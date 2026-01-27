"""
Market Scanner Full Implementation

This module provides comprehensive market scanning capabilities by integrating
multiple data providers with intelligent fallback chains and caching.

Data Providers (Priority Order):
1. Yahoo Finance (yfinance) - Unlimited, primary for prices/fundamentals
2. Finnhub (60/min) - Real-time quotes, news, sentiment
3. Alpha Vantage (25/day, 5/min) - Historical, fundamentals, technical
4. Polygon (5/min) - Market data, aggregates
5. SEC EDGAR (Unlimited) - Fundamental data, filings
6. Financial Modeling Prep - Fundamentals, DCF (if API key available)
7. News API (100/day) - News headlines
8. FRED (1000/day) - Economic data

Fallback Chain Strategy:
- Prices: yfinance -> Finnhub -> Alpha Vantage -> Polygon
- Fundamentals: SEC EDGAR -> yfinance -> Alpha Vantage -> Finnhub
- News: Finnhub -> Polygon -> News API
- Technical: yfinance (calculated) -> Alpha Vantage -> Finnhub
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import json
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

# Third-party imports
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    yf = None

try:
    from newsapi import NewsApiClient
    NEWSAPI_AVAILABLE = True
except ImportError:
    NEWSAPI_AVAILABLE = False
    NewsApiClient = None

# Local imports
from backend.config.settings import settings
from backend.utils.cache import get_redis, stock_cache, market_cache
from backend.utils.cost_monitor import cost_monitor
from backend.utils.circuit_breaker import CircuitBreaker, CircuitBreakerError, get_api_circuit_breaker
from backend.data_ingestion.finnhub_client import FinnhubClient
from backend.data_ingestion.alpha_vantage_client import AlphaVantageClient
from backend.data_ingestion.polygon_client import PolygonClient
from backend.data_ingestion.sec_edgar_client import SECEdgarClient

logger = logging.getLogger(__name__)


class DataProvider(Enum):
    """Available data providers"""
    YFINANCE = "yfinance"
    FINNHUB = "finnhub"
    ALPHA_VANTAGE = "alpha_vantage"
    POLYGON = "polygon"
    SEC_EDGAR = "sec_edgar"
    FMP = "fmp"
    NEWS_API = "news_api"
    FRED = "fred"


@dataclass
class ProviderHealth:
    """Health status for a data provider"""
    name: str
    is_available: bool = True
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    consecutive_failures: int = 0
    success_count: int = 0
    failure_count: int = 0

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return (self.success_count / total) if total > 0 else 1.0

    @property
    def is_healthy(self) -> bool:
        return self.is_available and self.consecutive_failures < 5


@dataclass
class StockQuote:
    """Real-time stock quote data"""
    ticker: str
    current_price: float
    open: float
    high: float
    low: float
    previous_close: float
    volume: int
    change: float
    change_percent: float
    timestamp: datetime
    source: str


@dataclass
class StockFundamentals:
    """Fundamental data for a stock"""
    ticker: str
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    ev_ebitda: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    market_cap: Optional[float] = None
    beta: Optional[float] = None
    dividend_yield: Optional[float] = None
    source: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)


class MarketScanner:
    """
    Production-ready Market Scanner with full provider integration.

    Provides comprehensive market scanning with:
    - Multi-provider data fetching with fallback chains
    - Intelligent caching to respect rate limits
    - Circuit breaker pattern for fault tolerance
    - Async operations for high throughput
    - Health monitoring for providers
    """

    def __init__(self):
        """Initialize the market scanner with all available providers."""
        self._initialized = False
        self._cache: Dict[str, Any] = {}
        self._cache_ttl = timedelta(hours=1)

        # Provider health tracking
        self._provider_health: Dict[str, ProviderHealth] = {
            p.value: ProviderHealth(name=p.value) for p in DataProvider
        }

        # Thread pool for sync operations (yfinance)
        self._executor = ThreadPoolExecutor(max_workers=10)

        # API clients (initialized lazily)
        self._finnhub_client: Optional[FinnhubClient] = None
        self._alpha_vantage_client: Optional[AlphaVantageClient] = None
        self._polygon_client: Optional[PolygonClient] = None
        self._sec_edgar_client: Optional[SECEdgarClient] = None
        self._news_api_client = None

        # Fallback chains for different data types
        self._price_fallback_chain = [
            DataProvider.YFINANCE,
            DataProvider.FINNHUB,
            DataProvider.POLYGON,
            DataProvider.ALPHA_VANTAGE
        ]

        self._fundamentals_fallback_chain = [
            DataProvider.YFINANCE,
            DataProvider.SEC_EDGAR,
            DataProvider.FINNHUB,
            DataProvider.ALPHA_VANTAGE
        ]

        self._news_fallback_chain = [
            DataProvider.FINNHUB,
            DataProvider.POLYGON,
            DataProvider.NEWS_API
        ]

        # Circuit breakers for each provider
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}

        logger.info("MarketScanner initialized with multi-provider support")

    async def initialize(self) -> None:
        """
        Initialize the market scanner and validate provider availability.
        """
        logger.info("Initializing MarketScanner...")

        # Initialize cost monitor
        if cost_monitor.redis is None:
            await cost_monitor.initialize()

        # Check which providers are available
        await self._check_provider_availability()

        self._initialized = True

        # Log provider status
        available = [p for p, h in self._provider_health.items() if h.is_available]
        logger.info(f"MarketScanner initialized with {len(available)} available providers: {available}")

    async def _check_provider_availability(self) -> None:
        """Check availability of all data providers."""

        # Yahoo Finance - always available if library installed
        self._provider_health[DataProvider.YFINANCE.value].is_available = YFINANCE_AVAILABLE
        if YFINANCE_AVAILABLE:
            logger.info("Yahoo Finance (yfinance) available - unlimited requests")
        else:
            logger.warning("yfinance not installed - pip install yfinance")

        # Finnhub - check API key
        if settings.FINNHUB_API_KEY:
            self._provider_health[DataProvider.FINNHUB.value].is_available = True
            logger.info("Finnhub available - 60 requests/minute")
        else:
            self._provider_health[DataProvider.FINNHUB.value].is_available = False
            logger.warning("Finnhub API key not configured")

        # Alpha Vantage - check API key
        if settings.ALPHA_VANTAGE_API_KEY:
            self._provider_health[DataProvider.ALPHA_VANTAGE.value].is_available = True
            logger.info("Alpha Vantage available - 25 requests/day, 5/minute")
        else:
            self._provider_health[DataProvider.ALPHA_VANTAGE.value].is_available = False
            logger.warning("Alpha Vantage API key not configured")

        # Polygon - check API key
        if settings.POLYGON_API_KEY:
            self._provider_health[DataProvider.POLYGON.value].is_available = True
            logger.info("Polygon available - 5 requests/minute")
        else:
            self._provider_health[DataProvider.POLYGON.value].is_available = False
            logger.warning("Polygon API key not configured")

        # SEC EDGAR - always available (no API key needed)
        self._provider_health[DataProvider.SEC_EDGAR.value].is_available = True
        logger.info("SEC EDGAR available - unlimited requests")

        # News API
        if settings.NEWS_API_KEY and NEWSAPI_AVAILABLE:
            self._provider_health[DataProvider.NEWS_API.value].is_available = True
            logger.info("News API available - 100 requests/day")
        else:
            self._provider_health[DataProvider.NEWS_API.value].is_available = False

        # FMP
        if settings.FMP_API_KEY:
            self._provider_health[DataProvider.FMP.value].is_available = True
            logger.info("Financial Modeling Prep available")
        else:
            self._provider_health[DataProvider.FMP.value].is_available = False

        # FRED
        if settings.FRED_API_KEY:
            self._provider_health[DataProvider.FRED.value].is_available = True
            logger.info("FRED available - 1000 requests/day")
        else:
            self._provider_health[DataProvider.FRED.value].is_available = False

    def _get_circuit_breaker(self, provider: str) -> CircuitBreaker:
        """Get or create circuit breaker for a provider."""
        if provider not in self._circuit_breakers:
            self._circuit_breakers[provider] = get_api_circuit_breaker(provider)
        return self._circuit_breakers[provider]

    def _record_provider_success(self, provider: str) -> None:
        """Record successful API call for a provider."""
        health = self._provider_health.get(provider)
        if health:
            health.last_success = datetime.utcnow()
            health.success_count += 1
            health.consecutive_failures = 0

    def _record_provider_failure(self, provider: str, error: str = "") -> None:
        """Record failed API call for a provider."""
        health = self._provider_health.get(provider)
        if health:
            health.last_failure = datetime.utcnow()
            health.failure_count += 1
            health.consecutive_failures += 1

            if health.consecutive_failures >= 5:
                health.is_available = False
                logger.warning(f"Provider {provider} marked unavailable after {health.consecutive_failures} failures")

    async def _get_finnhub_client(self) -> FinnhubClient:
        """Get or create Finnhub client."""
        if self._finnhub_client is None:
            self._finnhub_client = FinnhubClient()
        return self._finnhub_client

    async def _get_alpha_vantage_client(self) -> AlphaVantageClient:
        """Get or create Alpha Vantage client."""
        if self._alpha_vantage_client is None:
            self._alpha_vantage_client = AlphaVantageClient()
        return self._alpha_vantage_client

    async def _get_sec_edgar_client(self) -> SECEdgarClient:
        """Get or create SEC EDGAR client."""
        if self._sec_edgar_client is None:
            self._sec_edgar_client = SECEdgarClient()
        return self._sec_edgar_client

    # ========================
    # YFINANCE DATA FETCHING
    # ========================

    def _fetch_yfinance_quote_sync(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Fetch quote from Yahoo Finance (synchronous)."""
        if not YFINANCE_AVAILABLE:
            return None

        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            if not info or info.get('regularMarketPrice') is None:
                return None

            return {
                'ticker': ticker,
                'current_price': info.get('regularMarketPrice', 0),
                'open': info.get('regularMarketOpen', 0),
                'high': info.get('regularMarketDayHigh', 0),
                'low': info.get('regularMarketDayLow', 0),
                'previous_close': info.get('regularMarketPreviousClose', 0),
                'volume': info.get('regularMarketVolume', 0),
                'change': info.get('regularMarketChange', 0),
                'change_percent': info.get('regularMarketChangePercent', 0),
                'market_cap': info.get('marketCap'),
                'beta': info.get('beta'),
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'dividend_yield': info.get('dividendYield'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'name': info.get('longName') or info.get('shortName'),
                'exchange': info.get('exchange'),
                'currency': info.get('currency', 'USD'),
                'timestamp': datetime.utcnow().isoformat(),
                'source': 'yfinance'
            }
        except Exception as e:
            logger.debug(f"yfinance quote fetch failed for {ticker}: {e}")
            return None

    def _fetch_yfinance_fundamentals_sync(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Fetch fundamentals from Yahoo Finance (synchronous)."""
        if not YFINANCE_AVAILABLE:
            return None

        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            if not info:
                return None

            return {
                'ticker': ticker,
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'pb_ratio': info.get('priceToBook'),
                'ps_ratio': info.get('priceToSalesTrailing12Months'),
                'peg_ratio': info.get('pegRatio'),
                'ev_to_revenue': info.get('enterpriseToRevenue'),
                'ev_to_ebitda': info.get('enterpriseToEbitda'),
                'profit_margin': info.get('profitMargins'),
                'operating_margin': info.get('operatingMargins'),
                'gross_margin': info.get('grossMargins'),
                'roe': info.get('returnOnEquity'),
                'roa': info.get('returnOnAssets'),
                'revenue_growth': info.get('revenueGrowth'),
                'earnings_growth': info.get('earningsGrowth'),
                'debt_to_equity': info.get('debtToEquity'),
                'current_ratio': info.get('currentRatio'),
                'quick_ratio': info.get('quickRatio'),
                'market_cap': info.get('marketCap'),
                'enterprise_value': info.get('enterpriseValue'),
                'beta': info.get('beta'),
                'dividend_yield': info.get('dividendYield'),
                'dividend_rate': info.get('dividendRate'),
                'payout_ratio': info.get('payoutRatio'),
                'book_value': info.get('bookValue'),
                'earnings_per_share': info.get('trailingEps'),
                'forward_eps': info.get('forwardEps'),
                'revenue': info.get('totalRevenue'),
                'gross_profit': info.get('grossProfits'),
                'ebitda': info.get('ebitda'),
                'net_income': info.get('netIncomeToCommon'),
                'free_cash_flow': info.get('freeCashflow'),
                'operating_cash_flow': info.get('operatingCashflow'),
                'total_cash': info.get('totalCash'),
                'total_debt': info.get('totalDebt'),
                'shares_outstanding': info.get('sharesOutstanding'),
                'float_shares': info.get('floatShares'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                '52_week_high': info.get('fiftyTwoWeekHigh'),
                '52_week_low': info.get('fiftyTwoWeekLow'),
                '50_day_average': info.get('fiftyDayAverage'),
                '200_day_average': info.get('twoHundredDayAverage'),
                'analyst_target_price': info.get('targetMeanPrice'),
                'analyst_recommendations': info.get('recommendationKey'),
                'timestamp': datetime.utcnow().isoformat(),
                'source': 'yfinance'
            }
        except Exception as e:
            logger.debug(f"yfinance fundamentals fetch failed for {ticker}: {e}")
            return None

    def _fetch_yfinance_history_sync(self, ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """Fetch price history from Yahoo Finance (synchronous)."""
        if not YFINANCE_AVAILABLE:
            return None

        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)

            if hist.empty:
                return None

            # Standardize column names
            hist = hist.reset_index()
            hist.columns = [c.lower().replace(' ', '_') for c in hist.columns]

            return hist
        except Exception as e:
            logger.debug(f"yfinance history fetch failed for {ticker}: {e}")
            return None

    async def _fetch_yfinance_quote(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Fetch quote from Yahoo Finance (async wrapper)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._fetch_yfinance_quote_sync,
            ticker
        )

    async def _fetch_yfinance_fundamentals(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Fetch fundamentals from Yahoo Finance (async wrapper)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._fetch_yfinance_fundamentals_sync,
            ticker
        )

    async def _fetch_yfinance_history(self, ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """Fetch price history from Yahoo Finance (async wrapper)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._fetch_yfinance_history_sync,
            ticker,
            period
        )

    # ========================
    # MULTI-PROVIDER FETCHING
    # ========================

    async def _fetch_quote_with_fallback(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Fetch quote using fallback chain:
        yfinance -> Finnhub -> Polygon -> Alpha Vantage
        """
        cache_key = f"quote:{ticker}"

        # Check cache first
        cached = await stock_cache.get(cache_key)
        if cached:
            logger.debug(f"Cache hit for quote {ticker}")
            return cached

        # Try each provider in fallback chain
        for provider in self._price_fallback_chain:
            health = self._provider_health.get(provider.value)
            if not health or not health.is_healthy:
                continue

            try:
                result = None

                if provider == DataProvider.YFINANCE:
                    result = await self._fetch_yfinance_quote(ticker)

                elif provider == DataProvider.FINNHUB:
                    if await cost_monitor.check_api_limit('finnhub'):
                        client = await self._get_finnhub_client()
                        async with client:
                            result = await client.get_quote(ticker)
                        if result:
                            result['source'] = 'finnhub'

                elif provider == DataProvider.POLYGON:
                    if await cost_monitor.check_api_limit('polygon'):
                        async with PolygonClient() as client:
                            snapshot = await client.get_snapshot(ticker)
                            if snapshot:
                                day = snapshot.get('day', {})
                                prev = snapshot.get('prev_day', {})
                                result = {
                                    'ticker': ticker,
                                    'current_price': day.get('c', 0),
                                    'open': day.get('o', 0),
                                    'high': day.get('h', 0),
                                    'low': day.get('l', 0),
                                    'volume': day.get('v', 0),
                                    'previous_close': prev.get('c', 0),
                                    'change': day.get('c', 0) - prev.get('c', 0),
                                    'change_percent': ((day.get('c', 0) - prev.get('c', 0)) / prev.get('c', 1)) * 100 if prev.get('c') else 0,
                                    'timestamp': datetime.utcnow().isoformat(),
                                    'source': 'polygon'
                                }

                elif provider == DataProvider.ALPHA_VANTAGE:
                    if await cost_monitor.check_api_limit('alpha_vantage'):
                        client = await self._get_alpha_vantage_client()
                        async with client:
                            result = await client.get_quote(ticker)
                        if result:
                            result['source'] = 'alpha_vantage'

                if result and result.get('current_price'):
                    self._record_provider_success(provider.value)

                    # Cache the result
                    await stock_cache.set(cache_key, result, ttl=60)  # 1 minute for quotes

                    logger.debug(f"Got quote for {ticker} from {provider.value}")
                    return result

            except CircuitBreakerError:
                logger.warning(f"Circuit breaker open for {provider.value}")
                continue
            except Exception as e:
                logger.debug(f"Failed to get quote from {provider.value} for {ticker}: {e}")
                self._record_provider_failure(provider.value, str(e))
                continue

        logger.warning(f"All providers failed for quote {ticker}")
        return None

    async def _fetch_fundamentals_with_fallback(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Fetch fundamentals using fallback chain:
        yfinance -> SEC EDGAR -> Finnhub -> Alpha Vantage
        """
        cache_key = f"fundamentals:{ticker}"

        # Check cache first (fundamentals cache for longer)
        cached = await stock_cache.get(cache_key)
        if cached:
            logger.debug(f"Cache hit for fundamentals {ticker}")
            return cached

        # Try each provider in fallback chain
        for provider in self._fundamentals_fallback_chain:
            health = self._provider_health.get(provider.value)
            if not health or not health.is_healthy:
                continue

            try:
                result = None

                if provider == DataProvider.YFINANCE:
                    result = await self._fetch_yfinance_fundamentals(ticker)

                elif provider == DataProvider.SEC_EDGAR:
                    client = await self._get_sec_edgar_client()
                    async with client:
                        facts = await client.get_company_facts(ticker)
                        if facts and facts.get('metrics'):
                            ratios = await client.calculate_fundamental_ratios(ticker)
                            result = {
                                'ticker': ticker,
                                'entity_name': facts.get('entity_name'),
                                'sector': facts.get('sic_description'),
                                **facts.get('metrics', {}),
                                **(ratios.get('ratios', {}) if ratios else {}),
                                'timestamp': datetime.utcnow().isoformat(),
                                'source': 'sec_edgar'
                            }

                elif provider == DataProvider.FINNHUB:
                    if await cost_monitor.check_api_limit('finnhub'):
                        client = await self._get_finnhub_client()
                        async with client:
                            financials = await client.get_basic_financials(ticker)
                            profile = await client.get_company_profile(ticker)

                            if financials or profile:
                                result = {
                                    'ticker': ticker,
                                    **(financials or {}),
                                    **(profile or {}),
                                    'timestamp': datetime.utcnow().isoformat(),
                                    'source': 'finnhub'
                                }

                elif provider == DataProvider.ALPHA_VANTAGE:
                    if await cost_monitor.check_api_limit('alpha_vantage'):
                        client = await self._get_alpha_vantage_client()
                        async with client:
                            result = await client.get_company_overview(ticker)
                        if result:
                            result['source'] = 'alpha_vantage'

                if result:
                    self._record_provider_success(provider.value)

                    # Cache the result (fundamentals for 6 hours)
                    await stock_cache.set(cache_key, result, ttl=21600)

                    logger.debug(f"Got fundamentals for {ticker} from {provider.value}")
                    return result

            except CircuitBreakerError:
                logger.warning(f"Circuit breaker open for {provider.value}")
                continue
            except Exception as e:
                logger.debug(f"Failed to get fundamentals from {provider.value} for {ticker}: {e}")
                self._record_provider_failure(provider.value, str(e))
                continue

        logger.warning(f"All providers failed for fundamentals {ticker}")
        return None

    async def _fetch_news_with_fallback(self, ticker: str) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch news using fallback chain:
        Finnhub -> Polygon -> News API
        """
        cache_key = f"news:{ticker}"

        # Check cache first
        cached = await stock_cache.get(cache_key)
        if cached:
            logger.debug(f"Cache hit for news {ticker}")
            return cached

        for provider in self._news_fallback_chain:
            health = self._provider_health.get(provider.value)
            if not health or not health.is_healthy:
                continue

            try:
                result = None

                if provider == DataProvider.FINNHUB:
                    if await cost_monitor.check_api_limit('finnhub'):
                        client = await self._get_finnhub_client()
                        async with client:
                            result = await client.get_news(ticker)

                elif provider == DataProvider.POLYGON:
                    if await cost_monitor.check_api_limit('polygon'):
                        async with PolygonClient() as client:
                            result = await client.get_news(ticker, limit=20)

                elif provider == DataProvider.NEWS_API:
                    if NEWSAPI_AVAILABLE and settings.NEWS_API_KEY:
                        if await cost_monitor.check_api_limit('news_api'):
                            # Run in executor since newsapi is sync
                            loop = asyncio.get_event_loop()
                            result = await loop.run_in_executor(
                                self._executor,
                                self._fetch_newsapi_sync,
                                ticker
                            )

                if result:
                    self._record_provider_success(provider.value)

                    # Cache news for 15 minutes
                    await stock_cache.set(cache_key, result, ttl=900)

                    logger.debug(f"Got news for {ticker} from {provider.value}")
                    return result

            except CircuitBreakerError:
                logger.warning(f"Circuit breaker open for {provider.value}")
                continue
            except Exception as e:
                logger.debug(f"Failed to get news from {provider.value} for {ticker}: {e}")
                self._record_provider_failure(provider.value, str(e))
                continue

        logger.warning(f"All providers failed for news {ticker}")
        return []

    def _fetch_newsapi_sync(self, ticker: str) -> Optional[List[Dict[str, Any]]]:
        """Fetch news from News API (synchronous)."""
        if not NEWSAPI_AVAILABLE or not settings.NEWS_API_KEY:
            return None

        try:
            client = NewsApiClient(api_key=settings.NEWS_API_KEY)
            response = client.get_everything(
                q=ticker,
                language='en',
                sort_by='publishedAt',
                page_size=20
            )

            if response.get('status') == 'ok':
                return [
                    {
                        'headline': article.get('title'),
                        'summary': article.get('description'),
                        'url': article.get('url'),
                        'source': article.get('source', {}).get('name'),
                        'datetime': article.get('publishedAt'),
                        'image': article.get('urlToImage')
                    }
                    for article in response.get('articles', [])
                ]
            return None
        except Exception as e:
            logger.debug(f"News API fetch failed for {ticker}: {e}")
            return None

    # ========================
    # PUBLIC API METHODS
    # ========================

    async def scan_market(
        self,
        sectors: Optional[List[str]] = None,
        market_cap_range: Optional[Tuple[float, float]] = None,
        max_stocks: int = 500,
        min_volume: Optional[float] = None,
        min_price: float = 5.0,
        exchanges: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Scan the market for candidate stocks based on filters.

        Args:
            sectors: List of sectors to include (e.g., ['Technology', 'Healthcare'])
            market_cap_range: Tuple of (min_market_cap, max_market_cap) in dollars
            max_stocks: Maximum number of stocks to return
            min_volume: Minimum average daily volume
            min_price: Minimum stock price (default $5 to avoid penny stocks)
            exchanges: List of exchanges to include (e.g., ['NYSE', 'NASDAQ'])

        Returns:
            List of stock dictionaries with basic data for analysis
        """
        logger.info(f"Scanning market with filters: sectors={sectors}, market_cap={market_cap_range}")

        # Check cache for stock universe
        cache_key = "market:stock_universe"
        stock_universe = await market_cache.get(cache_key)

        if not stock_universe:
            # Build stock universe from multiple sources
            stock_universe = await self._build_stock_universe()

            if stock_universe:
                # Cache for 24 hours
                await market_cache.set(cache_key, stock_universe, ttl=86400)

        if not stock_universe:
            logger.warning("Failed to build stock universe")
            return []

        # Apply filters
        filtered_stocks = []

        for stock in stock_universe:
            # Sector filter
            if sectors and stock.get('sector') not in sectors:
                continue

            # Market cap filter
            market_cap = stock.get('market_cap', 0)
            if market_cap_range:
                if market_cap < market_cap_range[0] or market_cap > market_cap_range[1]:
                    continue

            # Volume filter
            if min_volume and stock.get('volume', 0) < min_volume:
                continue

            # Price filter
            if stock.get('current_price', 0) < min_price:
                continue

            # Exchange filter
            if exchanges and stock.get('exchange') not in exchanges:
                continue

            filtered_stocks.append(stock)

            if len(filtered_stocks) >= max_stocks:
                break

        logger.info(f"Market scan returned {len(filtered_stocks)} stocks")
        return filtered_stocks

    async def _build_stock_universe(self) -> List[Dict[str, Any]]:
        """Build comprehensive stock universe from available sources."""
        stocks = []

        # Try to get from Finnhub (best source for US stock list)
        if self._provider_health[DataProvider.FINNHUB.value].is_healthy:
            try:
                if await cost_monitor.check_api_limit('finnhub'):
                    client = await self._get_finnhub_client()
                    async with client:
                        # Finnhub returns all US stocks with one call
                        response = await client._make_request(
                            "stock/symbol",
                            {'exchange': 'US'}
                        )

                        if response:
                            for stock in response:
                                symbol = stock.get('symbol', '')
                                # Filter for common stocks only
                                if stock.get('type') in ['Common Stock', 'CS', 'EQS']:
                                    # Skip symbols with special characters
                                    if len(symbol) <= 5 and '.' not in symbol and '-' not in symbol:
                                        stocks.append({
                                            'ticker': symbol,
                                            'name': stock.get('description', symbol),
                                            'exchange': 'NASDAQ' if 'NAS' in stock.get('mic', '') else 'NYSE',
                                            'type': stock.get('type')
                                        })

                            logger.info(f"Built stock universe with {len(stocks)} stocks from Finnhub")
                            return stocks

            except Exception as e:
                logger.warning(f"Failed to build stock universe from Finnhub: {e}")

        # Fallback: Use a curated list of major stocks
        major_tickers = self._get_major_stock_list()

        # Fetch basic data for each ticker
        for ticker in major_tickers[:max(500, len(major_tickers))]:
            quote = await self._fetch_quote_with_fallback(ticker)
            if quote:
                stocks.append({
                    'ticker': ticker,
                    'name': quote.get('name', ticker),
                    'current_price': quote.get('current_price'),
                    'market_cap': quote.get('market_cap'),
                    'volume': quote.get('volume'),
                    'sector': quote.get('sector'),
                    'industry': quote.get('industry'),
                    'exchange': quote.get('exchange', 'NYSE')
                })

            # Brief delay to avoid overwhelming providers
            await asyncio.sleep(0.1)

        logger.info(f"Built stock universe with {len(stocks)} stocks (fallback method)")
        return stocks

    def _get_major_stock_list(self) -> List[str]:
        """Get list of major US stock tickers."""
        # S&P 500 + additional major stocks
        return [
            # Technology
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC',
            'CRM', 'ORCL', 'ADBE', 'NFLX', 'CSCO', 'IBM', 'QCOM', 'TXN', 'AVGO', 'NOW',
            # Finance
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'V', 'MA', 'PYPL',
            # Healthcare
            'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT', 'DHR', 'BMY',
            # Consumer
            'WMT', 'PG', 'KO', 'PEP', 'COST', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT',
            # Industrial
            'CAT', 'BA', 'HON', 'UPS', 'RTX', 'LMT', 'GE', 'MMM', 'DE', 'UNP',
            # Energy
            'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'KMI',
            # Telecom
            'VZ', 'T', 'TMUS', 'CMCSA', 'CHTR',
            # Real Estate
            'PLD', 'AMT', 'CCI', 'EQIX', 'SPG',
            # Utilities
            'NEE', 'DUK', 'SO', 'D', 'AEP',
            # Materials
            'LIN', 'APD', 'SHW', 'ECL', 'DD', 'NEM', 'FCX',
        ]

    async def get_stock_data(
        self,
        ticker: str,
        include_fundamentals: bool = True,
        include_news: bool = True,
        include_social: bool = False,
        lookback_days: int = 365
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch comprehensive data for a single stock.

        Args:
            ticker: Stock ticker symbol
            include_fundamentals: Whether to fetch fundamental data
            include_news: Whether to fetch recent news
            include_social: Whether to fetch social media mentions (not implemented)
            lookback_days: Number of days of historical price data

        Returns:
            Comprehensive stock data dictionary
        """
        logger.info(f"Fetching comprehensive data for {ticker}")

        # Check cache first
        cache_key = f"stock_data:{ticker}:{include_fundamentals}:{include_news}:{lookback_days}"
        cached = await stock_cache.get(cache_key)
        if cached:
            logger.debug(f"Cache hit for comprehensive data {ticker}")
            return cached

        # Fetch data in parallel
        tasks = []

        # Quote is always fetched
        tasks.append(self._fetch_quote_with_fallback(ticker))

        # Fundamentals
        if include_fundamentals:
            tasks.append(self._fetch_fundamentals_with_fallback(ticker))
        else:
            tasks.append(asyncio.coroutine(lambda: None)())

        # News
        if include_news:
            tasks.append(self._fetch_news_with_fallback(ticker))
        else:
            tasks.append(asyncio.coroutine(lambda: None)())

        # Price history
        period = "1y" if lookback_days >= 365 else f"{lookback_days}d"
        tasks.append(self._fetch_yfinance_history(ticker, period))

        # Analyst data from Finnhub
        tasks.append(self._fetch_analyst_data(ticker))

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        quote_data = results[0] if not isinstance(results[0], Exception) else None
        fundamentals = results[1] if not isinstance(results[1], Exception) else None
        news = results[2] if not isinstance(results[2], Exception) else []
        price_history = results[3] if not isinstance(results[3], Exception) else None
        analyst_data = results[4] if not isinstance(results[4], Exception) else {}

        if not quote_data:
            logger.warning(f"Failed to fetch any data for {ticker}")
            return None

        # Convert price history DataFrame to dict
        price_history_dict = None
        if price_history is not None and isinstance(price_history, pd.DataFrame):
            price_history_dict = price_history.to_dict('records')

        # Compile comprehensive data
        stock_data = {
            'ticker': ticker,
            'current_price': quote_data.get('current_price'),
            'price_history': price_history_dict,
            'fundamentals': fundamentals or {},
            'market_cap': quote_data.get('market_cap') or (fundamentals.get('market_cap') if fundamentals else None),
            'beta': quote_data.get('beta') or (fundamentals.get('beta') if fundamentals else None),
            'sector': quote_data.get('sector') or (fundamentals.get('sector') if fundamentals else None),
            'industry': quote_data.get('industry') or (fundamentals.get('industry') if fundamentals else None),
            'name': quote_data.get('name'),
            'exchange': quote_data.get('exchange'),
            'currency': quote_data.get('currency', 'USD'),
            'news': news or [],
            'social_mentions': [],  # Not implemented
            'analyst_opinions': analyst_data.get('recommendations', []),
            'price_target': analyst_data.get('price_target'),
            'sentiment': analyst_data.get('sentiment'),
            'next_earnings_date': fundamentals.get('next_earnings_date') if fundamentals else None,
            'quote': quote_data,
            'peer_data': None,  # Could be implemented
            'data_sources': self._get_data_sources(quote_data, fundamentals, news),
            'timestamp': datetime.utcnow().isoformat()
        }

        # Cache for 5 minutes
        await stock_cache.set(cache_key, stock_data, ttl=300)

        logger.info(f"Successfully fetched comprehensive data for {ticker}")
        return stock_data

    async def _fetch_analyst_data(self, ticker: str) -> Dict[str, Any]:
        """Fetch analyst recommendations and price targets."""
        result = {}

        if self._provider_health[DataProvider.FINNHUB.value].is_healthy:
            try:
                if await cost_monitor.check_api_limit('finnhub'):
                    client = await self._get_finnhub_client()
                    async with client:
                        # Fetch multiple analyst data points concurrently
                        tasks = [
                            client.get_recommendations(ticker),
                            client.get_price_target(ticker),
                            client.get_sentiment(ticker)
                        ]

                        results = await asyncio.gather(*tasks, return_exceptions=True)

                        if not isinstance(results[0], Exception):
                            result['recommendations'] = results[0]

                        if not isinstance(results[1], Exception):
                            result['price_target'] = results[1]

                        if not isinstance(results[2], Exception):
                            result['sentiment'] = results[2]

                        self._record_provider_success(DataProvider.FINNHUB.value)

            except Exception as e:
                logger.debug(f"Failed to fetch analyst data for {ticker}: {e}")
                self._record_provider_failure(DataProvider.FINNHUB.value, str(e))

        return result

    def _get_data_sources(
        self,
        quote: Optional[Dict],
        fundamentals: Optional[Dict],
        news: Optional[List]
    ) -> List[str]:
        """Get list of data sources used."""
        sources = []

        if quote:
            sources.append(quote.get('source', 'unknown'))

        if fundamentals:
            sources.append(fundamentals.get('source', 'unknown'))

        if news:
            sources.append('news')

        return list(set(sources))

    async def get_trending_stocks(
        self,
        limit: int = 10,
        timeframe: str = 'day'
    ) -> List[Dict[str, Any]]:
        """
        Get trending/most active stocks.

        Args:
            limit: Number of stocks to return
            timeframe: Time period to consider ('day', 'week', 'month')

        Returns:
            List of trending stock info dictionaries
        """
        logger.info(f"Fetching trending stocks (limit={limit}, timeframe={timeframe})")

        cache_key = f"trending:{timeframe}:{limit}"
        cached = await market_cache.get(cache_key)
        if cached:
            return cached

        trending = []

        # Try Finnhub for market movers
        if self._provider_health[DataProvider.FINNHUB.value].is_healthy:
            try:
                if await cost_monitor.check_api_limit('finnhub'):
                    client = await self._get_finnhub_client()
                    async with client:
                        # Get market news to identify trending tickers
                        news = await client.get_news(category='general')

                        if news:
                            # Extract and count mentioned tickers
                            ticker_counts = {}
                            for article in news[:50]:
                                for related in article.get('related', []):
                                    ticker_counts[related] = ticker_counts.get(related, 0) + 1

                            # Sort by mention count
                            sorted_tickers = sorted(
                                ticker_counts.items(),
                                key=lambda x: x[1],
                                reverse=True
                            )

                            # Fetch data for top mentioned tickers
                            for ticker, count in sorted_tickers[:limit]:
                                quote = await self._fetch_quote_with_fallback(ticker)
                                if quote:
                                    trending.append({
                                        'ticker': ticker,
                                        'name': quote.get('name', ticker),
                                        'current_price': quote.get('current_price'),
                                        'change': quote.get('change'),
                                        'change_percent': quote.get('change_percent'),
                                        'volume': quote.get('volume'),
                                        'mention_count': count,
                                        'source': 'finnhub_news'
                                    })

                        self._record_provider_success(DataProvider.FINNHUB.value)

            except Exception as e:
                logger.warning(f"Failed to get trending from Finnhub: {e}")
                self._record_provider_failure(DataProvider.FINNHUB.value, str(e))

        # Fallback: Use major stocks with highest volume
        if not trending:
            major_tickers = self._get_major_stock_list()[:20]

            for ticker in major_tickers:
                quote = await self._fetch_quote_with_fallback(ticker)
                if quote:
                    trending.append({
                        'ticker': ticker,
                        'name': quote.get('name', ticker),
                        'current_price': quote.get('current_price'),
                        'change': quote.get('change'),
                        'change_percent': quote.get('change_percent'),
                        'volume': quote.get('volume'),
                        'source': 'major_stocks'
                    })

            # Sort by volume
            trending.sort(key=lambda x: x.get('volume', 0) or 0, reverse=True)
            trending = trending[:limit]

        # Cache for 15 minutes
        if trending:
            await market_cache.set(cache_key, trending, ttl=900)

        logger.info(f"Returning {len(trending)} trending stocks")
        return trending

    async def get_sector_performance(self) -> Dict[str, Dict[str, float]]:
        """
        Get sector performance metrics.

        Returns:
            Dictionary mapping sector names to performance metrics
        """
        logger.info("Fetching sector performance")

        cache_key = "sector_performance"
        cached = await market_cache.get(cache_key)
        if cached:
            return cached

        # Sector ETFs as proxies
        sector_etfs = {
            'Technology': 'XLK',
            'Healthcare': 'XLV',
            'Financials': 'XLF',
            'Consumer Discretionary': 'XLY',
            'Consumer Staples': 'XLP',
            'Energy': 'XLE',
            'Industrials': 'XLI',
            'Materials': 'XLB',
            'Utilities': 'XLU',
            'Real Estate': 'XLRE',
            'Communication Services': 'XLC'
        }

        sector_performance = {}

        for sector, etf in sector_etfs.items():
            try:
                quote = await self._fetch_quote_with_fallback(etf)

                if quote:
                    sector_performance[sector] = {
                        'etf': etf,
                        'current_price': quote.get('current_price'),
                        'change': quote.get('change'),
                        'change_percent': quote.get('change_percent'),
                        'volume': quote.get('volume')
                    }
            except Exception as e:
                logger.debug(f"Failed to get sector performance for {sector}: {e}")

        # Cache for 5 minutes
        if sector_performance:
            await market_cache.set(cache_key, sector_performance, ttl=300)

        return sector_performance

    async def refresh_stock_universe(self) -> int:
        """
        Refresh the stock universe from exchanges.

        Returns:
            Number of stocks in the refreshed universe
        """
        logger.info("Refreshing stock universe")

        # Clear cached universe
        await market_cache.delete("market:stock_universe")

        # Rebuild
        universe = await self._build_stock_universe()

        return len(universe)

    def clear_cache(self) -> None:
        """Clear the data cache."""
        self._cache.clear()
        logger.info("MarketScanner in-memory cache cleared")

    @property
    def is_initialized(self) -> bool:
        """Check if the scanner is initialized."""
        return self._initialized

    async def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all data providers."""
        return {
            provider: {
                'is_available': health.is_available,
                'is_healthy': health.is_healthy,
                'success_rate': health.success_rate,
                'success_count': health.success_count,
                'failure_count': health.failure_count,
                'consecutive_failures': health.consecutive_failures,
                'last_success': health.last_success.isoformat() if health.last_success else None,
                'last_failure': health.last_failure.isoformat() if health.last_failure else None
            }
            for provider, health in self._provider_health.items()
        }

    async def cleanup(self) -> None:
        """Clean up resources."""
        self._executor.shutdown(wait=False)
        logger.info("MarketScanner cleanup completed")


# Module-level singleton for convenience
_scanner_instance: Optional[MarketScanner] = None


async def get_market_scanner() -> MarketScanner:
    """Get or create the market scanner singleton."""
    global _scanner_instance

    if _scanner_instance is None:
        _scanner_instance = MarketScanner()
        await _scanner_instance.initialize()

    return _scanner_instance
