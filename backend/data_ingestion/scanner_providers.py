"""
Market Scanner Provider Methods

Contains all provider-specific data-fetching methods extracted from MarketScanner.
Each method group is responsible for one data provider (yfinance, Finnhub,
Alpha Vantage, Polygon, SEC EDGAR, News API) and is intended to be used as a
mixin by the MarketScanner orchestrator.

This module is NOT meant to be imported directly by application code; use
backend.data_ingestion.market_scanner instead, which re-exports everything
through the MarketScanner class.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

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

from backend.config.settings import settings
from backend.utils.cache import stock_cache
from backend.utils.cost_monitor import cost_monitor
from backend.utils.circuit_breaker import CircuitBreakerError
from backend.data_ingestion.scanner_types import DataProvider

logger = logging.getLogger(__name__)


class ProviderMixin:
    """
    Mixin providing all provider-specific fetch methods for MarketScanner.

    Expects the following attributes to exist on the host class:
        _executor          - ThreadPoolExecutor for sync provider wrappers
        _provider_health   - Dict[str, ProviderHealth]
        _finnhub_client    - Optional[FinnhubClient]
        _alpha_vantage_client - Optional[AlphaVantageClient]
        _polygon_client    - Optional[PolygonClient]
        _sec_edgar_client  - Optional[SECEdgarClient]
        _price_fallback_chain
        _fundamentals_fallback_chain
        _news_fallback_chain
    And the helper methods:
        _get_finnhub_client(), _get_alpha_vantage_client(),
        _get_sec_edgar_client(), _record_provider_success(),
        _record_provider_failure(), _get_circuit_breaker()
    """

    # ------------------------------------------------------------------
    # YFINANCE – synchronous helpers wrapped in async executors
    # ------------------------------------------------------------------

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
                'timestamp': datetime.now(timezone.utc).isoformat(),
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
                'timestamp': datetime.now(timezone.utc).isoformat(),
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

    # ------------------------------------------------------------------
    # MULTI-PROVIDER FALLBACK CHAINS
    # ------------------------------------------------------------------

    async def _fetch_quote_with_fallback(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Fetch quote using fallback chain:
        yfinance -> Finnhub -> Polygon -> Alpha Vantage
        """
        from backend.data_ingestion.polygon_client import PolygonClient

        cache_key = f"quote:{ticker}"

        cached = await stock_cache.get(cache_key)
        if cached:
            logger.debug(f"Cache hit for quote {ticker}")
            return cached

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
                                    'timestamp': datetime.now(timezone.utc).isoformat(),
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
                    await stock_cache.set(cache_key, result, ttl=60)
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

        cached = await stock_cache.get(cache_key)
        if cached:
            logger.debug(f"Cache hit for fundamentals {ticker}")
            return cached

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
                                'timestamp': datetime.now(timezone.utc).isoformat(),
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
                                    'timestamp': datetime.now(timezone.utc).isoformat(),
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
        from backend.data_ingestion.polygon_client import PolygonClient

        cache_key = f"news:{ticker}"

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
                            loop = asyncio.get_event_loop()
                            result = await loop.run_in_executor(
                                self._executor,
                                self._fetch_newsapi_sync,
                                ticker
                            )

                if result:
                    self._record_provider_success(provider.value)
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

    # ------------------------------------------------------------------
    # ANALYST DATA (Finnhub)
    # ------------------------------------------------------------------

    async def _fetch_analyst_data(self, ticker: str) -> Dict[str, Any]:
        """Fetch analyst recommendations and price targets."""
        result = {}

        if self._provider_health[DataProvider.FINNHUB.value].is_healthy:
            try:
                if await cost_monitor.check_api_limit('finnhub'):
                    client = await self._get_finnhub_client()
                    async with client:
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


# Module-level availability flags re-exported for convenience
__all__ = [
    "ProviderMixin",
    "YFINANCE_AVAILABLE",
    "NEWSAPI_AVAILABLE",
]
