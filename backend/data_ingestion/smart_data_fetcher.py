"""
Smart Data Fetcher - Intelligent data fetching with caching and rate limiting.

This module provides a unified interface for fetching stock data from
multiple sources with intelligent caching and rate limit management.

F-05-004 (audit 2026-04, G2a sub-theme C step 25): the previous
implementation returned hardcoded zeros / empty lists from every
``_fetch_*`` method, which silently degraded any downstream analytics
that consumed it. Each fetcher now delegates to the existing real
clients (``FinnhubClient``, ``AlphaVantageClient``, ``PolygonClient``,
``SECEdgarClient``) and only falls back to a ``source: "unavailable"``
sentinel when every client either failed or is not configured.
"""

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from datetime import datetime, timezone

if TYPE_CHECKING:
    # ``BaseAPIClient`` is used only for type annotations. Importing it
    # at runtime would drag in the backend.config chain (Pydantic
    # Settings, secrets manager, etc.) which is too heavy for the
    # surface this module presents.
    from backend.data_ingestion.base_client import BaseAPIClient

logger = logging.getLogger(__name__)


class SmartDataFetcher:
    """Unified data fetcher backed by real API clients with graceful fallback."""

    def __init__(self, cache_manager=None, rate_limiter=None):
        """Initialize the smart data fetcher.

        Args:
            cache_manager: Optional cache manager for caching data.
            rate_limiter: Optional rate limiter for managing API calls.
        """
        self.cache_manager = cache_manager
        self.rate_limiter = rate_limiter
        # ``None`` values are cached for clients that failed to
        # initialize (e.g. missing API key) so we don't retry them
        # on every fetch.
        self._clients: Dict[str, Optional["BaseAPIClient"]] = {}

    # ------------------------------------------------------------------
    # Client lazy init
    # ------------------------------------------------------------------

    def _get_client(self, name: str) -> Optional["BaseAPIClient"]:
        """Lazily build the requested client, swallowing config errors.

        Returns ``None`` if the client is missing required configuration
        (e.g. unset API key) so callers can move on to the next source.
        """
        if name in self._clients:
            return self._clients[name]

        client: Optional["BaseAPIClient"] = None
        try:
            if name == "finnhub":
                from backend.data_ingestion.finnhub_client import FinnhubClient
                client = FinnhubClient()
            elif name == "alpha_vantage":
                from backend.data_ingestion.alpha_vantage_client import AlphaVantageClient
                client = AlphaVantageClient()
            elif name == "polygon":
                from backend.data_ingestion.polygon_client import PolygonClient
                client = PolygonClient()
            elif name == "sec_edgar":
                from backend.data_ingestion.sec_edgar_client import SECEdgarClient
                client = SECEdgarClient()
        except Exception as e:  # noqa: BLE001 - log and degrade
            logger.warning(f"Smart fetcher: client {name!r} unavailable: {e}")
            client = None

        self._clients[name] = client
        return client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_stock_data(self, ticker: str, data_type: str) -> Dict[str, Any]:
        """Fetch stock data with intelligent source selection."""
        data_fetchers = {
            "price": self._fetch_price_data,
            "fundamentals": self._fetch_fundamentals,
            "news": self._fetch_news,
            "financials": self._fetch_financials,
            "earnings": self._fetch_earnings,
            "sentiment": self._fetch_sentiment,
        }
        fetcher = data_fetchers.get(data_type, self._fetch_generic)
        return await fetcher(ticker)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _unavailable(self, ticker: str, **extra: Any) -> Dict[str, Any]:
        """Sentinel response when no source can satisfy the request."""
        payload = {
            "ticker": ticker,
            "source": "unavailable",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        payload.update(extra)
        return payload

    async def _try_client(self, name: str, method: str, *args, **kwargs):
        """Call ``client.method(*args, **kwargs)`` if the client exists."""
        client = self._get_client(name)
        if client is None:
            return None
        func = getattr(client, method, None)
        if func is None:
            return None
        try:
            return await func(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - log and try next source
            logger.warning(f"Smart fetcher: {name}.{method}({args!r}) failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Concrete fetchers
    # ------------------------------------------------------------------

    async def _fetch_price_data(self, ticker: str) -> Dict[str, Any]:
        """Fetch current price data; tries finnhub → alpha_vantage → polygon."""
        quote = await self._try_client("finnhub", "get_quote", ticker)
        if quote:
            return {
                "ticker": ticker,
                "price": quote.get("c") or quote.get("current_price") or 0.0,
                "change": quote.get("d") or 0.0,
                "change_percent": quote.get("dp") or 0.0,
                "volume": quote.get("v") or 0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "finnhub",
            }

        av_quote = await self._try_client("alpha_vantage", "get_quote", ticker)
        if av_quote:
            return {
                "ticker": ticker,
                "price": av_quote.get("price", 0.0),
                "change": av_quote.get("change", 0.0),
                "change_percent": av_quote.get("change_percent", 0.0),
                "volume": av_quote.get("volume", 0),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "alpha_vantage",
            }

        last_quote = await self._try_client("polygon", "get_last_quote", ticker)
        if last_quote:
            return {
                "ticker": ticker,
                "price": last_quote.get("last_price", 0.0),
                "change": 0.0,
                "change_percent": 0.0,
                "volume": last_quote.get("volume", 0),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "polygon",
            }

        return self._unavailable(
            ticker,
            price=0.0,
            change=0.0,
            change_percent=0.0,
            volume=0,
        )

    async def _fetch_fundamentals(self, ticker: str) -> Dict[str, Any]:
        """Fetch fundamental metrics; finnhub basic_financials → alpha_vantage overview."""
        fin = await self._try_client("finnhub", "get_basic_financials", ticker)
        if fin and isinstance(fin, dict):
            metrics = fin.get("metric", {}) or {}
            return {
                "ticker": ticker,
                "pe_ratio": metrics.get("peNormalizedAnnual") or metrics.get("peBasicExclExtraTTM", 0.0),
                "market_cap": metrics.get("marketCapitalization", 0),
                "eps": metrics.get("epsAnnual") or metrics.get("epsBasicExclExtraItemsTTM", 0.0),
                "dividend_yield": metrics.get("dividendYieldIndicatedAnnual", 0.0),
                "source": "finnhub",
            }

        overview = await self._try_client("alpha_vantage", "get_company_overview", ticker)
        if overview:
            return {
                "ticker": ticker,
                "pe_ratio": float(overview.get("PERatio", 0.0) or 0.0),
                "market_cap": int(float(overview.get("MarketCapitalization", 0) or 0)),
                "eps": float(overview.get("EPS", 0.0) or 0.0),
                "dividend_yield": float(overview.get("DividendYield", 0.0) or 0.0),
                "source": "alpha_vantage",
            }

        return self._unavailable(
            ticker, pe_ratio=0.0, market_cap=0, eps=0.0, dividend_yield=0.0
        )

    async def _fetch_news(self, ticker: str) -> Dict[str, Any]:
        """Fetch news headlines; finnhub get_news is the canonical source."""
        articles = await self._try_client("finnhub", "get_news", ticker)
        if articles is not None:
            return {"ticker": ticker, "articles": articles, "source": "finnhub"}
        return self._unavailable(ticker, articles=[])

    async def _fetch_financials(self, ticker: str) -> Dict[str, Any]:
        """Fetch financial statements; SEC EDGAR is the authoritative source."""
        sec = self._get_client("sec_edgar")
        if sec is not None and hasattr(sec, "get_company_facts"):
            try:
                facts = await sec.get_company_facts(ticker)
                if facts:
                    return {"ticker": ticker, **facts, "source": "sec_edgar"}
            except Exception as e:  # noqa: BLE001
                logger.warning(f"SEC EDGAR facts for {ticker} failed: {e}")

        return self._unavailable(
            ticker,
            income_statement={},
            balance_sheet={},
            cash_flow={},
        )

    async def _fetch_earnings(self, ticker: str) -> Dict[str, Any]:
        """Fetch earnings data via Finnhub recommendations / price targets."""
        recs = await self._try_client("finnhub", "get_recommendations", ticker)
        target = await self._try_client("finnhub", "get_price_target", ticker)
        if recs is not None or target is not None:
            return {
                "ticker": ticker,
                "recommendations": recs or [],
                "price_target": target or {},
                "source": "finnhub",
            }
        return self._unavailable(
            ticker, next_earnings_date=None, eps_history=[],
        )

    async def _fetch_sentiment(self, ticker: str) -> Dict[str, Any]:
        """Fetch sentiment via Finnhub."""
        sent = await self._try_client("finnhub", "get_sentiment", ticker)
        if sent:
            return {
                "ticker": ticker,
                "sentiment_score": sent.get("sentiment", 0.0),
                "sentiment_label": sent.get("label", "neutral"),
                "source": "finnhub",
            }
        return self._unavailable(
            ticker, sentiment_score=0.0, sentiment_label="neutral",
        )

    async def _fetch_generic(self, ticker: str) -> Dict[str, Any]:
        """Unknown data_type — explicit unavailability."""
        return self._unavailable(ticker, data={})

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    async def get_available_sources(self) -> List[str]:
        """Get list of available data sources (only those that initialized)."""
        return [
            name
            for name in ("alpha_vantage", "finnhub", "polygon", "sec_edgar")
            if self._get_client(name) is not None
        ]

    async def get_source_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all data sources."""
        available = await self.get_available_sources()
        return {
            name: {
                "available": name in available,
                "rate_limit_remaining": None,
            }
            for name in ("alpha_vantage", "finnhub", "polygon", "sec_edgar")
        }


# Global instance
_smart_fetcher: Optional[SmartDataFetcher] = None


async def get_smart_fetcher() -> SmartDataFetcher:
    """Get or create the global smart data fetcher instance."""
    global _smart_fetcher
    if _smart_fetcher is None:
        _smart_fetcher = SmartDataFetcher()
    return _smart_fetcher
