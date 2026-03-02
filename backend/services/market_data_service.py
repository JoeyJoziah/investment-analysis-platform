"""
Market Data Service
Fetches real-time and historical stock prices from multiple providers.
Provider fallback chain: Finnhub -> Polygon -> Alpha Vantage -> FMP
"""

import asyncio
import json
import logging
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx

from backend.config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Simple in-memory cache (TTL-based, avoids Redis dependency for service layer)
# ---------------------------------------------------------------------------

_price_cache: Dict[str, Tuple[float, object]] = {}
_PRICE_CACHE_TTL = 60  # 1 minute for real-time quotes


def _cache_get(key: str) -> Optional[object]:
    entry = _price_cache.get(key)
    if entry is None:
        return None
    expires_at, value = entry
    if time.monotonic() > expires_at:
        del _price_cache[key]
        return None
    return value


def _cache_set(key: str, value: object, ttl: int = _PRICE_CACHE_TTL) -> None:
    _price_cache[key] = (time.monotonic() + ttl, value)


# ---------------------------------------------------------------------------
# Canonical price dict shape
# ---------------------------------------------------------------------------

def _make_price_record(
    symbol: str,
    current_price: float,
    open_price: float = 0.0,
    high: float = 0.0,
    low: float = 0.0,
    previous_close: float = 0.0,
    volume: int = 0,
    change: float = 0.0,
    percent_change: float = 0.0,
    provider: str = "unknown",
    timestamp: Optional[datetime] = None,
) -> Dict[str, Any]:
    return {
        "symbol": symbol,
        "current_price": current_price,
        "open": open_price,
        "high": high,
        "low": low,
        "previous_close": previous_close,
        "volume": volume,
        "change": change,
        "percent_change": percent_change,
        "provider": provider,
        "timestamp": (timestamp or datetime.now(timezone.utc)).isoformat(),
    }


# ---------------------------------------------------------------------------
# Provider-specific fetchers
# ---------------------------------------------------------------------------

async def _fetch_finnhub_quote(symbol: str, client: httpx.AsyncClient) -> Optional[Dict[str, Any]]:
    """Fetch real-time quote from Finnhub (60 calls/minute free tier)."""
    api_key = settings.FINNHUB_API_KEY
    if not api_key:
        logger.warning("FINNHUB_API_KEY not configured - skipping Finnhub")
        return None

    try:
        resp = await client.get(
            "https://finnhub.io/api/v1/quote",
            params={"symbol": symbol, "token": api_key},
            timeout=10,
        )
        if resp.status_code != 200:
            logger.warning(f"Finnhub quote returned HTTP {resp.status_code} for {symbol}")
            return None

        data = resp.json()
        current = data.get("c", 0)
        if not current:
            return None

        return _make_price_record(
            symbol=symbol,
            current_price=float(current),
            open_price=float(data.get("o", 0)),
            high=float(data.get("h", 0)),
            low=float(data.get("l", 0)),
            previous_close=float(data.get("pc", 0)),
            volume=0,  # Finnhub quote endpoint does not include volume
            change=float(data.get("d", 0)),
            percent_change=float(data.get("dp", 0)),
            provider="finnhub",
            timestamp=datetime.fromtimestamp(data.get("t", time.time()), tz=timezone.utc),
        )
    except Exception as exc:
        logger.warning(f"Finnhub quote fetch failed for {symbol}: {exc}")
        return None


async def _fetch_polygon_quote(symbol: str, client: httpx.AsyncClient) -> Optional[Dict[str, Any]]:
    """Fetch previous-day close from Polygon.io (5 calls/minute free tier)."""
    api_key = settings.POLYGON_API_KEY
    if not api_key:
        logger.warning("POLYGON_API_KEY not configured - skipping Polygon")
        return None

    try:
        # Use the previous close endpoint (free tier compatible)
        resp = await client.get(
            f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev",
            params={"adjusted": "true", "apiKey": api_key},
            timeout=10,
        )
        if resp.status_code != 200:
            logger.warning(f"Polygon quote returned HTTP {resp.status_code} for {symbol}")
            return None

        data = resp.json()
        results = data.get("results", [])
        if not results:
            return None

        result = results[0]
        close_price = float(result.get("c", 0))
        if not close_price:
            return None

        return _make_price_record(
            symbol=symbol,
            current_price=close_price,
            open_price=float(result.get("o", 0)),
            high=float(result.get("h", 0)),
            low=float(result.get("l", 0)),
            previous_close=float(result.get("pc", 0)) if result.get("pc") else close_price,
            volume=int(result.get("v", 0)),
            provider="polygon",
        )
    except Exception as exc:
        logger.warning(f"Polygon quote fetch failed for {symbol}: {exc}")
        return None


async def _fetch_alpha_vantage_quote(symbol: str, client: httpx.AsyncClient) -> Optional[Dict[str, Any]]:
    """Fetch quote from Alpha Vantage (25 calls/day free tier)."""
    api_key = settings.ALPHA_VANTAGE_API_KEY
    if not api_key:
        logger.warning("ALPHA_VANTAGE_API_KEY not configured - skipping Alpha Vantage")
        return None

    try:
        resp = await client.get(
            "https://www.alphavantage.co/query",
            params={"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": api_key},
            timeout=10,
        )
        if resp.status_code != 200:
            logger.warning(f"Alpha Vantage returned HTTP {resp.status_code} for {symbol}")
            return None

        data = resp.json()
        quote = data.get("Global Quote", {})
        price_str = quote.get("05. price", "")
        if not price_str:
            # Check for API limit message
            if "Note" in data or "Information" in data:
                logger.warning(f"Alpha Vantage rate limit hit for {symbol}")
            return None

        current = float(price_str)
        change_pct_str = quote.get("10. change percent", "0%").replace("%", "")

        return _make_price_record(
            symbol=symbol,
            current_price=current,
            open_price=float(quote.get("02. open", 0)),
            high=float(quote.get("03. high", 0)),
            low=float(quote.get("04. low", 0)),
            previous_close=float(quote.get("08. previous close", 0)),
            volume=int(quote.get("06. volume", 0)),
            change=float(quote.get("09. change", 0)),
            percent_change=float(change_pct_str) if change_pct_str else 0.0,
            provider="alpha_vantage",
        )
    except Exception as exc:
        logger.warning(f"Alpha Vantage quote fetch failed for {symbol}: {exc}")
        return None


async def _fetch_fmp_quote(symbol: str, client: httpx.AsyncClient) -> Optional[Dict[str, Any]]:
    """Fetch quote from Financial Modeling Prep (250 calls/day free tier)."""
    api_key = settings.FMP_API_KEY
    if not api_key:
        logger.warning("FMP_API_KEY not configured - skipping FMP")
        return None

    try:
        resp = await client.get(
            f"https://financialmodelingprep.com/api/v3/quote/{symbol}",
            params={"apikey": api_key},
            timeout=10,
        )
        if resp.status_code != 200:
            logger.warning(f"FMP returned HTTP {resp.status_code} for {symbol}")
            return None

        data = resp.json()
        if not isinstance(data, list) or not data:
            return None

        item = data[0]
        current = float(item.get("price", 0))
        if not current:
            return None

        return _make_price_record(
            symbol=symbol,
            current_price=current,
            open_price=float(item.get("open", 0)),
            high=float(item.get("dayHigh", 0)),
            low=float(item.get("dayLow", 0)),
            previous_close=float(item.get("previousClose", 0)),
            volume=int(item.get("volume", 0)),
            change=float(item.get("change", 0)),
            percent_change=float(item.get("changesPercentage", 0)),
            provider="fmp",
        )
    except Exception as exc:
        logger.warning(f"FMP quote fetch failed for {symbol}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def get_stock_price(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Fetch the current price for a single symbol using the provider fallback chain.

    Fallback order: Finnhub -> Polygon -> Alpha Vantage -> FMP

    Results are cached in-memory for 60 seconds.

    Args:
        symbol: Stock ticker symbol (e.g., "AAPL").

    Returns:
        Price record dict or None if all providers fail.
    """
    symbol = symbol.upper()
    cache_key = f"price:{symbol}"
    cached = _cache_get(cache_key)
    if cached is not None:
        logger.debug(f"Price cache hit for {symbol}")
        return cached  # type: ignore[return-value]

    async with httpx.AsyncClient() as client:
        for fetcher in (
            _fetch_finnhub_quote,
            _fetch_polygon_quote,
            _fetch_alpha_vantage_quote,
            _fetch_fmp_quote,
        ):
            result = await fetcher(symbol, client)
            if result is not None:
                _cache_set(cache_key, result, ttl=_PRICE_CACHE_TTL)
                return result

    logger.error(f"All price providers failed for {symbol}")
    return None


async def get_stock_prices_batch(symbols: List[str]) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Fetch prices for multiple symbols concurrently.

    Args:
        symbols: List of stock ticker symbols.

    Returns:
        Dict mapping symbol -> price record (or None on failure).
    """
    tasks = [get_stock_price(sym) for sym in symbols]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    output: Dict[str, Optional[Dict[str, Any]]] = {}
    for sym, result in zip(symbols, results):
        if isinstance(result, Exception):
            logger.warning(f"Price fetch exception for {sym}: {result}")
            output[sym] = None
        else:
            output[sym] = result  # type: ignore[assignment]
    return output


async def update_prices_in_db(
    symbol: str,
    price_data: Dict[str, Any],
    db_session,  # sqlalchemy.orm.Session (sync)
) -> bool:
    """
    Persist a price record to the database.

    Uses the sync SQLAlchemy session expected by the Celery task context.
    Creates or updates today's PriceHistory record for the given symbol.

    Args:
        symbol: Stock ticker symbol.
        price_data: Price record dict from get_stock_price().
        db_session: Synchronous SQLAlchemy Session.

    Returns:
        True on success, False on failure.
    """
    from backend.models.unified_models import Stock, PriceHistory
    from sqlalchemy import and_

    try:
        stock = db_session.query(Stock).filter(Stock.symbol == symbol).first()
        if not stock:
            logger.warning(f"Stock {symbol} not found in database - skipping price update")
            return False

        today = date.today()
        current_price = price_data.get("current_price", 0)
        volume = price_data.get("volume", 0)
        open_price = price_data.get("open", current_price)
        high_price = price_data.get("high", current_price)
        low_price = price_data.get("low", current_price)

        if not current_price:
            logger.warning(f"Price data for {symbol} has zero/null price - skipping")
            return False

        existing = db_session.query(PriceHistory).filter(
            and_(
                PriceHistory.stock_id == stock.id,
                PriceHistory.date == today,
            )
        ).first()

        if existing:
            existing.close = current_price
            existing.high = max(existing.high or current_price, current_price)
            existing.low = min(existing.low or current_price, current_price)
            if volume:
                existing.volume = volume
        else:
            price_record = PriceHistory(
                stock_id=stock.id,
                date=today,
                open=open_price,
                high=high_price,
                low=low_price,
                close=current_price,
                volume=volume,
            )
            db_session.add(price_record)

        # Keep the stock's last_price_update timestamp fresh
        if hasattr(stock, "last_price_update"):
            stock.last_price_update = datetime.now(timezone.utc)

        db_session.commit()
        logger.info(f"Price updated in DB for {symbol}: {current_price}")
        return True

    except Exception as exc:
        logger.error(f"DB price update failed for {symbol}: {exc}")
        try:
            db_session.rollback()
        except Exception:
            pass
        return False
