"""
Stocks Service
Business logic for stock data retrieval, quoting, and analysis.
Extracted from backend/api/routers/stocks.py to keep the router thin.
"""

import logging
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from backend.repositories import (
    stock_repository,
    price_repository,
    FilterCriteria,
    PaginationParams,
    SortParams,
    SortDirection,
)
from backend.repositories.alert_repository import alert_repository
from backend.data_ingestion.alpha_vantage_client import AlphaVantageClient
from backend.data_ingestion.finnhub_client import FinnhubClient
from backend.data_ingestion.polygon_client import PolygonClient
from backend.utils.api_cache_decorators import api_cache
from backend.config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# External data provider clients (initialized once at import time)
# ---------------------------------------------------------------------------

alpha_vantage_client = AlphaVantageClient() if settings.ALPHA_VANTAGE_API_KEY else None
finnhub_client = FinnhubClient() if settings.FINNHUB_API_KEY else None
try:
    polygon_client = PolygonClient() if settings.POLYGON_API_KEY else None
except Exception as e:
    logger.warning(f"Failed to initialize Polygon client: {e}")
    polygon_client = None


# ---------------------------------------------------------------------------
# Cached data-fetching helpers
# ---------------------------------------------------------------------------

@api_cache(
    data_type="real_time_quote",
    ttl_override={'l1': 60, 'l2': 300, 'l3': 1800},
    cost_tracking=True,
)
async def get_real_time_quote(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch real-time quote from available providers with intelligent caching."""
    try:
        if finnhub_client:
            return await finnhub_client.get_quote(symbol)
        if alpha_vantage_client:
            return await alpha_vantage_client.get_quote(symbol)
        if polygon_client:
            return await polygon_client.get_quote(symbol)
        return None
    except Exception as e:
        logger.error(f"Error fetching real-time quote for {symbol}: {e}")
        return None


@api_cache(
    data_type="company_overview",
    ttl_override={'l1': 7200, 'l2': 43200, 'l3': 604800},
    cost_tracking=True,
)
async def fetch_company_overview(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch company overview from available providers with intelligent caching."""
    try:
        if alpha_vantage_client:
            return await alpha_vantage_client.get_company_overview(symbol)
        if finnhub_client:
            return await finnhub_client.get_company_profile(symbol)
        return None
    except Exception as e:
        logger.error(f"Error fetching company overview for {symbol}: {e}")
        return None


# ---------------------------------------------------------------------------
# StocksService
# ---------------------------------------------------------------------------

class StocksService:
    """
    Business logic for all stock-related operations.

    Each public method corresponds to a route handler in the stocks router.
    The router delegates to these methods after parsing request parameters
    and returns the result wrapped in an ``ApiResponse``.
    """

    # ------------------------------------------------------------------
    # List / Search
    # ------------------------------------------------------------------

    async def get_stocks(
        self,
        *,
        sector: Optional[str],
        min_market_cap: Optional[float],
        max_market_cap: Optional[float],
        is_active: bool,
        limit: int,
        offset: int,
        sort_by: str,
        order: str,
        db: AsyncSession,
    ) -> List:
        """Return a filtered, sorted, paginated list of Stock ORM objects."""
        filters: list[FilterCriteria] = []

        if is_active:
            filters.append(FilterCriteria(field='is_active', operator='eq', value=True))
            filters.append(FilterCriteria(field='is_tradable', operator='eq', value=True))
        if sector:
            filters.append(FilterCriteria(field='sector', operator='eq', value=sector))
        if min_market_cap is not None:
            filters.append(FilterCriteria(field='market_cap', operator='gte', value=int(min_market_cap)))
        if max_market_cap is not None:
            filters.append(FilterCriteria(field='market_cap', operator='lte', value=int(max_market_cap)))

        sort_direction = SortDirection.DESC if order == "desc" else SortDirection.ASC
        sort_params = [SortParams(field=sort_by, direction=sort_direction)]
        pagination = PaginationParams(offset=offset, limit=limit)

        return await stock_repository.get_multi(
            filters=filters,
            sort_params=sort_params,
            pagination=pagination,
            session=db,
        )

    async def search_stocks(
        self,
        *,
        query: str,
        limit: int,
        db: AsyncSession,
    ) -> List:
        """Search stocks by symbol or company name."""
        return await stock_repository.search_stocks(
            query=query,
            limit=limit,
            session=db,
        )

    # ------------------------------------------------------------------
    # Detail
    # ------------------------------------------------------------------

    async def get_stock_detail(
        self,
        *,
        symbol: str,
        db: AsyncSession,
    ):
        """Return a single Stock ORM object or ``None``."""
        return await stock_repository.get_by_symbol(symbol, session=db)

    # ------------------------------------------------------------------
    # Quote  (real-time with database fallback)
    # ------------------------------------------------------------------

    async def get_stock_quote(
        self,
        *,
        symbol: str,
        real_time_data: Optional[Dict[str, Any]],
        db: AsyncSession,
    ) -> Dict[str, Any]:
        """
        Build a quote dict for *symbol*.

        The caller is responsible for fetching ``real_time_data`` from the
        external provider (so that test patches on the module-level
        ``get_real_time_quote`` function continue to work).  When
        ``real_time_data`` is ``None``, this method falls back to the
        latest database price.

        Returns a flat dict that maps directly onto ``StockQuoteResponse``
        field names.
        """
        symbol = symbol.upper()
        logger.info(f"Fetching quote for {symbol}")

        if real_time_data:
            data_source = real_time_data.get('source', 'external_api')
            return self._build_quote_from_external(symbol, real_time_data, data_source)

        # Fallback to database
        logger.info(f"Falling back to database for {symbol}")
        return await self._build_quote_from_db(symbol, db)

    def _build_quote_from_external(
        self,
        symbol: str,
        quote_data: Dict[str, Any],
        data_source: str,
    ) -> Dict[str, Any]:
        """Transform raw external provider data into a normalised quote dict."""
        current_price = float(quote_data.get('price', quote_data.get('c', 0)))
        previous_close = float(quote_data.get('previous_close', quote_data.get('pc', current_price)))

        change = current_price - previous_close if previous_close else 0.0
        change_percent = (change / previous_close * 100) if previous_close else 0.0

        return {
            "symbol": symbol,
            "price": current_price,
            "change": change,
            "change_percent": change_percent,
            "volume": int(quote_data.get('volume', quote_data.get('v', 0))),
            "timestamp": datetime.now(timezone.utc),
            "open": float(quote_data.get('open', quote_data.get('o'))) if quote_data.get('open') or quote_data.get('o') else None,
            "high": float(quote_data.get('high', quote_data.get('h'))) if quote_data.get('high') or quote_data.get('h') else None,
            "low": float(quote_data.get('low', quote_data.get('l'))) if quote_data.get('low') or quote_data.get('l') else None,
            "previous_close": previous_close if previous_close != current_price else None,
            "bid": float(quote_data.get('bid')) if quote_data.get('bid') else None,
            "ask": float(quote_data.get('ask')) if quote_data.get('ask') else None,
            "fifty_two_week_high": float(quote_data.get('52_week_high')) if quote_data.get('52_week_high') else None,
            "fifty_two_week_low": float(quote_data.get('52_week_low')) if quote_data.get('52_week_low') else None,
            "pe_ratio": float(quote_data.get('pe')) if quote_data.get('pe') else None,
            "data_source": data_source,
            "last_updated": datetime.now(timezone.utc),
            "is_real_time": True,
        }

    async def _build_quote_from_db(
        self,
        symbol: str,
        db: AsyncSession,
    ) -> Dict[str, Any]:
        """Build a quote dict from database price history."""
        from fastapi import HTTPException, status

        stock = await stock_repository.get_by_symbol(symbol, session=db)
        if not stock:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Stock '{symbol}' not found in database",
            )

        latest_price = await price_repository.get_latest_price(symbol, session=db)
        if not latest_price:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No price data found for symbol '{symbol}'",
            )

        previous_price = await price_repository.get_previous_price(symbol, latest_price.date, session=db)
        previous_close = float(previous_price.close) if previous_price else float(latest_price.close)

        current_price = float(latest_price.close)
        change = current_price - previous_close
        change_percent = (change / previous_close * 100) if previous_close else 0.0

        return {
            "symbol": symbol,
            "price": current_price,
            "change": change,
            "change_percent": change_percent,
            "volume": latest_price.volume,
            "timestamp": datetime.combine(latest_price.date, datetime.min.time()),
            "open": float(latest_price.open),
            "high": float(latest_price.high),
            "low": float(latest_price.low),
            "previous_close": previous_close if previous_price else None,
            "market_cap": stock.market_cap,
            "data_source": "database",
            "last_updated": datetime.now(timezone.utc),
            "is_real_time": False,
        }

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    async def get_price_history(
        self,
        *,
        symbol: str,
        start_date: Optional[date],
        end_date: Optional[date],
        limit: Optional[int],
        db: AsyncSession,
    ) -> List:
        """Return price history ORM rows for *symbol* within a date range."""
        if not end_date:
            end_date = date.today()
        if not start_date:
            start_date = end_date - timedelta(days=365)

        return await price_repository.get_price_history(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            session=db,
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    async def get_stock_statistics(
        self,
        *,
        symbol: str,
        days: int,
        db: AsyncSession,
    ) -> Optional[Dict[str, Any]]:
        """Return price statistics dict (or ``None`` when no data exists)."""
        statistics = await price_repository.get_price_statistics(
            symbol=symbol,
            days=days,
            session=db,
        )
        if not statistics:
            return None

        volatility = await price_repository.get_volatility(
            symbol=symbol,
            days=min(days, 30),
            session=db,
        )
        if volatility is not None:
            statistics['volatility_annualized'] = volatility

        return statistics

    # ------------------------------------------------------------------
    # Sectors
    # ------------------------------------------------------------------

    async def get_sectors(self, *, db: AsyncSession) -> List[str]:
        """Return a flat list of sector name strings."""
        sector_summary = await stock_repository.get_sector_summary(session=db)
        return [item['sector'] for item in sector_summary if item['sector']]

    async def get_sector_summary(self, *, db: AsyncSession) -> List[Dict[str, Any]]:
        """Return sector summary dicts with statistics."""
        return await stock_repository.get_sector_summary(session=db)

    # ------------------------------------------------------------------
    # Top performers
    # ------------------------------------------------------------------

    async def get_top_performers(
        self,
        *,
        timeframe: str,
        limit: int,
        db: AsyncSession,
    ) -> List[Dict[str, Any]]:
        """Return a list of ``{stock, start_price, end_price, performance_pct}`` dicts."""
        return await stock_repository.get_top_performers(
            timeframe=timeframe,
            limit=limit,
            session=db,
        )

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------

    async def create_price_alert(
        self,
        *,
        user_id: int,
        symbol: str,
        condition: str,
        threshold_price: float,
        is_recurring: bool,
        db: AsyncSession,
    ) -> Dict[str, Any]:
        """
        Persist a new price alert and return a dict suitable for AlertResponse.

        Raises ``HTTPException`` if the stock does not exist.
        """
        from fastapi import HTTPException, status

        stock = await stock_repository.get_by_symbol(symbol, session=db)
        if not stock:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Stock with symbol '{symbol}' not found",
            )

        condition_payload = {
            "type": "price_threshold",
            "condition": condition,
            "threshold_price": threshold_price,
        }

        alert = await alert_repository.create(
            data={
                "user_id": user_id,
                "stock_id": stock.id,
                "alert_type": "price_threshold",
                "condition": condition_payload,
                "is_active": True,
                "is_recurring": is_recurring,
            },
            session=db,
        )

        return {
            "alert_id": alert.alert_id,
            "symbol": symbol,
            "condition": condition,
            "threshold_price": threshold_price,
            "is_active": alert.is_active,
            "is_recurring": alert.is_recurring,
            "status": "active",
            "created_at": alert.created_at or datetime.now(timezone.utc),
        }


# Module-level singleton (matches the pattern used by other services)
stocks_service = StocksService()
