"""
Stocks API Router - Production-Ready Implementation
Enhanced with real data integration, comprehensive error handling, and performance optimizations.

Business logic lives in backend.services.stocks_service.StocksService.
This module contains only route definitions, request/response schemas, and
thin handler functions that delegate to the service layer.
"""

from fastapi import APIRouter, Query, HTTPException, Depends, status, Path
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import date, datetime, timezone
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from backend.config.database import get_async_db_session
# NOTE: These module-level imports are required so that existing test patch
# paths (e.g. "backend.api.routers.stocks.get_real_time_quote" and
# "backend.api.routers.stocks.price_repository") continue to resolve.
from backend.repositories import price_repository  # noqa: F401 -- used by tests
from backend.models.unified_models import User
from backend.auth.oauth2 import get_current_user
from backend.config.settings import settings
from backend.utils.api_cache_decorators import (
    cache_stock_data,
    cache_analysis_result,
    api_cache,
)
from backend.models.api_response import ApiResponse, success_response
from backend.utils.response_utils import filter_response_fields
from backend.services.stocks_service import (
    stocks_service,
    get_real_time_quote,  # noqa: F401 -- re-exported for test patch paths
    fetch_company_overview,  # noqa: F401 -- re-exported
)
from backend.services.news_service import fetch_news

# Configure logging
logger = logging.getLogger(__name__)

# Import error handling utilities with fallback implementations
try:
    from backend.utils.enhanced_error_handling import (
        handle_api_error,
        validate_stock_symbol
    )
except ImportError:
    import re

    async def handle_api_error(error: Exception, operation: str, context: dict = None):
        """Fallback error handler that logs the error."""
        logger.error(f"API error during {operation}: {error}", exc_info=True)

    def validate_stock_symbol(symbol: str) -> bool:
        """Validate stock symbol format - fallback implementation."""
        if not symbol or not isinstance(symbol, str):
            return False
        symbol = symbol.strip().upper()
        return bool(re.match(r'^[A-Z]{1,5}$', symbol))

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic response / request models
# ---------------------------------------------------------------------------

class StockResponse(BaseModel):
    """Stock response model"""
    id: int
    symbol: str
    name: str
    exchange: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[int] = None
    is_active: bool
    is_tradable: bool
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = {"from_attributes": True}

    @classmethod
    def from_orm(cls, obj):
        """
        Build a StockResponse from a Stock ORM instance.

        Resolves relationship objects to plain strings and maps
        ``last_updated`` to the created/updated timestamp fields.
        """
        exchange_name = None
        if obj.exchange is not None:
            exchange_name = getattr(obj.exchange, "name", None) or getattr(obj.exchange, "code", None)

        sector_name = None
        if obj.sector is not None:
            sector_name = getattr(obj.sector, "name", None)

        industry_name = None
        if obj.industry is not None:
            industry_name = getattr(obj.industry, "name", None)

        last_updated = getattr(obj, "last_updated", None)

        return cls(
            id=obj.id,
            symbol=obj.symbol,
            name=obj.name,
            exchange=exchange_name,
            sector=sector_name,
            industry=industry_name,
            market_cap=int(obj.market_cap) if obj.market_cap is not None else None,
            is_active=obj.is_active,
            is_tradable=obj.is_tradable,
            created_at=last_updated,
            updated_at=last_updated,
        )


class StockDetailResponse(StockResponse):
    """Detailed stock response with additional fields"""
    shares_outstanding: Optional[int] = None
    float_shares: Optional[int] = None
    country: Optional[str] = None
    currency: Optional[str] = None
    ipo_date: Optional[date] = None
    description: Optional[str] = None
    website: Optional[str] = None
    employees: Optional[int] = None

    @classmethod
    def from_orm(cls, obj):
        """Extend the base from_orm to include detail-level fields."""
        base = StockResponse.from_orm(obj)
        return cls(
            **base.model_dump(),
            shares_outstanding=getattr(obj, "shares_outstanding", None),
            float_shares=getattr(obj, "float_shares", None),
            country=getattr(obj, "country", None),
            currency=getattr(obj, "currency", None),
            ipo_date=getattr(obj, "ipo_date", None),
            description=getattr(obj, "description", None),
            website=getattr(obj, "website", None),
            employees=getattr(obj, "employees", None),
        )


class PriceHistoryResponse(BaseModel):
    """Price history response model"""
    date: date
    open: float
    high: float
    low: float
    close: float
    adjusted_close: Optional[float] = None
    volume: int
    split_coefficient: Optional[float] = 1.0
    dividend_amount: Optional[float] = 0.0

    class Config:
        from_attributes = True


class StockQuoteResponse(BaseModel):
    """Enhanced real-time stock quote response"""
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    timestamp: datetime

    # Enhanced quote data
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    previous_close: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None

    # Market data
    market_cap: Optional[int] = None
    pe_ratio: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None
    avg_volume: Optional[int] = None

    # Data source info
    data_source: Optional[str] = None
    last_updated: Optional[datetime] = None
    is_real_time: bool = True


class StockSearchResponse(BaseModel):
    """Stock search result"""
    stocks: List[StockResponse]
    total_count: int
    page: int
    per_page: int


class AlertConditionEnum(str, Enum):
    """Supported price alert condition types."""
    ABOVE = "above"
    BELOW = "below"


class CreateAlertRequest(BaseModel):
    """Request body for creating a price threshold alert."""
    symbol: str = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Stock ticker symbol (e.g. AAPL)",
        json_schema_extra={"example": "AAPL"},
    )
    condition: AlertConditionEnum = Field(
        ...,
        description="Trigger when price goes 'above' or 'below' the threshold",
    )
    threshold_price: float = Field(
        ...,
        gt=0,
        description="Price threshold that triggers the alert",
        json_schema_extra={"example": 150.00},
    )
    is_recurring: bool = Field(
        False,
        description="If true, alert stays active after triggering",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "condition": "above",
                "threshold_price": 200.00,
                "is_recurring": False,
            }
        }


class AlertResponse(BaseModel):
    """Response model for a created price alert."""
    alert_id: str = Field(..., description="Unique alert identifier (UUID)")
    symbol: str
    condition: str
    threshold_price: float
    is_active: bool
    is_recurring: bool
    status: str = Field("active", description="Current status of the alert")
    created_at: datetime

    class Config:
        from_attributes = True


class SectorSummaryResponse(BaseModel):
    """Sector summary statistics"""
    sector: str
    stock_count: int
    total_market_cap: float
    avg_market_cap: float


class PerformanceResponse(BaseModel):
    """Stock performance data"""
    symbol: str
    start_price: float
    end_price: float
    performance_pct: float
    timeframe: str


# ---------------------------------------------------------------------------
# Route handlers  (thin -- delegate to stocks_service)
# ---------------------------------------------------------------------------

@router.get("")
@api_cache(data_type="db_query", ttl_override={'l1': 1800, 'l2': 7200, 'l3': 28800})
async def get_stocks(
    sector: Optional[str] = Query(None, description="Filter by sector"),
    min_market_cap: Optional[float] = Query(None, description="Minimum market cap"),
    max_market_cap: Optional[float] = Query(None, description="Maximum market cap"),
    is_active: bool = Query(True, description="Filter active stocks only"),
    limit: int = Query(100, le=500, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    sort_by: str = Query("market_cap", pattern="^(symbol|name|market_cap|created_at)$", description="Sort field"),
    order: str = Query("desc", pattern="^(asc|desc)$", description="Sort order"),
    fields: Optional[str] = Query(None, description="Comma-separated list of fields to include (e.g., symbol,name,market_cap)"),
    db: AsyncSession = Depends(get_async_db_session),
) -> ApiResponse[List[StockResponse]]:
    """
    Get list of stocks with optional filtering, sorting, and pagination.

    - **sector**: Filter stocks by sector
    - **min_market_cap**: Filter by minimum market capitalization
    - **max_market_cap**: Filter by maximum market capitalization
    - **is_active**: Include only active stocks
    - **limit**: Maximum number of results (up to 500)
    - **offset**: Number of results to skip for pagination
    - **sort_by**: Field to sort by
    - **order**: Sort order (asc or desc)
    """
    try:
        stocks = await stocks_service.get_stocks(
            sector=sector,
            min_market_cap=min_market_cap,
            max_market_cap=max_market_cap,
            is_active=is_active,
            limit=limit,
            offset=offset,
            sort_by=sort_by,
            order=order,
            db=db,
        )

        stock_responses = [StockResponse.from_orm(stock) for stock in stocks]

        if fields:
            response_dicts = [resp.dict() for resp in stock_responses]
            filtered_data = filter_response_fields(response_dicts, fields)
            return success_response(data=filtered_data)

        return success_response(data=stock_responses)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving stocks: {str(e)}",
        )


@router.get("/search")
@api_cache(data_type="db_query", ttl_override={'l1': 3600, 'l2': 14400, 'l3': 86400})
async def search_stocks(
    q: str = Query(..., min_length=1, alias="q", description="Search term (ticker symbol or company name)"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of results"),
    fields: Optional[str] = Query(None, description="Comma-separated list of fields to include (e.g., symbol,name,exchange)"),
    db: AsyncSession = Depends(get_async_db_session),
) -> ApiResponse[StockSearchResponse]:
    """
    Search stocks by ticker symbol or company name.

    - **q**: Search term (minimum 1 character)
    - **limit**: Maximum number of results (default 10, max 100)
    """
    try:
        stocks = await stocks_service.search_stocks(query=q, limit=limit, db=db)
        total_count = len(stocks)

        stock_responses = [StockResponse.from_orm(stock) for stock in stocks]

        if fields:
            filtered_stocks = filter_response_fields(
                [resp.dict() for resp in stock_responses], fields,
            )
            return success_response(data=StockSearchResponse(
                stocks=filtered_stocks, total_count=total_count, page=1, per_page=limit,
            ))

        return success_response(data=StockSearchResponse(
            stocks=stock_responses, total_count=total_count, page=1, per_page=limit,
        ))

    except Exception as e:
        logger.error(f"Error searching stocks for q='{q}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error searching stocks: {str(e)}",
        )


@router.post("/alerts", status_code=status.HTTP_201_CREATED)
async def create_price_alert(
    alert_request: CreateAlertRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db_session),
) -> ApiResponse[AlertResponse]:
    """
    Create a price threshold alert for a stock.

    Requires authentication.
    """
    try:
        symbol = alert_request.symbol.strip().upper()

        if not validate_stock_symbol(symbol):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid stock symbol format: '{symbol}'",
            )

        alert_data = await stocks_service.create_price_alert(
            user_id=current_user.id,
            symbol=symbol,
            condition=alert_request.condition.value,
            threshold_price=alert_request.threshold_price,
            is_recurring=alert_request.is_recurring,
            db=db,
        )

        return success_response(data=AlertResponse(**alert_data))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating price alert for {alert_request.symbol}: {e}")
        await handle_api_error(e, f"create price alert for {alert_request.symbol}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating price alert: {str(e)}",
        )


@router.get("/sectors")
async def get_sectors(
    db: AsyncSession = Depends(get_async_db_session),
) -> ApiResponse[List[str]]:
    """Get list of available sectors."""
    try:
        sectors = await stocks_service.get_sectors(db=db)
        return success_response(data=sectors)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving sectors: {str(e)}",
        )


@router.get("/sectors/summary")
async def get_sector_summary(
    db: AsyncSession = Depends(get_async_db_session),
) -> ApiResponse[List[SectorSummaryResponse]]:
    """Get sector summary with statistics."""
    try:
        sector_data = await stocks_service.get_sector_summary(db=db)
        summaries = [SectorSummaryResponse(**item) for item in sector_data]
        return success_response(data=summaries)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving sector summary: {str(e)}",
        )


@router.get("/top-performers")
async def get_top_performers(
    timeframe: str = Query("1d", pattern="^(1d|1w|1m|3m|6m|1y)$", description="Performance timeframe"),
    limit: int = Query(100, le=500, description="Maximum number of results"),
    db: AsyncSession = Depends(get_async_db_session),
) -> ApiResponse[List[PerformanceResponse]]:
    """
    Get top performing stocks by timeframe.

    - **timeframe**: Time period (1d, 1w, 1m, 3m, 6m, 1y)
    - **limit**: Maximum number of results
    """
    try:
        performers = await stocks_service.get_top_performers(
            timeframe=timeframe, limit=limit, db=db,
        )

        performance_list = [
            PerformanceResponse(
                symbol=perf['stock'].symbol,
                start_price=perf['start_price'],
                end_price=perf['end_price'],
                performance_pct=perf['performance_pct'],
                timeframe=timeframe,
            )
            for perf in performers
        ]
        return success_response(data=performance_list)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving top performers: {str(e)}",
        )


@router.get("/{symbol}")
async def get_stock_detail(
    symbol: str = Path(..., description="Stock symbol"),
    fields: Optional[str] = Query(None, description="Comma-separated list of fields to include"),
    db: AsyncSession = Depends(get_async_db_session),
) -> ApiResponse[StockDetailResponse]:
    """
    Get detailed information about a specific stock.

    - **symbol**: Stock symbol (e.g., AAPL, GOOGL)
    """
    try:
        stock = await stocks_service.get_stock_detail(symbol=symbol, db=db)

        if not stock:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Stock with symbol '{symbol}' not found",
            )

        response = StockDetailResponse.from_orm(stock)

        if fields:
            filtered_data = filter_response_fields(response.dict(), fields)
            return success_response(data=filtered_data)

        return success_response(data=response)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving stock details: {str(e)}",
        )


@router.get("/{symbol}/quote")
@cache_stock_data(ttl_hours=0.01)
async def get_stock_quote(
    symbol: str = Path(..., description="Stock symbol"),
    force_refresh: bool = Query(False, description="Force refresh from external APIs"),
    fields: Optional[str] = Query(None, description="Comma-separated list of fields to include"),
    db: AsyncSession = Depends(get_async_db_session),
) -> ApiResponse[StockQuoteResponse]:
    """
    Get enhanced real-time quote for a stock with fallback data sources.

    - **symbol**: Stock symbol (e.g., AAPL, GOOGL)
    - **force_refresh**: Force refresh from external APIs instead of cache
    """
    try:
        if not validate_stock_symbol(symbol):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid stock symbol format: '{symbol}'",
            )

        # Fetch real-time data here (not in the service) so that test
        # patches on "backend.api.routers.stocks.get_real_time_quote"
        # continue to intercept the call.
        real_time_data = await get_real_time_quote(symbol.upper())

        quote_data = await stocks_service.get_stock_quote(
            symbol=symbol, real_time_data=real_time_data, db=db,
        )

        quote_response = StockQuoteResponse(**quote_data)

        if fields:
            filtered_data = filter_response_fields(quote_response.dict(), fields)
            return success_response(data=filtered_data)

        return success_response(data=quote_response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving stock quote for {symbol}: {e}")
        await handle_api_error(e, f"retrieve quote for {symbol}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving stock quote: {str(e)}",
        )


@router.get("/{symbol}/history")
@api_cache(data_type="daily_prices", ttl_override={'l1': 3600, 'l2': 14400, 'l3': 86400})
async def get_stock_history(
    symbol: str = Path(..., description="Stock symbol"),
    start_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[date] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: Optional[int] = Query(252, le=1000, description="Maximum number of records"),
    db: AsyncSession = Depends(get_async_db_session),
) -> ApiResponse[List[PriceHistoryResponse]]:
    """
    Get historical price data for a stock.

    - **symbol**: Stock symbol
    - **start_date**: Start date for historical data
    - **end_date**: End date for historical data
    - **limit**: Maximum number of records (defaults to 1 year ~ 252 trading days)
    """
    try:
        price_history = await stocks_service.get_price_history(
            symbol=symbol, start_date=start_date, end_date=end_date, limit=limit, db=db,
        )

        if not price_history:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No price history found for symbol '{symbol}' in the specified date range",
            )

        history_responses = [PriceHistoryResponse.from_orm(price) for price in price_history]
        return success_response(data=history_responses)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving price history: {str(e)}",
        )


@router.get("/{symbol}/statistics")
@cache_analysis_result(ttl_hours=2)
async def get_stock_statistics(
    symbol: str = Path(..., description="Stock symbol"),
    days: int = Query(252, le=1000, description="Number of days for analysis"),
    db: AsyncSession = Depends(get_async_db_session),
) -> ApiResponse[Dict[str, Any]]:
    """
    Get comprehensive price statistics for a stock.

    - **symbol**: Stock symbol
    - **days**: Number of days to analyze (default 252 ~ 1 year)
    """
    try:
        statistics = await stocks_service.get_stock_statistics(
            symbol=symbol, days=days, db=db,
        )

        if not statistics:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No price data found for symbol '{symbol}'",
            )

        return success_response(data=statistics)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving stock statistics: {str(e)}",
        )


# ---------------------------------------------------------------------------
# Stock-detail routes consumed by the frontend Analysis page.
#
# Placed among the other "/{symbol}/..." routes (FastAPI matches the more
# specific literal paths first, so these never shadow the bare "/{symbol}"
# detail route). All payloads use camelCase keys matching the frontend
# TypeScript interfaces and are wrapped by ``success_response``.
#
# Data strategy (no fabricated financials): serve from our DB / existing
# services first, fall back to existing provider helpers only when needed,
# and otherwise return graceful EMPTY payloads. Representative/sample data is
# only ever produced under ``settings.DEMO_MODE`` (production default = real
# data only), mirroring the recommendations router pattern.
# ---------------------------------------------------------------------------


@router.get("/{symbol}/chart")
@api_cache(data_type="daily_prices", ttl_override={'l1': 3600, 'l2': 14400, 'l3': 86400})
async def get_stock_chart(
    symbol: str = Path(..., description="Stock symbol"),
    interval: str = Query("1d", description="Bar interval (e.g. 1d)"),
    db: AsyncSession = Depends(get_async_db_session),
) -> ApiResponse[Dict[str, Any]]:
    """
    Get historical OHLC bars for a stock chart.

    Serves from stored price history (``stocks_service.get_price_history``).
    Returns ``data: []`` when no history is stored yet — the frontend renders
    a graceful empty chart state.
    """
    try:
        if not validate_stock_symbol(symbol):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid stock symbol format: '{symbol}'",
            )

        symbol = symbol.upper()
        price_history = await stocks_service.get_price_history(
            symbol=symbol, start_date=None, end_date=None, limit=252, db=db,
        )

        # Repository returns newest-first; present chart bars oldest-first.
        bars = [
            {
                "date": (
                    price.date.isoformat()
                    if hasattr(price.date, "isoformat")
                    else str(price.date)
                ),
                "open": float(price.open),
                "high": float(price.high),
                "low": float(price.low),
                "close": float(price.close),
                "volume": int(price.volume),
            }
            for price in reversed(price_history or [])
        ]

        return success_response(data={
            "ticker": symbol,
            "interval": interval,
            "data": bars,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving chart for {symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving stock chart: {str(e)}",
        )


def _neutral_technical_indicators() -> Dict[str, Any]:
    """Graceful all-zero/neutral technical payload (no fabricated values)."""
    return {
        "rsi": 0,
        "macd": {"macd": 0, "signal": 0, "histogram": 0},
        "sma": {"sma20": 0, "sma50": 0, "sma200": 0},
        "ema": {"ema12": 0, "ema26": 0},
        "bollingerBands": {"upper": 0, "middle": 0, "lower": 0},
        "stochastic": {"k": 0, "d": 0},
        "atr": 0,
        "adx": 0,
        "obv": 0,
        "volumeProfile": [],
        "signals": {
            "trend": "neutral",
            "momentum": "weak",
            "volatility": "low",
            "recommendation": "hold",
        },
    }


@router.get("/{symbol}/technical")
@cache_analysis_result(ttl_hours=2)
async def get_stock_technical(
    symbol: str = Path(..., description="Stock symbol"),
    db: AsyncSession = Depends(get_async_db_session),
) -> ApiResponse[Dict[str, Any]]:
    """
    Compute technical indicators from stored price history.

    Reuses the indicator functions backing
    ``backend.analytics.technical_analysis`` (the same engine used by the
    analysis router). When there is insufficient stored history to compute,
    returns the structured payload with zeroed numeric fields and neutral
    signals rather than fabricating values.
    """
    try:
        if not validate_stock_symbol(symbol):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid stock symbol format: '{symbol}'",
            )

        symbol = symbol.upper()
        price_history = await stocks_service.get_price_history(
            symbol=symbol, start_date=None, end_date=None, limit=252, db=db,
        )

        # Need a reasonable window to compute meaningful indicators.
        if not price_history or len(price_history) < 35:
            return success_response(data=_neutral_technical_indicators())

        import numpy as np
        from backend.analytics.technical_analysis import (
            calculate_rsi,
            calculate_macd,
            calculate_ema,
            calculate_bollinger_bands,
            calculate_stochastic,
            calculate_atr,
            calculate_adx,
        )

        # Repository is newest-first; indicator functions expect chronological.
        ordered = list(reversed(price_history))
        close = np.array([float(p.close) for p in ordered], dtype=float)
        high = np.array([float(p.high) for p in ordered], dtype=float)
        low = np.array([float(p.low) for p in ordered], dtype=float)
        volume = np.array([float(p.volume) for p in ordered], dtype=float)

        def _sma(arr: "np.ndarray", window: int):
            """Simple moving average of the last *window* values, or None."""
            if len(arr) < window:
                return None
            return round(float(np.mean(arr[-window:])), 4)

        rsi_value = round(float(calculate_rsi(close)), 4)
        macd_data = calculate_macd(close)
        bb_data = calculate_bollinger_bands(close)
        stoch_data = calculate_stochastic(high, low, close)
        atr_value = round(float(calculate_atr(high, low, close)), 4)
        adx_data = calculate_adx(high, low, close)

        sma20 = _sma(close, 20)
        sma50 = _sma(close, 50)
        sma200 = _sma(close, 200)

        # On-Balance Volume from real closes/volumes.
        obv = 0.0
        for i in range(1, len(close)):
            if close[i] > close[i - 1]:
                obv += volume[i]
            elif close[i] < close[i - 1]:
                obv -= volume[i]

        # Volume profile: aggregate real volume into price buckets.
        volume_profile: List[Dict[str, float]] = []
        if len(close) >= 2 and float(np.max(close)) > float(np.min(close)):
            num_bins = 10
            edges = np.linspace(float(np.min(close)), float(np.max(close)), num_bins + 1)
            for b in range(num_bins):
                mask = (close >= edges[b]) & (
                    close <= edges[b + 1] if b == num_bins - 1 else close < edges[b + 1]
                )
                bucket_volume = float(np.sum(volume[mask]))
                if bucket_volume > 0:
                    volume_profile.append({
                        "price": round(float((edges[b] + edges[b + 1]) / 2), 4),
                        "volume": round(bucket_volume, 2),
                    })

        # Derive qualitative signals from the real computed indicators.
        last_close = float(close[-1])
        if sma50 is not None and sma20 is not None:
            if last_close > sma20 > sma50:
                trend = "bullish"
            elif last_close < sma20 < sma50:
                trend = "bearish"
            else:
                trend = "neutral"
        else:
            trend = "neutral"

        if rsi_value >= 60 or rsi_value <= 40:
            momentum = "strong"
        elif 45 <= rsi_value <= 55:
            momentum = "weak"
        else:
            momentum = "moderate"

        atr_pct = (atr_value / last_close) if last_close else 0.0
        if atr_pct >= 0.03:
            volatility = "high"
        elif atr_pct >= 0.015:
            volatility = "medium"
        else:
            volatility = "low"

        if trend == "bullish" and rsi_value < 70:
            recommendation = "buy" if rsi_value < 60 else "hold"
        elif trend == "bearish" and rsi_value > 30:
            recommendation = "sell" if rsi_value > 40 else "hold"
        else:
            recommendation = "hold"

        payload = {
            "rsi": rsi_value,
            "macd": {
                "macd": round(float(macd_data.get("macd", 0)), 4),
                "signal": round(float(macd_data.get("signal", 0)), 4),
                "histogram": round(float(macd_data.get("histogram", 0)), 4),
            },
            "sma": {
                "sma20": sma20 if sma20 is not None else 0,
                "sma50": sma50 if sma50 is not None else 0,
                "sma200": sma200 if sma200 is not None else 0,
            },
            "ema": {
                "ema12": round(float(calculate_ema(close, 12)), 4),
                "ema26": round(float(calculate_ema(close, 26)), 4),
            },
            "bollingerBands": {
                "upper": round(float(bb_data.get("upper", 0)), 4),
                "middle": round(float(bb_data.get("middle", 0)), 4),
                "lower": round(float(bb_data.get("lower", 0)), 4),
            },
            "stochastic": {
                "k": round(float(stoch_data.get("k", 0)), 4),
                "d": round(float(stoch_data.get("d", 0)), 4),
            },
            "atr": atr_value,
            "adx": round(float(adx_data.get("adx", 0)), 4),
            "obv": round(obv, 2),
            "volumeProfile": volume_profile,
            "signals": {
                "trend": trend,
                "momentum": momentum,
                "volatility": volatility,
                "recommendation": recommendation,
            },
        }

        return success_response(data=payload)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error computing technical indicators for {symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error computing technical indicators: {str(e)}",
        )


@router.get("/{symbol}/fundamental")
# NOTE: intentionally NOT decorated with @cache_analysis_result — that cache
# wrapper swallows the HTTPException(404) we raise for "no fundamentals stored"
# and returns None, which fails response validation as a 500. The DB read here
# is cheap, so we skip caching to keep the clean 404.
async def get_stock_fundamental(
    symbol: str = Path(..., description="Stock symbol"),
    db: AsyncSession = Depends(get_async_db_session),
) -> ApiResponse[Dict[str, Any]]:
    """
    Get fundamental financials for a stock from stored data.

    Reads the latest ``Fundamentals`` row from our database. Fields we have
    not yet ingested are returned as ``null`` (the frontend tolerates missing
    fundamentals). Returns HTTP 404 only when there is no fundamentals record
    at all for the symbol. No ratios are fabricated.
    """
    try:
        if not validate_stock_symbol(symbol):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid stock symbol format: '{symbol}'",
            )

        symbol = symbol.upper()
        fundamentals = await stocks_service.get_latest_fundamentals(symbol=symbol, db=db)

        if not fundamentals:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No fundamental data stored for symbol '{symbol}'",
            )

        def _num(value):
            """Coerce a stored Decimal/float to float, preserving None."""
            return float(value) if value is not None else None

        payload = {
            # Income statement
            "revenue": _num(fundamentals.revenue),
            # Growth metrics are not stored as columns yet — left null until ingested.
            "revenueGrowth": None,
            "earnings": _num(fundamentals.net_income),
            "earningsGrowth": None,
            "profitMargin": _num(fundamentals.net_margin),
            "operatingMargin": _num(fundamentals.operating_margin),
            # Returns / leverage ratios
            "roe": _num(fundamentals.roe),
            "roa": _num(fundamentals.roa),
            "debtToEquity": _num(fundamentals.debt_to_equity),
            "currentRatio": _num(fundamentals.current_ratio),
            "quickRatio": _num(fundamentals.quick_ratio),
            # Cash flow / valuation
            "freeCashFlow": _num(fundamentals.free_cash_flow),
            # Per-share book value is not a stored column; derive only from real
            # equity / shares when both are available, else null.
            "bookValue": None,
            "priceToBook": _num(fundamentals.pb_ratio),
            "priceToSales": _num(fundamentals.ps_ratio),
            "pegRatio": _num(fundamentals.peg_ratio),
            "forwardPE": _num(fundamentals.pe_ratio),
            # Dividend / ownership / short-interest are not stored yet — null.
            "dividendRate": None,
            "payoutRatio": None,
            "insiderOwnership": None,
            "institutionalOwnership": None,
            "shortInterest": None,
            # Analyst ratings are not stored yet — return an empty consensus
            # rather than fabricating analyst counts.
            "analystRating": {
                "consensus": "",
                "targetPrice": 0,
                "strongBuy": 0,
                "buy": 0,
                "hold": 0,
                "sell": 0,
                "strongSell": 0,
            },
        }

        # Derive book value per share only from genuinely stored real values.
        target_stock = await stocks_service.get_stock_detail(symbol=symbol, db=db)
        shares = getattr(target_stock, "shares_outstanding", None) if target_stock else None
        if fundamentals.total_equity is not None and shares:
            payload["bookValue"] = round(float(fundamentals.total_equity) / float(shares), 4)

        return success_response(data=payload)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving fundamentals for {symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving fundamental data: {str(e)}",
        )


@router.get("/{symbol}/news")
async def get_stock_news(
    symbol: str = Path(..., description="Stock symbol"),
    limit: int = Query(20, ge=1, le=50, description="Maximum number of articles"),
    db: AsyncSession = Depends(get_async_db_session),
) -> ApiResponse[List[Dict[str, Any]]]:
    """
    Get per-ticker financial news.

    Reuses ``backend.services.news_service.fetch_news`` (Finnhub -> NewsAPI ->
    MarketAux fallback chain with caching and keyword sentiment scoring).
    Returns an empty list when no provider returns articles.
    """
    try:
        if not validate_stock_symbol(symbol):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid stock symbol format: '{symbol}'",
            )

        symbol = symbol.upper()
        raw_articles = await fetch_news(symbols=[symbol], limit=limit)

        articles = [
            {
                "id": article["id"],
                "title": article.get("title", ""),
                "summary": article.get("description") or "",
                "url": article.get("url", ""),
                "source": article.get("source", ""),
                "publishedAt": (
                    article["published_at"].isoformat()
                    if hasattr(article.get("published_at"), "isoformat")
                    else str(article.get("published_at", ""))
                ),
                "sentiment": article.get("sentiment") or "neutral",
                "relevanceScore": (
                    abs(float(article["sentiment_score"]))
                    if article.get("sentiment_score") is not None
                    else 0
                ),
            }
            for article in (raw_articles or [])
        ]

        return success_response(data=articles)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving news for {symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving stock news: {str(e)}",
        )


@router.get("/{symbol}/options")
async def get_stock_options(
    symbol: str = Path(..., description="Stock symbol"),
    db: AsyncSession = Depends(get_async_db_session),
) -> ApiResponse[Dict[str, Any]]:
    """
    Get the options chain for a stock.

    Free-tier providers in use do not supply options data, so this returns a
    graceful empty chain. The structure matches the frontend interface so the
    options tab renders an empty state. No options data is fabricated; when an
    options provider is integrated this endpoint will populate from it.
    """
    try:
        if not validate_stock_symbol(symbol):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid stock symbol format: '{symbol}'",
            )

        symbol = symbol.upper()
        return success_response(data={
            "ticker": symbol,
            "expirationDates": [],
            "calls": [],
            "puts": [],
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving options for {symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving options chain: {str(e)}",
        )


@router.get("/{symbol}/similar")
@cache_analysis_result(ttl_hours=6)
async def get_similar_stocks(
    symbol: str = Path(..., description="Stock symbol"),
    limit: int = Query(6, ge=1, le=20, description="Maximum number of peers"),
    db: AsyncSession = Depends(get_async_db_session),
) -> ApiResponse[List[Dict[str, Any]]]:
    """
    Get similar / peer stocks computed from real data in our database.

    Peers are same-sector active stocks (excluding the symbol itself).
    ``changePercent`` is derived from each peer's latest stored prices and
    ``correlation`` is a real Pearson correlation of overlapping daily returns
    (``0.0`` when there is insufficient shared history — never fabricated).
    Returns an empty list when no peers exist.
    """
    try:
        if not validate_stock_symbol(symbol):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid stock symbol format: '{symbol}'",
            )

        symbol = symbol.upper()
        peers = await stocks_service.get_similar_stocks(symbol=symbol, limit=limit, db=db)
        return success_response(data=peers)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving similar stocks for {symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving similar stocks: {str(e)}",
        )


@router.post("/{symbol}/watchlist")
async def add_to_watchlist(
    symbol: str = Path(..., description="Stock symbol"),
    db: AsyncSession = Depends(get_async_db_session),
) -> Dict[str, Any]:
    """
    Add a stock to user's default watchlist.

    DEPRECATED: Use POST /api/watchlists/default/symbols/{symbol} instead.
    """
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={
            "message": "Authentication required. Use the watchlist API endpoints.",
            "redirect": f"/api/watchlists/default/symbols/{symbol.upper()}",
            "method": "POST",
            "note": "This endpoint is deprecated. Please use the authenticated watchlist API.",
        },
    )


@router.delete("/{symbol}/watchlist")
async def remove_from_watchlist(
    symbol: str = Path(..., description="Stock symbol"),
    db: AsyncSession = Depends(get_async_db_session),
) -> Dict[str, Any]:
    """
    Remove a stock from user's default watchlist.

    DEPRECATED: Use DELETE /api/watchlists/default/symbols/{symbol} instead.
    """
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={
            "message": "Authentication required. Use the watchlist API endpoints.",
            "redirect": f"/api/watchlists/default/symbols/{symbol.upper()}",
            "method": "DELETE",
            "note": "This endpoint is deprecated. Please use the authenticated watchlist API.",
        },
    )
