"""
Stocks API Router - Production-Ready Implementation
Enhanced with real data integration, comprehensive error handling, and performance optimizations.
"""

from fastapi import APIRouter, Query, HTTPException, Depends, status, Path, BackgroundTasks
from typing import List, Optional, Dict, Any
from datetime import date, datetime, timedelta
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
import logging
import asyncio
from decimal import Decimal

from backend.config.database import get_async_db_session
from backend.repositories import (
    stock_repository, 
    price_repository, 
    FilterCriteria, 
    PaginationParams, 
    SortParams,
    SortDirection
)
from backend.models.unified_models import Stock as StockModel, PriceHistory as PriceHistoryModel
from backend.data_ingestion.alpha_vantage_client import AlphaVantageClient
from backend.data_ingestion.finnhub_client import FinnhubClient
from backend.data_ingestion.polygon_client import PolygonClient
from backend.utils.api_cache_decorators import (
    cache_stock_data, 
    cache_analysis_result,
    api_cache,
    generate_cache_key
)
from backend.utils.database_query_cache import cached_query
# TODO: Fix these imports - functions don't exist in enhanced_error_handling
from backend.utils.enhanced_error_handling import (
    handle_api_error,
    validate_stock_symbol
)
from backend.config.settings import settings

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize data clients
alpha_vantage_client = AlphaVantageClient() if settings.ALPHA_VANTAGE_API_KEY else None
finnhub_client = FinnhubClient() if settings.FINNHUB_API_KEY else None
try:
    polygon_client = PolygonClient() if settings.POLYGON_API_KEY else None
except Exception as e:
    logger.warning(f"Failed to initialize Polygon client: {e}")
    polygon_client = None

# Helper functions for data fetching with intelligent caching
@api_cache(
    data_type="real_time_quote", 
    ttl_override={'l1': 60, 'l2': 300, 'l3': 1800},
    cost_tracking=True
)
async def get_real_time_quote(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch real-time quote from available providers with intelligent caching"""
    try:
        # Try Finnhub first (fastest for quotes)
        if finnhub_client:
            return await finnhub_client.get_quote(symbol)
        
        # Fallback to Alpha Vantage
        if alpha_vantage_client:
            return await alpha_vantage_client.get_quote(symbol)
        
        # Last resort: Polygon
        if polygon_client:
            return await polygon_client.get_quote(symbol)
        
        return None
    except Exception as e:
        logger.error(f"Error fetching real-time quote for {symbol}: {e}")
        return None

@api_cache(
    data_type="company_overview",
    ttl_override={'l1': 7200, 'l2': 43200, 'l3': 604800},  # 2hr/12hr/7days
    cost_tracking=True
)
async def fetch_company_overview(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch company overview from available providers with intelligent caching"""
    try:
        # Alpha Vantage has good company overview data
        if alpha_vantage_client:
            return await alpha_vantage_client.get_company_overview(symbol)
        
        # Finnhub company profile
        if finnhub_client:
            return await finnhub_client.get_company_profile(symbol)
        
        return None
    except Exception as e:
        logger.error(f"Error fetching company overview for {symbol}: {e}")
        return None

# Enhanced Pydantic response models
class StockResponse(BaseModel):
    """Stock response model"""
    id: int
    symbol: str
    name: str
    exchange: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[int] = None
    is_active: bool
    is_tradable: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class StockDetailResponse(StockResponse):
    """Detailed stock response with additional fields"""
    shares_outstanding: Optional[int] = None
    float_shares: Optional[int] = None
    country: str
    currency: str
    ipo_date: Optional[date] = None
    description: Optional[str] = None
    website: Optional[str] = None
    employees: Optional[int] = None


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


@router.get("", response_model=List[StockResponse])
@api_cache(data_type="db_query", ttl_override={'l1': 1800, 'l2': 7200, 'l3': 28800})
async def get_stocks(
    sector: Optional[str] = Query(None, description="Filter by sector"),
    min_market_cap: Optional[float] = Query(None, description="Minimum market cap"),
    max_market_cap: Optional[float] = Query(None, description="Maximum market cap"),
    is_active: bool = Query(True, description="Filter active stocks only"),
    limit: int = Query(100, le=500, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    sort_by: str = Query("market_cap", regex="^(symbol|name|market_cap|created_at)$", description="Sort field"),
    order: str = Query("desc", regex="^(asc|desc)$", description="Sort order"),
    db: AsyncSession = Depends(get_async_db_session)
) -> List[StockResponse]:
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
        # Build filters
        filters = []
        
        if is_active:
            filters.append(FilterCriteria(field='is_active', operator='eq', value=True))
            filters.append(FilterCriteria(field='is_tradable', operator='eq', value=True))
        
        if sector:
            filters.append(FilterCriteria(field='sector', operator='eq', value=sector))
        
        if min_market_cap is not None:
            filters.append(FilterCriteria(field='market_cap', operator='gte', value=int(min_market_cap)))
        
        if max_market_cap is not None:
            filters.append(FilterCriteria(field='market_cap', operator='lte', value=int(max_market_cap)))
        
        # Sort parameters
        sort_direction = SortDirection.DESC if order == "desc" else SortDirection.ASC
        sort_params = [SortParams(field=sort_by, direction=sort_direction)]
        
        # Pagination
        pagination = PaginationParams(offset=offset, limit=limit)
        
        # Get stocks from repository
        stocks = await stock_repository.get_multi(
            filters=filters,
            sort_params=sort_params,
            pagination=pagination,
            session=db
        )
        
        return [StockResponse.from_orm(stock) for stock in stocks]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving stocks: {str(e)}"
        )


@router.get("/search", response_model=StockSearchResponse)
@api_cache(data_type="db_query", ttl_override={'l1': 3600, 'l2': 14400, 'l3': 86400})
async def search_stocks(
    query: str = Query(..., min_length=1, description="Search query (symbol or company name)"),
    limit: int = Query(50, le=100, description="Maximum number of results"),
    db: AsyncSession = Depends(get_async_db_session)
) -> StockSearchResponse:
    """
    Search stocks by symbol or company name.
    
    - **query**: Search term (minimum 1 character)
    - **limit**: Maximum number of results
    """
    try:
        stocks = await stock_repository.search_stocks(
            query=query,
            limit=limit,
            session=db
        )
        
        # Get total count for pagination info
        total_count = len(stocks)
        
        return StockSearchResponse(
            stocks=[StockResponse.from_orm(stock) for stock in stocks],
            total_count=total_count,
            page=1,
            per_page=limit
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error searching stocks: {str(e)}"
        )


@router.get("/sectors", response_model=List[str])
async def get_sectors(
    db: AsyncSession = Depends(get_async_db_session)
) -> List[str]:
    """Get list of available sectors."""
    try:
        sector_summary = await stock_repository.get_sector_summary(session=db)
        return [item['sector'] for item in sector_summary if item['sector']]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving sectors: {str(e)}"
        )


@router.get("/sectors/summary", response_model=List[SectorSummaryResponse])
async def get_sector_summary(
    db: AsyncSession = Depends(get_async_db_session)
) -> List[SectorSummaryResponse]:
    """Get sector summary with statistics."""
    try:
        sector_data = await stock_repository.get_sector_summary(session=db)
        return [SectorSummaryResponse(**item) for item in sector_data]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving sector summary: {str(e)}"
        )


@router.get("/top-performers", response_model=List[PerformanceResponse])
async def get_top_performers(
    timeframe: str = Query("1d", regex="^(1d|1w|1m|3m|6m|1y)$", description="Performance timeframe"),
    limit: int = Query(100, le=500, description="Maximum number of results"),
    db: AsyncSession = Depends(get_async_db_session)
) -> List[PerformanceResponse]:
    """
    Get top performing stocks by timeframe.
    
    - **timeframe**: Time period (1d, 1w, 1m, 3m, 6m, 1y)
    - **limit**: Maximum number of results
    """
    try:
        performers = await stock_repository.get_top_performers(
            timeframe=timeframe,
            limit=limit,
            session=db
        )
        
        return [
            PerformanceResponse(
                symbol=perf['stock'].symbol,
                start_price=perf['start_price'],
                end_price=perf['end_price'],
                performance_pct=perf['performance_pct'],
                timeframe=timeframe
            )
            for perf in performers
        ]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving top performers: {str(e)}"
        )


@router.get("/{symbol}", response_model=StockDetailResponse)
async def get_stock_detail(
    symbol: str = Path(..., description="Stock symbol"),
    db: AsyncSession = Depends(get_async_db_session)
) -> StockDetailResponse:
    """
    Get detailed information about a specific stock.
    
    - **symbol**: Stock symbol (e.g., AAPL, GOOGL)
    """
    try:
        stock = await stock_repository.get_by_symbol(symbol, session=db)
        
        if not stock:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Stock with symbol '{symbol}' not found"
            )
        
        return StockDetailResponse.from_orm(stock)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving stock details: {str(e)}"
        )


@router.get("/{symbol}/quote", response_model=StockQuoteResponse)
@cache_stock_data(ttl_hours=0.01)  # Cache for ~30 seconds for real-time data
async def get_stock_quote(
    symbol: str = Path(..., description="Stock symbol"),
    force_refresh: bool = Query(False, description="Force refresh from external APIs"),
    db: AsyncSession = Depends(get_async_db_session)
) -> StockQuoteResponse:
    """
    Get enhanced real-time quote for a stock with fallback data sources.
    
    - **symbol**: Stock symbol (e.g., AAPL, GOOGL)
    - **force_refresh**: Force refresh from external APIs instead of cache
    """
    try:
        # Validate stock symbol format
        if not validate_stock_symbol(symbol):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid stock symbol format: '{symbol}'"
            )
        
        symbol = symbol.upper()
        logger.info(f"Fetching quote for {symbol}")
        
        # Try to get real-time data from external APIs first
        real_time_data = None
        data_source = "database"
        
        if not force_refresh:
            # Check cache first
            cache_key = generate_cache_key(f"quote:{symbol}")
            # Cache logic would be handled by the decorator
        
        if force_refresh or not real_time_data:
            real_time_data = await get_real_time_quote(symbol)
            if real_time_data:
                data_source = real_time_data.get('source', 'external_api')
        
        # If we have real-time data, use it
        if real_time_data:
            quote_data = real_time_data
            current_price = float(quote_data.get('price', quote_data.get('c', 0)))
            previous_close = float(quote_data.get('previous_close', quote_data.get('pc', current_price)))
            
            change = current_price - previous_close if previous_close else 0.0
            change_percent = (change / previous_close * 100) if previous_close else 0.0
            
            return StockQuoteResponse(
                symbol=symbol,
                price=current_price,
                change=change,
                change_percent=change_percent,
                volume=int(quote_data.get('volume', quote_data.get('v', 0))),
                timestamp=datetime.utcnow(),
                
                # Enhanced data
                open=float(quote_data.get('open', quote_data.get('o'))) if quote_data.get('open') or quote_data.get('o') else None,
                high=float(quote_data.get('high', quote_data.get('h'))) if quote_data.get('high') or quote_data.get('h') else None,
                low=float(quote_data.get('low', quote_data.get('l'))) if quote_data.get('low') or quote_data.get('l') else None,
                previous_close=previous_close if previous_close != current_price else None,
                bid=float(quote_data.get('bid')) if quote_data.get('bid') else None,
                ask=float(quote_data.get('ask')) if quote_data.get('ask') else None,
                
                # Additional market data
                fifty_two_week_high=float(quote_data.get('52_week_high')) if quote_data.get('52_week_high') else None,
                fifty_two_week_low=float(quote_data.get('52_week_low')) if quote_data.get('52_week_low') else None,
                pe_ratio=float(quote_data.get('pe')) if quote_data.get('pe') else None,
                
                # Meta data
                data_source=data_source,
                last_updated=datetime.utcnow(),
                is_real_time=True
            )
        
        # Fallback to database data
        logger.info(f"Falling back to database for {symbol}")
        
        # Get stock info to verify it exists
        stock = await stock_repository.get_by_symbol(symbol, session=db)
        if not stock:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Stock '{symbol}' not found in database"
            )
        
        # Get latest price from database
        latest_price = await price_repository.get_latest_price(symbol, session=db)
        if not latest_price:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No price data found for symbol '{symbol}'"
            )
        
        # Get previous price for change calculation
        previous_price = await price_repository.get_previous_price(symbol, latest_price.date, session=db)
        previous_close = float(previous_price.close) if previous_price else float(latest_price.close)
        
        current_price = float(latest_price.close)
        change = current_price - previous_close
        change_percent = (change / previous_close * 100) if previous_close else 0.0
        
        return StockQuoteResponse(
            symbol=symbol,
            price=current_price,
            change=change,
            change_percent=change_percent,
            volume=latest_price.volume,
            timestamp=datetime.combine(latest_price.date, datetime.min.time()),
            
            # Basic OHLC data
            open=float(latest_price.open),
            high=float(latest_price.high),
            low=float(latest_price.low),
            previous_close=previous_close if previous_price else None,
            
            # Market data from stock model
            market_cap=stock.market_cap,
            
            # Meta data
            data_source="database",
            last_updated=latest_price.updated_at or datetime.utcnow(),
            is_real_time=False
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving stock quote for {symbol}: {e}")
        await handle_api_error(e, f"retrieve quote for {symbol}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving stock quote: {str(e)}"
        )


@router.get("/{symbol}/history", response_model=List[PriceHistoryResponse])
@api_cache(data_type="daily_prices", ttl_override={'l1': 3600, 'l2': 14400, 'l3': 86400})
async def get_stock_history(
    symbol: str = Path(..., description="Stock symbol"),
    start_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[date] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: Optional[int] = Query(252, le=1000, description="Maximum number of records"),
    db: AsyncSession = Depends(get_async_db_session)
) -> List[PriceHistoryResponse]:
    """
    Get historical price data for a stock.
    
    - **symbol**: Stock symbol
    - **start_date**: Start date for historical data
    - **end_date**: End date for historical data  
    - **limit**: Maximum number of records (defaults to 1 year ~ 252 trading days)
    """
    try:
        # Set default date range if not provided
        if not end_date:
            end_date = date.today()
        if not start_date:
            start_date = end_date - timedelta(days=365)
        
        price_history = await price_repository.get_price_history(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            session=db
        )
        
        if not price_history:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No price history found for symbol '{symbol}' in the specified date range"
            )
        
        return [PriceHistoryResponse.from_orm(price) for price in price_history]
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving price history: {str(e)}"
        )


@router.get("/{symbol}/statistics")
@cache_analysis_result(ttl_hours=2)
async def get_stock_statistics(
    symbol: str = Path(..., description="Stock symbol"),
    days: int = Query(252, le=1000, description="Number of days for analysis"),
    db: AsyncSession = Depends(get_async_db_session)
) -> Dict[str, Any]:
    """
    Get comprehensive price statistics for a stock.
    
    - **symbol**: Stock symbol
    - **days**: Number of days to analyze (default 252 ~ 1 year)
    """
    try:
        statistics = await price_repository.get_price_statistics(
            symbol=symbol,
            days=days,
            session=db
        )
        
        if not statistics:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No price data found for symbol '{symbol}'"
            )
        
        # Add volatility calculation
        volatility = await price_repository.get_volatility(
            symbol=symbol,
            days=min(days, 30),  # Use last 30 days for volatility
            session=db
        )
        
        if volatility is not None:
            statistics['volatility_annualized'] = volatility
        
        return statistics
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving stock statistics: {str(e)}"
        )


@router.post("/{symbol}/watchlist")
async def add_to_watchlist(
    symbol: str = Path(..., description="Stock symbol"),
    db: AsyncSession = Depends(get_async_db_session)
) -> Dict[str, Any]:
    """
    Add a stock to user's watchlist.
    
    Note: This is a placeholder endpoint. In a full implementation,
    this would require user authentication and actual watchlist management.
    """
    try:
        # Verify stock exists
        stock = await stock_repository.get_by_symbol(symbol, session=db)
        
        if not stock:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Stock with symbol '{symbol}' not found"
            )
        
        # TODO: Implement actual watchlist functionality with user authentication
        return {
            "message": f"Stock {symbol.upper()} added to watchlist",
            "success": True,
            "stock_id": stock.id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error adding to watchlist: {str(e)}"
        )


@router.delete("/{symbol}/watchlist")
async def remove_from_watchlist(
    symbol: str = Path(..., description="Stock symbol"),
    db: AsyncSession = Depends(get_async_db_session)
) -> Dict[str, Any]:
    """
    Remove a stock from user's watchlist.
    
    Note: This is a placeholder endpoint. In a full implementation,
    this would require user authentication and actual watchlist management.
    """
    try:
        # Verify stock exists
        stock = await stock_repository.get_by_symbol(symbol, session=db)
        
        if not stock:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Stock with symbol '{symbol}' not found"
            )
        
        # TODO: Implement actual watchlist functionality with user authentication
        return {
            "message": f"Stock {symbol.upper()} removed from watchlist",
            "success": True,
            "stock_id": stock.id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error removing from watchlist: {str(e)}"
        )