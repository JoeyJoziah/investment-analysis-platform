"""
Stocks API Router - Async Database Operations
Updated to use async database operations with proper repository pattern.
"""

from fastapi import APIRouter, Query, HTTPException, Depends, status, Path
from typing import List, Optional, Dict, Any
from datetime import date, datetime, timedelta
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

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

router = APIRouter()

# Pydantic response models
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
    """Real-time stock quote response"""
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    timestamp: datetime


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
async def get_stock_quote(
    symbol: str = Path(..., description="Stock symbol"),
    db: AsyncSession = Depends(get_async_db_session)
) -> StockQuoteResponse:
    """
    Get real-time quote for a stock.
    
    - **symbol**: Stock symbol (e.g., AAPL, GOOGL)
    """
    try:
        # Get latest price from price history
        latest_price = await price_repository.get_latest_price(symbol, session=db)
        
        if not latest_price:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No price data found for symbol '{symbol}'"
            )
        
        # Calculate change (simplified - would need previous day's price)
        # For now, using a placeholder calculation
        change = 0.0
        change_percent = 0.0
        
        return StockQuoteResponse(
            symbol=symbol.upper(),
            price=float(latest_price.close),
            change=change,
            change_percent=change_percent,
            volume=latest_price.volume,
            timestamp=datetime.combine(latest_price.date, datetime.min.time())
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving stock quote: {str(e)}"
        )


@router.get("/{symbol}/history", response_model=List[PriceHistoryResponse])
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