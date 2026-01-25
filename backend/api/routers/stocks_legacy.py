from fastapi import APIRouter, Query, HTTPException, Depends
from typing import List, Optional, Dict, Any
from datetime import date, datetime, timedelta
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.database import get_async_db_session
from backend.repositories import stock_repository, price_repository, FilterCriteria, PaginationParams, SortParams
from backend.models.unified_models import Stock, PriceHistory

router = APIRouter()

# Pydantic models
class Stock(BaseModel):
    symbol: str
    name: str
    sector: Optional[str] = None
    market_cap: Optional[float] = None
    current_price: Optional[float] = None
    change_percent: Optional[float] = None
    volume: Optional[int] = None

class StockDetail(Stock):
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    high_52w: Optional[float] = None
    low_52w: Optional[float] = None
    avg_volume: Optional[int] = None
    beta: Optional[float] = None
    eps: Optional[float] = None
    market_cap_formatted: Optional[str] = None

class PriceHistory(BaseModel):
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int

class StockQuote(BaseModel):
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    timestamp: datetime

# Sample data for development
SAMPLE_STOCKS = [
    Stock(
        symbol="AAPL",
        name="Apple Inc.",
        sector="Technology",
        market_cap=3000000000000,
        current_price=195.89,
        change_percent=1.25,
        volume=50000000
    ),
    Stock(
        symbol="GOOGL",
        name="Alphabet Inc.",
        sector="Technology",
        market_cap=1800000000000,
        current_price=150.42,
        change_percent=-0.75,
        volume=25000000
    ),
    Stock(
        symbol="MSFT",
        name="Microsoft Corporation",
        sector="Technology",
        market_cap=2800000000000,
        current_price=420.55,
        change_percent=0.50,
        volume=20000000
    ),
    Stock(
        symbol="AMZN",
        name="Amazon.com Inc.",
        sector="Consumer Cyclical",
        market_cap=1700000000000,
        current_price=175.35,
        change_percent=2.10,
        volume=40000000
    ),
    Stock(
        symbol="META",
        name="Meta Platforms Inc.",
        sector="Technology",
        market_cap=1300000000000,
        current_price=500.12,
        change_percent=1.80,
        volume=15000000
    )
]

@router.get("", response_model=List[Stock])
async def get_stocks(
    sector: Optional[str] = None,
    min_market_cap: Optional[float] = None,
    max_market_cap: Optional[float] = None,
    limit: int = Query(100, le=500),
    offset: int = 0,
    sort_by: str = Query("market_cap", pattern="^(symbol|name|market_cap|current_price|change_percent)$"),
    order: str = Query("desc", pattern="^(asc|desc)$")
) -> List[Stock]:
    """Get list of stocks with optional filters"""
    
    # Filter stocks based on criteria
    filtered_stocks = SAMPLE_STOCKS.copy()
    
    if sector:
        filtered_stocks = [s for s in filtered_stocks if s.sector == sector]
    
    if min_market_cap:
        filtered_stocks = [s for s in filtered_stocks if s.market_cap and s.market_cap >= min_market_cap]
    
    if max_market_cap:
        filtered_stocks = [s for s in filtered_stocks if s.market_cap and s.market_cap <= max_market_cap]
    
    # Sort stocks
    reverse = (order == "desc")
    filtered_stocks.sort(key=lambda x: getattr(x, sort_by, 0) or 0, reverse=reverse)
    
    # Apply pagination
    return filtered_stocks[offset:offset + limit]

@router.get("/sectors")
async def get_sectors() -> List[str]:
    """Get list of available sectors"""
    sectors = list(set([s.sector for s in SAMPLE_STOCKS if s.sector]))
    return sorted(sectors)

@router.get("/search")
async def search_stocks(
    query: str = Query(..., min_length=1),
    limit: int = Query(10, le=50)
) -> List[Stock]:
    """Search stocks by symbol or name"""
    query_lower = query.lower()
    results = [
        s for s in SAMPLE_STOCKS 
        if query_lower in s.symbol.lower() or query_lower in s.name.lower()
    ]
    return results[:limit]

@router.get("/{symbol}", response_model=StockDetail)
async def get_stock_detail(symbol: str) -> StockDetail:
    """Get detailed information about a specific stock"""
    
    symbol_upper = symbol.upper()
    
    # Find stock in sample data
    stock = next((s for s in SAMPLE_STOCKS if s.symbol == symbol_upper), None)
    
    if not stock:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
    
    # Create detailed response with additional fields
    return StockDetail(
        **stock.dict(),
        pe_ratio=25.5,
        dividend_yield=1.5,
        high_52w=stock.current_price * 1.2 if stock.current_price else None,
        low_52w=stock.current_price * 0.8 if stock.current_price else None,
        avg_volume=stock.volume,
        beta=1.1,
        eps=7.50,
        market_cap_formatted=f"${stock.market_cap/1000000000:.2f}B" if stock.market_cap else None
    )

@router.get("/{symbol}/quote", response_model=StockQuote)
async def get_stock_quote(symbol: str) -> StockQuote:
    """Get real-time quote for a stock"""
    
    symbol_upper = symbol.upper()
    stock = next((s for s in SAMPLE_STOCKS if s.symbol == symbol_upper), None)
    
    if not stock:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
    
    return StockQuote(
        symbol=symbol_upper,
        price=stock.current_price or 0,
        change=stock.current_price * (stock.change_percent / 100) if stock.current_price and stock.change_percent else 0,
        change_percent=stock.change_percent or 0,
        volume=stock.volume or 0,
        timestamp=datetime.utcnow()
    )

@router.get("/{symbol}/history", response_model=List[PriceHistory])
async def get_stock_history(
    symbol: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    interval: str = Query("daily", pattern="^(daily|weekly|monthly)$")
) -> List[PriceHistory]:
    """Get historical price data for a stock"""
    
    symbol_upper = symbol.upper()
    stock = next((s for s in SAMPLE_STOCKS if s.symbol == symbol_upper), None)
    
    if not stock:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
    
    # Generate sample historical data
    if not end_date:
        end_date = date.today()
    if not start_date:
        start_date = end_date - timedelta(days=30)
    
    history = []
    current_date = start_date
    base_price = stock.current_price or 100
    
    while current_date <= end_date:
        # Generate random price variations
        import random
        variation = random.uniform(-0.05, 0.05)
        day_price = base_price * (1 + variation)
        
        history.append(PriceHistory(
            date=current_date,
            open=day_price * random.uniform(0.98, 1.02),
            high=day_price * random.uniform(1.01, 1.05),
            low=day_price * random.uniform(0.95, 0.99),
            close=day_price,
            volume=random.randint(10000000, 100000000)
        ))
        
        # Move to next period based on interval
        if interval == "daily":
            current_date += timedelta(days=1)
        elif interval == "weekly":
            current_date += timedelta(days=7)
        else:  # monthly
            current_date += timedelta(days=30)
    
    return history

@router.post("/{symbol}/watchlist")
async def add_to_watchlist(symbol: str):
    """Add a stock to user's watchlist"""
    symbol_upper = symbol.upper()
    stock = next((s for s in SAMPLE_STOCKS if s.symbol == symbol_upper), None)
    
    if not stock:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
    
    # In production, this would add to user's watchlist in database
    return {"message": f"Stock {symbol_upper} added to watchlist", "success": True}

@router.delete("/{symbol}/watchlist")
async def remove_from_watchlist(symbol: str):
    """Remove a stock from user's watchlist"""
    symbol_upper = symbol.upper()
    
    # In production, this would remove from user's watchlist in database
    return {"message": f"Stock {symbol_upper} removed from watchlist", "success": True}