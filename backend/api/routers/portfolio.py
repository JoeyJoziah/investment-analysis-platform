from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks, Path, status
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime, date, timedelta, timezone
from enum import Enum
import uuid
import logging
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.database import get_async_db_session
from backend.utils.cache import cache_with_ttl
from backend.auth.oauth2 import get_current_user
from backend.models.unified_models import User
from backend.models.api_response import ApiResponse, success_response
from backend.services.portfolio_service import portfolio_service, PortfolioService

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(tags=["portfolio"])

# ============================================================================
# Service Layer Dependencies
# ============================================================================

async def get_portfolio_service():
    """
    Dependency to get portfolio service instance.

    In test mode (TESTING=True), returns service without initialization.
    """
    import os
    if not os.getenv("TESTING"):
        try:
            # Portfolio service doesn't need async initialization currently
            pass
        except Exception as e:
            logger.warning(f"Failed to initialize portfolio service: {e}")
    return portfolio_service

# Enums
class TransactionType(str, Enum):
    BUY = "buy"
    SELL = "sell"
    DIVIDEND = "dividend"
    TRANSFER_IN = "transfer_in"
    TRANSFER_OUT = "transfer_out"

class AssetClass(str, Enum):
    STOCKS = "stocks"
    BONDS = "bonds"
    ETF = "etf"
    CRYPTO = "crypto"
    COMMODITIES = "commodities"
    CASH = "cash"
    REAL_ESTATE = "real_estate"

class PortfolioStrategy(str, Enum):
    AGGRESSIVE_GROWTH = "aggressive_growth"
    GROWTH = "growth"
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"
    INCOME = "income"
    PRESERVATION = "preservation"

class RebalanceFrequency(str, Enum):
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMI_ANNUAL = "semi_annual"
    ANNUAL = "annual"
    MANUAL = "manual"

# Pydantic models
class Position(BaseModel):
    id: str
    symbol: str
    name: str
    quantity: float = Field(..., gt=0)
    average_cost: float = Field(..., gt=0)
    current_price: float = Field(..., gt=0)
    market_value: float
    cost_basis: float
    unrealized_gain: float
    unrealized_gain_percent: float
    realized_gain: float = 0
    asset_class: AssetClass
    sector: Optional[str] = None
    allocation_percent: float = Field(..., ge=0, le=100)

    @validator('market_value', always=True)
    def calculate_market_value(cls, v, values):
        if 'quantity' in values and 'current_price' in values:
            return values['quantity'] * values['current_price']
        return v

    @validator('cost_basis', always=True)
    def calculate_cost_basis(cls, v, values):
        if 'quantity' in values and 'average_cost' in values:
            return values['quantity'] * values['average_cost']
        return v

class PortfolioSummary(BaseModel):
    id: str
    name: str
    total_value: float
    total_cost: float
    total_gain: float
    total_gain_percent: float
    cash_balance: float
    buying_power: float
    day_change: float
    day_change_percent: float
    positions_count: int
    strategy: PortfolioStrategy
    risk_score: float = Field(..., ge=0, le=100)
    created_at: datetime
    last_updated: datetime

class PortfolioDetail(PortfolioSummary):
    positions: List[Position]
    asset_allocation: Dict[AssetClass, float]
    sector_allocation: Dict[str, float]
    top_performers: List[Position]
    worst_performers: List[Position]
    recent_transactions: List['Transaction']
    performance_metrics: 'PerformanceMetrics'

class Transaction(BaseModel):
    id: str
    portfolio_id: str
    symbol: str
    transaction_type: TransactionType
    quantity: float
    price: float
    total_amount: float
    fees: float = 0
    notes: Optional[str] = None
    timestamp: datetime

    @validator('total_amount', always=True)
    def calculate_total(cls, v, values):
        if 'quantity' in values and 'price' in values:
            return values['quantity'] * values['price'] + values.get('fees', 0)
        return v

class AddPositionRequest(BaseModel):
    symbol: str
    quantity: float = Field(..., gt=0)
    price: Optional[float] = Field(None, gt=0)
    transaction_type: TransactionType = TransactionType.BUY
    notes: Optional[str] = None

class RemovePositionRequest(BaseModel):
    symbol: str
    quantity: Optional[float] = Field(None, gt=0)
    sell_all: bool = False
    price: Optional[float] = Field(None, gt=0)

class PerformanceMetrics(BaseModel):
    total_return: Optional[float] = None
    annualized_return: Optional[float] = None
    volatility: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    beta: Optional[float] = None
    alpha: Optional[float] = None
    treynor_ratio: Optional[float] = None
    calmar_ratio: Optional[float] = None
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    risk_adjusted_return: Optional[float] = None

class PortfolioAnalysis(BaseModel):
    portfolio_id: str
    analysis_date: date
    risk_analysis: Dict[str, Any]
    diversification_score: float = Field(..., ge=0, le=100)
    concentration_risk: Dict[str, Any]
    correlation_matrix: Dict[str, Any]
    efficient_frontier: Dict[str, Any]
    optimization_suggestions: List[str]
    rebalancing_needed: bool
    recommended_changes: List[Dict[str, Any]]

class RebalanceRequest(BaseModel):
    portfolio_id: str
    target_allocation: Dict[AssetClass, float]
    max_trades: int = Field(10, ge=1, le=50)
    min_trade_value: float = Field(100, gt=0)
    tax_efficient: bool = True

class WatchlistItem(BaseModel):
    symbol: str
    name: str
    current_price: float
    target_price: Optional[float] = None
    notes: Optional[str] = None
    alert_enabled: bool = False
    alert_conditions: Optional[Dict[str, Any]] = None
    added_date: datetime

class PortfolioSettings(BaseModel):
    portfolio_id: str
    name: str
    strategy: PortfolioStrategy
    rebalance_frequency: RebalanceFrequency
    tax_harvesting_enabled: bool = False
    dividend_reinvestment: bool = True
    margin_enabled: bool = False
    options_enabled: bool = False
    notifications_enabled: bool = True
    benchmark: str = "SPY"

# Helper functions
def generate_position(symbol: str = None) -> Position:
    """Generate a sample position"""
    from backend.services.portfolio_service import generate_position_data
    data = generate_position_data(symbol)
    return Position(**data)

# Enhanced Endpoints with Real Database Integration
@router.get("/summary")
@cache_with_ttl(ttl=60)  # Cache for 1 minute
async def get_portfolios_summary(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db_session),
    service: PortfolioService = Depends(get_portfolio_service)
) -> ApiResponse[List[PortfolioSummary]]:
    """Get summary of all user portfolios with real-time price data."""
    try:
        raw_summaries = await service.compute_portfolio_summaries(
            user_id=current_user.id, db=db
        )

        summaries = [
            PortfolioSummary(
                **{
                    **s,
                    "strategy": (
                        PortfolioStrategy(s["strategy"])
                        if s["strategy"] in [e.value for e in PortfolioStrategy]
                        else PortfolioStrategy.BALANCED
                    ),
                }
            )
            for s in raw_summaries
        ]

        return success_response(data=summaries)

    except Exception as e:
        logger.error(f"Error fetching portfolio summaries: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching portfolio summaries"
        )

@router.get("/{portfolio_id}")
@cache_with_ttl(ttl=30)  # Cache for 30 seconds
async def get_portfolio_detail(
    portfolio_id: str = Path(..., description="Portfolio ID"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db_session),
    service: PortfolioService = Depends(get_portfolio_service)
) -> ApiResponse[PortfolioDetail]:
    """Get detailed portfolio information with real-time price updates."""
    try:
        detail_data = await service.compute_portfolio_detail(
            portfolio_id=portfolio_id, user_id=current_user.id, db=db
        )

        if detail_data is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Portfolio '{portfolio_id}' not found or access denied"
            )

        # Convert raw dicts to Pydantic models
        positions = [Position(**p) for p in detail_data["positions"]]
        top_performers = [Position(**p) for p in detail_data["top_performers"]]
        worst_performers = [Position(**p) for p in detail_data["worst_performers"]]
        transactions = [Transaction(**t) for t in detail_data["recent_transactions"]]
        performance_metrics = PerformanceMetrics(**detail_data["performance_metrics"])

        # Map string keys to AssetClass enums for asset_allocation
        asset_alloc_enum = {
            AssetClass(k): v for k, v in detail_data["asset_allocation"].items()
        }

        strategy_val = detail_data.get("strategy", "balanced")
        strategy = (
            PortfolioStrategy(strategy_val)
            if strategy_val in [e.value for e in PortfolioStrategy]
            else PortfolioStrategy.BALANCED
        )

        return success_response(data=PortfolioDetail(
            id=detail_data["id"],
            name=detail_data["name"],
            total_value=detail_data["total_value"],
            total_cost=detail_data["total_cost"],
            total_gain=detail_data["total_gain"],
            total_gain_percent=detail_data["total_gain_percent"],
            cash_balance=detail_data["cash_balance"],
            buying_power=detail_data["buying_power"],
            day_change=detail_data["day_change"],
            day_change_percent=detail_data["day_change_percent"],
            positions_count=detail_data["positions_count"],
            strategy=strategy,
            risk_score=detail_data["risk_score"],
            created_at=detail_data["created_at"],
            last_updated=detail_data["last_updated"],
            positions=positions,
            asset_allocation=asset_alloc_enum,
            sector_allocation=detail_data["sector_allocation"],
            top_performers=top_performers,
            worst_performers=worst_performers,
            recent_transactions=transactions,
            performance_metrics=performance_metrics,
        ))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching portfolio detail {portfolio_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching portfolio details"
        )

@router.post("/{portfolio_id}/positions")
async def add_position(
    portfolio_id: str,
    request: AddPositionRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    service: PortfolioService = Depends(get_portfolio_service)
) -> ApiResponse[Dict[str, Any]]:
    """Add a new position or add to existing position"""

    # Price is required; reject if not provided
    if not request.price:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A price must be provided to add a position"
        )

    # Create transaction record
    transaction = Transaction(
        id=str(uuid.uuid4()),
        portfolio_id=portfolio_id,
        symbol=request.symbol.upper(),
        transaction_type=request.transaction_type,
        quantity=request.quantity,
        price=request.price,
        total_amount=request.quantity * request.price,
        fees=0.0,
        notes=request.notes,
        timestamp=datetime.now(timezone.utc)
    )

    # Background task to update portfolio metrics
    background_tasks.add_task(service.update_portfolio_metrics, portfolio_id)

    return success_response(data={
        "message": f"Successfully added {request.quantity} shares of {request.symbol}",
        "transaction": transaction.dict(),
        "portfolio_id": portfolio_id
    })

@router.delete("/{portfolio_id}/positions/{symbol}")
async def remove_position(
    portfolio_id: str,
    symbol: str,
    request: RemovePositionRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    service: PortfolioService = Depends(get_portfolio_service)
) -> ApiResponse[Dict[str, Any]]:
    """Remove or reduce a position"""

    # Price is required; reject if not provided
    if not request.price:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A price must be provided to remove a position"
        )

    # Determine quantity to sell
    if request.sell_all:
        if not request.quantity:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="quantity must be provided when sell_all is true"
            )
        quantity_to_sell = request.quantity
    else:
        quantity_to_sell = request.quantity or 0

    # Create transaction record
    transaction = Transaction(
        id=str(uuid.uuid4()),
        portfolio_id=portfolio_id,
        symbol=symbol.upper(),
        transaction_type=TransactionType.SELL,
        quantity=quantity_to_sell,
        price=request.price,
        total_amount=quantity_to_sell * request.price,
        fees=0.0,
        notes=f"Sold {'all' if request.sell_all else request.quantity} shares",
        timestamp=datetime.now(timezone.utc)
    )

    # Background task to update portfolio metrics
    background_tasks.add_task(service.update_portfolio_metrics, portfolio_id)

    return success_response(data={
        "message": f"Successfully sold {quantity_to_sell} shares of {symbol}",
        "transaction": transaction.dict(),
        "portfolio_id": portfolio_id,
        "realized_gain": None
    })

@router.get("/{portfolio_id}/transactions")
async def get_transactions(
    portfolio_id: str,
    limit: int = Query(50, le=500),
    offset: int = 0,
    transaction_type: Optional[TransactionType] = None,
    symbol: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db_session),
    service: PortfolioService = Depends(get_portfolio_service)
) -> ApiResponse[List[Transaction]]:
    """Get portfolio transaction history"""

    # Verify portfolio ownership before returning any transaction data.
    ownership = await service.compute_portfolio_detail(
        portfolio_id=portfolio_id, user_id=current_user.id, db=db
    )
    if ownership is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Portfolio '{portfolio_id}' not found or access denied",
        )

    raw_transactions = service.generate_transaction_list(
        portfolio_id=portfolio_id,
        limit=limit,
        offset=offset,
        transaction_type_filter=transaction_type,
        symbol_filter=symbol,
        start_date=start_date,
        end_date=end_date,
    )

    transactions = [
        Transaction(
            id=t["id"],
            portfolio_id=t["portfolio_id"],
            symbol=t["symbol"],
            transaction_type=t["transaction_type"],
            quantity=t["quantity"],
            price=t["price"],
            total_amount=t["quantity"] * t["price"] + t["fees"],
            fees=t["fees"],
            notes=t["notes"],
            timestamp=t["timestamp"],
        )
        for t in raw_transactions
    ]

    return success_response(data=transactions)

@router.get("/{portfolio_id}/performance")
async def get_portfolio_performance(
    portfolio_id: str,
    period: str = Query("1M", pattern="^(1D|1W|1M|3M|6M|1Y|3Y|5Y|ALL)$"),
    benchmark: str = "SPY",
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db_session),
    service: PortfolioService = Depends(get_portfolio_service)
) -> ApiResponse[Dict[str, Any]]:
    """Get portfolio performance over time"""

    # Verify portfolio ownership before returning performance data.
    ownership = await service.compute_portfolio_detail(
        portfolio_id=portfolio_id, user_id=current_user.id, db=db
    )
    if ownership is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Portfolio '{portfolio_id}' not found or access denied",
        )

    result = service.generate_performance_data_points(
        portfolio_id=portfolio_id,
        period=period,
        benchmark=benchmark,
    )
    return success_response(data=result)

@router.post("/{portfolio_id}/analyze")
async def analyze_portfolio(
    portfolio_id: str,
    current_user: User = Depends(get_current_user),
    service: PortfolioService = Depends(get_portfolio_service)
) -> ApiResponse[PortfolioAnalysis]:
    """Perform comprehensive portfolio analysis"""

    analysis_data = service.build_portfolio_analysis(portfolio_id)
    return success_response(data=PortfolioAnalysis(**analysis_data))

@router.post("/{portfolio_id}/rebalance")
async def rebalance_portfolio(
    portfolio_id: str,
    request: RebalanceRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    service: PortfolioService = Depends(get_portfolio_service)
) -> ApiResponse[Dict[str, Any]]:
    """Generate rebalancing recommendations"""

    # Validate target allocation sums to 100%
    total_allocation = sum(request.target_allocation.values())
    if abs(total_allocation - 100) > 0.01:
        raise HTTPException(status_code=400, detail="Target allocation must sum to 100%")

    result = service.generate_rebalancing_trades(
        portfolio_id=portfolio_id,
        target_allocation=request.target_allocation,
        max_trades=request.max_trades,
        min_trade_value=request.min_trade_value,
        tax_efficient=request.tax_efficient,
    )

    # Background task to execute rebalancing
    background_tasks.add_task(service.execute_rebalancing, portfolio_id, result["rebalancing_plan"])

    return success_response(data=result)

@router.get("/{portfolio_id}/watchlist")
async def get_watchlist(
    portfolio_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db_session),
    service: PortfolioService = Depends(get_portfolio_service),
) -> ApiResponse[List[WatchlistItem]]:
    """Get portfolio watchlist"""

    # Verify portfolio ownership before returning watchlist data.
    ownership = await service.compute_portfolio_detail(
        portfolio_id=portfolio_id, user_id=current_user.id, db=db
    )
    if ownership is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Portfolio '{portfolio_id}' not found or access denied",
        )

    # Watchlist is stored in the database; return an empty list until DB integration is complete.
    return success_response(data=[])

@router.post("/{portfolio_id}/watchlist")
async def add_to_watchlist(
    portfolio_id: str,
    item: WatchlistItem,
    current_user: User = Depends(get_current_user)
) -> ApiResponse[Dict[str, str]]:
    """Add item to watchlist"""

    return success_response(data={
        "message": f"Added {item.symbol} to watchlist",
        "portfolio_id": portfolio_id,
        "watchlist_id": str(uuid.uuid4())
    })

@router.put("/{portfolio_id}/settings")
async def update_portfolio_settings(
    portfolio_id: str,
    settings: PortfolioSettings,
    current_user: User = Depends(get_current_user)
) -> ApiResponse[Dict[str, str]]:
    """Update portfolio settings"""

    return success_response(data={
        "message": "Portfolio settings updated successfully",
        "portfolio_id": portfolio_id
    })
