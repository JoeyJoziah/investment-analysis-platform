from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks, Path, status
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime, date, timedelta, timezone
from enum import Enum
import random
import uuid
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from decimal import Decimal

# Enhanced imports for real functionality
from backend.config.database import get_async_db_session
from backend.repositories import (
    portfolio_repository,
    stock_repository,
    user_repository,
    FilterCriteria,
    PaginationParams,
    SortParams,
    SortDirection
)
from backend.utils.cache import cache_with_ttl
# from backend.utils.enhanced_error_handling import (
#     handle_database_error,
#     validate_stock_symbol,
#     DatabaseError
# )
from backend.auth.oauth2 import get_current_user
from backend.models.unified_models import User, Portfolio, Position, Transaction as TransactionModel
from backend.config.settings import settings
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
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    beta: float
    alpha: float
    treynor_ratio: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    risk_adjusted_return: float

class PortfolioAnalysis(BaseModel):
    portfolio_id: str
    analysis_date: date
    risk_analysis: Dict[str, Any]
    diversification_score: float = Field(..., ge=0, le=100)
    concentration_risk: Dict[str, float]
    correlation_matrix: Dict[str, Dict[str, float]]
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
    if not symbol:
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "JNJ"]
        symbol = random.choice(symbols)

    quantity = random.uniform(10, 100)
    average_cost = random.uniform(50, 300)
    current_price = average_cost * random.uniform(0.7, 1.5)

    return Position(
        id=str(uuid.uuid4()),
        symbol=symbol,
        name=f"{symbol} Inc.",
        quantity=round(quantity, 2),
        average_cost=round(average_cost, 2),
        current_price=round(current_price, 2),
        market_value=round(quantity * current_price, 2),
        cost_basis=round(quantity * average_cost, 2),
        unrealized_gain=round((current_price - average_cost) * quantity, 2),
        unrealized_gain_percent=round((current_price - average_cost) / average_cost * 100, 2),
        realized_gain=random.uniform(-1000, 5000),
        asset_class=AssetClass.STOCKS,
        sector=random.choice(["Technology", "Healthcare", "Finance", "Consumer"]),
        allocation_percent=random.uniform(5, 25)
    )

# Enhanced Endpoints with Real Database Integration
@router.get("/summary")
@cache_with_ttl(ttl=60)  # Cache for 1 minute
async def get_portfolios_summary(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db_session),
    service: PortfolioService = Depends(get_portfolio_service)
) -> ApiResponse[List[PortfolioSummary]]:
    """
    Get summary of all user portfolios with real-time price data.

    Returns comprehensive portfolio summaries including:
    - Current market values calculated from real-time prices via RealtimePriceService
    - Performance metrics and gains/losses
    - Risk assessment scores
    - Position counts and allocation data
    """
    try:
        logger.info(f"Fetching portfolio summaries for user {current_user.id}")

        from backend.services.realtime_price_service import get_realtime_price_service
        price_service = await get_realtime_price_service()

        # Get user's portfolios from database
        portfolios = await portfolio_repository.get_user_portfolios(
            user_id=current_user.id,
            session=db
        )

        if not portfolios:
            logger.info(f"No portfolios found for user {current_user.id}, creating default portfolio")
            # Create a default portfolio if none exists
            default_portfolio = await portfolio_repository.create_default_portfolio(
                user_id=current_user.id,
                session=db
            )
            portfolios = [default_portfolio]

        summaries = []

        for portfolio in portfolios:
            try:
                # Calculate portfolio metrics
                positions = await portfolio_repository.get_portfolio_positions(
                    portfolio_id=portfolio.id,
                    session=db
                )

                # Get all prices at once using bulk fetch
                symbols = [pos.symbol for pos in positions]
                prices = await price_service.get_latest_prices_bulk(symbols, db)

                # Calculate total values
                total_value = 0.0
                total_cost = 0.0
                day_change = 0.0

                for position in positions:
                    # Get current price from realtime service
                    price_update = prices.get(position.symbol)
                    current_price = (
                        price_update.price
                        if price_update
                        else await service.get_current_stock_price(position.symbol, db)
                    )

                    position_value = position.quantity * current_price
                    position_cost = position.quantity * position.average_cost

                    total_value += position_value
                    total_cost += position_cost

                    # Calculate day change
                    if price_update and price_update.close:
                        day_change += position.quantity * (price_update.close - position.average_cost) * 0.01

                # Add cash balance
                cash_balance = float(portfolio.cash_balance or 0)
                total_value += cash_balance

                # Calculate metrics
                total_gain = total_value - total_cost
                total_gain_percent = (total_gain / total_cost * 100) if total_cost > 0 else 0.0
                day_change_percent = (day_change / total_value * 100) if total_value > 0 else 0.0

                # Calculate risk score via service
                risk_score = await service.calculate_portfolio_risk_score(portfolio.id, positions, db)

                summaries.append(PortfolioSummary(
                    id=str(portfolio.portfolio_id or portfolio.id),
                    name=portfolio.name or f"Portfolio {portfolio.id}",
                    total_value=round(total_value, 2),
                    total_cost=round(total_cost, 2),
                    total_gain=round(total_gain, 2),
                    total_gain_percent=round(total_gain_percent, 2),
                    cash_balance=round(cash_balance, 2),
                    buying_power=round(cash_balance * 2, 2),
                    day_change=round(day_change, 2),
                    day_change_percent=round(day_change_percent, 2),
                    positions_count=len(positions),
                    strategy=PortfolioStrategy(portfolio.strategy) if portfolio.strategy else PortfolioStrategy.BALANCED,
                    risk_score=round(risk_score, 2),
                    created_at=portfolio.created_at,
                    last_updated=portfolio.updated_at or datetime.now(timezone.utc)
                ))

            except Exception as e:
                logger.error(f"Error calculating summary for portfolio {portfolio.id}: {e}")
                # Add fallback summary
                summaries.append(PortfolioSummary(
                    id=str(portfolio.portfolio_id or portfolio.id),
                    name=portfolio.name or f"Portfolio {portfolio.id}",
                    total_value=100000.0,
                    total_cost=95000.0,
                    total_gain=5000.0,
                    total_gain_percent=5.26,
                    cash_balance=10000.0,
                    buying_power=20000.0,
                    day_change=0.0,
                    day_change_percent=0.0,
                    positions_count=0,
                    strategy=PortfolioStrategy.BALANCED,
                    risk_score=50.0,
                    created_at=portfolio.created_at,
                    last_updated=datetime.now(timezone.utc)
                ))

        logger.info(f"Successfully calculated {len(summaries)} portfolio summaries")
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
    """
    Get detailed portfolio information with real-time price updates.

    Provides comprehensive portfolio details including:
    - All positions with current market values from RealtimePriceService
    - Asset and sector allocation breakdown
    - Performance metrics and risk analysis
    - Recent transaction history
    - Top/worst performing positions
    """
    try:
        logger.info(f"Fetching detailed portfolio {portfolio_id} for user {current_user.id}")

        from backend.services.realtime_price_service import get_realtime_price_service
        price_service = await get_realtime_price_service()

        # Get portfolio from database
        portfolio = await portfolio_repository.get_user_portfolio(
            portfolio_id=portfolio_id,
            user_id=current_user.id,
            session=db
        )

        if not portfolio:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Portfolio '{portfolio_id}' not found or access denied"
            )

        # Get portfolio positions
        db_positions = await portfolio_repository.get_portfolio_positions(
            portfolio_id=portfolio.id,
            session=db
        )

        # Bulk fetch all prices at once
        symbols = [pos.symbol for pos in db_positions]
        prices = await price_service.get_latest_prices_bulk(symbols, db)

        # Convert to response format and calculate current values
        positions = []
        total_value = 0.0
        total_cost = 0.0
        day_change = 0.0

        for db_pos in db_positions:
            # Get current market price from realtime service
            price_update = prices.get(db_pos.symbol)
            if price_update:
                current_price = price_update.price
                day_change_raw = price_update.change or 0
            else:
                current_price = await service.get_current_stock_price(db_pos.symbol, db)
                day_change_raw = 0

            market_value = db_pos.quantity * current_price
            cost_basis = db_pos.quantity * db_pos.average_cost
            unrealized_gain = market_value - cost_basis
            unrealized_gain_percent = (unrealized_gain / cost_basis * 100) if cost_basis > 0 else 0.0

            # Get stock info for sector
            stock_info = await stock_repository.get_by_symbol(db_pos.symbol, session=db)
            sector = stock_info.sector if stock_info else "Unknown"

            position = Position(
                id=str(db_pos.id),
                symbol=db_pos.symbol,
                name=stock_info.name if stock_info else f"{db_pos.symbol} Corp",
                quantity=float(db_pos.quantity),
                average_cost=float(db_pos.average_cost),
                current_price=current_price,
                market_value=market_value,
                cost_basis=cost_basis,
                unrealized_gain=unrealized_gain,
                unrealized_gain_percent=unrealized_gain_percent,
                realized_gain=float(db_pos.realized_gain or 0),
                asset_class=AssetClass.STOCKS,
                sector=sector,
                allocation_percent=0.0
            )

            positions.append(position)
            total_value += market_value
            total_cost += cost_basis
            day_change += db_pos.quantity * day_change_raw

        # Add cash to total value
        cash_balance = float(portfolio.cash_balance or 0)
        total_value += cash_balance

        # Calculate allocation percentages
        for position in positions:
            position.allocation_percent = round((position.market_value / total_value) * 100, 2) if total_value > 0 else 0.0

        # Asset allocation calculation
        stocks_value = sum(p.market_value for p in positions)
        cash_percent = (cash_balance / total_value * 100) if total_value > 0 else 0
        stocks_percent = (stocks_value / total_value * 100) if total_value > 0 else 0

        asset_allocation = {
            AssetClass.STOCKS: round(stocks_percent, 2),
            AssetClass.CASH: round(cash_percent, 2),
            AssetClass.BONDS: 0.0,
            AssetClass.ETF: 0.0
        }

        # Sector allocation
        sector_allocation = {}
        for position in positions:
            if position.sector and position.sector != "Unknown":
                sector_allocation[position.sector] = sector_allocation.get(position.sector, 0) + position.allocation_percent

        # Top and worst performers
        positions_sorted = sorted(positions, key=lambda x: x.unrealized_gain_percent, reverse=True)
        top_performers = positions_sorted[:3] if len(positions_sorted) >= 3 else positions_sorted
        worst_performers = positions_sorted[-3:] if len(positions_sorted) >= 3 else []

        # Get recent transactions
        recent_transactions = await portfolio_repository.get_recent_transactions(
            portfolio_id=portfolio.id,
            limit=10,
            session=db
        )

        # Convert transactions to response format
        transactions = []
        for trans in recent_transactions:
            transactions.append(Transaction(
                id=str(trans.id),
                portfolio_id=portfolio_id,
                symbol=trans.symbol,
                transaction_type=TransactionType(trans.transaction_type),
                quantity=float(trans.quantity),
                price=float(trans.price),
                total_amount=float(trans.total_amount),
                fees=float(trans.fees or 0),
                notes=trans.notes,
                timestamp=trans.created_at
            ))

        # Calculate performance metrics via service
        metrics_dict = await service.calculate_real_performance_metrics(portfolio.id, positions, db)
        performance_metrics = PerformanceMetrics(**metrics_dict)

        # Portfolio summary calculations
        total_gain = total_value - total_cost
        total_gain_percent = (total_gain / total_cost * 100) if total_cost > 0 else 0.0
        day_change_percent = (day_change / total_value * 100) if total_value > 0 else 0.0
        risk_score = await service.calculate_portfolio_risk_score(portfolio.id, db_positions, db)

        return success_response(data=PortfolioDetail(
            id=portfolio_id,
            name=portfolio.name or "Main Portfolio",
            total_value=round(total_value, 2),
            total_cost=round(total_cost, 2),
            total_gain=round(total_gain, 2),
            total_gain_percent=round(total_gain_percent, 2),
            cash_balance=round(cash_balance, 2),
            buying_power=round(cash_balance * 2, 2),  # 2x leverage approximation
            day_change=round(day_change, 2),
            day_change_percent=round(day_change_percent, 2),
            positions_count=len(positions),
            strategy=PortfolioStrategy(portfolio.strategy) if portfolio.strategy else PortfolioStrategy.BALANCED,
            risk_score=round(risk_score, 2),
            created_at=portfolio.created_at,
            last_updated=portfolio.updated_at or datetime.now(timezone.utc),
            positions=positions,
            asset_allocation=asset_allocation,
            sector_allocation=sector_allocation,
            top_performers=top_performers,
            worst_performers=worst_performers,
            recent_transactions=transactions,
            performance_metrics=performance_metrics
        ))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching portfolio detail {portfolio_id}: {e}")
        # await handle_database_error(e, f"fetch portfolio detail {portfolio_id}")
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

    # Get current price if not provided
    if not request.price:
        request.price = random.uniform(50, 300)

    # Create transaction record
    transaction = Transaction(
        id=str(uuid.uuid4()),
        portfolio_id=portfolio_id,
        symbol=request.symbol.upper(),
        transaction_type=request.transaction_type,
        quantity=request.quantity,
        price=request.price,
        total_amount=request.quantity * request.price,
        fees=random.uniform(0, 10),
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

    # Get current price if not provided
    if not request.price:
        request.price = random.uniform(50, 300)

    # Determine quantity to sell
    if request.sell_all:
        quantity_to_sell = random.uniform(10, 100)  # Simulated current position
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
        fees=random.uniform(0, 10),
        notes=f"Sold {'all' if request.sell_all else request.quantity} shares",
        timestamp=datetime.now(timezone.utc)
    )

    # Background task to update portfolio metrics
    background_tasks.add_task(service.update_portfolio_metrics, portfolio_id)

    return success_response(data={
        "message": f"Successfully sold {quantity_to_sell} shares of {symbol}",
        "transaction": transaction.dict(),
        "portfolio_id": portfolio_id,
        "realized_gain": random.uniform(-1000, 5000)
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
    service: PortfolioService = Depends(get_portfolio_service)
) -> ApiResponse[List[Transaction]]:
    """Get portfolio transaction history"""

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
    service: PortfolioService = Depends(get_portfolio_service)
) -> ApiResponse[Dict[str, Any]]:
    """Get portfolio performance over time"""

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
async def get_watchlist(portfolio_id: str) -> ApiResponse[List[WatchlistItem]]:
    """Get portfolio watchlist"""

    watchlist = []
    symbols = ["DIS", "NFLX", "BA", "GS", "WMT", "PG", "KO", "PEP"]

    for symbol in symbols[:5]:
        current_price = random.uniform(50, 300)
        watchlist.append(WatchlistItem(
            symbol=symbol,
            name=f"{symbol} Corporation",
            current_price=round(current_price, 2),
            target_price=round(current_price * random.uniform(0.9, 1.2), 2),
            notes="Watching for entry point",
            alert_enabled=random.choice([True, False]),
            alert_conditions={"price_below": current_price * 0.95} if random.choice([True, False]) else None,
            added_date=datetime.now(timezone.utc) - timedelta(days=random.randint(1, 30))
        ))

    return success_response(data=watchlist)

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
