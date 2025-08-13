from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime, date, timedelta
from enum import Enum
import random
import uuid

router = APIRouter(prefix="/portfolio", tags=["portfolio"])

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

def calculate_performance_metrics() -> PerformanceMetrics:
    """Calculate portfolio performance metrics"""
    return PerformanceMetrics(
        total_return=random.uniform(-0.1, 0.3),
        annualized_return=random.uniform(0.05, 0.15),
        volatility=random.uniform(0.1, 0.3),
        sharpe_ratio=random.uniform(0.5, 2.0),
        sortino_ratio=random.uniform(0.7, 2.5),
        max_drawdown=random.uniform(-0.3, -0.05),
        beta=random.uniform(0.8, 1.2),
        alpha=random.uniform(-0.02, 0.05),
        treynor_ratio=random.uniform(0.1, 0.3),
        calmar_ratio=random.uniform(0.5, 2.0),
        win_rate=random.uniform(0.4, 0.7),
        profit_factor=random.uniform(1.2, 2.5),
        risk_adjusted_return=random.uniform(0.08, 0.20)
    )

# Endpoints
@router.get("/summary", response_model=List[PortfolioSummary])
async def get_portfolios_summary() -> List[PortfolioSummary]:
    """Get summary of all user portfolios"""
    
    portfolios = []
    for i in range(3):
        portfolio_value = random.uniform(10000, 1000000)
        portfolio_cost = portfolio_value * random.uniform(0.7, 1.0)
        
        portfolios.append(PortfolioSummary(
            id=str(uuid.uuid4()),
            name=f"Portfolio {i+1}",
            total_value=round(portfolio_value, 2),
            total_cost=round(portfolio_cost, 2),
            total_gain=round(portfolio_value - portfolio_cost, 2),
            total_gain_percent=round((portfolio_value - portfolio_cost) / portfolio_cost * 100, 2),
            cash_balance=random.uniform(1000, 50000),
            buying_power=random.uniform(5000, 100000),
            day_change=random.uniform(-5000, 5000),
            day_change_percent=random.uniform(-2, 2),
            positions_count=random.randint(5, 30),
            strategy=random.choice(list(PortfolioStrategy)),
            risk_score=random.uniform(30, 70),
            created_at=datetime.utcnow() - timedelta(days=random.randint(30, 365)),
            last_updated=datetime.utcnow()
        ))
    
    return portfolios

@router.get("/{portfolio_id}", response_model=PortfolioDetail)
async def get_portfolio_detail(portfolio_id: str) -> PortfolioDetail:
    """Get detailed portfolio information"""
    
    # Generate positions
    positions = [generate_position() for _ in range(random.randint(5, 15))]
    
    # Calculate totals
    total_value = sum(p.market_value for p in positions)
    total_cost = sum(p.cost_basis for p in positions)
    
    # Normalize allocation percentages
    for position in positions:
        position.allocation_percent = round((position.market_value / total_value) * 100, 2)
    
    # Asset allocation
    asset_allocation = {
        AssetClass.STOCKS: 70,
        AssetClass.BONDS: 15,
        AssetClass.ETF: 10,
        AssetClass.CASH: 5
    }
    
    # Sector allocation
    sector_allocation = {}
    for position in positions:
        if position.sector:
            sector_allocation[position.sector] = sector_allocation.get(position.sector, 0) + position.allocation_percent
    
    # Top and worst performers
    positions_sorted = sorted(positions, key=lambda x: x.unrealized_gain_percent, reverse=True)
    top_performers = positions_sorted[:3]
    worst_performers = positions_sorted[-3:]
    
    # Recent transactions
    recent_transactions = []
    for i in range(5):
        recent_transactions.append(Transaction(
            id=str(uuid.uuid4()),
            portfolio_id=portfolio_id,
            symbol=random.choice([p.symbol for p in positions]),
            transaction_type=random.choice(list(TransactionType)),
            quantity=random.uniform(1, 20),
            price=random.uniform(50, 300),
            total_amount=0,  # Will be calculated by validator
            fees=random.uniform(0, 10),
            notes="Market order executed",
            timestamp=datetime.utcnow() - timedelta(days=random.randint(0, 30))
        ))
    
    return PortfolioDetail(
        id=portfolio_id,
        name="Main Portfolio",
        total_value=round(total_value, 2),
        total_cost=round(total_cost, 2),
        total_gain=round(total_value - total_cost, 2),
        total_gain_percent=round((total_value - total_cost) / total_cost * 100, 2),
        cash_balance=random.uniform(1000, 50000),
        buying_power=random.uniform(5000, 100000),
        day_change=random.uniform(-5000, 5000),
        day_change_percent=random.uniform(-2, 2),
        positions_count=len(positions),
        strategy=PortfolioStrategy.BALANCED,
        risk_score=random.uniform(30, 70),
        created_at=datetime.utcnow() - timedelta(days=180),
        last_updated=datetime.utcnow(),
        positions=positions,
        asset_allocation=asset_allocation,
        sector_allocation=sector_allocation,
        top_performers=top_performers,
        worst_performers=worst_performers,
        recent_transactions=recent_transactions,
        performance_metrics=calculate_performance_metrics()
    )

@router.post("/{portfolio_id}/positions", response_model=Dict[str, Any])
async def add_position(
    portfolio_id: str,
    request: AddPositionRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
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
        timestamp=datetime.utcnow()
    )
    
    # Background task to update portfolio metrics
    background_tasks.add_task(update_portfolio_metrics, portfolio_id)
    
    return {
        "message": f"Successfully added {request.quantity} shares of {request.symbol}",
        "transaction": transaction.dict(),
        "portfolio_id": portfolio_id
    }

@router.delete("/{portfolio_id}/positions/{symbol}", response_model=Dict[str, Any])
async def remove_position(
    portfolio_id: str,
    symbol: str,
    request: RemovePositionRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
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
        timestamp=datetime.utcnow()
    )
    
    # Background task to update portfolio metrics
    background_tasks.add_task(update_portfolio_metrics, portfolio_id)
    
    return {
        "message": f"Successfully sold {quantity_to_sell} shares of {symbol}",
        "transaction": transaction.dict(),
        "portfolio_id": portfolio_id,
        "realized_gain": random.uniform(-1000, 5000)
    }

@router.get("/{portfolio_id}/transactions", response_model=List[Transaction])
async def get_transactions(
    portfolio_id: str,
    limit: int = Query(50, le=500),
    offset: int = 0,
    transaction_type: Optional[TransactionType] = None,
    symbol: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None
) -> List[Transaction]:
    """Get portfolio transaction history"""
    
    # Generate sample transactions
    transactions = []
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "NVDA", "TSLA"]
    
    for i in range(100):
        trans_date = datetime.utcnow() - timedelta(days=random.randint(0, 365))
        
        if start_date and trans_date.date() < start_date:
            continue
        if end_date and trans_date.date() > end_date:
            continue
        
        trans = Transaction(
            id=str(uuid.uuid4()),
            portfolio_id=portfolio_id,
            symbol=symbol.upper() if symbol else random.choice(symbols),
            transaction_type=transaction_type or random.choice(list(TransactionType)),
            quantity=random.uniform(1, 50),
            price=random.uniform(50, 500),
            total_amount=0,  # Will be calculated
            fees=random.uniform(0, 10),
            notes="Transaction note",
            timestamp=trans_date
        )
        
        if transaction_type and trans.transaction_type != transaction_type:
            continue
        if symbol and trans.symbol != symbol.upper():
            continue
        
        transactions.append(trans)
    
    # Sort by timestamp descending
    transactions.sort(key=lambda x: x.timestamp, reverse=True)
    
    return transactions[offset:offset + limit]

@router.get("/{portfolio_id}/performance", response_model=Dict[str, Any])
async def get_portfolio_performance(
    portfolio_id: str,
    period: str = Query("1M", regex="^(1D|1W|1M|3M|6M|1Y|3Y|5Y|ALL)$"),
    benchmark: str = "SPY"
) -> Dict[str, Any]:
    """Get portfolio performance over time"""
    
    # Generate performance data points
    data_points = []
    
    # Determine number of points based on period
    if period == "1D":
        num_points = 24
    elif period == "1W":
        num_points = 7
    elif period == "1M":
        num_points = 30
    elif period == "3M":
        num_points = 90
    elif period == "6M":
        num_points = 180
    elif period == "1Y":
        num_points = 252
    else:
        num_points = 365
    
    base_value = 100000
    for i in range(num_points):
        date_point = datetime.utcnow() - timedelta(days=num_points - i)
        value = base_value * (1 + random.uniform(-0.02, 0.02))
        base_value = value
        
        data_points.append({
            "date": date_point.date().isoformat(),
            "value": round(value, 2),
            "benchmark_value": round(value * random.uniform(0.95, 1.05), 2)
        })
    
    # Calculate metrics
    start_value = data_points[0]["value"]
    end_value = data_points[-1]["value"]
    total_return = (end_value - start_value) / start_value
    
    return {
        "portfolio_id": portfolio_id,
        "period": period,
        "data_points": data_points,
        "metrics": {
            "total_return": round(total_return, 4),
            "annualized_return": round(total_return * (365 / num_points), 4),
            "volatility": random.uniform(0.1, 0.3),
            "sharpe_ratio": random.uniform(0.5, 2.0),
            "max_drawdown": random.uniform(-0.2, -0.05),
            "benchmark_correlation": random.uniform(0.6, 0.95)
        },
        "vs_benchmark": {
            "excess_return": random.uniform(-0.05, 0.1),
            "tracking_error": random.uniform(0.02, 0.1),
            "information_ratio": random.uniform(-0.5, 1.5)
        }
    }

@router.post("/{portfolio_id}/analyze", response_model=PortfolioAnalysis)
async def analyze_portfolio(portfolio_id: str) -> PortfolioAnalysis:
    """Perform comprehensive portfolio analysis"""
    
    return PortfolioAnalysis(
        portfolio_id=portfolio_id,
        analysis_date=date.today(),
        risk_analysis={
            "var_95": random.uniform(-0.1, -0.02),
            "cvar_95": random.uniform(-0.15, -0.03),
            "downside_deviation": random.uniform(0.05, 0.15),
            "upside_potential": random.uniform(0.1, 0.3)
        },
        diversification_score=random.uniform(60, 90),
        concentration_risk={
            "top_holding": random.uniform(0.1, 0.3),
            "top_3_holdings": random.uniform(0.3, 0.5),
            "top_5_holdings": random.uniform(0.5, 0.7)
        },
        correlation_matrix={
            "AAPL": {"GOOGL": 0.7, "MSFT": 0.65, "AMZN": 0.6},
            "GOOGL": {"AAPL": 0.7, "MSFT": 0.75, "AMZN": 0.65},
            "MSFT": {"AAPL": 0.65, "GOOGL": 0.75, "AMZN": 0.6}
        },
        efficient_frontier={
            "current_position": {"return": 0.12, "risk": 0.15},
            "optimal_position": {"return": 0.14, "risk": 0.14},
            "improvement_potential": 0.02
        },
        optimization_suggestions=[
            "Reduce concentration in Technology sector",
            "Consider adding international exposure",
            "Increase allocation to fixed income for better risk-adjusted returns",
            "Review positions with high correlation"
        ],
        rebalancing_needed=random.choice([True, False]),
        recommended_changes=[
            {"action": "reduce", "symbol": "AAPL", "percent": 5},
            {"action": "increase", "symbol": "BND", "percent": 10},
            {"action": "add", "symbol": "VXUS", "percent": 5}
        ]
    )

@router.post("/{portfolio_id}/rebalance", response_model=Dict[str, Any])
async def rebalance_portfolio(
    portfolio_id: str,
    request: RebalanceRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Generate rebalancing recommendations"""
    
    # Validate target allocation sums to 100%
    total_allocation = sum(request.target_allocation.values())
    if abs(total_allocation - 100) > 0.01:
        raise HTTPException(status_code=400, detail="Target allocation must sum to 100%")
    
    # Generate rebalancing trades
    trades = []
    for asset_class, target_percent in request.target_allocation.items():
        current_percent = random.uniform(0, 30)
        difference = target_percent - current_percent
        
        if abs(difference) > 1:  # Only rebalance if difference > 1%
            action = "buy" if difference > 0 else "sell"
            trades.append({
                "asset_class": asset_class,
                "action": action,
                "amount": abs(difference) * 1000,  # Convert to dollar amount
                "current_allocation": round(current_percent, 2),
                "target_allocation": target_percent,
                "impact": round(difference, 2)
            })
    
    # Limit number of trades
    trades = trades[:request.max_trades]
    
    # Background task to execute rebalancing
    background_tasks.add_task(execute_rebalancing, portfolio_id, trades)
    
    return {
        "portfolio_id": portfolio_id,
        "rebalancing_plan": trades,
        "estimated_cost": sum(t["amount"] * 0.001 for t in trades),  # 0.1% transaction cost
        "tax_impact": random.uniform(-1000, -100) if request.tax_efficient else 0,
        "execution_status": "pending"
    }

@router.get("/{portfolio_id}/watchlist", response_model=List[WatchlistItem])
async def get_watchlist(portfolio_id: str) -> List[WatchlistItem]:
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
            added_date=datetime.utcnow() - timedelta(days=random.randint(1, 30))
        ))
    
    return watchlist

@router.post("/{portfolio_id}/watchlist", response_model=Dict[str, str])
async def add_to_watchlist(
    portfolio_id: str,
    item: WatchlistItem
) -> Dict[str, str]:
    """Add item to watchlist"""
    
    return {
        "message": f"Added {item.symbol} to watchlist",
        "portfolio_id": portfolio_id,
        "watchlist_id": str(uuid.uuid4())
    }

@router.put("/{portfolio_id}/settings", response_model=Dict[str, str])
async def update_portfolio_settings(
    portfolio_id: str,
    settings: PortfolioSettings
) -> Dict[str, str]:
    """Update portfolio settings"""
    
    return {
        "message": "Portfolio settings updated successfully",
        "portfolio_id": portfolio_id
    }

# Background task functions
async def update_portfolio_metrics(portfolio_id: str):
    """Update portfolio metrics after transaction"""
    # In production, this would recalculate portfolio metrics
    print(f"Updating metrics for portfolio {portfolio_id}")

async def execute_rebalancing(portfolio_id: str, trades: List[Dict]):
    """Execute rebalancing trades"""
    # In production, this would execute the trades
    print(f"Executing {len(trades)} trades for portfolio {portfolio_id}")