"""
Pydantic schemas for request/response models and data validation
"""
from pydantic import BaseModel, Field, EmailStr, field_validator, constr, model_validator
from datetime import datetime, date, time
from typing import Optional, List, Dict, Any, Union
from decimal import Decimal
from enum import Enum
from typing_extensions import Annotated

# Enums
class UserRole(str, Enum):
    SUPER_ADMIN = "super_admin"
    ADMIN = "admin"
    ANALYST = "analyst"
    PREMIUM_USER = "premium_user"
    BASIC_USER = "basic_user"
    FREE_USER = "free_user"

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class AssetType(str, Enum):
    STOCK = "stock"
    ETF = "etf"
    BOND = "bond"
    OPTION = "option"
    CRYPTO = "crypto"
    COMMODITY = "commodity"
    FOREX = "forex"

class TimeInForce(str, Enum):
    DAY = "day"
    GTC = "gtc"  # Good Till Cancelled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill
    GTD = "gtd"  # Good Till Date

class AnalysisType(str, Enum):
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    QUANTITATIVE = "quantitative"
    COMPREHENSIVE = "comprehensive"

class RecommendationType(str, Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"

# Base Schemas
class BaseSchema(BaseModel):
    class Config:
        from_attributes = True
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat(),
            Decimal: lambda v: float(v)
        }

# User Schemas
class UserBase(BaseSchema):
    email: EmailStr
    full_name: str
    role: UserRole = UserRole.FREE_USER
    is_active: bool = True
    phone_number: Optional[Annotated[str, Field(pattern=r'^\+?1?\d{9,15}$')]] = None
    country: Optional[str] = None
    timezone: Optional[str] = "UTC"
    preferences: Optional[Dict[str, Any]] = {}

class UserCreate(UserBase):
    password: Annotated[str, Field(min_length=8, max_length=128)]
    confirm_password: str
    accept_terms: bool

    @field_validator('confirm_password')
    @classmethod
    def passwords_match(cls, v, info):
        if 'password' in info.data and v != info.data['password']:
            raise ValueError('Passwords do not match')
        return v

    @field_validator('accept_terms')
    @classmethod
    def terms_accepted(cls, v):
        if not v:
            raise ValueError('Terms must be accepted')
        return v

class UserUpdate(BaseSchema):
    full_name: Optional[str] = None
    phone_number: Optional[str] = None
    country: Optional[str] = None
    timezone: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None
    notification_settings: Optional[Dict[str, bool]] = None

class UserInDB(UserBase):
    id: int
    hashed_password: str
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None
    email_verified: bool = False
    two_factor_enabled: bool = False
    api_key: Optional[str] = None
    subscription_end_date: Optional[datetime] = None

class UserResponse(UserBase):
    id: int
    created_at: datetime
    last_login: Optional[datetime]
    email_verified: bool
    subscription_status: str

# Stock Schemas
class StockBase(BaseSchema):
    symbol: Annotated[str, Field(min_length=1, max_length=10)]
    name: str
    exchange: str
    asset_type: AssetType = AssetType.STOCK
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[float] = None
    country: str = "US"
    currency: str = "USD"
    ipo_date: Optional[date] = None
    description: Optional[str] = None

    @field_validator('symbol')
    @classmethod
    def uppercase_stock_symbol(cls, v):
        return v.upper() if v else v

class StockCreate(StockBase):
    pass

class StockUpdate(BaseSchema):
    name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[float] = None
    description: Optional[str] = None

class StockInDB(StockBase):
    id: int
    is_active: bool = True
    created_at: datetime
    updated_at: datetime
    last_price_update: Optional[datetime] = None
    
class Stock(StockInDB):
    current_price: Optional[float] = None
    change: Optional[float] = None
    change_percent: Optional[float] = None
    volume: Optional[int] = None
    avg_volume: Optional[int] = None
    market_cap_formatted: Optional[str] = None

    @model_validator(mode='after')
    def format_market_cap(self):
        if self.market_cap:
            cap = self.market_cap
            if cap >= 1e12:
                self.market_cap_formatted = f"${cap/1e12:.2f}T"
            elif cap >= 1e9:
                self.market_cap_formatted = f"${cap/1e9:.2f}B"
            elif cap >= 1e6:
                self.market_cap_formatted = f"${cap/1e6:.2f}M"
            else:
                self.market_cap_formatted = f"${cap:,.0f}"
        return self

# Price History Schemas
class PriceHistoryBase(BaseSchema):
    stock_id: int
    date: date
    open: float = Field(..., gt=0)
    high: float = Field(..., gt=0)
    low: float = Field(..., gt=0)
    close: float = Field(..., gt=0)
    adjusted_close: Optional[float] = Field(None, gt=0)
    volume: int = Field(..., ge=0)

    @model_validator(mode='after')
    def validate_prices(self):
        if self.high < self.low:
            raise ValueError('High must be greater than or equal to low')
        if self.close < self.low or self.close > self.high:
            raise ValueError('Close must be between low and high')
        return self

class PriceHistoryCreate(PriceHistoryBase):
    pass

class PriceHistory(PriceHistoryBase):
    id: int
    created_at: datetime
    
class PriceHistoryBulkCreate(BaseSchema):
    stock_id: int
    data: List[PriceHistoryBase]

# Portfolio Schemas
class PortfolioBase(BaseSchema):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    is_public: bool = False
    benchmark: str = "SPY"
    target_allocation: Optional[Dict[str, float]] = None
    rebalance_frequency: Optional[str] = "quarterly"

    @field_validator('target_allocation')
    @classmethod
    def validate_allocation(cls, v):
        if v:
            total = sum(v.values())
            if abs(total - 100.0) > 0.01:
                raise ValueError('Target allocation must sum to 100%')
        return v

class PortfolioCreate(PortfolioBase):
    initial_cash: float = Field(0, ge=0)

class PortfolioUpdate(BaseSchema):
    name: Optional[str] = None
    description: Optional[str] = None
    is_public: Optional[bool] = None
    benchmark: Optional[str] = None
    target_allocation: Optional[Dict[str, float]] = None

class Portfolio(PortfolioBase):
    id: int
    user_id: int
    cash_balance: float
    total_value: float
    total_cost: float
    total_return: float
    total_return_percent: float
    created_at: datetime
    updated_at: datetime
    
class PortfolioWithPositions(Portfolio):
    positions: List['Position']
    transactions: Optional[List['Transaction']] = []
    performance: Optional['PortfolioPerformance'] = None

# Position Schemas
class PositionBase(BaseSchema):
    portfolio_id: int
    stock_id: int
    quantity: float = Field(..., gt=0)
    average_cost: float = Field(..., gt=0)
    
class PositionCreate(PositionBase):
    pass

class PositionUpdate(BaseSchema):
    quantity: Optional[float] = Field(None, gt=0)
    average_cost: Optional[float] = Field(None, gt=0)

class Position(PositionBase):
    id: int
    current_price: float
    market_value: float
    cost_basis: float
    unrealized_gain: float
    unrealized_gain_percent: float
    weight: float  # Portfolio weight percentage
    created_at: datetime
    updated_at: datetime
    stock: Optional[Stock] = None

# Transaction Schemas
class TransactionBase(BaseSchema):
    portfolio_id: int
    stock_id: int
    transaction_type: OrderSide
    quantity: float = Field(..., gt=0)
    price: float = Field(..., gt=0)
    commission: float = Field(0, ge=0)
    notes: Optional[str] = None
    executed_at: datetime

class TransactionCreate(TransactionBase):
    pass

class Transaction(TransactionBase):
    id: int
    total_amount: float = 0.0
    created_at: datetime
    stock: Optional[Stock] = None

    @model_validator(mode='after')
    def calculate_total(self):
        base_amount = self.quantity * self.price
        commission = self.commission or 0
        if self.transaction_type == OrderSide.BUY:
            self.total_amount = base_amount + commission
        else:
            self.total_amount = base_amount - commission
        return self

# Order Schemas
class OrderBase(BaseSchema):
    portfolio_id: int
    stock_id: int
    order_type: OrderType
    order_side: OrderSide
    quantity: float = Field(..., gt=0)
    limit_price: Optional[float] = Field(None, gt=0)
    stop_price: Optional[float] = Field(None, gt=0)
    time_in_force: TimeInForce = TimeInForce.DAY
    extended_hours: bool = False

    @model_validator(mode='after')
    def validate_order_prices(self):
        if self.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and not self.limit_price:
            raise ValueError('Limit price required for limit orders')
        if self.order_type in [OrderType.STOP, OrderType.STOP_LIMIT, OrderType.TRAILING_STOP] and not self.stop_price:
            raise ValueError('Stop price required for stop orders')
        return self

class OrderCreate(OrderBase):
    pass

class OrderUpdate(BaseSchema):
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    quantity: Optional[float] = None

class Order(OrderBase):
    id: int
    user_id: int
    status: OrderStatus
    filled_quantity: float = 0
    average_fill_price: Optional[float] = None
    commission: float = 0
    created_at: datetime
    updated_at: datetime
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None
    stock: Optional[Stock] = None

# Recommendation Schemas
class RecommendationBase(BaseSchema):
    stock_id: int
    recommendation_type: RecommendationType
    confidence_score: float = Field(..., ge=0, le=1)
    target_price: Optional[float] = Field(None, gt=0)
    stop_loss: Optional[float] = Field(None, gt=0)
    time_horizon_days: int = Field(30, gt=0)
    reasoning: str
    key_factors: List[str]
    risk_level: str

class RecommendationCreate(RecommendationBase):
    pass

class Recommendation(RecommendationBase):
    id: int
    current_price: float
    expected_return: float
    created_at: datetime
    valid_until: datetime
    is_active: bool
    stock: Optional[Stock] = None
    performance: Optional['RecommendationPerformance'] = None

class RecommendationPerformance(BaseSchema):
    recommendation_id: int
    entry_price: float
    current_price: float
    highest_price: float
    lowest_price: float
    actual_return: float
    days_active: int
    target_hit: bool
    stop_loss_hit: bool
    last_updated: datetime

# Analysis Schemas
class AnalysisRequest(BaseSchema):
    symbol: str
    analysis_types: List[AnalysisType]
    period_days: int = Field(30, gt=0, le=365)
    include_predictions: bool = True
    include_comparisons: bool = False
    comparison_symbols: Optional[List[str]] = None

class TechnicalIndicators(BaseSchema):
    rsi: Optional[float] = Field(None, ge=0, le=100)
    macd: Optional[Dict[str, float]] = None
    bollinger_bands: Optional[Dict[str, float]] = None
    moving_averages: Optional[Dict[str, float]] = None
    stochastic: Optional[Dict[str, float]] = None
    adx: Optional[float] = None
    atr: Optional[float] = None
    obv: Optional[float] = None
    fibonacci_levels: Optional[Dict[str, float]] = None
    pivot_points: Optional[Dict[str, float]] = None
    support_resistance: Optional[Dict[str, List[float]]] = None

class FundamentalMetrics(BaseSchema):
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    peg_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ev_to_ebitda: Optional[float] = None
    ev_to_revenue: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    roic: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    fcf_yield: Optional[float] = None
    dividend_yield: Optional[float] = None
    payout_ratio: Optional[float] = None

class SentimentData(BaseSchema):
    overall_sentiment: float = Field(..., ge=-1, le=1)
    news_sentiment: Optional[float] = Field(None, ge=-1, le=1)
    social_sentiment: Optional[float] = Field(None, ge=-1, le=1)
    analyst_sentiment: Optional[float] = Field(None, ge=-1, le=1)
    insider_sentiment: Optional[float] = Field(None, ge=-1, le=1)
    options_sentiment: Optional[float] = Field(None, ge=-1, le=1)
    sentiment_momentum: Optional[str] = None
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0
    key_topics: List[str] = []
    recent_news: List[Dict[str, Any]] = []

class AnalysisResult(BaseSchema):
    symbol: str
    analysis_date: datetime
    technical: Optional[TechnicalIndicators] = None
    fundamental: Optional[FundamentalMetrics] = None
    sentiment: Optional[SentimentData] = None
    overall_score: float = Field(..., ge=0, le=100)
    recommendation: RecommendationType
    confidence: float = Field(..., ge=0, le=1)
    price_target: Optional[float] = None
    risk_score: float = Field(..., ge=0, le=100)
    opportunities: List[str] = []
    risks: List[str] = []
    key_insights: List[str] = []

# Alert Schemas
class AlertBase(BaseSchema):
    user_id: int
    alert_type: str
    condition: Dict[str, Any]
    is_active: bool = True
    notification_methods: List[str] = ["email"]
    
class AlertCreate(AlertBase):
    pass

class Alert(AlertBase):
    id: int
    triggered_count: int = 0
    last_triggered: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

# Watchlist Schemas
class WatchlistBase(BaseSchema):
    """Base watchlist schema."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    is_public: bool = False


class WatchlistCreate(WatchlistBase):
    """Schema for creating a watchlist."""
    pass


class WatchlistUpdate(BaseSchema):
    """Schema for updating a watchlist."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    is_public: Optional[bool] = None


class Watchlist(WatchlistBase):
    """Basic watchlist response schema."""
    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime


# Watchlist Item Schemas
class WatchlistItemBase(BaseSchema):
    """Base watchlist item schema."""
    target_price: Optional[float] = Field(None, gt=0, description="Target price alert")
    notes: Optional[str] = Field(None, max_length=500)
    alert_enabled: bool = False


class WatchlistItemCreate(WatchlistItemBase):
    """Schema for adding item to watchlist."""
    symbol: str = Field(..., min_length=1, max_length=10, description="Stock symbol to add")

    @field_validator('symbol')
    @classmethod
    def uppercase_symbol(cls, v):
        return v.upper() if v else v


class WatchlistItemUpdate(BaseSchema):
    """Schema for updating watchlist item."""
    target_price: Optional[float] = Field(None, ge=0)  # 0 clears the target price
    notes: Optional[str] = Field(None, max_length=500)
    alert_enabled: Optional[bool] = None


class WatchlistItemResponse(WatchlistItemBase):
    """Response schema with stock details."""
    id: int
    watchlist_id: int
    stock_id: int
    added_at: datetime
    # Stock details
    symbol: str
    company_name: Optional[str] = None
    current_price: Optional[float] = None
    price_change: Optional[float] = None
    price_change_percent: Optional[float] = None
    volume: Optional[int] = None
    market_cap: Optional[int] = None
    sector: Optional[str] = None


class WatchlistResponse(WatchlistBase):
    """Full watchlist response with items."""
    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime
    items: List[WatchlistItemResponse] = []
    item_count: int = 0

    @model_validator(mode='after')
    def set_item_count(self):
        self.item_count = len(self.items)
        return self


class WatchlistSummary(BaseSchema):
    """Summary for list view."""
    id: int
    name: str
    description: Optional[str] = None
    item_count: int = 0
    total_value: Optional[float] = None
    daily_change_percent: Optional[float] = None
    created_at: datetime
    updated_at: Optional[datetime] = None


class WatchlistWithStocks(WatchlistBase):
    """Legacy schema for backward compatibility."""
    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime
    stocks: Optional[List[Stock]] = []

# Performance Schemas
class PortfolioPerformance(BaseSchema):
    portfolio_id: int
    period: str
    start_date: date
    end_date: date
    starting_value: float
    ending_value: float
    total_return: float
    total_return_percent: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration_days: int
    win_rate: float
    profit_factor: float
    best_day: Optional[Dict[str, Any]] = None
    worst_day: Optional[Dict[str, Any]] = None
    benchmark_return: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    correlation: Optional[float] = None

# Settings Schemas
class UserSettings(BaseSchema):
    user_id: int
    theme: str = "dark"
    language: str = "en"
    currency: str = "USD"
    date_format: str = "YYYY-MM-DD"
    time_format: str = "24h"
    notifications: Dict[str, bool] = {
        "email_daily_summary": True,
        "email_trade_alerts": True,
        "email_price_alerts": True,
        "push_notifications": False
    }
    trading_preferences: Dict[str, Any] = {
        "default_order_type": "market",
        "default_time_in_force": "day",
        "confirm_orders": True,
        "extended_hours": False
    }
    display_preferences: Dict[str, Any] = {
        "show_positions": True,
        "show_watchlist": True,
        "show_news": True,
        "chart_type": "candlestick",
        "chart_indicators": ["volume", "sma20", "sma50"]
    }

class SystemSettings(BaseSchema):
    maintenance_mode: bool = False
    allow_registrations: bool = True
    require_email_verification: bool = True
    max_api_calls_per_minute: int = 60
    max_portfolio_size: int = 100
    max_watchlist_size: int = 50
    cache_ttl_seconds: int = 300
    data_retention_days: int = 365

# Investment Thesis Schemas
class InvestmentThesisBase(BaseSchema):
    investment_objective: str = Field(..., min_length=10, max_length=500)
    time_horizon: str = Field(..., pattern=r'^(short-term|medium-term|long-term)$')
    target_price: Optional[Decimal] = Field(None, gt=0)
    business_model: Optional[str] = Field(None, max_length=5000)
    competitive_advantages: Optional[str] = Field(None, max_length=5000)
    financial_health: Optional[str] = Field(None, max_length=5000)
    growth_drivers: Optional[str] = Field(None, max_length=5000)
    risks: Optional[str] = Field(None, max_length=5000)
    valuation_rationale: Optional[str] = Field(None, max_length=5000)
    exit_strategy: Optional[str] = Field(None, max_length=5000)
    content: Optional[str] = Field(None, max_length=50000)

class InvestmentThesisCreate(InvestmentThesisBase):
    stock_id: int = Field(..., gt=0)

class InvestmentThesisUpdate(BaseSchema):
    investment_objective: Optional[str] = Field(None, min_length=10, max_length=500)
    time_horizon: Optional[str] = Field(None, pattern=r'^(short-term|medium-term|long-term)$')
    target_price: Optional[Decimal] = Field(None, gt=0)
    business_model: Optional[str] = Field(None, max_length=5000)
    competitive_advantages: Optional[str] = Field(None, max_length=5000)
    financial_health: Optional[str] = Field(None, max_length=5000)
    growth_drivers: Optional[str] = Field(None, max_length=5000)
    risks: Optional[str] = Field(None, max_length=5000)
    valuation_rationale: Optional[str] = Field(None, max_length=5000)
    exit_strategy: Optional[str] = Field(None, max_length=5000)
    content: Optional[str] = Field(None, max_length=50000)

class InvestmentThesisResponse(InvestmentThesisBase):
    id: int
    user_id: int
    stock_id: int
    version: int
    created_at: datetime
    updated_at: datetime
    stock_symbol: Optional[str] = None
    stock_name: Optional[str] = None

# Forward references update
Portfolio.model_rebuild()
PortfolioWithPositions.model_rebuild()
Position.model_rebuild()
Transaction.model_rebuild()
Order.model_rebuild()
Recommendation.model_rebuild()
Watchlist.model_rebuild()
WatchlistResponse.model_rebuild()
WatchlistItemResponse.model_rebuild()
WatchlistWithStocks.model_rebuild()
InvestmentThesisResponse.model_rebuild()