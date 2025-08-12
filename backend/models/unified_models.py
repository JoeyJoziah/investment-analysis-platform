"""
Unified Database Models for Investment Analysis Platform
Consolidates all SQLAlchemy ORM models into a single source of truth
"""

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Date, Boolean, 
    ForeignKey, Text, JSON, Numeric, Index, UniqueConstraint,
    CheckConstraint, Enum as SQLEnum, DECIMAL, BigInteger, func
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref
from datetime import datetime
import enum
import uuid

Base = declarative_base()

# ============================================================================
# ENUM DEFINITIONS
# ============================================================================

class UserRoleEnum(enum.Enum):
    SUPER_ADMIN = "super_admin"
    ADMIN = "admin"
    ANALYST = "analyst"
    PREMIUM_USER = "premium_user"
    BASIC_USER = "basic_user"
    FREE_USER = "free_user"

class OrderTypeEnum(enum.Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

class OrderSideEnum(enum.Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatusEnum(enum.Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class AssetTypeEnum(enum.Enum):
    STOCK = "stock"
    ETF = "etf"
    BOND = "bond"
    OPTION = "option"
    CRYPTO = "crypto"
    COMMODITY = "commodity"
    FOREX = "forex"

class RecommendationTypeEnum(enum.Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"

class TimeInForceEnum(enum.Enum):
    DAY = "day"
    GTC = "gtc"  # Good Till Cancelled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill
    GTD = "gtd"  # Good Till Date

# ============================================================================
# USER & AUTHENTICATION MODELS
# ============================================================================

class User(Base):
    """User accounts for the platform"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(36), default=lambda: str(uuid.uuid4()), unique=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    
    # Roles and permissions
    role = Column(String(50), default="free_user", nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    is_admin = Column(Boolean, default=False)
    is_premium = Column(Boolean, default=False)
    
    # Contact and preferences
    phone_number = Column(String(20))
    country = Column(String(2))
    timezone = Column(String(50), default="UTC")
    preferences = Column(JSON, default={})
    notification_settings = Column(JSON, default={
        "email_daily_summary": True,
        "email_trade_alerts": True,
        "email_price_alerts": True,
        "push_notifications": False
    })
    
    # Security
    two_factor_enabled = Column(Boolean, default=False)
    two_factor_secret = Column(String(255))
    api_key = Column(String(255), unique=True, index=True)
    api_secret = Column(String(255))
    
    # Subscription
    subscription_tier = Column(String(50))
    subscription_end_date = Column(DateTime)
    
    # Investment preferences
    risk_tolerance = Column(String(20))  # conservative, moderate, aggressive
    investment_style = Column(String(50))  # value, growth, momentum, etc.
    preferred_sectors = Column(JSON)
    excluded_sectors = Column(JSON)
    
    # Activity tracking
    last_login = Column(DateTime)
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    portfolios = relationship("Portfolio", back_populates="user", cascade="all, delete-orphan")
    watchlists = relationship("Watchlist", back_populates="user", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="user", cascade="all, delete-orphan")
    orders = relationship("Order", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_user_email_active', 'email', 'is_active'),
        Index('idx_user_role', 'role'),
        CheckConstraint('failed_login_attempts >= 0', name='check_failed_login_attempts'),
    )

class UserSession(Base):
    """User session tracking"""
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    session_token = Column(String(255), unique=True, nullable=False)
    ip_address = Column(String(45))
    user_agent = Column(String(500))
    device_info = Column(JSON)
    location = Column(String(255))
    
    created_at = Column(DateTime, default=func.now())
    expires_at = Column(DateTime, nullable=False)
    last_activity = Column(DateTime, default=func.now())
    is_active = Column(Boolean, default=True)
    
    # Relationship
    user = relationship("User", back_populates="sessions")
    
    __table_args__ = (
        Index('idx_session_token', 'session_token'),
        Index('idx_session_user_active', 'user_id', 'is_active'),
    )

# ============================================================================
# MARKET DATA MODELS
# ============================================================================

class Exchange(Base):
    """Stock exchanges"""
    __tablename__ = "exchanges"
    
    id = Column(Integer, primary_key=True)
    code = Column(String(10), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    timezone = Column(String(50))
    country = Column(String(2))
    currency = Column(String(3))
    
    # Trading hours
    market_open = Column(String(5))  # HH:MM format
    market_close = Column(String(5))  # HH:MM format
    
    # Relationships
    stocks = relationship("Stock", back_populates="exchange")

class Sector(Base):
    """Market sectors"""
    __tablename__ = "sectors"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text)
    
    # Relationships
    stocks = relationship("Stock", back_populates="sector")
    industries = relationship("Industry", back_populates="sector")

class Industry(Base):
    """Industries within sectors"""
    __tablename__ = "industries"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    sector_id = Column(Integer, ForeignKey("sectors.id"), nullable=False)
    description = Column(Text)
    
    # Relationships
    sector = relationship("Sector", back_populates="industries")
    stocks = relationship("Stock", back_populates="industry")

class Stock(Base):
    """Master stock table for all tickers"""
    __tablename__ = "stocks"
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    exchange_id = Column(Integer, ForeignKey("exchanges.id"), nullable=False)
    sector_id = Column(Integer, ForeignKey("sectors.id"))
    industry_id = Column(Integer, ForeignKey("industries.id"))
    
    # Stock details
    asset_type = Column(String(20), default="stock")
    market_cap = Column(Float)
    shares_outstanding = Column(BigInteger)
    ipo_date = Column(Date)
    country = Column(String(2), default="US")
    currency = Column(String(3), default="USD")
    description = Column(Text)
    website = Column(String(255))
    logo_url = Column(String(500))
    
    # Status
    is_active = Column(Boolean, default=True)
    is_tradeable = Column(Boolean, default=True)
    is_delisted = Column(Boolean, default=False)
    delisted_date = Column(Date)
    
    # Tracking
    last_updated = Column(DateTime, default=func.now(), onupdate=func.now())
    last_price_update = Column(DateTime)
    
    # Relationships
    exchange = relationship("Exchange", back_populates="stocks")
    sector = relationship("Sector", back_populates="stocks")
    industry = relationship("Industry", back_populates="stocks")
    price_history = relationship("PriceHistory", back_populates="stock", cascade="all, delete-orphan")
    fundamentals = relationship("Fundamentals", back_populates="stock", cascade="all, delete-orphan")
    technical_indicators = relationship("TechnicalIndicators", back_populates="stock", cascade="all, delete-orphan")
    news_sentiment = relationship("NewsSentiment", back_populates="stock", cascade="all, delete-orphan")
    predictions = relationship("MLPrediction", back_populates="stock", cascade="all, delete-orphan")
    recommendations = relationship("Recommendation", back_populates="stock", cascade="all, delete-orphan")
    positions = relationship("Position", back_populates="stock")
    watchlist_items = relationship("Watchlist", back_populates="stock")
    
    __table_args__ = (
        Index('idx_stock_exchange_sector', 'exchange_id', 'sector_id'),
        Index('idx_stock_active_tradeable', 'is_active', 'is_tradeable'),
        Index('idx_stock_symbol', 'symbol'),
    )

class PriceHistory(Base):
    """Historical price data (OHLCV)"""
    __tablename__ = "price_history"
    
    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    date = Column(DateTime, nullable=False)
    
    # OHLCV data
    open = Column(DECIMAL(10, 4), nullable=False)
    high = Column(DECIMAL(10, 4), nullable=False)
    low = Column(DECIMAL(10, 4), nullable=False)
    close = Column(DECIMAL(10, 4), nullable=False)
    adjusted_close = Column(DECIMAL(10, 4))
    volume = Column(BigInteger, nullable=False)
    
    # Additional metrics
    intraday_volatility = Column(Float)
    typical_price = Column(DECIMAL(10, 4))  # (H+L+C)/3
    vwap = Column(DECIMAL(10, 4))  # Volume Weighted Average Price
    
    # Relationship
    stock = relationship("Stock", back_populates="price_history")
    
    __table_args__ = (
        UniqueConstraint('stock_id', 'date', name='uq_stock_date'),
        Index('idx_price_date', 'date'),
        Index('idx_price_stock_date', 'stock_id', 'date'),
    )

# ============================================================================
# FUNDAMENTAL DATA MODELS
# ============================================================================

class Fundamentals(Base):
    """Fundamental financial data from SEC filings"""
    __tablename__ = "fundamentals"
    
    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    period_date = Column(Date, nullable=False)
    period_type = Column(String(10))  # quarterly, annual
    filing_date = Column(Date)
    
    # Income Statement
    revenue = Column(DECIMAL(20, 2))
    cost_of_revenue = Column(DECIMAL(20, 2))
    gross_profit = Column(DECIMAL(20, 2))
    operating_expenses = Column(DECIMAL(20, 2))
    operating_income = Column(DECIMAL(20, 2))
    net_income = Column(DECIMAL(20, 2))
    eps = Column(DECIMAL(10, 4))
    diluted_eps = Column(DECIMAL(10, 4))
    
    # Balance Sheet
    total_assets = Column(DECIMAL(20, 2))
    current_assets = Column(DECIMAL(20, 2))
    total_liabilities = Column(DECIMAL(20, 2))
    current_liabilities = Column(DECIMAL(20, 2))
    total_equity = Column(DECIMAL(20, 2))
    cash = Column(DECIMAL(20, 2))
    total_debt = Column(DECIMAL(20, 2))
    
    # Cash Flow
    operating_cash_flow = Column(DECIMAL(20, 2))
    investing_cash_flow = Column(DECIMAL(20, 2))
    financing_cash_flow = Column(DECIMAL(20, 2))
    free_cash_flow = Column(DECIMAL(20, 2))
    capex = Column(DECIMAL(20, 2))
    dividends_paid = Column(DECIMAL(20, 2))
    
    # Ratios (calculated)
    pe_ratio = Column(Float)
    pb_ratio = Column(Float)
    ps_ratio = Column(Float)
    peg_ratio = Column(Float)
    ev_ebitda = Column(Float)
    roe = Column(Float)
    roa = Column(Float)
    roic = Column(Float)
    current_ratio = Column(Float)
    quick_ratio = Column(Float)
    debt_to_equity = Column(Float)
    gross_margin = Column(Float)
    operating_margin = Column(Float)
    net_margin = Column(Float)
    
    # Relationship
    stock = relationship("Stock", back_populates="fundamentals")
    
    __table_args__ = (
        UniqueConstraint('stock_id', 'period_date', 'period_type', name='uq_fundamentals'),
        Index('idx_fundamentals_period', 'period_date', 'period_type'),
        Index('idx_fundamentals_stock', 'stock_id'),
    )

# ============================================================================
# TECHNICAL ANALYSIS MODELS
# ============================================================================

class TechnicalIndicators(Base):
    """Pre-calculated technical indicators"""
    __tablename__ = "technical_indicators"
    
    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    date = Column(DateTime, nullable=False)
    
    # Moving Averages
    sma_5 = Column(Float)
    sma_10 = Column(Float)
    sma_20 = Column(Float)
    sma_50 = Column(Float)
    sma_100 = Column(Float)
    sma_200 = Column(Float)
    ema_12 = Column(Float)
    ema_26 = Column(Float)
    ema_50 = Column(Float)
    
    # Momentum Indicators
    rsi_14 = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    macd_histogram = Column(Float)
    stochastic_k = Column(Float)
    stochastic_d = Column(Float)
    williams_r = Column(Float)
    cci = Column(Float)  # Commodity Channel Index
    
    # Volatility Indicators
    bollinger_upper = Column(Float)
    bollinger_middle = Column(Float)
    bollinger_lower = Column(Float)
    bollinger_width = Column(Float)
    atr_14 = Column(Float)  # Average True Range
    
    # Volume Indicators
    obv = Column(Float)  # On Balance Volume
    ad_line = Column(Float)  # Accumulation/Distribution
    mfi_14 = Column(Float)  # Money Flow Index
    vwap = Column(Float)  # Volume Weighted Average Price
    
    # Trend Indicators
    adx_14 = Column(Float)
    plus_di = Column(Float)
    minus_di = Column(Float)
    parabolic_sar = Column(Float)
    
    # Custom Indicators
    support_level = Column(Float)
    resistance_level = Column(Float)
    pivot_point = Column(Float)
    trend_strength = Column(Float)
    
    # Relationship
    stock = relationship("Stock", back_populates="technical_indicators")
    
    __table_args__ = (
        UniqueConstraint('stock_id', 'date', name='uq_technical_indicators'),
        Index('idx_technical_date', 'date'),
        Index('idx_technical_stock_date', 'stock_id', 'date'),
    )

# ============================================================================
# SENTIMENT & NEWS MODELS
# ============================================================================

class NewsSentiment(Base):
    """News and sentiment analysis data"""
    __tablename__ = "news_sentiment"
    
    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    source = Column(String(100))  # newsapi, reddit, twitter, benzinga, etc.
    source_id = Column(String(255))  # External ID from source
    
    # Content
    headline = Column(Text, nullable=False)
    summary = Column(Text)
    content = Column(Text)
    url = Column(String(500))
    author = Column(String(255))
    
    # Timestamps
    published_at = Column(DateTime, nullable=False)
    scraped_at = Column(DateTime, default=func.now())
    
    # Sentiment scores
    sentiment_score = Column(Float)  # -1 to 1
    sentiment_label = Column(String(20))  # positive, negative, neutral
    confidence = Column(Float)  # 0 to 1
    
    # Analysis metadata
    relevance_score = Column(Float)
    source_credibility = Column(Float)
    virality_score = Column(Float)
    impact_score = Column(Float)
    
    # Extracted entities
    keywords = Column(JSON)
    entities = Column(JSON)
    topics = Column(JSON)
    
    # Relationship
    stock = relationship("Stock", back_populates="news_sentiment")
    
    __table_args__ = (
        Index('idx_sentiment_date', 'published_at'),
        Index('idx_sentiment_stock_date', 'stock_id', 'published_at'),
        Index('idx_sentiment_score', 'sentiment_score'),
        Index('idx_sentiment_source', 'source'),
    )

# ============================================================================
# ML & PREDICTION MODELS
# ============================================================================

class MLPrediction(Base):
    """ML model predictions"""
    __tablename__ = "ml_predictions"
    
    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(20))
    prediction_date = Column(DateTime, default=func.now())
    target_date = Column(DateTime, nullable=False)
    prediction_horizon = Column(String(20))  # 1d, 1w, 1m, 3m, 6m, 1y
    
    # Price predictions
    predicted_price = Column(Float)
    predicted_low = Column(Float)
    predicted_high = Column(Float)
    predicted_return = Column(Float)
    predicted_volatility = Column(Float)
    
    # Direction predictions
    predicted_direction = Column(String(10))  # up, down, neutral
    direction_confidence = Column(Float)
    
    # Probability distributions
    prob_strong_buy = Column(Float)
    prob_buy = Column(Float)
    prob_hold = Column(Float)
    prob_sell = Column(Float)
    prob_strong_sell = Column(Float)
    
    # Model metadata
    features_used = Column(JSON)
    feature_importance = Column(JSON)
    model_params = Column(JSON)
    training_metrics = Column(JSON)
    
    # Validation
    actual_price = Column(Float)
    actual_return = Column(Float)
    prediction_error = Column(Float)
    was_correct = Column(Boolean)
    
    # Relationship
    stock = relationship("Stock", back_populates="predictions")
    
    __table_args__ = (
        Index('idx_prediction_date', 'prediction_date'),
        Index('idx_prediction_target', 'target_date'),
        Index('idx_prediction_stock_model', 'stock_id', 'model_name'),
        Index('idx_prediction_horizon', 'prediction_horizon'),
    )

class Recommendation(Base):
    """Final investment recommendations"""
    __tablename__ = "recommendations"
    
    id = Column(Integer, primary_key=True)
    recommendation_id = Column(String(36), default=lambda: str(uuid.uuid4()), unique=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    created_at = Column(DateTime, default=func.now())
    valid_until = Column(DateTime)
    
    # Recommendation
    action = Column(String(20), nullable=False)  # strong_buy, buy, hold, sell, strong_sell
    confidence = Column(Float, nullable=False)
    priority = Column(Integer)  # 1-10
    
    # Price targets
    entry_price = Column(DECIMAL(10, 4))
    target_price = Column(DECIMAL(10, 4))
    stop_loss = Column(DECIMAL(10, 4))
    expected_return = Column(Float)
    time_horizon_days = Column(Integer)
    
    # Risk metrics
    risk_score = Column(Float)
    risk_level = Column(String(20))  # low, medium, high
    volatility_percentile = Column(Float)
    beta = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    
    # Analysis scores
    technical_score = Column(Float)
    fundamental_score = Column(Float)
    sentiment_score = Column(Float)
    macro_score = Column(Float)
    alternative_score = Column(Float)
    overall_score = Column(Float)
    
    # Explanation
    reasoning = Column(Text)
    key_factors = Column(JSON)
    risks = Column(JSON)
    opportunities = Column(JSON)
    catalysts = Column(JSON)
    
    # Performance tracking
    is_active = Column(Boolean, default=True)
    actual_return = Column(Float)
    max_gain = Column(Float)
    max_loss = Column(Float)
    outcome = Column(String(20))  # success, partial, failure
    closed_at = Column(DateTime)
    
    # Relationship
    stock = relationship("Stock", back_populates="recommendations")
    
    __table_args__ = (
        Index('idx_recommendation_created', 'created_at'),
        Index('idx_recommendation_action', 'action', 'confidence'),
        Index('idx_recommendation_active', 'is_active', 'created_at'),
        Index('idx_recommendation_stock', 'stock_id'),
        CheckConstraint('confidence >= 0 AND confidence <= 1', name='check_confidence_range'),
        CheckConstraint('priority >= 1 AND priority <= 10', name='check_priority_range'),
    )

# ============================================================================
# PORTFOLIO & TRADING MODELS
# ============================================================================

class Portfolio(Base):
    """User portfolios"""
    __tablename__ = "portfolios"
    
    id = Column(Integer, primary_key=True)
    portfolio_id = Column(String(36), default=lambda: str(uuid.uuid4()), unique=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    
    # Portfolio settings
    is_public = Column(Boolean, default=False)
    is_default = Column(Boolean, default=False)
    benchmark = Column(String(10), default="SPY")
    target_allocation = Column(JSON)
    rebalance_frequency = Column(String(20))  # daily, weekly, monthly, quarterly
    
    # Portfolio metrics
    total_value = Column(DECIMAL(20, 2), default=0.0)
    cash_balance = Column(DECIMAL(20, 2), default=0.0)
    invested_value = Column(DECIMAL(20, 2), default=0.0)
    total_cost_basis = Column(DECIMAL(20, 2), default=0.0)
    
    # Performance
    total_return = Column(DECIMAL(20, 2), default=0.0)
    total_return_pct = Column(Float, default=0.0)
    daily_return = Column(Float)
    weekly_return = Column(Float)
    monthly_return = Column(Float)
    yearly_return = Column(Float)
    
    # Risk metrics
    volatility = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    beta = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    last_rebalanced = Column(DateTime)
    
    # Relationships
    user = relationship("User", back_populates="portfolios")
    positions = relationship("Position", back_populates="portfolio", cascade="all, delete-orphan")
    transactions = relationship("Transaction", back_populates="portfolio", cascade="all, delete-orphan")
    orders = relationship("Order", back_populates="portfolio")
    
    __table_args__ = (
        UniqueConstraint('user_id', 'name', name='uq_user_portfolio'),
        Index('idx_portfolio_user', 'user_id'),
    )

class Position(Base):
    """Portfolio positions"""
    __tablename__ = "positions"
    
    id = Column(Integer, primary_key=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    
    # Position details
    quantity = Column(DECIMAL(15, 6), nullable=False)
    avg_cost_basis = Column(DECIMAL(10, 4), nullable=False)
    total_cost_basis = Column(DECIMAL(20, 2))
    
    # Current values
    current_price = Column(DECIMAL(10, 4))
    market_value = Column(DECIMAL(20, 2))
    
    # Performance
    unrealized_gain_loss = Column(DECIMAL(20, 2))
    unrealized_gain_loss_pct = Column(Float)
    realized_gain_loss = Column(DECIMAL(20, 2))
    total_gain_loss = Column(DECIMAL(20, 2))
    
    # Position metrics
    weight = Column(Float)  # Portfolio weight percentage
    target_weight = Column(Float)
    
    # Dates
    first_purchase_date = Column(DateTime)
    last_transaction_date = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="positions")
    stock = relationship("Stock", back_populates="positions")
    
    __table_args__ = (
        UniqueConstraint('portfolio_id', 'stock_id', name='uq_portfolio_stock'),
        Index('idx_position_portfolio', 'portfolio_id'),
    )

class Transaction(Base):
    """Transaction history"""
    __tablename__ = "transactions"
    
    id = Column(Integer, primary_key=True)
    transaction_id = Column(String(36), default=lambda: str(uuid.uuid4()), unique=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    
    # Transaction details
    transaction_type = Column(String(10), nullable=False)  # buy, sell
    quantity = Column(DECIMAL(15, 6), nullable=False)
    price = Column(DECIMAL(10, 4), nullable=False)
    total_amount = Column(DECIMAL(20, 2), nullable=False)
    
    # Fees
    commission = Column(DECIMAL(10, 2), default=0)
    fees = Column(DECIMAL(10, 2), default=0)
    tax = Column(DECIMAL(10, 2), default=0)
    
    # Settlement
    trade_date = Column(DateTime, nullable=False)
    settlement_date = Column(DateTime)
    
    # Notes
    notes = Column(Text)
    order_id = Column(Integer, ForeignKey("orders.id"))
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="transactions")
    
    __table_args__ = (
        Index('idx_transaction_portfolio', 'portfolio_id'),
        Index('idx_transaction_date', 'trade_date'),
    )

class Order(Base):
    """Trading orders"""
    __tablename__ = "orders"
    
    id = Column(Integer, primary_key=True)
    order_id = Column(String(36), default=lambda: str(uuid.uuid4()), unique=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"))
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    
    # Order details
    order_type = Column(String(20), nullable=False)
    order_side = Column(String(10), nullable=False)
    quantity = Column(DECIMAL(15, 6), nullable=False)
    limit_price = Column(DECIMAL(10, 4))
    stop_price = Column(DECIMAL(10, 4))
    
    # Order settings
    time_in_force = Column(String(10), default="day")
    extended_hours = Column(Boolean, default=False)
    
    # Status
    status = Column(String(20), default="pending", nullable=False)
    filled_quantity = Column(DECIMAL(15, 6), default=0)
    average_fill_price = Column(DECIMAL(10, 4))
    
    # Fees
    commission = Column(DECIMAL(10, 2), default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    submitted_at = Column(DateTime)
    filled_at = Column(DateTime)
    cancelled_at = Column(DateTime)
    expires_at = Column(DateTime)
    
    # Error handling
    rejection_reason = Column(Text)
    error_message = Column(Text)
    
    # Relationships
    user = relationship("User", back_populates="orders")
    portfolio = relationship("Portfolio", back_populates="orders")
    
    __table_args__ = (
        Index('idx_order_user', 'user_id'),
        Index('idx_order_status', 'status'),
        Index('idx_order_created', 'created_at'),
    )

# ============================================================================
# WATCHLIST & ALERT MODELS
# ============================================================================

class Watchlist(Base):
    """User watchlists"""
    __tablename__ = "watchlists"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    name = Column(String(100))
    
    # Watchlist details
    added_date = Column(DateTime, default=func.now())
    notes = Column(Text)
    tags = Column(JSON)
    priority = Column(Integer, default=0)
    
    # Price tracking
    target_price = Column(DECIMAL(10, 4))
    stop_loss = Column(DECIMAL(10, 4))
    
    # Alerts
    alert_rules = Column(JSON)
    
    # Relationships
    user = relationship("User", back_populates="watchlists")
    stock = relationship("Stock", back_populates="watchlist_items")
    
    __table_args__ = (
        UniqueConstraint('user_id', 'stock_id', 'name', name='uq_user_watchlist'),
        Index('idx_watchlist_user', 'user_id'),
    )

class Alert(Base):
    """User alerts and notifications"""
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True)
    alert_id = Column(String(36), default=lambda: str(uuid.uuid4()), unique=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    stock_id = Column(Integer, ForeignKey("stocks.id"))
    
    # Alert details
    alert_type = Column(String(50), nullable=False)  # price, volume, news, recommendation, etc.
    condition = Column(JSON, nullable=False)
    message_template = Column(Text)
    
    # Status
    is_active = Column(Boolean, default=True)
    is_recurring = Column(Boolean, default=False)
    
    # Notification methods
    notification_methods = Column(JSON, default=["email"])
    
    # Tracking
    triggered_count = Column(Integer, default=0)
    last_triggered = Column(DateTime)
    last_checked = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    expires_at = Column(DateTime)
    
    # Relationships
    user = relationship("User", back_populates="alerts")
    
    __table_args__ = (
        Index('idx_alert_user_active', 'user_id', 'is_active'),
        Index('idx_alert_type', 'alert_type'),
    )

# ============================================================================
# SYSTEM & MONITORING MODELS
# ============================================================================

class APIUsage(Base):
    """Track API usage for cost monitoring"""
    __tablename__ = "api_usage"
    
    id = Column(Integer, primary_key=True)
    provider = Column(String(50), nullable=False)
    endpoint = Column(String(100))
    timestamp = Column(DateTime, default=func.now())
    
    # Usage metrics
    calls_count = Column(Integer, default=1)
    data_points = Column(Integer)
    response_time_ms = Column(Integer)
    success = Column(Boolean, default=True)
    
    # Error tracking
    status_code = Column(Integer)
    error_message = Column(Text)
    
    # Cost tracking
    estimated_cost = Column(DECIMAL(10, 6), default=0.0)
    
    __table_args__ = (
        Index('idx_api_usage_provider_time', 'provider', 'timestamp'),
        Index('idx_api_usage_timestamp', 'timestamp'),
    )

class AuditLog(Base):
    """Audit logging for compliance"""
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    
    # Action details
    action = Column(String(100), nullable=False)
    resource_type = Column(String(50))
    resource_id = Column(String(100))
    
    # Request details
    ip_address = Column(String(45))
    user_agent = Column(String(500))
    request_method = Column(String(10))
    request_path = Column(String(500))
    request_body = Column(JSON)
    
    # Response
    response_status = Column(Integer)
    response_time_ms = Column(Integer)
    
    # Additional metadata (renamed from 'metadata' to avoid SQLAlchemy conflict)
    meta_data = Column(JSON)
    
    # Timestamp
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="audit_logs")
    
    __table_args__ = (
        Index('idx_audit_user', 'user_id'),
        Index('idx_audit_action', 'action'),
        Index('idx_audit_timestamp', 'created_at'),
    )

class SystemMetrics(Base):
    """System performance metrics"""
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True)
    metric_type = Column(String(50), nullable=False)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    
    # Metadata
    labels = Column(JSON)
    unit = Column(String(20))
    
    # Timestamp
    timestamp = Column(DateTime, default=func.now())
    
    __table_args__ = (
        Index('idx_metrics_type_time', 'metric_type', 'timestamp'),
        Index('idx_metrics_name_time', 'metric_name', 'timestamp'),
    )

class CostMetrics(Base):
    """Cost metrics for API usage tracking"""
    __tablename__ = "cost_metrics"
    
    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False)
    provider = Column(String(50), nullable=False)
    api_calls = Column(Integer, default=0)
    successful_calls = Column(Integer, default=0)
    failed_calls = Column(Integer, default=0)
    cached_hits = Column(Integer, default=0)
    estimated_cost = Column(Float, default=0.0)
    actual_cost = Column(Float)  # Can be updated when actual bills arrive
    data_points_fetched = Column(Integer, default=0)
    average_latency_ms = Column(Float)
    error_rate = Column(Float)
    meta_data = Column(JSON, default={})  # Renamed from 'metadata' to avoid SQLAlchemy conflict
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    __table_args__ = (
        UniqueConstraint('date', 'provider', name='uq_cost_metrics_date_provider'),
        Index('idx_cost_metrics_date', 'date'),
        Index('idx_cost_metrics_provider', 'provider'),
        Index('idx_cost_metrics_date_provider', 'date', 'provider'),
    )

# ============================================================================
# CREATE ALL TABLES FUNCTION
# ============================================================================

def create_all_tables(engine):
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)

def drop_all_tables(engine):
    """Drop all database tables"""
    Base.metadata.drop_all(bind=engine)