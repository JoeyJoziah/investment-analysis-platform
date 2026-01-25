"""
SQLAlchemy ORM models for database tables
"""
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Date, Boolean, 
    ForeignKey, Text, JSON, Numeric, Index, UniqueConstraint,
    CheckConstraint, Enum as SQLEnum, DECIMAL, BigInteger
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref
from sqlalchemy.sql import func
from datetime import datetime
import enum

Base = declarative_base()

# Enum classes for database
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

# ML Operations Enums
class ModelTypeEnum(enum.Enum):
    SKLEARN = "sklearn"
    PYTORCH = "pytorch"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    PROPHET = "prophet"
    ENSEMBLE = "ensemble"

class ModelStageEnum(enum.Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    RETIRED = "retired"
    ARCHIVED = "archived"

class FeatureTypeEnum(enum.Enum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"

class ComputeModeEnum(enum.Enum):
    BATCH = "batch"
    STREAMING = "streaming"
    ON_DEMAND = "on_demand"

class FeatureStatusEnum(enum.Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    RETIRED = "retired"

class DriftTypeEnum(enum.Enum):
    DATA_DRIFT = "data_drift"
    PREDICTION_DRIFT = "prediction_drift"
    CONCEPT_DRIFT = "concept_drift"
    PERFORMANCE_DRIFT = "performance_drift"

class AlertSeverityEnum(enum.Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    ERROR = "error"

class ModelHealthEnum(enum.Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    UNKNOWN = "unknown"

# User model
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    role = Column(SQLEnum(UserRoleEnum), default=UserRoleEnum.FREE_USER, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    phone_number = Column(String(20))
    country = Column(String(2))
    timezone = Column(String(50), default="UTC")
    preferences = Column(JSON, default={})
    notification_settings = Column(JSON, default={})
    two_factor_enabled = Column(Boolean, default=False)
    two_factor_secret = Column(String(255))
    api_key = Column(String(255), unique=True, index=True)
    api_secret = Column(String(255))
    subscription_tier = Column(String(50))
    subscription_end_date = Column(DateTime)
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
    api_logs = relationship("ApiLog", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_user_email_active', 'email', 'is_active'),
        Index('idx_user_role', 'role'),
        Index('idx_user_created', 'created_at'),
    )

# Stock model
class Stock(Base):
    __tablename__ = "stocks"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=False)
    exchange = Column(String(50), nullable=False)
    asset_type = Column(SQLEnum(AssetTypeEnum), default=AssetTypeEnum.STOCK, nullable=False)
    sector = Column(String(100), index=True)
    industry = Column(String(100))
    market_cap = Column(BigInteger)
    shares_outstanding = Column(BigInteger)
    float_shares = Column(BigInteger)
    country = Column(String(2), default="US")
    currency = Column(String(3), default="USD")
    ipo_date = Column(Date)
    description = Column(Text)
    website = Column(String(255))
    logo_url = Column(String(255))
    employees = Column(Integer)
    is_active = Column(Boolean, default=True, nullable=False)
    is_tradable = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    last_price_update = Column(DateTime)
    
    # Relationships
    price_history = relationship("PriceHistory", back_populates="stock", cascade="all, delete-orphan")
    recommendations = relationship("Recommendation", back_populates="stock", cascade="all, delete-orphan")
    positions = relationship("Position", back_populates="stock")
    transactions = relationship("Transaction", back_populates="stock")
    orders = relationship("Order", back_populates="stock")
    fundamentals = relationship("Fundamental", back_populates="stock", cascade="all, delete-orphan")
    news = relationship("News", back_populates="stock", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_stock_symbol_active', 'symbol', 'is_active'),
        Index('idx_stock_sector_industry', 'sector', 'industry'),
        Index('idx_stock_market_cap', 'market_cap'),
    )

# Price History model
class PriceHistory(Base):
    __tablename__ = "price_history"
    
    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id", ondelete="CASCADE"), nullable=False)
    date = Column(Date, nullable=False)
    open = Column(DECIMAL(10, 4), nullable=False)
    high = Column(DECIMAL(10, 4), nullable=False)
    low = Column(DECIMAL(10, 4), nullable=False)
    close = Column(DECIMAL(10, 4), nullable=False)
    adjusted_close = Column(DECIMAL(10, 4))
    volume = Column(BigInteger, nullable=False)
    split_coefficient = Column(Float, default=1.0)
    dividend_amount = Column(DECIMAL(10, 4), default=0)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    # Relationships
    stock = relationship("Stock", back_populates="price_history")
    
    # Constraints and Indexes
    __table_args__ = (
        UniqueConstraint('stock_id', 'date', name='uq_stock_date'),
        Index('idx_price_history_stock_date', 'stock_id', 'date'),
        Index('idx_price_history_date', 'date'),
        CheckConstraint('high >= low', name='check_high_gte_low'),
        CheckConstraint('close >= low AND close <= high', name='check_close_in_range'),
    )

# Portfolio model
class Portfolio(Base):
    __tablename__ = "portfolios"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    cash_balance = Column(DECIMAL(15, 2), default=0, nullable=False)
    is_public = Column(Boolean, default=False, nullable=False)
    is_default = Column(Boolean, default=False, nullable=False)
    benchmark = Column(String(10), default="SPY")
    target_allocation = Column(JSON)
    rebalance_frequency = Column(String(20))
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="portfolios")
    positions = relationship("Position", back_populates="portfolio", cascade="all, delete-orphan")
    transactions = relationship("Transaction", back_populates="portfolio", cascade="all, delete-orphan")
    orders = relationship("Order", back_populates="portfolio", cascade="all, delete-orphan")
    performance_history = relationship("PortfolioPerformance", back_populates="portfolio", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_portfolio_user', 'user_id'),
        Index('idx_portfolio_public', 'is_public'),
        UniqueConstraint('user_id', 'name', name='uq_user_portfolio_name'),
    )

# Position model
class Position(Base):
    __tablename__ = "positions"
    
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    quantity = Column(DECIMAL(15, 4), nullable=False)
    average_cost = Column(DECIMAL(10, 4), nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="positions")
    stock = relationship("Stock", back_populates="positions")
    
    # Constraints and Indexes
    __table_args__ = (
        UniqueConstraint('portfolio_id', 'stock_id', name='uq_portfolio_stock'),
        Index('idx_position_portfolio', 'portfolio_id'),
        Index('idx_position_stock', 'stock_id'),
        CheckConstraint('quantity > 0', name='check_quantity_positive'),
        CheckConstraint('average_cost > 0', name='check_cost_positive'),
    )

# Transaction model
class Transaction(Base):
    __tablename__ = "transactions"
    
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    transaction_type = Column(SQLEnum(OrderSideEnum), nullable=False)
    quantity = Column(DECIMAL(15, 4), nullable=False)
    price = Column(DECIMAL(10, 4), nullable=False)
    commission = Column(DECIMAL(10, 2), default=0)
    fees = Column(DECIMAL(10, 2), default=0)
    notes = Column(Text)
    executed_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="transactions")
    stock = relationship("Stock", back_populates="transactions")
    
    # Indexes
    __table_args__ = (
        Index('idx_transaction_portfolio', 'portfolio_id'),
        Index('idx_transaction_stock', 'stock_id'),
        Index('idx_transaction_executed', 'executed_at'),
        CheckConstraint('quantity > 0', name='check_trans_quantity_positive'),
        CheckConstraint('price > 0', name='check_trans_price_positive'),
    )

# Order model
class Order(Base):
    __tablename__ = "orders"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    order_type = Column(SQLEnum(OrderTypeEnum), nullable=False)
    order_side = Column(SQLEnum(OrderSideEnum), nullable=False)
    status = Column(SQLEnum(OrderStatusEnum), default=OrderStatusEnum.PENDING, nullable=False)
    quantity = Column(DECIMAL(15, 4), nullable=False)
    filled_quantity = Column(DECIMAL(15, 4), default=0)
    limit_price = Column(DECIMAL(10, 4))
    stop_price = Column(DECIMAL(10, 4))
    average_fill_price = Column(DECIMAL(10, 4))
    time_in_force = Column(String(10), default="day")
    extended_hours = Column(Boolean, default=False)
    commission = Column(DECIMAL(10, 2), default=0)
    external_order_id = Column(String(100))
    rejection_reason = Column(Text)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    submitted_at = Column(DateTime)
    filled_at = Column(DateTime)
    cancelled_at = Column(DateTime)
    expired_at = Column(DateTime)
    
    # Relationships
    user = relationship("User", back_populates="orders")
    portfolio = relationship("Portfolio", back_populates="orders")
    stock = relationship("Stock", back_populates="orders")
    
    # Indexes
    __table_args__ = (
        Index('idx_order_user_status', 'user_id', 'status'),
        Index('idx_order_portfolio', 'portfolio_id'),
        Index('idx_order_created', 'created_at'),
        Index('idx_order_external', 'external_order_id'),
        CheckConstraint('quantity > 0', name='check_order_quantity_positive'),
    )

# Recommendation model
class Recommendation(Base):
    __tablename__ = "recommendations"
    
    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id", ondelete="CASCADE"), nullable=False)
    recommendation_type = Column(SQLEnum(RecommendationTypeEnum), nullable=False)
    confidence_score = Column(Float, nullable=False)
    current_price = Column(DECIMAL(10, 4), nullable=False)
    target_price = Column(DECIMAL(10, 4))
    stop_loss = Column(DECIMAL(10, 4))
    time_horizon_days = Column(Integer, default=30)
    reasoning = Column(Text, nullable=False)
    key_factors = Column(JSON)
    risk_level = Column(String(20))
    technical_score = Column(Float)
    fundamental_score = Column(Float)
    sentiment_score = Column(Float)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    valid_until = Column(DateTime, nullable=False)
    
    # Relationships
    stock = relationship("Stock", back_populates="recommendations")
    performance = relationship("RecommendationPerformance", back_populates="recommendation", uselist=False, cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_recommendation_stock_active', 'stock_id', 'is_active'),
        Index('idx_recommendation_created', 'created_at'),
        Index('idx_recommendation_valid_until', 'valid_until'),
        CheckConstraint('confidence_score >= 0 AND confidence_score <= 1', name='check_confidence_range'),
        CheckConstraint('target_price > 0', name='check_target_price_positive'),
    )

# Recommendation Performance model
class RecommendationPerformance(Base):
    __tablename__ = "recommendation_performance"
    
    id = Column(Integer, primary_key=True, index=True)
    recommendation_id = Column(Integer, ForeignKey("recommendations.id", ondelete="CASCADE"), unique=True, nullable=False)
    entry_price = Column(DECIMAL(10, 4), nullable=False)
    current_price = Column(DECIMAL(10, 4), nullable=False)
    highest_price = Column(DECIMAL(10, 4), nullable=False)
    lowest_price = Column(DECIMAL(10, 4), nullable=False)
    actual_return = Column(Float)
    max_return = Column(Float)
    max_drawdown = Column(Float)
    days_active = Column(Integer)
    target_hit = Column(Boolean, default=False)
    stop_loss_hit = Column(Boolean, default=False)
    last_updated = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    recommendation = relationship("Recommendation", back_populates="performance")

# Watchlist model
class Watchlist(Base):
    __tablename__ = "watchlists"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    is_public = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="watchlists")
    items = relationship("WatchlistItem", back_populates="watchlist", cascade="all, delete-orphan")
    
    # Constraints and Indexes
    __table_args__ = (
        UniqueConstraint('user_id', 'name', name='uq_user_watchlist_name'),
        Index('idx_watchlist_user', 'user_id'),
        Index('idx_watchlist_public', 'is_public'),
    )

# Watchlist Item model
class WatchlistItem(Base):
    __tablename__ = "watchlist_items"
    
    id = Column(Integer, primary_key=True, index=True)
    watchlist_id = Column(Integer, ForeignKey("watchlists.id", ondelete="CASCADE"), nullable=False)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    target_price = Column(DECIMAL(10, 4))
    notes = Column(Text)
    alert_enabled = Column(Boolean, default=False)
    added_at = Column(DateTime, default=func.now(), nullable=False)
    
    # Relationships
    watchlist = relationship("Watchlist", back_populates="items")
    stock = relationship("Stock")
    
    # Constraints and Indexes
    __table_args__ = (
        UniqueConstraint('watchlist_id', 'stock_id', name='uq_watchlist_stock'),
        Index('idx_watchlist_item_watchlist', 'watchlist_id'),
    )

# Alert model
class Alert(Base):
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    stock_id = Column(Integer, ForeignKey("stocks.id"))
    alert_type = Column(String(50), nullable=False)
    condition = Column(JSON, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    notification_methods = Column(JSON, default=["email"])
    triggered_count = Column(Integer, default=0)
    last_triggered = Column(DateTime)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="alerts")
    stock = relationship("Stock")
    
    # Indexes
    __table_args__ = (
        Index('idx_alert_user_active', 'user_id', 'is_active'),
        Index('idx_alert_stock', 'stock_id'),
    )

# Fundamental data model
class Fundamental(Base):
    __tablename__ = "fundamentals"
    
    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id", ondelete="CASCADE"), nullable=False)
    report_date = Column(Date, nullable=False)
    period = Column(String(10), nullable=False)  # Q1, Q2, Q3, Q4, FY
    revenue = Column(BigInteger)
    gross_profit = Column(BigInteger)
    operating_income = Column(BigInteger)
    net_income = Column(BigInteger)
    eps = Column(DECIMAL(10, 4))
    eps_diluted = Column(DECIMAL(10, 4))
    total_assets = Column(BigInteger)
    total_liabilities = Column(BigInteger)
    total_equity = Column(BigInteger)
    cash = Column(BigInteger)
    debt = Column(BigInteger)
    free_cash_flow = Column(BigInteger)
    pe_ratio = Column(Float)
    peg_ratio = Column(Float)
    ps_ratio = Column(Float)
    pb_ratio = Column(Float)
    dividend_yield = Column(Float)
    roe = Column(Float)
    roa = Column(Float)
    roic = Column(Float)
    gross_margin = Column(Float)
    operating_margin = Column(Float)
    net_margin = Column(Float)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    # Relationships
    stock = relationship("Stock", back_populates="fundamentals")
    
    # Constraints and Indexes
    __table_args__ = (
        UniqueConstraint('stock_id', 'report_date', 'period', name='uq_stock_fundamental_period'),
        Index('idx_fundamental_stock_date', 'stock_id', 'report_date'),
    )

# News model
class News(Base):
    __tablename__ = "news"
    
    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id", ondelete="CASCADE"))
    headline = Column(String(500), nullable=False)
    summary = Column(Text)
    content = Column(Text)
    source = Column(String(100))
    url = Column(String(500))
    author = Column(String(100))
    sentiment_score = Column(Float)
    published_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    # Relationships
    stock = relationship("Stock", back_populates="news")
    
    # Indexes
    __table_args__ = (
        Index('idx_news_stock_published', 'stock_id', 'published_at'),
        Index('idx_news_published', 'published_at'),
        Index('idx_news_sentiment', 'sentiment_score'),
    )

# Portfolio Performance model
class PortfolioPerformance(Base):
    __tablename__ = "portfolio_performance"
    
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False)
    date = Column(Date, nullable=False)
    total_value = Column(DECIMAL(15, 2), nullable=False)
    cash_value = Column(DECIMAL(15, 2), nullable=False)
    positions_value = Column(DECIMAL(15, 2), nullable=False)
    daily_return = Column(Float)
    cumulative_return = Column(Float)
    benchmark_return = Column(Float)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="performance_history")
    
    # Constraints and Indexes
    __table_args__ = (
        UniqueConstraint('portfolio_id', 'date', name='uq_portfolio_performance_date'),
        Index('idx_portfolio_performance_date', 'portfolio_id', 'date'),
    )

# API Log model
class ApiLog(Base):
    __tablename__ = "api_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"))
    endpoint = Column(String(255), nullable=False)
    method = Column(String(10), nullable=False)
    status_code = Column(Integer)
    response_time_ms = Column(Integer)
    ip_address = Column(String(45))
    user_agent = Column(String(255))
    request_body = Column(JSON)
    response_body = Column(JSON)
    error_message = Column(Text)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="api_logs")
    
    # Indexes
    __table_args__ = (
        Index('idx_api_log_user', 'user_id'),
        Index('idx_api_log_endpoint', 'endpoint'),
        Index('idx_api_log_created', 'created_at'),
    )

# User Session model
class UserSession(Base):
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    session_token = Column(String(255), unique=True, index=True, nullable=False)
    refresh_token = Column(String(255), unique=True, index=True)
    ip_address = Column(String(45))
    user_agent = Column(String(255))
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    expires_at = Column(DateTime, nullable=False)
    last_activity = Column(DateTime, default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="sessions")
    
    # Indexes
    __table_args__ = (
        Index('idx_session_user_active', 'user_id', 'is_active'),
        Index('idx_session_expires', 'expires_at'),
    )

# System Settings model (singleton)
class SystemSettings(Base):
    __tablename__ = "system_settings"
    
    id = Column(Integer, primary_key=True, default=1)
    maintenance_mode = Column(Boolean, default=False, nullable=False)
    maintenance_message = Column(Text)
    allow_registrations = Column(Boolean, default=True, nullable=False)
    require_email_verification = Column(Boolean, default=True, nullable=False)
    max_api_calls_per_minute = Column(Integer, default=60)
    max_portfolio_size = Column(Integer, default=100)
    max_watchlist_size = Column(Integer, default=50)
    cache_ttl_seconds = Column(Integer, default=300)
    data_retention_days = Column(Integer, default=365)
    settings = Column(JSON, default={})
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    # Constraint to ensure only one row
    __table_args__ = (
        CheckConstraint('id = 1', name='check_single_row'),
    )


# Cost Metrics for API usage tracking
class CostMetrics(Base):
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


# Technical Indicators - Enhanced version
class TechnicalIndicators(Base):
    __tablename__ = "technical_indicators"
    
    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey('stocks.id'), nullable=False)
    date = Column(Date, nullable=False)
    
    # Moving Averages
    sma_20 = Column(Float)
    sma_50 = Column(Float)
    sma_200 = Column(Float)
    ema_12 = Column(Float)
    ema_26 = Column(Float)
    
    # Momentum Indicators
    rsi = Column(Float)  # Relative Strength Index
    macd = Column(Float)  # MACD Line
    macd_signal = Column(Float)  # Signal Line
    macd_histogram = Column(Float)  # MACD Histogram
    
    # Volatility Indicators
    bb_upper = Column(Float)  # Bollinger Band Upper
    bb_middle = Column(Float)  # Bollinger Band Middle
    bb_lower = Column(Float)  # Bollinger Band Lower
    atr = Column(Float)  # Average True Range
    
    # Volume Indicators
    volume_sma = Column(Float)  # Volume Simple Moving Average
    obv = Column(Float)  # On-Balance Volume
    vwap = Column(Float)  # Volume Weighted Average Price
    
    # Trend Indicators
    adx = Column(Float)  # Average Directional Index
    plus_di = Column(Float)  # Positive Directional Indicator
    minus_di = Column(Float)  # Negative Directional Indicator
    
    # Oscillators
    stochastic_k = Column(Float)  # Stochastic %K
    stochastic_d = Column(Float)  # Stochastic %D
    williams_r = Column(Float)  # Williams %R
    cci = Column(Float)  # Commodity Channel Index
    
    # Support/Resistance
    pivot_point = Column(Float)
    resistance_1 = Column(Float)
    resistance_2 = Column(Float)
    support_1 = Column(Float)
    support_2 = Column(Float)
    
    # Pattern Scores
    trend_strength = Column(Float)  # -100 to 100
    momentum_score = Column(Float)  # -100 to 100
    volatility_score = Column(Float)  # 0 to 100
    
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    stock = relationship("Stock", backref="technical_indicators")
    
    __table_args__ = (
        UniqueConstraint('stock_id', 'date', name='uq_technical_indicators_stock_date'),
        Index('idx_technical_indicators_stock_date', 'stock_id', 'date'),
        Index('idx_technical_indicators_date', 'date'),
        Index('idx_technical_indicators_rsi', 'rsi'),
        Index('idx_technical_indicators_trend', 'trend_strength'),
    )


# Audit Log for tracking system events
class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    action = Column(String(100), nullable=False)
    entity_type = Column(String(50), nullable=True)
    entity_id = Column(Integer, nullable=True)
    details = Column(JSON, nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)

    # Relationships
    user = relationship("User", backref="audit_logs")

    __table_args__ = (
        Index('idx_audit_logs_user', 'user_id'),
        Index('idx_audit_logs_action', 'action'),
        Index('idx_audit_logs_created_at', 'created_at'),
        Index('idx_audit_logs_entity', 'entity_type', 'entity_id'),
    )


# System Metrics for monitoring
class SystemMetrics(Base):
    __tablename__ = "system_metrics"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=func.now(), nullable=False)
    metric_type = Column(String(50), nullable=False)
    metrics = Column(JSON, nullable=False)

    __table_args__ = (
        Index('idx_system_metrics_timestamp', 'timestamp'),
        Index('idx_system_metrics_type', 'metric_type'),
        Index('idx_system_metrics_type_timestamp', 'metric_type', 'timestamp'),
    )