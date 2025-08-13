"""
Consolidated Database Models for Investment Analysis Platform
This file consolidates and resolves conflicts between database.py and unified_models.py
"""

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Date, Boolean, 
    ForeignKey, Text, JSON, Numeric, Index, UniqueConstraint,
    CheckConstraint, DECIMAL, BigInteger, func, create_engine
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session
from datetime import datetime
import uuid

Base = declarative_base()

# ============================================================================
# REFERENCE DATA MODELS (Core Exchange/Sector data)
# ============================================================================

class Exchange(Base):
    """Stock exchanges - FIXED: Ensures 'code' field exists"""
    __tablename__ = "exchanges"
    
    id = Column(Integer, primary_key=True)
    code = Column(String(10), unique=True, nullable=False, index=True)  # NASDAQ, NYSE, AMEX
    name = Column(String(100), nullable=False)
    timezone = Column(String(50), default="America/New_York")
    country = Column(String(2), default="US")
    currency = Column(String(3), default="USD")
    
    # Trading hours
    market_open = Column(String(5), default="09:30")  # HH:MM format
    market_close = Column(String(5), default="16:00")  # HH:MM format
    
    # Relationships
    stocks = relationship("Stock", back_populates="exchange")
    
    __table_args__ = (
        Index('idx_exchange_code', 'code'),
    )

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

# ============================================================================
# STOCK DATA MODELS (Core ticker information)
# ============================================================================

class Stock(Base):
    """
    Master stock table - FIXED: Uses 'ticker' consistently
    Resolves field naming conflict between different model files
    """
    __tablename__ = "stocks"
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String(10), unique=True, nullable=False, index=True)  # FIXED: Use 'ticker' consistently
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
    
    # Status tracking
    is_active = Column(Boolean, default=True, nullable=False)
    is_tradeable = Column(Boolean, default=True, nullable=False)
    is_delisted = Column(Boolean, default=False, nullable=False)
    delisted_date = Column(Date)
    
    # Data quality tracking
    data_quality_score = Column(Float, default=100.0)  # 0-100 scale
    last_data_check = Column(DateTime)
    
    # Update tracking
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
    
    __table_args__ = (
        Index('idx_stock_ticker', 'ticker'),
        Index('idx_stock_exchange_sector', 'exchange_id', 'sector_id'),
        Index('idx_stock_active_tradeable', 'is_active', 'is_tradeable'),
        Index('idx_stock_market_cap', 'market_cap'),
    )

# ============================================================================
# PRICE HISTORY & TECHNICAL DATA
# ============================================================================

class PriceHistory(Base):
    """Historical price data with enhanced data quality validation"""
    __tablename__ = "price_history"
    
    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    date = Column(DateTime, nullable=False)
    
    # OHLCV data with precision for financial calculations
    open = Column(DECIMAL(12, 4), nullable=False)
    high = Column(DECIMAL(12, 4), nullable=False)
    low = Column(DECIMAL(12, 4), nullable=False)
    close = Column(DECIMAL(12, 4), nullable=False)
    adjusted_close = Column(DECIMAL(12, 4))
    volume = Column(BigInteger, nullable=False)
    
    # Data quality indicators
    is_validated = Column(Boolean, default=False)
    has_anomalies = Column(Boolean, default=False)
    anomaly_flags = Column(JSON)  # List of detected issues
    
    # Additional calculated fields
    daily_return = Column(Float)
    intraday_volatility = Column(Float)
    typical_price = Column(DECIMAL(12, 4))  # (H+L+C)/3
    
    # Data source tracking
    data_source = Column(String(50))  # alpha_vantage, yfinance, etc.
    created_at = Column(DateTime, default=func.now())
    
    # Relationship
    stock = relationship("Stock", back_populates="price_history")
    
    __table_args__ = (
        UniqueConstraint('stock_id', 'date', name='uq_stock_price_date'),
        Index('idx_price_date', 'date'),
        Index('idx_price_stock_date', 'stock_id', 'date'),
        Index('idx_price_validated', 'is_validated'),
        # Data quality constraints
        CheckConstraint('high >= low', name='check_high_low'),
        CheckConstraint('high >= open', name='check_high_open'),
        CheckConstraint('high >= close', name='check_high_close'),
        CheckConstraint('low <= open', name='check_low_open'),
        CheckConstraint('low <= close', name='check_low_close'),
        CheckConstraint('open > 0', name='check_positive_open'),
        CheckConstraint('high > 0', name='check_positive_high'),
        CheckConstraint('low > 0', name='check_positive_low'),
        CheckConstraint('close > 0', name='check_positive_close'),
        CheckConstraint('volume >= 0', name='check_positive_volume'),
    )

class TechnicalIndicators(Base):
    """Pre-calculated technical indicators"""
    __tablename__ = "technical_indicators"
    
    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    date = Column(DateTime, nullable=False)
    
    # Moving Averages
    sma_5 = Column(Float)
    sma_20 = Column(Float)
    sma_50 = Column(Float)
    sma_200 = Column(Float)
    ema_12 = Column(Float)
    ema_26 = Column(Float)
    
    # Momentum Indicators
    rsi_14 = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    macd_histogram = Column(Float)
    stochastic_k = Column(Float)
    stochastic_d = Column(Float)
    
    # Volatility Indicators
    bollinger_upper = Column(Float)
    bollinger_middle = Column(Float)
    bollinger_lower = Column(Float)
    atr_14 = Column(Float)
    
    # Volume Indicators
    obv = Column(Float)  # On Balance Volume
    ad_line = Column(Float)  # Accumulation/Distribution
    mfi_14 = Column(Float)  # Money Flow Index
    
    # Trend Indicators
    adx_14 = Column(Float)
    plus_di = Column(Float)
    minus_di = Column(Float)
    
    # Support/Resistance
    support_level = Column(Float)
    resistance_level = Column(Float)
    trend_strength = Column(Float)
    
    # Calculation metadata
    calculated_at = Column(DateTime, default=func.now())
    
    # Relationship
    stock = relationship("Stock", back_populates="technical_indicators")
    
    __table_args__ = (
        UniqueConstraint('stock_id', 'date', name='uq_technical_stock_date'),
        Index('idx_technical_date', 'date'),
        Index('idx_technical_stock_date', 'stock_id', 'date'),
    )

# ============================================================================
# FUNDAMENTAL DATA
# ============================================================================

class Fundamentals(Base):
    """Fundamental financial data from SEC filings"""
    __tablename__ = "fundamentals"
    
    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    period_date = Column(Date, nullable=False)
    period_type = Column(String(10))  # quarterly, annual
    filing_date = Column(Date)
    
    # Income Statement (in millions USD)
    revenue = Column(DECIMAL(20, 2))
    cost_of_revenue = Column(DECIMAL(20, 2))
    gross_profit = Column(DECIMAL(20, 2))
    operating_income = Column(DECIMAL(20, 2))
    net_income = Column(DECIMAL(20, 2))
    eps = Column(DECIMAL(10, 4))
    diluted_eps = Column(DECIMAL(10, 4))
    
    # Balance Sheet (in millions USD)
    total_assets = Column(DECIMAL(20, 2))
    total_liabilities = Column(DECIMAL(20, 2))
    total_equity = Column(DECIMAL(20, 2))
    cash = Column(DECIMAL(20, 2))
    total_debt = Column(DECIMAL(20, 2))
    
    # Cash Flow (in millions USD)
    operating_cash_flow = Column(DECIMAL(20, 2))
    free_cash_flow = Column(DECIMAL(20, 2))
    capex = Column(DECIMAL(20, 2))
    
    # Calculated Ratios
    pe_ratio = Column(Float)
    pb_ratio = Column(Float)
    ps_ratio = Column(Float)
    peg_ratio = Column(Float)
    ev_ebitda = Column(Float)
    roe = Column(Float)
    roa = Column(Float)
    roic = Column(Float)
    current_ratio = Column(Float)
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
# NEWS & SENTIMENT
# ============================================================================

class NewsSentiment(Base):
    """News and sentiment analysis data"""
    __tablename__ = "news_sentiment"
    
    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    source = Column(String(100))  # newsapi, reddit, benzinga, etc.
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
    
    # Sentiment scores (-1 to 1)
    sentiment_score = Column(Float)
    sentiment_label = Column(String(20))  # positive, negative, neutral
    confidence = Column(Float)  # 0 to 1
    
    # Analysis metadata
    relevance_score = Column(Float)
    source_credibility = Column(Float)
    impact_score = Column(Float)
    
    # Extracted data
    keywords = Column(JSON)
    entities = Column(JSON)
    
    # Relationship
    stock = relationship("Stock", back_populates="news_sentiment")
    
    __table_args__ = (
        Index('idx_sentiment_date', 'published_at'),
        Index('idx_sentiment_stock_date', 'stock_id', 'published_at'),
        Index('idx_sentiment_score', 'sentiment_score'),
        Index('idx_sentiment_source', 'source'),
    )

# ============================================================================
# ML PREDICTIONS & RECOMMENDATIONS
# ============================================================================

class MLPrediction(Base):
    """ML model predictions with enhanced validation"""
    __tablename__ = "ml_predictions"
    
    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(20))
    prediction_date = Column(DateTime, default=func.now())
    target_date = Column(DateTime, nullable=False)
    prediction_horizon = Column(String(20))  # 1d, 1w, 1m, 3m, 6m, 1y
    
    # Price predictions
    predicted_price = Column(DECIMAL(10, 4))
    predicted_low = Column(DECIMAL(10, 4))
    predicted_high = Column(DECIMAL(10, 4))
    predicted_return = Column(Float)
    predicted_volatility = Column(Float)
    
    # Direction predictions
    predicted_direction = Column(String(10))  # up, down, neutral
    direction_confidence = Column(Float)
    
    # Model metadata
    features_used = Column(JSON)
    model_accuracy = Column(Float)
    training_date = Column(DateTime)
    
    # Validation (updated when target date reached)
    actual_price = Column(DECIMAL(10, 4))
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
    """Final investment recommendations with performance tracking"""
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
    
    # Analysis scores (0-100)
    technical_score = Column(Float)
    fundamental_score = Column(Float)
    sentiment_score = Column(Float)
    macro_score = Column(Float)
    overall_score = Column(Float)
    
    # Explanation
    reasoning = Column(Text)
    key_factors = Column(JSON)
    risks = Column(JSON)
    opportunities = Column(JSON)
    
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
# COST & MONITORING
# ============================================================================

class APIUsage(Base):
    """Enhanced API usage tracking for cost monitoring"""
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
    retry_count = Column(Integer, default=0)
    
    # Cost tracking
    estimated_cost = Column(DECIMAL(10, 6), default=0.0)
    
    # Request details for debugging
    request_params = Column(JSON)
    response_size = Column(Integer)
    
    __table_args__ = (
        Index('idx_api_usage_provider_time', 'provider', 'timestamp'),
        Index('idx_api_usage_timestamp', 'timestamp'),
        Index('idx_api_usage_success', 'success', 'timestamp'),
    )

class CostMetrics(Base):
    """Daily cost metrics aggregation"""
    __tablename__ = "cost_metrics"
    
    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False)
    provider = Column(String(50), nullable=False)
    
    # Daily aggregations
    api_calls = Column(Integer, default=0)
    successful_calls = Column(Integer, default=0)
    failed_calls = Column(Integer, default=0)
    cached_hits = Column(Integer, default=0)
    
    # Cost tracking
    estimated_cost = Column(DECIMAL(10, 6), default=0.0)
    actual_cost = Column(DECIMAL(10, 6))  # Updated when bills arrive
    
    # Performance metrics
    data_points_fetched = Column(Integer, default=0)
    average_latency_ms = Column(Float)
    error_rate = Column(Float)
    cache_hit_rate = Column(Float)
    
    # Metadata
    notes = Column(Text)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        UniqueConstraint('date', 'provider', name='uq_cost_metrics_date_provider'),
        Index('idx_cost_metrics_date', 'date'),
        Index('idx_cost_metrics_provider', 'provider'),
        Index('idx_cost_metrics_date_provider', 'date', 'provider'),
    )

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_all_tables(engine):
    """Create all database tables with proper error handling"""
    try:
        Base.metadata.create_all(bind=engine)
        return True
    except Exception as e:
        print(f"Error creating tables: {e}")
        return False

def drop_all_tables(engine):
    """Drop all database tables"""
    Base.metadata.drop_all(bind=engine)

def verify_schema(engine):
    """Verify database schema matches model definitions"""
    from sqlalchemy import inspect
    
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()
    
    # Check critical tables exist
    required_tables = ['exchanges', 'sectors', 'industries', 'stocks', 'price_history']
    missing_tables = [table for table in required_tables if table not in existing_tables]
    
    if missing_tables:
        return False, f"Missing tables: {missing_tables}"
    
    # Check exchanges table has 'code' column
    exchanges_columns = [col['name'] for col in inspector.get_columns('exchanges')]
    if 'code' not in exchanges_columns:
        return False, "exchanges table missing 'code' column"
    
    # Check stocks table has 'ticker' column  
    stocks_columns = [col['name'] for col in inspector.get_columns('stocks')]
    if 'ticker' not in stocks_columns:
        return False, "stocks table missing 'ticker' column"
    
    return True, "Schema verification passed"

def get_exchange_by_code(session: Session, code: str):
    """Safe exchange lookup with proper error handling"""
    try:
        exchange = session.query(Exchange).filter(Exchange.code == code).first()
        if not exchange:
            # Create missing exchange
            exchange = Exchange(
                code=code,
                name=f"{code} Stock Exchange",
                timezone="America/New_York"
            )
            session.add(exchange)
            session.commit()
            print(f"Created missing exchange: {code}")
        return exchange
    except Exception as e:
        session.rollback()
        print(f"Error getting exchange {code}: {e}")
        return None