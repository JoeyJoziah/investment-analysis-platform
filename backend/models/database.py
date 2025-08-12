"""
Database Models for Investment Analysis Platform
"""

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, Text, JSON,
    ForeignKey, Index, UniqueConstraint, CheckConstraint, func
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()


class Exchange(Base):
    """
    Stock exchanges
    """
    __tablename__ = "exchanges"
    
    id = Column(Integer, primary_key=True)
    code = Column(String(10), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    timezone = Column(String(50))
    
    # Relationships
    stocks = relationship("Stock", back_populates="exchange")


class Sector(Base):
    """
    Market sectors
    """
    __tablename__ = "sectors"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    
    # Relationships
    stocks = relationship("Stock", back_populates="sector")
    industries = relationship("Industry", back_populates="sector")


class Industry(Base):
    """
    Industries within sectors
    """
    __tablename__ = "industries"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    sector_id = Column(Integer, ForeignKey("sectors.id"), nullable=False)
    
    # Relationships
    sector = relationship("Sector", back_populates="industries")
    stocks = relationship("Stock", back_populates="industry")


class Stock(Base):
    """
    Master stock table for all tickers
    """
    __tablename__ = "stocks"
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String(10), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    exchange_id = Column(Integer, ForeignKey("exchanges.id"), nullable=False)
    sector_id = Column(Integer, ForeignKey("sectors.id"))
    industry_id = Column(Integer, ForeignKey("industries.id"))
    market_cap = Column(Float)
    
    # Status
    is_active = Column(Boolean, default=True)
    is_tradeable = Column(Boolean, default=True)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    exchange = relationship("Exchange", back_populates="stocks")
    sector = relationship("Sector", back_populates="stocks")
    industry = relationship("Industry", back_populates="stocks")
    price_history = relationship("PriceHistory", back_populates="stock", cascade="all, delete-orphan")
    fundamentals = relationship("Fundamentals", back_populates="stock", cascade="all, delete-orphan")
    technical_indicators = relationship("TechnicalIndicators", back_populates="stock", cascade="all, delete-orphan")
    news_sentiment = relationship("NewsSentiment", back_populates="stock", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="stock", cascade="all, delete-orphan")
    recommendations = relationship("Recommendation", back_populates="stock", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_stock_exchange_sector', 'exchange_id', 'sector_id'),
        Index('idx_stock_active_tradeable', 'is_active', 'is_tradeable'),
    )


class PriceHistory(Base):
    """
    Historical price data (OHLCV)
    """
    __tablename__ = "price_history"
    
    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    date = Column(DateTime, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    adjusted_close = Column(Float)
    volume = Column(Integer, nullable=False)
    
    # Additional metrics
    intraday_volatility = Column(Float)
    typical_price = Column(Float)  # (H+L+C)/3
    vwap = Column(Float)  # Volume Weighted Average Price
    
    # Relationship
    stock = relationship("Stock", back_populates="price_history")
    
    __table_args__ = (
        UniqueConstraint('stock_id', 'date', name='uq_stock_date'),
        Index('idx_price_date', 'date'),
        Index('idx_price_stock_date', 'stock_id', 'date'),
    )


class Fundamentals(Base):
    """
    Fundamental financial data from SEC filings
    """
    __tablename__ = "fundamentals"
    
    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    period_date = Column(DateTime, nullable=False)
    period_type = Column(String(10))  # quarterly, annual
    
    # Income Statement
    revenue = Column(Float)
    gross_profit = Column(Float)
    operating_income = Column(Float)
    net_income = Column(Float)
    eps = Column(Float)
    diluted_eps = Column(Float)
    
    # Balance Sheet
    total_assets = Column(Float)
    total_liabilities = Column(Float)
    total_equity = Column(Float)
    cash = Column(Float)
    debt = Column(Float)
    
    # Cash Flow
    operating_cash_flow = Column(Float)
    free_cash_flow = Column(Float)
    capex = Column(Float)
    
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
    debt_to_equity = Column(Float)
    gross_margin = Column(Float)
    operating_margin = Column(Float)
    net_margin = Column(Float)
    
    # Relationship
    stock = relationship("Stock", back_populates="fundamentals")
    
    __table_args__ = (
        UniqueConstraint('stock_id', 'period_date', 'period_type', name='uq_fundamentals'),
        Index('idx_fundamentals_period', 'period_date', 'period_type'),
    )


class TechnicalIndicators(Base):
    """
    Pre-calculated technical indicators
    """
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
    
    # Custom Indicators
    support_level = Column(Float)
    resistance_level = Column(Float)
    trend_strength = Column(Float)
    
    # Relationship
    stock = relationship("Stock", back_populates="technical_indicators")
    
    __table_args__ = (
        UniqueConstraint('stock_id', 'date', name='uq_technical_indicators'),
        Index('idx_technical_date', 'date'),
        Index('idx_technical_stock_date', 'stock_id', 'date'),
    )


class NewsSentiment(Base):
    """
    News and sentiment analysis data
    """
    __tablename__ = "news_sentiment"
    
    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    source = Column(String(100))  # newsapi, reddit, twitter, etc.
    headline = Column(Text)
    content = Column(Text)
    url = Column(String(500))
    published_at = Column(DateTime, nullable=False)
    
    # Sentiment scores
    sentiment_score = Column(Float)  # -1 to 1
    sentiment_label = Column(String(20))  # positive, negative, neutral
    confidence = Column(Float)  # 0 to 1
    
    # Analysis metadata
    relevance_score = Column(Float)
    source_credibility = Column(Float)
    virality_score = Column(Float)
    
    # Keywords
    keywords = Column(JSON)
    entities = Column(JSON)
    
    # Relationship
    stock = relationship("Stock", back_populates="news_sentiment")
    
    __table_args__ = (
        Index('idx_sentiment_date', 'published_at'),
        Index('idx_sentiment_stock_date', 'stock_id', 'published_at'),
        Index('idx_sentiment_score', 'sentiment_score'),
    )


class AlternativeData(Base):
    """
    Alternative data sources
    """
    __tablename__ = "alternative_data"
    
    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    data_type = Column(String(50))  # google_trends, weather, satellite, etc.
    date = Column(DateTime, nullable=False)
    
    # Flexible data storage
    metrics = Column(JSON)
    
    # Common fields
    value = Column(Float)
    change_pct = Column(Float)
    trend = Column(String(20))
    
    __table_args__ = (
        Index('idx_alt_data_type_date', 'data_type', 'date'),
        Index('idx_alt_data_stock', 'stock_id', 'data_type', 'date'),
    )


class Prediction(Base):
    """
    ML model predictions
    """
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(20))
    prediction_date = Column(DateTime, default=datetime.utcnow)
    target_date = Column(DateTime, nullable=False)
    
    # Predictions
    predicted_price = Column(Float)
    predicted_return = Column(Float)
    predicted_direction = Column(String(10))  # up, down, neutral
    confidence = Column(Float)
    
    # Probability distribution
    prob_strong_buy = Column(Float)
    prob_buy = Column(Float)
    prob_hold = Column(Float)
    prob_sell = Column(Float)
    prob_strong_sell = Column(Float)
    
    # Model metadata
    features_used = Column(JSON)
    model_params = Column(JSON)
    
    # Relationship
    stock = relationship("Stock", back_populates="predictions")
    
    __table_args__ = (
        Index('idx_prediction_date', 'prediction_date'),
        Index('idx_prediction_target', 'target_date'),
        Index('idx_prediction_stock_model', 'stock_id', 'model_name'),
    )


class Recommendation(Base):
    """
    Final investment recommendations
    """
    __tablename__ = "recommendations"
    
    id = Column(Integer, primary_key=True)
    recommendation_id = Column(String(36), default=lambda: str(uuid.uuid4()), unique=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    valid_until = Column(DateTime)
    
    # Recommendation
    action = Column(String(20), nullable=False)  # strong_buy, buy, hold, sell, strong_sell
    confidence = Column(Float, nullable=False)
    priority = Column(Integer)  # 1-10
    
    # Price targets
    entry_price = Column(Float)
    target_price = Column(Float)
    stop_loss = Column(Float)
    expected_return = Column(Float)
    time_horizon_days = Column(Integer)
    
    # Risk metrics
    risk_score = Column(Float)
    volatility_percentile = Column(Float)
    beta = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    
    # Analysis summary
    technical_score = Column(Float)
    fundamental_score = Column(Float)
    sentiment_score = Column(Float)
    macro_score = Column(Float)
    alternative_score = Column(Float)
    
    # Explanation
    reasoning = Column(Text)
    key_factors = Column(JSON)
    risks = Column(JSON)
    catalysts = Column(JSON)
    
    # Performance tracking
    is_active = Column(Boolean, default=True)
    actual_return = Column(Float)
    outcome = Column(String(20))  # success, partial, failure
    
    # Relationship
    stock = relationship("Stock", back_populates="recommendations")
    
    __table_args__ = (
        Index('idx_recommendation_created', 'created_at'),
        Index('idx_recommendation_action', 'action', 'confidence'),
        Index('idx_recommendation_active', 'is_active', 'created_at'),
        CheckConstraint('confidence >= 0 AND confidence <= 1', name='check_confidence_range'),
        CheckConstraint('priority >= 1 AND priority <= 10', name='check_priority_range'),
    )


class APIUsage(Base):
    """
    Track API usage for cost monitoring
    """
    __tablename__ = "api_usage"
    
    id = Column(Integer, primary_key=True)
    provider = Column(String(50), nullable=False)
    endpoint = Column(String(100))
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Usage metrics
    calls_count = Column(Integer, default=1)
    data_points = Column(Integer)
    response_time_ms = Column(Integer)
    success = Column(Boolean, default=True)
    
    # Cost tracking
    estimated_cost = Column(Float, default=0.0)
    
    __table_args__ = (
        Index('idx_api_usage_provider_time', 'provider', 'timestamp'),
        Index('idx_api_usage_timestamp', 'timestamp'),
    )


class User(Base):
    """
    User accounts for the platform
    """
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(36), default=lambda: str(uuid.uuid4()), unique=True)
    email = Column(String(255), unique=True, nullable=False)
    username = Column(String(100), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    
    # Profile
    full_name = Column(String(255))
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    is_premium = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    
    # Preferences
    risk_tolerance = Column(String(20))  # conservative, moderate, aggressive
    investment_style = Column(String(50))  # value, growth, momentum, etc.
    preferred_sectors = Column(JSON)
    excluded_sectors = Column(JSON)
    
    # Relationships
    portfolios = relationship("Portfolio", back_populates="user", cascade="all, delete-orphan")
    watchlists = relationship("Watchlist", back_populates="user", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_user_email', 'email'),
        Index('idx_user_username', 'username'),
    )


class Portfolio(Base):
    """
    User portfolios
    """
    __tablename__ = "portfolios"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Portfolio metrics
    total_value = Column(Float, default=0.0)
    cash_balance = Column(Float, default=0.0)
    total_return = Column(Float, default=0.0)
    total_return_pct = Column(Float, default=0.0)
    
    # Relationships
    user = relationship("User", back_populates="portfolios")
    positions = relationship("Position", back_populates="portfolio", cascade="all, delete-orphan")
    
    __table_args__ = (
        UniqueConstraint('user_id', 'name', name='uq_user_portfolio'),
    )


class Position(Base):
    """
    Portfolio positions
    """
    __tablename__ = "positions"
    
    id = Column(Integer, primary_key=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    
    # Position details
    quantity = Column(Float, nullable=False)
    avg_cost_basis = Column(Float, nullable=False)
    current_price = Column(Float)
    market_value = Column(Float)
    
    # Performance
    unrealized_gain_loss = Column(Float)
    unrealized_gain_loss_pct = Column(Float)
    realized_gain_loss = Column(Float)
    
    # Dates
    first_purchase_date = Column(DateTime)
    last_transaction_date = Column(DateTime)
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="positions")
    
    __table_args__ = (
        UniqueConstraint('portfolio_id', 'stock_id', name='uq_portfolio_stock'),
    )


class Watchlist(Base):
    """
    User watchlists
    """
    __tablename__ = "watchlists"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    
    # Watchlist details
    added_date = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text)
    alert_rules = Column(JSON)
    
    # Relationships
    user = relationship("User", back_populates="watchlists")
    
    __table_args__ = (
        UniqueConstraint('user_id', 'stock_id', name='uq_user_watchlist'),
        Index('idx_watchlist_user', 'user_id'),
    )