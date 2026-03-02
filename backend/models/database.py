"""
Database Models - BACKWARD COMPATIBILITY SHIM

All ORM models are now defined in backend.models.unified_models (the canonical source).
This module re-exports everything for backward compatibility. New code should import
from backend.models.unified_models directly.
"""

# Re-export the canonical Base and all models from unified_models
from backend.models.unified_models import (  # noqa: F401
    Base,
    # Enums
    UserRoleEnum,
    OrderTypeEnum,
    OrderSideEnum,
    OrderStatusEnum,
    AssetTypeEnum,
    RecommendationTypeEnum,
    TimeInForceEnum,
    # User & Auth
    User,
    UserSession,
    # Market Data
    Exchange,
    Sector,
    Industry,
    Stock,
    PriceHistory,
    # Fundamentals
    Fundamentals,
    Fundamental,
    # Technical
    TechnicalIndicators,
    # Sentiment & News
    NewsSentiment,
    News,
    # ML
    MLPrediction,
    # Recommendations
    Recommendation,
    RecommendationPerformance,
    # Portfolio & Trading
    Portfolio,
    Position,
    Transaction,
    Order,
    # Watchlist & Alerts
    Watchlist,
    WatchlistItem,
    Alert,
    # System & Monitoring
    APIUsage,
    AuditLog,
    SystemMetrics,
    CostMetrics,
    ApiLog,
    SystemSettings,
    PortfolioPerformance,
    DividendHistory,
    # Utility functions
    create_all_tables,
    drop_all_tables,
)

# Legacy alias: some scripts imported "Prediction" from this module
# In unified_models it's called MLPrediction; provide alias for compat
Prediction = MLPrediction

# Legacy alias: some scripts imported "AlternativeData" from the old database.py
# This class was never migrated to unified_models. If needed, define a stub or skip.
# For now, we do not export AlternativeData since no tests depend on it.

# Backward-compatible SessionLocal stub.
# The real SessionLocal lives in backend.utils.database.  Some monitoring code
# previously imported it from here by mistake.  Provide a lazy proxy so those
# imports don't crash at module load time (they'll still fail at call time if
# the DB isn't configured, but that's the existing behavior).
def __getattr__(name):
    if name == "SessionLocal":
        from backend.utils.database import SessionLocal
        return SessionLocal
    if name == "get_db_session":
        from backend.utils.database import get_db
        return get_db
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
