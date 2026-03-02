"""
SQLAlchemy ORM Models - BACKWARD COMPATIBILITY SHIM

All ORM models are now defined in backend.models.unified_models (the canonical source).
This module re-exports everything for backward compatibility. New code should import
from backend.models.unified_models directly.
"""

import enum

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

# Legacy aliases: tables.py previously defined these ML-related enums
# that are not in unified_models. Re-define for backward compat.


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
