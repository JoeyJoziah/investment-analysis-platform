"""
Application Configuration Settings
"""

from pydantic_settings import BaseSettings
from typing import List, Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    """
    Application settings with environment variable support
    """
    
    # Application
    APP_NAME: str = "Investment Analysis Platform"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "production"
    SECRET_KEY: str
    JWT_SECRET_KEY: str
    LOG_LEVEL: str = "INFO"
    
    # API Keys (Free Tier)
    ALPHA_VANTAGE_API_KEY: Optional[str] = None
    FINNHUB_API_KEY: Optional[str] = None
    POLYGON_API_KEY: Optional[str] = None
    FMP_API_KEY: Optional[str] = None
    NEWS_API_KEY: Optional[str] = None
    MARKETAUX_API_KEY: Optional[str] = None
    FRED_API_KEY: Optional[str] = None
    OPENWEATHER_API_KEY: Optional[str] = None
    
    # Database
    DATABASE_URL: str
    REDIS_URL: str
    # Elasticsearch removed - using PostgreSQL full-text search instead (saves $15-20/month)
    ELASTICSEARCH_URL: Optional[str] = None
    
    # API Rate Limits (Free Tier)
    ALPHA_VANTAGE_DAILY_LIMIT: int = 25
    ALPHA_VANTAGE_MINUTE_LIMIT: int = 5
    POLYGON_MINUTE_LIMIT: int = 5
    FINNHUB_MINUTE_LIMIT: int = 60
    NEWS_API_DAILY_LIMIT: int = 100
    
    # Cost Monitoring
    MONTHLY_BUDGET_USD: float = 50.0
    ALERT_THRESHOLD_PERCENT: int = 80
    
    # ML Model Settings
    MODEL_CACHE_DIR: Path = Path("/app/models")
    ENABLE_GPU: bool = False
    BATCH_SIZE: int = 32
    MODEL_UPDATE_FREQUENCY_DAYS: int = 7
    
    # Security
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]

    # JWT Settings - NOTE: These are kept for backward compatibility.
    # The SINGLE SOURCE OF TRUTH for JWT config is backend/security/security_config.py
    # New code should import from SecurityConfig, not from settings.
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    ALGORITHM: str = "HS256"

    # GDPR Compliance
    GDPR_ENCRYPTION_KEY: Optional[str] = None  # Fernet key for PII encryption
    GDPR_DATA_RETENTION_DAYS: int = 2555  # 7 years for SEC compliance

    # SEC Compliance
    SEC_AUDIT_RETENTION_YEARS: int = 7
    SEC_RECOMMENDATION_DISCLOSURE_ENABLED: bool = True
    
    # Cache Settings
    CACHE_TTL_SECONDS: int = 3600  # 1 hour
    CACHE_MAX_SIZE: int = 10000
    
    # Analysis Settings
    MAX_STOCKS_PER_REQUEST: int = 100
    DEFAULT_ANALYSIS_PERIOD_DAYS: int = 365
    ENABLE_REAL_TIME_ANALYSIS: bool = True
    
    # Recommendation Settings
    MIN_CONFIDENCE_THRESHOLD: float = 0.6
    MAX_RECOMMENDATIONS_PER_DAY: int = 50
    RECOMMENDATION_UPDATE_HOUR: int = 6  # 6 AM UTC
    
    # Performance Settings
    MAX_WORKERS: int = 4
    ASYNC_TIMEOUT_SECONDS: int = 30
    API_TIMEOUT_SECONDS: int = 60
    
    # Monitoring
    PROMETHEUS_PORT: int = 9090
    GRAFANA_PORT: int = 3000
    ENABLE_METRICS: bool = True
    
    # Feature Flags
    ENABLE_ALTERNATIVE_DATA: bool = True
    ENABLE_SENTIMENT_ANALYSIS: bool = True
    ENABLE_TECHNICAL_ANALYSIS: bool = True
    ENABLE_FUNDAMENTAL_ANALYSIS: bool = True
    ENABLE_ML_PREDICTIONS: bool = True
    ENABLE_PORTFOLIO_OPTIMIZATION: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"  # Allow extra fields from .env file
        
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"
    
    @property
    def database_url_async(self) -> str:
        """Convert sync database URL to async"""
        return self.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a specific provider"""
        key_map = {
            "alpha_vantage": self.ALPHA_VANTAGE_API_KEY,
            "finnhub": self.FINNHUB_API_KEY,
            "polygon": self.POLYGON_API_KEY,
            "fmp": self.FMP_API_KEY,
            "news_api": self.NEWS_API_KEY,
            "marketaux": self.MARKETAUX_API_KEY,
            "fred": self.FRED_API_KEY,
            "openweather": self.OPENWEATHER_API_KEY
        }
        return key_map.get(provider.lower())
    
    def validate_api_keys(self) -> dict:
        """Validate which API keys are configured"""
        providers = [
            "alpha_vantage", "finnhub", "polygon", "fmp",
            "news_api", "marketaux", "fred", "openweather"
        ]
        return {
            provider: bool(self.get_api_key(provider))
            for provider in providers
        }


# Create settings instance
settings = Settings()

# Validate configuration on startup
if settings.is_production:
    assert settings.SECRET_KEY, "Must set SECRET_KEY in production"
    assert settings.JWT_SECRET_KEY, "Must set JWT_SECRET_KEY in production"
    assert settings.DATABASE_URL, "Must set DATABASE_URL in production"
    assert not settings.DATABASE_URL.startswith("postgresql://postgres:password"), "Must set proper DATABASE_URL in production"
    assert not settings.SECRET_KEY.startswith("your-"), "Must set proper SECRET_KEY in production"
    assert not settings.JWT_SECRET_KEY.startswith("your-"), "Must set proper JWT_SECRET_KEY in production"