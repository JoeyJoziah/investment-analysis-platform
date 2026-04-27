"""
Application Configuration Settings - Security Hardened Version

This module provides security-hardened configuration with validation for:
- Secure secret key generation and validation (rejects weak/default keys)
- DEBUG mode protection in production
- Environment-based security controls

To use this instead of the original settings.py, update imports to:
    from backend.config.settings_secure import settings
"""

from pydantic_settings import BaseSettings
from pydantic import field_validator, model_validator
from typing import List, Optional
import os
import secrets
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


def _generate_secure_key(length: int = 64) -> str:
    """Generate a cryptographically secure random key"""
    return secrets.token_urlsafe(length)


def _validate_secret_key(value: str, key_name: str) -> str:
    """
    Validate that a secret key is secure and not a default/weak value.

    Security requirements:
    - Must not be empty
    - Must not contain common weak patterns
    - Must be at least 32 characters long
    """
    if not value:
        raise ValueError(f"{key_name} must be set - use a cryptographically secure random value")

    # List of weak patterns that indicate a default/placeholder value
    weak_patterns = [
        "your-", "change-", "secret", "password", "default",
        "example", "test", "dev", "123", "abc", "xxx", "changeme",
        "placeholder", "replace", "todo", "fixme"
    ]

    value_lower = value.lower()
    for pattern in weak_patterns:
        if pattern in value_lower:
            raise ValueError(
                f"{key_name} contains weak pattern '{pattern}'. "
                f"Use a cryptographically secure random value instead. "
                f"Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(64))\""
            )

    if len(value) < 32:
        raise ValueError(
            f"{key_name} must be at least 32 characters long for security. "
            f"Current length: {len(value)}. "
            f"Generate a secure key with: python -c \"import secrets; print(secrets.token_urlsafe(64))\""
        )

    return value


class Settings(BaseSettings):
    """
    Application settings with environment variable support and security validation
    """

    # Application
    APP_NAME: str = "Investment Analysis Platform"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "production"
    SECRET_KEY: str
    JWT_SECRET_KEY: str
    LOG_LEVEL: str = "INFO"

    @field_validator('SECRET_KEY')
    @classmethod
    def validate_secret_key(cls, v):
        """Validate SECRET_KEY is secure and not a default value"""
        return _validate_secret_key(v, "SECRET_KEY")

    @field_validator('JWT_SECRET_KEY')
    @classmethod
    def validate_jwt_secret_key(cls, v):
        """Validate JWT_SECRET_KEY is secure and not a default value"""
        return _validate_secret_key(v, "JWT_SECRET_KEY")

    @model_validator(mode='after')
    def validate_production_settings(self):
        """Ensure security settings are appropriate for production environment"""
        if self.ENVIRONMENT == 'production':
            if self.DEBUG:
                raise ValueError(
                    "DEBUG mode cannot be enabled in production environment. "
                    "Set ENVIRONMENT to 'development' or 'staging' to enable DEBUG mode, "
                    "or set DEBUG=false for production."
                )
            if self.LOG_LEVEL == 'DEBUG':
                # Warning: DEBUG logging in production can expose sensitive data
                logger.warning(
                    "DEBUG log level in production may expose sensitive information. "
                    "Consider using INFO or WARNING level instead."
                )
        return self

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
    ELASTICSEARCH_URL: str = "http://localhost:9200"

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
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    ALGORITHM: str = "HS256"

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

# Validate configuration on startup (legacy validation - now handled by pydantic validators)
if settings.is_production:
    assert settings.DATABASE_URL, "Must set DATABASE_URL in production"
    assert not settings.DATABASE_URL.startswith("postgresql://postgres:password"), "Must set proper DATABASE_URL in production"
