"""
Monitoring and logging configuration for the investment analysis application.
Centralizes all monitoring, logging, and alerting settings.
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    level: str = "INFO"
    format: str = "json"  # json or text
    service_name: str = "investment_analysis"
    environment: str = os.getenv("ENVIRONMENT", "production")
    version: str = os.getenv("APP_VERSION", "1.0.0")
    
    # File logging
    enable_file_logging: bool = True
    log_file_path: str = "/app/logs/application.log"
    log_file_max_size: int = 100 * 1024 * 1024  # 100MB
    log_file_backup_count: int = 10
    
    # Structured logging fields
    include_correlation_id: bool = True
    include_request_id: bool = True
    include_user_id: bool = True
    include_performance_metrics: bool = True
    
    # Log aggregation
    enable_log_aggregation: bool = True
    log_aggregation_endpoint: str = os.getenv("LOG_AGGREGATION_ENDPOINT", "")
    
    # Sensitive data masking
    mask_sensitive_data: bool = True
    sensitive_fields: List[str] = field(default_factory=lambda: [
        "password", "api_key", "token", "secret", "credit_card", "ssn"
    ])


@dataclass
class CacheConfig:
    """Cache configuration settings."""
    # Cache optimization
    enable_consistent_hashing: bool = True
    enable_key_hashing: bool = True
    hash_long_keys: bool = True
    hash_threshold: int = 128
    max_key_length: int = 512
    cache_version: str = "v2"
    
    # Cache nodes for distributed caching
    cache_nodes: List[str] = field(default_factory=lambda: [
        os.getenv("REDIS_NODE_1", "redis:6379"),
        os.getenv("REDIS_NODE_2", "redis:6380"),
        os.getenv("REDIS_NODE_3", "redis:6381"),
    ])
    
    # Virtual nodes for consistent hashing
    virtual_nodes_per_physical: int = 150
    
    # Cache TTL configurations by data type (seconds)
    ttl_config: Dict[str, int] = field(default_factory=lambda: {
        "price": 300,           # 5 minutes
        "fundamentals": 86400,  # 1 day
        "technical": 900,       # 15 minutes
        "sentiment": 3600,      # 1 hour
        "analysis": 1800,       # 30 minutes
        "recommendation": 3600, # 1 hour
    })
    
    # Cache warming settings
    enable_cache_warming: bool = True
    cache_warming_interval: int = 3600  # 1 hour
    cache_warming_batch_size: int = 100


@dataclass
class MetricsConfig:
    """Metrics and monitoring configuration."""
    # Prometheus settings
    enable_metrics: bool = True
    metrics_port: int = 8001
    metrics_path: str = "/metrics"
    
    # Data quality metrics
    enable_data_quality_metrics: bool = True
    dq_metrics_port: int = 8002
    dq_metrics_path: str = "/data-quality/metrics"
    
    # Metrics push gateway (optional)
    enable_push_gateway: bool = False
    push_gateway_url: str = os.getenv("PROMETHEUS_PUSH_GATEWAY", "")
    push_interval: int = 60  # seconds
    
    # Custom metrics endpoints
    custom_metrics_endpoints: Dict[str, str] = field(default_factory=lambda: {
        "circuit_breaker": "/circuit-breaker/metrics",
        "cost_monitor": "/cost-monitor/metrics",
        "rate_limiter": "/rate-limiter/metrics",
    })
    
    # Grafana settings
    grafana_url: str = os.getenv("GRAFANA_URL", "http://grafana:3000")
    grafana_api_key: str = os.getenv("GRAFANA_API_KEY", "")


@dataclass
class AlertingConfig:
    """Alerting configuration."""
    # Alert manager settings
    enable_alerting: bool = True
    alertmanager_url: str = os.getenv("ALERTMANAGER_URL", "http://alertmanager:9093")
    
    # Alert rules paths
    alert_rules_path: str = "/app/infrastructure/monitoring/prometheus/alerts"
    
    # Alert notification channels
    notification_channels: Dict[str, Dict] = field(default_factory=lambda: {
        "slack": {
            "enabled": os.getenv("SLACK_ALERTS_ENABLED", "false") == "true",
            "webhook_url": os.getenv("SLACK_WEBHOOK_URL", ""),
            "channel": "#alerts",
            "username": "Investment Analysis Bot"
        },
        "email": {
            "enabled": os.getenv("EMAIL_ALERTS_ENABLED", "false") == "true",
            "smtp_host": os.getenv("SMTP_HOST", ""),
            "smtp_port": int(os.getenv("SMTP_PORT", "587")),
            "from_address": os.getenv("ALERT_FROM_EMAIL", "alerts@investment-analysis.com"),
            "to_addresses": os.getenv("ALERT_TO_EMAILS", "").split(",")
        },
        "pagerduty": {
            "enabled": os.getenv("PAGERDUTY_ENABLED", "false") == "true",
            "integration_key": os.getenv("PAGERDUTY_KEY", "")
        }
    })
    
    # Alert thresholds
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        "budget_warning": 40.0,      # $40 warning threshold
        "budget_critical": 45.0,     # $45 critical threshold
        "api_latency_warning": 2.0,  # 2s warning
        "api_latency_critical": 5.0, # 5s critical
        "error_rate_warning": 0.05,  # 5% warning
        "error_rate_critical": 0.10, # 10% critical
        "data_quality_warning": 70,  # Quality score < 70
        "data_quality_critical": 50, # Quality score < 50
    })


@dataclass
class DataQualityConfig:
    """Data quality monitoring configuration."""
    # Quality check settings
    enable_quality_checks: bool = True
    check_interval: int = 300  # 5 minutes
    
    # Quality thresholds
    min_quality_score: float = 70.0
    max_missing_data_percent: float = 5.0
    max_price_gap_percent: float = 30.0
    max_volume_zscore: float = 3.0
    
    # Staleness thresholds (seconds)
    staleness_thresholds: Dict[str, int] = field(default_factory=lambda: {
        "price": 3600,         # 1 hour
        "fundamentals": 86400 * 7,  # 1 week
        "technical": 3600,     # 1 hour
        "sentiment": 7200,     # 2 hours
    })
    
    # Validation rules
    validation_rules: Dict[str, Dict] = field(default_factory=lambda: {
        "price": {
            "required_fields": ["open", "high", "low", "close", "volume"],
            "max_daily_change": 0.5,  # 50%
            "min_volume": 100,
        },
        "fundamentals": {
            "required_fields": ["market_cap", "pe_ratio", "eps"],
            "max_pe_ratio": 1000,
            "min_market_cap": 1000000,  # $1M
        }
    })


@dataclass
class ExceptionHandlingConfig:
    """Exception handling and recovery configuration."""
    # Retry settings
    enable_retries: bool = True
    max_retries: int = 3
    retry_backoff_base: float = 2.0  # Exponential backoff base
    
    # Circuit breaker settings
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: int = 60  # seconds
    circuit_breaker_expected_exception_types: List[str] = field(default_factory=lambda: [
        "APITimeoutException",
        "APIProviderException",
        "DatabaseConnectionException",
    ])
    
    # Recovery strategies by exception type
    recovery_strategies: Dict[str, str] = field(default_factory=lambda: {
        "RateLimitException": "fallback",
        "APITimeoutException": "retry",
        "APIDataException": "cache",
        "CacheConnectionException": "degrade",
        "BudgetExceededException": "cache",
        "DataQualityException": "degrade",
    })
    
    # Fallback data sources
    fallback_providers: Dict[str, List[str]] = field(default_factory=lambda: {
        "finnhub": ["alpha_vantage", "polygon", "cache"],
        "alpha_vantage": ["finnhub", "polygon", "cache"],
        "polygon": ["finnhub", "alpha_vantage", "cache"],
    })


@dataclass
class MonitoringConfig:
    """Master monitoring configuration."""
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    alerting: AlertingConfig = field(default_factory=AlertingConfig)
    data_quality: DataQualityConfig = field(default_factory=DataQualityConfig)
    exception_handling: ExceptionHandlingConfig = field(default_factory=ExceptionHandlingConfig)
    
    # Global settings
    enable_monitoring: bool = True
    enable_tracing: bool = True
    enable_profiling: bool = os.getenv("ENABLE_PROFILING", "false") == "true"
    
    # Performance settings
    metrics_collection_interval: int = 15  # seconds
    health_check_interval: int = 30  # seconds
    
    # Cost monitoring
    enable_cost_monitoring: bool = True
    monthly_budget: float = 50.0
    emergency_mode_threshold: float = 45.0
    
    # Compliance monitoring
    enable_compliance_monitoring: bool = True
    audit_log_retention_days: int = 90
    gdpr_data_retention_days: int = 365


# Global configuration instance
monitoring_config = MonitoringConfig()


# Helper function to initialize monitoring
def initialize_monitoring():
    """Initialize all monitoring components based on configuration."""
    from backend.utils.structured_logging import configure_structured_logging
    from backend.monitoring.data_quality_metrics import dq_metrics
    from backend.utils.monitoring import metrics as base_metrics
    
    # Configure structured logging
    if monitoring_config.logging.enable_file_logging:
        configure_structured_logging(
            service_name=monitoring_config.logging.service_name,
            environment=monitoring_config.logging.environment,
            version=monitoring_config.logging.version,
            log_level=monitoring_config.logging.level,
            log_file=monitoring_config.logging.log_file_path
        )
    else:
        configure_structured_logging(
            service_name=monitoring_config.logging.service_name,
            environment=monitoring_config.logging.environment,
            version=monitoring_config.logging.version,
            log_level=monitoring_config.logging.level
        )
    
    # Initialize metrics collectors
    if monitoring_config.metrics.enable_metrics:
        # Metrics are auto-initialized on import
        pass
    
    # Initialize data quality metrics
    if monitoring_config.metrics.enable_data_quality_metrics:
        # Data quality metrics auto-initialized on import
        pass
    
    return monitoring_config


# Export configuration
__all__ = [
    'MonitoringConfig',
    'LoggingConfig',
    'CacheConfig',
    'MetricsConfig',
    'AlertingConfig',
    'DataQualityConfig',
    'ExceptionHandlingConfig',
    'monitoring_config',
    'initialize_monitoring'
]