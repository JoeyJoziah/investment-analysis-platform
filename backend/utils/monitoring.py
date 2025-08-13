"""Comprehensive monitoring and metrics collection"""

import asyncio
import time
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, Optional

from prometheus_client import (
    CollectorRegistry, Counter, Gauge, Histogram, Info, Summary,
    generate_latest, push_to_gateway
)

from backend.config.settings import settings
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Create registry
registry = CollectorRegistry()

# System metrics
system_info = Info(
    "investment_analysis_system",
    "System information",
    registry=registry
)

system_info.info({
    "version": settings.VERSION,
    "environment": settings.ENVIRONMENT,
    "python_version": "3.11"
})

# API metrics
api_requests_total = Counter(
    "api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"],
    registry=registry
)

api_request_duration = Histogram(
    "api_request_duration_seconds",
    "API request duration",
    ["method", "endpoint"],
    registry=registry
)

api_errors_total = Counter(
    "api_errors_total",
    "Total API errors",
    ["method", "endpoint", "error_type"],
    registry=registry
)

# External API metrics
api_calls_total = Counter(
    "external_api_calls_total",
    "Total external API calls",
    ["provider", "endpoint"],
    registry=registry
)

api_call_duration = Histogram(
    "external_api_call_duration_seconds",
    "External API call duration",
    ["provider", "endpoint"],
    registry=registry
)

api_failures_total = Counter(
    "external_api_failures_total",
    "Total external API failures",
    ["provider", "endpoint", "error_type"],
    registry=registry
)

api_cost_total = Counter(
    "api_cost_dollars_total",
    "Total API cost in dollars",
    ["provider"],
    registry=registry
)

# Rate limiting metrics
rate_limit_hits = Counter(
    "rate_limit_hits_total",
    "Total rate limit hits",
    ["limit_type", "resource"],
    registry=registry
)

rate_limit_remaining = Gauge(
    "rate_limit_remaining",
    "Remaining rate limit",
    ["provider", "limit_type"],
    registry=registry
)

# Database metrics
db_queries_total = Counter(
    "db_queries_total",
    "Total database queries",
    ["operation", "table"],
    registry=registry
)

db_query_duration = Histogram(
    "db_query_duration_seconds",
    "Database query duration",
    ["operation", "table"],
    registry=registry
)

db_connections_active = Gauge(
    "db_connections_active",
    "Active database connections",
    registry=registry
)

db_connection_errors = Counter(
    "db_connection_errors_total",
    "Database connection errors",
    ["error_type"],
    registry=registry
)

# Cache metrics
cache_hits_total = Counter(
    "cache_hits_total",
    "Total cache hits",
    ["cache_type", "operation"],
    registry=registry
)

cache_misses_total = Counter(
    "cache_misses_total",
    "Total cache misses",
    ["cache_type", "operation"],
    registry=registry
)

cache_size_bytes = Gauge(
    "cache_size_bytes",
    "Cache size in bytes",
    ["cache_type"],
    registry=registry
)

# ML model metrics
ml_predictions_total = Counter(
    "ml_predictions_total",
    "Total ML predictions",
    ["model", "ticker"],
    registry=registry
)

ml_prediction_duration = Histogram(
    "ml_prediction_duration_seconds",
    "ML prediction duration",
    ["model"],
    registry=registry
)

ml_prediction_accuracy = Gauge(
    "ml_prediction_accuracy",
    "ML prediction accuracy",
    ["model", "horizon"],
    registry=registry
)

ml_model_load_time = Histogram(
    "ml_model_load_time_seconds",
    "ML model load time",
    ["model"],
    registry=registry
)

# Analysis metrics
analysis_completed_total = Counter(
    "analysis_completed_total",
    "Total analyses completed",
    ["analysis_type", "status"],
    registry=registry
)

analysis_duration = Histogram(
    "analysis_duration_seconds",
    "Analysis duration",
    ["analysis_type"],
    registry=registry
)

stocks_analyzed_total = Counter(
    "stocks_analyzed_total",
    "Total stocks analyzed",
    ["exchange"],
    registry=registry
)

recommendations_generated = Counter(
    "recommendations_generated_total",
    "Total recommendations generated",
    ["action", "confidence_level"],
    registry=registry
)

# Cost tracking metrics
daily_cost_gauge = Gauge(
    "daily_cost_dollars",
    "Daily cost in dollars",
    ["service"],
    registry=registry
)

monthly_cost_projection = Gauge(
    "monthly_cost_projection_dollars",
    "Projected monthly cost in dollars",
    registry=registry
)

cost_budget_usage_percent = Gauge(
    "cost_budget_usage_percent",
    "Percentage of budget used",
    registry=registry
)

# Kafka metrics
kafka_messages_sent = Counter(
    "kafka_messages_sent_total",
    "Total Kafka messages sent",
    ["topic"],
    registry=registry
)

kafka_messages_received = Counter(
    "kafka_messages_received_total",
    "Total Kafka messages received",
    ["topic"],
    registry=registry
)

kafka_processing_lag = Histogram(
    "kafka_processing_lag_seconds",
    "Kafka message processing lag",
    ["topic"],
    registry=registry
)

# Circuit breaker metrics
circuit_breaker_state = Gauge(
    "circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=open, 2=half-open)",
    ["breaker_name"],
    registry=registry
)

circuit_breaker_failures = Counter(
    "circuit_breaker_failures_total",
    "Total circuit breaker failures",
    ["breaker_name"],
    registry=registry
)

# Compliance metrics
audit_logs_created = Counter(
    "audit_logs_created_total",
    "Total audit logs created",
    ["action", "user_type"],
    registry=registry
)

gdpr_requests = Counter(
    "gdpr_requests_total",
    "Total GDPR requests",
    ["request_type"],
    registry=registry
)

data_anonymization_operations = Counter(
    "data_anonymization_operations_total",
    "Total data anonymization operations",
    ["operation_type"],
    registry=registry
)


class MetricsCollector:
    """Collects and manages metrics"""
    
    def __init__(self):
        self.custom_metrics: Dict[str, Any] = {}
        
    @contextmanager
    def timer(self, metric: Histogram, labels: Optional[Dict[str, str]] = None):
        """Context manager for timing operations"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            if labels:
                metric.labels(**labels).observe(duration)
            else:
                metric.observe(duration)
                
    def track_request(
        self,
        method: str,
        endpoint: str,
        status: int,
        duration: float
    ):
        """Track API request metrics"""
        api_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status=str(status)
        ).inc()
        
        api_request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
        
    def track_external_api_call(
        self,
        provider: str,
        endpoint: str,
        duration: float,
        success: bool,
        cost: float = 0.0
    ):
        """Track external API call metrics"""
        api_calls_total.labels(
            provider=provider,
            endpoint=endpoint
        ).inc()
        
        api_call_duration.labels(
            provider=provider,
            endpoint=endpoint
        ).observe(duration)
        
        if not success:
            api_failures_total.labels(
                provider=provider,
                endpoint=endpoint,
                error_type="api_error"
            ).inc()
            
        if cost > 0:
            api_cost_total.labels(provider=provider).inc(cost)
            
    def track_cache_operation(
        self,
        cache_type: str,
        operation: str,
        hit: bool
    ):
        """Track cache operation metrics"""
        if hit:
            cache_hits_total.labels(
                cache_type=cache_type,
                operation=operation
            ).inc()
        else:
            cache_misses_total.labels(
                cache_type=cache_type,
                operation=operation
            ).inc()
            
    def track_ml_prediction(
        self,
        model: str,
        ticker: str,
        duration: float,
        accuracy: Optional[float] = None
    ):
        """Track ML prediction metrics"""
        ml_predictions_total.labels(
            model=model,
            ticker=ticker
        ).inc()
        
        ml_prediction_duration.labels(model=model).observe(duration)
        
        if accuracy is not None:
            ml_prediction_accuracy.labels(
                model=model,
                horizon="5d"  # Example
            ).set(accuracy)
            
    def track_analysis(
        self,
        analysis_type: str,
        duration: float,
        success: bool = True
    ):
        """Track analysis metrics"""
        status = "success" if success else "failure"
        
        analysis_completed_total.labels(
            analysis_type=analysis_type,
            status=status
        ).inc()
        
        analysis_duration.labels(
            analysis_type=analysis_type
        ).observe(duration)
        
    def update_cost_metrics(
        self,
        daily_costs: Dict[str, float],
        monthly_projection: float,
        budget: float = 50.0
    ):
        """Update cost tracking metrics"""
        for service, cost in daily_costs.items():
            daily_cost_gauge.labels(service=service).set(cost)
            
        monthly_cost_projection.set(monthly_projection)
        
        if budget > 0:
            usage_percent = (monthly_projection / budget) * 100
            cost_budget_usage_percent.set(usage_percent)
            
    def track_circuit_breaker(
        self,
        breaker_name: str,
        state: str,
        failure: bool = False
    ):
        """Track circuit breaker metrics"""
        state_map = {"closed": 0, "open": 1, "half_open": 2}
        circuit_breaker_state.labels(breaker_name=breaker_name).set(
            state_map.get(state, -1)
        )
        
        if failure:
            circuit_breaker_failures.labels(breaker_name=breaker_name).inc()
            
    def track_audit_log(
        self,
        action: str,
        user_type: str = "user"
    ):
        """Track audit log creation"""
        audit_logs_created.labels(
            action=action,
            user_type=user_type
        ).inc()
        
    def track_gdpr_request(self, request_type: str):
        """Track GDPR request"""
        gdpr_requests.labels(request_type=request_type).inc()
        
    def get_metrics(self) -> bytes:
        """Get metrics in Prometheus format"""
        return generate_latest(registry)
        
    def push_metrics(self, gateway_url: Optional[str] = None):
        """Push metrics to Prometheus gateway"""
        if gateway_url:
            push_to_gateway(
                gateway_url,
                job="investment_analysis",
                registry=registry
            )


# Decorator for timing functions
def timed_operation(
    metric: Histogram,
    labels_func: Optional[Callable[..., Dict[str, str]]] = None
):
    """Decorator to time function execution"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            labels = labels_func(*args, **kwargs) if labels_func else {}
            with metrics.timer(metric, labels):
                return await func(*args, **kwargs)
                
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            labels = labels_func(*args, **kwargs) if labels_func else {}
            with metrics.timer(metric, labels):
                return func(*args, **kwargs)
                
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator


# Global metrics collector
metrics = MetricsCollector()


# Health check endpoint data
class HealthStatus:
    """System health status"""
    
    def __init__(self):
        self.database = "unknown"
        self.redis = "unknown"
        self.kafka = "unknown"
        self.external_apis = {}
        self.ml_models = "unknown"
        self.last_check = None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "status": self.overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "database": self.database,
                "redis": self.redis,
                "kafka": self.kafka,
                "external_apis": self.external_apis,
                "ml_models": self.ml_models
            },
            "last_check": self.last_check.isoformat() if self.last_check else None
        }
        
    @property
    def overall_status(self) -> str:
        """Get overall system status"""
        statuses = [
            self.database,
            self.redis,
            self.kafka,
            self.ml_models
        ] + list(self.external_apis.values())
        
        if all(s == "healthy" for s in statuses):
            return "healthy"
        elif any(s == "unhealthy" for s in statuses):
            return "unhealthy"
        else:
            return "degraded"


# Global health status
health_status = HealthStatus()


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting Prometheus metrics"""
    
    async def dispatch(self, request: Request, call_next):
        """Process request and collect metrics"""
        start_time = time.time()
        
        try:
            # Process request
            response = await call_next(request)
            
            # Record response metrics
            duration = time.time() - start_time
            api_requests_total.labels(
                method=request.method,
                endpoint=request.url.path,
                status=str(response.status_code)
            ).inc()
            
            api_request_duration.labels(
                method=request.method,
                endpoint=request.url.path
            ).observe(duration)
            
            return response
            
        except Exception as e:
            # Record error
            duration = time.time() - start_time
            api_errors_total.labels(
                method=request.method,
                endpoint=request.url.path,
                error_type=type(e).__name__
            ).inc()
            
            api_request_duration.labels(
                method=request.method,
                endpoint=request.url.path
            ).observe(duration)
            
            raise


# Export metrics function
def export_metrics():
    """Get current metrics in Prometheus format"""
    return generate_latest(registry)