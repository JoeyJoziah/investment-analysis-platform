"""
API Performance Monitoring and Tracking
Advanced FastAPI middleware for comprehensive API monitoring.
"""

import time
import asyncio
import logging
import uuid
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json

from fastapi import Request, Response, HTTPException
from fastapi.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import httpx
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info
)

from backend.monitoring.metrics_collector import metrics_collector
from backend.config.monitoring_config import monitoring_config
from backend.utils.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)

# Enhanced API Metrics
api_response_size = Histogram(
    'api_response_size_bytes',
    'API response size in bytes',
    ['method', 'endpoint', 'status_class']
)

api_request_size = Histogram(
    'api_request_size_bytes', 
    'API request size in bytes',
    ['method', 'endpoint']
)

api_rate_limit_hits = Counter(
    'api_rate_limit_hits_total',
    'API rate limit violations',
    ['endpoint', 'user_id', 'ip']
)

api_dependency_duration = Histogram(
    'api_dependency_duration_seconds',
    'Duration of external API dependencies',
    ['dependency', 'endpoint']
)

api_cache_operations = Counter(
    'api_cache_operations_total',
    'API cache operations',
    ['operation', 'cache_type', 'endpoint']
)

api_user_sessions = Gauge(
    'api_active_user_sessions',
    'Active user sessions',
    ['endpoint_category']
)

api_business_metrics = Counter(
    'api_business_operations_total',
    'Business logic operations via API',
    ['operation_type', 'success']
)

# SLA Tracking
api_sla_violations = Counter(
    'api_sla_violations_total',
    'API SLA violations',
    ['sla_type', 'endpoint', 'severity']
)

api_availability = Gauge(
    'api_availability_ratio',
    'API availability ratio',
    ['endpoint', 'time_window']
)

# Security Metrics
api_security_events = Counter(
    'api_security_events_total',
    'API security events',
    ['event_type', 'severity', 'endpoint']
)

suspicious_requests = Counter(
    'api_suspicious_requests_total',
    'Suspicious API requests',
    ['pattern_type', 'endpoint', 'source_ip']
)

# Performance Percentiles
api_p50_latency = Gauge('api_p50_latency_seconds', 'API P50 latency', ['endpoint'])
api_p95_latency = Gauge('api_p95_latency_seconds', 'API P95 latency', ['endpoint'])
api_p99_latency = Gauge('api_p99_latency_seconds', 'API P99 latency', ['endpoint'])


class RequestTracker:
    """Track request metrics and performance."""
    
    def __init__(self):
        self.request_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.total_requests: Dict[str, int] = defaultdict(int)
        self.active_requests: Dict[str, int] = defaultdict(int)
    
    def record_request(self, endpoint: str, duration: float, status: int):
        """Record request metrics."""
        self.request_times[endpoint].append(duration)
        self.total_requests[endpoint] += 1
        
        if status >= 400:
            self.error_counts[endpoint] += 1
    
    def get_percentiles(self, endpoint: str) -> Dict[str, float]:
        """Calculate latency percentiles."""
        times = list(self.request_times[endpoint])
        if not times:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        
        times.sort()
        n = len(times)
        
        return {
            "p50": times[int(n * 0.5)] if n > 0 else 0.0,
            "p95": times[int(n * 0.95)] if n > 1 else 0.0,
            "p99": times[int(n * 0.99)] if n > 2 else 0.0
        }
    
    def get_error_rate(self, endpoint: str) -> float:
        """Calculate error rate for endpoint."""
        total = self.total_requests[endpoint]
        errors = self.error_counts[endpoint]
        return (errors / total * 100) if total > 0 else 0.0
    
    def start_request(self, endpoint: str):
        """Mark request as started."""
        self.active_requests[endpoint] += 1
    
    def end_request(self, endpoint: str):
        """Mark request as ended."""
        self.active_requests[endpoint] = max(0, self.active_requests[endpoint] - 1)


class APIPerformanceMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive API performance monitoring middleware.
    """
    
    def __init__(
        self,
        app: ASGIApp,
        tracker: Optional[RequestTracker] = None,
        enable_detailed_logging: bool = True,
        slow_request_threshold: float = 2.0,
        enable_business_metrics: bool = True
    ):
        super().__init__(app)
        self.tracker = tracker or RequestTracker()
        self.enable_detailed_logging = enable_detailed_logging
        self.slow_request_threshold = slow_request_threshold
        self.enable_business_metrics = enable_business_metrics
        
        # SLA thresholds
        self.sla_thresholds = {
            "latency_warning": 1.0,  # 1 second
            "latency_critical": 3.0,  # 3 seconds
            "error_rate_warning": 5.0,  # 5%
            "error_rate_critical": 10.0  # 10%
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Main middleware dispatch method."""
        # Generate correlation ID
        correlation_id = str(uuid.uuid4())
        request.state.correlation_id = correlation_id
        
        # Extract request metadata
        method = request.method
        endpoint = self._normalize_endpoint(str(request.url.path))
        user_id = await self._extract_user_id(request)
        client_ip = self._get_client_ip(request)
        
        # Security checks
        await self._security_checks(request, endpoint, client_ip)
        
        # Start tracking
        start_time = time.time()
        self.tracker.start_request(endpoint)
        
        # Request size
        request_size = await self._get_request_size(request)
        if request_size > 0:
            api_request_size.labels(
                method=method,
                endpoint=endpoint
            ).observe(request_size)
        
        # Process request
        response = None
        status_code = 500
        error_type = None
        
        try:
            # Call next middleware/handler
            response = await call_next(request)
            status_code = response.status_code
            
            # Response size
            response_size = self._get_response_size(response)
            if response_size > 0:
                status_class = f"{status_code // 100}xx"
                api_response_size.labels(
                    method=method,
                    endpoint=endpoint,
                    status_class=status_class
                ).observe(response_size)
            
        except HTTPException as e:
            status_code = e.status_code
            error_type = "http_exception"
            logger.warning(
                f"HTTP Exception in {endpoint}",
                extra={
                    "correlation_id": correlation_id,
                    "status_code": status_code,
                    "detail": str(e.detail)
                }
            )
            raise
        
        except Exception as e:
            status_code = 500
            error_type = "internal_error"
            logger.error(
                f"Internal error in {endpoint}",
                extra={
                    "correlation_id": correlation_id,
                    "error": str(e)
                },
                exc_info=True
            )
            raise
        
        finally:
            # Calculate duration
            duration = time.time() - start_time
            
            # Record metrics
            self._record_request_metrics(
                method, endpoint, status_code, duration, 
                user_id, client_ip, correlation_id
            )
            
            # Check SLA violations
            await self._check_sla_violations(endpoint, duration, status_code)
            
            # Business metrics
            if self.enable_business_metrics:
                await self._record_business_metrics(request, endpoint, status_code)
            
            # End tracking
            self.tracker.end_request(endpoint)
            
            # Detailed logging for slow requests
            if duration > self.slow_request_threshold and self.enable_detailed_logging:
                await self._log_slow_request(
                    request, endpoint, duration, status_code, correlation_id
                )
        
        return response
    
    def _normalize_endpoint(self, path: str) -> str:
        """Normalize endpoint path for consistent metrics."""
        # Replace IDs and UUIDs with placeholders
        import re
        
        # Replace numeric IDs
        path = re.sub(r'/\d+(?=/|$)', '/{id}', path)
        
        # Replace UUIDs
        uuid_pattern = r'/[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}(?=/|$)'
        path = re.sub(uuid_pattern, '/{uuid}', path, flags=re.IGNORECASE)
        
        # Replace ticker symbols (3-5 uppercase letters)
        ticker_pattern = r'/[A-Z]{2,5}(?=/|$)'
        path = re.sub(ticker_pattern, '/{ticker}', path)
        
        return path
    
    async def _extract_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID from request."""
        try:
            # Check JWT token
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                # In a real implementation, decode JWT to get user_id
                # For now, return a placeholder
                return "authenticated_user"
            
            # Check for API key
            api_key = request.headers.get("X-API-Key")
            if api_key:
                return f"api_key_user:{api_key[:8]}"
            
            return None
        except Exception:
            return None
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        forwarded = request.headers.get("X-Forwarded")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    async def _security_checks(self, request: Request, endpoint: str, client_ip: str):
        """Perform security checks on request."""
        try:
            # Rate limiting check (simplified)
            user_agent = request.headers.get("User-Agent", "")
            
            # Detect suspicious patterns
            if "bot" in user_agent.lower() and endpoint.startswith("/api/"):
                suspicious_requests.labels(
                    pattern_type="bot_traffic",
                    endpoint=endpoint,
                    source_ip=client_ip
                ).inc()
            
            # Check for SQL injection patterns in query parameters
            for param, value in request.query_params.items():
                if any(sql_keyword in str(value).lower() for sql_keyword in ['union', 'select', 'drop', 'insert']):
                    api_security_events.labels(
                        event_type="sql_injection_attempt",
                        severity="high",
                        endpoint=endpoint
                    ).inc()
                    logger.warning(
                        f"Potential SQL injection attempt",
                        extra={
                            "endpoint": endpoint,
                            "client_ip": client_ip,
                            "parameter": param,
                            "value": str(value)[:100]
                        }
                    )
        
        except Exception as e:
            logger.error(f"Error in security checks: {e}")
    
    async def _get_request_size(self, request: Request) -> int:
        """Get request body size."""
        try:
            content_length = request.headers.get("Content-Length")
            if content_length:
                return int(content_length)
            return 0
        except (ValueError, TypeError):
            return 0
    
    def _get_response_size(self, response: Response) -> int:
        """Get response size."""
        try:
            if hasattr(response, 'body') and response.body:
                return len(response.body)
            return 0
        except Exception:
            return 0
    
    def _record_request_metrics(
        self, 
        method: str, 
        endpoint: str, 
        status_code: int, 
        duration: float,
        user_id: Optional[str],
        client_ip: str,
        correlation_id: str
    ):
        """Record comprehensive request metrics."""
        # Basic metrics via metrics collector
        metrics_collector.record_api_request(method, endpoint, status_code, duration)
        
        # Enhanced tracking
        self.tracker.record_request(endpoint, duration, status_code)
        
        # Update percentile metrics
        percentiles = self.tracker.get_percentiles(endpoint)
        api_p50_latency.labels(endpoint=endpoint).set(percentiles["p50"])
        api_p95_latency.labels(endpoint=endpoint).set(percentiles["p95"])
        api_p99_latency.labels(endpoint=endpoint).set(percentiles["p99"])
        
        # Log request details
        logger.info(
            f"API request completed",
            extra={
                "correlation_id": correlation_id,
                "method": method,
                "endpoint": endpoint,
                "status_code": status_code,
                "duration_ms": duration * 1000,
                "user_id": user_id,
                "client_ip": client_ip
            }
        )
    
    async def _check_sla_violations(self, endpoint: str, duration: float, status_code: int):
        """Check for SLA violations and record them."""
        try:
            # Latency SLA checks
            if duration > self.sla_thresholds["latency_critical"]:
                api_sla_violations.labels(
                    sla_type="latency",
                    endpoint=endpoint,
                    severity="critical"
                ).inc()
            elif duration > self.sla_thresholds["latency_warning"]:
                api_sla_violations.labels(
                    sla_type="latency",
                    endpoint=endpoint,
                    severity="warning"
                ).inc()
            
            # Error rate SLA checks
            error_rate = self.tracker.get_error_rate(endpoint)
            
            if error_rate > self.sla_thresholds["error_rate_critical"]:
                api_sla_violations.labels(
                    sla_type="error_rate",
                    endpoint=endpoint,
                    severity="critical"
                ).inc()
            elif error_rate > self.sla_thresholds["error_rate_warning"]:
                api_sla_violations.labels(
                    sla_type="error_rate",
                    endpoint=endpoint,
                    severity="warning"
                ).inc()
        
        except Exception as e:
            logger.error(f"Error checking SLA violations: {e}")
    
    async def _record_business_metrics(self, request: Request, endpoint: str, status_code: int):
        """Record business-specific metrics based on endpoint."""
        try:
            success = "true" if status_code < 400 else "false"
            
            # Map endpoints to business operations
            if "/recommendations" in endpoint:
                api_business_metrics.labels(
                    operation_type="recommendation_generation",
                    success=success
                ).inc()
            elif "/portfolio" in endpoint:
                api_business_metrics.labels(
                    operation_type="portfolio_management",
                    success=success
                ).inc()
            elif "/stocks" in endpoint and request.method == "GET":
                api_business_metrics.labels(
                    operation_type="stock_data_retrieval",
                    success=success
                ).inc()
            elif "/analysis" in endpoint:
                api_business_metrics.labels(
                    operation_type="stock_analysis",
                    success=success
                ).inc()
        
        except Exception as e:
            logger.error(f"Error recording business metrics: {e}")
    
    async def _log_slow_request(
        self, 
        request: Request, 
        endpoint: str, 
        duration: float, 
        status_code: int,
        correlation_id: str
    ):
        """Log detailed information for slow requests."""
        try:
            request_details = {
                "correlation_id": correlation_id,
                "endpoint": endpoint,
                "method": request.method,
                "duration_ms": duration * 1000,
                "status_code": status_code,
                "query_params": dict(request.query_params),
                "headers": {k: v for k, v in request.headers.items() 
                          if k.lower() not in ['authorization', 'cookie']},
                "user_agent": request.headers.get("User-Agent", ""),
                "content_type": request.headers.get("Content-Type", ""),
            }
            
            logger.warning(
                f"Slow API request detected: {endpoint}",
                extra=request_details
            )
        
        except Exception as e:
            logger.error(f"Error logging slow request: {e}")


class APIHealthChecker:
    """Monitor API endpoint health and availability."""
    
    def __init__(self):
        self.endpoint_stats: Dict[str, Dict] = defaultdict(lambda: {
            "total_requests": 0,
            "successful_requests": 0,
            "last_success": None,
            "consecutive_failures": 0
        })
        self._monitoring_task: Optional[asyncio.Task] = None
    
    async def start_monitoring(self):
        """Start health monitoring."""
        if not self._monitoring_task:
            self._monitoring_task = asyncio.create_task(self._health_monitoring_loop())
            logger.info("Started API health monitoring")
    
    async def stop_monitoring(self):
        """Stop health monitoring."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped API health monitoring")
    
    async def _health_monitoring_loop(self):
        """Monitor API health continuously."""
        while True:
            try:
                await self._calculate_availability_metrics()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _calculate_availability_metrics(self):
        """Calculate availability metrics for endpoints."""
        try:
            current_time = datetime.now()
            
            for endpoint, stats in self.endpoint_stats.items():
                if stats["total_requests"] > 0:
                    # Calculate availability ratio
                    availability = (stats["successful_requests"] / stats["total_requests"]) * 100
                    
                    # Set metrics for different time windows
                    api_availability.labels(
                        endpoint=endpoint,
                        time_window="1h"
                    ).set(availability)
                    
                    # Check if endpoint is healthy
                    if stats["consecutive_failures"] > 5:
                        logger.warning(f"Endpoint {endpoint} has {stats['consecutive_failures']} consecutive failures")
        
        except Exception as e:
            logger.error(f"Error calculating availability metrics: {e}")
    
    def record_request(self, endpoint: str, success: bool):
        """Record request result for health tracking."""
        stats = self.endpoint_stats[endpoint]
        stats["total_requests"] += 1
        
        if success:
            stats["successful_requests"] += 1
            stats["last_success"] = datetime.now()
            stats["consecutive_failures"] = 0
        else:
            stats["consecutive_failures"] += 1


# Global instances
request_tracker = RequestTracker()
api_health_checker = APIHealthChecker()


# Setup function
def setup_api_monitoring(app):
    """Setup API performance monitoring."""
    # Add middleware
    app.add_middleware(
        APIPerformanceMiddleware,
        tracker=request_tracker,
        enable_detailed_logging=monitoring_config.logging.level == "DEBUG",
        slow_request_threshold=monitoring_config.alerting.thresholds["api_latency_warning"]
    )
    
    logger.info("API performance monitoring middleware added")
    return app


# Utility functions
async def get_api_metrics_summary() -> Dict[str, Any]:
    """Get summary of API performance metrics."""
    try:
        summary = {
            "endpoints": {},
            "overall": {
                "total_endpoints": len(request_tracker.request_times),
                "active_requests": sum(request_tracker.active_requests.values()),
                "timestamp": datetime.now().isoformat()
            }
        }
        
        for endpoint in request_tracker.request_times.keys():
            percentiles = request_tracker.get_percentiles(endpoint)
            error_rate = request_tracker.get_error_rate(endpoint)
            
            summary["endpoints"][endpoint] = {
                "total_requests": request_tracker.total_requests[endpoint],
                "error_rate_percent": error_rate,
                "active_requests": request_tracker.active_requests[endpoint],
                "latency_percentiles": percentiles
            }
        
        return summary
    
    except Exception as e:
        logger.error(f"Error generating API metrics summary: {e}")
        return {"error": str(e)}