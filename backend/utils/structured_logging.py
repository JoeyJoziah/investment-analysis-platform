"""
Structured logging with correlation IDs for distributed tracing.
Provides JSON-formatted logs with consistent fields and request tracking.
"""

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Dict, Optional

import structlog
# from pythonjsonlogger import jsonlogger  # TODO: Install python-json-logger

# Context variable for correlation ID
correlation_id_var: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar('user_id', default=None)


class CorrelationIDProcessor:
    """Add correlation ID to all log entries."""
    
    def __call__(self, logger, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Add correlation ID and other context to log entry."""
        correlation_id = correlation_id_var.get()
        if correlation_id:
            event_dict['correlation_id'] = correlation_id
            
        request_id = request_id_var.get()
        if request_id:
            event_dict['request_id'] = request_id
            
        user_id = user_id_var.get()
        if user_id:
            event_dict['user_id'] = user_id
            
        return event_dict


class ServiceContextProcessor:
    """Add service context to all log entries."""
    
    def __init__(self, service_name: str, environment: str, version: str):
        self.service_name = service_name
        self.environment = environment
        self.version = version
    
    def __call__(self, logger, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Add service context to log entry."""
        event_dict.update({
            'service': self.service_name,
            'environment': self.environment,
            'version': self.version,
            'timestamp': datetime.utcnow().isoformat(),
        })
        return event_dict


class PerformanceProcessor:
    """Add performance metrics to log entries."""
    
    def __call__(self, logger, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Add performance context if available."""
        if 'duration_ms' in event_dict:
            # Categorize performance
            duration = event_dict['duration_ms']
            if duration < 100:
                event_dict['performance'] = 'fast'
            elif duration < 500:
                event_dict['performance'] = 'normal'
            elif duration < 1000:
                event_dict['performance'] = 'slow'
            else:
                event_dict['performance'] = 'very_slow'
        
        return event_dict


class ErrorDetailsProcessor:
    """Enhanced error details in log entries."""
    
    def __call__(self, logger, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Add detailed error information."""
        if 'exception' in event_dict:
            exc = event_dict['exception']
            if isinstance(exc, Exception):
                event_dict['error'] = {
                    'type': type(exc).__name__,
                    'message': str(exc),
                    'module': type(exc).__module__,
                }
                # Add specific fields for known exceptions
                if hasattr(exc, 'status_code'):
                    event_dict['error']['status_code'] = exc.status_code
                if hasattr(exc, 'detail'):
                    event_dict['error']['detail'] = exc.detail
        
        return event_dict


def configure_structured_logging(
    service_name: str = "investment_analysis",
    environment: str = "production",
    version: str = "1.0.0",
    log_level: str = "INFO",
    log_file: Optional[str] = None
) -> None:
    """
    Configure structured logging for the application.
    
    Args:
        service_name: Name of the service
        environment: Environment (development, staging, production)
        version: Application version
        log_level: Logging level
        log_file: Optional log file path
    """
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            CorrelationIDProcessor(),
            ServiceContextProcessor(service_name, environment, version),
            PerformanceProcessor(),
            ErrorDetailsProcessor(),
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer() if environment == "development" 
            else structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler with JSON formatter
    console_handler = logging.StreamHandler(sys.stdout)
    if environment == "development":
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    else:
        # Use standard formatter until python-json-logger is installed
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        # Use standard formatter until python-json-logger is installed
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)


class StructuredLogger:
    """
    Wrapper for structured logging with correlation IDs.
    """
    
    def __init__(self, name: str):
        self.logger = structlog.get_logger(name)
    
    def with_correlation_id(self, correlation_id: str) -> 'StructuredLogger':
        """Set correlation ID for this logger instance."""
        correlation_id_var.set(correlation_id)
        return self
    
    def with_request_id(self, request_id: str) -> 'StructuredLogger':
        """Set request ID for this logger instance."""
        request_id_var.set(request_id)
        return self
    
    def with_user_id(self, user_id: str) -> 'StructuredLogger':
        """Set user ID for this logger instance."""
        user_id_var.set(user_id)
        return self
    
    def debug(self, msg: str, **kwargs):
        """Log debug message."""
        self.logger.debug(msg, **kwargs)
    
    def info(self, msg: str, **kwargs):
        """Log info message."""
        self.logger.info(msg, **kwargs)
    
    def warning(self, msg: str, **kwargs):
        """Log warning message."""
        self.logger.warning(msg, **kwargs)
    
    def error(self, msg: str, **kwargs):
        """Log error message."""
        self.logger.error(msg, **kwargs)
    
    def critical(self, msg: str, **kwargs):
        """Log critical message."""
        self.logger.critical(msg, **kwargs)
    
    def log_api_call(
        self,
        provider: str,
        endpoint: str,
        duration_ms: float,
        status_code: Optional[int] = None,
        success: bool = True,
        **kwargs
    ):
        """Log external API call with structured data."""
        self.info(
            "external_api_call",
            provider=provider,
            endpoint=endpoint,
            duration_ms=duration_ms,
            status_code=status_code,
            success=success,
            **kwargs
        )
    
    def log_database_query(
        self,
        operation: str,
        table: str,
        duration_ms: float,
        rows_affected: Optional[int] = None,
        **kwargs
    ):
        """Log database query with structured data."""
        self.info(
            "database_query",
            operation=operation,
            table=table,
            duration_ms=duration_ms,
            rows_affected=rows_affected,
            **kwargs
        )
    
    def log_cache_operation(
        self,
        operation: str,
        key: str,
        hit: bool,
        duration_ms: float,
        **kwargs
    ):
        """Log cache operation with structured data."""
        self.debug(
            "cache_operation",
            operation=operation,
            key=key,
            cache_hit=hit,
            duration_ms=duration_ms,
            **kwargs
        )
    
    def log_analysis(
        self,
        analysis_type: str,
        symbol: str,
        duration_ms: float,
        result: Optional[str] = None,
        **kwargs
    ):
        """Log analysis operation with structured data."""
        self.info(
            "analysis_completed",
            analysis_type=analysis_type,
            symbol=symbol,
            duration_ms=duration_ms,
            result=result,
            **kwargs
        )


def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return str(uuid.uuid4())


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID from context."""
    return correlation_id_var.get()


def set_correlation_id(correlation_id: str) -> None:
    """Set correlation ID in context."""
    correlation_id_var.set(correlation_id)


# Middleware for FastAPI to add correlation IDs
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """Middleware to add correlation IDs to all requests."""
    
    async def dispatch(self, request: Request, call_next):
        """Process request and add correlation ID."""
        # Get or generate correlation ID
        correlation_id = request.headers.get('X-Correlation-ID', generate_correlation_id())
        request_id = request.headers.get('X-Request-ID', generate_correlation_id())
        
        # Set in context
        correlation_id_var.set(correlation_id)
        request_id_var.set(request_id)
        
        # Get user ID from request if authenticated
        user_id = getattr(request.state, 'user_id', None)
        if user_id:
            user_id_var.set(user_id)
        
        # Process request
        response = await call_next(request)
        
        # Add correlation ID to response headers
        response.headers['X-Correlation-ID'] = correlation_id
        response.headers['X-Request-ID'] = request_id
        
        return response


# Example usage with context manager
from contextlib import contextmanager
import time


@contextmanager
def log_operation(logger: StructuredLogger, operation: str, **context):
    """Context manager for logging operations with timing."""
    start_time = time.time()
    logger.info(f"{operation}_started", **context)
    
    try:
        yield logger
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error(
            f"{operation}_failed",
            duration_ms=duration_ms,
            exception=e,
            **context
        )
        raise
    else:
        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            f"{operation}_completed",
            duration_ms=duration_ms,
            **context
        )


# Create default logger instance
default_logger = StructuredLogger(__name__)


def get_structured_logger(name: str) -> StructuredLogger:
    """
    Get or create a structured logger for the given name.

    This is a convenience function that creates a StructuredLogger instance
    for the given module name, similar to logging.getLogger().

    Args:
        name: The name for the logger, typically __name__

    Returns:
        A StructuredLogger instance
    """
    return StructuredLogger(name)