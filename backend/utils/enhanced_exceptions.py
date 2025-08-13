"""
Enhanced exception handling with specific error types, recovery strategies,
and detailed error tracking for the integration layer.
"""

from typing import Any, Dict, List, Optional, Type
from datetime import datetime
from enum import Enum
import traceback
import sys


class ErrorSeverity(Enum):
    """Error severity levels for prioritization."""
    CRITICAL = "critical"  # System failure, immediate attention
    HIGH = "high"          # Service degraded, needs attention
    MEDIUM = "medium"      # Partial failure, can recover
    LOW = "low"           # Minor issue, self-healing
    INFO = "info"         # Informational only


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types."""
    RETRY = "retry"                    # Retry the operation
    FALLBACK = "fallback"              # Use fallback data/service
    CIRCUIT_BREAK = "circuit_break"    # Open circuit breaker
    CACHE = "cache"                    # Use cached data
    DEGRADE = "degrade"                # Provide degraded service
    FAIL = "fail"                      # Fail immediately
    QUEUE = "queue"                    # Queue for later processing


class BaseIntegrationException(Exception):
    """Base exception for integration layer."""
    
    severity = ErrorSeverity.MEDIUM
    recovery_strategy = RecoveryStrategy.FAIL
    retry_after = None  # Seconds to wait before retry
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.cause = cause
        self.timestamp = datetime.utcnow()
        self.traceback = self._capture_traceback()
    
    def _capture_traceback(self) -> str:
        """Capture current traceback."""
        return ''.join(traceback.format_tb(sys.exc_info()[2]))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/monitoring."""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'severity': self.severity.value,
            'recovery_strategy': self.recovery_strategy.value,
            'details': self.details,
            'timestamp': self.timestamp.isoformat(),
            'cause': str(self.cause) if self.cause else None,
            'traceback': self.traceback,
            'retry_after': self.retry_after
        }


# API-related exceptions
class APIException(BaseIntegrationException):
    """Base exception for API-related errors."""
    pass


class RateLimitException(APIException):
    """Rate limit exceeded for API."""
    
    severity = ErrorSeverity.MEDIUM
    recovery_strategy = RecoveryStrategy.FALLBACK
    
    def __init__(
        self,
        provider: str,
        limit: int,
        reset_time: Optional[datetime] = None,
        **kwargs
    ):
        message = f"Rate limit exceeded for {provider}: {limit} requests"
        details = {
            'provider': provider,
            'limit': limit,
            'reset_time': reset_time.isoformat() if reset_time else None
        }
        super().__init__(message, details, **kwargs)
        
        # Calculate retry_after
        if reset_time:
            self.retry_after = max(0, (reset_time - datetime.utcnow()).total_seconds())


class APITimeoutException(APIException):
    """API request timeout."""
    
    severity = ErrorSeverity.MEDIUM
    recovery_strategy = RecoveryStrategy.RETRY
    retry_after = 5  # Default retry after 5 seconds
    
    def __init__(self, provider: str, endpoint: str, timeout: float, **kwargs):
        message = f"API timeout for {provider}/{endpoint} after {timeout}s"
        details = {
            'provider': provider,
            'endpoint': endpoint,
            'timeout': timeout
        }
        super().__init__(message, details, **kwargs)


class APIAuthenticationException(APIException):
    """API authentication failed."""
    
    severity = ErrorSeverity.HIGH
    recovery_strategy = RecoveryStrategy.FAIL
    
    def __init__(self, provider: str, reason: str, **kwargs):
        message = f"Authentication failed for {provider}: {reason}"
        details = {
            'provider': provider,
            'reason': reason
        }
        super().__init__(message, details, **kwargs)


class APIDataException(APIException):
    """Invalid or corrupted API data."""
    
    severity = ErrorSeverity.MEDIUM
    recovery_strategy = RecoveryStrategy.CACHE
    
    def __init__(self, provider: str, data_type: str, reason: str, **kwargs):
        message = f"Invalid data from {provider} for {data_type}: {reason}"
        details = {
            'provider': provider,
            'data_type': data_type,
            'reason': reason
        }
        super().__init__(message, details, **kwargs)


class APIProviderException(APIException):
    """API provider service error."""
    
    severity = ErrorSeverity.HIGH
    recovery_strategy = RecoveryStrategy.CIRCUIT_BREAK
    
    def __init__(self, provider: str, status_code: int, response: str, **kwargs):
        message = f"Provider error from {provider}: HTTP {status_code}"
        details = {
            'provider': provider,
            'status_code': status_code,
            'response': response[:500]  # Limit response size
        }
        super().__init__(message, details, **kwargs)


# Cache-related exceptions
class CacheException(BaseIntegrationException):
    """Base exception for cache-related errors."""
    pass


class CacheConnectionException(CacheException):
    """Cache connection failed."""
    
    severity = ErrorSeverity.HIGH
    recovery_strategy = RecoveryStrategy.DEGRADE
    
    def __init__(self, cache_type: str, reason: str, **kwargs):
        message = f"Cache connection failed for {cache_type}: {reason}"
        details = {
            'cache_type': cache_type,
            'reason': reason
        }
        super().__init__(message, details, **kwargs)


class CacheKeyException(CacheException):
    """Invalid cache key."""
    
    severity = ErrorSeverity.LOW
    recovery_strategy = RecoveryStrategy.FAIL
    
    def __init__(self, key: str, reason: str, **kwargs):
        message = f"Invalid cache key '{key}': {reason}"
        details = {
            'key': key,
            'reason': reason
        }
        super().__init__(message, details, **kwargs)


class CacheSerializationException(CacheException):
    """Cache serialization/deserialization failed."""
    
    severity = ErrorSeverity.MEDIUM
    recovery_strategy = RecoveryStrategy.FAIL
    
    def __init__(self, operation: str, data_type: str, **kwargs):
        message = f"Cache {operation} failed for type {data_type}"
        details = {
            'operation': operation,
            'data_type': data_type
        }
        super().__init__(message, details, **kwargs)


# Database-related exceptions
class DatabaseException(BaseIntegrationException):
    """Base exception for database-related errors."""
    pass


class DatabaseConnectionException(DatabaseException):
    """Database connection failed."""
    
    severity = ErrorSeverity.CRITICAL
    recovery_strategy = RecoveryStrategy.CIRCUIT_BREAK
    
    def __init__(self, db_name: str, reason: str, **kwargs):
        message = f"Database connection failed for {db_name}: {reason}"
        details = {
            'database': db_name,
            'reason': reason
        }
        super().__init__(message, details, **kwargs)


class DatabaseQueryException(DatabaseException):
    """Database query failed."""
    
    severity = ErrorSeverity.HIGH
    recovery_strategy = RecoveryStrategy.RETRY
    retry_after = 2
    
    def __init__(self, query: str, reason: str, **kwargs):
        message = f"Database query failed: {reason}"
        details = {
            'query': query[:500],  # Limit query size
            'reason': reason
        }
        super().__init__(message, details, **kwargs)


class DatabaseIntegrityException(DatabaseException):
    """Database integrity constraint violation."""
    
    severity = ErrorSeverity.HIGH
    recovery_strategy = RecoveryStrategy.FAIL
    
    def __init__(self, constraint: str, table: str, **kwargs):
        message = f"Integrity constraint {constraint} violated on {table}"
        details = {
            'constraint': constraint,
            'table': table
        }
        super().__init__(message, details, **kwargs)


# Processing-related exceptions
class ProcessingException(BaseIntegrationException):
    """Base exception for data processing errors."""
    pass


class DataValidationException(ProcessingException):
    """Data validation failed."""
    
    severity = ErrorSeverity.MEDIUM
    recovery_strategy = RecoveryStrategy.FAIL
    
    def __init__(self, data_type: str, errors: List[str], **kwargs):
        message = f"Validation failed for {data_type}: {len(errors)} errors"
        details = {
            'data_type': data_type,
            'errors': errors[:10]  # Limit to first 10 errors
        }
        super().__init__(message, details, **kwargs)


class DataQualityException(ProcessingException):
    """Data quality check failed."""
    
    severity = ErrorSeverity.MEDIUM
    recovery_strategy = RecoveryStrategy.DEGRADE
    
    def __init__(self, symbol: str, quality_score: float, issues: List[Dict], **kwargs):
        message = f"Data quality check failed for {symbol}: score {quality_score}"
        details = {
            'symbol': symbol,
            'quality_score': quality_score,
            'issues': issues[:5]  # Limit to first 5 issues
        }
        super().__init__(message, details, **kwargs)


class ProcessingTimeoutException(ProcessingException):
    """Processing timeout exceeded."""
    
    severity = ErrorSeverity.MEDIUM
    recovery_strategy = RecoveryStrategy.QUEUE
    
    def __init__(self, operation: str, timeout: float, **kwargs):
        message = f"Processing timeout for {operation} after {timeout}s"
        details = {
            'operation': operation,
            'timeout': timeout
        }
        super().__init__(message, details, **kwargs)


# Cost-related exceptions
class CostException(BaseIntegrationException):
    """Base exception for cost-related errors."""
    pass


class BudgetExceededException(CostException):
    """Budget limit exceeded."""
    
    severity = ErrorSeverity.CRITICAL
    recovery_strategy = RecoveryStrategy.CACHE
    
    def __init__(self, current_cost: float, budget: float, **kwargs):
        message = f"Budget exceeded: ${current_cost:.2f} > ${budget:.2f}"
        details = {
            'current_cost': current_cost,
            'budget': budget,
            'overage': current_cost - budget
        }
        super().__init__(message, details, **kwargs)


class EmergencyModeException(CostException):
    """System in emergency mode due to cost."""
    
    severity = ErrorSeverity.HIGH
    recovery_strategy = RecoveryStrategy.CACHE
    
    def __init__(self, reason: str, **kwargs):
        message = f"Emergency mode activated: {reason}"
        details = {
            'reason': reason,
            'mode': 'emergency'
        }
        super().__init__(message, details, **kwargs)


# Exception handler with recovery
class ExceptionHandler:
    """
    Centralized exception handler with recovery strategies.
    """
    
    def __init__(self, logger=None):
        self.logger = logger
        self.error_counts = {}
        self.recovery_actions = {
            RecoveryStrategy.RETRY: self._retry_action,
            RecoveryStrategy.FALLBACK: self._fallback_action,
            RecoveryStrategy.CIRCUIT_BREAK: self._circuit_break_action,
            RecoveryStrategy.CACHE: self._cache_action,
            RecoveryStrategy.DEGRADE: self._degrade_action,
            RecoveryStrategy.FAIL: self._fail_action,
            RecoveryStrategy.QUEUE: self._queue_action
        }
    
    async def handle_exception(
        self,
        exception: BaseIntegrationException,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """
        Handle exception with appropriate recovery strategy.
        
        Args:
            exception: The exception to handle
            context: Additional context for recovery
            
        Returns:
            Recovery result or None
        """
        # Track error
        error_type = type(exception).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Log error
        if self.logger:
            self.logger.error(
                f"Handling {error_type}",
                exception=exception.to_dict(),
                context=context
            )
        
        # Execute recovery strategy
        recovery_strategy = exception.recovery_strategy
        if recovery_strategy in self.recovery_actions:
            return await self.recovery_actions[recovery_strategy](exception, context)
        
        return None
    
    async def _retry_action(
        self,
        exception: BaseIntegrationException,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """Implement retry logic."""
        if context and 'retry_func' in context:
            retry_func = context['retry_func']
            retry_count = context.get('retry_count', 0)
            max_retries = context.get('max_retries', 3)
            
            if retry_count < max_retries:
                if exception.retry_after:
                    import asyncio
                    await asyncio.sleep(exception.retry_after)
                
                context['retry_count'] = retry_count + 1
                return await retry_func()
        
        return None
    
    async def _fallback_action(
        self,
        exception: BaseIntegrationException,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """Implement fallback logic."""
        if context and 'fallback_func' in context:
            return await context['fallback_func']()
        
        return None
    
    async def _circuit_break_action(
        self,
        exception: BaseIntegrationException,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Implement circuit breaker logic."""
        if context and 'circuit_breaker' in context:
            circuit_breaker = context['circuit_breaker']
            circuit_breaker.record_failure()
        
        return None
    
    async def _cache_action(
        self,
        exception: BaseIntegrationException,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """Implement cache fallback logic."""
        if context and 'cache_key' in context:
            from backend.utils.cache import CacheManager
            cache = CacheManager()
            return await cache.get(context['cache_key'])
        
        return None
    
    async def _degrade_action(
        self,
        exception: BaseIntegrationException,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """Implement degraded service logic."""
        if context and 'degraded_response' in context:
            return context['degraded_response']
        
        return {'status': 'degraded', 'message': str(exception)}
    
    async def _fail_action(
        self,
        exception: BaseIntegrationException,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Implement immediate failure logic."""
        raise exception
    
    async def _queue_action(
        self,
        exception: BaseIntegrationException,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """Implement queue for later processing logic."""
        if context and 'queue_func' in context:
            return await context['queue_func'](exception, context)
        
        return {'status': 'queued', 'message': 'Operation queued for later processing'}
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        total_errors = sum(self.error_counts.values())
        
        return {
            'total_errors': total_errors,
            'error_counts': self.error_counts,
            'top_errors': sorted(
                self.error_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }


# Global exception handler instance
exception_handler = ExceptionHandler()