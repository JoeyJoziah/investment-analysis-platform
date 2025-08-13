"""
Deadlock Detection and Retry Logic
Comprehensive deadlock handling for concurrent database operations with exponential backoff and circuit breaker patterns.
"""

import asyncio
import logging
from typing import Callable, Any, Optional, Dict, List, TypeVar, Awaitable
from functools import wraps
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import random
import asyncpg
from sqlalchemy.exc import DisconnectionError, TimeoutError, OperationalError

logger = logging.getLogger(__name__)

T = TypeVar('T')


class DeadlockErrorType(Enum):
    """Types of deadlock-related errors"""
    SERIALIZATION_FAILURE = "serialization_failure"
    DEADLOCK_DETECTED = "deadlock_detected" 
    CONNECTION_TIMEOUT = "connection_timeout"
    LOCK_TIMEOUT = "lock_timeout"
    UNKNOWN = "unknown"


@dataclass
class RetryMetrics:
    """Metrics for retry operations"""
    total_attempts: int = 0
    successful_retries: int = 0
    failed_retries: int = 0
    deadlock_errors: int = 0
    timeout_errors: int = 0
    last_error: Optional[str] = None
    last_retry_at: Optional[datetime] = None


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests  
    HALF_OPEN = "half_open" # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    success_threshold: int = 3  # Successes needed to close from half-open


class CircuitBreaker:
    """Circuit breaker for database operations"""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.metrics = RetryMetrics()
    
    def should_allow_request(self) -> bool:
        """Check if request should be allowed"""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        
        if self.state == CircuitBreakerState.OPEN:
            # Check if recovery timeout has passed
            if (self.last_failure_time and 
                datetime.utcnow() - self.last_failure_time >= timedelta(seconds=self.config.recovery_timeout)):
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                logger.info("Circuit breaker transitioning to HALF_OPEN")
                return True
            return False
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            return True
        
        return False
    
    def record_success(self):
        """Record successful operation"""
        self.failure_count = 0
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                logger.info("Circuit breaker CLOSED after successful recovery")
        
        self.metrics.successful_retries += 1
    
    def record_failure(self, error: Exception):
        """Record failed operation"""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        self.metrics.failed_retries += 1
        self.metrics.last_error = str(error)
        self.metrics.last_retry_at = datetime.utcnow()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            logger.warning("Circuit breaker OPEN after failure in HALF_OPEN")
        elif self.state == CircuitBreakerState.CLOSED and self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker OPEN after {self.failure_count} failures")


class DeadlockHandler:
    """
    Comprehensive deadlock detection and retry handler with circuit breaker pattern.
    """
    
    def __init__(
        self,
        max_retries: int = 5,
        base_delay: float = 0.1,
        max_delay: float = 5.0,
        exponential_factor: float = 2.0,
        jitter: bool = True
    ):
        """
        Initialize deadlock handler.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
            exponential_factor: Exponential backoff factor
            jitter: Whether to add random jitter to delays
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_factor = exponential_factor
        self.jitter = jitter
        
        # Circuit breakers for different operation types
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Global metrics
        self.global_metrics = RetryMetrics()
    
    def _get_circuit_breaker(self, operation_type: str) -> CircuitBreaker:
        """Get or create circuit breaker for operation type"""
        if operation_type not in self.circuit_breakers:
            self.circuit_breakers[operation_type] = CircuitBreaker(
                CircuitBreakerConfig()
            )
        return self.circuit_breakers[operation_type]
    
    def _classify_error(self, error: Exception) -> DeadlockErrorType:
        """Classify error type for appropriate handling"""
        if isinstance(error, asyncpg.exceptions.SerializationFailureError):
            return DeadlockErrorType.SERIALIZATION_FAILURE
        elif isinstance(error, asyncpg.exceptions.DeadlockDetectedError):
            return DeadlockErrorType.DEADLOCK_DETECTED
        elif isinstance(error, (TimeoutError, asyncio.TimeoutError)):
            return DeadlockErrorType.CONNECTION_TIMEOUT
        elif isinstance(error, OperationalError):
            if "lock" in str(error).lower():
                return DeadlockErrorType.LOCK_TIMEOUT
            elif "timeout" in str(error).lower():
                return DeadlockErrorType.CONNECTION_TIMEOUT
        
        return DeadlockErrorType.UNKNOWN
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if error is retryable"""
        error_type = self._classify_error(error)
        
        # These error types are generally retryable
        retryable_types = {
            DeadlockErrorType.SERIALIZATION_FAILURE,
            DeadlockErrorType.DEADLOCK_DETECTED,
            DeadlockErrorType.CONNECTION_TIMEOUT,
            DeadlockErrorType.LOCK_TIMEOUT
        }
        
        return error_type in retryable_types
    
    def _calculate_delay(self, attempt: int, error_type: DeadlockErrorType) -> float:
        """Calculate delay for retry attempt"""
        # Base delay with exponential backoff
        delay = self.base_delay * (self.exponential_factor ** attempt)
        
        # Apply error-type specific adjustments
        if error_type == DeadlockErrorType.SERIALIZATION_FAILURE:
            delay *= 0.5  # Shorter delay for serialization failures
        elif error_type == DeadlockErrorType.DEADLOCK_DETECTED:
            delay *= 1.5  # Longer delay for deadlocks
        
        # Cap at max delay
        delay = min(delay, self.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.jitter:
            delay += random.uniform(0, delay * 0.1)
        
        return delay
    
    async def execute_with_retry(
        self,
        operation: Callable[..., Awaitable[T]],
        *args,
        operation_type: str = "default",
        **kwargs
    ) -> T:
        """
        Execute operation with comprehensive retry logic.
        
        Args:
            operation: Async function to execute
            operation_type: Type of operation for circuit breaker
            *args, **kwargs: Arguments to pass to operation
        
        Returns:
            Result of successful operation
        
        Raises:
            Exception: If all retries are exhausted
        """
        circuit_breaker = self._get_circuit_breaker(operation_type)
        last_exception = None
        
        # Check circuit breaker
        if not circuit_breaker.should_allow_request():
            raise RuntimeError(
                f"Circuit breaker OPEN for operation type '{operation_type}'. "
                f"Too many failures detected."
            )
        
        for attempt in range(self.max_retries + 1):
            try:
                self.global_metrics.total_attempts += 1
                circuit_breaker.metrics.total_attempts += 1
                
                # Execute the operation
                result = await operation(*args, **kwargs)
                
                # Success - record in circuit breaker
                circuit_breaker.record_success()
                
                if attempt > 0:
                    logger.info(
                        f"Operation '{operation_type}' succeeded after {attempt} retries"
                    )
                
                return result
                
            except Exception as e:
                last_exception = e
                error_type = self._classify_error(e)
                
                # Update metrics
                if error_type == DeadlockErrorType.DEADLOCK_DETECTED:
                    self.global_metrics.deadlock_errors += 1
                    circuit_breaker.metrics.deadlock_errors += 1
                elif error_type in (DeadlockErrorType.CONNECTION_TIMEOUT, DeadlockErrorType.LOCK_TIMEOUT):
                    self.global_metrics.timeout_errors += 1
                    circuit_breaker.metrics.timeout_errors += 1
                
                # Check if error is retryable
                if not self._is_retryable_error(e):
                    circuit_breaker.record_failure(e)
                    logger.error(
                        f"Non-retryable error in operation '{operation_type}': {e}"
                    )
                    raise
                
                # Check if we have retries left
                if attempt >= self.max_retries:
                    circuit_breaker.record_failure(e)
                    logger.error(
                        f"All {self.max_retries + 1} attempts failed for operation '{operation_type}'. "
                        f"Last error: {e}"
                    )
                    break
                
                # Calculate delay and wait
                delay = self._calculate_delay(attempt, error_type)
                
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries + 1} failed for operation '{operation_type}' "
                    f"with {error_type.value}: {e}. Retrying in {delay:.2f}s"
                )
                
                await asyncio.sleep(delay)
        
        # All retries exhausted
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError(f"Operation '{operation_type}' failed after all retries")
    
    def get_metrics(self, operation_type: Optional[str] = None) -> Dict[str, Any]:
        """Get retry metrics"""
        if operation_type:
            if operation_type in self.circuit_breakers:
                cb = self.circuit_breakers[operation_type]
                return {
                    'operation_type': operation_type,
                    'circuit_breaker_state': cb.state.value,
                    'failure_count': cb.failure_count,
                    'success_count': cb.success_count,
                    'metrics': {
                        'total_attempts': cb.metrics.total_attempts,
                        'successful_retries': cb.metrics.successful_retries,
                        'failed_retries': cb.metrics.failed_retries,
                        'deadlock_errors': cb.metrics.deadlock_errors,
                        'timeout_errors': cb.metrics.timeout_errors,
                        'last_error': cb.metrics.last_error,
                        'last_retry_at': cb.metrics.last_retry_at.isoformat() if cb.metrics.last_retry_at else None
                    }
                }
            else:
                return {'operation_type': operation_type, 'status': 'no_data'}
        else:
            return {
                'global_metrics': {
                    'total_attempts': self.global_metrics.total_attempts,
                    'successful_retries': self.global_metrics.successful_retries,
                    'failed_retries': self.global_metrics.failed_retries,
                    'deadlock_errors': self.global_metrics.deadlock_errors,
                    'timeout_errors': self.global_metrics.timeout_errors,
                },
                'circuit_breakers': {
                    op_type: {
                        'state': cb.state.value,
                        'failure_count': cb.failure_count,
                        'success_count': cb.success_count
                    }
                    for op_type, cb in self.circuit_breakers.items()
                }
            }
    
    def reset_circuit_breaker(self, operation_type: str):
        """Reset circuit breaker for operation type"""
        if operation_type in self.circuit_breakers:
            cb = self.circuit_breakers[operation_type]
            cb.state = CircuitBreakerState.CLOSED
            cb.failure_count = 0
            cb.success_count = 0
            cb.last_failure_time = None
            logger.info(f"Reset circuit breaker for operation type '{operation_type}'")


# Global deadlock handler instance
deadlock_handler = DeadlockHandler()


# Decorator for automatic retry handling
def with_deadlock_retry(
    max_retries: int = 3,
    operation_type: str = "default"
):
    """
    Decorator to add automatic deadlock retry to async functions.
    
    Args:
        max_retries: Maximum number of retry attempts
        operation_type: Type of operation for circuit breaker
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            # Create a handler with specific retry count for this function
            handler = DeadlockHandler(max_retries=max_retries)
            
            return await handler.execute_with_retry(
                func,
                *args,
                operation_type=operation_type,
                **kwargs
            )
        
        return wrapper
    return decorator


# Convenience functions for common operations
async def execute_database_operation_with_retry(
    operation: Callable[..., Awaitable[T]],
    *args,
    **kwargs
) -> T:
    """Execute database operation with standard retry logic"""
    return await deadlock_handler.execute_with_retry(
        operation,
        *args,
        operation_type="database",
        **kwargs
    )


async def execute_portfolio_operation_with_retry(
    operation: Callable[..., Awaitable[T]],
    *args,
    **kwargs
) -> T:
    """Execute portfolio operation with retry logic"""
    return await deadlock_handler.execute_with_retry(
        operation,
        *args,
        operation_type="portfolio",
        **kwargs
    )


async def execute_price_update_with_retry(
    operation: Callable[..., Awaitable[T]],
    *args,
    **kwargs
) -> T:
    """Execute price update operation with retry logic"""
    return await deadlock_handler.execute_with_retry(
        operation,
        *args,
        operation_type="price_update",
        **kwargs
    )


# Health check function
async def get_deadlock_handler_status() -> Dict[str, Any]:
    """Get comprehensive status of deadlock handler"""
    return {
        'handler_config': {
            'max_retries': deadlock_handler.max_retries,
            'base_delay': deadlock_handler.base_delay,
            'max_delay': deadlock_handler.max_delay,
            'exponential_factor': deadlock_handler.exponential_factor
        },
        'metrics': deadlock_handler.get_metrics(),
        'circuit_breakers_count': len(deadlock_handler.circuit_breakers),
        'active_circuit_breakers': [
            op_type for op_type, cb in deadlock_handler.circuit_breakers.items()
            if cb.state != CircuitBreakerState.CLOSED
        ]
    }