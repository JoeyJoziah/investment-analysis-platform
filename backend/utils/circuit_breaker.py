"""
Circuit Breaker Pattern Implementation
Prevents cascading failures in distributed systems by failing fast
"""

import asyncio
import time
from typing import Callable, Optional, Any, Dict, Union
from functools import wraps
from enum import Enum
from datetime import datetime, timedelta
import logging
from collections import deque
import threading

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open"""
    pass


class CircuitBreaker:
    """
    Circuit breaker implementation with configurable thresholds
    
    The circuit breaker pattern prevents cascading failures by:
    1. Monitoring failure rates
    2. Opening the circuit when threshold exceeded
    3. Failing fast without calling the protected function
    4. Periodically testing if the service has recovered
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception,
        success_threshold: int = 2,
        name: Optional[str] = None,
        on_open: Optional[Callable] = None,
        on_close: Optional[Callable] = None
    ):
        """
        Initialize circuit breaker
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            expected_exception: Exception type to catch
            success_threshold: Successes needed to close circuit
            name: Circuit breaker name for logging
            on_open: Callback when circuit opens
            on_close: Callback when circuit closes
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.success_threshold = success_threshold
        self.name = name or "CircuitBreaker"
        self.on_open = on_open
        self.on_close = on_close
        
        # State management
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._last_attempt_time = None
        
        # Metrics
        self._call_count = 0
        self._failure_history = deque(maxlen=100)
        self._response_times = deque(maxlen=100)
        
        # Thread safety
        self._lock = threading.Lock()
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state"""
        with self._lock:
            self._update_state()
            return self._state
    
    @property
    def is_open(self) -> bool:
        """Check if circuit is open"""
        return self.state == CircuitState.OPEN
    
    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed"""
        return self.state == CircuitState.CLOSED
    
    def _update_state(self):
        """Update circuit state based on current conditions"""
        if self._state == CircuitState.OPEN:
            if self._last_failure_time and \
               time.time() - self._last_failure_time > self.recovery_timeout:
                logger.info(f"{self.name}: Attempting recovery, moving to HALF_OPEN")
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
    
    def _record_success(self):
        """Record successful call"""
        with self._lock:
            self._failure_count = 0
            
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._state = CircuitState.CLOSED
                    logger.info(f"{self.name}: Circuit closed after recovery")
                    if self.on_close:
                        self.on_close()
    
    def _record_failure(self):
        """Record failed call"""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            self._failure_history.append(datetime.now())
            
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning(f"{self.name}: Recovery failed, circuit reopened")
            elif self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.error(f"{self.name}: Circuit opened after {self._failure_count} failures")
                if self.on_open:
                    self.on_open()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call function through circuit breaker
        
        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: If circuit is open
        """
        if self.state == CircuitState.OPEN:
            raise CircuitBreakerError(f"{self.name}: Circuit is open")
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            self._record_success()
            self._response_times.append(time.time() - start_time)
            return result
        except self.expected_exception as e:
            self._record_failure()
            raise e
    
    async def async_call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call async function through circuit breaker
        
        Args:
            func: Async function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: If circuit is open
        """
        if self.state == CircuitState.OPEN:
            raise CircuitBreakerError(f"{self.name}: Circuit is open")
        
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            self._record_success()
            self._response_times.append(time.time() - start_time)
            return result
        except self.expected_exception as e:
            self._record_failure()
            raise e
    
    def __call__(self, func: Callable) -> Callable:
        """
        Decorator usage of circuit breaker
        
        Example:
            @CircuitBreaker(failure_threshold=3)
            def risky_operation():
                # code that might fail
                pass
        """
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self.async_call(func, *args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return self.call(func, *args, **kwargs)
            return sync_wrapper
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics"""
        with self._lock:
            recent_failures = len([
                f for f in self._failure_history
                if (datetime.now() - f).seconds < 300
            ])
            
            avg_response_time = (
                sum(self._response_times) / len(self._response_times)
                if self._response_times else 0
            )
            
            return {
                'name': self.name,
                'state': self._state.value,
                'failure_count': self._failure_count,
                'success_count': self._success_count,
                'recent_failures_5min': recent_failures,
                'avg_response_time_ms': avg_response_time * 1000,
                'last_failure': self._last_failure_time
            }
    
    def reset(self):
        """Manually reset circuit breaker"""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            logger.info(f"{self.name}: Circuit manually reset")


class APICircuitBreaker(CircuitBreaker):
    """
    Specialized circuit breaker for API calls with additional features
    """
    
    def __init__(
        self,
        api_name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        rate_limit_threshold: int = 3,
        timeout_threshold: int = 3
    ):
        """
        Initialize API-specific circuit breaker
        
        Args:
            api_name: Name of the API
            failure_threshold: Total failures before opening
            recovery_timeout: Recovery timeout in seconds
            rate_limit_threshold: Rate limit errors before opening
            timeout_threshold: Timeout errors before opening
        """
        super().__init__(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            name=f"API_{api_name}"
        )
        
        self.api_name = api_name
        self.rate_limit_threshold = rate_limit_threshold
        self.timeout_threshold = timeout_threshold
        
        # Specific error counters
        self._rate_limit_count = 0
        self._timeout_count = 0
        
    def record_rate_limit(self):
        """Record rate limit error"""
        with self._lock:
            self._rate_limit_count += 1
            if self._rate_limit_count >= self.rate_limit_threshold:
                self._state = CircuitState.OPEN
                self._last_failure_time = time.time()
                logger.warning(f"{self.api_name}: Circuit opened due to rate limiting")
    
    def record_timeout(self):
        """Record timeout error"""
        with self._lock:
            self._timeout_count += 1
            if self._timeout_count >= self.timeout_threshold:
                self._state = CircuitState.OPEN
                self._last_failure_time = time.time()
                logger.warning(f"{self.api_name}: Circuit opened due to timeouts")
    
    def _record_success(self):
        """Override to reset specific counters"""
        super()._record_success()
        with self._lock:
            self._rate_limit_count = 0
            self._timeout_count = 0


class CircuitBreakerManager:
    """
    Manages multiple circuit breakers for different services
    """
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()
    
    def register(self, name: str, breaker: CircuitBreaker):
        """Register a circuit breaker"""
        with self._lock:
            self._breakers[name] = breaker
            logger.info(f"Registered circuit breaker: {name}")
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name"""
        return self._breakers.get(name)
    
    def get_or_create(
        self,
        name: str,
        **kwargs
    ) -> CircuitBreaker:
        """Get existing or create new circuit breaker"""
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name=name, **kwargs)
            return self._breakers[name]
    
    def get_all_metrics(self) -> Dict[str, Dict]:
        """Get metrics for all circuit breakers"""
        return {
            name: breaker.get_metrics()
            for name, breaker in self._breakers.items()
        }
    
    def reset_all(self):
        """Reset all circuit breakers"""
        for breaker in self._breakers.values():
            breaker.reset()
    
    def get_open_circuits(self) -> List[str]:
        """Get list of open circuits"""
        return [
            name for name, breaker in self._breakers.items()
            if breaker.is_open
        ]


# Global circuit breaker manager
circuit_manager = CircuitBreakerManager()


# Pre-configured circuit breakers for different APIs
def get_api_circuit_breaker(api_name: str) -> APICircuitBreaker:
    """Get or create API-specific circuit breaker"""
    
    # API-specific configurations
    configs = {
        'finnhub': {
            'failure_threshold': 5,
            'recovery_timeout': 30,
            'rate_limit_threshold': 3
        },
        'alpha_vantage': {
            'failure_threshold': 3,
            'recovery_timeout': 60,
            'rate_limit_threshold': 2
        },
        'polygon': {
            'failure_threshold': 4,
            'recovery_timeout': 45,
            'rate_limit_threshold': 2
        },
        'yahoo_finance': {
            'failure_threshold': 10,
            'recovery_timeout': 20,
            'rate_limit_threshold': 5
        }
    }
    
    config = configs.get(api_name.lower(), {
        'failure_threshold': 5,
        'recovery_timeout': 60,
        'rate_limit_threshold': 3
    })
    
    breaker = circuit_manager.get(api_name)
    if not breaker:
        breaker = APICircuitBreaker(api_name, **config)
        circuit_manager.register(api_name, breaker)
    
    return breaker


# Decorator for easy usage
def with_circuit_breaker(
    name: Optional[str] = None,
    failure_threshold: int = 5,
    recovery_timeout: int = 60
):
    """
    Decorator to apply circuit breaker to function
    
    Example:
        @with_circuit_breaker(name="external_api", failure_threshold=3)
        async def call_external_api():
            # risky API call
            pass
    """
    def decorator(func):
        breaker_name = name or func.__name__
        breaker = circuit_manager.get_or_create(
            breaker_name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout
        )
        return breaker(func)
    
    return decorator