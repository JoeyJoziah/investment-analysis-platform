"""
Redis Resilience Layer with Circuit Breaker and Sentinel Support
Provides high availability and fault tolerance for Redis operations.
"""

import asyncio
import logging
import time
from typing import Optional, Any, Dict, List, Callable, Union
from enum import Enum
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import json
import random

import redis.asyncio as aioredis
from redis.sentinel import Sentinel
from redis.exceptions import RedisError, ConnectionError as RedisConnectionError
import redis

from pybreaker import CircuitBreaker, CircuitBreakerError

from backend.utils.bounded_cache import BoundedFallbackCache

logger = logging.getLogger(__name__)


class RedisMode(Enum):
    """Redis deployment modes."""
    STANDALONE = "standalone"
    SENTINEL = "sentinel"
    CLUSTER = "cluster"


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class RedisCircuitBreaker:
    """
    Circuit breaker for Redis operations.
    Prevents cascading failures and provides fallback mechanisms.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = RedisError,
        name: str = "redis"
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            expected_exception: Exception type to catch
            name: Circuit breaker name
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name
        
        # Circuit breaker from pybreaker
        self._breaker = CircuitBreaker(
            fail_max=failure_threshold,
            reset_timeout=recovery_timeout,
            exclude=[KeyError, ValueError],  # Don't trip on logical errors
            name=name
        )
        
        # Metrics
        self._metrics = {
            'calls': 0,
            'failures': 0,
            'successes': 0,
            'fallbacks': 0,
            'state_changes': []
        }
        
        # Bounded fallback cache for when Redis is down
        self._fallback_cache = BoundedFallbackCache(max_size=500, default_ttl=600)
    
    async def call(
        self,
        func: Callable,
        *args,
        fallback: Optional[Callable] = None,
        cache_key: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Async function to execute
            fallback: Fallback function if circuit is open
            cache_key: Key for fallback cache
            *args, **kwargs: Arguments for func
        
        Returns:
            Result from func or fallback
        """
        self._metrics['calls'] += 1
        
        try:
            # Try to execute through circuit breaker
            if asyncio.iscoroutinefunction(func):
                result = await self._breaker(func)(*args, **kwargs)
            else:
                result = self._breaker(func)(*args, **kwargs)
            
            self._metrics['successes'] += 1
            
            # Cache successful result for fallback
            if cache_key:
                self._update_fallback_cache(cache_key, result)
            
            return result
            
        except CircuitBreakerError:
            # Circuit is open
            logger.warning(f"Circuit breaker {self.name} is OPEN")
            self._metrics['failures'] += 1
            
            # Try fallback
            if fallback:
                self._metrics['fallbacks'] += 1
                if asyncio.iscoroutinefunction(fallback):
                    return await fallback(*args, **kwargs)
                return fallback(*args, **kwargs)
            
            # Try fallback cache
            if cache_key:
                cached_value = self._fallback_cache.get(cache_key)
                if cached_value is not None:
                    self._metrics['fallbacks'] += 1
                    logger.info(f"Using fallback cache for {cache_key}")
                    return cached_value
            
            # No fallback available
            raise
            
        except self.expected_exception as e:
            # Expected error, increment failure count
            self._metrics['failures'] += 1
            logger.error(f"Redis operation failed: {e}")
            
            # Try fallback on failure
            if fallback:
                self._metrics['fallbacks'] += 1
                if asyncio.iscoroutinefunction(fallback):
                    return await fallback(*args, **kwargs)
                return fallback(*args, **kwargs)
            
            raise
    
    def _update_fallback_cache(self, key: str, value: Any, ttl: int = 300):
        """Update fallback cache with TTL."""
        self._fallback_cache.set(key, value, ttl=ttl)
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached entry is still valid."""
        value = self._fallback_cache.get(key)
        return value is not None
    
    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        if self._breaker.closed:
            return CircuitState.CLOSED
        elif self._breaker.opened:
            return CircuitState.OPEN
        else:
            return CircuitState.HALF_OPEN
    
    def reset(self):
        """Reset circuit breaker."""
        self._breaker.reset()
        self._metrics['state_changes'].append({
            'state': 'reset',
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        metrics = self._metrics.copy()
        metrics['state'] = self.get_state().value
        metrics['failure_rate'] = (
            self._metrics['failures'] / self._metrics['calls'] * 100
            if self._metrics['calls'] > 0 else 0
        )
        return metrics


class ResilientRedisClient:
    """
    Resilient Redis client with Sentinel support and circuit breaker.
    """
    
    def __init__(
        self,
        mode: RedisMode = RedisMode.STANDALONE,
        standalone_url: str = "redis://localhost:6379",
        sentinel_hosts: Optional[List[tuple]] = None,
        sentinel_service: str = "mymaster",
        circuit_breaker: Optional[RedisCircuitBreaker] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize resilient Redis client.
        
        Args:
            mode: Redis deployment mode
            standalone_url: URL for standalone Redis
            sentinel_hosts: List of (host, port) tuples for Sentinel
            sentinel_service: Sentinel service name
            circuit_breaker: Circuit breaker instance
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries
        """
        self.mode = mode
        self.standalone_url = standalone_url
        self.sentinel_hosts = sentinel_hosts or [('localhost', 26379)]
        self.sentinel_service = sentinel_service
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Circuit breaker
        self.circuit_breaker = circuit_breaker or RedisCircuitBreaker()
        
        # Redis clients
        self._redis_client: Optional[aioredis.Redis] = None
        self._sentinel: Optional[Sentinel] = None
        self._sync_client: Optional[redis.Redis] = None
        
        # Connection pool settings
        self._pool_settings = {
            'max_connections': 50,
            'socket_keepalive': True,
            'socket_connect_timeout': 5,
            'retry_on_timeout': True,
            'health_check_interval': 30
        }
        
        # Metrics
        self._metrics = {
            'connections': 0,
            'reconnections': 0,
            'failures': 0,
            'sentinel_failovers': 0
        }
    
    async def initialize(self) -> None:
        """Initialize Redis connection based on mode."""
        try:
            if self.mode == RedisMode.SENTINEL:
                await self._initialize_sentinel()
            else:
                await self._initialize_standalone()
            
            logger.info(f"Redis client initialized in {self.mode.value} mode")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis client: {e}")
            raise
    
    async def _initialize_standalone(self) -> None:
        """Initialize standalone Redis connection."""
        self._redis_client = await aioredis.from_url(
            self.standalone_url,
            encoding="utf-8",
            decode_responses=True,
            **self._pool_settings
        )
        
        # Test connection
        await self._redis_client.ping()
        self._metrics['connections'] += 1
    
    async def _initialize_sentinel(self) -> None:
        """Initialize Redis Sentinel connection."""
        # Create sentinel instance
        self._sentinel = Sentinel(
            self.sentinel_hosts,
            socket_timeout=5,
            socket_connect_timeout=5
        )
        
        # Discover master
        master_info = self._sentinel.discover_master(self.sentinel_service)
        logger.info(f"Discovered Redis master at {master_info}")
        
        # Get master connection
        master_host, master_port = master_info
        redis_url = f"redis://{master_host}:{master_port}"
        
        self._redis_client = await aioredis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=True,
            **self._pool_settings
        )
        
        # Test connection
        await self._redis_client.ping()
        self._metrics['connections'] += 1
        
        # Start sentinel monitoring
        self._monitor_task = asyncio.create_task(self._monitor_sentinel())
    
    async def _monitor_sentinel(self) -> None:
        """Monitor Sentinel for master changes."""
        last_master = None
        
        while True:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                if self._sentinel:
                    current_master = self._sentinel.discover_master(self.sentinel_service)
                    
                    if last_master and current_master != last_master:
                        logger.warning(f"Redis master changed from {last_master} to {current_master}")
                        self._metrics['sentinel_failovers'] += 1
                        
                        # Reconnect to new master
                        await self._handle_failover(current_master)
                    
                    last_master = current_master
                    
            except Exception as e:
                logger.error(f"Sentinel monitoring error: {e}")
                await asyncio.sleep(30)  # Back off on error
    
    async def _handle_failover(self, new_master: tuple) -> None:
        """Handle Redis master failover."""
        try:
            # Close old connection
            if self._redis_client:
                await self._redis_client.close()
            
            # Connect to new master
            master_host, master_port = new_master
            redis_url = f"redis://{master_host}:{master_port}"
            
            self._redis_client = await aioredis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=True,
                **self._pool_settings
            )
            
            await self._redis_client.ping()
            self._metrics['reconnections'] += 1
            
            logger.info(f"Successfully connected to new master at {new_master}")
            
        except Exception as e:
            logger.error(f"Failed to handle failover: {e}")
            self._metrics['failures'] += 1
    
    async def _execute_with_retry(
        self,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute operation with retry logic."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return await operation(*args, **kwargs)
                
            except (RedisConnectionError, ConnectionError) as e:
                last_error = e
                logger.warning(f"Redis operation failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries - 1:
                    # Exponential backoff with jitter
                    delay = self.retry_delay * (2 ** attempt) + random.uniform(0, 1)
                    await asyncio.sleep(delay)
                    
                    # Try to reconnect
                    try:
                        await self.initialize()
                    except (RedisConnectionError, ConnectionError, RedisError) as reconnect_error:
                        logger.warning(f"Failed to reconnect: {reconnect_error}")
        
        raise last_error
    
    @asynccontextmanager
    async def pipeline(self):
        """Get Redis pipeline with circuit breaker protection."""
        async def create_pipeline():
            return self._redis_client.pipeline()
        
        pipeline = await self.circuit_breaker.call(create_pipeline)
        try:
            yield pipeline
        finally:
            await self.circuit_breaker.call(pipeline.execute)
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value with circuit breaker and retry."""
        async def _get():
            return await self._execute_with_retry(
                self._redis_client.get,
                key
            )
        
        try:
            result = await self.circuit_breaker.call(
                _get,
                cache_key=f"get:{key}"
            )
            return result if result is not None else default
            
        except (CircuitBreakerError, RedisError):
            return default
    
    async def set(
        self,
        key: str,
        value: Any,
        ex: Optional[int] = None
    ) -> bool:
        """Set value with circuit breaker and retry."""
        async def _set():
            return await self._execute_with_retry(
                self._redis_client.set,
                key,
                value,
                ex=ex
            )
        
        try:
            await self.circuit_breaker.call(_set)
            return True
        except (CircuitBreakerError, RedisError):
            logger.error(f"Failed to set {key}")
            return False
    
    async def delete(self, *keys: str) -> int:
        """Delete keys with circuit breaker."""
        async def _delete():
            return await self._execute_with_retry(
                self._redis_client.delete,
                *keys
            )
        
        try:
            return await self.circuit_breaker.call(_delete)
        except (CircuitBreakerError, RedisError):
            return 0
    
    async def exists(self, *keys: str) -> int:
        """Check if keys exist with circuit breaker."""
        async def _exists():
            return await self._execute_with_retry(
                self._redis_client.exists,
                *keys
            )
        
        try:
            return await self.circuit_breaker.call(_exists)
        except (CircuitBreakerError, RedisError):
            return 0
    
    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiry with circuit breaker."""
        async def _expire():
            return await self._execute_with_retry(
                self._redis_client.expire,
                key,
                seconds
            )
        
        try:
            return await self.circuit_breaker.call(_expire)
        except (CircuitBreakerError, RedisError):
            return False
    
    async def incr(self, key: str) -> int:
        """Increment with circuit breaker."""
        async def _incr():
            return await self._execute_with_retry(
                self._redis_client.incr,
                key
            )
        
        try:
            return await self.circuit_breaker.call(_incr)
        except (CircuitBreakerError, RedisError):
            return 0
    
    async def hset(self, key: str, field: str, value: Any) -> int:
        """Hash set with circuit breaker."""
        async def _hset():
            return await self._execute_with_retry(
                self._redis_client.hset,
                key,
                field,
                value
            )
        
        try:
            return await self.circuit_breaker.call(_hset)
        except (CircuitBreakerError, RedisError):
            return 0
    
    async def hget(self, key: str, field: str) -> Optional[str]:
        """Hash get with circuit breaker."""
        async def _hget():
            return await self._execute_with_retry(
                self._redis_client.hget,
                key,
                field
            )
        
        try:
            return await self.circuit_breaker.call(
                _hget,
                cache_key=f"hget:{key}:{field}"
            )
        except (CircuitBreakerError, RedisError):
            return None
    
    async def zadd(self, key: str, mapping: Dict[str, float]) -> int:
        """Sorted set add with circuit breaker."""
        async def _zadd():
            return await self._execute_with_retry(
                self._redis_client.zadd,
                key,
                mapping
            )
        
        try:
            return await self.circuit_breaker.call(_zadd)
        except (CircuitBreakerError, RedisError):
            return 0
    
    async def zrange(
        self,
        key: str,
        start: int,
        stop: int,
        withscores: bool = False
    ) -> List:
        """Sorted set range with circuit breaker."""
        async def _zrange():
            return await self._execute_with_retry(
                self._redis_client.zrange,
                key,
                start,
                stop,
                withscores=withscores
            )
        
        try:
            return await self.circuit_breaker.call(
                _zrange,
                cache_key=f"zrange:{key}:{start}:{stop}"
            )
        except (CircuitBreakerError, RedisError):
            return []
    
    async def ping(self) -> bool:
        """Ping Redis with circuit breaker."""
        async def _ping():
            return await self._redis_client.ping()
        
        try:
            await self.circuit_breaker.call(_ping)
            return True
        except (CircuitBreakerError, RedisError):
            return False
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of Redis connection."""
        return {
            'mode': self.mode.value,
            'connected': self._redis_client is not None,
            'circuit_state': self.circuit_breaker.get_state().value,
            'circuit_metrics': self.circuit_breaker.get_metrics(),
            'connection_metrics': self._metrics,
            'sentinel_active': self._sentinel is not None
        }
    
    async def close(self) -> None:
        """Close Redis connections and cleanup tasks."""
        # Cancel monitoring task if exists
        if hasattr(self, '_monitor_task') and self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        # Close Redis connections
        if self._redis_client:
            await self._redis_client.close()
        if self._sync_client:
            self._sync_client.close()


# Factory function for creating resilient Redis clients
def create_resilient_redis(
    mode: str = "standalone",
    **kwargs
) -> ResilientRedisClient:
    """
    Factory function to create resilient Redis client.
    
    Args:
        mode: "standalone", "sentinel", or "cluster"
        **kwargs: Additional arguments for ResilientRedisClient
    
    Returns:
        Configured ResilientRedisClient instance
    """
    redis_mode = RedisMode(mode.lower())
    
    # Create circuit breaker with appropriate settings
    circuit_breaker = RedisCircuitBreaker(
        failure_threshold=5,
        recovery_timeout=60,
        name=f"redis_{mode}"
    )
    
    # Create client
    client = ResilientRedisClient(
        mode=redis_mode,
        circuit_breaker=circuit_breaker,
        **kwargs
    )
    
    return client


# Global resilient Redis instance
resilient_redis = create_resilient_redis()