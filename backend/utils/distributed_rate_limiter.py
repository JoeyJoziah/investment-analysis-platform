"""
Distributed Rate Limiter using Redis
Provides thread-safe, distributed rate limiting across multiple instances.
"""

import asyncio
import time
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from redis.exceptions import RedisError, ConnectionError as RedisConnectionError
import redis

logger = logging.getLogger(__name__)


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""
    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class DistributedRateLimiter:
    """
    Distributed rate limiter using Redis for coordination across instances.
    Supports multiple strategies and automatic failover.
    """
    
    # Lua script for atomic sliding window rate limiting
    SLIDING_WINDOW_SCRIPT = """
    local key = KEYS[1]
    local now = tonumber(ARGV[1])
    local window = tonumber(ARGV[2])
    local limit = tonumber(ARGV[3])
    
    -- Remove old entries outside the window
    redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
    
    -- Count current entries in window
    local current = redis.call('ZCARD', key)
    
    if current < limit then
        -- Add new entry
        redis.call('ZADD', key, now, now)
        redis.call('EXPIRE', key, window)
        return {1, limit - current - 1}  -- allowed, remaining
    else
        -- Get oldest entry to calculate retry_after
        local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
        local retry_after = 0
        if #oldest > 0 then
            retry_after = oldest[2] + window - now
        end
        return {0, 0, retry_after}  -- denied, remaining, retry_after
    end
    """
    
    # Lua script for token bucket algorithm
    TOKEN_BUCKET_SCRIPT = """
    local key = KEYS[1]
    local capacity = tonumber(ARGV[1])
    local refill_rate = tonumber(ARGV[2])
    local refill_period = tonumber(ARGV[3])
    local requested = tonumber(ARGV[4])
    local now = tonumber(ARGV[5])
    
    local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
    local tokens = tonumber(bucket[1]) or capacity
    local last_refill = tonumber(bucket[2]) or now
    
    -- Calculate tokens to add based on time passed
    local time_passed = now - last_refill
    local tokens_to_add = math.floor(time_passed / refill_period * refill_rate)
    tokens = math.min(capacity, tokens + tokens_to_add)
    
    if tokens >= requested then
        tokens = tokens - requested
        redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
        redis.call('EXPIRE', key, capacity * refill_period / refill_rate)
        return {1, tokens}  -- allowed, remaining tokens
    else
        local wait_time = (requested - tokens) * refill_period / refill_rate
        return {0, tokens, wait_time}  -- denied, available tokens, wait time
    end
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        key_prefix: str = "rate_limit",
        strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW,
        fallback_to_local: bool = True
    ):
        """
        Initialize distributed rate limiter.
        
        Args:
            redis_url: Redis connection URL
            key_prefix: Prefix for Redis keys
            strategy: Rate limiting strategy to use
            fallback_to_local: Fall back to local limiting if Redis unavailable
        """
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.strategy = strategy
        self.fallback_to_local = fallback_to_local
        
        self._redis_client: Optional[aioredis.Redis] = None
        self._sync_redis_client: Optional[redis.Redis] = None
        self._scripts: Dict[str, Any] = {}
        self._local_limits: Dict[str, List[float]] = {}  # Fallback local storage
        
        # Performance metrics
        self._metrics = {
            'requests': 0,
            'allowed': 0,
            'denied': 0,
            'errors': 0,
            'redis_failures': 0
        }
    
    async def initialize(self) -> None:
        """Initialize Redis connection and register scripts."""
        try:
            self._redis_client = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_keepalive=True,
                socket_connect_timeout=5,
                retry_on_timeout=True,
                max_connections=50
            )
            
            # Test connection
            await self._redis_client.ping()
            
            # Register Lua scripts
            self._scripts['sliding_window'] = await self._redis_client.script_load(
                self.SLIDING_WINDOW_SCRIPT
            )
            self._scripts['token_bucket'] = await self._redis_client.script_load(
                self.TOKEN_BUCKET_SCRIPT
            )
            
            logger.info("Distributed rate limiter initialized successfully")
            
        except (RedisError, RedisConnectionError) as e:
            logger.error(f"Failed to initialize Redis connection: {e}")
            if not self.fallback_to_local:
                raise
            logger.warning("Falling back to local rate limiting")
    
    async def check_rate_limit(
        self,
        identifier: str,
        limit: int,
        window: int,
        cost: int = 1
    ) -> Tuple[bool, int, Optional[float]]:
        """
        Check if request is within rate limit.
        
        Args:
            identifier: Unique identifier (e.g., user_id, ip_address, api_key)
            limit: Maximum requests allowed
            window: Time window in seconds
            cost: Cost of this request (for token bucket)
        
        Returns:
            Tuple of (allowed, remaining, retry_after)
        """
        self._metrics['requests'] += 1
        
        try:
            if self.strategy == RateLimitStrategy.SLIDING_WINDOW:
                result = await self._check_sliding_window(identifier, limit, window)
            elif self.strategy == RateLimitStrategy.TOKEN_BUCKET:
                result = await self._check_token_bucket(identifier, limit, window, cost)
            elif self.strategy == RateLimitStrategy.FIXED_WINDOW:
                result = await self._check_fixed_window(identifier, limit, window)
            else:
                result = await self._check_leaky_bucket(identifier, limit, window)
            
            if result[0]:
                self._metrics['allowed'] += 1
            else:
                self._metrics['denied'] += 1
            
            return result
            
        except (RedisError, RedisConnectionError) as e:
            logger.error(f"Redis error in rate limiting: {e}")
            self._metrics['errors'] += 1
            self._metrics['redis_failures'] += 1
            
            if self.fallback_to_local:
                return self._check_local_fallback(identifier, limit, window)
            raise
    
    async def _check_sliding_window(
        self,
        identifier: str,
        limit: int,
        window: int
    ) -> Tuple[bool, int, Optional[float]]:
        """Check rate limit using sliding window algorithm."""
        if not self._redis_client:
            await self.initialize()
        
        key = f"{self.key_prefix}:sliding:{identifier}"
        now = time.time()
        
        result = await self._redis_client.evalsha(
            self._scripts['sliding_window'],
            1,
            key,
            now,
            window,
            limit
        )
        
        allowed = bool(result[0])
        remaining = int(result[1]) if allowed else 0
        retry_after = float(result[2]) if len(result) > 2 else None
        
        return allowed, remaining, retry_after
    
    async def _check_token_bucket(
        self,
        identifier: str,
        limit: int,
        window: int,
        cost: int
    ) -> Tuple[bool, int, Optional[float]]:
        """Check rate limit using token bucket algorithm."""
        if not self._redis_client:
            await self.initialize()
        
        key = f"{self.key_prefix}:bucket:{identifier}"
        now = time.time()
        refill_rate = limit  # Tokens per period
        refill_period = window  # Period in seconds
        
        result = await self._redis_client.evalsha(
            self._scripts['token_bucket'],
            1,
            key,
            limit,  # capacity
            refill_rate,
            refill_period,
            cost,
            now
        )
        
        allowed = bool(result[0])
        remaining = int(result[1])
        retry_after = float(result[2]) if len(result) > 2 else None
        
        return allowed, remaining, retry_after
    
    async def _check_fixed_window(
        self,
        identifier: str,
        limit: int,
        window: int
    ) -> Tuple[bool, int, Optional[float]]:
        """Check rate limit using fixed window algorithm."""
        if not self._redis_client:
            await self.initialize()
        
        # Calculate window start
        now = int(time.time())
        window_start = (now // window) * window
        key = f"{self.key_prefix}:fixed:{identifier}:{window_start}"
        
        # Increment counter
        pipe = self._redis_client.pipeline()
        pipe.incr(key)
        pipe.expire(key, window)
        results = await pipe.execute()
        
        current = results[0]
        allowed = current <= limit
        remaining = max(0, limit - current)
        retry_after = window - (now - window_start) if not allowed else None
        
        return allowed, remaining, retry_after
    
    async def _check_leaky_bucket(
        self,
        identifier: str,
        limit: int,
        window: int
    ) -> Tuple[bool, int, Optional[float]]:
        """Check rate limit using leaky bucket algorithm."""
        if not self._redis_client:
            await self.initialize()
        
        key = f"{self.key_prefix}:leaky:{identifier}"
        now = time.time()
        leak_rate = limit / window  # Requests leaked per second
        
        # Get current bucket state
        pipe = self._redis_client.pipeline()
        pipe.hget(key, "level")
        pipe.hget(key, "last_leak")
        results = await pipe.execute()
        
        level = float(results[0] or 0)
        last_leak = float(results[1] or now)
        
        # Calculate leaked amount
        time_passed = now - last_leak
        leaked = time_passed * leak_rate
        level = max(0, level - leaked)
        
        # Check if request fits
        if level < limit:
            level += 1
            allowed = True
            remaining = int(limit - level)
            retry_after = None
            
            # Update bucket
            pipe = self._redis_client.pipeline()
            pipe.hset(key, "level", level)
            pipe.hset(key, "last_leak", now)
            pipe.expire(key, window * 2)
            await pipe.execute()
        else:
            allowed = False
            remaining = 0
            retry_after = (level - limit + 1) / leak_rate
        
        return allowed, remaining, retry_after
    
    def _check_local_fallback(
        self,
        identifier: str,
        limit: int,
        window: int
    ) -> Tuple[bool, int, Optional[float]]:
        """Fallback to local rate limiting when Redis is unavailable."""
        now = time.time()
        key = f"{identifier}:{window}"
        
        # Clean old entries
        if key in self._local_limits:
            self._local_limits[key] = [
                t for t in self._local_limits[key]
                if t > now - window
            ]
        else:
            self._local_limits[key] = []
        
        current = len(self._local_limits[key])
        
        if current < limit:
            self._local_limits[key].append(now)
            return True, limit - current - 1, None
        else:
            oldest = min(self._local_limits[key])
            retry_after = oldest + window - now
            return False, 0, retry_after
    
    @asynccontextmanager
    async def rate_limit_context(
        self,
        identifier: str,
        limit: int,
        window: int,
        raise_on_limit: bool = True
    ):
        """
        Context manager for rate limiting.
        
        Usage:
            async with limiter.rate_limit_context("user_123", 100, 60):
                # Protected code
                await make_api_call()
        """
        allowed, remaining, retry_after = await self.check_rate_limit(
            identifier, limit, window
        )
        
        if not allowed and raise_on_limit:
            raise RateLimitExceeded(
                f"Rate limit exceeded. Try again in {retry_after:.1f} seconds",
                retry_after
            )
        
        try:
            yield allowed, remaining
        finally:
            pass  # Could add cleanup here if needed
    
    async def reset_limit(self, identifier: str) -> None:
        """Reset rate limit for an identifier."""
        if not self._redis_client:
            await self.initialize()
        
        patterns = [
            f"{self.key_prefix}:sliding:{identifier}",
            f"{self.key_prefix}:bucket:{identifier}",
            f"{self.key_prefix}:fixed:{identifier}:*",
            f"{self.key_prefix}:leaky:{identifier}"
        ]
        
        for pattern in patterns:
            keys = await self._redis_client.keys(pattern)
            if keys:
                await self._redis_client.delete(*keys)
    
    async def get_usage(self, identifier: str) -> Dict[str, Any]:
        """Get current usage statistics for an identifier."""
        if not self._redis_client:
            await self.initialize()
        
        usage = {}
        
        if self.strategy == RateLimitStrategy.SLIDING_WINDOW:
            key = f"{self.key_prefix}:sliding:{identifier}"
            count = await self._redis_client.zcard(key)
            usage['current_requests'] = count
            
        elif self.strategy == RateLimitStrategy.TOKEN_BUCKET:
            key = f"{self.key_prefix}:bucket:{identifier}"
            bucket = await self._redis_client.hgetall(key)
            usage['available_tokens'] = float(bucket.get('tokens', 0))
            usage['last_refill'] = float(bucket.get('last_refill', 0))
        
        usage['strategy'] = self.strategy.value
        return usage
    
    def get_metrics(self) -> Dict[str, int]:
        """Get rate limiter performance metrics."""
        return self._metrics.copy()
    
    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis_client:
            await self._redis_client.close()


class APIRateLimiter:
    """
    Specialized rate limiter for API providers with different limits.
    """
    
    # API provider limits (calls per minute, calls per day)
    PROVIDER_LIMITS = {
        'alpha_vantage': {'per_minute': 5, 'per_day': 25},
        'finnhub': {'per_minute': 60, 'per_day': float('inf')},
        'polygon': {'per_minute': 5, 'per_day': float('inf')},
        'fmp': {'per_minute': float('inf'), 'per_day': 250},
        'newsapi': {'per_minute': float('inf'), 'per_day': 500},
        'sec': {'per_minute': 10, 'per_day': float('inf')},
    }
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """Initialize API rate limiter."""
        self.limiters = {
            'minute': DistributedRateLimiter(
                redis_url,
                key_prefix="api_limit_minute",
                strategy=RateLimitStrategy.SLIDING_WINDOW
            ),
            'day': DistributedRateLimiter(
                redis_url,
                key_prefix="api_limit_day",
                strategy=RateLimitStrategy.FIXED_WINDOW
            )
        }
    
    async def initialize(self) -> None:
        """Initialize all rate limiters."""
        for limiter in self.limiters.values():
            await limiter.initialize()
    
    async def check_api_limit(
        self,
        provider: str,
        cost: int = 1
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if API call is within limits.
        
        Args:
            provider: API provider name
            cost: Number of API calls this request costs
        
        Returns:
            Tuple of (allowed, details)
        """
        if provider not in self.PROVIDER_LIMITS:
            return True, {'error': 'Unknown provider'}
        
        limits = self.PROVIDER_LIMITS[provider]
        details = {'provider': provider}
        
        # Check minute limit
        if limits['per_minute'] != float('inf'):
            allowed, remaining, retry_after = await self.limiters['minute'].check_rate_limit(
                provider,
                int(limits['per_minute']),
                60,  # 1 minute window
                cost
            )
            
            if not allowed:
                details['limited_by'] = 'per_minute'
                details['retry_after'] = retry_after
                details['remaining_minute'] = remaining
                return False, details
            
            details['remaining_minute'] = remaining
        
        # Check daily limit
        if limits['per_day'] != float('inf'):
            allowed, remaining, retry_after = await self.limiters['day'].check_rate_limit(
                provider,
                int(limits['per_day']),
                86400,  # 24 hour window
                cost
            )
            
            if not allowed:
                details['limited_by'] = 'per_day'
                details['retry_after'] = retry_after
                details['remaining_day'] = remaining
                return False, details
            
            details['remaining_day'] = remaining
        
        details['allowed'] = True
        return True, details
    
    async def get_provider_usage(self, provider: str) -> Dict[str, Any]:
        """Get current usage for a provider."""
        usage = {
            'provider': provider,
            'limits': self.PROVIDER_LIMITS.get(provider, {})
        }
        
        if provider in self.PROVIDER_LIMITS:
            minute_usage = await self.limiters['minute'].get_usage(provider)
            day_usage = await self.limiters['day'].get_usage(provider)
            
            usage['current'] = {
                'per_minute': minute_usage,
                'per_day': day_usage
            }
        
        return usage
    
    async def reset_provider_limits(self, provider: str) -> None:
        """Reset limits for a provider (useful for testing)."""
        await self.limiters['minute'].reset_limit(provider)
        await self.limiters['day'].reset_limit(provider)
    
    async def close(self) -> None:
        """Close all rate limiters."""
        for limiter in self.limiters.values():
            await limiter.close()


# Global instances
rate_limiter = DistributedRateLimiter()
api_rate_limiter = APIRateLimiter()