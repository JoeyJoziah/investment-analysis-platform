"""Advanced Rate Limiting and API Usage Tracking"""

import asyncio
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

import redis.asyncio as redis
from fastapi import HTTPException, Request
from pydantic import BaseModel

from backend.config.settings import settings
from backend.utils.exceptions import RateLimitException
from backend.utils.monitoring import metrics

# Rate limit configurations for different API providers
API_RATE_LIMITS = {
    "alpha_vantage": {
        "calls_per_minute": 5,
        "calls_per_day": 25,
        "burst_size": 5,
        "cooldown_seconds": 60
    },
    "finnhub": {
        "calls_per_minute": 60,
        "calls_per_day": None,  # No daily limit
        "burst_size": 30,
        "cooldown_seconds": 10
    },
    "polygon": {
        "calls_per_minute": 5,
        "calls_per_day": None,
        "burst_size": 5,
        "cooldown_seconds": 60
    },
    "newsapi": {
        "calls_per_minute": 30,
        "calls_per_day": 100,
        "burst_size": 10,
        "cooldown_seconds": 30
    },
    "sec_edgar": {
        "calls_per_minute": 10,
        "calls_per_day": None,
        "burst_size": 5,
        "cooldown_seconds": 60
    }
}


class RateLimitInfo(BaseModel):
    """Rate limit information"""
    limit: int
    remaining: int
    reset: datetime
    retry_after: Optional[int] = None


class APIUsageStats(BaseModel):
    """API usage statistics"""
    provider: str
    calls_today: int
    calls_this_hour: int
    calls_this_minute: int
    daily_limit: Optional[int]
    hourly_limit: Optional[int]
    minute_limit: Optional[int]
    last_call: Optional[datetime]
    estimated_daily_cost: float
    is_available: bool


class TokenBucket:
    """Token bucket algorithm for rate limiting"""
    
    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket
        
        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self._lock = asyncio.Lock()
        
    async def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False otherwise
        """
        async with self._lock:
            # Refill tokens based on time passed
            now = time.time()
            elapsed = now - self.last_refill
            self.tokens = min(
                self.capacity,
                self.tokens + (elapsed * self.refill_rate)
            )
            self.last_refill = now
            
            # Try to consume tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
            
    async def wait_for_tokens(self, tokens: int = 1) -> float:
        """
        Calculate wait time for tokens to be available
        
        Args:
            tokens: Number of tokens needed
            
        Returns:
            Seconds to wait (0 if tokens available)
        """
        async with self._lock:
            if self.tokens >= tokens:
                return 0
                
            tokens_needed = tokens - self.tokens
            wait_time = tokens_needed / self.refill_rate
            return wait_time


class SlidingWindowCounter:
    """Sliding window counter for rate limiting"""
    
    def __init__(self, window_size: int):
        """
        Initialize sliding window counter
        
        Args:
            window_size: Window size in seconds
        """
        self.window_size = window_size
        self.requests = deque()
        self._lock = asyncio.Lock()
        
    async def add_request(self, timestamp: Optional[float] = None) -> int:
        """
        Add a request and return current count
        
        Args:
            timestamp: Request timestamp (current time if None)
            
        Returns:
            Current request count in window
        """
        async with self._lock:
            now = timestamp or time.time()
            
            # Remove old requests outside window
            cutoff = now - self.window_size
            while self.requests and self.requests[0] < cutoff:
                self.requests.popleft()
                
            # Add new request
            self.requests.append(now)
            
            return len(self.requests)
            
    async def get_count(self) -> int:
        """Get current request count in window"""
        async with self._lock:
            now = time.time()
            cutoff = now - self.window_size
            
            # Remove old requests
            while self.requests and self.requests[0] < cutoff:
                self.requests.popleft()
                
            return len(self.requests)


class RateLimiter:
    """Advanced rate limiter with multiple algorithms"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.local_buckets: Dict[str, TokenBucket] = {}
        self.local_windows: Dict[str, SlidingWindowCounter] = {}
        self.usage_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
    async def check_rate_limit(
        self,
        key: str,
        limit: int,
        window: int,
        cost: int = 1
    ) -> Tuple[bool, RateLimitInfo]:
        """
        Check if request is within rate limit
        
        Args:
            key: Rate limit key
            limit: Maximum requests allowed
            window: Time window in seconds
            cost: Cost of this request
            
        Returns:
            Tuple of (allowed, rate_limit_info)
        """
        if self.redis_client:
            return await self._check_redis_rate_limit(key, limit, window, cost)
        else:
            return await self._check_local_rate_limit(key, limit, window, cost)
            
    async def _check_redis_rate_limit(
        self,
        key: str,
        limit: int,
        window: int,
        cost: int
    ) -> Tuple[bool, RateLimitInfo]:
        """Check rate limit using Redis"""
        now = time.time()
        window_start = now - window
        
        # Use Redis sorted set for sliding window
        pipe = self.redis_client.pipeline()
        
        # Remove old entries
        pipe.zremrangebyscore(key, 0, window_start)
        
        # Count current requests
        pipe.zcard(key)
        
        # Add current request
        pipe.zadd(key, {str(now): now})
        
        # Set expiry
        pipe.expire(key, window + 1)
        
        results = await pipe.execute()
        current_count = results[1]
        
        # Check if within limit
        allowed = current_count + cost <= limit
        
        # Calculate reset time
        if results[0] > 0:
            oldest_request = await self.redis_client.zrange(key, 0, 0, withscores=True)
            if oldest_request:
                reset_time = datetime.fromtimestamp(oldest_request[0][1] + window)
            else:
                reset_time = datetime.fromtimestamp(now + window)
        else:
            reset_time = datetime.fromtimestamp(now + window)
            
        info = RateLimitInfo(
            limit=limit,
            remaining=max(0, limit - current_count - cost),
            reset=reset_time,
            retry_after=int(window) if not allowed else None
        )
        
        return allowed, info
        
    async def _check_local_rate_limit(
        self,
        key: str,
        limit: int,
        window: int,
        cost: int
    ) -> Tuple[bool, RateLimitInfo]:
        """Check rate limit using local memory"""
        # Get or create sliding window
        if key not in self.local_windows:
            self.local_windows[key] = SlidingWindowCounter(window)
            
        window_counter = self.local_windows[key]
        current_count = await window_counter.get_count()
        
        # Check if within limit
        allowed = current_count + cost <= limit
        
        if allowed:
            await window_counter.add_request()
            
        # Calculate reset time
        reset_time = datetime.fromtimestamp(time.time() + window)
        
        info = RateLimitInfo(
            limit=limit,
            remaining=max(0, limit - current_count - cost),
            reset=reset_time,
            retry_after=int(window) if not allowed else None
        )
        
        return allowed, info
        
    async def track_api_usage(
        self,
        provider: str,
        endpoint: str,
        cost: float = 0.0
    ):
        """Track API usage for cost monitoring"""
        now = datetime.utcnow()
        
        # Track in Redis if available
        if self.redis_client:
            # Daily counter
            daily_key = f"api_usage:{provider}:{now.strftime('%Y-%m-%d')}"
            await self.redis_client.hincrby(daily_key, endpoint, 1)
            await self.redis_client.expire(daily_key, 86400 * 7)  # Keep for 7 days
            
            # Hourly counter
            hourly_key = f"api_usage:{provider}:{now.strftime('%Y-%m-%d-%H')}"
            await self.redis_client.hincrby(hourly_key, endpoint, 1)
            await self.redis_client.expire(hourly_key, 3600 * 24)  # Keep for 24 hours
            
            # Cost tracking
            if cost > 0:
                cost_key = f"api_cost:{provider}:{now.strftime('%Y-%m')}"
                await self.redis_client.hincrbyfloat(cost_key, now.strftime('%Y-%m-%d'), cost)
                await self.redis_client.expire(cost_key, 86400 * 31)  # Keep for 31 days
                
        # Track locally
        self.usage_stats[provider]["daily"] += 1
        self.usage_stats[provider]["hourly"] += 1
        self.usage_stats[provider]["cost"] += cost
        
        # Emit metrics
        metrics.api_calls_total.labels(
            provider=provider,
            endpoint=endpoint
        ).inc()
        
        if cost > 0:
            metrics.api_cost_total.labels(provider=provider).inc(cost)
            
    async def get_api_usage_stats(self, provider: str) -> APIUsageStats:
        """Get API usage statistics for a provider"""
        config = API_RATE_LIMITS.get(provider, {})
        now = datetime.utcnow()
        
        # Get usage counts
        if self.redis_client:
            # Daily usage
            daily_key = f"api_usage:{provider}:{now.strftime('%Y-%m-%d')}"
            daily_usage = await self.redis_client.hvals(daily_key)
            calls_today = sum(int(x) for x in daily_usage) if daily_usage else 0
            
            # Hourly usage
            hourly_key = f"api_usage:{provider}:{now.strftime('%Y-%m-%d-%H')}"
            hourly_usage = await self.redis_client.hvals(hourly_key)
            calls_this_hour = sum(int(x) for x in hourly_usage) if hourly_usage else 0
            
            # Minute usage (estimate from last 60 seconds)
            calls_this_minute = calls_this_hour // 60  # Simple estimate
            
            # Cost
            cost_key = f"api_cost:{provider}:{now.strftime('%Y-%m')}"
            daily_cost = await self.redis_client.hget(cost_key, now.strftime('%Y-%m-%d'))
            estimated_daily_cost = float(daily_cost) if daily_cost else 0.0
            
        else:
            # Use local stats
            calls_today = self.usage_stats[provider]["daily"]
            calls_this_hour = self.usage_stats[provider]["hourly"]
            calls_this_minute = calls_this_hour // 60
            estimated_daily_cost = self.usage_stats[provider]["cost"]
            
        # Check availability
        is_available = True
        if config.get("calls_per_day") and calls_today >= config["calls_per_day"]:
            is_available = False
        elif config.get("calls_per_minute") and calls_this_minute >= config["calls_per_minute"]:
            is_available = False
            
        return APIUsageStats(
            provider=provider,
            calls_today=calls_today,
            calls_this_hour=calls_this_hour,
            calls_this_minute=calls_this_minute,
            daily_limit=config.get("calls_per_day"),
            hourly_limit=None,  # Not configured
            minute_limit=config.get("calls_per_minute"),
            last_call=now,
            estimated_daily_cost=estimated_daily_cost,
            is_available=is_available
        )
        
    async def wait_if_needed(self, provider: str) -> Optional[float]:
        """
        Wait if rate limit exceeded
        
        Returns:
            Wait time in seconds (None if no wait needed)
        """
        config = API_RATE_LIMITS.get(provider)
        if not config:
            return None
            
        # Check minute limit
        if config.get("calls_per_minute"):
            key = f"rate_limit:{provider}:minute"
            allowed, info = await self.check_rate_limit(
                key, 
                config["calls_per_minute"],
                60
            )
            
            if not allowed and info.retry_after:
                await asyncio.sleep(info.retry_after)
                return float(info.retry_after)
                
        return None


class RateLimitMiddleware:
    """FastAPI middleware for rate limiting"""
    
    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
        
    async def __call__(self, request: Request, call_next):
        """Apply rate limiting to requests"""
        # Get user/IP for rate limiting
        user_id = getattr(request.state, "user_id", None)
        client_ip = request.client.host if request.client else "unknown"
        
        # Determine rate limit key
        if user_id:
            key = f"user:{user_id}"
            limit = 1000  # Authenticated users
        else:
            key = f"ip:{client_ip}"
            limit = 100  # Anonymous users
            
        # Check rate limit
        allowed, info = await self.rate_limiter.check_rate_limit(
            key=key,
            limit=limit,
            window=3600  # 1 hour
        )
        
        if not allowed:
            raise RateLimitException(
                "Rate limit exceeded",
                retry_after=info.retry_after,
                details={
                    "limit": info.limit,
                    "remaining": info.remaining,
                    "reset": info.reset.isoformat()
                }
            )
            
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(info.limit)
        response.headers["X-RateLimit-Remaining"] = str(info.remaining)
        response.headers["X-RateLimit-Reset"] = str(int(info.reset.timestamp()))
        
        return response


# Global rate limiter instance
rate_limiter = RateLimiter()

def rate_limit(
    requests_per_minute: int = 60,
    requests_per_hour: int = 1000,
    key_func: Optional[callable] = None
):
    """
    Simple rate limiting decorator for API endpoints.
    
    This is a simplified rate limiter that can be applied to FastAPI endpoints.
    For more advanced rate limiting features, use the security.rate_limiter module.
    
    Args:
        requests_per_minute: Maximum requests allowed per minute
        requests_per_hour: Maximum requests allowed per hour
        key_func: Optional function to extract the rate limit key from request
        
    Returns:
        Decorator function
    """
    from functools import wraps
    from fastapi import Request, HTTPException
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get request from kwargs if available
            request: Optional[Request] = kwargs.get('request')
            
            # Try to get rate limiter
            try:
                # Get the key for rate limiting
                if key_func and request:
                    key = key_func(request)
                elif request:
                    # Default to client IP
                    key = f"ip:{request.client.host if request.client else 'unknown'}"
                else:
                    key = "default"
                
                # Check rate limit using global rate limiter
                allowed, info = await rate_limiter.check_rate_limit(
                    key=f"ratelimit:{key}",
                    limit=requests_per_minute,
                    window=60  # 1 minute window
                )
                
                if not allowed:
                    raise HTTPException(
                        status_code=429,
                        detail={
                            "error": "Rate limit exceeded",
                            "limit": info.limit,
                            "remaining": info.remaining,
                            "retry_after": info.retry_after
                        },
                        headers={
                            "X-RateLimit-Limit": str(info.limit),
                            "X-RateLimit-Remaining": str(info.remaining),
                            "Retry-After": str(info.retry_after) if info.retry_after else "60"
                        }
                    )
                    
            except Exception as e:
                # If rate limiting fails, log and continue
                import logging
                logging.getLogger(__name__).warning(f"Rate limiting check failed: {e}")
            
            return await func(*args, **kwargs)
        
        return wrapper
    
    return decorator
