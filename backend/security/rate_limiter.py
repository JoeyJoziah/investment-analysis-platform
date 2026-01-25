"""
Advanced Rate Limiting for Authentication and API Endpoints

This module provides sophisticated rate limiting with:
- Multiple rate limiting algorithms (Token Bucket, Sliding Window, Fixed Window)
- IP-based and user-based rate limiting
- Distributed rate limiting with Redis
- Adaptive rate limiting based on threat detection
- Rate limiting for different endpoint categories
"""

import time
import json
import logging
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import redis
import hashlib
from fastapi import HTTPException, status, Request
from ipaddress import ip_address, ip_network
import asyncio
from functools import wraps

logger = logging.getLogger(__name__)


class RateLimitAlgorithm(str, Enum):
    """Rate limiting algorithms"""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    ADAPTIVE = "adaptive"


class RateLimitCategory(str, Enum):
    """Categories of rate limits"""
    AUTHENTICATION = "auth"
    API_READ = "api_read"
    API_WRITE = "api_write"
    ADMIN = "admin"
    PASSWORD_RESET = "password_reset"
    REGISTRATION = "registration"
    FILE_UPLOAD = "file_upload"


@dataclass
class RateLimitRule:
    """Rate limit rule configuration"""
    requests: int  # Number of requests allowed
    window_seconds: int  # Time window in seconds
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.SLIDING_WINDOW
    block_duration_seconds: int = 300  # 5 minutes default
    burst_allowance: int = 0  # Additional burst requests
    adaptive_factor: float = 1.0  # Multiplier for adaptive limiting


@dataclass
class RateLimitStatus:
    """Current rate limit status"""
    allowed: bool
    remaining: int
    reset_time: datetime
    retry_after_seconds: Optional[int] = None
    blocked_until: Optional[datetime] = None


@dataclass
class RateLimitViolation:
    """Rate limit violation record"""
    timestamp: datetime
    ip_address: str
    user_id: Optional[int]
    category: RateLimitCategory
    requests_made: int
    limit: int
    user_agent: Optional[str] = None
    blocked_duration_seconds: int = 300


class AdvancedRateLimiter:
    """
    Advanced rate limiting system with multiple algorithms and threat detection.
    
    Features:
    - Multiple rate limiting algorithms
    - IP-based and user-based limits
    - Distributed limiting with Redis
    - Adaptive limits based on behavior
    - Comprehensive logging and monitoring
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client or self._get_redis_client()
        
        # Default rate limit rules
        self.default_rules = {
            RateLimitCategory.AUTHENTICATION: RateLimitRule(
                requests=5, 
                window_seconds=300,  # 5 attempts per 5 minutes
                block_duration_seconds=900,  # 15 minutes block
                algorithm=RateLimitAlgorithm.SLIDING_WINDOW
            ),
            RateLimitCategory.PASSWORD_RESET: RateLimitRule(
                requests=3,
                window_seconds=3600,  # 3 attempts per hour
                block_duration_seconds=1800  # 30 minutes block
            ),
            RateLimitCategory.REGISTRATION: RateLimitRule(
                requests=5,
                window_seconds=3600,  # 5 registrations per hour
                block_duration_seconds=3600  # 1 hour block
            ),
            RateLimitCategory.API_READ: RateLimitRule(
                requests=1000,
                window_seconds=3600,  # 1000 requests per hour
                burst_allowance=100
            ),
            RateLimitCategory.API_WRITE: RateLimitRule(
                requests=200,
                window_seconds=3600,  # 200 writes per hour
                block_duration_seconds=600  # 10 minutes block
            ),
            RateLimitCategory.ADMIN: RateLimitRule(
                requests=100,
                window_seconds=3600,  # 100 admin requests per hour
                algorithm=RateLimitAlgorithm.TOKEN_BUCKET
            ),
            RateLimitCategory.FILE_UPLOAD: RateLimitRule(
                requests=10,
                window_seconds=3600,  # 10 uploads per hour
                block_duration_seconds=1800
            )
        }
        
        # Trusted IP networks (e.g., internal networks)
        self.trusted_networks = [
            ip_network("127.0.0.0/8"),  # Loopback
            ip_network("10.0.0.0/8"),   # Private
            ip_network("172.16.0.0/12"), # Private
            ip_network("192.168.0.0/16") # Private
        ]
        
        # Violation tracking
        self.violation_threshold = 10  # Block after 10 violations in 24 hours
    
    def _get_redis_client(self) -> Optional[redis.Redis]:
        """Get Redis client for distributed rate limiting"""
        try:
            from backend.config.settings import settings
            return redis.from_url(settings.REDIS_URL, decode_responses=True)
        except Exception as e:
            logger.warning(f"Redis not available for rate limiting: {e}")
            return None
    
    def _is_trusted_ip(self, ip: str) -> bool:
        """Check if IP is in trusted networks"""
        try:
            ip_addr = ip_address(ip)
            return any(ip_addr in network for network in self.trusted_networks)
        except Exception:
            return False
    
    def _get_rate_limit_key(self, identifier: str, category: RateLimitCategory) -> str:
        """Generate Redis key for rate limiting"""
        return f"rate_limit:{category.value}:{identifier}"
    
    def _get_violation_key(self, ip: str) -> str:
        """Generate Redis key for violation tracking"""
        return f"rate_limit_violations:{ip}"
    
    def _get_block_key(self, identifier: str, category: RateLimitCategory) -> str:
        """Generate Redis key for blocking"""
        return f"rate_limit_block:{category.value}:{identifier}"
    
    async def check_rate_limit(
        self,
        request: Request,
        category: RateLimitCategory,
        user_id: Optional[int] = None,
        custom_rule: Optional[RateLimitRule] = None
    ) -> RateLimitStatus:
        """
        Check if request is within rate limits.
        
        Args:
            request: FastAPI request object
            category: Rate limit category
            user_id: Optional user ID for user-based limiting
            custom_rule: Optional custom rate limit rule
            
        Returns:
            RateLimitStatus indicating if request is allowed
        """
        try:
            # Extract client information
            client_ip = self._get_client_ip(request)
            user_agent = request.headers.get("user-agent", "unknown")
            
            # Check if IP is trusted
            if self._is_trusted_ip(client_ip):
                return RateLimitStatus(
                    allowed=True,
                    remaining=999999,
                    reset_time=datetime.utcnow() + timedelta(hours=1)
                )
            
            # Get rate limit rule
            rule = custom_rule or self.default_rules.get(category)
            if not rule:
                logger.warning(f"No rate limit rule found for category: {category}")
                return RateLimitStatus(
                    allowed=True,
                    remaining=999999,
                    reset_time=datetime.utcnow() + timedelta(hours=1)
                )
            
            # Determine identifier (IP or user-based)
            identifier = str(user_id) if user_id else client_ip
            
            # Check if currently blocked
            block_status = await self._check_block_status(identifier, category)
            if not block_status.allowed:
                return block_status
            
            # Check violation history for adaptive limiting
            violation_count = await self._get_violation_count(client_ip)
            if violation_count > 5:
                # Apply stricter limits for problematic IPs
                rule = self._apply_adaptive_limiting(rule, violation_count)
            
            # Apply rate limiting algorithm
            if rule.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
                status = await self._token_bucket_check(identifier, category, rule)
            elif rule.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
                status = await self._fixed_window_check(identifier, category, rule)
            else:  # SLIDING_WINDOW (default)
                status = await self._sliding_window_check(identifier, category, rule)
            
            # If rate limit exceeded, record violation and potentially block
            if not status.allowed:
                await self._record_violation(
                    client_ip, user_id, category, rule, user_agent
                )
                
                # Apply block if configured
                if rule.block_duration_seconds > 0:
                    await self._apply_block(identifier, category, rule.block_duration_seconds)
                    status.blocked_until = datetime.utcnow() + timedelta(seconds=rule.block_duration_seconds)
                    status.retry_after_seconds = rule.block_duration_seconds
            
            return status
            
        except Exception as e:
            logger.error(f"Error in rate limit check: {e}")
            # Fail open - allow request if rate limiting fails
            return RateLimitStatus(
                allowed=True,
                remaining=0,
                reset_time=datetime.utcnow() + timedelta(hours=1)
            )
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request"""
        # Check X-Forwarded-For header first (for proxy/load balancer scenarios)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            # Take the first IP in the chain
            return forwarded_for.split(",")[0].strip()
        
        # Check X-Real-IP header
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()
        
        # Fall back to direct client IP
        return request.client.host if request.client else "unknown"
    
    async def _check_block_status(
        self,
        identifier: str,
        category: RateLimitCategory
    ) -> RateLimitStatus:
        """Check if identifier is currently blocked"""
        if not self.redis_client:
            return RateLimitStatus(allowed=True, remaining=0, reset_time=datetime.utcnow())
        
        try:
            block_key = self._get_block_key(identifier, category)
            block_data = self.redis_client.get(block_key)
            
            if block_data:
                block_info = json.loads(block_data)
                blocked_until = datetime.fromisoformat(block_info["blocked_until"])
                
                if datetime.utcnow() < blocked_until:
                    retry_after = int((blocked_until - datetime.utcnow()).total_seconds())
                    return RateLimitStatus(
                        allowed=False,
                        remaining=0,
                        reset_time=blocked_until,
                        retry_after_seconds=retry_after,
                        blocked_until=blocked_until
                    )
                else:
                    # Block expired, remove it
                    self.redis_client.delete(block_key)
            
            return RateLimitStatus(allowed=True, remaining=0, reset_time=datetime.utcnow())
            
        except Exception as e:
            logger.error(f"Error checking block status: {e}")
            return RateLimitStatus(allowed=True, remaining=0, reset_time=datetime.utcnow())
    
    async def _sliding_window_check(
        self,
        identifier: str,
        category: RateLimitCategory,
        rule: RateLimitRule
    ) -> RateLimitStatus:
        """Implement sliding window rate limiting"""
        if not self.redis_client:
            return RateLimitStatus(allowed=True, remaining=0, reset_time=datetime.utcnow())
        
        try:
            now = time.time()
            window_start = now - rule.window_seconds
            key = self._get_rate_limit_key(identifier, category)
            
            # Use Redis sorted set for sliding window
            pipe = self.redis_client.pipeline()
            
            # Remove old entries
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count current entries
            pipe.zcard(key)
            
            # Add current request
            pipe.zadd(key, {str(now): now})
            
            # Set expiration
            pipe.expire(key, rule.window_seconds)
            
            results = pipe.execute()
            current_count = results[1]
            
            # Calculate remaining requests
            allowed_requests = rule.requests + rule.burst_allowance
            remaining = max(0, allowed_requests - current_count - 1)
            
            # Reset time is the start of next window
            reset_time = datetime.fromtimestamp(now + rule.window_seconds)
            
            return RateLimitStatus(
                allowed=current_count < allowed_requests,
                remaining=remaining,
                reset_time=reset_time
            )
            
        except Exception as e:
            logger.error(f"Error in sliding window rate limit: {e}")
            return RateLimitStatus(allowed=True, remaining=0, reset_time=datetime.utcnow())
    
    async def _token_bucket_check(
        self,
        identifier: str,
        category: RateLimitCategory,
        rule: RateLimitRule
    ) -> RateLimitStatus:
        """Implement token bucket rate limiting"""
        if not self.redis_client:
            return RateLimitStatus(allowed=True, remaining=0, reset_time=datetime.utcnow())
        
        try:
            now = time.time()
            key = self._get_rate_limit_key(identifier, category)
            
            # Token bucket parameters
            bucket_size = rule.requests + rule.burst_allowance
            refill_rate = rule.requests / rule.window_seconds  # tokens per second
            
            # Get current bucket state
            bucket_data = self.redis_client.hgetall(key)
            
            if bucket_data:
                tokens = float(bucket_data.get("tokens", bucket_size))
                last_refill = float(bucket_data.get("last_refill", now))
            else:
                tokens = bucket_size
                last_refill = now
            
            # Calculate tokens to add
            time_passed = now - last_refill
            tokens_to_add = time_passed * refill_rate
            tokens = min(bucket_size, tokens + tokens_to_add)
            
            # Check if we can consume a token
            if tokens >= 1:
                tokens -= 1
                allowed = True
            else:
                allowed = False
            
            # Update bucket state
            self.redis_client.hset(key, mapping={
                "tokens": tokens,
                "last_refill": now
            })
            self.redis_client.expire(key, rule.window_seconds)
            
            # Calculate reset time
            if tokens == 0:
                time_to_next_token = 1 / refill_rate
                reset_time = datetime.fromtimestamp(now + time_to_next_token)
            else:
                reset_time = datetime.fromtimestamp(now + rule.window_seconds)
            
            return RateLimitStatus(
                allowed=allowed,
                remaining=int(tokens),
                reset_time=reset_time
            )
            
        except Exception as e:
            logger.error(f"Error in token bucket rate limit: {e}")
            return RateLimitStatus(allowed=True, remaining=0, reset_time=datetime.utcnow())
    
    async def _fixed_window_check(
        self,
        identifier: str,
        category: RateLimitCategory,
        rule: RateLimitRule
    ) -> RateLimitStatus:
        """Implement fixed window rate limiting"""
        if not self.redis_client:
            return RateLimitStatus(allowed=True, remaining=0, reset_time=datetime.utcnow())
        
        try:
            now = time.time()
            window_id = int(now // rule.window_seconds)
            key = f"{self._get_rate_limit_key(identifier, category)}:{window_id}"
            
            # Increment counter
            current_count = self.redis_client.incr(key)
            
            # Set expiration on first request in window
            if current_count == 1:
                self.redis_client.expire(key, rule.window_seconds)
            
            # Check if limit exceeded
            allowed_requests = rule.requests + rule.burst_allowance
            remaining = max(0, allowed_requests - current_count)
            
            # Calculate reset time (start of next window)
            next_window = (window_id + 1) * rule.window_seconds
            reset_time = datetime.fromtimestamp(next_window)
            
            return RateLimitStatus(
                allowed=current_count <= allowed_requests,
                remaining=remaining,
                reset_time=reset_time
            )
            
        except Exception as e:
            logger.error(f"Error in fixed window rate limit: {e}")
            return RateLimitStatus(allowed=True, remaining=0, reset_time=datetime.utcnow())
    
    async def _record_violation(
        self,
        ip: str,
        user_id: Optional[int],
        category: RateLimitCategory,
        rule: RateLimitRule,
        user_agent: Optional[str]
    ):
        """Record a rate limit violation"""
        if not self.redis_client:
            return
        
        try:
            violation = RateLimitViolation(
                timestamp=datetime.utcnow(),
                ip_address=ip,
                user_id=user_id,
                category=category,
                requests_made=rule.requests + 1,
                limit=rule.requests,
                user_agent=user_agent,
                blocked_duration_seconds=rule.block_duration_seconds
            )
            
            # Store violation
            violation_key = self._get_violation_key(ip)
            violation_data = json.dumps(asdict(violation), default=str)
            
            self.redis_client.lpush(violation_key, violation_data)
            self.redis_client.ltrim(violation_key, 0, 99)  # Keep last 100 violations
            self.redis_client.expire(violation_key, 86400 * 7)  # 7 days
            
            # Log security event
            logger.warning(
                f"Rate limit violation: IP={ip}, Category={category.value}, "
                f"UserID={user_id}, Requests={rule.requests + 1}, Limit={rule.requests}"
            )
            
        except Exception as e:
            logger.error(f"Error recording violation: {e}")
    
    async def _get_violation_count(self, ip: str) -> int:
        """Get violation count for IP in last 24 hours"""
        if not self.redis_client:
            return 0
        
        try:
            violation_key = self._get_violation_key(ip)
            violations = self.redis_client.lrange(violation_key, 0, -1)
            
            # Count violations in last 24 hours
            cutoff_time = datetime.utcnow() - timedelta(days=1)
            recent_violations = 0
            
            for violation_data in violations:
                try:
                    violation = json.loads(violation_data)
                    violation_time = datetime.fromisoformat(violation["timestamp"])
                    if violation_time > cutoff_time:
                        recent_violations += 1
                except Exception:
                    continue
            
            return recent_violations
            
        except Exception as e:
            logger.error(f"Error getting violation count: {e}")
            return 0
    
    def _apply_adaptive_limiting(self, rule: RateLimitRule, violation_count: int) -> RateLimitRule:
        """Apply adaptive limiting based on violation history"""
        # Reduce limits for problematic IPs
        adaptive_factor = max(0.1, 1.0 - (violation_count * 0.1))
        
        adapted_rule = RateLimitRule(
            requests=max(1, int(rule.requests * adaptive_factor)),
            window_seconds=rule.window_seconds,
            algorithm=rule.algorithm,
            block_duration_seconds=min(3600, rule.block_duration_seconds * 2),  # Double block time
            burst_allowance=max(0, int(rule.burst_allowance * adaptive_factor)),
            adaptive_factor=adaptive_factor
        )
        
        return adapted_rule
    
    async def _apply_block(
        self,
        identifier: str,
        category: RateLimitCategory,
        duration_seconds: int
    ):
        """Apply a temporary block"""
        if not self.redis_client:
            return
        
        try:
            block_key = self._get_block_key(identifier, category)
            blocked_until = datetime.utcnow() + timedelta(seconds=duration_seconds)
            
            block_data = {
                "blocked_until": blocked_until.isoformat(),
                "category": category.value,
                "identifier": identifier
            }
            
            self.redis_client.setex(
                block_key,
                duration_seconds,
                json.dumps(block_data)
            )
            
            logger.info(
                f"Applied rate limit block: {identifier} for {category.value}, "
                f"duration: {duration_seconds}s"
            )
            
        except Exception as e:
            logger.error(f"Error applying block: {e}")


# Global rate limiter instance
_rate_limiter: Optional[AdvancedRateLimiter] = None


def get_rate_limiter() -> AdvancedRateLimiter:
    """Get or create the global rate limiter instance"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = AdvancedRateLimiter()
    return _rate_limiter


# FastAPI dependency for rate limiting
async def rate_limit_dependency(
    request: Request,
    category: RateLimitCategory = RateLimitCategory.API_READ,
    user_id: Optional[int] = None
):
    """FastAPI dependency for rate limiting"""
    rate_limiter = get_rate_limiter()
    rate_status = await rate_limiter.check_rate_limit(request, category, user_id)

    if not rate_status.allowed:
        # Add rate limit headers
        headers = {
            "X-RateLimit-Limit": str(rate_limiter.default_rules[category].requests),
            "X-RateLimit-Remaining": str(rate_status.remaining),
            "X-RateLimit-Reset": str(int(rate_status.reset_time.timestamp()))
        }

        if rate_status.retry_after_seconds:
            headers["Retry-After"] = str(rate_status.retry_after_seconds)

        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers=headers
        )

    return rate_status


# Decorator for rate limiting
def rate_limit(category: RateLimitCategory, custom_rule: Optional[RateLimitRule] = None):
    """Decorator for adding rate limiting to functions"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request from arguments
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            if not request:
                # Look in kwargs
                request = kwargs.get('request')

            if request:
                rate_limiter = get_rate_limiter()
                rate_status = await rate_limiter.check_rate_limit(request, category, None, custom_rule)

                if not rate_status.allowed:
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail="Rate limit exceeded"
                    )

            return await func(*args, **kwargs)
        return wrapper
    return decorator