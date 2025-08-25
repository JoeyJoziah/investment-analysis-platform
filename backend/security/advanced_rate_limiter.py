"""
Advanced Rate Limiting and DDoS Protection System
Implements multiple algorithms and adaptive protection mechanisms
"""

import time
import asyncio
import hashlib
import json
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import redis.asyncio as aioredis
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import logging
import ipaddress
from collections import defaultdict, deque
import math
from concurrent.futures import ThreadPoolExecutor
import geoip2.database
import geoip2.errors
import user_agents

logger = logging.getLogger(__name__)


class RateLimitStrategy(str, Enum):
    """Rate limiting strategies"""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"
    ADAPTIVE = "adaptive"


class ThreatLevel(str, Enum):
    """Threat assessment levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ActionType(str, Enum):
    """Actions to take when limits are exceeded"""
    LOG_ONLY = "log_only"
    THROTTLE = "throttle"
    BLOCK_TEMPORARY = "block_temporary"
    BLOCK_PERMANENT = "block_permanent"
    CAPTCHA = "captcha"
    REQUIRE_AUTH = "require_auth"


@dataclass
class RateLimitRule:
    """Rate limiting rule configuration"""
    name: str
    strategy: RateLimitStrategy
    requests_per_window: int
    window_size_seconds: int
    burst_capacity: Optional[int] = None  # For token bucket
    refill_rate: Optional[int] = None     # For token bucket
    block_duration_seconds: int = 300     # 5 minutes default
    action: ActionType = ActionType.BLOCK_TEMPORARY
    applies_to: List[str] = None          # IP, user_id, API key, etc.
    endpoints: List[str] = None           # Specific endpoints
    exclude_endpoints: List[str] = None   # Endpoints to exclude


@dataclass
class ClientInfo:
    """Client information for threat assessment"""
    ip_address: str
    user_agent: str
    user_id: Optional[str] = None
    api_key_id: Optional[str] = None
    country: Optional[str] = None
    is_tor: bool = False
    is_vpn: bool = False
    is_proxy: bool = False
    reputation_score: float = 1.0  # 0.0 = bad, 1.0 = good
    first_seen: Optional[datetime] = None
    request_count: int = 0
    failed_auth_count: int = 0


@dataclass
class RequestContext:
    """Request context for rate limiting"""
    client_info: ClientInfo
    endpoint: str
    method: str
    timestamp: datetime
    request_size: int = 0
    response_time: Optional[float] = None


class ThreatDetector:
    """Advanced threat detection system"""
    
    def __init__(self):
        self.suspicious_patterns = [
            # Common attack patterns
            r"(?i)(union|select|insert|delete|update|drop|create|alter)",
            r"(?i)(script|javascript|vbscript|onload|onerror|onclick)",
            r"(?i)(eval|exec|system|cmd|shell)",
            r"\.\.\/|\.\.\\",  # Directory traversal
            r"<iframe|<script|<object|<embed",  # XSS attempts
        ]
        
        # Known malicious User-Agent patterns
        self.malicious_user_agents = [
            "sqlmap", "nikto", "nmap", "masscan", "zmap",
            "burp", "w3af", "acunetix", "appscan", "netsparker"
        ]
        
        # Suspicious countries (can be configured)
        self.high_risk_countries = {"CN", "RU", "IR", "KP"}
        
        self.executor = ThreadPoolExecutor(max_workers=2)
    
    def assess_threat_level(self, client_info: ClientInfo, request_context: RequestContext) -> ThreatLevel:
        """Assess threat level based on multiple factors"""
        risk_score = 0.0
        
        # Geographic risk
        if client_info.country in self.high_risk_countries:
            risk_score += 0.3
        
        # Anonymization services
        if client_info.is_tor:
            risk_score += 0.4
        elif client_info.is_vpn or client_info.is_proxy:
            risk_score += 0.2
        
        # User agent analysis
        if self._is_suspicious_user_agent(client_info.user_agent):
            risk_score += 0.5
        
        # Request patterns
        if client_info.failed_auth_count > 5:
            risk_score += 0.4
        
        # Reputation score
        risk_score += (1.0 - client_info.reputation_score) * 0.3
        
        # Request size (potential for DoS)
        if request_context.request_size > 1024 * 1024:  # 1MB
            risk_score += 0.2
        
        # Convert to threat level
        if risk_score >= 0.8:
            return ThreatLevel.CRITICAL
        elif risk_score >= 0.6:
            return ThreatLevel.HIGH
        elif risk_score >= 0.4:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW
    
    def _is_suspicious_user_agent(self, user_agent: str) -> bool:
        """Check if user agent is suspicious"""
        if not user_agent or len(user_agent) < 10:
            return True
        
        user_agent_lower = user_agent.lower()
        return any(pattern in user_agent_lower for pattern in self.malicious_user_agents)
    
    def detect_bot_behavior(self, client_info: ClientInfo, recent_requests: List[RequestContext]) -> bool:
        """Detect bot-like behavior patterns"""
        if len(recent_requests) < 10:
            return False
        
        # Check for consistent timing patterns
        intervals = []
        for i in range(1, len(recent_requests)):
            interval = (recent_requests[i].timestamp - recent_requests[i-1].timestamp).total_seconds()
            intervals.append(interval)
        
        # Calculate variance in request intervals
        if intervals:
            avg_interval = sum(intervals) / len(intervals)
            variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
            
            # Very low variance suggests bot behavior
            if variance < 0.1 and avg_interval < 2.0:
                return True
        
        # Check for repetitive endpoints
        endpoints = [req.endpoint for req in recent_requests[-20:]]
        unique_endpoints = set(endpoints)
        
        # If accessing only 1-2 endpoints repeatedly, likely a bot
        if len(unique_endpoints) <= 2 and len(endpoints) > 15:
            return True
        
        return False


class RateLimitStorage:
    """Redis-based storage for rate limiting data"""
    
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self._redis: Optional[aioredis.Redis] = None
    
    async def get_redis(self) -> aioredis.Redis:
        """Get Redis connection"""
        if not self._redis:
            self._redis = aioredis.from_url(self.redis_url)
        return self._redis
    
    async def get_rate_limit_data(self, key: str) -> Optional[Dict]:
        """Get rate limit data for key"""
        redis = await self.get_redis()
        data = await redis.get(f"rate_limit:{key}")
        return json.loads(data) if data else None
    
    async def set_rate_limit_data(self, key: str, data: Dict, ttl: int):
        """Set rate limit data with TTL"""
        redis = await self.get_redis()
        await redis.setex(f"rate_limit:{key}", ttl, json.dumps(data))
    
    async def increment_counter(self, key: str, window: int) -> int:
        """Increment counter and set TTL if new"""
        redis = await self.get_redis()
        counter_key = f"counter:{key}:{window}"
        
        pipe = redis.pipeline()
        pipe.incr(counter_key)
        pipe.expire(counter_key, window)
        results = await pipe.execute()
        
        return results[0]
    
    async def get_sliding_window_count(self, key: str, window_seconds: int) -> int:
        """Get count for sliding window"""
        redis = await self.get_redis()
        now = time.time()
        start_time = now - window_seconds
        
        # Use sorted set for sliding window
        set_key = f"sliding:{key}"
        
        # Remove old entries
        await redis.zremrangebyscore(set_key, 0, start_time)
        
        # Get current count
        count = await redis.zcard(set_key)
        return count
    
    async def add_sliding_window_request(self, key: str, window_seconds: int):
        """Add request to sliding window"""
        redis = await self.get_redis()
        now = time.time()
        set_key = f"sliding:{key}"
        
        # Add current request
        await redis.zadd(set_key, {str(now): now})
        
        # Set expiry
        await redis.expire(set_key, window_seconds + 1)
    
    async def is_blocked(self, key: str) -> Tuple[bool, Optional[int]]:
        """Check if key is blocked and get remaining time"""
        redis = await self.get_redis()
        block_key = f"blocked:{key}"
        ttl = await redis.ttl(block_key)
        
        if ttl > 0:
            return True, ttl
        elif ttl == 0:  # Key exists but no TTL
            await redis.delete(block_key)  # Clean up
        
        return False, None
    
    async def block_key(self, key: str, duration_seconds: int, reason: str):
        """Block a key for specified duration"""
        redis = await self.get_redis()
        block_key = f"blocked:{key}"
        block_data = {
            "blocked_at": time.time(),
            "duration": duration_seconds,
            "reason": reason
        }
        await redis.setex(block_key, duration_seconds, json.dumps(block_data))
    
    async def get_client_info(self, identifier: str) -> Optional[ClientInfo]:
        """Get stored client information"""
        redis = await self.get_redis()
        data = await redis.get(f"client:{identifier}")
        if data:
            client_data = json.loads(data)
            return ClientInfo(**client_data)
        return None
    
    async def update_client_info(self, client_info: ClientInfo, ttl: int = 86400):
        """Update client information"""
        redis = await self.get_redis()
        await redis.setex(
            f"client:{client_info.ip_address}",
            ttl,
            json.dumps(asdict(client_info), default=str)
        )


class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts based on system load and threat level"""
    
    def __init__(self, storage: RateLimitStorage):
        self.storage = storage
        self.threat_detector = ThreatDetector()
        
        # Base rate limits by threat level
        self.base_limits = {
            ThreatLevel.LOW: {"requests": 100, "window": 60},
            ThreatLevel.MEDIUM: {"requests": 50, "window": 60},
            ThreatLevel.HIGH: {"requests": 20, "window": 60},
            ThreatLevel.CRITICAL: {"requests": 5, "window": 60}
        }
        
        # System load factors (would be updated by monitoring system)
        self.system_load_factor = 1.0  # 1.0 = normal, >1.0 = high load
    
    async def check_rate_limit(
        self,
        client_info: ClientInfo,
        request_context: RequestContext,
        rule: RateLimitRule
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if request should be rate limited"""
        
        # Assess threat level
        threat_level = self.threat_detector.assess_threat_level(client_info, request_context)
        
        # Check if blocked
        block_key = self._get_block_key(client_info, rule)
        is_blocked, block_ttl = await self.storage.is_blocked(block_key)
        
        if is_blocked:
            return False, {
                "reason": "blocked",
                "blocked_for": block_ttl,
                "threat_level": threat_level.value
            }
        
        # Get adaptive limits
        adaptive_limits = self._calculate_adaptive_limits(threat_level, rule)
        
        # Apply rate limiting strategy
        allowed, metadata = await self._apply_rate_limit_strategy(
            client_info, request_context, rule, adaptive_limits
        )
        
        # Update client info
        await self._update_client_statistics(client_info, request_context, allowed)
        
        # Take action if limit exceeded
        if not allowed:
            await self._handle_rate_limit_exceeded(
                client_info, request_context, rule, threat_level, metadata
            )
        
        metadata.update({
            "threat_level": threat_level.value,
            "adaptive_limits": adaptive_limits
        })
        
        return allowed, metadata
    
    def _calculate_adaptive_limits(self, threat_level: ThreatLevel, rule: RateLimitRule) -> Dict[str, int]:
        """Calculate adaptive limits based on threat level and system load"""
        base_limit = self.base_limits[threat_level]
        
        # Apply system load factor
        adjusted_requests = int(base_limit["requests"] / self.system_load_factor)
        
        # Ensure minimum limits
        adjusted_requests = max(adjusted_requests, 1)
        
        # Use rule limits if more restrictive
        if rule.requests_per_window < adjusted_requests:
            adjusted_requests = rule.requests_per_window
        
        return {
            "requests": adjusted_requests,
            "window": rule.window_size_seconds
        }
    
    async def _apply_rate_limit_strategy(
        self,
        client_info: ClientInfo,
        request_context: RequestContext,
        rule: RateLimitRule,
        limits: Dict[str, int]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Apply specific rate limiting strategy"""
        
        if rule.strategy == RateLimitStrategy.FIXED_WINDOW:
            return await self._fixed_window_check(client_info, rule, limits)
        
        elif rule.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return await self._sliding_window_check(client_info, rule, limits)
        
        elif rule.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return await self._token_bucket_check(client_info, rule, limits)
        
        elif rule.strategy == RateLimitStrategy.LEAKY_BUCKET:
            return await self._leaky_bucket_check(client_info, rule, limits)
        
        else:  # Default to sliding window
            return await self._sliding_window_check(client_info, rule, limits)
    
    async def _fixed_window_check(
        self,
        client_info: ClientInfo,
        rule: RateLimitRule,
        limits: Dict[str, int]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Fixed window rate limiting"""
        window = int(time.time() // limits["window"])
        key = f"{rule.name}:{client_info.ip_address}:{window}"
        
        count = await self.storage.increment_counter(key, limits["window"])
        
        allowed = count <= limits["requests"]
        remaining = max(0, limits["requests"] - count)
        
        return allowed, {
            "strategy": "fixed_window",
            "count": count,
            "limit": limits["requests"],
            "remaining": remaining,
            "reset_time": (window + 1) * limits["window"]
        }
    
    async def _sliding_window_check(
        self,
        client_info: ClientInfo,
        rule: RateLimitRule,
        limits: Dict[str, int]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Sliding window rate limiting"""
        key = f"{rule.name}:{client_info.ip_address}"
        
        count = await self.storage.get_sliding_window_count(key, limits["window"])
        allowed = count < limits["requests"]
        
        if allowed:
            await self.storage.add_sliding_window_request(key, limits["window"])
            count += 1
        
        remaining = max(0, limits["requests"] - count)
        
        return allowed, {
            "strategy": "sliding_window",
            "count": count,
            "limit": limits["requests"],
            "remaining": remaining,
            "window_size": limits["window"]
        }
    
    async def _token_bucket_check(
        self,
        client_info: ClientInfo,
        rule: RateLimitRule,
        limits: Dict[str, int]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Token bucket rate limiting"""
        key = f"{rule.name}:{client_info.ip_address}"
        
        bucket_data = await self.storage.get_rate_limit_data(key)
        now = time.time()
        
        if not bucket_data:
            bucket_data = {
                "tokens": rule.burst_capacity or limits["requests"],
                "last_refill": now
            }
        else:
            # Refill tokens
            time_passed = now - bucket_data["last_refill"]
            refill_rate = rule.refill_rate or (limits["requests"] / limits["window"])
            new_tokens = time_passed * refill_rate
            
            bucket_data["tokens"] = min(
                rule.burst_capacity or limits["requests"],
                bucket_data["tokens"] + new_tokens
            )
            bucket_data["last_refill"] = now
        
        # Check if token available
        allowed = bucket_data["tokens"] >= 1
        if allowed:
            bucket_data["tokens"] -= 1
        
        # Store updated bucket
        await self.storage.set_rate_limit_data(key, bucket_data, limits["window"] * 2)
        
        return allowed, {
            "strategy": "token_bucket",
            "tokens": bucket_data["tokens"],
            "capacity": rule.burst_capacity or limits["requests"]
        }
    
    async def _leaky_bucket_check(
        self,
        client_info: ClientInfo,
        rule: RateLimitRule,
        limits: Dict[str, int]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Leaky bucket rate limiting"""
        key = f"{rule.name}:{client_info.ip_address}"
        
        bucket_data = await self.storage.get_rate_limit_data(key)
        now = time.time()
        
        if not bucket_data:
            bucket_data = {
                "queue_size": 0,
                "last_leak": now
            }
        else:
            # Leak requests
            time_passed = now - bucket_data["last_leak"]
            leak_rate = limits["requests"] / limits["window"]
            leaked = int(time_passed * leak_rate)
            
            bucket_data["queue_size"] = max(0, bucket_data["queue_size"] - leaked)
            bucket_data["last_leak"] = now
        
        # Check if can accept request
        capacity = rule.burst_capacity or limits["requests"]
        allowed = bucket_data["queue_size"] < capacity
        
        if allowed:
            bucket_data["queue_size"] += 1
        
        # Store updated bucket
        await self.storage.set_rate_limit_data(key, bucket_data, limits["window"] * 2)
        
        return allowed, {
            "strategy": "leaky_bucket",
            "queue_size": bucket_data["queue_size"],
            "capacity": capacity
        }
    
    def _get_block_key(self, client_info: ClientInfo, rule: RateLimitRule) -> str:
        """Generate block key for client"""
        if client_info.user_id:
            return f"{rule.name}:user:{client_info.user_id}"
        elif client_info.api_key_id:
            return f"{rule.name}:api_key:{client_info.api_key_id}"
        else:
            return f"{rule.name}:ip:{client_info.ip_address}"
    
    async def _handle_rate_limit_exceeded(
        self,
        client_info: ClientInfo,
        request_context: RequestContext,
        rule: RateLimitRule,
        threat_level: ThreatLevel,
        metadata: Dict[str, Any]
    ):
        """Handle rate limit exceeded based on rule action"""
        
        if rule.action == ActionType.LOG_ONLY:
            logger.warning(f"Rate limit exceeded for {client_info.ip_address} on {rule.name}")
        
        elif rule.action in [ActionType.BLOCK_TEMPORARY, ActionType.BLOCK_PERMANENT]:
            block_duration = rule.block_duration_seconds
            
            # Escalate block duration for higher threats
            if threat_level == ThreatLevel.HIGH:
                block_duration *= 2
            elif threat_level == ThreatLevel.CRITICAL:
                block_duration *= 5
            
            if rule.action == ActionType.BLOCK_PERMANENT:
                block_duration = 86400 * 365  # 1 year
            
            block_key = self._get_block_key(client_info, rule)
            reason = f"Rate limit exceeded: {rule.name} ({threat_level.value} threat)"
            
            await self.storage.block_key(block_key, block_duration, reason)
            
            logger.warning(f"Blocked {client_info.ip_address} for {block_duration}s: {reason}")
    
    async def _update_client_statistics(
        self,
        client_info: ClientInfo,
        request_context: RequestContext,
        allowed: bool
    ):
        """Update client statistics for reputation scoring"""
        client_info.request_count += 1
        
        if not allowed:
            # Decrease reputation for rate limit violations
            client_info.reputation_score = max(0.0, client_info.reputation_score - 0.05)
        else:
            # Slowly improve reputation for good behavior
            client_info.reputation_score = min(1.0, client_info.reputation_score + 0.001)
        
        await self.storage.update_client_info(client_info)


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for advanced rate limiting and DDoS protection"""
    
    def __init__(self, app, rules: List[RateLimitRule], redis_url: str):
        super().__init__(app)
        self.rules = {rule.name: rule for rule in rules}
        self.storage = RateLimitStorage(redis_url)
        self.rate_limiter = AdaptiveRateLimiter(self.storage)
        
        # GeoIP database (optional)
        try:
            self.geoip_reader = geoip2.database.Reader('/usr/share/GeoIP/GeoLite2-Country.mmdb')
        except:
            self.geoip_reader = None
            logger.warning("GeoIP database not available")
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request through rate limiting pipeline"""
        start_time = time.time()
        
        try:
            # Skip rate limiting for health checks
            if request.url.path in ["/api/health", "/api/metrics"]:
                return await call_next(request)
            
            # Extract client information
            client_info = await self._extract_client_info(request)
            
            # Create request context
            request_context = RequestContext(
                client_info=client_info,
                endpoint=request.url.path,
                method=request.method,
                timestamp=datetime.utcnow(),
                request_size=len(await request.body()) if request.method in ["POST", "PUT", "PATCH"] else 0
            )
            
            # Apply rate limiting rules
            for rule_name, rule in self.rules.items():
                if self._should_apply_rule(request, rule):
                    allowed, metadata = await self.rate_limiter.check_rate_limit(
                        client_info, request_context, rule
                    )
                    
                    if not allowed:
                        return self._create_rate_limit_response(rule, metadata)
            
            # Process request
            response = await call_next(request)
            
            # Update request context with response time
            request_context.response_time = time.time() - start_time
            
            return response
            
        except Exception as e:
            logger.error(f"Rate limiting middleware error: {e}")
            return await call_next(request)
    
    async def _extract_client_info(self, request: Request) -> ClientInfo:
        """Extract client information from request"""
        ip_address = self._get_client_ip(request)
        user_agent = request.headers.get("User-Agent", "")
        
        # Get stored client info or create new
        client_info = await self.storage.get_client_info(ip_address)
        
        if not client_info:
            # Create new client info
            client_info = ClientInfo(
                ip_address=ip_address,
                user_agent=user_agent,
                first_seen=datetime.utcnow(),
                country=self._get_country(ip_address),
                is_tor=self._is_tor_exit_node(ip_address),
                is_vpn=False,  # Would integrate with VPN detection service
                is_proxy=False,  # Would integrate with proxy detection service
                reputation_score=1.0
            )
        else:
            # Update user agent if different
            if client_info.user_agent != user_agent:
                client_info.user_agent = user_agent
        
        # Extract user ID from JWT token if available
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            try:
                # Would decode JWT to get user_id
                # client_info.user_id = decode_jwt_user_id(auth_header[7:])
                pass
            except:
                pass
        
        return client_info
    
    def _get_client_ip(self, request: Request) -> str:
        """Get real client IP considering proxies"""
        # Check common proxy headers in order of preference
        headers_to_check = [
            "CF-Connecting-IP",      # Cloudflare
            "X-Real-IP",            # Nginx
            "X-Forwarded-For",      # Standard proxy header
            "X-Client-IP",          # Alternative
            "X-Cluster-Client-IP",  # Cluster environments
        ]
        
        for header in headers_to_check:
            ip = request.headers.get(header)
            if ip:
                # X-Forwarded-For can contain multiple IPs
                if "," in ip:
                    ip = ip.split(",")[0].strip()
                
                # Validate IP address
                try:
                    ipaddress.ip_address(ip)
                    return ip
                except ValueError:
                    continue
        
        # Fall back to direct connection IP
        return request.client.host
    
    def _get_country(self, ip_address: str) -> Optional[str]:
        """Get country code for IP address"""
        if not self.geoip_reader:
            return None
        
        try:
            response = self.geoip_reader.country(ip_address)
            return response.country.iso_code
        except geoip2.errors.AddressNotFoundError:
            return None
        except Exception as e:
            logger.error(f"GeoIP lookup error: {e}")
            return None
    
    def _is_tor_exit_node(self, ip_address: str) -> bool:
        """Check if IP is a known Tor exit node"""
        # This would integrate with Tor exit node lists
        # For now, return False
        return False
    
    def _should_apply_rule(self, request: Request, rule: RateLimitRule) -> bool:
        """Check if rule should be applied to this request"""
        path = request.url.path
        
        # Check excluded endpoints
        if rule.exclude_endpoints:
            for exclude_pattern in rule.exclude_endpoints:
                if path.startswith(exclude_pattern):
                    return False
        
        # Check included endpoints
        if rule.endpoints:
            for endpoint_pattern in rule.endpoints:
                if endpoint_pattern.endswith("/*"):
                    if path.startswith(endpoint_pattern[:-2]):
                        return True
                elif path == endpoint_pattern:
                    return True
            return False
        
        # Apply to all endpoints by default
        return True
    
    def _create_rate_limit_response(self, rule: RateLimitRule, metadata: Dict[str, Any]) -> JSONResponse:
        """Create rate limit exceeded response"""
        
        if metadata.get("reason") == "blocked":
            status_code = status.HTTP_429_TOO_MANY_REQUESTS
            message = "You have been temporarily blocked due to suspicious activity"
            headers = {
                "Retry-After": str(metadata.get("blocked_for", 300))
            }
        else:
            status_code = status.HTTP_429_TOO_MANY_REQUESTS
            message = "Rate limit exceeded"
            headers = {
                "X-RateLimit-Limit": str(metadata.get("limit", "N/A")),
                "X-RateLimit-Remaining": str(metadata.get("remaining", 0)),
                "X-RateLimit-Reset": str(metadata.get("reset_time", int(time.time() + 60))),
                "Retry-After": "60"
            }
        
        return JSONResponse(
            status_code=status_code,
            content={
                "error": "Rate limit exceeded",
                "message": message,
                "rule": rule.name,
                "threat_level": metadata.get("threat_level", "unknown"),
                "timestamp": datetime.utcnow().isoformat()
            },
            headers=headers
        )


# Default rate limiting rules for the investment platform
def get_default_rate_limiting_rules() -> List[RateLimitRule]:
    """Get default rate limiting rules for the investment platform"""
    
    return [
        # Global API rate limit
        RateLimitRule(
            name="global_api",
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            requests_per_window=1000,
            window_size_seconds=3600,  # 1 hour
            action=ActionType.BLOCK_TEMPORARY,
            block_duration_seconds=3600
        ),
        
        # Authentication endpoints - strict limits
        RateLimitRule(
            name="auth_endpoints",
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            requests_per_window=10,
            window_size_seconds=900,  # 15 minutes
            action=ActionType.BLOCK_TEMPORARY,
            block_duration_seconds=3600,
            endpoints=["/api/auth/*"]
        ),
        
        # Stock data endpoints - moderate limits
        RateLimitRule(
            name="stock_data",
            strategy=RateLimitStrategy.TOKEN_BUCKET,
            requests_per_window=100,
            window_size_seconds=60,
            burst_capacity=20,
            refill_rate=2,  # 2 requests per second
            action=ActionType.THROTTLE,
            endpoints=["/api/stocks/*", "/api/analysis/*"]
        ),
        
        # Portfolio endpoints - per user limits
        RateLimitRule(
            name="portfolio_operations",
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            requests_per_window=50,
            window_size_seconds=300,  # 5 minutes
            action=ActionType.BLOCK_TEMPORARY,
            block_duration_seconds=600,
            endpoints=["/api/portfolio/*"]
        ),
        
        # WebSocket connections
        RateLimitRule(
            name="websocket_connections",
            strategy=RateLimitStrategy.FIXED_WINDOW,
            requests_per_window=5,
            window_size_seconds=3600,  # 1 hour
            action=ActionType.BLOCK_TEMPORARY,
            block_duration_seconds=3600,
            endpoints=["/api/ws/*"]
        ),
        
        # Admin endpoints - very strict
        RateLimitRule(
            name="admin_endpoints",
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            requests_per_window=20,
            window_size_seconds=3600,  # 1 hour
            action=ActionType.BLOCK_PERMANENT,
            endpoints=["/api/admin/*"]
        )
    ]