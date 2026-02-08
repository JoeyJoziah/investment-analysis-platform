"""
Rate Limiter Tests

Comprehensive test suite for the rate limiting modules:
- backend/security/rate_limiter.py (AdvancedRateLimiter, categories, algorithms)
- backend/security/advanced_rate_limiter.py (AdaptiveRateLimiter, ThreatDetector, strategies)

Covers:
- Basic rate limit enforcement across algorithms
- Category-specific limits (AUTH, REGISTRATION, API_READ, etc.)
- Redis fallback behavior (fail-open when Redis is unavailable)
- Rate limit headers in HTTP 429 responses
- Rate limit reset timing and block durations
- Trusted IP bypass
- Adaptive limiting based on violation history
- Threat detection and risk scoring

Created: 2026-02-08
"""

import pytest
import time
import json
from typing import Optional
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock
from dataclasses import asdict

from backend.security.rate_limiter import (
    AdvancedRateLimiter,
    RateLimitAlgorithm,
    RateLimitCategory,
    RateLimitRule,
    RateLimitStatus,
    RateLimitViolation,
    get_rate_limiter,
    rate_limit_dependency,
)
from backend.security.advanced_rate_limiter import (
    AdaptiveRateLimiter,
    RateLimitStrategy,
    ThreatLevel,
    ActionType,
    ThreatDetector,
    RateLimitStorage,
    ClientInfo,
    RequestContext,
    RateLimitRule as AdvancedRateLimitRule,
    RateLimitingMiddleware,
    get_default_rate_limiting_rules,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_request(
    ip: str = "203.0.113.50",
    path: str = "/api/v1/stocks",
    method: str = "GET",
    headers: Optional[dict] = None,
):
    """Build a minimal mock FastAPI Request."""
    mock = MagicMock()
    mock.client.host = ip
    mock.url.path = path
    mock.method = method
    _headers = {
        "user-agent": "Mozilla/5.0 TestAgent",
        "x-forwarded-for": "",
        "x-real-ip": "",
    }
    if headers:
        _headers.update(headers)
    mock.headers = MagicMock()
    mock.headers.get = lambda key, default="": _headers.get(key.lower(), default)
    mock.headers.__contains__ = lambda self_inner, key: key.lower() in _headers
    return mock


def _make_redis_mock(*, connected: bool = True):
    """Return a MagicMock that behaves like a synchronous redis.Redis client."""
    mock = MagicMock()
    if not connected:
        # Simulate Redis being unavailable by returning None for the client
        return None

    # Sorted-set helpers for sliding window
    mock.zremrangebyscore = MagicMock(return_value=0)
    mock.zcard = MagicMock(return_value=0)
    mock.zadd = MagicMock(return_value=1)
    mock.expire = MagicMock(return_value=True)

    # Pipeline support (sliding window uses pipeline)
    pipe = MagicMock()
    pipe.zremrangebyscore = MagicMock()
    pipe.zcard = MagicMock()
    pipe.zadd = MagicMock()
    pipe.expire = MagicMock()
    pipe.execute = MagicMock(return_value=[0, 0, 1, True])  # [removed, count, added, expire_ok]
    mock.pipeline = MagicMock(return_value=pipe)

    # Hash helpers for token bucket
    mock.hgetall = MagicMock(return_value={})
    mock.hset = MagicMock()

    # Key/value helpers for block checking
    mock.get = MagicMock(return_value=None)
    mock.setex = MagicMock()
    mock.delete = MagicMock()
    mock.incr = MagicMock(return_value=1)
    mock.ttl = MagicMock(return_value=-2)  # key does not exist

    # List helpers for violations
    mock.lpush = MagicMock()
    mock.ltrim = MagicMock()
    mock.lrange = MagicMock(return_value=[])

    return mock


# ---------------------------------------------------------------------------
# 1) Basic rate limit enforcement  (AdvancedRateLimiter)
# ---------------------------------------------------------------------------

class TestBasicRateLimitEnforcement:
    """Verify that the sliding-window algorithm blocks requests beyond the limit."""

    async def test_allow_request_within_limit(self):
        """Requests within the configured limit should be allowed."""
        redis_mock = _make_redis_mock()
        limiter = AdvancedRateLimiter(redis_client=redis_mock)
        request = _make_request()

        status = await limiter.check_rate_limit(
            request, RateLimitCategory.API_READ
        )

        assert status.allowed is True
        assert status.remaining >= 0

    async def test_block_request_exceeding_limit(self):
        """When count already equals the limit the next request should be denied."""
        redis_mock = _make_redis_mock()
        # Pipeline returns a count that already equals the rule limit
        auth_rule = RateLimitRule(requests=5, window_seconds=300)
        # Simulate zcard returning 5 (already at limit)
        pipe = redis_mock.pipeline()
        pipe.execute.return_value = [0, 5, 1, True]

        limiter = AdvancedRateLimiter(redis_client=redis_mock)
        request = _make_request()

        status = await limiter.check_rate_limit(
            request, RateLimitCategory.AUTHENTICATION, custom_rule=auth_rule
        )

        assert status.allowed is False
        assert status.remaining == 0

    async def test_remaining_count_decreases(self):
        """Remaining count should reflect how many requests are left."""
        redis_mock = _make_redis_mock()
        pipe = redis_mock.pipeline()
        # 3 requests already consumed out of a limit of 5
        pipe.execute.return_value = [0, 3, 1, True]

        rule = RateLimitRule(requests=5, window_seconds=300)
        limiter = AdvancedRateLimiter(redis_client=redis_mock)
        request = _make_request()

        status = await limiter.check_rate_limit(
            request, RateLimitCategory.AUTHENTICATION, custom_rule=rule
        )

        assert status.allowed is True
        # remaining = max(0, (5 + 0) - 3 - 1) = 1
        assert status.remaining == 1


# ---------------------------------------------------------------------------
# 2) Category-specific rate limits
# ---------------------------------------------------------------------------

class TestRateLimitCategories:
    """Different categories should have different default thresholds."""

    async def test_auth_category_has_strict_limit(self):
        """Authentication category defaults to 5 requests per 5 minutes."""
        redis_mock = _make_redis_mock()
        limiter = AdvancedRateLimiter(redis_client=redis_mock)

        rule = limiter.default_rules[RateLimitCategory.AUTHENTICATION]
        assert rule.requests == 5
        assert rule.window_seconds == 300

    async def test_registration_category_limit(self):
        """Registration category defaults to 5 requests per hour."""
        redis_mock = _make_redis_mock()
        limiter = AdvancedRateLimiter(redis_client=redis_mock)

        rule = limiter.default_rules[RateLimitCategory.REGISTRATION]
        assert rule.requests == 5
        assert rule.window_seconds == 3600

    async def test_api_read_category_has_generous_limit(self):
        """API read category allows 1000 requests per hour with burst."""
        redis_mock = _make_redis_mock()
        limiter = AdvancedRateLimiter(redis_client=redis_mock)

        rule = limiter.default_rules[RateLimitCategory.API_READ]
        assert rule.requests == 1000
        assert rule.burst_allowance == 100

    async def test_password_reset_category(self):
        """Password reset should be tightly limited (3 per hour)."""
        redis_mock = _make_redis_mock()
        limiter = AdvancedRateLimiter(redis_client=redis_mock)

        rule = limiter.default_rules[RateLimitCategory.PASSWORD_RESET]
        assert rule.requests == 3
        assert rule.window_seconds == 3600
        assert rule.block_duration_seconds == 1800

    async def test_all_categories_have_rules(self):
        """Every RateLimitCategory should have a default rule configured."""
        redis_mock = _make_redis_mock()
        limiter = AdvancedRateLimiter(redis_client=redis_mock)

        for category in RateLimitCategory:
            assert category in limiter.default_rules, (
                f"Missing default rule for category {category.value}"
            )


# ---------------------------------------------------------------------------
# 3) Redis fallback behaviour (fail-open)
# ---------------------------------------------------------------------------

class TestRedisFallback:
    """When Redis is unavailable the limiter should fail open (allow requests)."""

    async def test_no_redis_allows_request(self):
        """With redis_client=None every check should return allowed=True."""
        limiter = AdvancedRateLimiter(redis_client=None)
        request = _make_request()

        status = await limiter.check_rate_limit(
            request, RateLimitCategory.AUTHENTICATION
        )

        assert status.allowed is True

    async def test_redis_exception_fails_open(self):
        """If Redis raises an exception the limiter should still allow the request."""
        redis_mock = _make_redis_mock()
        pipe = redis_mock.pipeline()
        pipe.execute.side_effect = Exception("Redis connection lost")

        limiter = AdvancedRateLimiter(redis_client=redis_mock)
        request = _make_request()

        status = await limiter.check_rate_limit(
            request, RateLimitCategory.API_READ
        )

        # The outer try/except in check_rate_limit catches and returns allowed=True
        assert status.allowed is True

    async def test_block_check_without_redis(self):
        """_check_block_status should return allowed when Redis is absent."""
        limiter = AdvancedRateLimiter(redis_client=None)

        status = await limiter._check_block_status("1.2.3.4", RateLimitCategory.AUTHENTICATION)
        assert status.allowed is True

    async def test_violation_count_without_redis(self):
        """_get_violation_count should return 0 when Redis is absent."""
        limiter = AdvancedRateLimiter(redis_client=None)

        count = await limiter._get_violation_count("1.2.3.4")
        assert count == 0


# ---------------------------------------------------------------------------
# 4) Rate limit headers in 429 responses (rate_limit_dependency)
# ---------------------------------------------------------------------------

class TestRateLimitHeaders:
    """The FastAPI dependency should raise HTTPException with standard headers."""

    async def test_dependency_raises_429_with_headers(self):
        """rate_limit_dependency should raise HTTP 429 with rate limit headers."""
        from fastapi import HTTPException

        redis_mock = _make_redis_mock()
        pipe = redis_mock.pipeline()
        # Exceed the default AUTH limit of 5
        pipe.execute.return_value = [0, 5, 1, True]

        limiter = AdvancedRateLimiter(redis_client=redis_mock)
        request = _make_request()

        with patch("backend.security.rate_limiter.get_rate_limiter", return_value=limiter):
            with pytest.raises(HTTPException) as exc_info:
                await rate_limit_dependency(
                    request,
                    category=RateLimitCategory.AUTHENTICATION,
                )

            exc = exc_info.value
            assert exc.status_code == 429
            assert exc.detail == "Rate limit exceeded"
            assert "X-RateLimit-Limit" in exc.headers
            assert "X-RateLimit-Remaining" in exc.headers
            assert "X-RateLimit-Reset" in exc.headers

    async def test_dependency_includes_retry_after_when_blocked(self):
        """When a client is blocked the response should include Retry-After."""
        from fastapi import HTTPException

        blocked_until_dt = datetime.now(timezone.utc) + timedelta(seconds=600)
        blocked_status = RateLimitStatus(
            allowed=False,
            remaining=0,
            reset_time=blocked_until_dt,
            retry_after_seconds=600,
            blocked_until=blocked_until_dt,
        )

        redis_mock = _make_redis_mock()
        limiter = AdvancedRateLimiter(redis_client=redis_mock)
        request = _make_request()

        with patch("backend.security.rate_limiter.get_rate_limiter", return_value=limiter), \
             patch.object(limiter, "check_rate_limit", return_value=blocked_status):
            with pytest.raises(HTTPException) as exc_info:
                await rate_limit_dependency(
                    request,
                    category=RateLimitCategory.AUTHENTICATION,
                )

            exc = exc_info.value
            assert exc.status_code == 429
            assert "Retry-After" in exc.headers
            retry_value = int(exc.headers["Retry-After"])
            assert retry_value > 0

    async def test_dependency_passes_when_allowed(self):
        """When within limits the dependency should return the status, not raise."""
        redis_mock = _make_redis_mock()
        limiter = AdvancedRateLimiter(redis_client=redis_mock)
        request = _make_request()

        with patch("backend.security.rate_limiter.get_rate_limiter", return_value=limiter):
            status = await rate_limit_dependency(
                request,
                category=RateLimitCategory.API_READ,
            )

        assert isinstance(status, RateLimitStatus)
        assert status.allowed is True


# ---------------------------------------------------------------------------
# 5) Rate limit reset timing
# ---------------------------------------------------------------------------

class TestRateLimitResetTiming:
    """Verify that reset_time is set correctly for different algorithms."""

    async def test_sliding_window_reset_in_future(self):
        """Sliding window reset_time should be in the future (local time)."""
        redis_mock = _make_redis_mock()
        limiter = AdvancedRateLimiter(redis_client=redis_mock)
        request = _make_request()

        # The source code uses datetime.fromtimestamp() which returns naive
        # local time, so compare using datetime.now() (also naive local).
        before = datetime.now()
        status = await limiter.check_rate_limit(
            request, RateLimitCategory.API_READ
        )

        assert status.reset_time > before

    async def test_fixed_window_reset_aligns_to_window_boundary(self):
        """Fixed window reset should align to the next window boundary."""
        redis_mock = _make_redis_mock()
        limiter = AdvancedRateLimiter(redis_client=redis_mock)
        request = _make_request()

        rule = RateLimitRule(
            requests=10,
            window_seconds=60,
            algorithm=RateLimitAlgorithm.FIXED_WINDOW,
        )

        status = await limiter.check_rate_limit(
            request, RateLimitCategory.API_READ, custom_rule=rule,
        )

        now_ts = time.time()
        window_id = int(now_ts // 60)
        expected_reset_ts = (window_id + 1) * 60

        # Allow 2-second tolerance for execution time
        assert abs(status.reset_time.timestamp() - expected_reset_ts) < 2

    async def test_block_duration_is_applied(self):
        """When a request is blocked, blocked_until should reflect block_duration_seconds."""
        redis_mock = _make_redis_mock()
        pipe = redis_mock.pipeline()
        # Exceed the auth limit (5 requests)
        pipe.execute.return_value = [0, 5, 1, True]

        limiter = AdvancedRateLimiter(redis_client=redis_mock)
        request = _make_request()

        status = await limiter.check_rate_limit(
            request, RateLimitCategory.AUTHENTICATION,
        )

        assert status.allowed is False
        # AUTH rule has block_duration_seconds=900 (15 minutes)
        # The source sets blocked_until as a naive datetime.
        # Normalize both to naive for comparison.
        if status.blocked_until:
            blocked = status.blocked_until.replace(tzinfo=None)
            now_naive = datetime.now(timezone.utc).replace(tzinfo=None)
            delta = (blocked - now_naive).total_seconds()
            assert 880 < delta < 910


# ---------------------------------------------------------------------------
# 6) Trusted IP bypass
# ---------------------------------------------------------------------------

class TestTrustedIPBypass:
    """Requests from trusted / private IPs should bypass rate limits."""

    async def test_loopback_ip_bypasses_limits(self):
        """127.0.0.1 should always be allowed."""
        redis_mock = _make_redis_mock()
        limiter = AdvancedRateLimiter(redis_client=redis_mock)
        request = _make_request(ip="127.0.0.1")

        status = await limiter.check_rate_limit(
            request, RateLimitCategory.AUTHENTICATION,
        )

        assert status.allowed is True
        assert status.remaining == 999999

    async def test_private_network_bypasses_limits(self):
        """10.x.x.x, 172.16.x.x, 192.168.x.x should bypass."""
        redis_mock = _make_redis_mock()
        limiter = AdvancedRateLimiter(redis_client=redis_mock)

        for ip in ["10.0.0.5", "172.16.0.1", "192.168.1.100"]:
            request = _make_request(ip=ip)
            status = await limiter.check_rate_limit(
                request, RateLimitCategory.AUTHENTICATION,
            )
            assert status.allowed is True, f"Expected {ip} to bypass rate limits"

    async def test_public_ip_is_rate_limited(self):
        """A public IP should be subject to normal rate limits."""
        redis_mock = _make_redis_mock()
        limiter = AdvancedRateLimiter(redis_client=redis_mock)

        assert not limiter._is_trusted_ip("203.0.113.50")


# ---------------------------------------------------------------------------
# 7) Adaptive limiting based on violation history
# ---------------------------------------------------------------------------

class TestAdaptiveLimiting:
    """When an IP has repeated violations, limits should be tightened."""

    def test_adaptive_factor_reduces_requests(self):
        """High violation count should reduce allowed requests."""
        redis_mock = _make_redis_mock()
        limiter = AdvancedRateLimiter(redis_client=redis_mock)

        original_rule = RateLimitRule(requests=100, window_seconds=3600)

        # 5 violations -> adaptive_factor = max(0.1, 1.0 - 0.5) = 0.5
        # Requests = max(1, int(100 * 0.5)) = 50
        adapted = limiter._apply_adaptive_limiting(original_rule, violation_count=5)

        assert adapted.requests < original_rule.requests
        assert adapted.requests == 50
        assert adapted.block_duration_seconds > original_rule.block_duration_seconds

    def test_adaptive_minimum_one_request(self):
        """Even extreme violations should leave at least 1 allowed request."""
        redis_mock = _make_redis_mock()
        limiter = AdvancedRateLimiter(redis_client=redis_mock)

        rule = RateLimitRule(requests=5, window_seconds=300)
        adapted = limiter._apply_adaptive_limiting(rule, violation_count=50)

        assert adapted.requests >= 1

    def test_adaptive_doubles_block_duration(self):
        """Block duration should double but cap at 3600."""
        redis_mock = _make_redis_mock()
        limiter = AdvancedRateLimiter(redis_client=redis_mock)

        rule = RateLimitRule(
            requests=10, window_seconds=300, block_duration_seconds=300
        )
        adapted = limiter._apply_adaptive_limiting(rule, violation_count=6)

        assert adapted.block_duration_seconds == 600  # doubled

    def test_adaptive_block_duration_capped_at_3600(self):
        """Block duration should not exceed 3600 seconds."""
        redis_mock = _make_redis_mock()
        limiter = AdvancedRateLimiter(redis_client=redis_mock)

        rule = RateLimitRule(
            requests=10, window_seconds=300, block_duration_seconds=2000
        )
        adapted = limiter._apply_adaptive_limiting(rule, violation_count=6)

        assert adapted.block_duration_seconds == 3600  # capped


# ---------------------------------------------------------------------------
# 8) ThreatDetector (advanced_rate_limiter.py)
# ---------------------------------------------------------------------------

class TestThreatDetector:
    """Verify threat level assessment and bot detection logic."""

    def test_low_threat_for_normal_client(self):
        """A normal client with good reputation should be LOW threat."""
        detector = ThreatDetector()

        client = ClientInfo(
            ip_address="203.0.113.50",
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            reputation_score=1.0,
        )
        ctx = RequestContext(
            client_info=client,
            endpoint="/api/v1/stocks",
            method="GET",
            timestamp=datetime.now(timezone.utc),
            request_size=256,
        )

        level = detector.assess_threat_level(client, ctx)
        assert level == ThreatLevel.LOW

    def test_critical_threat_for_malicious_agent(self):
        """A Tor exit node with a scanner user-agent should be CRITICAL."""
        detector = ThreatDetector()

        client = ClientInfo(
            ip_address="198.51.100.1",
            user_agent="sqlmap/1.5",
            is_tor=True,
            reputation_score=0.2,
            failed_auth_count=10,
        )
        ctx = RequestContext(
            client_info=client,
            endpoint="/api/v1/admin/users",
            method="POST",
            timestamp=datetime.now(timezone.utc),
            request_size=2 * 1024 * 1024,  # 2MB body
        )

        level = detector.assess_threat_level(client, ctx)
        assert level == ThreatLevel.CRITICAL

    def test_suspicious_user_agent_detection(self):
        """Known scanner user-agents should be flagged as suspicious."""
        detector = ThreatDetector()

        for ua in ["sqlmap/1.0", "Nikto/2.1.5", "Nmap scripting engine"]:
            assert detector._is_suspicious_user_agent(ua) is True

    def test_empty_user_agent_is_suspicious(self):
        """An empty or very short user-agent string should be suspicious."""
        detector = ThreatDetector()

        assert detector._is_suspicious_user_agent("") is True
        assert detector._is_suspicious_user_agent("short") is True

    def test_normal_user_agent_is_not_suspicious(self):
        """Standard browser user-agents should not be flagged."""
        detector = ThreatDetector()

        ua = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        assert detector._is_suspicious_user_agent(ua) is False

    def test_bot_detection_low_variance_intervals(self):
        """Requests with near-identical timing intervals should be flagged as bots."""
        detector = ThreatDetector()

        client = ClientInfo(ip_address="1.2.3.4", user_agent="BotAgent/1.0")
        base_time = datetime.now(timezone.utc)

        # 15 requests exactly 1s apart to the same endpoint
        requests = [
            RequestContext(
                client_info=client,
                endpoint="/api/v1/stocks",
                method="GET",
                timestamp=base_time + timedelta(seconds=i),
            )
            for i in range(15)
        ]

        is_bot = detector.detect_bot_behavior(client, requests)
        assert is_bot is True

    def test_bot_detection_normal_traffic(self):
        """Normal human browsing patterns should not be flagged as bots."""
        detector = ThreatDetector()

        client = ClientInfo(ip_address="1.2.3.4", user_agent="Mozilla/5.0 Normal")
        base_time = datetime.now(timezone.utc)

        # Fewer than 10 requests -- method returns False early
        requests = [
            RequestContext(
                client_info=client,
                endpoint=f"/api/endpoint{i}",
                method="GET",
                timestamp=base_time + timedelta(seconds=i * 5),
            )
            for i in range(8)
        ]

        is_bot = detector.detect_bot_behavior(client, requests)
        assert is_bot is False


# ---------------------------------------------------------------------------
# 9) Advanced rate limiter strategies
# ---------------------------------------------------------------------------

class TestAdvancedRateLimiterStrategies:
    """Test the AdaptiveRateLimiter with different RateLimitStrategy values."""

    def _make_async_storage(self):
        """Create a mock RateLimitStorage."""
        storage = MagicMock(spec=RateLimitStorage)
        storage.is_blocked = AsyncMock(return_value=(False, None))
        storage.increment_counter = AsyncMock(return_value=1)
        storage.get_sliding_window_count = AsyncMock(return_value=0)
        storage.add_sliding_window_request = AsyncMock()
        storage.get_rate_limit_data = AsyncMock(return_value=None)
        storage.set_rate_limit_data = AsyncMock()
        storage.update_client_info = AsyncMock()
        storage.block_key = AsyncMock()
        return storage

    async def test_fixed_window_strategy_allows(self):
        """Fixed window strategy should allow when count is below the limit."""
        storage = self._make_async_storage()
        storage.increment_counter.return_value = 3

        limiter = AdaptiveRateLimiter(storage)

        client = ClientInfo(ip_address="1.2.3.4", user_agent="Mozilla/5.0 Test")
        ctx = RequestContext(
            client_info=client, endpoint="/api/test", method="GET",
            timestamp=datetime.now(timezone.utc),
        )
        rule = AdvancedRateLimitRule(
            name="test_fixed",
            strategy=RateLimitStrategy.FIXED_WINDOW,
            requests_per_window=10,
            window_size_seconds=60,
        )

        allowed, metadata = await limiter.check_rate_limit(client, ctx, rule)

        assert allowed is True
        assert metadata["adaptive_limits"]["requests"] <= 100

    async def test_sliding_window_strategy_blocks_at_limit(self):
        """Sliding window should deny when count reaches the limit."""
        storage = self._make_async_storage()
        # Return count at the limit
        storage.get_sliding_window_count.return_value = 100

        limiter = AdaptiveRateLimiter(storage)

        client = ClientInfo(ip_address="1.2.3.4", user_agent="Mozilla/5.0 Test")
        ctx = RequestContext(
            client_info=client, endpoint="/api/test", method="GET",
            timestamp=datetime.now(timezone.utc),
        )
        rule = AdvancedRateLimitRule(
            name="test_sliding",
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            requests_per_window=100,
            window_size_seconds=60,
        )

        allowed, metadata = await limiter.check_rate_limit(client, ctx, rule)

        assert allowed is False

    async def test_token_bucket_allows_burst(self):
        """Token bucket with burst capacity should allow initial burst requests."""
        storage = self._make_async_storage()
        # No existing bucket data -> starts full
        storage.get_rate_limit_data.return_value = None

        limiter = AdaptiveRateLimiter(storage)

        client = ClientInfo(ip_address="1.2.3.4", user_agent="Mozilla/5.0 Test")
        ctx = RequestContext(
            client_info=client, endpoint="/api/test", method="GET",
            timestamp=datetime.now(timezone.utc),
        )
        rule = AdvancedRateLimitRule(
            name="test_bucket",
            strategy=RateLimitStrategy.TOKEN_BUCKET,
            requests_per_window=10,
            window_size_seconds=60,
            burst_capacity=20,
            refill_rate=2,
        )

        allowed, metadata = await limiter.check_rate_limit(client, ctx, rule)

        assert allowed is True
        assert metadata.get("threat_level") is not None

    async def test_blocked_client_is_denied(self):
        """A blocked client should be denied immediately regardless of strategy."""
        storage = self._make_async_storage()
        storage.is_blocked.return_value = (True, 120)

        limiter = AdaptiveRateLimiter(storage)

        client = ClientInfo(ip_address="1.2.3.4", user_agent="Mozilla/5.0 Test")
        ctx = RequestContext(
            client_info=client, endpoint="/api/test", method="GET",
            timestamp=datetime.now(timezone.utc),
        )
        rule = AdvancedRateLimitRule(
            name="test_blocked",
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            requests_per_window=100,
            window_size_seconds=60,
        )

        allowed, metadata = await limiter.check_rate_limit(client, ctx, rule)

        assert allowed is False
        assert metadata["reason"] == "blocked"
        assert metadata["blocked_for"] == 120


# ---------------------------------------------------------------------------
# 10) Default rules from advanced_rate_limiter
# ---------------------------------------------------------------------------

class TestDefaultRateLimitingRules:
    """Validate the set of default rules returned by get_default_rate_limiting_rules."""

    def test_default_rules_are_returned(self):
        rules = get_default_rate_limiting_rules()
        assert len(rules) >= 5

    def test_auth_endpoints_rule_exists(self):
        rules = get_default_rate_limiting_rules()
        names = [r.name for r in rules]
        assert "auth_endpoints" in names

    def test_auth_endpoints_has_strict_limit(self):
        rules = get_default_rate_limiting_rules()
        auth_rule = next(r for r in rules if r.name == "auth_endpoints")
        assert auth_rule.requests_per_window == 10
        assert auth_rule.window_size_seconds == 900

    def test_global_api_rule_covers_all_endpoints(self):
        rules = get_default_rate_limiting_rules()
        global_rule = next(r for r in rules if r.name == "global_api")
        # No specific endpoints means it applies to all
        assert global_rule.endpoints is None


# ---------------------------------------------------------------------------
# 11) RateLimitingMiddleware rule matching
# ---------------------------------------------------------------------------

class TestMiddlewareRuleMatching:
    """Verify _should_apply_rule correctly matches endpoints."""

    def _make_middleware(self):
        """Construct middleware with mock app and default rules."""
        rules = get_default_rate_limiting_rules()
        mock_app = MagicMock()
        with patch.object(RateLimitingMiddleware, "__init__", lambda self, *a, **kw: None):
            mw = RateLimitingMiddleware.__new__(RateLimitingMiddleware)
            mw.rules = {r.name: r for r in rules}
        return mw

    def test_auth_rule_matches_auth_path(self):
        """Test that auth rule correctly matches authentication paths"""
        mw = self._make_middleware()
        rule = mw.rules["auth_endpoints"]

        # Verify rule has auth endpoints configured
        assert rule.endpoints is not None, "auth_endpoints rule should have endpoints defined"
        assert any("/auth/" in str(ep) for ep in rule.endpoints), "auth_endpoints should include /auth/ paths"

        # The default rule uses /api/auth/* pattern, not /api/v1/auth/*
        # Test with the path that matches the pattern
        request = _make_request(path="/api/auth/login")
        result = mw._should_apply_rule(request, rule)
        assert result is True, f"Expected auth rule to match /api/auth/login. Rule endpoints: {rule.endpoints}"

    def test_auth_rule_does_not_match_stocks_path(self):
        mw = self._make_middleware()
        rule = mw.rules["auth_endpoints"]
        request = _make_request(path="/api/v1/stocks/AAPL")
        assert mw._should_apply_rule(request, rule) is False

    def test_global_rule_matches_any_path(self):
        mw = self._make_middleware()
        rule = mw.rules["global_api"]
        request = _make_request(path="/api/anything")
        assert mw._should_apply_rule(request, rule) is True

    def test_excluded_endpoint_is_skipped(self):
        mw = self._make_middleware()
        rule = AdvancedRateLimitRule(
            name="test_exclude",
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            requests_per_window=10,
            window_size_seconds=60,
            exclude_endpoints=["/api/health"],
        )
        mw.rules["test_exclude"] = rule
        request = _make_request(path="/api/health")
        assert mw._should_apply_rule(request, rule) is False


# ---------------------------------------------------------------------------
# 12) Rate limit 429 response structure (advanced middleware)
# ---------------------------------------------------------------------------

class TestMiddlewareResponseStructure:
    """Verify the JSON body and headers of the 429 response."""

    def test_rate_limit_response_has_correct_structure(self):
        """The 429 response should include error, message, rule, threat_level, timestamp."""
        rules = get_default_rate_limiting_rules()
        mock_app = MagicMock()
        with patch.object(RateLimitingMiddleware, "__init__", lambda self, *a, **kw: None):
            mw = RateLimitingMiddleware.__new__(RateLimitingMiddleware)
            mw.rules = {r.name: r for r in rules}

        metadata = {
            "limit": 10,
            "remaining": 0,
            "reset_time": int(time.time()) + 60,
            "threat_level": "medium",
        }

        response = mw._create_rate_limit_response(rules[1], metadata)

        assert response.status_code == 429
        body = json.loads(response.body.decode())
        assert "error" in body
        assert "message" in body
        assert "rule" in body
        assert "threat_level" in body
        assert "timestamp" in body

    def test_blocked_response_has_retry_after(self):
        """A blocked response should carry a Retry-After header."""
        rules = get_default_rate_limiting_rules()
        with patch.object(RateLimitingMiddleware, "__init__", lambda self, *a, **kw: None):
            mw = RateLimitingMiddleware.__new__(RateLimitingMiddleware)
            mw.rules = {r.name: r for r in rules}

        metadata = {
            "reason": "blocked",
            "blocked_for": 300,
            "threat_level": "high",
        }

        response = mw._create_rate_limit_response(rules[0], metadata)

        assert response.status_code == 429
        assert response.headers.get("retry-after") == "300"


# ---------------------------------------------------------------------------
# 13) Client IP extraction
# ---------------------------------------------------------------------------

class TestClientIPExtraction:
    """Verify correct IP extraction from various proxy headers."""

    async def test_ip_from_x_forwarded_for(self):
        """Should use the first IP in X-Forwarded-For."""
        redis_mock = _make_redis_mock()
        limiter = AdvancedRateLimiter(redis_client=redis_mock)

        request = _make_request(
            ip="10.0.0.1",  # private, would be trusted
            headers={"x-forwarded-for": "198.51.100.5, 10.0.0.1"},
        )

        ip = limiter._get_client_ip(request)
        assert ip == "198.51.100.5"

    async def test_ip_from_x_real_ip(self):
        """Should fall back to X-Real-IP when X-Forwarded-For is absent."""
        redis_mock = _make_redis_mock()
        limiter = AdvancedRateLimiter(redis_client=redis_mock)

        request = _make_request(
            ip="10.0.0.1",
            headers={"x-real-ip": "198.51.100.10"},
        )

        ip = limiter._get_client_ip(request)
        assert ip == "198.51.100.10"

    async def test_ip_falls_back_to_client_host(self):
        """When no proxy headers are present, use request.client.host."""
        redis_mock = _make_redis_mock()
        limiter = AdvancedRateLimiter(redis_client=redis_mock)

        request = _make_request(ip="203.0.113.50")

        ip = limiter._get_client_ip(request)
        assert ip == "203.0.113.50"


# ---------------------------------------------------------------------------
# 14) Token bucket algorithm
# ---------------------------------------------------------------------------

class TestTokenBucketAlgorithm:
    """Direct tests for the token bucket algorithm in rate_limiter.py."""

    async def test_token_bucket_allows_initial_request(self):
        """First request should be allowed when bucket is full."""
        redis_mock = _make_redis_mock()
        # hgetall returns empty dict -> bucket starts full
        redis_mock.hgetall.return_value = {}

        limiter = AdvancedRateLimiter(redis_client=redis_mock)

        rule = RateLimitRule(
            requests=10,
            window_seconds=60,
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
            burst_allowance=5,
        )

        status = await limiter._token_bucket_check(
            "test_user", RateLimitCategory.ADMIN, rule
        )

        assert status.allowed is True
        # bucket_size = 10 + 5 = 15, after consuming one: 14
        assert status.remaining == 14

    async def test_token_bucket_denies_when_empty(self):
        """Request should be denied when no tokens remain."""
        redis_mock = _make_redis_mock()
        redis_mock.hgetall.return_value = {
            "tokens": "0.0",
            "last_refill": str(time.time()),
        }

        limiter = AdvancedRateLimiter(redis_client=redis_mock)

        rule = RateLimitRule(
            requests=10,
            window_seconds=60,
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
        )

        status = await limiter._token_bucket_check(
            "test_user", RateLimitCategory.ADMIN, rule
        )

        assert status.allowed is False
        assert status.remaining == 0


# ---------------------------------------------------------------------------
# 15) Data class / enum completeness
# ---------------------------------------------------------------------------

class TestDataClassesAndEnums:
    """Ensure enums and dataclasses are well-formed."""

    def test_rate_limit_category_values(self):
        expected = {"auth", "api_read", "api_write", "admin",
                    "password_reset", "registration", "file_upload"}
        actual = {c.value for c in RateLimitCategory}
        assert expected == actual

    def test_rate_limit_algorithm_values(self):
        expected = {"token_bucket", "sliding_window", "fixed_window", "adaptive"}
        actual = {a.value for a in RateLimitAlgorithm}
        assert expected == actual

    def test_rate_limit_status_dataclass(self):
        status = RateLimitStatus(
            allowed=True, remaining=5, reset_time=datetime.now(timezone.utc)
        )
        assert status.allowed is True
        assert status.retry_after_seconds is None
        assert status.blocked_until is None

    def test_rate_limit_violation_dataclass(self):
        violation = RateLimitViolation(
            timestamp=datetime.now(timezone.utc),
            ip_address="1.2.3.4",
            user_id=42,
            category=RateLimitCategory.AUTHENTICATION,
            requests_made=6,
            limit=5,
        )
        assert violation.blocked_duration_seconds == 300  # default

    def test_threat_level_ordering(self):
        """ThreatLevel should cover four severity tiers."""
        levels = [t.value for t in ThreatLevel]
        assert "low" in levels
        assert "medium" in levels
        assert "high" in levels
        assert "critical" in levels
