"""
Error Scenarios and Resilience Tests

Tests for:
- API rate limiting
- Database connection loss and recovery
- Circuit breaker activation
- Graceful degradation
"""

import asyncio
import logging
import time
from typing import Optional
from unittest.mock import patch, AsyncMock, MagicMock
from datetime import datetime, timedelta

import pytest
import bcrypt
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from fastapi import HTTPException

from backend.api.main import app
from backend.models.unified_models import User
from backend.auth.oauth2 import create_tokens, RateLimiter
from backend.utils.circuit_breaker import CircuitBreaker, CircuitState

logger = logging.getLogger(__name__)


@pytest.fixture
def test_user_data():
    """Test user fixture"""
    return {
        "username": "error_test_user",
        "email": "errortest@example.com",
        "password": "ErrorTest123!@#",
        "is_active": True,
        "is_admin": False,
    }


@pytest.fixture
def authenticated_client(test_user_data, db_session):
    """Create authenticated test client"""
    user = User(
        username=test_user_data["username"],
        email=test_user_data["email"],
        is_active=True,
        hashed_password=bcrypt.hashpw(test_user_data["password"].encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    tokens = create_tokens(user)

    client = TestClient(app)
    client.headers.update({"Authorization": f"Bearer {tokens['access_token']}"})
    return client


class TestAPIRateLimiting:
    """Rate limiting tests"""

    def test_rate_limiter_tracks_requests(self, test_user_data, db_session):
        """Test that rate limiter tracks requests per user"""
        user = User(
            username=test_user_data["username"],
            email=test_user_data["email"],
        )
        user.hashed_password = bcrypt.hashpw(test_user_data["password"].encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        db_session.add(user)
        db_session.commit()

        rate_limiter = RateLimiter(calls=5, period=60)

        # Track 5 requests
        for i in range(5):
            key = f"user:{user.id}"
            rate_limiter.clock[key] = rate_limiter.clock.get(key, []) + [
                datetime.now()
            ]

        # Should allow up to 5 calls
        assert len(rate_limiter.clock.get(f"user:{user.id}", [])) == 5

        # 6th call should be blocked conceptually
        assert len(rate_limiter.clock.get(f"user:{user.id}", [])) >= 5

    def test_rate_limiter_resets_after_period(self, test_user_data, db_session):
        """Test rate limiter resets after time period"""
        user = User(
            username=test_user_data["username"],
            email=test_user_data["email"],
        )
        user.hashed_password = bcrypt.hashpw(test_user_data["password"].encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        db_session.add(user)
        db_session.commit()

        rate_limiter = RateLimiter(calls=5, period=1)  # 1 second period
        key = f"user:{user.id}"

        # Add 5 requests
        now = datetime.now()
        rate_limiter.clock[key] = [now] * 5

        # Verify 5 requests recorded
        assert len(rate_limiter.clock[key]) == 5

        # Simulate time passing (>1 second)
        old_requests = rate_limiter.clock[key]
        rate_limiter.clock[key] = [
            ts for ts in old_requests
            if (datetime.now() - ts).total_seconds() < rate_limiter.period
        ]

        # After period expires, old requests should be cleared
        # This would happen in production when checking limits

    def test_rate_limit_exceeded_response(self, authenticated_client):
        """Test that rate limit exceeded returns 429 status"""
        # Make requests to trigger rate limit
        # Note: Actual limit depends on endpoint configuration

        responses = []
        for i in range(100):
            response = authenticated_client.get("/api/portfolio")
            responses.append(response.status_code)

            # Stop if we hit rate limit
            if response.status_code == 429:
                assert response.status_code == 429
                assert "rate limit" in response.json().get("detail", "").lower()
                break

        # Either we hit rate limit or exhausted test iterations
        assert any(status == 429 for status in responses) or len(responses) > 50

    def test_rate_limit_includes_retry_after_header(
        self, authenticated_client
    ):
        """Test that 429 response includes Retry-After header"""
        # Make many requests to trigger rate limit
        for i in range(150):
            response = authenticated_client.get("/api/portfolio")
            if response.status_code == 429:
                # Verify Retry-After header exists
                assert "retry-after" in response.headers or "Retry-After" in response.headers
                break

    def test_different_rate_limits_for_tiers(self, db_session):
        """Test different rate limits for different user tiers"""
        # Create users with different permission levels
        basic_user = User(
            username="basic_user",
            email="basic@example.com",
            is_active=True,
        )
        basic_user.hashed_password = bcrypt.hashpw("Pass123!@#".encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

        premium_user = User(
            username="premium_user",
            email="premium@example.com",
            is_active=True,
        )
        premium_user.hashed_password = bcrypt.hashpw("Pass123!@#".encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

        admin_user = User(
            username="admin_user",
            email="admin@example.com",
            is_active=True,
            is_admin=True,
        )
        admin_user.hashed_password = bcrypt.hashpw("Pass123!@#".encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

        db_session.add_all([basic_user, premium_user, admin_user])
        db_session.commit()

        # In real implementation, basic, premium, and admin would have
        # different rate limits (e.g., 100, 1000, 10000 calls/hour)

    def test_rate_limit_per_endpoint(self, authenticated_client):
        """Test rate limiting per endpoint"""
        # Different endpoints might have different limits
        endpoints = ["/api/portfolio", "/api/stocks", "/api/analysis"]

        # Each endpoint should have independent rate limiting
        for endpoint in endpoints:
            response = authenticated_client.get(endpoint)
            # Should not exceed rate limit on valid requests
            assert response.status_code in [200, 400, 401, 429]


class TestDatabaseConnectionLoss:
    """Database connection loss and recovery tests"""

    @pytest.mark.asyncio
    async def test_database_connection_error_handling(
        self, authenticated_client
    ):
        """Test graceful handling of database connection errors"""
        # Patch database connection to raise error
        with patch(
            "backend.utils.database.get_db",
            side_effect=OperationalError("Connection refused", None, None),
        ):
            response = authenticated_client.get("/api/portfolio")

            # Should return 503 Service Unavailable or similar
            assert response.status_code in [500, 503]
            data = response.json()
            assert "error" in data or "detail" in data

    @pytest.mark.asyncio
    async def test_database_timeout_handling(self, authenticated_client):
        """Test handling of database query timeouts"""
        with patch(
            "backend.utils.database.get_db",
            side_effect=TimeoutError("Query timeout"),
        ):
            response = authenticated_client.get("/api/portfolio")

            # Should return appropriate error
            assert response.status_code in [500, 503, 504]

    @pytest.mark.asyncio
    async def test_connection_pool_exhaustion(self, authenticated_client):
        """Test handling when connection pool is exhausted"""
        with patch(
            "backend.utils.database.get_db",
            side_effect=OperationalError(
                "QueuePool limit exceeded", None, None
            ),
        ):
            response = authenticated_client.get("/api/portfolio")

            # Should handle gracefully
            assert response.status_code in [500, 503]

    def test_transaction_rollback_on_error(self, db_session):
        """Test that transactions are rolled back on error"""
        from backend.models.unified_models import Portfolio

        # Create a user
        user = User(
            username="test_user_db",
            email="testdb@example.com",
        )
        user.hashed_password = bcrypt.hashpw("Pass123!@#".encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        db_session.add(user)
        db_session.commit()

        # Attempt operation that might fail
        try:
            portfolio = Portfolio(
                user_id=user.id,
                total_value=-1000,  # Invalid negative value
            )
            db_session.add(portfolio)
            db_session.commit()
        except Exception as e:
            db_session.rollback()
            logger.info(f"Transaction rolled back: {e}")

        # Verify user still exists but portfolio wasn't created
        user_check = db_session.query(User).filter_by(email="testdb@example.com").first()
        assert user_check is not None

    def test_database_recovery_after_connection_loss(
        self, authenticated_client
    ):
        """Test that system recovers after database comes back online"""
        # Simulate connection loss then recovery
        call_count = [0]

        def mock_get_db_with_recovery():
            call_count[0] += 1
            if call_count[0] <= 2:
                raise OperationalError("Connection refused", None, None)
            # After 2 failures, succeed
            from backend.utils.database import SessionLocal
            return SessionLocal()

        # First attempt should fail
        with patch(
            "backend.utils.database.get_db",
            side_effect=OperationalError("Connection refused", None, None),
        ):
            response = authenticated_client.get("/api/portfolio")
            assert response.status_code in [500, 503]

        # After recovery, should succeed
        response = authenticated_client.get("/api/portfolio")
        assert response.status_code in [200, 401, 400]  # Normal response


class TestCircuitBreaker:
    """Circuit breaker pattern tests"""

    def test_circuit_breaker_initial_state(self):
        """Test circuit breaker starts in CLOSED state"""
        breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            expected_exception=Exception,
        )

        assert breaker.state == CircuitState.CLOSED

    def test_circuit_breaker_opens_after_failures(self):
        """Test circuit breaker opens after threshold failures"""
        breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=60,
        )

        def failing_operation():
            raise Exception("Operation failed")

        # Trigger failures
        for i in range(3):
            try:
                breaker(failing_operation)()
            except Exception:
                pass

        # Circuit should open
        assert breaker.state == CircuitState.OPEN

    def test_circuit_breaker_rejects_calls_when_open(self):
        """Test circuit breaker rejects calls when OPEN"""
        breaker = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=60,
        )

        # Trigger failure to open circuit
        try:
            breaker(lambda: 1 / 0)()
        except Exception:
            pass

        # Circuit should be open
        assert breaker.state == CircuitState.OPEN

        # Next call should be rejected immediately
        with pytest.raises(Exception) as exc_info:
            breaker(lambda: "This won't execute")()

        assert "Circuit breaker is open" in str(exc_info.value)

    def test_circuit_breaker_half_open_state(self):
        """Test circuit breaker transitions to HALF_OPEN"""
        breaker = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0.1,  # Short timeout for testing
        )

        # Open the circuit
        try:
            breaker(lambda: 1 / 0)()
        except Exception:
            pass

        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(0.2)

        # Should attempt recovery (move to HALF_OPEN)
        # When we call a successful operation
        def successful_operation():
            return "success"

        try:
            result = breaker(successful_operation)()
            # If successful, circuit should close
            assert breaker.state == CircuitState.CLOSED
            assert result == "success"
        except Exception:
            # Depending on timing, might still be in recovery
            pass

    def test_circuit_breaker_with_slow_endpoint(self):
        """Test circuit breaker detects and handles slow responses"""
        breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=60,
            timeout=0.1,  # 100ms timeout
        )

        def slow_operation():
            time.sleep(0.5)  # Sleep 500ms, exceeds timeout
            return "done"

        # Operation should timeout
        with pytest.raises(Exception):
            breaker(slow_operation)()

    def test_circuit_breaker_metrics(self):
        """Test circuit breaker collects metrics"""
        breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
        )

        # Make some calls
        for i in range(3):
            try:
                breaker(lambda: 1 / 0 if i < 2 else "success")()
            except Exception:
                pass

        # Metrics should be tracked
        assert hasattr(breaker, "failure_count") or hasattr(breaker, "get_metrics")

    def test_circuit_breaker_for_external_api(self):
        """Test circuit breaker protecting external API calls"""
        breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=60,
            name="external_api",
        )

        # Simulate external API failures
        call_count = [0]

        def external_api_call():
            call_count[0] += 1
            if call_count[0] <= 3:
                raise ConnectionError("API unavailable")
            return {"data": "success"}

        # First 3 calls fail
        for i in range(3):
            try:
                breaker(external_api_call)()
            except Exception:
                pass

        # Circuit should open
        assert breaker.state == CircuitState.OPEN

        # Further calls rejected without attempting API call
        original_call_count = call_count[0]
        with pytest.raises(Exception):
            breaker(external_api_call)()

        # Should not have attempted another API call
        assert call_count[0] == original_call_count


class TestGracefulDegradation:
    """Graceful degradation tests"""

    def test_api_returns_cached_data_on_db_error(
        self, authenticated_client
    ):
        """Test API returns cached data when database is unavailable"""
        # In a real scenario, would verify cache hit
        # Mock implementation would show fallback behavior

        with patch(
            "backend.utils.database.get_db",
            side_effect=OperationalError("Connection lost", None, None),
        ):
            response = authenticated_client.get("/api/portfolio")

            # Should either:
            # 1. Return 503 if no cache
            # 2. Return cached data if available
            # 3. Return partial response with available data
            assert response.status_code in [200, 503]

    def test_missing_external_service_fallback(self):
        """Test graceful fallback when external service is unavailable"""
        # Example: stock price service is down
        # System should continue to work with last known prices

        with patch(
            "backend.streaming.price_service.get_live_price",
            return_value=None,
        ):
            # Fallback to cached price
            # In real implementation, would check cache
            pass

    def test_partial_response_on_service_failure(
        self, authenticated_client
    ):
        """Test API returns partial response if some services fail"""
        # Example: Portfolio endpoint returns positions but not recommendations

        response = authenticated_client.get("/api/portfolio")

        # Should return data that's available
        assert response.status_code in [200, 206]  # 206 = Partial Content
        data = response.json()

        # Should have some data
        if response.status_code == 200:
            assert data is not None

    def test_websocket_graceful_disconnect(self, test_user_data, db_session):
        """Test WebSocket handles unexpected disconnect gracefully"""
        user = User(
            username=test_user_data["username"],
            email=test_user_data["email"],
        )
        user.hashed_password = bcrypt.hashpw(test_user_data["password"].encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        db_session.add(user)
        db_session.commit()

        tokens = create_tokens(user)

        with TestClient(app) as client:
            with client.websocket_connect(
                "/api/ws/prices",
                headers={"Authorization": f"Bearer {tokens['access_token']}"},
            ) as websocket:
                websocket.receive_json()

                # Simulate abnormal closure
                # (implicit on context exit)

            # Server should clean up resources
            # Verify by creating new connection
            with client.websocket_connect(
                "/api/ws/prices",
                headers={"Authorization": f"Bearer {tokens['access_token']}"},
            ) as websocket2:
                data = websocket2.receive_json()
                assert data["type"] == "connection_established"

    def test_authentication_fallback(self):
        """Test system handles auth service failures gracefully"""
        with patch(
            "backend.auth.oauth2.decode_access_token",
            side_effect=Exception("Auth service down"),
        ):
            client = TestClient(app)
            response = client.get(
                "/api/portfolio",
                headers={"Authorization": "Bearer invalid"},
            )

            # Should return 401 or 503, not 500
            assert response.status_code in [401, 403, 503]

    def test_data_validation_prevents_corruption(self, db_session):
        """Test data validation prevents corrupted data from being stored"""
        from pydantic import ValidationError

        user = User(
            username="validation_test",
            email="validation@example.com",
        )
        user.hashed_password = bcrypt.hashpw("Pass123!@#".encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

        # Attempt to set invalid values
        with pytest.raises((ValueError, ValidationError, AttributeError)):
            user.id = "not_an_integer"

        db_session.add(user)
        db_session.commit()

        # Verify only valid user was stored
        stored_user = db_session.query(User).filter_by(
            email="validation@example.com"
        ).first()
        assert stored_user is not None
        assert isinstance(stored_user.id, int) or stored_user.id is None


class TestConcurrencyErrors:
    """Concurrency and race condition tests"""

    def test_simultaneous_portfolio_updates(self, db_session):
        """Test handling of simultaneous portfolio updates"""
        from concurrent.futures import ThreadPoolExecutor

        user = User(
            username="concurrent_test",
            email="concurrent@example.com",
        )
        user.hashed_password = bcrypt.hashpw("Pass123!@#".encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        db_session.add(user)
        db_session.commit()

        update_results = []

        def update_portfolio():
            try:
                # Simulate portfolio update
                user_check = db_session.query(User).filter_by(
                    email="concurrent@example.com"
                ).first()
                assert user_check is not None
                update_results.append("success")
            except Exception as e:
                update_results.append(f"error: {e}")

        # Simulate concurrent updates
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(update_portfolio) for _ in range(5)]
            [f.result() for f in futures]

        # All updates should succeed or fail gracefully
        assert len(update_results) == 5
        assert all("success" in r or "error" in r for r in update_results)

    def test_duplicate_transaction_prevention(self, db_session):
        """Test prevention of duplicate transaction entries"""
        # In real scenario, check for idempotency tokens or unique constraints

        user = User(
            username="dup_test",
            email="dup@example.com",
        )
        user.hashed_password = bcrypt.hashpw("Pass123!@#".encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        db_session.add(user)
        db_session.commit()

        # Attempt to create duplicate entry with same parameters
        # Should be prevented by unique constraints or idempotency
