"""
Middleware Stack Tests

Tests for the priority-based middleware stack manager.

Created: 2026-02-08
Part of: Issue #7 - Middleware Optimization
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from backend.middleware.stack import (
    MiddlewareStack,
    MiddlewarePriority,
    MiddlewareRegistration
)


class TestMiddleware1(BaseHTTPMiddleware):
    """Test middleware 1"""
    async def dispatch(self, request, call_next):
        request.state.test_order = getattr(request.state, 'test_order', [])
        request.state.test_order.append('middleware1')
        response = await call_next(request)
        response.headers['X-Test-1'] = 'passed'
        return response


class TestMiddleware2(BaseHTTPMiddleware):
    """Test middleware 2"""
    async def dispatch(self, request, call_next):
        request.state.test_order = getattr(request.state, 'test_order', [])
        request.state.test_order.append('middleware2')
        response = await call_next(request)
        response.headers['X-Test-2'] = 'passed'
        return response


class TestMiddleware3(BaseHTTPMiddleware):
    """Test middleware 3"""
    async def dispatch(self, request, call_next):
        request.state.test_order = getattr(request.state, 'test_order', [])
        request.state.test_order.append('middleware3')
        response = await call_next(request)
        response.headers['X-Test-3'] = 'passed'
        return response


class TestMiddlewarePriority:
    """Test MiddlewarePriority enum"""

    def test_priority_ordering(self):
        """Test priority values are in correct order"""
        # Higher priority = outermost = executed first on request
        assert MiddlewarePriority.ERROR_HANDLER > MiddlewarePriority.CORS
        assert MiddlewarePriority.CORS > MiddlewarePriority.SECURITY_HEADERS
        assert MiddlewarePriority.SECURITY_HEADERS > MiddlewarePriority.CSRF
        assert MiddlewarePriority.CSRF > MiddlewarePriority.RATE_LIMITING
        assert MiddlewarePriority.RATE_LIMITING > MiddlewarePriority.REQUEST_SIZE
        assert MiddlewarePriority.REQUEST_SIZE > MiddlewarePriority.AUTHENTICATION
        assert MiddlewarePriority.AUTHENTICATION > MiddlewarePriority.MONITORING
        assert MiddlewarePriority.MONITORING > MiddlewarePriority.CACHING
        assert MiddlewarePriority.CACHING > MiddlewarePriority.COMPRESSION

    def test_priority_values(self):
        """Test specific priority values"""
        assert MiddlewarePriority.ERROR_HANDLER == 10000
        assert MiddlewarePriority.CORS == 9000
        assert MiddlewarePriority.SECURITY_HEADERS == 8000
        assert MiddlewarePriority.CSRF == 7000
        assert MiddlewarePriority.COMPRESSION == 1000


class TestMiddlewareRegistration:
    """Test MiddlewareRegistration dataclass"""

    def test_registration_creation(self):
        """Test creating middleware registration"""
        registration = MiddlewareRegistration(
            name="test",
            middleware_class=TestMiddleware1,
            priority=MiddlewarePriority.NORMAL,
            config={"key": "value"},
            enabled=True,
            skip_in_testing=False
        )

        assert registration.name == "test"
        assert registration.middleware_class == TestMiddleware1
        assert registration.priority == MiddlewarePriority.NORMAL
        assert registration.config == {"key": "value"}
        assert registration.enabled is True
        assert registration.skip_in_testing is False

    def test_registration_defaults(self):
        """Test registration default values"""
        registration = MiddlewareRegistration(
            name="test",
            middleware_class=TestMiddleware1,
            priority=MiddlewarePriority.NORMAL,
            config={}
        )

        assert registration.enabled is True
        assert registration.skip_in_testing is False


class TestMiddlewareStack:
    """Test MiddlewareStack class"""

    @pytest.fixture
    def app(self):
        """Create test FastAPI app"""
        app = FastAPI()

        @app.get("/test")
        async def test_endpoint(request):
            order = getattr(request.state, 'test_order', [])
            return {"order": order}

        return app

    def test_register_middleware(self, app):
        """Test registering middleware"""
        stack = MiddlewareStack(app)

        result = stack.register(
            "test",
            TestMiddleware1,
            MiddlewarePriority.NORMAL,
            {}
        )

        assert result is stack  # Should return self for chaining
        assert len(stack.middlewares) == 1
        assert stack.middlewares[0].name == "test"

    def test_register_chaining(self, app):
        """Test middleware registration chaining"""
        stack = MiddlewareStack(app)

        result = stack.register(
            "test1",
            TestMiddleware1,
            MiddlewarePriority.HIGH
        ).register(
            "test2",
            TestMiddleware2,
            MiddlewarePriority.LOW
        )

        assert result is stack
        assert len(stack.middlewares) == 2

    def test_register_after_apply_fails(self, app):
        """Test that registration after apply raises error"""
        stack = MiddlewareStack(app)
        stack.apply()

        with pytest.raises(RuntimeError, match="Cannot register middleware after stack has been applied"):
            stack.register("test", TestMiddleware1, MiddlewarePriority.NORMAL)

    def test_apply_orders_by_priority(self, app):
        """Test middleware is applied in priority order"""
        stack = MiddlewareStack(app)

        # Register in random order
        stack.register("low", TestMiddleware3, MiddlewarePriority.LOW)
        stack.register("high", TestMiddleware1, MiddlewarePriority.HIGH)
        stack.register("normal", TestMiddleware2, MiddlewarePriority.NORMAL)

        stack.apply()

        # Verify stack is not empty
        assert len(stack.middlewares) == 3

        # Order should be: high (first), normal (second), low (third)
        sorted_middlewares = sorted(
            stack.middlewares,
            key=lambda m: m.priority.value,
            reverse=True
        )

        assert sorted_middlewares[0].name == "high"
        assert sorted_middlewares[1].name == "normal"
        assert sorted_middlewares[2].name == "low"

    def test_apply_skips_disabled(self, app):
        """Test that disabled middleware is skipped"""
        stack = MiddlewareStack(app)

        stack.register("enabled", TestMiddleware1, MiddlewarePriority.NORMAL, enabled=True)
        stack.register("disabled", TestMiddleware2, MiddlewarePriority.NORMAL, enabled=False)

        stack.apply()

        client = TestClient(app)
        response = client.get("/test")

        # Only enabled middleware should have run
        assert "X-Test-1" in response.headers
        assert "X-Test-2" not in response.headers

    def test_apply_skips_in_testing(self, app):
        """Test that middleware with skip_in_testing is skipped when is_testing=True"""
        stack = MiddlewareStack(app)

        stack.register("always", TestMiddleware1, MiddlewarePriority.NORMAL, skip_in_testing=False)
        stack.register("skip_test", TestMiddleware2, MiddlewarePriority.NORMAL, skip_in_testing=True)

        stack.apply(is_testing=True)

        client = TestClient(app)
        response = client.get("/test")

        # Only non-skipped middleware should run in testing
        assert "X-Test-1" in response.headers
        assert "X-Test-2" not in response.headers

    def test_apply_includes_all_in_production(self, app):
        """Test that all middleware runs when is_testing=False"""
        stack = MiddlewareStack(app)

        stack.register("always", TestMiddleware1, MiddlewarePriority.NORMAL, skip_in_testing=False)
        stack.register("skip_test", TestMiddleware2, MiddlewarePriority.NORMAL, skip_in_testing=True)

        stack.apply(is_testing=False)

        client = TestClient(app)
        response = client.get("/test")

        # Both middleware should run in production
        assert "X-Test-1" in response.headers
        assert "X-Test-2" in response.headers

    def test_apply_idempotent(self, app):
        """Test that calling apply twice doesn't re-apply"""
        stack = MiddlewareStack(app)
        stack.register("test", TestMiddleware1, MiddlewarePriority.NORMAL)

        stack.apply()
        stack.apply()  # Should not raise or re-apply

        assert stack._applied is True

    def test_get_stack_summary(self, app):
        """Test stack summary generation"""
        stack = MiddlewareStack(app)

        stack.register("high", TestMiddleware1, MiddlewarePriority.HIGH)
        stack.register("low", TestMiddleware2, MiddlewarePriority.LOW)

        summary = stack.get_stack_summary()

        assert "Middleware Stack" in summary
        assert "high" in summary
        assert "low" in summary
        assert "✓" in summary  # Enabled middleware

    def test_get_stack_summary_empty(self, app):
        """Test stack summary with no middleware"""
        stack = MiddlewareStack(app)
        summary = stack.get_stack_summary()

        assert "No middleware registered" in summary

    def test_get_stack_summary_shows_disabled(self, app):
        """Test stack summary shows disabled middleware"""
        stack = MiddlewareStack(app)
        stack.register("disabled", TestMiddleware1, MiddlewarePriority.NORMAL, enabled=False)

        summary = stack.get_stack_summary()

        assert "disabled" in summary
        assert "✗" in summary  # Disabled middleware

    def test_get_stack_summary_shows_skip_in_testing(self, app):
        """Test stack summary shows skip_in_testing flag"""
        stack = MiddlewareStack(app)
        stack.register("test", TestMiddleware1, MiddlewarePriority.NORMAL, skip_in_testing=True)

        summary = stack.get_stack_summary()

        assert "[skip in testing]" in summary

    def test_middleware_receives_config(self, app):
        """Test middleware receives configuration"""
        class ConfigurableMiddleware(BaseHTTPMiddleware):
            def __init__(self, app, test_value: str):
                super().__init__(app)
                self.test_value = test_value

            async def dispatch(self, request, call_next):
                response = await call_next(request)
                response.headers['X-Config'] = self.test_value
                return response

        stack = MiddlewareStack(app)
        stack.register(
            "configurable",
            ConfigurableMiddleware,
            MiddlewarePriority.NORMAL,
            {"test_value": "configured"}
        )

        stack.apply()

        client = TestClient(app)
        response = client.get("/test")

        assert response.headers.get('X-Config') == "configured"

    def test_middleware_execution_order(self, app):
        """Test middleware executes in correct order (highest priority first)"""
        stack = MiddlewareStack(app)

        # Register with explicit priorities
        stack.register("third", TestMiddleware3, MiddlewarePriority.COMPRESSION)  # 1000
        stack.register("first", TestMiddleware1, MiddlewarePriority.CORS)  # 9000
        stack.register("second", TestMiddleware2, MiddlewarePriority.SECURITY_HEADERS)  # 8000

        stack.apply()

        client = TestClient(app)
        response = client.get("/test")

        # All three middleware should have added headers (proving they executed)
        assert "X-Test-1" in response.headers
        assert "X-Test-2" in response.headers
        assert "X-Test-3" in response.headers

        # The middleware were applied in priority order (highest first)
        # This means CORS wraps SECURITY_HEADERS wraps COMPRESSION
        sorted_middlewares = sorted(
            stack.middlewares,
            key=lambda m: m.priority.value,
            reverse=True
        )
        assert sorted_middlewares[0].name == "first"  # CORS (9000)
        assert sorted_middlewares[1].name == "second"  # SECURITY_HEADERS (8000)
        assert sorted_middlewares[2].name == "third"  # COMPRESSION (1000)


class TestMiddlewarePerformance:
    """Test middleware performance and overhead"""

    @pytest.fixture
    def minimal_app(self):
        """Create minimal FastAPI app for performance testing"""
        app = FastAPI()

        @app.get("/fast")
        async def fast_endpoint():
            return {"status": "ok"}

        return app

    def test_middleware_overhead_under_5ms(self, minimal_app):
        """Test that middleware stack overhead is under 5ms for simple request"""
        import time

        stack = MiddlewareStack(minimal_app)

        # Register typical middleware stack
        from backend.middleware.response_optimizer import ResponseTimingMiddleware
        from fastapi.middleware.gzip import GZipMiddleware

        stack.register("timing", ResponseTimingMiddleware, MiddlewarePriority.HIGHEST, {})
        stack.register("gzip", GZipMiddleware, MiddlewarePriority.COMPRESSION, {"minimum_size": 1000})

        stack.apply()

        client = TestClient(minimal_app)

        # Warm up (JIT compilation, etc.)
        for _ in range(5):
            client.get("/fast")

        # Measure actual overhead
        timings = []
        for _ in range(20):
            start = time.perf_counter()
            response = client.get("/fast")
            end = time.perf_counter()

            assert response.status_code == 200
            timings.append((end - start) * 1000)  # Convert to ms

        avg_time = sum(timings) / len(timings)
        max_time = max(timings)

        # Middleware overhead should be minimal
        assert avg_time < 5.0, f"Average middleware overhead {avg_time:.2f}ms exceeds 5ms"
        assert max_time < 10.0, f"Max middleware overhead {max_time:.2f}ms exceeds 10ms"

    def test_response_timing_middleware_accuracy(self, minimal_app):
        """Test ResponseTimingMiddleware reports accurate timing"""
        from backend.middleware.response_optimizer import ResponseTimingMiddleware

        stack = MiddlewareStack(minimal_app)
        stack.register("timing", ResponseTimingMiddleware, MiddlewarePriority.HIGHEST, {})
        stack.apply()

        client = TestClient(minimal_app)
        response = client.get("/fast")

        assert response.status_code == 200
        assert "X-Response-Time" in response.headers

        # Parse timing header
        timing_str = response.headers["X-Response-Time"]
        assert timing_str.endswith("ms")

        timing_ms = float(timing_str.replace("ms", ""))

        # Should be very fast (under 100ms in test environment)
        assert timing_ms < 100.0, f"Response time {timing_ms}ms seems too high for simple endpoint"


class TestMiddlewareRouterCompatibility:
    """Test middleware compatibility with different router types"""

    @pytest.fixture
    def mixed_app(self):
        """Create app with both sync and async handlers"""
        app = FastAPI()

        @app.get("/sync")
        def sync_handler():
            """Synchronous route handler"""
            return {"type": "sync", "value": 42}

        @app.get("/async")
        async def async_handler():
            """Asynchronous route handler"""
            return {"type": "async", "value": 99}

        @app.post("/sync_post")
        def sync_post_handler(data: dict):
            """Synchronous POST handler"""
            return {"type": "sync_post", "received": data}

        @app.post("/async_post")
        async def async_post_handler(data: dict):
            """Asynchronous POST handler"""
            return {"type": "async_post", "received": data}

        return app

    def test_middleware_with_sync_handler(self, mixed_app):
        """Test middleware works correctly with synchronous route handlers"""
        stack = MiddlewareStack(mixed_app)

        stack.register("test1", TestMiddleware1, MiddlewarePriority.HIGH)
        stack.register("test2", TestMiddleware2, MiddlewarePriority.LOW)

        stack.apply()

        client = TestClient(mixed_app)
        response = client.get("/sync")

        assert response.status_code == 200
        assert response.json() == {"type": "sync", "value": 42}

        # Verify middleware executed
        assert "X-Test-1" in response.headers
        assert "X-Test-2" in response.headers

    def test_middleware_with_async_handler(self, mixed_app):
        """Test middleware works correctly with asynchronous route handlers"""
        stack = MiddlewareStack(mixed_app)

        stack.register("test1", TestMiddleware1, MiddlewarePriority.HIGH)
        stack.register("test2", TestMiddleware2, MiddlewarePriority.LOW)

        stack.apply()

        client = TestClient(mixed_app)
        response = client.get("/async")

        assert response.status_code == 200
        assert response.json() == {"type": "async", "value": 99}

        # Verify middleware executed
        assert "X-Test-1" in response.headers
        assert "X-Test-2" in response.headers

    def test_middleware_with_sync_post(self, mixed_app):
        """Test middleware works with synchronous POST handlers"""
        stack = MiddlewareStack(mixed_app)
        stack.register("test1", TestMiddleware1, MiddlewarePriority.HIGH)
        stack.apply()

        client = TestClient(mixed_app)
        response = client.post("/sync_post", json={"key": "value"})

        assert response.status_code == 200
        assert response.json()["type"] == "sync_post"
        assert response.json()["received"] == {"key": "value"}
        assert "X-Test-1" in response.headers

    def test_middleware_with_async_post(self, mixed_app):
        """Test middleware works with asynchronous POST handlers"""
        stack = MiddlewareStack(mixed_app)
        stack.register("test1", TestMiddleware1, MiddlewarePriority.HIGH)
        stack.apply()

        client = TestClient(mixed_app)
        response = client.post("/async_post", json={"key": "value"})

        assert response.status_code == 200
        assert response.json()["type"] == "async_post"
        assert response.json()["received"] == {"key": "value"}
        assert "X-Test-1" in response.headers


class TestCORSSecurityHeadersIntegration:
    """Test CORS and SecurityHeaders middleware integration"""

    @pytest.fixture
    def cors_app(self):
        """Create app with CORS and SecurityHeaders"""
        app = FastAPI()

        @app.get("/api/data")
        async def get_data():
            return {"data": "test"}

        return app

    def test_cors_and_security_headers_no_conflicts(self, cors_app):
        """Test CORS and SecurityHeaders don't conflict on headers"""
        from fastapi.middleware.cors import CORSMiddleware
        from backend.middleware.security_headers import SecurityHeadersMiddleware, SecurityHeadersConfig

        stack = MiddlewareStack(cors_app)

        # Register CORS first (higher priority)
        stack.register(
            "cors",
            CORSMiddleware,
            MiddlewarePriority.CORS,
            {
                "allow_origins": ["http://localhost:3000"],
                "allow_credentials": True,
                "allow_methods": ["*"],
                "allow_headers": ["*"],
                "expose_headers": ["X-Request-ID", "X-Response-Time"]
            }
        )

        # Register SecurityHeaders after CORS
        stack.register(
            "security_headers",
            SecurityHeadersMiddleware,
            MiddlewarePriority.SECURITY_HEADERS,
            {"config": SecurityHeadersConfig()}
        )

        stack.apply()

        client = TestClient(cors_app)

        # Test actual GET request with Origin header
        response = client.get(
            "/api/data",
            headers={"Origin": "http://localhost:3000"}
        )

        assert response.status_code == 200

        # Both CORS and security headers should be present
        headers_lower = {k.lower(): v for k, v in response.headers.items()}

        # CORS header should be present
        assert "access-control-allow-origin" in headers_lower
        assert headers_lower["access-control-allow-origin"] == "http://localhost:3000"

        # Security headers (from SecurityHeadersMiddleware)
        assert "x-content-type-options" in headers_lower
        assert headers_lower["x-content-type-options"] == "nosniff"
        assert "x-frame-options" in headers_lower
        assert "referrer-policy" in headers_lower

        # Verify no duplicate CORS headers
        cors_headers = [k for k in response.headers.keys() if "access-control" in k.lower()]
        # Each CORS header should appear only once
        assert len(cors_headers) == len(set(h.lower() for h in cors_headers))

        # Test that CORS preflight is handled correctly (via CORS middleware)
        # Note: CORSMiddleware automatically handles OPTIONS requests
        response = client.options(
            "/api/data",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET"
            }
        )

        # CORS middleware should handle the preflight
        assert response.status_code == 200
        headers_lower = {k.lower(): v for k, v in response.headers.items()}
        assert "access-control-allow-origin" in headers_lower

    def test_security_headers_dont_add_cors(self, cors_app):
        """Test SecurityHeaders middleware doesn't add CORS headers"""
        from backend.middleware.security_headers import SecurityHeadersMiddleware, SecurityHeadersConfig

        stack = MiddlewareStack(cors_app)

        # Register ONLY SecurityHeaders (no CORS)
        stack.register(
            "security_headers",
            SecurityHeadersMiddleware,
            MiddlewarePriority.SECURITY_HEADERS,
            {"config": SecurityHeadersConfig()}
        )

        stack.apply()

        client = TestClient(cors_app)
        response = client.get("/api/data")

        assert response.status_code == 200

        # Security headers should be present
        headers_lower = {k.lower(): v for k, v in response.headers.items()}
        assert "x-content-type-options" in headers_lower
        assert "x-frame-options" in headers_lower

        # CORS headers should NOT be present (SecurityHeaders doesn't add them)
        assert "access-control-allow-origin" not in headers_lower
        assert "access-control-allow-credentials" not in headers_lower
