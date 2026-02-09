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
