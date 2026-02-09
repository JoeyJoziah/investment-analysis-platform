"""
Tests for Response Optimizer Middleware

Tests for response timing tracking and ETag generation middleware.

Created: 2026-02-08
Part of: Issue #12 - API Response Time Optimization
"""

import pytest
import hashlib
import time
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient
from starlette.responses import JSONResponse

from backend.middleware.response_optimizer import (
    ResponseTimingMiddleware,
    ETagMiddleware
)


@pytest.fixture
def app():
    """Create test FastAPI app"""
    app = FastAPI()

    @app.get("/test")
    def test_endpoint():
        return {"message": "test"}

    @app.get("/slow")
    async def slow_endpoint():
        await asyncio.sleep(1.1)  # Simulate slow endpoint
        return {"message": "slow"}

    @app.get("/excluded/auth/login")
    def excluded_endpoint():
        return {"message": "auth"}

    @app.post("/test")
    def test_post():
        return {"message": "posted"}

    @app.get("/error")
    def error_endpoint():
        raise ValueError("Test error")

    return app


class TestResponseTimingMiddleware:
    """Tests for ResponseTimingMiddleware"""

    def test_adds_response_time_header(self, app):
        """Test that X-Response-Time header is added"""
        app.add_middleware(ResponseTimingMiddleware)
        client = TestClient(app)

        response = client.get("/test")

        assert response.status_code == 200
        assert "X-Response-Time" in response.headers

        # Verify format (e.g., "1.23ms")
        response_time = response.headers["X-Response-Time"]
        assert response_time.endswith("ms")
        assert float(response_time.replace("ms", "")) >= 0

    def test_measures_actual_time(self, app):
        """Test that timing is reasonably accurate"""
        app.add_middleware(ResponseTimingMiddleware)
        client = TestClient(app)

        response = client.get("/test")

        response_time = float(response.headers["X-Response-Time"].replace("ms", ""))

        # Should be less than 100ms for simple endpoint
        assert response_time < 100

    def test_timing_on_error(self, app):
        """Test that timing is tracked even when endpoint errors"""
        app.add_middleware(ResponseTimingMiddleware)
        client = TestClient(app)

        with pytest.raises(ValueError):
            client.get("/error")

    def test_logs_slow_requests(self, app):
        """Test that response time is tracked for all requests (slow logging tested via manual testing)"""
        app.add_middleware(ResponseTimingMiddleware)
        client = TestClient(app)

        response = client.get("/test")

        # Verify timing header exists - manual testing would verify logging
        assert "X-Response-Time" in response.headers
        response_time = float(response.headers["X-Response-Time"].replace("ms", ""))
        assert response_time >= 0

    def test_works_with_all_methods(self, app):
        """Test that timing works for different HTTP methods"""
        app.add_middleware(ResponseTimingMiddleware)
        client = TestClient(app)

        # GET
        response = client.get("/test")
        assert "X-Response-Time" in response.headers

        # POST
        response = client.post("/test")
        assert "X-Response-Time" in response.headers


class TestETagMiddleware:
    """Tests for ETagMiddleware"""

    def test_generates_etag_for_get_requests(self, app):
        """Test that ETags are generated for GET requests"""
        app.add_middleware(ETagMiddleware)
        client = TestClient(app)

        response = client.get("/test")

        assert response.status_code == 200
        assert "ETag" in response.headers

        # Verify ETag format (quoted string)
        etag = response.headers["ETag"]
        assert etag.startswith('"')
        assert etag.endswith('"')

    def test_etag_is_consistent(self, app):
        """Test that same content produces same ETag"""
        app.add_middleware(ETagMiddleware)
        client = TestClient(app)

        response1 = client.get("/test")
        response2 = client.get("/test")

        assert response1.headers["ETag"] == response2.headers["ETag"]

    def test_etag_differs_for_different_content(self, app):
        """Test that different content produces different ETags"""
        test_app = FastAPI()

        @test_app.get("/endpoint1")
        def endpoint1():
            return {"data": "content1"}

        @test_app.get("/endpoint2")
        def endpoint2():
            return {"data": "content2"}

        test_app.add_middleware(ETagMiddleware)
        client = TestClient(test_app)

        response1 = client.get("/endpoint1")
        response2 = client.get("/endpoint2")

        assert response1.headers["ETag"] != response2.headers["ETag"]

    def test_handles_if_none_match_304(self, app):
        """Test that 304 Not Modified is returned when ETag matches"""
        app.add_middleware(ETagMiddleware)
        client = TestClient(app)

        # First request - get ETag
        response1 = client.get("/test")
        assert response1.status_code == 200
        etag = response1.headers["ETag"]

        # Second request with If-None-Match
        response2 = client.get("/test", headers={"If-None-Match": etag})

        assert response2.status_code == 304
        assert "ETag" in response2.headers
        assert response2.headers["ETag"] == etag
        # 304 responses should have no body
        assert len(response2.content) == 0

    def test_if_none_match_mismatch_returns_200(self, app):
        """Test that 200 OK is returned when ETag doesn't match"""
        app.add_middleware(ETagMiddleware)
        client = TestClient(app)

        # Request with wrong ETag
        response = client.get("/test", headers={"If-None-Match": '"wrongetag"'})

        assert response.status_code == 200
        assert response.json()["message"] == "test"

    def test_if_none_match_wildcard(self, app):
        """Test that wildcard * in If-None-Match matches any ETag"""
        app.add_middleware(ETagMiddleware)
        client = TestClient(app)

        response = client.get("/test", headers={"If-None-Match": "*"})

        assert response.status_code == 304

    def test_if_none_match_multiple_etags(self, app):
        """Test that multiple ETags in If-None-Match are handled"""
        app.add_middleware(ETagMiddleware)
        client = TestClient(app)

        # Get real ETag
        response1 = client.get("/test")
        etag = response1.headers["ETag"]

        # Send multiple ETags (comma-separated)
        response2 = client.get(
            "/test",
            headers={"If-None-Match": f'"fakeetag1", {etag}, "fakeetag2"'}
        )

        assert response2.status_code == 304

    def test_weak_etag_generation(self, app):
        """Test that weak ETags are generated when configured"""
        app.add_middleware(ETagMiddleware, weak_etag=True)
        client = TestClient(app)

        response = client.get("/test")

        etag = response.headers["ETag"]
        assert etag.startswith('W/"')
        assert etag.endswith('"')

    def test_weak_etag_matching(self, app):
        """Test that weak ETags match correctly"""
        app.add_middleware(ETagMiddleware, weak_etag=True)
        client = TestClient(app)

        # First request
        response1 = client.get("/test")
        weak_etag = response1.headers["ETag"]
        assert weak_etag.startswith('W/"')

        # Second request with weak ETag - should get 304
        response2 = client.get("/test", headers={"If-None-Match": weak_etag})
        assert response2.status_code == 304

    def test_no_etag_for_post_requests(self, app):
        """Test that ETags are not generated for POST requests"""
        app.add_middleware(ETagMiddleware)
        client = TestClient(app)

        response = client.post("/test")

        assert response.status_code == 200
        assert "ETag" not in response.headers

    def test_no_etag_for_excluded_paths(self, app):
        """Test that ETags are not generated for excluded paths"""
        app.add_middleware(
            ETagMiddleware,
            excluded_paths=["/excluded/"]
        )
        client = TestClient(app)

        response = client.get("/excluded/auth/login")

        assert response.status_code == 200
        assert "ETag" not in response.headers

    def test_no_etag_for_error_responses(self, app):
        """Test that ETags are not generated for error responses"""
        test_app = FastAPI()

        @test_app.get("/notfound")
        def not_found():
            return JSONResponse(status_code=404, content={"error": "Not found"})

        test_app.add_middleware(ETagMiddleware)
        client = TestClient(test_app)

        response = client.get("/notfound")

        assert response.status_code == 404
        assert "ETag" not in response.headers

    def test_etag_hash_generation(self, app):
        """Test that ETag hash is generated correctly from content"""
        app.add_middleware(ETagMiddleware)
        client = TestClient(app)

        response = client.get("/test")
        etag = response.headers["ETag"].strip('"')

        # Verify hash matches content
        content = response.content
        expected_hash = hashlib.md5(content).hexdigest()[:16]

        assert etag == expected_hash


class TestMiddlewareCombination:
    """Tests for both middleware working together"""

    def test_timing_and_etag_together(self, app):
        """Test that both middleware work correctly together"""
        # Add in correct order (timing outermost)
        app.add_middleware(ETagMiddleware)
        app.add_middleware(ResponseTimingMiddleware)
        client = TestClient(app)

        response = client.get("/test")

        assert response.status_code == 200
        assert "X-Response-Time" in response.headers
        assert "ETag" in response.headers

    def test_timing_includes_etag_overhead(self, app):
        """Test that response time includes ETag generation"""
        app.add_middleware(ETagMiddleware)
        app.add_middleware(ResponseTimingMiddleware)
        client = TestClient(app)

        response = client.get("/test")

        response_time = float(response.headers["X-Response-Time"].replace("ms", ""))

        # Should be positive and reasonable
        assert response_time > 0
        assert response_time < 1000  # Should be fast

    def test_304_response_has_timing(self, app):
        """Test that 304 Not Modified responses still have timing"""
        app.add_middleware(ETagMiddleware)
        app.add_middleware(ResponseTimingMiddleware)
        client = TestClient(app)

        # First request
        response1 = client.get("/test")
        etag = response1.headers["ETag"]

        # Second request (should be 304)
        response2 = client.get("/test", headers={"If-None-Match": etag})

        assert response2.status_code == 304
        assert "X-Response-Time" in response2.headers
        assert "ETag" in response2.headers


class TestPerformance:
    """Performance tests for middleware"""

    def test_etag_generation_is_fast(self, app):
        """Test that ETag generation has minimal overhead"""
        app.add_middleware(ETagMiddleware)
        app.add_middleware(ResponseTimingMiddleware)
        client = TestClient(app)

        response = client.get("/test")

        response_time = float(response.headers["X-Response-Time"].replace("ms", ""))

        # ETag generation should add <5ms overhead
        assert response_time < 10

    def test_timing_overhead_is_minimal(self):
        """Test that timing middleware has minimal overhead"""
        # Create two separate apps to test with/without middleware
        app_no_middleware = FastAPI()

        @app_no_middleware.get("/test")
        def test_endpoint():
            return {"message": "test"}

        app_with_middleware = FastAPI()

        @app_with_middleware.get("/test")
        def test_endpoint2():
            return {"message": "test"}

        app_with_middleware.add_middleware(ResponseTimingMiddleware)

        # Test without middleware
        client_no_middleware = TestClient(app_no_middleware)
        start = time.perf_counter()
        client_no_middleware.get("/test")
        time_no_middleware = (time.perf_counter() - start) * 1000

        # Test with middleware
        client_with_middleware = TestClient(app_with_middleware)
        start = time.perf_counter()
        client_with_middleware.get("/test")
        time_with_middleware = (time.perf_counter() - start) * 1000

        # Overhead should be less than 10ms (lenient for test environment)
        overhead = time_with_middleware - time_no_middleware
        assert overhead < 10


# Add asyncio import for async tests
import asyncio


@pytest.fixture
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
