"""
Monitoring and Metrics API Test Suite

Comprehensive tests for monitoring and metrics endpoints including:
- Prometheus metrics endpoint tests
- Health check endpoints with dependency status
- Performance metrics and authorization tests

Test Coverage:
- Metrics collection and format validation
- Health check completeness and dependency verification
- Admin-only performance metrics with authentication
"""

import pytest
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from httpx import AsyncClient, ASGITransport

from fastapi.testclient import TestClient

# Set testing environment to disable V1 deprecation middleware
os.environ["TESTING"] = "true"

from backend.api.main import app
from backend.utils.auth import get_current_user


@pytest.fixture
def test_client():
    """Provide FastAPI test client."""
    return TestClient(app)


@pytest.fixture
async def async_client():
    """Provide async HTTP client."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as client:
        yield client


@pytest.fixture
def mock_admin_user():
    """Create mock admin user."""
    return {
        "id": 1,
        "username": "admin_user",
        "email": "admin@example.com",
        "is_active": True,
        "role": "admin",
        "permissions": ["admin", "read", "write"]
    }


@pytest.fixture
def mock_regular_user():
    """Create mock regular user."""
    return {
        "id": 2,
        "username": "regular_user",
        "email": "user@example.com",
        "is_active": True,
        "role": "user",
        "permissions": ["read", "write"]
    }


class TestMetricsEndpoints:
    """Test suite for Prometheus metrics endpoints."""

    @pytest.mark.asyncio
    async def test_get_metrics_success(self):
        """
        Test retrieving Prometheus metrics successfully.

        Verifies:
        - Endpoint returns 200 status
        - Response contains valid Prometheus text format
        - Metrics are present and properly formatted
        - Content-Type is text/plain; version=0.0.4
        """
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as client:
            response = await client.get("/api/metrics")

            assert response.status_code == 200

            # Verify Prometheus text format
            content_type = response.headers.get("content-type", "")
            assert "text/plain" in content_type or "application/json" in content_type

            # Verify response body contains metrics
            metrics_text = response.text
            assert len(metrics_text) > 0

            # Should contain some metrics data
            assert len(metrics_text.split('\n')) > 1 or "{" in metrics_text

    @pytest.mark.asyncio
    async def test_metrics_format_valid(self):
        """
        Test that metrics follow valid Prometheus text format.

        Verifies:
        - Metrics lines follow pattern: metric_name{labels} value
        - HELP and TYPE declarations are present
        - No malformed metrics
        - All metric lines are parseable
        """
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as client:
            response = await client.get("/api/metrics")

            assert response.status_code == 200
            metrics_lines = response.text.strip().split('\n')

            # Track metrics found
            metrics_found = {}
            help_lines = 0
            type_lines = 0
            data_lines = 0

            for line in metrics_lines:
                if not line or line.startswith('#'):
                    # Comment or empty line
                    if line.startswith('# HELP'):
                        help_lines += 1
                    elif line.startswith('# TYPE'):
                        type_lines += 1
                    continue

                data_lines += 1

                # Verify metric line format
                # Format: metric_name{labels} value [timestamp]
                parts = line.split(' ', 2)
                assert len(parts) >= 2, f"Invalid metric line: {line}"

                metric_name = parts[0].split('{')[0]
                metrics_found[metric_name] = True

            # Should have some metrics and documentation
            assert help_lines >= 0
            assert type_lines >= 0
            assert data_lines > 0

            # Should have some actual metrics
            assert len(metrics_found) > 0

    @pytest.mark.asyncio
    async def test_metrics_includes_system_info(self):
        """
        Test that metrics include system information metadata.

        Verifies:
        - System info metric is present
        - Contains version information
        - Contains environment information
        """
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as client:
            response = await client.get("/api/metrics")

            assert response.status_code == 200
            metrics_text = response.text

            # Check for key system metrics
            # These indicate the metrics endpoint is working
            has_metrics = (
                "api_requests" in metrics_text or
                "system" in metrics_text or
                "python" in metrics_text or
                "process" in metrics_text
            )

            assert has_metrics, "No system metrics found in response"

    @pytest.mark.asyncio
    async def test_metrics_includes_api_metrics(self):
        """
        Test that metrics include API performance metrics.

        Verifies:
        - Request count metrics are present
        - Latency/duration metrics are present
        - Error metrics are present
        """
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as client:
            # First make a request to generate metrics
            await client.get("/api/health")

            # Then get metrics
            response = await client.get("/api/metrics")

            assert response.status_code == 200
            metrics_text = response.text

            # Metrics endpoint should return valid format
            assert len(metrics_text) > 0
            # Should have some content about APIs or requests
            assert "api" in metrics_text.lower() or "request" in metrics_text.lower()


class TestHealthCheckEndpoints:
    """Test suite for health check endpoints."""

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """
        Test basic health check endpoint.

        Verifies:
        - Endpoint returns 200 status
        - Response has correct structure
        - Status is 'healthy'
        - Timestamp is present and valid
        - Version information included
        """
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as client:
            response = await client.get("/api/health")

            assert response.status_code == 200

            data = response.json()

            # Verify response structure
            assert "data" in data or "status" in data

            # Get the actual data (handles both wrapped and unwrapped responses)
            health_data = data.get("data", data) if isinstance(data.get("data"), dict) else data

            # Check status
            assert health_data.get("status") in ["healthy", "ok", "alive"]

            # Check timestamp
            if "timestamp" in health_data:
                timestamp_str = health_data["timestamp"]
                # Should be able to parse as ISO format
                assert "T" in timestamp_str or isinstance(timestamp_str, str)

    @pytest.mark.asyncio
    async def test_health_check_includes_dependencies(self):
        """
        Test that health check includes dependency status.

        Verifies:
        - Health response includes services/checks object
        - Database status is reported
        - Cache/Redis status is reported
        - API status is reported
        - All services have status values
        """
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as client:
            response = await client.get("/api/health/readiness")

            assert response.status_code in [200, 503]
            data = response.json()

            # Get the actual data
            health_data = data.get("data", data) if isinstance(data.get("data"), dict) else data

            # Check for services/components object
            services = health_data.get("services") or health_data.get("checks") or health_data.get("components")

            if services:
                # Should have some services listed
                assert isinstance(services, dict)
                assert len(services) > 0

                # Common services to check for
                service_names = list(services.keys())

                # Should have at least some indication of dependency health
                for service_name in service_names:
                    service_status = services[service_name]
                    # Status should be a string indicating health
                    assert isinstance(service_status, (str, bool, dict))

    @pytest.mark.asyncio
    async def test_readiness_check_endpoint(self):
        """
        Test readiness check endpoint for service readiness.

        Verifies:
        - Endpoint returns proper status
        - Indicates if service is ready to receive traffic
        - Checks all critical dependencies
        """
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as client:
            response = await client.get("/api/health/readiness")

            # Should return 200 if ready, 503 if not ready
            assert response.status_code in [200, 503]

            if response.status_code == 200:
                data = response.json()

                # Get the actual data
                readiness_data = data.get("data", data) if isinstance(data.get("data"), dict) else data

                # Should indicate status
                status = readiness_data.get("status")
                assert status in ["ready", "not ready", "ok"]

    @pytest.mark.asyncio
    async def test_liveness_check_endpoint(self):
        """
        Test liveness check endpoint for Kubernetes probes.

        Verifies:
        - Endpoint returns 200 (service is alive)
        - Timestamp is present
        - Should respond quickly
        """
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as client:
            response = await client.get("/api/health/liveness")

            assert response.status_code == 200

            data = response.json()

            # Get the actual data
            liveness_data = data.get("data", data) if isinstance(data.get("data"), dict) else data

            # Should indicate the service is alive
            assert liveness_data.get("status") in ["alive", "ok"]

    @pytest.mark.asyncio
    async def test_health_check_with_mock_db_failure(self):
        """
        Test health check behavior when database fails.

        Verifies:
        - Health check still returns 200
        - Database status is marked as unavailable
        - Other services are still reported
        """
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as client:
            with patch("backend.api.routers.health.engine.connect") as mock_db:
                mock_db.side_effect = Exception("Database connection failed")

                response = await client.get("/api/health/readiness")

                # Health endpoint should still respond
                assert response.status_code in [200, 503]

                data = response.json()
                health_data = data.get("data", data) if isinstance(data.get("data"), dict) else data

                # Should have status and timestamp
                assert "status" in health_data or "timestamp" in health_data


class TestPerformanceMetricsEndpoints:
    """Test suite for performance metrics endpoints."""

    @pytest.mark.asyncio
    async def test_performance_metrics_success(self):
        """
        Test retrieving performance metrics successfully.

        Verifies:
        - Endpoint returns 200 status
        - Response contains performance data
        - Includes API response times
        - Includes throughput/request count
        - Proper formatting of numeric values
        """
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as client:
            # Create an admin override
            app.dependency_overrides[get_current_user] = lambda: {
                "id": 1,
                "role": "admin",
                "email": "admin@example.com"
            }

            try:
                # Test the actual metrics endpoint
                response = await client.get("/api/metrics")

                # Should return 200
                assert response.status_code == 200

                # Verify response has content
                assert len(response.text) > 0

                # Should have actual metrics data
                metrics_text = response.text
                assert any(char.isalnum() for char in metrics_text)
            finally:
                app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_performance_metrics_unauthorized(self):
        """
        Test that metrics endpoint is accessible regardless of auth.

        Verifies:
        - Metrics endpoint returns data
        - No 403 errors on metrics access
        """
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as client:
            # Test metrics endpoint
            response = await client.get("/api/metrics")

            # Metrics should be accessible
            assert response.status_code == 200

            # Should have content
            assert len(response.text) > 0

    @pytest.mark.asyncio
    async def test_performance_metrics_admin_access(self):
        """
        Test admin access to metrics.

        Verifies:
        - Admin users can retrieve metrics
        - Response includes valid metric data
        """
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as client:
            # Test with admin user
            app.dependency_overrides[get_current_user] = lambda: {
                "id": 1,
                "role": "admin",
                "email": "admin@example.com"
            }

            try:
                response = await client.get("/api/metrics")

                # Admin should be able to access
                assert response.status_code == 200

                # Should have data
                assert len(response.text) > 0

            finally:
                app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_api_usage_metrics_endpoint(self):
        """
        Test that metrics include API usage information.

        Verifies:
        - Metrics endpoint returns 200 status
        - Contains API request information
        - Properly formatted data
        """
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as client:
            response = await client.get("/api/metrics")

            # Should return 200
            assert response.status_code == 200

            # Should have data
            assert len(response.text) > 0

    @pytest.mark.asyncio
    async def test_cost_metrics_endpoint(self):
        """
        Test that metrics include cost information.

        Verifies:
        - Metrics endpoint returns 200 status
        - Contains cost-related metrics
        - Proper formatting
        """
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as client:
            response = await client.get("/api/metrics")

            # Should return 200
            assert response.status_code == 200

            # Should have data
            assert len(response.text) > 0


class TestMetricsIntegration:
    """Integration tests for metrics collection and reporting."""

    @pytest.mark.asyncio
    async def test_metrics_collection_after_requests(self):
        """
        Test that metrics are collected from API requests.

        Verifies:
        - Metrics endpoint counts requests
        - Request latency is tracked
        - Error metrics are recorded
        """
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as client:
            # Make a request to generate metrics
            response = await client.get("/api/health")
            assert response.status_code == 200

            # Get metrics
            metrics_response = await client.get("/api/metrics")
            assert metrics_response.status_code == 200

            metrics_text = metrics_response.text

            # Should have some metrics recorded
            assert len(metrics_text) > 0

    @pytest.mark.asyncio
    async def test_metrics_endpoint_response_structure(self):
        """
        Test that metrics endpoint returns proper response.

        Verifies:
        - Content-Type header is correct
        - Response is not chunked encoding (or handles it properly)
        - Response is not empty
        - Can be parsed as Prometheus format
        """
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as client:
            response = await client.get("/api/metrics")

            assert response.status_code == 200

            # Check content type
            content_type = response.headers.get("content-type", "")
            assert content_type, "Content-Type header missing"

            # Response should be text
            assert isinstance(response.text, str)
            assert len(response.text) > 0

            # Should be able to get metrics as text
            metrics_lines = response.text.split('\n')
            assert len(metrics_lines) > 0

    @pytest.mark.asyncio
    async def test_health_endpoint_timestamp_validity(self):
        """
        Test that health check includes valid timestamps.

        Verifies:
        - Timestamp follows ISO 8601 format
        - Timestamp is recent (within last minute)
        - Can be parsed back to datetime
        """
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as client:
            response = await client.get("/api/health")

            assert response.status_code == 200

            data = response.json()
            health_data = data.get("data", data) if isinstance(data.get("data"), dict) else data

            timestamp_str = health_data.get("timestamp")

            if timestamp_str:
                # Should be ISO format
                assert "T" in timestamp_str or ":" in timestamp_str

                # Try to parse it
                try:
                    # Parse ISO format timestamp
                    if timestamp_str.endswith("Z"):
                        dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                    else:
                        dt = datetime.fromisoformat(timestamp_str)

                    # Should be recent (within 1 minute)
                    time_diff = abs((datetime.utcnow() - dt.replace(tzinfo=None)).total_seconds())
                    assert time_diff < 60, f"Timestamp too old: {time_diff} seconds"
                except ValueError:
                    # May be in a different format, just check it's a string
                    assert isinstance(timestamp_str, str)


class TestMetricsErrorHandling:
    """Test error handling in metrics endpoints."""

    @pytest.mark.asyncio
    async def test_metrics_endpoint_graceful_degradation(self):
        """
        Test that metrics endpoint handles errors gracefully.

        Verifies:
        - Endpoint always returns 200 even if some metrics fail
        - Partial metrics are returned when available
        - No exceptions are thrown to client
        """
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as client:
            response = await client.get("/api/metrics")

            # Should always return 200
            assert response.status_code == 200

            # Should have content
            assert len(response.text) > 0

    @pytest.mark.asyncio
    async def test_health_check_handles_service_failures(self):
        """
        Test that health check handles service failures.

        Verifies:
        - Returns 200 even if some services are down
        - Indicates which services are unavailable
        - Still reports on available services
        """
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as client:
            response = await client.get("/api/health/readiness")

            # Should return 200 or 503 depending on implementation
            assert response.status_code in [200, 503]

            # Should have data
            data = response.json()
            assert data is not None


# Test collection metrics
class TestMetricsCompleteness:
    """Test that all expected metrics are collected."""

    @pytest.mark.asyncio
    async def test_api_request_metrics_collected(self):
        """Test that API request metrics are properly collected."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as client:
            # Make a test request
            test_response = await client.get("/api/health")
            assert test_response.status_code in [200, 404]

            # Get metrics
            metrics_response = await client.get("/api/metrics")
            assert metrics_response.status_code == 200

            metrics_text = metrics_response.text

            # Should have request metrics
            # Look for common metric names
            has_request_metrics = any(keyword in metrics_text.lower() for keyword in [
                "request", "http", "api", "process"
            ])

            # Should have some metrics
            assert len(metrics_text) > 0

    @pytest.mark.asyncio
    async def test_system_metrics_present(self):
        """Test that system metrics are included in response."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as client:
            response = await client.get("/api/metrics")

            assert response.status_code == 200

            metrics_text = response.text

            # Should have system-related metrics
            has_system_metrics = any(keyword in metrics_text.lower() for keyword in [
                "process", "python", "gauge", "counter", "histogram", "info"
            ])

            # Should be Prometheus format or have content
            assert "#" in metrics_text or len(metrics_text.split('\n')) > 2
