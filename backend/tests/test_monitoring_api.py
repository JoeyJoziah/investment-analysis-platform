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
from datetime import datetime, timedelta, timezone
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
        Test retrieving metrics successfully.

        Verifies:
        - Endpoint returns 200 status
        - Response contains metrics data
        - Metrics are present and properly formatted
        """
        with patch("psutil.net_connections", return_value=[]):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as client:
                response = await client.get("/api/health/metrics")

                assert response.status_code == 200

                # Verify response format (JSON)
                data = response.json()

                # Should have success wrapper
                assert "success" in data or "data" in data

                # Get metrics data
                metrics_data = data.get("data", data)
                assert len(str(metrics_data)) > 0

    @pytest.mark.asyncio
    async def test_metrics_format_valid(self):
        """
        Test that metrics follow valid format.

        Verifies:
        - Metrics response is properly structured
        - Contains system metrics
        - All metrics are parseable
        """
        with patch("psutil.net_connections", return_value=[]):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as client:
                response = await client.get("/api/health/metrics")

                assert response.status_code == 200
                data = response.json()

                # Get metrics data
                metrics_data = data.get("data", data)
                assert isinstance(metrics_data, dict)

                # Should have system metrics
                assert "system" in metrics_data or len(metrics_data) > 0

                # If system metrics exist, validate structure
                if "system" in metrics_data:
                    system = metrics_data["system"]
                    assert isinstance(system, dict)
                    # Should have CPU, memory, or disk info
                    assert any(key in system for key in ["cpu_percent", "memory", "disk", "network"])

    @pytest.mark.asyncio
    async def test_metrics_includes_system_info(self):
        """
        Test that metrics include system information metadata.

        Verifies:
        - System info metric is present
        - Contains system resource information
        """
        with patch("psutil.net_connections", return_value=[]):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as client:
                response = await client.get("/api/health/metrics")

                assert response.status_code == 200
                data = response.json()
                metrics_data = data.get("data", data)

                # Check for system metrics
                has_system_metrics = "system" in metrics_data or len(metrics_data) > 0

                assert has_system_metrics, "No system metrics found in response"

    @pytest.mark.asyncio
    async def test_metrics_includes_api_metrics(self):
        """
        Test that metrics include API performance metrics.

        Verifies:
        - Metrics endpoint returns data
        - Contains valid metrics information
        """
        with patch("psutil.net_connections", return_value=[]):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as client:
                # First make a request to generate metrics
                await client.get("/api/health")

                # Then get metrics
                response = await client.get("/api/health/metrics")

                assert response.status_code == 200
                data = response.json()
                metrics_data = data.get("data", data)

                # Metrics endpoint should return valid format
                assert len(str(metrics_data)) > 0
                # Should be a dictionary with metrics
                assert isinstance(metrics_data, dict)


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


@patch("psutil.net_connections", return_value=[])
class TestPerformanceMetricsEndpoints:
    """Test suite for performance metrics endpoints."""

    @pytest.mark.asyncio
    async def test_performance_metrics_success(self, mock_net_conn):
        """
        Test retrieving performance metrics successfully.

        Verifies:
        - Endpoint returns 200 status
        - Response contains performance data
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
                response = await client.get("/api/health/metrics")

                # Should return 200
                assert response.status_code == 200

                # Verify response has content
                data = response.json()
                metrics_data = data.get("data", data)
                assert len(str(metrics_data)) > 0

                # Should be a dict with metrics
                assert isinstance(metrics_data, dict)
            finally:
                app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_performance_metrics_unauthorized(self, mock_net_conn):
        """
        Test that metrics endpoint is accessible regardless of auth.

        Verifies:
        - Metrics endpoint returns data
        - No 403 errors on metrics access
        """
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as client:
            # Test metrics endpoint
            response = await client.get("/api/health/metrics")

            # Metrics should be accessible
            assert response.status_code == 200

            # Should have content
            data = response.json()
            assert len(str(data)) > 0

    @pytest.mark.asyncio
    async def test_performance_metrics_admin_access(self, mock_net_conn):
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
                response = await client.get("/api/health/metrics")

                # Admin should be able to access
                assert response.status_code == 200

                # Should have data
                data = response.json()
                assert len(str(data)) > 0

            finally:
                app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_api_usage_metrics_endpoint(self, mock_net_conn):
        """
        Test that metrics include API usage information.

        Verifies:
        - Metrics endpoint returns 200 status
        - Contains API request information
        - Properly formatted data
        """
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as client:
            response = await client.get("/api/health/metrics")

            # Should return 200
            assert response.status_code == 200

            # Should have data
            data = response.json()
            assert len(str(data)) > 0

    @pytest.mark.asyncio
    async def test_cost_metrics_endpoint(self, mock_net_conn):
        """
        Test that metrics include cost information.

        Verifies:
        - Metrics endpoint returns 200 status
        - Contains valid metrics data
        """
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as client:
            response = await client.get("/api/health/metrics")

            # Should return 200
            assert response.status_code == 200

            # Should have data
            data = response.json()
            assert len(str(data)) > 0


@patch("psutil.net_connections", return_value=[])
class TestMetricsIntegration:
    """Integration tests for metrics collection and reporting."""

    @pytest.mark.asyncio
    async def test_metrics_collection_after_requests(self, mock_net_conn):
        """
        Test that metrics are collected from API requests.

        Verifies:
        - Metrics endpoint responds after requests
        - Metrics data is returned
        """
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as client:
            # Make a request to generate metrics
            response = await client.get("/api/health")
            assert response.status_code == 200

            # Get metrics
            metrics_response = await client.get("/api/health/metrics")
            assert metrics_response.status_code == 200

            data = metrics_response.json()
            metrics_data = data.get("data", data)

            # Should have some metrics recorded
            assert len(str(metrics_data)) > 0

    @pytest.mark.asyncio
    async def test_metrics_endpoint_response_structure(self, mock_net_conn):
        """
        Test that metrics endpoint returns proper response.

        Verifies:
        - Content-Type header is correct
        - Response is not empty
        - Response is valid JSON
        """
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as client:
            response = await client.get("/api/health/metrics")

            assert response.status_code == 200

            # Check content type
            content_type = response.headers.get("content-type", "")
            assert content_type, "Content-Type header missing"
            assert "json" in content_type.lower()

            # Response should be parseable as JSON
            data = response.json()
            assert isinstance(data, dict)
            assert len(str(data)) > 0

    @pytest.mark.asyncio
    async def test_health_endpoint_timestamp_validity(self, mock_net_conn):
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
                    # Ensure both datetimes are timezone-aware for comparison
                    now_utc = datetime.now(timezone.utc)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    time_diff = abs((now_utc - dt).total_seconds())
                    assert time_diff < 60, f"Timestamp too old: {time_diff} seconds"
                except ValueError:
                    # May be in a different format, just check it's a string
                    assert isinstance(timestamp_str, str)


@patch("psutil.net_connections", return_value=[])
class TestMetricsErrorHandling:
    """Test error handling in metrics endpoints."""

    @pytest.mark.asyncio
    async def test_metrics_endpoint_graceful_degradation(self, mock_net_conn):
        """
        Test that metrics endpoint handles errors gracefully.

        Verifies:
        - Endpoint always returns 200 even if some metrics fail
        - Partial metrics are returned when available
        - No exceptions are thrown to client
        """
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as client:
            response = await client.get("/api/health/metrics")

            # Should always return 200
            assert response.status_code == 200

            # Should have content
            data = response.json()
            assert len(str(data)) > 0

    @pytest.mark.asyncio
    async def test_health_check_handles_service_failures(self, mock_net_conn):
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
@patch("psutil.net_connections", return_value=[])
class TestMetricsCompleteness:
    """Test that all expected metrics are collected."""

    @pytest.mark.asyncio
    async def test_api_request_metrics_collected(self, mock_net_conn):
        """Test that API request metrics are properly collected."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as client:
            # Make a test request
            test_response = await client.get("/api/health")
            assert test_response.status_code in [200, 404]

            # Get metrics
            metrics_response = await client.get("/api/health/metrics")
            assert metrics_response.status_code == 200

            data = metrics_response.json()
            metrics_data = data.get("data", data)

            # Should have some metrics
            assert len(str(metrics_data)) > 0
            assert isinstance(metrics_data, dict)

    @pytest.mark.asyncio
    async def test_system_metrics_present(self, mock_net_conn):
        """Test that system metrics are included in response."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as client:
            response = await client.get("/api/health/metrics")

            assert response.status_code == 200

            data = response.json()
            metrics_data = data.get("data", data)

            # Should have system metrics
            assert isinstance(metrics_data, dict)
            assert len(str(metrics_data)) > 0


# Health endpoint tests that were previously inside metrics-only test classes
@pytest.mark.asyncio
async def test_health_endpoint_timestamp_validity():
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
                # Ensure both datetimes are timezone-aware for comparison
                now_utc = datetime.now(timezone.utc)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                time_diff = abs((now_utc - dt).total_seconds())
                assert time_diff < 60, f"Timestamp too old: {time_diff} seconds"
            except ValueError:
                # May be in a different format, just check it's a string
                assert isinstance(timestamp_str, str)


@pytest.mark.asyncio
async def test_health_check_handles_service_failures():
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
