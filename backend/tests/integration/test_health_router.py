"""
Integration tests for the health check router.

Tests cover all health endpoints: basic health, readiness, liveness,
startup, metrics, and ping. External dependencies (database engine,
Redis) are mocked so the test suite is self-contained.
"""

import contextlib

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock, PropertyMock
from httpx import AsyncClient


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_redis_client(*, ping_ok: bool = True, info: dict | None = None):
    """Return a mock Redis client with configurable behaviour."""
    client = MagicMock()
    if ping_ok:
        client.ping.return_value = True
    else:
        client.ping.side_effect = ConnectionError("Redis unreachable")

    if info is None:
        info = {
            "used_memory_human": "1.50M",
            "connected_clients": 4,
            "total_commands_processed": 500,
            "keyspace_hits": 300,
            "keyspace_misses": 50,
        }
    client.info.return_value = info
    return client


def _mock_psutil():
    """Return a dict of psutil patches to avoid OS permission errors in CI."""
    mock_vm = MagicMock(total=16_000_000_000, used=8_000_000_000,
                        percent=50.0, available=8_000_000_000)
    mock_disk = MagicMock(total=500_000_000_000, used=250_000_000_000,
                          free=250_000_000_000, percent=50.0)
    return {
        "backend.api.routers.health.psutil.cpu_percent": MagicMock(return_value=25.0),
        "backend.api.routers.health.psutil.virtual_memory": MagicMock(return_value=mock_vm),
        "backend.api.routers.health.psutil.disk_usage": MagicMock(return_value=mock_disk),
        "backend.api.routers.health.psutil.net_connections": MagicMock(return_value=[]),
    }


def _mock_engine(*, select_ok: bool = True, table_count: int = 12):
    """Return a mock SQLAlchemy engine with a fake connection pool."""
    engine = MagicMock()

    # Pool statistics
    pool = MagicMock()
    pool.size.return_value = 10
    pool.checkedin.return_value = 8
    pool.overflow.return_value = 0
    pool.total.return_value = 10
    engine.pool = pool

    # Connection context manager
    conn = MagicMock()

    def _execute_side_effect(stmt):
        sql_text = str(stmt)
        if "SELECT 1" in sql_text:
            result = MagicMock()
            if not select_ok:
                raise ConnectionError("Database unreachable")
            result.fetchone.return_value = (1,)
            return result
        if "information_schema" in sql_text:
            result = MagicMock()
            result.scalar.return_value = table_count
            return result
        return MagicMock()

    conn.execute.side_effect = _execute_side_effect
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)
    engine.connect.return_value = conn

    return engine


# ---------------------------------------------------------------------------
# 1. Basic health check
# ---------------------------------------------------------------------------

class TestBasicHealthCheck:
    """Tests for GET /api/health -- the lightweight health endpoint."""

    @pytest.mark.asyncio
    async def test_returns_healthy_status(self, async_client: AsyncClient):
        """Health check should return status=healthy with service metadata."""
        response = await async_client.get("/api/health")

        assert response.status_code == 200
        body = response.json()

        # ApiResponse wrapper
        assert body["success"] is True
        data = body["data"]

        assert data["status"] == "healthy"
        assert data["service"] == "investment-analysis-api"
        assert data["version"] == "1.0.0"

    @pytest.mark.asyncio
    async def test_includes_iso_timestamp(self, async_client: AsyncClient):
        """Health check response must contain a valid ISO-8601 timestamp."""
        response = await async_client.get("/api/health")

        data = response.json()["data"]
        # Should parse without error
        parsed = datetime.fromisoformat(data["timestamp"])
        assert isinstance(parsed, datetime)

    @pytest.mark.asyncio
    async def test_response_format_matches_api_response_schema(
        self, async_client: AsyncClient
    ):
        """Response must conform to the standard ApiResponse envelope."""
        response = await async_client.get("/api/health")
        body = response.json()

        # Top-level keys mandated by ApiResponse
        assert "success" in body
        assert "data" in body
        assert "error" in body
        assert "meta" in body

        # Success path: error and meta should be null
        assert body["error"] is None
        assert body["meta"] is None


# ---------------------------------------------------------------------------
# 2. Readiness check  (database + Redis)
# ---------------------------------------------------------------------------

class TestReadinessCheck:
    """Tests for GET /api/health/readiness -- verifies dependency connectivity."""

    @pytest.mark.asyncio
    async def test_all_services_ready(self, async_client: AsyncClient):
        """When both database and Redis are reachable, status should be 'ready'."""
        mock_engine = _mock_engine(select_ok=True, table_count=15)
        mock_redis = _mock_redis_client(ping_ok=True)

        with (
            patch("backend.api.routers.health.engine", mock_engine),
            patch(
                "backend.api.routers.health.get_redis_client",
                return_value=mock_redis,
            ),
        ):
            response = await async_client.get("/api/health/readiness")

        assert response.status_code == 200
        data = response.json()["data"]

        assert data["status"] == "ready"
        assert data["checks"]["database"] is True
        assert data["checks"]["cache"] is True
        assert data["checks"]["api"] is True
        assert "errors" not in data

    @pytest.mark.asyncio
    async def test_database_down_marks_not_ready(self, async_client: AsyncClient):
        """When the database is unreachable, status should be 'not ready'."""
        mock_engine = _mock_engine(select_ok=False)
        mock_redis = _mock_redis_client(ping_ok=True)

        with (
            patch("backend.api.routers.health.engine", mock_engine),
            patch(
                "backend.api.routers.health.get_redis_client",
                return_value=mock_redis,
            ),
        ):
            response = await async_client.get("/api/health/readiness")

        assert response.status_code == 200
        data = response.json()["data"]

        assert data["status"] == "not ready"
        assert data["checks"]["database"] is False
        assert data["checks"]["cache"] is True
        assert "database" in data["errors"]

    @pytest.mark.asyncio
    async def test_redis_down_marks_not_ready(self, async_client: AsyncClient):
        """When Redis is unreachable, status should be 'not ready'."""
        mock_engine = _mock_engine(select_ok=True, table_count=10)
        mock_redis = _mock_redis_client(ping_ok=False)

        with (
            patch("backend.api.routers.health.engine", mock_engine),
            patch(
                "backend.api.routers.health.get_redis_client",
                return_value=mock_redis,
            ),
        ):
            response = await async_client.get("/api/health/readiness")

        assert response.status_code == 200
        data = response.json()["data"]

        assert data["status"] == "not ready"
        assert data["checks"]["cache"] is False
        assert "cache" in data["errors"]

    @pytest.mark.asyncio
    async def test_no_tables_marks_database_not_ready(
        self, async_client: AsyncClient
    ):
        """When the database is reachable but has zero tables, it is not ready."""
        mock_engine = _mock_engine(select_ok=True, table_count=0)
        mock_redis = _mock_redis_client(ping_ok=True)

        with (
            patch("backend.api.routers.health.engine", mock_engine),
            patch(
                "backend.api.routers.health.get_redis_client",
                return_value=mock_redis,
            ),
        ):
            response = await async_client.get("/api/health/readiness")

        data = response.json()["data"]

        assert data["checks"]["database"] is False
        assert data["status"] == "not ready"
        assert "database" in data["errors"]
        assert "No tables" in data["errors"]["database"]

    @pytest.mark.asyncio
    async def test_readiness_includes_timestamp(self, async_client: AsyncClient):
        """Readiness response must include a valid timestamp."""
        mock_engine = _mock_engine()
        mock_redis = _mock_redis_client()

        with (
            patch("backend.api.routers.health.engine", mock_engine),
            patch(
                "backend.api.routers.health.get_redis_client",
                return_value=mock_redis,
            ),
        ):
            response = await async_client.get("/api/health/readiness")

        data = response.json()["data"]
        parsed = datetime.fromisoformat(data["timestamp"])
        assert isinstance(parsed, datetime)


# ---------------------------------------------------------------------------
# 3. Liveness probe
# ---------------------------------------------------------------------------

class TestLivenessCheck:
    """Tests for GET /api/health/liveness -- Kubernetes liveness probe."""

    @pytest.mark.asyncio
    async def test_returns_alive(self, async_client: AsyncClient):
        """Liveness probe should always return status=alive."""
        response = await async_client.get("/api/health/liveness")

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["status"] == "alive"

    @pytest.mark.asyncio
    async def test_liveness_response_envelope(self, async_client: AsyncClient):
        """Liveness response must use the standard ApiResponse wrapper."""
        response = await async_client.get("/api/health/liveness")
        body = response.json()
        assert body["success"] is True
        assert body["data"]["status"] == "alive"
        assert "timestamp" in body["data"]


# ---------------------------------------------------------------------------
# 4. Startup probe (database + Redis required)
# ---------------------------------------------------------------------------

class TestStartupCheck:
    """Tests for GET /api/health/startup -- Kubernetes startup probe."""

    @pytest.mark.asyncio
    async def test_startup_succeeds_when_services_available(
        self, async_client: AsyncClient
    ):
        """Startup probe succeeds when both database and Redis respond."""
        mock_engine = _mock_engine(select_ok=True)
        mock_redis = _mock_redis_client(ping_ok=True)

        with (
            patch("backend.api.routers.health.engine", mock_engine),
            patch(
                "backend.api.routers.health.get_redis_client",
                return_value=mock_redis,
            ),
        ):
            response = await async_client.get("/api/health/startup")

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["status"] == "started"

    @pytest.mark.asyncio
    async def test_startup_fails_when_database_unavailable(
        self, async_client: AsyncClient
    ):
        """Startup probe returns 503 when database is unreachable."""
        mock_engine = _mock_engine(select_ok=False)
        mock_redis = _mock_redis_client(ping_ok=True)

        with (
            patch("backend.api.routers.health.engine", mock_engine),
            patch(
                "backend.api.routers.health.get_redis_client",
                return_value=mock_redis,
            ),
        ):
            response = await async_client.get("/api/health/startup")

        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_startup_fails_when_redis_unavailable(
        self, async_client: AsyncClient
    ):
        """Startup probe returns 503 when Redis is unreachable."""
        mock_engine = _mock_engine(select_ok=True)
        mock_redis = _mock_redis_client(ping_ok=False)

        with (
            patch("backend.api.routers.health.engine", mock_engine),
            patch(
                "backend.api.routers.health.get_redis_client",
                return_value=mock_redis,
            ),
        ):
            response = await async_client.get("/api/health/startup")

        assert response.status_code == 503


# ---------------------------------------------------------------------------
# 5. Metrics endpoint
# ---------------------------------------------------------------------------

class TestMetricsEndpoint:
    """Tests for GET /api/health/metrics -- system and service metrics.

    All tests in this class mock psutil to avoid PermissionError from
    ``psutil.net_connections()`` on macOS / CI sandboxes.
    """

    def _apply_patches(self, stack, mock_engine, mock_redis):
        """Enter engine, redis, and psutil patches on *stack*."""
        stack.enter_context(patch("backend.api.routers.health.engine", mock_engine))
        stack.enter_context(
            patch("backend.api.routers.health.get_redis_client", return_value=mock_redis)
        )
        for target, mock_obj in _mock_psutil().items():
            stack.enter_context(patch(target, mock_obj))

    @pytest.mark.asyncio
    async def test_returns_system_metrics(self, async_client: AsyncClient):
        """Metrics endpoint should return CPU, memory, and disk data."""
        mock_engine = _mock_engine()
        mock_redis = _mock_redis_client()

        with contextlib.ExitStack() as stack:
            self._apply_patches(stack, mock_engine, mock_redis)
            response = await async_client.get("/api/health/metrics")

        assert response.status_code == 200
        data = response.json()["data"]

        # System metrics structure
        assert "system" in data
        system = data["system"]
        assert "cpu_percent" in system
        assert "memory" in system
        assert "disk" in system
        assert "network" in system

        # Memory sub-fields
        mem = system["memory"]
        assert "total" in mem
        assert "used" in mem
        assert "percent" in mem
        assert "available" in mem

        # Disk sub-fields
        disk = system["disk"]
        assert "total" in disk
        assert "used" in disk
        assert "free" in disk
        assert "percent" in disk

    @pytest.mark.asyncio
    async def test_includes_database_pool_stats(self, async_client: AsyncClient):
        """Metrics should include database connection pool statistics."""
        mock_engine = _mock_engine()
        mock_redis = _mock_redis_client()

        with contextlib.ExitStack() as stack:
            self._apply_patches(stack, mock_engine, mock_redis)
            response = await async_client.get("/api/health/metrics")

        data = response.json()["data"]
        assert "database_pool" in data

        pool = data["database_pool"]
        assert pool["size"] == 10
        assert pool["checked_in"] == 8
        assert pool["overflow"] == 0
        assert pool["total"] == 10

    @pytest.mark.asyncio
    async def test_includes_redis_info(self, async_client: AsyncClient):
        """Metrics should include Redis memory and command stats."""
        mock_engine = _mock_engine()
        mock_redis = _mock_redis_client(info={
            "used_memory_human": "2.00M",
            "connected_clients": 6,
            "total_commands_processed": 1200,
            "keyspace_hits": 800,
            "keyspace_misses": 100,
        })

        with contextlib.ExitStack() as stack:
            self._apply_patches(stack, mock_engine, mock_redis)
            response = await async_client.get("/api/health/metrics")

        data = response.json()["data"]
        assert "redis" in data

        redis_info = data["redis"]
        assert redis_info["used_memory"] == "2.00M"
        assert redis_info["connected_clients"] == 6
        assert redis_info["total_commands_processed"] == 1200
        assert redis_info["keyspace_hits"] == 800
        assert redis_info["keyspace_misses"] == 100

    @pytest.mark.asyncio
    async def test_metrics_resilient_when_pool_unavailable(
        self, async_client: AsyncClient
    ):
        """Metrics should still return system data even if pool stats fail."""
        mock_engine = MagicMock()
        mock_engine.pool.size.side_effect = Exception("Pool error")
        mock_redis = _mock_redis_client()

        with contextlib.ExitStack() as stack:
            self._apply_patches(stack, mock_engine, mock_redis)
            response = await async_client.get("/api/health/metrics")

        assert response.status_code == 200
        data = response.json()["data"]
        assert "system" in data
        # database_pool should be absent when pool stats fail
        assert "database_pool" not in data

    @pytest.mark.asyncio
    async def test_metrics_includes_timestamp(self, async_client: AsyncClient):
        """Metrics response must include a valid timestamp."""
        mock_engine = _mock_engine()
        mock_redis = _mock_redis_client()

        with contextlib.ExitStack() as stack:
            self._apply_patches(stack, mock_engine, mock_redis)
            response = await async_client.get("/api/health/metrics")

        data = response.json()["data"]
        parsed = datetime.fromisoformat(data["timestamp"])
        assert isinstance(parsed, datetime)


# ---------------------------------------------------------------------------
# 6. Ping endpoint
# ---------------------------------------------------------------------------

class TestPingEndpoint:
    """Tests for GET /api/health/ping -- simple connectivity test."""

    @pytest.mark.asyncio
    async def test_ping_returns_pong(self, async_client: AsyncClient):
        """Ping should return status=pong."""
        response = await async_client.get("/api/health/ping")

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["status"] == "pong"

    @pytest.mark.asyncio
    async def test_ping_response_format(self, async_client: AsyncClient):
        """Ping response should have ApiResponse wrapper and timestamp."""
        response = await async_client.get("/api/health/ping")
        body = response.json()

        assert body["success"] is True
        assert "timestamp" in body["data"]
