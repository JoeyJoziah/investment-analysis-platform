"""Wave 7: product/testing backlog contracts already largely implemented."""
from __future__ import annotations

import inspect
import os
from pathlib import Path

os.environ.setdefault("SECRET_KEY", "test-secret-wave7")
os.environ.setdefault("JWT_SECRET_KEY", "test-jwt-wave7")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("TESTING", "True")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("MASTER_SECRET_KEY", "test-master-wave7")


def test_ci_has_backend_coverage_reporting():
    """#93: CI coverage reporting + floor gate."""
    ci = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "codecov" in ci.lower() or "codecov-action" in ci
    assert "--cov" in ci or "coverage-" in ci
    assert "60" in ci  # blocking floor documented in workflow


def test_stocks_search_endpoint_exists():
    """#39: search stocks by ticker/name."""
    from backend.api.routers import stocks as stocks_mod

    source = inspect.getsource(stocks_mod)
    assert '@router.get("/search")' in source or 'get("/search")' in source
    assert "search_stocks" in source


def test_portfolio_add_position_exists_and_is_authed():
    """#39: add to portfolio with auth."""
    from backend.api.routers import portfolio as port

    source = inspect.getsource(port.add_position)
    assert "get_current_user" in source
    assert "AddPositionRequest" in inspect.getsource(port) or "price" in source


def test_integration_tests_exist_for_core_routers():
    """#87–#89: integration test modules present."""
    root = Path("backend/tests/integration")
    assert (root / "test_stocks_router.py").exists()
    assert (root / "test_analysis_router.py").exists()
    assert (root / "test_recommendations_router.py").exists()
    # Non-trivial size
    assert (root / "test_stocks_router.py").stat().st_size > 1000
    assert (root / "test_analysis_router.py").stat().st_size > 1000
    assert (root / "test_recommendations_router.py").stat().st_size > 1000


def test_no_stocks_legacy_module():
    """#98: legacy stocks_legacy module should be gone."""
    assert not Path("backend/api/routers/stocks_legacy.py").exists()
    assert not Path("backend/stocks_legacy.py").exists()
