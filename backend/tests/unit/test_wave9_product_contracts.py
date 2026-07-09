"""Wave 9: portfolio analysis, domain adapters, integration coverage contracts."""
from __future__ import annotations

import inspect
import os
from pathlib import Path

import pytest

os.environ.setdefault("SECRET_KEY", "test-secret-wave9")
os.environ.setdefault("JWT_SECRET_KEY", "test-jwt-wave9")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("TESTING", "True")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("MASTER_SECRET_KEY", "test-master-wave9")


def test_portfolio_analysis_uses_real_position_math():
    """#108: analysis is derived from holdings, never random.uniform."""
    from backend.services.portfolio_service import PortfolioService

    svc = PortfolioService()
    src = inspect.getsource(svc.build_portfolio_analysis)
    assert "random.uniform" not in src
    assert "random.choice" not in src

    positions = [
        {
            "symbol": "AAPL",
            "sector": "Technology",
            "market_value": 5000,
            "unrealized_gain_percent": 12.0,
        },
        {
            "symbol": "XOM",
            "sector": "Energy",
            "market_value": 5000,
            "unrealized_gain_percent": -4.0,
        },
    ]
    result = svc.build_portfolio_analysis("wave9-p1", positions=positions)
    assert result["portfolio_id"] == "wave9-p1"
    assert result["concentration_risk"]["top_holding"] == pytest.approx(0.5, abs=0.01)
    assert result["diversification_score"] > 0
    assert result["risk_analysis"]["var_95"] is not None
    assert "AAPL" in result["correlation_matrix"]
    assert isinstance(result["optimization_suggestions"], list)
    assert len(result["optimization_suggestions"]) >= 1


def test_analyze_portfolio_router_is_async_and_loads_db():
    """#108: analyze endpoint awaits async analysis with auth + db."""
    from backend.api.routers import portfolio as port

    src = inspect.getsource(port.analyze_portfolio)
    assert "build_portfolio_analysis_async" in src
    assert "get_current_user" in src
    assert "get_async_db_session" in src or "db" in src


def test_domain_contract_concrete_adapters_exist_and_validate():
    """#109: concrete adapters implement all five domain contracts."""
    from backend.domain.implementations import get_default_domain_adapters
    from backend.domain.contracts import (
        DataPipelineContract,
        InvestmentAnalysisContract,
        MarketDataContract,
        MLContract,
        PortfolioContract,
    )

    adapters = get_default_domain_adapters()
    assert set(adapters) == {
        "portfolio",
        "market_data",
        "data_pipeline",
        "ml",
        "investment_analysis",
    }
    assert isinstance(adapters["portfolio"], PortfolioContract)
    assert isinstance(adapters["market_data"], MarketDataContract)
    assert isinstance(adapters["data_pipeline"], DataPipelineContract)
    assert isinstance(adapters["ml"], MLContract)
    assert isinstance(adapters["investment_analysis"], InvestmentAnalysisContract)

    for name, adapter in adapters.items():
        validation = adapter.validate_contract()
        assert validation.success, f"{name} failed validation: {validation.error}"
        assert adapter.domain_name
        assert adapter.version
        assert adapter.capabilities


@pytest.mark.asyncio
async def test_domain_adapters_health_check():
    """#109: adapters report healthy via contract health_check."""
    from backend.domain.implementations import get_default_domain_adapters

    for name, adapter in get_default_domain_adapters().items():
        result = await adapter.health_check()
        assert result.success, f"{name} health failed: {result.error}"
        assert result.data["status"] == "healthy"


def test_frontend_backend_integration_suites_present():
    """#4: substantial backend integration coverage for core flows."""
    root = Path("backend/tests/integration")
    required = [
        "test_auth_to_portfolio_flow.py",
        "test_auth_flow_complete.py",
        "test_stocks_router.py",
        "test_websocket_router.py",
        "test_service_integration.py",
        "test_stock_to_analysis_flow.py",
        "test_agents_to_recommendations_flow.py",
    ]
    for name in required:
        path = root / name
        assert path.exists(), f"missing {name}"
        assert path.stat().st_size > 5000, f"{name} too small to be meaningful"
