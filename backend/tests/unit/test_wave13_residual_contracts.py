"""Wave 13: residual anti-fabrication, compliance extract, module size progress."""
from __future__ import annotations

import inspect
import os
from pathlib import Path

import pytest

os.environ.setdefault("SECRET_KEY", "test-secret-wave13")
os.environ.setdefault("JWT_SECRET_KEY", "test-jwt-wave13")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("TESTING", "True")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("MASTER_SECRET_KEY", "test-master-wave13")


def test_portfolio_rebalancing_refuses_outside_demo_mode(monkeypatch):
    """Wave 13: leftover rebalancing helpers refuse fabrication in production."""
    from backend.config.settings import settings
    from backend.exceptions import ModelUnavailableError
    from backend.services import portfolio_rebalancing as pr

    monkeypatch.setattr(settings, "DEMO_MODE", False)

    with pytest.raises(ModelUnavailableError):
        pr.build_portfolio_analysis("p1")
    with pytest.raises(ModelUnavailableError):
        pr.generate_rebalancing_trades("p1", {"Equities": 100}, 5, 100.0, False)
    with pytest.raises(ModelUnavailableError):
        pr.generate_transaction_list("p1", 10, 0, None, None, None, None)
    with pytest.raises(ModelUnavailableError):
        pr.generate_performance_data_points("p1", "1M")


def test_portfolio_rebalancing_demo_mode_tags_simulated(monkeypatch):
    """DEMO_MODE may synthesize data but must tag data_source=simulated."""
    from backend.config.settings import settings
    from backend.services import portfolio_rebalancing as pr

    monkeypatch.setattr(settings, "DEMO_MODE", True)
    perf = pr.generate_performance_data_points("p1", "1W")
    assert perf.get("data_source") == "simulated"
    analysis = pr.build_portfolio_analysis("p2")
    assert analysis.get("data_source") == "simulated"


def test_recommendation_compliance_module_extracted():
    """Wave 13: SEC constants live in recommendation_compliance."""
    from backend.services import recommendation_compliance as rc
    from backend.services.recommendation_service import RecommendationService

    assert rc.SEC_RISK_WARNING
    assert "Past performance" in rc.SEC_RISK_WARNING
    disc = rc.build_sec_disclosure(confidence_score=0.9)
    assert disc["confidence_level"] == "high"
    assert disc["risk_warning"] == rc.SEC_RISK_WARNING

    # Service still exposes same API
    svc = RecommendationService()
    disc2 = svc.generate_sec_disclosure(confidence_score=0.7)
    assert disc2["confidence_level"] == "moderate"
    assert "build_sec_disclosure" in inspect.getsource(svc.generate_sec_disclosure)


def test_primary_portfolio_analysis_still_non_random():
    """Regression: PortfolioService analysis remains holdings-based (wave 9)."""
    from backend.services.portfolio_service import PortfolioService

    src = inspect.getsource(PortfolioService.build_portfolio_analysis)
    assert "random.uniform" not in src
    svc = PortfolioService()
    result = svc.build_portfolio_analysis(
        "w13",
        positions=[
            {"symbol": "A", "sector": "T", "market_value": 50, "unrealized_gain_percent": 1},
            {"symbol": "B", "sector": "E", "market_value": 50, "unrealized_gain_percent": -1},
        ],
    )
    assert result["concentration_risk"]["top_holding"] == pytest.approx(0.5, abs=0.01)


def test_wave_stack_contract_suite_present():
    """Waves 2–13 contract modules exist for continuous verification."""
    unit = Path("backend/tests/unit")
    for name in [
        "test_wave2_no_fabrication_surfaces.py",
        "test_wave11_ops_contracts.py",
        "test_wave12_ops_contracts.py",
        "test_wave13_residual_contracts.py",
    ]:
        assert (unit / name).exists(), name


def test_security_config_remains_under_800_after_wave12():
    """Regression: security_config stays ≤800 lines."""
    path = Path("backend/security/security_config.py")
    lines = len(path.read_text(encoding="utf-8").splitlines())
    assert lines <= 800, lines
