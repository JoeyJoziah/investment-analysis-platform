"""Wave 2 residual fabrication surface gates."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("SECRET_KEY", "test-secret-wave2")
os.environ.setdefault("JWT_SECRET_KEY", "test-jwt-wave2")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("TESTING", "True")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("MASTER_SECRET_KEY", "test-master-wave2")
os.environ.setdefault("DEMO_MODE", "false")


def _force_demo_mode(monkeypatch, enabled: bool) -> None:
    """Patch settings wherever Wave 2 code imports it."""
    from backend.config.settings import settings

    monkeypatch.setattr(settings, "DEMO_MODE", enabled, raising=False)
    monkeypatch.setenv("DEMO_MODE", "true" if enabled else "false")


def test_websocket_market_overview_refuses_when_not_demo(monkeypatch):
    _force_demo_mode(monkeypatch, False)
    from backend.services.websocket_service import generate_market_overview_data

    payload = generate_market_overview_data()
    assert payload.get("error") == "model_unavailable"
    assert payload.get("indices") == {}


def test_websocket_portfolio_update_refuses_when_not_demo(monkeypatch):
    _force_demo_mode(monkeypatch, False)
    from backend.services.websocket_service import generate_portfolio_update_data

    payload = generate_portfolio_update_data("pf-1")
    assert payload.get("error") == "model_unavailable"
    assert payload.get("positions") == []


def test_sample_recommendation_refuses_when_not_demo(monkeypatch):
    _force_demo_mode(monkeypatch, False)
    from backend.exceptions import ModelUnavailableError
    from backend.services.recommendation_service import RecommendationService

    svc = RecommendationService()
    with pytest.raises(ModelUnavailableError):
        svc.generate_sample_recommendation("AAPL")


def test_recommendation_crud_has_no_mixin_fabricator():
    import backend.services.recommendation_crud as crud

    assert not hasattr(crud, "RecommendationCrudMixin")
    assert crud.RECOMMENDATION_MODEL_VERSION


def test_monthly_returns_single_month_no_crash():
    import pandas as pd
    from backend.ml.backtesting import BacktestEngine

    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    rets = pd.Series([0.001] * len(idx), index=idx)
    engine = BacktestEngine(data_provider=None, allow_synthetic=False)
    matrix = engine._calculate_monthly_returns(rets)
    assert list(matrix.columns) == [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    assert not matrix.empty
