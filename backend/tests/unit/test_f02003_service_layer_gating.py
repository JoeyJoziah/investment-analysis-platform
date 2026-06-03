"""F-02-003 fail-first regression tests for remaining random-data service methods.

Per PRD audit 2026-04 Workstream D §3 Step 2 (Q4 default, recorded 2026-04-28),
each call site listed in F-02-003 must either return real data or refuse with
HTTP 503 ``model_unavailable`` in production. The router layer already gates
``/users``, ``/backtest``, ``/performance/track``, and ``/alerts/history`` at
HTTP boundary via ``_refuse_when_models_in_fallback``. These tests pin the
defense-in-depth contract at the **service layer**, matching the established
``portfolio_rebalancing.generate_performance_data_points`` pattern: when
``settings.DEMO_MODE`` is False (production default), the service method
itself raises ``ModelUnavailableError`` rather than fabricating values for
internal callers, future test code, or scripts that bypass the router.

Pre-fix expectation: these tests FAIL because the four methods unconditionally
return ``random.uniform/randint/choice`` data regardless of DEMO_MODE.
"""
from __future__ import annotations

import pytest

from backend.config.settings import settings
from backend.exceptions import ModelUnavailableError


@pytest.fixture
def production_mode(monkeypatch):
    """Force DEMO_MODE=False so production-gate paths execute."""
    monkeypatch.setattr(settings, "DEMO_MODE", False)
    yield


@pytest.fixture
def demo_mode(monkeypatch):
    """Force DEMO_MODE=True so legacy synthetic paths remain available."""
    monkeypatch.setattr(settings, "DEMO_MODE", True)
    yield


class TestAdminListUsersF02003:
    """F-02-003: admin_service.list_users must refuse in production."""

    def test_refuses_in_production(self, production_mode):
        from backend.services.admin_service import list_users
        with pytest.raises(ModelUnavailableError) as exc:
            list_users(limit=10)
        assert exc.value.model in {"admin_user_directory", "user_directory"}
        assert exc.value.reason in {
            "not_implemented",
            "fallback_active",
            "live_feed_not_configured",
        }

    def test_returns_synthetic_in_demo_mode(self, demo_mode):
        from backend.services.admin_service import list_users
        users = list_users(limit=5)
        assert isinstance(users, list)
        assert len(users) <= 5


class TestRecommendationServiceRunBacktestF02003:
    """F-02-003: RecommendationService.run_backtest must refuse in production."""

    def test_refuses_in_production(self, production_mode):
        from datetime import date, timedelta
        from backend.services.recommendation_service import RecommendationService
        svc = RecommendationService()
        with pytest.raises(ModelUnavailableError):
            svc.run_backtest(
                strategy="growth",
                start_date=date.today() - timedelta(days=180),
                end_date=date.today(),
            )

    def test_returns_synthetic_in_demo_mode(self, demo_mode):
        from datetime import date, timedelta
        from backend.services.recommendation_service import RecommendationService
        svc = RecommendationService()
        result = svc.run_backtest(
            strategy="growth",
            start_date=date.today() - timedelta(days=180),
            end_date=date.today(),
        )
        assert isinstance(result, dict)


class TestRecommendationServiceGeneratePerformanceRecordsF02003:
    """F-02-003: generate_performance_records must refuse in production."""

    def test_refuses_in_production(self, production_mode):
        from backend.services.recommendation_service import RecommendationService
        svc = RecommendationService()
        with pytest.raises(ModelUnavailableError):
            svc.generate_performance_records(days_back=30)

    def test_returns_synthetic_in_demo_mode(self, demo_mode):
        from backend.services.recommendation_service import RecommendationService
        svc = RecommendationService()
        result = svc.generate_performance_records(days_back=7)
        assert isinstance(result, list)


class TestRecommendationServiceGenerateAlertHistoryF02003:
    """F-02-003: generate_alert_history must refuse in production."""

    def test_refuses_in_production(self, production_mode):
        from backend.services.recommendation_service import RecommendationService
        svc = RecommendationService()
        with pytest.raises(ModelUnavailableError):
            svc.generate_alert_history(days_back=7)

    def test_returns_synthetic_in_demo_mode(self, demo_mode):
        from backend.services.recommendation_service import RecommendationService
        svc = RecommendationService()
        result = svc.generate_alert_history(days_back=7)
        assert isinstance(result, list)
