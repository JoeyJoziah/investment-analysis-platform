"""
Regression tests for P0 findings #199 and #200 — recommendations.py slice.

Finding #199 — Broken Access Control (recommendations.py):
  Six endpoints lacked auth; /portfolio/{portfolio_id} lacked ownership check.
  Fix: all nine non-daily endpoints now declare Depends(get_current_user);
       portfolio endpoint additionally verifies portfolio.user_id == current_user.id.

Finding #200 — Fabricated data returned on failure (recommendations.py):
  /daily and /trending catch blocks returned synthetic random financial data
  instead of signalling unavailability.
  Fix: failure path raises HTTP 503; synthetic path gated behind BOOTSTRAP_MODELS
  env flag (off in production).

Strategy
  - Route-signature tests confirm Depends(get_current_user) is wired in for every
    endpoint that was missing it (no app spin-up required).
  - Failure-path unit tests monkeypatch the upstream service call to raise, then
    assert the router raises HTTPException(503) rather than returning synthetic data.
  - Ownership unit test monkeypatches portfolio_repository to return a portfolio
    whose user_id differs from the requester and asserts 403.
"""

# CRITICAL: env vars must be set before any backend imports so that
# pydantic-settings can construct the Settings singleton without a real .env.
import os
os.environ.setdefault("TESTING", "True")
os.environ.setdefault("DEBUG", "True")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret-key-at-least-64-chars-long-for-pydantic-settings-ok")
os.environ.setdefault("JWT_SECRET_KEY", "test-jwt-secret-key-64chars-minimum-for-pydantic-settings-valid")
os.environ.setdefault("MASTER_SECRET_KEY", "test-master-secret-key-minimum-128-chars-long-padded-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")

import inspect
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import params as fa_params
from fastapi import HTTPException

from backend.auth.oauth2 import get_current_user


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _depends_on_get_current_user(func) -> bool:
    """Return True when *func* has a parameter whose Depends() dependency is
    get_current_user (the canonical backend.auth.oauth2 version)."""
    sig = inspect.signature(func)
    for param in sig.parameters.values():
        default = param.default
        if (
            isinstance(default, fa_params.Depends)
            and default.dependency is get_current_user
        ):
            return True
    return False


def _make_user(user_id: int = 1):
    """Build a minimal User-like object sufficient for ownership checks."""
    user = MagicMock()
    user.id = user_id
    return user


# ---------------------------------------------------------------------------
# #199 — Route-signature tests: all nine endpoints must declare auth
# ---------------------------------------------------------------------------

class TestRecommendationsEndpointSignatures:
    """Finding #199: every previously-unguarded endpoint must now declare
    current_user: User = Depends(get_current_user)."""

    def test_get_recommendations_list_requires_auth(self):
        from backend.api.routers.recommendations import get_recommendations
        assert _depends_on_get_current_user(get_recommendations), (
            "GET /list must declare Depends(get_current_user) — finding #199 regression"
        )

    def test_get_recommendation_detail_requires_auth(self):
        from backend.api.routers.recommendations import get_recommendation_detail
        assert _depends_on_get_current_user(get_recommendation_detail), (
            "GET /{recommendation_id} must declare Depends(get_current_user) — finding #199 regression"
        )

    def test_filter_recommendations_requires_auth(self):
        from backend.api.routers.recommendations import filter_recommendations
        assert _depends_on_get_current_user(filter_recommendations), (
            "POST /filter must declare Depends(get_current_user) — finding #199 regression"
        )

    def test_get_portfolio_recommendations_requires_auth(self):
        from backend.api.routers.recommendations import get_portfolio_recommendations
        assert _depends_on_get_current_user(get_portfolio_recommendations), (
            "GET /portfolio/{portfolio_id} must declare Depends(get_current_user) — finding #199 regression"
        )

    def test_track_recommendation_performance_requires_auth(self):
        from backend.api.routers.recommendations import track_recommendation_performance
        assert _depends_on_get_current_user(track_recommendation_performance), (
            "GET /performance/track must declare Depends(get_current_user) — finding #199 regression"
        )

    def test_update_alert_settings_requires_auth(self):
        from backend.api.routers.recommendations import update_alert_settings
        assert _depends_on_get_current_user(update_alert_settings), (
            "POST /alerts/settings must declare Depends(get_current_user) — finding #199 regression"
        )

    def test_get_alert_history_requires_auth(self):
        from backend.api.routers.recommendations import get_alert_history
        assert _depends_on_get_current_user(get_alert_history), (
            "GET /alerts/history must declare Depends(get_current_user) — finding #199 regression"
        )

    def test_backtest_strategy_requires_auth(self):
        from backend.api.routers.recommendations import backtest_strategy
        assert _depends_on_get_current_user(backtest_strategy), (
            "POST /backtest must declare Depends(get_current_user) — finding #199 regression"
        )

    def test_get_trending_recommendations_requires_auth(self):
        from backend.api.routers.recommendations import get_trending_recommendations
        assert _depends_on_get_current_user(get_trending_recommendations), (
            "GET /trending must declare Depends(get_current_user) — finding #199 regression"
        )


# ---------------------------------------------------------------------------
# #199 — Ownership enforcement on /portfolio/{portfolio_id}
# ---------------------------------------------------------------------------

class TestPortfolioOwnershipEnforcement:
    """Finding #199: /portfolio/{portfolio_id} must reject requests from users
    who do not own the requested portfolio."""

    @pytest.mark.asyncio
    async def test_foreign_portfolio_raises_403(self):
        """A valid authenticated user requesting another user's portfolio must get 403."""
        from backend.api.routers.recommendations import get_portfolio_recommendations

        requester = _make_user(user_id=1)

        # Portfolio exists but belongs to user 99, not the requester (user 1).
        foreign_portfolio = MagicMock()
        foreign_portfolio.user_id = 99

        mock_db = MagicMock()

        with patch(
            "backend.api.routers.recommendations.portfolio_repository.get_portfolio_with_positions",
            new=AsyncMock(return_value=foreign_portfolio),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_portfolio_recommendations(
                    portfolio_id="42",
                    current_user=requester,
                    db=mock_db,
                )

        assert exc_info.value.status_code == 403, (
            "Non-owner must receive 403 Forbidden — finding #199 ownership regression"
        )

    @pytest.mark.asyncio
    async def test_missing_portfolio_raises_404(self):
        """A portfolio_id that does not exist must yield 404."""
        from backend.api.routers.recommendations import get_portfolio_recommendations

        requester = _make_user(user_id=1)
        mock_db = MagicMock()

        with patch(
            "backend.api.routers.recommendations.portfolio_repository.get_portfolio_with_positions",
            new=AsyncMock(return_value=None),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_portfolio_recommendations(
                    portfolio_id="9999",
                    current_user=requester,
                    db=mock_db,
                )

        assert exc_info.value.status_code == 404, (
            "Missing portfolio must return 404 — finding #199 ownership regression"
        )

    @pytest.mark.asyncio
    async def test_non_numeric_portfolio_id_raises_404(self):
        """A non-numeric portfolio_id (slug-style) that can't be resolved must yield 404."""
        from backend.api.routers.recommendations import get_portfolio_recommendations

        requester = _make_user(user_id=1)
        mock_db = MagicMock()

        with pytest.raises(HTTPException) as exc_info:
            await get_portfolio_recommendations(
                portfolio_id="not-a-number",
                current_user=requester,
                db=mock_db,
            )

        assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# #200 — Failure path must raise 503, never return fabricated data
# ---------------------------------------------------------------------------

class TestNoFabricatedDataOnFailure:
    """Finding #200: when the upstream service fails, the router must raise
    HTTPException(503) rather than returning synthetic/random financial data."""

    @pytest.mark.asyncio
    async def test_daily_service_failure_raises_503_not_synthetic_data(self):
        """When build_daily_recommendations raises, the endpoint must propagate
        as HTTP 503 — never return a fabricated DailyRecommendations payload.

        Calls __wrapped__ to bypass the @cache_with_ttl decorator (which would
        attempt a Redis connection and block in a test environment).
        """
        from backend.api.routers.recommendations import get_daily_recommendations
        from datetime import date as date_type

        current_user = _make_user(user_id=1)
        mock_db = MagicMock()

        failing_service = MagicMock()
        failing_service.build_daily_recommendations = AsyncMock(
            side_effect=RuntimeError("upstream model unavailable")
        )

        # Bypass the @cache_with_ttl decorator via __wrapped__.
        # Ensure BOOTSTRAP_MODELS is NOT set so the 503 path is exercised.
        with patch.dict(os.environ, {}, clear=False) as patched_env:
            patched_env.pop("BOOTSTRAP_MODELS", None)
            with pytest.raises(HTTPException) as exc_info:
                await get_daily_recommendations.__wrapped__(
                    date_param=date_type.today(),
                    risk_level=None,
                    current_user=current_user,
                    db=mock_db,
                    rec_service=failing_service,
                )

        assert exc_info.value.status_code == 503, (
            "Service failure must yield HTTP 503, not fabricated data — finding #200 regression"
        )

    @pytest.mark.asyncio
    async def test_daily_bootstrap_flag_returns_empty_payload_not_random_data(self):
        """When BOOTSTRAP_MODELS is set, failure path returns an empty (non-random)
        placeholder — still no hardcoded random financial values."""
        from backend.api.routers.recommendations import get_daily_recommendations
        from datetime import date as date_type

        current_user = _make_user(user_id=1)
        mock_db = MagicMock()

        failing_service = MagicMock()
        failing_service.build_daily_recommendations = AsyncMock(
            side_effect=RuntimeError("model not loaded")
        )

        # Bypass @cache_with_ttl via __wrapped__ and set BOOTSTRAP_MODELS.
        with patch.dict(os.environ, {"BOOTSTRAP_MODELS": "1"}, clear=False):
            response = await get_daily_recommendations.__wrapped__(
                date_param=date_type.today(),
                risk_level=None,
                current_user=current_user,
                db=mock_db,
                rec_service=failing_service,
            )

        # Must be an ApiResponse wrapper, not an exception.
        assert response is not None
        # The top_picks list must be empty — no synthetic recommendations injected.
        data = response.data if hasattr(response, "data") else response["data"]
        assert data.top_picks == [], (
            "Bootstrap fallback must return empty top_picks, not fabricated recommendations"
        )
        assert data.watchlist == [], (
            "Bootstrap fallback must return empty watchlist, not hardcoded tickers"
        )

    @pytest.mark.asyncio
    async def test_trending_service_failure_raises_503_not_synthetic_data(self):
        """When get_trending raises, the endpoint must propagate as HTTP 503."""
        from backend.api.routers.recommendations import get_trending_recommendations

        current_user = _make_user(user_id=1)

        failing_service = MagicMock()
        failing_service.get_trending = AsyncMock(
            side_effect=RuntimeError("market data feed unavailable")
        )

        with pytest.raises(HTTPException) as exc_info:
            await get_trending_recommendations(
                timeframe="24h",
                limit=10,
                risk_tolerance="moderate",
                current_user=current_user,
                rec_service=failing_service,
            )

        assert exc_info.value.status_code == 503, (
            "Trending service failure must yield HTTP 503, not fabricated data — finding #200 regression"
        )

    @pytest.mark.asyncio
    async def test_trending_503_detail_does_not_leak_internal_error(self):
        """The 503 detail message must not expose raw exception text to callers."""
        from backend.api.routers.recommendations import get_trending_recommendations

        current_user = _make_user(user_id=1)
        internal_message = "INTERNAL_CREDENTIAL_db_pw_secret_leak"

        failing_service = MagicMock()
        failing_service.get_trending = AsyncMock(
            side_effect=RuntimeError(internal_message)
        )

        with pytest.raises(HTTPException) as exc_info:
            await get_trending_recommendations(
                timeframe="24h",
                limit=5,
                risk_tolerance="conservative",
                current_user=current_user,
                rec_service=failing_service,
            )

        assert internal_message not in exc_info.value.detail, (
            "503 detail must not leak raw exception text"
        )
