"""
Integration tests for the recommendations router endpoints.

Tests cover recommendation generation, SEC-compliant disclosures, methodology
transparency, risk warnings, input validation, filtering, performance tracking,
and error handling across all /api/recommendations/* endpoints.

GitHub Issue: #89
"""

import pytest
import pytest_asyncio
from datetime import date, timedelta
from unittest.mock import patch, AsyncMock

from backend.api.main import app
from backend.api.routers.recommendations import (
    SEC_RISK_WARNING,
    SEC_LIMITATIONS_STATEMENT,
    SEC_METHODOLOGY_DISCLOSURE_TEMPLATE,
    RECOMMENDATION_MODEL_VERSION,
    RECOMMENDATION_MODEL_TRAINING_DATE,
    generate_sec_disclosure,
    generate_recommendation,
    RecommendationType,
    RiskLevel,
    RecommendationCategory,
    TimeHorizon,
)
from backend.auth.oauth2 import get_current_user
from backend.models.unified_models import User
from httpx import AsyncClient


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unwrap(response, expected_status=200):
    """Assert ApiResponse wrapper and return the inner data payload."""
    assert response.status_code == expected_status, (
        f"Expected {expected_status}, got {response.status_code}: {response.text}"
    )
    body = response.json()
    assert body["success"] is True
    assert "data" in body
    return body["data"]


# ---------------------------------------------------------------------------
# 1. GET /api/recommendations/list -- basic retrieval
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_recommendations_returns_expected_structure(async_client: AsyncClient):
    """
    GET /api/recommendations/list returns a list of recommendations
    with the standard ApiResponse wrapper and each item carrying required fields.
    """
    response = await async_client.get("/api/v1/recommendations/list")
    data = _unwrap(response)

    assert isinstance(data, list)
    assert len(data) > 0

    first = data[0]
    required_fields = [
        "id", "symbol", "company_name", "recommendation_type",
        "confidence_score", "target_price", "current_price",
        "expected_return", "time_horizon", "risk_level",
        "reasoning", "key_factors", "technical_signals",
        "fundamental_metrics", "risk_factors",
        "entry_points", "exit_points", "stop_loss",
        "sector", "market_cap", "volume",
    ]
    for field in required_fields:
        assert field in first, f"Missing required field: {field}"


# ---------------------------------------------------------------------------
# 2. SEC-compliant disclaimer validation on individual recommendations
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sec_disclosure_present_on_recommendations(async_client: AsyncClient):
    """
    Each recommendation returned by /list must include an sec_disclosure
    object containing all SEC 2025 mandated fields.
    """
    response = await async_client.get("/api/v1/recommendations/list", params={"limit": 5})
    data = _unwrap(response)

    for rec in data:
        disclosure = rec.get("sec_disclosure")
        assert disclosure is not None, (
            f"Recommendation {rec['id']} missing sec_disclosure"
        )

        sec_required_fields = [
            "methodology_disclosure",
            "data_sources",
            "model_version",
            "model_training_date",
            "risk_warning",
            "limitations_statement",
            "confidence_level",
        ]
        for field in sec_required_fields:
            assert field in disclosure, (
                f"sec_disclosure missing required field: {field}"
            )


# ---------------------------------------------------------------------------
# 3. Risk warning text matches SEC standard
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_risk_warning_matches_sec_standard(async_client: AsyncClient):
    """
    The risk_warning field inside sec_disclosure must contain the
    exact SEC-mandated risk warning text defined in SEC_RISK_WARNING.
    """
    response = await async_client.get("/api/v1/recommendations/list", params={"limit": 1})
    data = _unwrap(response)

    disclosure = data[0]["sec_disclosure"]
    assert disclosure["risk_warning"] == SEC_RISK_WARNING


# ---------------------------------------------------------------------------
# 4. Methodology disclosure includes model version and training date
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_methodology_disclosure_contains_model_metadata(async_client: AsyncClient):
    """
    The methodology_disclosure field must reference the model version and
    training date so investors can assess staleness and provenance.
    """
    response = await async_client.get("/api/v1/recommendations/list", params={"limit": 1})
    data = _unwrap(response)

    disclosure = data[0]["sec_disclosure"]
    methodology = disclosure["methodology_disclosure"]

    assert RECOMMENDATION_MODEL_VERSION in methodology, (
        "Methodology disclosure must include model version"
    )
    assert RECOMMENDATION_MODEL_TRAINING_DATE in methodology, (
        "Methodology disclosure must include model training date"
    )
    assert disclosure["model_version"] == RECOMMENDATION_MODEL_VERSION
    assert disclosure["model_training_date"] == RECOMMENDATION_MODEL_TRAINING_DATE


# ---------------------------------------------------------------------------
# 5. Limitations statement present and accurate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_limitations_statement_present(async_client: AsyncClient):
    """
    The limitations_statement in sec_disclosure must match the canonical
    SEC_LIMITATIONS_STATEMENT to inform users of analysis boundaries.
    """
    response = await async_client.get("/api/v1/recommendations/list", params={"limit": 1})
    data = _unwrap(response)

    disclosure = data[0]["sec_disclosure"]
    assert disclosure["limitations_statement"] == SEC_LIMITATIONS_STATEMENT


# ---------------------------------------------------------------------------
# 6. Input validation -- limit and min_confidence boundaries
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_respects_limit_parameter(async_client: AsyncClient):
    """
    The limit query parameter constrains the number of returned items.
    """
    response = await async_client.get("/api/v1/recommendations/list", params={"limit": 3})
    data = _unwrap(response)
    assert len(data) <= 3


@pytest.mark.asyncio
async def test_list_rejects_limit_above_maximum(async_client: AsyncClient):
    """
    Requesting limit > 100 should return a 422 validation error.
    """
    response = await async_client.get(
        "/api/v1/recommendations/list", params={"limit": 200}
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_list_rejects_invalid_min_confidence(async_client: AsyncClient):
    """
    min_confidence must be between 0 and 1. Values outside that range
    should be rejected with a 422 validation error.
    """
    response = await async_client.get(
        "/api/v1/recommendations/list", params={"min_confidence": 1.5}
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 7. Filtering by recommendation type
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_filters_by_recommendation_type(async_client: AsyncClient):
    """
    When recommendation_type is specified, all returned items must match.
    """
    response = await async_client.get(
        "/api/v1/recommendations/list",
        params={"recommendation_type": "buy", "limit": 50},
    )
    data = _unwrap(response)

    for rec in data:
        assert rec["recommendation_type"] == "buy"


# ---------------------------------------------------------------------------
# 8. Filtering by risk level
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_filters_by_risk_level(async_client: AsyncClient):
    """
    When risk_level is specified, all returned items must match.
    """
    response = await async_client.get(
        "/api/v1/recommendations/list",
        params={"risk_level": "conservative", "limit": 50},
    )
    data = _unwrap(response)

    for rec in data:
        assert rec["risk_level"] == "conservative"


# ---------------------------------------------------------------------------
# 9. GET /api/recommendations/{recommendation_id} -- detail endpoint
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_recommendation_by_id(async_client: AsyncClient):
    """
    Fetching a recommendation by its ID returns a single detail object
    with the matching ID and full SEC disclosure.
    """
    rec_id = "REC-TEST-001"
    response = await async_client.get(f"/api/v1/recommendations/{rec_id}")
    data = _unwrap(response)

    assert data["id"] == rec_id
    assert "sec_disclosure" in data
    assert data["sec_disclosure"]["risk_warning"] == SEC_RISK_WARNING


# ---------------------------------------------------------------------------
# 10. POST /api/recommendations/filter -- advanced filtering
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_advanced_filter_by_category_and_risk(async_client: AsyncClient):
    """
    POST /filter accepts a RecommendationFilter body and returns
    only matching recommendations.
    """
    filter_body = {
        "categories": ["growth"],
        "risk_levels": ["moderate"],
    }
    response = await async_client.post(
        "/api/v1/recommendations/filter",
        json=filter_body,
        params={"limit": 50},
    )
    data = _unwrap(response)

    for rec in data:
        assert rec["category"] == "growth"
        assert rec["risk_level"] == "moderate"


@pytest.mark.asyncio
async def test_advanced_filter_min_confidence(async_client: AsyncClient):
    """
    min_confidence in the filter body should exclude low-confidence results.
    """
    filter_body = {"min_confidence": 0.8}
    response = await async_client.post(
        "/api/v1/recommendations/filter", json=filter_body
    )
    data = _unwrap(response)

    for rec in data:
        assert rec["confidence_score"] >= 0.8


# ---------------------------------------------------------------------------
# 11. GET /api/recommendations/portfolio/{portfolio_id}
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_portfolio_recommendations(async_client: AsyncClient):
    """
    Portfolio-specific endpoint returns recommendations together with
    rebalancing suggestions and diversification metrics.
    """
    response = await async_client.get("/api/v1/recommendations/portfolio/port-123")
    data = _unwrap(response)

    assert data["portfolio_id"] == "port-123"
    assert "recommendations" in data
    assert isinstance(data["recommendations"], list)
    assert "rebalancing_suggestions" in data
    assert "risk_score" in data
    assert "diversification_score" in data
    assert "expected_portfolio_return" in data

    # Each nested recommendation must carry SEC disclosure
    for rec in data["recommendations"]:
        assert "sec_disclosure" in rec


# ---------------------------------------------------------------------------
# 12. GET /api/recommendations/performance/track
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_performance_tracking_returns_metrics(async_client: AsyncClient):
    """
    Performance tracking endpoint returns historical recommendation
    metrics including actual vs expected returns.
    """
    response = await async_client.get(
        "/api/v1/recommendations/performance/track", params={"days_back": 30}
    )
    data = _unwrap(response)

    assert isinstance(data, list)
    assert len(data) > 0

    first = data[0]
    perf_fields = [
        "recommendation_id", "symbol", "recommended_date",
        "recommendation_type", "entry_price", "current_price",
        "target_price", "actual_return", "expected_return",
        "days_since_recommendation", "status", "performance_rating",
    ]
    for field in perf_fields:
        assert field in first, f"Missing performance field: {field}"


@pytest.mark.asyncio
async def test_performance_tracking_filters_by_status(async_client: AsyncClient):
    """
    When status is specified, only matching performance records are returned.
    """
    response = await async_client.get(
        "/api/v1/recommendations/performance/track",
        params={"status": "active"},
    )
    data = _unwrap(response)

    for perf in data:
        assert perf["status"] == "active"


# ---------------------------------------------------------------------------
# 13. POST /api/recommendations/alerts/settings
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_update_alert_settings(async_client: AsyncClient):
    """
    Posting alert settings returns a success acknowledgement.
    """
    settings_body = {
        "email_notifications": True,
        "push_notifications": False,
        "alert_types": ["strong_buy", "target_reached"],
        "min_confidence": 0.75,
        "categories": ["growth", "value"],
    }
    response = await async_client.post(
        "/api/v1/recommendations/alerts/settings", json=settings_body
    )
    data = _unwrap(response)
    assert data["status"] == "success"


# ---------------------------------------------------------------------------
# 14. GET /api/recommendations/alerts/history
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_alert_history_returns_sorted_alerts(async_client: AsyncClient):
    """
    Alert history endpoint returns alerts sorted by timestamp descending.
    """
    response = await async_client.get(
        "/api/v1/recommendations/alerts/history", params={"days_back": 7}
    )
    data = _unwrap(response)

    assert isinstance(data, list)
    assert len(data) > 0

    # Verify descending timestamp order
    timestamps = [a["timestamp"] for a in data]
    assert timestamps == sorted(timestamps, reverse=True)


# ---------------------------------------------------------------------------
# 15. POST /api/recommendations/backtest -- strategy backtesting
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_backtest_strategy_returns_performance_summary(async_client: AsyncClient):
    """
    Backtest endpoint returns a complete performance summary including
    return metrics, trade statistics, and risk measures.
    """
    response = await async_client.post(
        "/api/v1/recommendations/backtest",
        params={
            "strategy": "growth",
            "start_date": "2025-01-01",
            "end_date": "2025-12-31",
            "initial_capital": 100000,
        },
    )
    data = _unwrap(response)

    backtest_fields = [
        "strategy", "period", "initial_capital", "final_value",
        "total_return", "annualized_return", "sharpe_ratio",
        "max_drawdown", "win_rate", "total_trades",
    ]
    for field in backtest_fields:
        assert field in data, f"Backtest missing field: {field}"

    assert data["strategy"] == "growth"
    assert data["initial_capital"] == 100000


@pytest.mark.asyncio
async def test_backtest_rejects_invalid_strategy(async_client: AsyncClient):
    """
    An invalid strategy enum value should produce a 422 validation error.
    """
    response = await async_client.post(
        "/api/v1/recommendations/backtest",
        params={
            "strategy": "not_a_real_strategy",
            "start_date": "2025-01-01",
            "end_date": "2025-12-31",
        },
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 16. GET /api/recommendations/trending
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_trending_recommendations(async_client: AsyncClient):
    """
    Trending endpoint returns recommendations ranked by trending_score.
    """
    response = await async_client.get(
        "/api/v1/recommendations/trending", params={"timeframe": "24h"}
    )
    data = _unwrap(response)

    assert isinstance(data, list)
    assert len(data) > 0

    # Verify descending trending_score order
    scores = [item["trending_score"] for item in data]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_trending_rejects_invalid_timeframe(async_client: AsyncClient):
    """
    An invalid timeframe value should produce a 422 validation error.
    """
    response = await async_client.get(
        "/api/v1/recommendations/trending", params={"timeframe": "99y"}
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 17. sort_by validation on /list
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_rejects_invalid_sort_by(async_client: AsyncClient):
    """
    sort_by only accepts confidence_score, expected_return, or created_at.
    Other values should produce a 422.
    """
    response = await async_client.get(
        "/api/v1/recommendations/list", params={"sort_by": "hacked_field"}
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 18. Unit-level: generate_sec_disclosure helper
# ---------------------------------------------------------------------------

def test_generate_sec_disclosure_confidence_levels():
    """
    The generate_sec_disclosure helper must map confidence scores to the
    correct confidence_level string (low / moderate / high).
    """
    low = generate_sec_disclosure(confidence_score=0.4)
    assert low.confidence_level == "low"

    moderate = generate_sec_disclosure(confidence_score=0.7)
    assert moderate.confidence_level == "moderate"

    high = generate_sec_disclosure(confidence_score=0.9)
    assert high.confidence_level == "high"


def test_generate_sec_disclosure_contains_all_required_fields():
    """
    Every SECDisclosure object must include all fields mandated by
    SEC 2025 regulations for algorithmic investment advice.
    """
    disclosure = generate_sec_disclosure()
    assert disclosure.risk_warning == SEC_RISK_WARNING
    assert disclosure.limitations_statement == SEC_LIMITATIONS_STATEMENT
    assert disclosure.model_version == RECOMMENDATION_MODEL_VERSION
    assert disclosure.model_training_date == RECOMMENDATION_MODEL_TRAINING_DATE
    assert isinstance(disclosure.data_sources, list)
    assert len(disclosure.data_sources) > 0
    assert disclosure.conflict_of_interest_statement is not None


def test_generate_sec_disclosure_methodology_references_algorithm():
    """
    Custom algorithm_type parameter must appear in the methodology disclosure.
    """
    disclosure = generate_sec_disclosure(algorithm_type="deep reinforcement learning")
    assert "deep reinforcement learning" in disclosure.methodology_disclosure


# ---------------------------------------------------------------------------
# 19. Daily recommendations (authenticated endpoint)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_daily_recommendations_requires_auth(async_client: AsyncClient):
    """
    GET /api/recommendations/daily without a valid token should return 401.
    """
    response = await async_client.get("/api/v1/recommendations/daily")
    assert response.status_code in (401, 403)


@pytest.mark.asyncio
async def test_daily_recommendations_with_auth(
    async_client: AsyncClient,
    auth_headers: dict,
    test_user: User,
):
    """
    GET /api/recommendations/daily with valid auth returns daily
    curated picks including SEC global disclosure fields.
    """
    # Override get_current_user so the endpoint receives a real User
    app.dependency_overrides[get_current_user] = lambda: test_user

    try:
        response = await async_client.get(
            "/api/v1/recommendations/daily", headers=auth_headers
        )
        # The endpoint may fall back to mock data if DB has no stocks,
        # which is acceptable -- we validate the response structure.
        data = _unwrap(response)

        daily_fields = [
            "date", "market_outlook", "top_picks", "watchlist",
            "avoid_list", "sector_focus", "market_sentiment",
            "risk_assessment", "sec_global_disclosure",
            "data_as_of", "recommendation_model_version",
        ]
        for field in daily_fields:
            assert field in data, f"Daily response missing field: {field}"

        # SEC global disclosure must match the canonical risk warning
        assert data["sec_global_disclosure"] == SEC_RISK_WARNING

        # Model version must be present
        assert data["recommendation_model_version"] == RECOMMENDATION_MODEL_VERSION

        # top_picks must be a list
        assert isinstance(data["top_picks"], list)

        # market_sentiment must be in [-1, 1]
        assert -1.0 <= data["market_sentiment"] <= 1.0
    finally:
        app.dependency_overrides.pop(get_current_user, None)
