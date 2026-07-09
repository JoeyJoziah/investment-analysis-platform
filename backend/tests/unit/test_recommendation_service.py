"""
Unit tests for backend/services/recommendation_service.py

Tests all public methods of RecommendationService with mocked dependencies.
No database or external services required.
"""

import pytest
import random
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from backend.services.recommendation_service import (
    RecommendationService,
    SEC_RISK_WARNING,
    SEC_LIMITATIONS_STATEMENT,
    SEC_METHODOLOGY_DISCLOSURE_TEMPLATE,
    RECOMMENDATION_MODEL_VERSION,
    RECOMMENDATION_MODEL_TRAINING_DATE,
)


# ---------------------------------------------------------------------------
# Fixture: fresh RecommendationService (engines mocked to avoid heavy init)
# ---------------------------------------------------------------------------

@pytest.fixture
def service():
    """Return a RecommendationService with mocked engines so no real
    analytics modules are initialised.

    Because the __init__.py of backend.services shadows the module name
    with the singleton instance, we patch using sys.modules directly."""
    import sys
    mod = sys.modules["backend.services.recommendation_service"]
    # T1.2 (D1): the synthetic generator now refuses to run unless DEMO_MODE is
    # on. These unit tests exercise that synthetic structure, so enable demo mode
    # here; the production fail-loud contract is asserted separately below.
    with patch.object(mod, "RecommendationEngine", return_value=MagicMock()), \
         patch.object(mod, "FundamentalAnalysisEngine", return_value=MagicMock()), \
         patch.object(mod.settings, "DEMO_MODE", True):
        svc = RecommendationService()
        yield svc


# =========================================================================
# generate_sample_recommendation — production fail-loud contract (T1.2/D1)
# =========================================================================

class TestGenerateSampleRecommendationFailLoud:

    def test_refuses_in_production_mode(self):
        """With DEMO_MODE off (production default) the synthetic generator must
        raise ModelUnavailableError (-> HTTP 503) instead of fabricating data."""
        import sys
        from backend.exceptions import ModelUnavailableError
        mod = sys.modules["backend.services.recommendation_service"]
        with patch.object(mod, "RecommendationEngine", return_value=MagicMock()), \
             patch.object(mod, "FundamentalAnalysisEngine", return_value=MagicMock()), \
             patch.object(mod.settings, "DEMO_MODE", False):
            svc = RecommendationService()
            with pytest.raises(ModelUnavailableError) as excinfo:
                svc.generate_sample_recommendation(symbol="AAPL")
            assert excinfo.value.model == "recommendation_engine"


# =========================================================================
# generate_sec_disclosure
# =========================================================================

class TestGenerateSecDisclosure:

    def test_returns_all_required_fields(self, service):
        """Disclosure dict must contain every SEC-mandated field."""
        result = service.generate_sec_disclosure()
        required_keys = {
            "methodology_disclosure",
            "data_sources",
            "model_version",
            "model_training_date",
            "risk_warning",
            "limitations_statement",
            "confidence_level",
            "conflict_of_interest_statement",
        }
        assert required_keys.issubset(result.keys())

    def test_risk_warning_matches_constant(self, service):
        """The risk_warning field must be the exact SEC constant."""
        result = service.generate_sec_disclosure()
        assert result["risk_warning"] == SEC_RISK_WARNING

    def test_limitations_statement_matches_constant(self, service):
        """The limitations_statement must match the module-level constant."""
        result = service.generate_sec_disclosure()
        assert result["limitations_statement"] == SEC_LIMITATIONS_STATEMENT

    def test_model_version_matches_constant(self, service):
        """Model version and training date must match module constants."""
        result = service.generate_sec_disclosure()
        assert result["model_version"] == RECOMMENDATION_MODEL_VERSION
        assert result["model_training_date"] == RECOMMENDATION_MODEL_TRAINING_DATE

    def test_high_confidence_level(self, service):
        """Confidence >= 0.8 should map to 'high'."""
        result = service.generate_sec_disclosure(confidence_score=0.85)
        assert result["confidence_level"] == "high"

    def test_moderate_confidence_level(self, service):
        """Confidence in [0.6, 0.8) should map to 'moderate'."""
        result = service.generate_sec_disclosure(confidence_score=0.65)
        assert result["confidence_level"] == "moderate"

    def test_low_confidence_level(self, service):
        """Confidence < 0.6 should map to 'low'."""
        result = service.generate_sec_disclosure(confidence_score=0.4)
        assert result["confidence_level"] == "low"

    def test_custom_algorithm_type_in_methodology(self, service):
        """The algorithm_type parameter should appear in the methodology text."""
        result = service.generate_sec_disclosure(algorithm_type="deep neural network")
        assert "deep neural network" in result["methodology_disclosure"]

    def test_custom_data_sources_passthrough(self, service):
        """When explicit data_sources are provided they should be used as-is."""
        custom_sources = ["Source A", "Source B"]
        result = service.generate_sec_disclosure(data_sources=custom_sources)
        assert result["data_sources"] == custom_sources

    def test_default_data_sources_populated(self, service):
        """When no data_sources given, the default list should be non-empty."""
        result = service.generate_sec_disclosure()
        assert isinstance(result["data_sources"], list)
        assert len(result["data_sources"]) >= 1

    def test_conflict_of_interest_present(self, service):
        """Conflict-of-interest statement must be a non-empty string."""
        result = service.generate_sec_disclosure()
        assert isinstance(result["conflict_of_interest_statement"], str)
        assert len(result["conflict_of_interest_statement"]) > 0


# =========================================================================
# generate_sample_recommendation
# =========================================================================

class TestGenerateSampleRecommendation:

    def test_returns_dict_with_required_keys(self, service):
        """Sample recommendation must have core investment fields."""
        rec = service.generate_sample_recommendation(symbol="AAPL")
        required = {
            "id", "symbol", "company_name", "recommendation_type",
            "category", "confidence_score", "target_price", "current_price",
            "expected_return", "time_horizon", "risk_level", "created_at",
            "valid_until", "reasoning", "key_factors", "technical_signals",
            "fundamental_metrics", "risk_factors", "entry_points",
            "exit_points", "stop_loss", "sector", "sec_disclosure",
        }
        assert required.issubset(rec.keys())

    def test_specified_symbol_used(self, service):
        """When a symbol is explicitly provided it must appear in the output."""
        rec = service.generate_sample_recommendation(symbol="TSLA")
        assert rec["symbol"] == "TSLA"

    def test_random_symbol_when_none(self, service):
        """When no symbol given, a random valid ticker should be chosen."""
        rec = service.generate_sample_recommendation(symbol=None)
        valid_symbols = {"AAPL", "GOOGL", "MSFT", "AMZN", "META",
                         "NVDA", "TSLA", "JPM", "V", "JNJ"}
        assert rec["symbol"] in valid_symbols

    def test_confidence_score_in_range(self, service):
        """Confidence score should be between 0.6 and 0.95."""
        rec = service.generate_sample_recommendation()
        assert 0.6 <= rec["confidence_score"] <= 0.95

    def test_sec_disclosure_embedded(self, service):
        """Sample recommendation must include an embedded SEC disclosure dict."""
        rec = service.generate_sample_recommendation()
        assert "sec_disclosure" in rec
        assert "risk_warning" in rec["sec_disclosure"]

    def test_valid_until_after_created_at(self, service):
        """valid_until must be later than created_at."""
        rec = service.generate_sample_recommendation()
        assert rec["valid_until"] > rec["created_at"]

    def test_recommendation_type_valid(self, service):
        """recommendation_type must be one of the five known values."""
        rec = service.generate_sample_recommendation()
        valid_types = {"strong_buy", "buy", "hold", "sell", "strong_sell"}
        assert rec["recommendation_type"] in valid_types


# =========================================================================
# build_daily_recommendations
# =========================================================================

class TestBuildDailyRecommendations:

    @pytest.mark.asyncio
    async def test_returns_expected_top_level_keys(self, service):
        """Daily digest must contain outlook, picks, watchlist, etc."""
        # Stub the two async methods that build_daily_recommendations calls
        sample_recs = [service.generate_sample_recommendation() for _ in range(10)]
        # Force all to moderate risk so the risk_level filter does not empty the list
        for r in sample_recs:
            r["risk_level"] = "moderate"
            r["recommendation_type"] = "buy"
            r["sector"] = "Technology"

        service.generate_ml_powered_recommendations = AsyncMock(return_value=sample_recs)
        service.generate_personalized_recommendations = AsyncMock(return_value=[])

        result = await service.build_daily_recommendations(
            user_id=1,
            target_date=date.today(),
            risk_level="moderate",
        )

        expected_keys = {
            "top_picks", "watchlist", "avoid_list",
            "sector_focus", "market_sentiment", "market_outlook",
            "risk_assessment", "special_situations",
        }
        assert expected_keys.issubset(result.keys())

    @pytest.mark.asyncio
    async def test_deduplicates_by_symbol_keeps_higher_confidence(self, service):
        """When ML and personalized produce overlapping symbols, the higher
        confidence version should survive."""
        low = service.generate_sample_recommendation(symbol="AAPL")
        low["confidence_score"] = 0.60
        low["risk_level"] = "moderate"
        low["recommendation_type"] = "buy"
        low["sector"] = "Technology"

        high = service.generate_sample_recommendation(symbol="AAPL")
        high["confidence_score"] = 0.90
        high["risk_level"] = "moderate"
        high["recommendation_type"] = "strong_buy"
        high["sector"] = "Technology"

        service.generate_ml_powered_recommendations = AsyncMock(return_value=[low])
        service.generate_personalized_recommendations = AsyncMock(return_value=[high])

        result = await service.build_daily_recommendations(
            user_id=1, target_date=date.today()
        )

        aapl_picks = [p for p in result["top_picks"] if p["symbol"] == "AAPL"]
        assert len(aapl_picks) == 1
        assert aapl_picks[0]["confidence_score"] == 0.90

    @pytest.mark.asyncio
    async def test_market_sentiment_bullish_for_all_buys(self, service):
        """When all top picks are strong_buy, sentiment should be positive."""
        recs = []
        for sym in ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]:
            r = service.generate_sample_recommendation(symbol=sym)
            r["recommendation_type"] = "strong_buy"
            r["confidence_score"] = 0.9
            r["risk_level"] = "moderate"
            r["sector"] = "Technology"
            recs.append(r)

        service.generate_ml_powered_recommendations = AsyncMock(return_value=recs)
        service.generate_personalized_recommendations = AsyncMock(return_value=[])

        result = await service.build_daily_recommendations(
            user_id=1, target_date=date.today()
        )
        assert result["market_sentiment"] > 0.3

    @pytest.mark.asyncio
    async def test_avoid_list_populated_for_sell_recs(self, service):
        """Sell/strong_sell recommendations should appear in avoid_list."""
        recs = []
        for sym in ["AAPL", "MSFT", "GOOGL"]:
            r = service.generate_sample_recommendation(symbol=sym)
            r["recommendation_type"] = "sell"
            r["confidence_score"] = 0.7
            r["risk_level"] = "moderate"
            r["sector"] = "Technology"
            recs.append(r)

        service.generate_ml_powered_recommendations = AsyncMock(return_value=recs)
        service.generate_personalized_recommendations = AsyncMock(return_value=[])

        result = await service.build_daily_recommendations(
            user_id=1, target_date=date.today()
        )
        assert len(result["avoid_list"]) >= 1


# =========================================================================
# run_backtest
# =========================================================================

class TestRunBacktest:
    """Exercises the legacy synthetic backtest behind ``DEMO_MODE=True``.

    Production refusal (``DEMO_MODE=False`` raising ``ModelUnavailableError``)
    is covered by ``test_f02003_service_layer_gating.py`` per PRD audit
    2026-04 §3 D Step 2.
    """

    @pytest.fixture(autouse=True)
    def _demo_mode(self, monkeypatch):
        from backend.config.settings import settings
        monkeypatch.setattr(settings, "DEMO_MODE", True)
        yield

    def test_returns_strategy_and_period(self, service):
        """Result must echo back the strategy name and date period."""
        start = date(2024, 1, 1)
        end = date(2024, 6, 30)
        result = service.run_backtest("growth", start, end)

        assert result["strategy"] == "growth"
        assert result["period"]["start"] == "2024-01-01"
        assert result["period"]["end"] == "2024-06-30"

    def test_final_value_consistent_with_total_return(self, service):
        """final_value should equal initial_capital * (1 + total_return)."""
        result = service.run_backtest(
            "value", date(2024, 1, 1), date(2024, 12, 31),
            initial_capital=50000,
        )
        expected = 50000 * (1 + result["total_return"])
        assert abs(result["final_value"] - expected) < 0.01

    def test_annualized_return_formula(self, service):
        """annualized_return = total_return * (365 / days)."""
        start = date(2024, 1, 1)
        end = date(2024, 7, 1)
        result = service.run_backtest("momentum", start, end)
        days = (end - start).days
        expected_annualized = result["total_return"] * (365 / days)
        assert abs(result["annualized_return"] - expected_annualized) < 1e-9

    def test_max_drawdown_is_negative(self, service):
        """max_drawdown should always be a negative number."""
        result = service.run_backtest("growth", date(2024, 1, 1), date(2024, 12, 31))
        assert result["max_drawdown"] < 0

    def test_custom_initial_capital(self, service):
        """initial_capital should pass through to the result."""
        result = service.run_backtest(
            "value", date(2024, 1, 1), date(2024, 12, 31),
            initial_capital=200000,
        )
        assert result["initial_capital"] == 200000

    def test_best_and_worst_trade_present(self, service):
        """Result must include best_trade and worst_trade dicts."""
        result = service.run_backtest("growth", date(2024, 1, 1), date(2024, 6, 30))
        assert "symbol" in result["best_trade"]
        assert "return" in result["best_trade"]
        assert "symbol" in result["worst_trade"]
        assert "return" in result["worst_trade"]


# =========================================================================
# generate_performance_records
# =========================================================================

class TestGeneratePerformanceRecords:
    """Exercises legacy synthetic records behind ``DEMO_MODE=True``.

    Production refusal covered by ``test_f02003_service_layer_gating.py``.
    """

    @pytest.fixture(autouse=True)
    def _demo_mode(self, monkeypatch):
        from backend.config.settings import settings
        monkeypatch.setattr(settings, "DEMO_MODE", True)
        yield


    def test_default_generates_20_records(self, service):
        """With no filter the raw generation produces 20 records."""
        records = service.generate_performance_records()
        assert len(records) == 20

    def test_status_filter_applied(self, service):
        """When status_filter is provided, all returned records must match."""
        records = service.generate_performance_records(status_filter="active")
        assert all(r["status"] == "active" for r in records)

    def test_recommended_date_within_range(self, service):
        """All recommended_date values should fall within the requested window."""
        days_back = 10
        records = service.generate_performance_records(days_back=days_back)
        cutoff = (date.today() - timedelta(days=days_back)).isoformat()
        for r in records:
            assert r["recommended_date"] >= cutoff

    def test_actual_return_formula(self, service):
        """actual_return = (current_price - entry_price) / entry_price."""
        records = service.generate_performance_records()
        for r in records:
            expected = (r["current_price"] - r["entry_price"]) / r["entry_price"]
            assert abs(r["actual_return"] - expected) < 1e-9

    def test_record_has_required_fields(self, service):
        """Each record must contain the full set of tracking fields."""
        records = service.generate_performance_records()
        required = {
            "recommendation_id", "symbol", "recommended_date",
            "recommendation_type", "entry_price", "current_price",
            "target_price", "actual_return", "expected_return",
            "days_since_recommendation", "status", "performance_rating",
        }
        for r in records:
            assert required.issubset(r.keys())


# =========================================================================
# build_portfolio_recommendations
# =========================================================================

class TestBuildPortfolioRecommendations:

    def test_returns_portfolio_id(self, service):
        """The response must echo back the portfolio_id."""
        result = service.build_portfolio_recommendations("port-123")
        assert result["portfolio_id"] == "port-123"

    def test_contains_five_recommendations(self, service):
        """Exactly 5 recommendations should be generated."""
        result = service.build_portfolio_recommendations("port-abc")
        assert len(result["recommendations"]) == 5

    def test_rebalancing_weights_sum_to_one(self, service):
        """Rebalancing allocation percentages must sum to 1.0."""
        result = service.build_portfolio_recommendations("port-xyz")
        total = sum(result["rebalancing_suggestions"].values())
        assert abs(total - 1.0) < 1e-9

    def test_risk_score_in_range(self, service):
        """risk_score should be between 30 and 70."""
        result = service.build_portfolio_recommendations("port-001")
        assert 30 <= result["risk_score"] <= 70

    def test_diversification_score_in_range(self, service):
        """diversification_score should be between 0.6 and 0.9."""
        result = service.build_portfolio_recommendations("port-002")
        assert 0.6 <= result["diversification_score"] <= 0.9

    def test_expected_return_positive(self, service):
        """expected_portfolio_return should be in [0.08, 0.15]."""
        result = service.build_portfolio_recommendations("port-003")
        assert 0.08 <= result["expected_portfolio_return"] <= 0.15


# =========================================================================
# generate_alert_history
# =========================================================================

class TestGenerateAlertHistory:
    """Exercises legacy synthetic alert history behind ``DEMO_MODE=True``.

    Production refusal covered by ``test_f02003_service_layer_gating.py``.
    """

    @pytest.fixture(autouse=True)
    def _demo_mode(self, monkeypatch):
        from backend.config.settings import settings
        monkeypatch.setattr(settings, "DEMO_MODE", True)
        yield


    def test_returns_ten_alerts(self, service):
        """Default call should produce exactly 10 alerts."""
        alerts = service.generate_alert_history()
        assert len(alerts) == 10

    def test_sorted_by_timestamp_descending(self, service):
        """Alerts must be sorted newest-first."""
        alerts = service.generate_alert_history()
        timestamps = [a["timestamp"] for a in alerts]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_alert_has_required_fields(self, service):
        """Each alert must include id, timestamp, type, symbol, message, read."""
        alerts = service.generate_alert_history()
        required = {"id", "timestamp", "type", "symbol", "message", "read"}
        for a in alerts:
            assert required.issubset(a.keys())

    def test_alert_type_is_valid(self, service):
        """Alert type must be one of the three known values."""
        alerts = service.generate_alert_history()
        valid_types = {"strong_buy", "target_reached", "stop_loss_triggered"}
        for a in alerts:
            assert a["type"] in valid_types


# =========================================================================
# generate_filtered_recommendations
# =========================================================================

class TestGenerateFilteredRecommendations:

    def test_default_returns_up_to_limit(self, service):
        """Without filters, result length should be <= limit."""
        recs = service.generate_filtered_recommendations(count=50, limit=10)
        assert len(recs) <= 10

    def test_filter_by_recommendation_type(self, service):
        """Only recs matching the specified recommendation_type should survive."""
        recs = service.generate_filtered_recommendations(
            count=200, recommendation_type="buy", limit=50,
        )
        assert all(r["recommendation_type"] == "buy" for r in recs)

    def test_min_confidence_threshold(self, service):
        """All returned recs must have confidence_score >= min_confidence."""
        recs = service.generate_filtered_recommendations(
            count=200, min_confidence=0.8, limit=50,
        )
        assert all(r["confidence_score"] >= 0.8 for r in recs)

    def test_sort_by_confidence_desc(self, service):
        """Default sort is by confidence_score descending."""
        recs = service.generate_filtered_recommendations(
            count=200, limit=20, sort_by="confidence_score", order="desc",
        )
        scores = [r["confidence_score"] for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_sort_by_expected_return_asc(self, service):
        """Ascending sort by expected_return should order low-to-high."""
        recs = service.generate_filtered_recommendations(
            count=200, limit=20, sort_by="expected_return", order="asc",
        )
        returns = [r["expected_return"] for r in recs]
        assert returns == sorted(returns)

    def test_pagination_offset(self, service):
        """Offset should skip the first N results."""
        # Use a fixed seed so both calls produce the same pool
        random.seed(42)
        all_recs = service.generate_filtered_recommendations(
            count=30, limit=30, offset=0,
        )
        random.seed(42)
        offset_recs = service.generate_filtered_recommendations(
            count=30, limit=10, offset=5,
        )
        # The first element of offset_recs should match the 6th of all_recs
        assert offset_recs[0]["id"] == all_recs[5]["id"]

    def test_multi_category_filter(self, service):
        """categories list filter should keep only matching categories."""
        recs = service.generate_filtered_recommendations(
            count=200, categories=["growth", "value"], limit=50,
        )
        assert all(r["category"] in ("growth", "value") for r in recs)

    def test_min_expected_return_filter(self, service):
        """Only recs with expected_return >= threshold should survive."""
        recs = service.generate_filtered_recommendations(
            count=200, min_expected_return=0.05, limit=50,
        )
        assert all(r["expected_return"] >= 0.05 for r in recs)
