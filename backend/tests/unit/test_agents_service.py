"""
Unit tests for backend/services/agents_service.py

Tests cover:
- Engine singleton management (_get_fundamental_engine, _get_technical_engine,
  _get_sentiment_engine)
- Per-engine analysis runners (run_fundamental_analysis, run_technical_analysis,
  run_sentiment_analysis)
- Analysis orchestration (run_multi_engine_analysis)
- ANALYSIS_RUNNERS dispatch map
- Metrics logging helpers (log_agent_analysis_metrics, log_analysis_metrics,
  log_batch_analysis_metrics)

All external dependencies (analysis engines, pandas, numpy, sanitize_numpy)
are mocked.  No database, network, or heavy computation required.
"""

import asyncio
import logging
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import backend.services.agents_service as agents_mod
from backend.services.agents_service import (
    ANALYSIS_RUNNERS,
    log_agent_analysis_metrics,
    log_analysis_metrics,
    log_batch_analysis_metrics,
    run_fundamental_analysis,
    run_multi_engine_analysis,
    run_sentiment_analysis,
    run_technical_analysis,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_singletons():
    """Reset the module-level engine singletons so each test starts clean."""
    agents_mod._fundamental_engine = None
    agents_mod._technical_engine = None
    agents_mod._sentiment_engine = None


@pytest.fixture(autouse=True)
def _clean_singletons():
    """Automatically reset engine singletons before and after every test."""
    _reset_singletons()
    yield
    _reset_singletons()


# ---------------------------------------------------------------------------
# Engine Singleton Management
# ---------------------------------------------------------------------------


class TestEngineSingletons:
    """Verify lazy-init singleton behaviour for each engine type."""

    def test_fundamental_engine_created_on_first_call(self):
        with patch(
            "backend.services.agents_service.FundamentalAnalysisEngine"
        ) as MockCls:
            instance = MagicMock()
            MockCls.return_value = instance

            engine = agents_mod._get_fundamental_engine()

            MockCls.assert_called_once()
            assert engine is instance

    def test_fundamental_engine_reused_on_second_call(self):
        with patch(
            "backend.services.agents_service.FundamentalAnalysisEngine"
        ) as MockCls:
            MockCls.return_value = MagicMock()

            e1 = agents_mod._get_fundamental_engine()
            e2 = agents_mod._get_fundamental_engine()

            MockCls.assert_called_once()
            assert e1 is e2

    def test_technical_engine_created_on_first_call(self):
        with patch(
            "backend.services.agents_service.TechnicalAnalysisEngine"
        ) as MockCls:
            instance = MagicMock()
            MockCls.return_value = instance

            engine = agents_mod._get_technical_engine()

            MockCls.assert_called_once()
            assert engine is instance

    def test_technical_engine_reused_on_second_call(self):
        with patch(
            "backend.services.agents_service.TechnicalAnalysisEngine"
        ) as MockCls:
            MockCls.return_value = MagicMock()

            e1 = agents_mod._get_technical_engine()
            e2 = agents_mod._get_technical_engine()

            MockCls.assert_called_once()
            assert e1 is e2

    def test_sentiment_engine_created_on_first_call(self):
        with patch(
            "backend.services.agents_service.SentimentAnalysisEngine"
        ) as MockCls:
            instance = MagicMock()
            MockCls.return_value = instance

            engine = agents_mod._get_sentiment_engine()

            MockCls.assert_called_once()
            assert engine is instance

    def test_sentiment_engine_reused_on_second_call(self):
        with patch(
            "backend.services.agents_service.SentimentAnalysisEngine"
        ) as MockCls:
            MockCls.return_value = MagicMock()

            e1 = agents_mod._get_sentiment_engine()
            e2 = agents_mod._get_sentiment_engine()

            MockCls.assert_called_once()
            assert e1 is e2

    def test_engines_are_independent(self):
        """Each engine type has its own singleton slot."""
        with patch(
            "backend.services.agents_service.FundamentalAnalysisEngine"
        ) as F, patch(
            "backend.services.agents_service.TechnicalAnalysisEngine"
        ) as T, patch(
            "backend.services.agents_service.SentimentAnalysisEngine"
        ) as S:
            F.return_value = MagicMock(name="fund")
            T.return_value = MagicMock(name="tech")
            S.return_value = MagicMock(name="sent")

            fund = agents_mod._get_fundamental_engine()
            tech = agents_mod._get_technical_engine()
            sent = agents_mod._get_sentiment_engine()

            assert fund is not tech
            assert tech is not sent
            assert fund is not sent


# ---------------------------------------------------------------------------
# ANALYSIS_RUNNERS dispatch map
# ---------------------------------------------------------------------------


class TestAnalysisRunnersMap:

    def test_contains_fundamental(self):
        assert "fundamental" in ANALYSIS_RUNNERS

    def test_contains_technical(self):
        assert "technical" in ANALYSIS_RUNNERS

    def test_contains_sentiment(self):
        assert "sentiment" in ANALYSIS_RUNNERS

    def test_exactly_three_entries(self):
        assert len(ANALYSIS_RUNNERS) == 3

    def test_values_are_callable(self):
        for name, runner in ANALYSIS_RUNNERS.items():
            assert callable(runner), f"{name} runner is not callable"


# ---------------------------------------------------------------------------
# run_fundamental_analysis
# ---------------------------------------------------------------------------


class TestRunFundamentalAnalysis:

    @pytest.mark.asyncio
    async def test_returns_score_summary_details(self):
        mock_engine = AsyncMock()
        mock_engine.analyze_company = AsyncMock(
            return_value={
                "composite_score": 72.5,
                "risks": [{"description": "High debt"}],
                "opportunities": [{"description": "Growing market"}],
            }
        )

        with patch.object(agents_mod, "_get_fundamental_engine", return_value=mock_engine), \
             patch("backend.services.agents_service.sanitize_numpy", side_effect=lambda x: x):
            result = await run_fundamental_analysis("AAPL", "standard")

        assert result["score"] == 72.5
        assert "72.5" in result["summary"]
        assert "Risks" in result["summary"]
        assert "Opportunities" in result["summary"]

    @pytest.mark.asyncio
    async def test_standard_depth_truncates_details(self):
        mock_engine = AsyncMock()
        mock_engine.analyze_company = AsyncMock(
            return_value={
                "composite_score": 60.0,
                "risks": [{"description": f"risk-{i}"} for i in range(5)],
                "opportunities": [{"description": f"opp-{i}"} for i in range(5)],
                "extra_key": "should_not_appear",
            }
        )

        with patch.object(agents_mod, "_get_fundamental_engine", return_value=mock_engine), \
             patch("backend.services.agents_service.sanitize_numpy", side_effect=lambda x: x):
            result = await run_fundamental_analysis("TSLA", "standard")

        details = result["details"]
        assert len(details["risks"]) == 3
        assert len(details["opportunities"]) == 3
        assert "extra_key" not in details

    @pytest.mark.asyncio
    async def test_deep_depth_returns_full_analysis(self):
        full_analysis = {
            "composite_score": 85.0,
            "risks": [],
            "opportunities": [],
            "extra_key": "present_in_deep",
        }
        mock_engine = AsyncMock()
        mock_engine.analyze_company = AsyncMock(return_value=full_analysis)

        with patch.object(agents_mod, "_get_fundamental_engine", return_value=mock_engine), \
             patch("backend.services.agents_service.sanitize_numpy", side_effect=lambda x: x):
            result = await run_fundamental_analysis("GOOG", "deep")

        assert result["details"] is full_analysis

    @pytest.mark.asyncio
    async def test_summary_without_risks_or_opportunities(self):
        mock_engine = AsyncMock()
        mock_engine.analyze_company = AsyncMock(
            return_value={
                "composite_score": 50.0,
                "risks": [],
                "opportunities": [],
            }
        )

        with patch.object(agents_mod, "_get_fundamental_engine", return_value=mock_engine), \
             patch("backend.services.agents_service.sanitize_numpy", side_effect=lambda x: x):
            result = await run_fundamental_analysis("MSFT", "standard")

        assert result["summary"] == "Fundamental score 50.0/100."

    @pytest.mark.asyncio
    async def test_risks_as_plain_strings(self):
        mock_engine = AsyncMock()
        mock_engine.analyze_company = AsyncMock(
            return_value={
                "composite_score": 40.0,
                "risks": ["plain string risk"],
                "opportunities": [],
            }
        )

        with patch.object(agents_mod, "_get_fundamental_engine", return_value=mock_engine), \
             patch("backend.services.agents_service.sanitize_numpy", side_effect=lambda x: x):
            result = await run_fundamental_analysis("XYZ", "standard")

        assert "plain string risk" in result["summary"]


# ---------------------------------------------------------------------------
# run_technical_analysis
# ---------------------------------------------------------------------------


class TestRunTechnicalAnalysis:

    @pytest.mark.asyncio
    async def test_returns_score_summary_details(self):
        mock_engine = MagicMock()
        mock_engine.analyze_stock.return_value = {
            "composite_score": 0.75,
            "signals": [{"name": "MACD_cross"}, {"name": "RSI_overbought"}],
            "market_structure": {"trend": "bullish"},
        }

        import pandas as pd

        fake_df = pd.DataFrame(
            {
                "open": [1.0] * 40,
                "high": [1.1] * 40,
                "low": [0.9] * 40,
                "close": [1.0] * 40,
                "volume": [1e6] * 40,
            }
        )
        with patch.object(agents_mod, "_get_technical_engine", return_value=mock_engine), \
             patch.object(agents_mod, "_load_ohlcv_frame", AsyncMock(return_value=fake_df)), \
             patch("backend.services.agents_service.sanitize_numpy", side_effect=lambda x: x):
            result = await run_technical_analysis("AAPL", "standard")

        assert result["score"] == 0.75
        assert "bullish" in result["summary"]
        assert "MACD_cross" in result["summary"]

    @pytest.mark.asyncio
    async def test_standard_depth_truncates_signals(self):
        import pandas as pd

        mock_engine = MagicMock()
        mock_engine.analyze_stock.return_value = {
            "composite_score": -0.3,
            "signals": [{"name": f"sig-{i}"} for i in range(10)],
            "market_structure": {"trend": "bearish"},
            "extra": "data",
        }
        fake_df = pd.DataFrame(
            {
                "open": [1.0] * 40,
                "high": [1.1] * 40,
                "low": [0.9] * 40,
                "close": [1.0] * 40,
                "volume": [1e6] * 40,
            }
        )

        with patch.object(agents_mod, "_get_technical_engine", return_value=mock_engine), \
             patch.object(agents_mod, "_load_ohlcv_frame", AsyncMock(return_value=fake_df)), \
             patch("backend.services.agents_service.sanitize_numpy", side_effect=lambda x: x):
            result = await run_technical_analysis("TSLA", "standard")

        assert len(result["details"]["signals"]) == 5
        assert "extra" not in result["details"]

    @pytest.mark.asyncio
    async def test_deep_depth_returns_full_analysis(self):
        import pandas as pd

        full = {
            "composite_score": 0.5,
            "signals": [],
            "market_structure": {"trend": "sideways"},
            "extras": True,
        }
        mock_engine = MagicMock()
        mock_engine.analyze_stock.return_value = full
        fake_df = pd.DataFrame(
            {
                "open": [1.0] * 40,
                "high": [1.1] * 40,
                "low": [0.9] * 40,
                "close": [1.0] * 40,
                "volume": [1e6] * 40,
            }
        )

        with patch.object(agents_mod, "_get_technical_engine", return_value=mock_engine), \
             patch.object(agents_mod, "_load_ohlcv_frame", AsyncMock(return_value=fake_df)), \
             patch("backend.services.agents_service.sanitize_numpy", side_effect=lambda x: x):
            result = await run_technical_analysis("GOOG", "deep")

        assert result["details"]["extras"] is True
        assert result["details"]["data_source"] == "price_history"

    @pytest.mark.asyncio
    async def test_no_signals_produces_clean_summary(self):
        import pandas as pd

        mock_engine = MagicMock()
        mock_engine.analyze_stock.return_value = {
            "composite_score": 0.0,
            "signals": [],
            "market_structure": {"trend": "neutral"},
        }
        fake_df = pd.DataFrame(
            {
                "open": [1.0] * 40,
                "high": [1.1] * 40,
                "low": [0.9] * 40,
                "close": [1.0] * 40,
                "volume": [1e6] * 40,
            }
        )

        with patch.object(agents_mod, "_get_technical_engine", return_value=mock_engine), \
             patch.object(agents_mod, "_load_ohlcv_frame", AsyncMock(return_value=fake_df)), \
             patch("backend.services.agents_service.sanitize_numpy", side_effect=lambda x: x):
            result = await run_technical_analysis("XYZ", "standard")

        assert "Signals:" not in result["summary"]
        assert "neutral" in result["summary"]

    @pytest.mark.asyncio
    async def test_refuses_when_no_price_history_and_not_demo(self, monkeypatch):
        from backend.config.settings import settings
        from backend.exceptions import ModelUnavailableError

        monkeypatch.setattr(settings, "DEMO_MODE", False, raising=False)
        with patch.object(agents_mod, "_load_ohlcv_frame", AsyncMock(return_value=None)):
            with pytest.raises(ModelUnavailableError) as exc:
                await run_technical_analysis("NODATA", "standard")
        assert exc.value.reason == "insufficient_price_history"


# ---------------------------------------------------------------------------
# run_sentiment_analysis
# ---------------------------------------------------------------------------


class TestRunSentimentAnalysis:

    @pytest.mark.asyncio
    async def test_returns_score_summary_details(self):
        mock_engine = AsyncMock()
        mock_engine.analyze_comprehensive_sentiment = AsyncMock(
            return_value={
                "overall_sentiment": {
                    "score": 0.45,
                    "label": "positive",
                    "confidence": 0.82,
                },
                "sources_analyzed": 15,
            }
        )

        with patch.object(agents_mod, "_get_sentiment_engine", return_value=mock_engine), \
             patch("backend.services.agents_service.sanitize_numpy", side_effect=lambda x: x):
            result = await run_sentiment_analysis("AAPL", "standard")

        assert result["score"] == 0.45
        assert "positive" in result["summary"]
        assert "82%" in result["summary"]

    @pytest.mark.asyncio
    async def test_standard_depth_limited_details(self):
        mock_engine = AsyncMock()
        mock_engine.analyze_comprehensive_sentiment = AsyncMock(
            return_value={
                "overall_sentiment": {
                    "score": -0.1,
                    "label": "neutral",
                    "confidence": 0.55,
                },
                "sources_analyzed": 3,
                "raw_data": "lots of stuff",
            }
        )

        with patch.object(agents_mod, "_get_sentiment_engine", return_value=mock_engine), \
             patch("backend.services.agents_service.sanitize_numpy", side_effect=lambda x: x):
            result = await run_sentiment_analysis("MSFT", "standard")

        details = result["details"]
        assert details["score"] == -0.1
        assert details["label"] == "neutral"
        assert details["confidence"] == 0.55
        assert details["sources_analyzed"] == 3
        assert "raw_data" not in details

    @pytest.mark.asyncio
    async def test_deep_depth_returns_full_analysis(self):
        full = {
            "overall_sentiment": {"score": 0.8, "label": "bullish", "confidence": 0.95},
            "sources_analyzed": 50,
            "raw_data": "everything",
        }
        mock_engine = AsyncMock()
        mock_engine.analyze_comprehensive_sentiment = AsyncMock(return_value=full)

        with patch.object(agents_mod, "_get_sentiment_engine", return_value=mock_engine), \
             patch("backend.services.agents_service.sanitize_numpy", side_effect=lambda x: x):
            result = await run_sentiment_analysis("GOOG", "deep")

        assert result["details"] is full

    @pytest.mark.asyncio
    async def test_missing_overall_sentiment_uses_defaults(self):
        mock_engine = AsyncMock()
        mock_engine.analyze_comprehensive_sentiment = AsyncMock(return_value={})

        with patch.object(agents_mod, "_get_sentiment_engine", return_value=mock_engine), \
             patch("backend.services.agents_service.sanitize_numpy", side_effect=lambda x: x):
            result = await run_sentiment_analysis("XYZ", "standard")

        assert result["score"] == 0.0
        assert "neutral" in result["summary"]


# ---------------------------------------------------------------------------
# run_multi_engine_analysis
# ---------------------------------------------------------------------------


class TestRunMultiEngineAnalysis:

    @pytest.mark.asyncio
    async def test_single_fundamental_analysis(self):
        mock_runner = AsyncMock(
            return_value={"score": 75.0, "summary": "ok", "details": {}}
        )

        with patch.dict(ANALYSIS_RUNNERS, {"fundamental": mock_runner}):
            result = await run_multi_engine_analysis("AAPL", ["fundamental"], "standard")

        assert "fundamental" in result["results"]
        assert result["results"]["fundamental"]["score"] == 75.0
        assert 0 <= result["confidence_score"] <= 1
        assert result["analysis_id"].startswith("agt-")
        assert result["duration"] >= 0

    @pytest.mark.asyncio
    async def test_concurrent_multiple_engines(self):
        fund_mock = AsyncMock(return_value={"score": 80.0, "summary": "f", "details": {}})
        tech_mock = AsyncMock(return_value={"score": 0.5, "summary": "t", "details": {}})
        sent_mock = AsyncMock(return_value={"score": 0.3, "summary": "s", "details": {}})

        with patch.dict(
            ANALYSIS_RUNNERS,
            {"fundamental": fund_mock, "technical": tech_mock, "sentiment": sent_mock},
        ):
            result = await run_multi_engine_analysis(
                "AAPL", ["fundamental", "technical", "sentiment"], "standard"
            )

        assert len(result["results"]) == 3
        assert result["confidence_score"] > 0

    @pytest.mark.asyncio
    async def test_unsupported_type_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported analysis type: astrology"):
            await run_multi_engine_analysis("AAPL", ["astrology"], "standard")

    @pytest.mark.asyncio
    async def test_all_engines_fail_raises_runtime_error(self):
        failing_mock = AsyncMock(side_effect=RuntimeError("engine down"))

        with patch.dict(ANALYSIS_RUNNERS, {"fundamental": failing_mock}):
            with pytest.raises(RuntimeError, match="All analysis engines failed"):
                await run_multi_engine_analysis("AAPL", ["fundamental"], "standard")

    @pytest.mark.asyncio
    async def test_partial_failure_returns_successful_results(self):
        ok_mock = AsyncMock(return_value={"score": 60.0, "summary": "ok", "details": {}})
        fail_mock = AsyncMock(side_effect=RuntimeError("boom"))

        with patch.dict(ANALYSIS_RUNNERS, {"fundamental": ok_mock, "technical": fail_mock}):
            result = await run_multi_engine_analysis(
                "AAPL", ["fundamental", "technical"], "standard"
            )

        assert "fundamental" in result["results"]
        assert "technical" not in result["results"]

    @pytest.mark.asyncio
    async def test_confidence_score_for_fundamental_only(self):
        """Fundamental scores are normalised from [0, 100] to [0, 1]."""
        mock = AsyncMock(return_value={"score": 80.0, "summary": "", "details": {}})

        with patch.dict(ANALYSIS_RUNNERS, {"fundamental": mock}):
            result = await run_multi_engine_analysis("AAPL", ["fundamental"], "standard")

        assert result["confidence_score"] == 0.8

    @pytest.mark.asyncio
    async def test_confidence_score_for_technical_only(self):
        """Technical scores are normalised from [-1, 1] to [0, 1]."""
        mock = AsyncMock(return_value={"score": 0.5, "summary": "", "details": {}})

        with patch.dict(ANALYSIS_RUNNERS, {"technical": mock}):
            result = await run_multi_engine_analysis("AAPL", ["technical"], "standard")

        # (0.5 + 1) / 2 = 0.75
        assert result["confidence_score"] == 0.75

    @pytest.mark.asyncio
    async def test_confidence_score_clamped_above(self):
        """Scores above the normalised range are clamped to 1.0."""
        mock = AsyncMock(return_value={"score": 150.0, "summary": "", "details": {}})

        with patch.dict(ANALYSIS_RUNNERS, {"fundamental": mock}):
            result = await run_multi_engine_analysis("AAPL", ["fundamental"], "standard")

        assert result["confidence_score"] == 1.0

    @pytest.mark.asyncio
    async def test_confidence_score_clamped_below(self):
        """Scores below the normalised range are clamped to 0.0."""
        mock = AsyncMock(return_value={"score": -50.0, "summary": "", "details": {}})

        with patch.dict(ANALYSIS_RUNNERS, {"fundamental": mock}):
            result = await run_multi_engine_analysis("AAPL", ["fundamental"], "standard")

        assert result["confidence_score"] == 0.0

    @pytest.mark.asyncio
    async def test_analysis_id_format(self):
        mock = AsyncMock(return_value={"score": 50.0, "summary": "", "details": {}})

        with patch.dict(ANALYSIS_RUNNERS, {"fundamental": mock}):
            result = await run_multi_engine_analysis("AAPL", ["fundamental"], "standard")

        assert result["analysis_id"].startswith("agt-")
        assert len(result["analysis_id"]) == 12  # "agt-" + 8 hex chars

    @pytest.mark.asyncio
    async def test_timestamp_is_iso_utc(self):
        mock = AsyncMock(return_value={"score": 50.0, "summary": "", "details": {}})

        with patch.dict(ANALYSIS_RUNNERS, {"fundamental": mock}):
            result = await run_multi_engine_analysis("AAPL", ["fundamental"], "standard")

        ts = datetime.fromisoformat(result["timestamp"])
        assert ts.tzinfo is not None

    @pytest.mark.asyncio
    async def test_empty_analysis_types_raises_runtime_error(self):
        """An empty analysis_types list means zero results, triggering RuntimeError."""
        with pytest.raises(RuntimeError, match="All analysis engines failed"):
            await run_multi_engine_analysis("AAPL", [], "standard")

    @pytest.mark.asyncio
    async def test_confidence_averaged_across_engines(self):
        """Confidence is the mean of all normalised engine scores."""
        fund = AsyncMock(return_value={"score": 100.0, "summary": "", "details": {}})
        tech = AsyncMock(return_value={"score": 0.0, "summary": "", "details": {}})

        with patch.dict(ANALYSIS_RUNNERS, {"fundamental": fund, "technical": tech}):
            result = await run_multi_engine_analysis(
                "AAPL", ["fundamental", "technical"], "standard"
            )

        # fundamental: 100/100 = 1.0, technical: (0+1)/2 = 0.5 => mean = 0.75
        assert result["confidence_score"] == 0.75


# ---------------------------------------------------------------------------
# Metrics Logging Helpers
# ---------------------------------------------------------------------------


class TestLogAgentAnalysisMetrics:

    @pytest.mark.asyncio
    async def test_logs_successfully(self, caplog):
        with caplog.at_level(logging.INFO, logger="backend.services.agents_service"):
            await log_agent_analysis_metrics(
                analysis_id="agt-abc12345",
                ticker="AAPL",
                analysis_types=["fundamental"],
                depth="standard",
                confidence_score=0.85,
                duration=2.5,
            )

        assert "agt-abc12345" in caplog.text
        assert "AAPL" in caplog.text

    @pytest.mark.asyncio
    async def test_handles_logging_error_gracefully(self, caplog):
        """If the logger itself raises, the function catches it."""
        with patch("backend.services.agents_service.logger") as mock_logger:
            mock_logger.info.side_effect = RuntimeError("logging broken")
            mock_logger.error = MagicMock()

            await log_agent_analysis_metrics(
                analysis_id="x",
                ticker="X",
                analysis_types=[],
                depth="standard",
                confidence_score=0.0,
                duration=0.0,
            )

            mock_logger.error.assert_called_once()


class TestLogAnalysisMetrics:

    @pytest.mark.asyncio
    async def test_logs_all_fields(self, caplog):
        with caplog.at_level(logging.INFO, logger="backend.services.agents_service"):
            await log_analysis_metrics(
                ticker="GOOG",
                complexity_level="high",
                cost=0.0150,
                duration=5.3,
                agent_count=3,
            )

        assert "GOOG" in caplog.text
        assert "high" in caplog.text

    @pytest.mark.asyncio
    async def test_handles_logging_error_gracefully(self):
        with patch("backend.services.agents_service.logger") as mock_logger:
            mock_logger.info.side_effect = RuntimeError("broken")
            mock_logger.error = MagicMock()

            await log_analysis_metrics(
                ticker="X", complexity_level="low",
                cost=0.0, duration=0.0, agent_count=0,
            )

            mock_logger.error.assert_called_once()


class TestLogBatchAnalysisMetrics:

    @pytest.mark.asyncio
    async def test_logs_batch_fields(self, caplog):
        with caplog.at_level(logging.INFO, logger="backend.services.agents_service"):
            await log_batch_analysis_metrics(
                requested=10,
                completed=8,
                total_cost=0.12,
                total_duration=30.0,
                agents_used_count=5,
            )

        assert "Requested: 10" in caplog.text
        assert "Completed: 8" in caplog.text

    @pytest.mark.asyncio
    async def test_handles_logging_error_gracefully(self):
        with patch("backend.services.agents_service.logger") as mock_logger:
            mock_logger.info.side_effect = RuntimeError("broken")
            mock_logger.error = MagicMock()

            await log_batch_analysis_metrics(
                requested=0, completed=0, total_cost=0.0,
                total_duration=0.0, agents_used_count=0,
            )

            mock_logger.error.assert_called_once()
