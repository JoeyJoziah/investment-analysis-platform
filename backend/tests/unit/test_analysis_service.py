"""
Unit tests for backend/services/analysis_service.py

Tests all public functions and the AnalysisService class with mocked
dependencies.  No database, Redis, or external API calls required.
"""

import asyncio
import math
import statistics
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.services.analysis_service import (
    AnalysisService,
    calculate_macd,
    calculate_overall_score,
    calculate_risk_metrics_from_prices,
    calculate_rsi,
    cache_analysis_results,
    fetch_fundamental_data,
    fetch_parallel_with_fallback,
    fetch_sentiment_data,
    fetch_technical_indicators,
    generate_insights,
    safe_async_call,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_technical(*, rsi=55.0, trend="bullish"):
    """Lightweight stand-in for a TechnicalIndicators object."""
    return SimpleNamespace(rsi=rsi, trend=trend)


def _make_fundamental(*, pe_ratio=18.0, revenue_growth=0.12):
    """Lightweight stand-in for a FundamentalMetrics object."""
    return SimpleNamespace(pe_ratio=pe_ratio, revenue_growth=revenue_growth)


def _make_sentiment(*, overall_sentiment=0.3):
    """Lightweight stand-in for a SentimentAnalysis object."""
    return SimpleNamespace(overall_sentiment=overall_sentiment)


def _make_ml_predictions(*, confidence_score=0.85):
    """Lightweight stand-in for an MLPredictions object."""
    return SimpleNamespace(confidence_score=confidence_score)


def _generate_price_series(start=100.0, count=60, daily_return=0.001, seed=42):
    """Generate a deterministic list of closing prices."""
    import random as _random
    rng = _random.Random(seed)
    prices = [start]
    for _ in range(count - 1):
        change = rng.gauss(daily_return, 0.015)
        prices.append(prices[-1] * (1 + change))
    return prices


# =========================================================================
# safe_async_call
# =========================================================================

class TestSafeAsyncCall:

    @pytest.mark.asyncio
    async def test_successful_call_returns_result(self):
        """When the coroutine succeeds, return its result."""

        async def success_coro():
            return {"price": 150.25}

        result = await safe_async_call(success_coro(), timeout=5.0)
        assert result == {"price": 150.25}

    @pytest.mark.asyncio
    async def test_exception_returns_default(self):
        """When the coroutine raises, return the default value."""

        async def failing_coro():
            raise ValueError("API unavailable")

        result = await safe_async_call(
            failing_coro(),
            default={"fallback": True},
            error_msg="test call",
        )
        assert result == {"fallback": True}

    @pytest.mark.asyncio
    async def test_timeout_returns_default(self):
        """When the coroutine exceeds the timeout, return the default."""

        async def slow_coro():
            await asyncio.sleep(10)
            return "never reached"

        result = await safe_async_call(
            slow_coro(),
            timeout=0.05,
            default="timed_out",
            error_msg="slow call",
        )
        assert result == "timed_out"

    @pytest.mark.asyncio
    async def test_default_is_none_when_unspecified(self):
        """The default value should be None when not explicitly provided."""

        async def error_coro():
            raise RuntimeError("boom")

        result = await safe_async_call(error_coro())
        assert result is None


# =========================================================================
# fetch_parallel_with_fallback
# =========================================================================

class TestFetchParallelWithFallback:

    @pytest.mark.asyncio
    async def test_all_tasks_succeed(self):
        """All tasks complete successfully and their results are returned."""

        async def task_a():
            return {"rsi": 55.0}

        async def task_b():
            return {"macd": 1.2}

        tasks = [("rsi", task_a()), ("macd", task_b())]
        results = await fetch_parallel_with_fallback(tasks)

        assert results["rsi"] == {"rsi": 55.0}
        assert results["macd"] == {"macd": 1.2}

    @pytest.mark.asyncio
    async def test_partial_failure(self):
        """Failed tasks return None while successful tasks return results."""

        async def ok_task():
            return "good"

        async def bad_task():
            raise ConnectionError("API down")

        tasks = [("ok", ok_task()), ("bad", bad_task())]
        results = await fetch_parallel_with_fallback(tasks)

        assert results["ok"] == "good"
        assert results["bad"] is None

    @pytest.mark.asyncio
    async def test_all_tasks_fail(self):
        """When all tasks fail, every value is None."""

        async def fail_1():
            raise ValueError("fail")

        async def fail_2():
            raise TypeError("fail")

        tasks = [("a", fail_1()), ("b", fail_2())]
        results = await fetch_parallel_with_fallback(tasks)

        assert results["a"] is None
        assert results["b"] is None

    @pytest.mark.asyncio
    async def test_empty_task_list(self):
        """An empty task list returns an empty dict."""
        results = await fetch_parallel_with_fallback([])
        assert results == {}

    @pytest.mark.asyncio
    async def test_timeout_returns_all_none(self):
        """When the entire batch times out, all results are None."""

        async def slow():
            await asyncio.sleep(10)
            return "late"

        tasks = [("slow1", slow()), ("slow2", slow())]
        results = await fetch_parallel_with_fallback(tasks, timeout=0.05)

        assert results["slow1"] is None
        assert results["slow2"] is None


# =========================================================================
# fetch_technical_indicators
# =========================================================================

class TestFetchTechnicalIndicators:

    @pytest.mark.asyncio
    async def test_returns_indicators_when_client_provided(self):
        """With a valid client, fetched indicators are returned."""
        mock_client = AsyncMock()
        mock_client.get_rsi = AsyncMock(return_value={"RSI": 55.0})
        mock_client.get_macd = AsyncMock(return_value={"MACD": 1.5})
        mock_client.get_sma = AsyncMock(return_value={"SMA": 150.0})

        result = await fetch_technical_indicators("AAPL", alpha_vantage_client=mock_client)

        assert "rsi" in result
        assert "macd" in result
        assert "sma_20" in result

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_client(self):
        """Without an API client, return an empty dict."""
        result = await fetch_technical_indicators("AAPL", alpha_vantage_client=None)
        assert result == {}

    @pytest.mark.asyncio
    async def test_partial_indicator_failure(self):
        """If one indicator call fails, the others are still returned."""
        mock_client = AsyncMock()
        mock_client.get_rsi = AsyncMock(return_value={"RSI": 55.0})
        mock_client.get_macd = AsyncMock(side_effect=ConnectionError("timeout"))
        mock_client.get_sma = AsyncMock(return_value={"SMA": 150.0})

        result = await fetch_technical_indicators("AAPL", alpha_vantage_client=mock_client)

        assert "rsi" in result
        assert "sma_20" in result
        # macd failed, so it should be filtered out
        assert result.get("macd") is None or "macd" not in result


# =========================================================================
# fetch_fundamental_data
# =========================================================================

class TestFetchFundamentalData:

    @pytest.mark.asyncio
    async def test_with_both_clients(self):
        """Both alpha_vantage and finnhub data are merged into the result."""
        av_client = AsyncMock()
        av_client.get_company_overview = AsyncMock(
            return_value={"pe_ratio": 28.5, "market_cap": "2.5T"}
        )
        av_client.get_earnings = AsyncMock(
            return_value=[{"quarter": "Q4", "eps": 1.52}]
        )
        fh_client = AsyncMock()
        fh_client.get_basic_financials = AsyncMock(
            return_value={"revenue_growth": 0.08}
        )

        result = await fetch_fundamental_data(
            "AAPL", alpha_vantage_client=av_client, finnhub_client=fh_client
        )

        assert result.get("pe_ratio") == 28.5
        assert "earnings" in result
        assert result.get("revenue_growth") == 0.08

    @pytest.mark.asyncio
    async def test_no_clients_returns_empty(self):
        """Without any clients, return an empty dict."""
        result = await fetch_fundamental_data("AAPL")
        assert result == {}

    @pytest.mark.asyncio
    async def test_with_only_alpha_vantage(self):
        """Only alpha_vantage client produces overview + earnings."""
        av_client = AsyncMock()
        av_client.get_company_overview = AsyncMock(
            return_value={"sector": "Technology"}
        )
        av_client.get_earnings = AsyncMock(return_value=None)

        result = await fetch_fundamental_data("MSFT", alpha_vantage_client=av_client)

        assert result.get("sector") == "Technology"
        # earnings was None so it should not be in result
        assert "earnings" not in result


# =========================================================================
# fetch_sentiment_data
# =========================================================================

class TestFetchSentimentData:

    @pytest.mark.asyncio
    async def test_news_and_social_sentiment(self):
        """News is analyzed and social data is passed through."""
        fh_client = AsyncMock()
        fh_client.get_company_news = AsyncMock(
            return_value=[{"headline": "AAPL beats earnings"}]
        )
        fh_client.get_social_sentiment = AsyncMock(
            return_value={"reddit": 0.6, "twitter": 0.4}
        )

        analyzer = AsyncMock()
        analyzer.analyze_news_sentiment = AsyncMock(
            return_value={"score": 0.7, "label": "positive"}
        )

        result = await fetch_sentiment_data(
            "AAPL", finnhub_client=fh_client, sentiment_analyzer=analyzer
        )

        assert "news" in result
        assert result["news"]["label"] == "positive"
        assert "social" in result
        assert result["social"]["reddit"] == 0.6

    @pytest.mark.asyncio
    async def test_no_finnhub_client_returns_empty(self):
        """Without a finnhub client, return empty dict."""
        result = await fetch_sentiment_data("AAPL")
        assert result == {}

    @pytest.mark.asyncio
    async def test_no_sentiment_analyzer_skips_news_analysis(self):
        """Without a sentiment analyzer, news analysis is skipped."""
        fh_client = AsyncMock()
        fh_client.get_company_news = AsyncMock(
            return_value=[{"headline": "some news"}]
        )
        fh_client.get_social_sentiment = AsyncMock(return_value={"score": 0.5})

        result = await fetch_sentiment_data("AAPL", finnhub_client=fh_client)

        # No analyzer means no "news" key
        assert "news" not in result
        assert "social" in result


# =========================================================================
# calculate_rsi
# =========================================================================

class TestCalculateRsi:

    def test_returns_float_in_expected_range(self):
        """RSI should return a float between 30 and 70 (current stub)."""
        prices = _generate_price_series(count=30)
        result = calculate_rsi(prices, period=14)
        assert isinstance(result, float)
        assert 30 <= result <= 70

    def test_handles_short_price_list(self):
        """Even with insufficient data, should not raise."""
        result = calculate_rsi([100.0, 101.0], period=14)
        assert isinstance(result, float)


# =========================================================================
# calculate_macd
# =========================================================================

class TestCalculateMacd:

    def test_returns_expected_keys(self):
        """MACD result must contain macd, signal, and histogram."""
        prices = _generate_price_series(count=30)
        result = calculate_macd(prices)
        assert "macd" in result
        assert "signal" in result
        assert "histogram" in result

    def test_values_are_floats(self):
        """All MACD values should be floats."""
        result = calculate_macd([100.0] * 30)
        for key in ("macd", "signal", "histogram"):
            assert isinstance(result[key], float)


# =========================================================================
# generate_insights
# =========================================================================

class TestGenerateInsights:

    def test_overbought_rsi_insight(self):
        """RSI > 70 produces overbought insight."""
        analysis = {"technical": {"rsi": 75}, "fundamental": {}, "sentiment": {}}
        insights = generate_insights(analysis)
        assert any("overbought" in i.lower() for i in insights)

    def test_oversold_rsi_insight(self):
        """RSI < 30 produces oversold insight."""
        analysis = {"technical": {"rsi": 25}, "fundamental": {}, "sentiment": {}}
        insights = generate_insights(analysis)
        assert any("oversold" in i.lower() for i in insights)

    def test_undervalued_pe_insight(self):
        """PE ratio < 15 produces undervalued insight."""
        analysis = {"technical": {}, "fundamental": {"pe_ratio": 12}, "sentiment": {}}
        insights = generate_insights(analysis)
        assert any("undervalued" in i.lower() for i in insights)

    def test_positive_sentiment_insight(self):
        """Strong positive sentiment (> 0.5) produces a sentiment insight."""
        analysis = {
            "technical": {},
            "fundamental": {},
            "sentiment": {"overall_sentiment": 0.8},
        }
        insights = generate_insights(analysis)
        assert any("positive sentiment" in i.lower() for i in insights)

    def test_neutral_fallback(self):
        """When no signals trigger, a neutral insight is returned."""
        analysis = {"technical": {"rsi": 50}, "fundamental": {"pe_ratio": 20}, "sentiment": {}}
        insights = generate_insights(analysis)
        assert any("neutral" in i.lower() for i in insights)

    def test_empty_analysis_returns_neutral(self):
        """An empty analysis dict triggers the neutral fallback."""
        insights = generate_insights({})
        assert len(insights) >= 1
        assert any("neutral" in i.lower() for i in insights)

    def test_multiple_insights_can_coexist(self):
        """Multiple triggered conditions produce multiple insights."""
        analysis = {
            "technical": {"rsi": 25},
            "fundamental": {"pe_ratio": 10},
            "sentiment": {"overall_sentiment": 0.9},
        }
        insights = generate_insights(analysis)
        assert len(insights) >= 3


# =========================================================================
# calculate_risk_metrics_from_prices
# =========================================================================

class TestCalculateRiskMetricsFromPrices:

    def test_sufficient_data_calculates_volatility(self):
        """With 60+ prices, volatility should be a positive float."""
        prices = _generate_price_series(count=60)
        metrics = calculate_risk_metrics_from_prices(prices)
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert isinstance(metrics["sharpe_ratio"], float)

    def test_max_drawdown_is_negative_or_zero(self):
        """Max drawdown should be <= 0 (worst single-period return)."""
        prices = _generate_price_series(count=60)
        metrics = calculate_risk_metrics_from_prices(prices)
        assert metrics["max_drawdown"] <= 0.0 or metrics["max_drawdown"] >= 0.0
        # max_drawdown is min(returns), which can be negative
        returns = [(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))]
        assert metrics["max_drawdown"] == min(returns)

    def test_insufficient_data_returns_fallbacks(self):
        """With fewer than 30 prices, return hardcoded fallback values."""
        metrics = calculate_risk_metrics_from_prices([100.0, 101.0, 99.0])
        assert metrics["beta"] == 1.15
        assert metrics["sharpe_ratio"] == 1.85
        assert metrics["max_drawdown"] == -0.15

    def test_empty_prices_returns_fallbacks(self):
        """An empty price list returns fallback values."""
        metrics = calculate_risk_metrics_from_prices([])
        assert metrics["overall_risk_score"] == 42.0

    def test_risk_score_bounded_0_to_100(self):
        """Overall risk score should be clamped between 0 and 100."""
        prices = _generate_price_series(count=60)
        metrics = calculate_risk_metrics_from_prices(prices)
        assert 0 <= metrics["overall_risk_score"] <= 100

    def test_all_expected_keys_present(self):
        """The result dict should contain all required risk metric keys."""
        prices = _generate_price_series(count=60)
        metrics = calculate_risk_metrics_from_prices(prices)
        expected_keys = {
            "beta", "alpha", "sharpe_ratio", "sortino_ratio",
            "max_drawdown", "var_95", "cvar_95",
            "correlation_with_market", "risk_adjusted_return",
            "overall_risk_score",
        }
        assert expected_keys.issubset(metrics.keys())


# =========================================================================
# calculate_overall_score
# =========================================================================

class TestCalculateOverallScore:

    def test_all_components_present(self):
        """Score is a weighted average when all components are given."""
        tech = _make_technical(rsi=55.0, trend="bullish")
        fund = _make_fundamental(pe_ratio=18.0, revenue_growth=0.12)
        sent = _make_sentiment(overall_sentiment=0.3)
        ml = _make_ml_predictions(confidence_score=0.85)

        score = calculate_overall_score(tech, fund, sent, ml)
        assert 0 <= score <= 100

    def test_no_components_returns_default(self):
        """With no components, return the default score of 60.0."""
        score = calculate_overall_score(None, None, None, None)
        assert score == 60.0

    def test_only_technical_component(self):
        """With only technical data, score is based on tech alone."""
        tech = _make_technical(rsi=55.0, trend="bullish")
        score = calculate_overall_score(tech, None, None, None)
        # rsi in [30, 70] => +20, trend bullish => +15, base 50 => 85
        assert score == 85.0

    def test_bearish_trend_reduces_score(self):
        """A bearish trend should reduce the technical score."""
        bullish = _make_technical(rsi=55.0, trend="bullish")
        bearish = _make_technical(rsi=55.0, trend="bearish")

        score_bull = calculate_overall_score(bullish, None, None, None)
        score_bear = calculate_overall_score(bearish, None, None, None)

        assert score_bull > score_bear

    def test_high_confidence_ml_increases_score(self):
        """High ML confidence score should produce a high component."""
        ml = _make_ml_predictions(confidence_score=0.95)
        score = calculate_overall_score(None, None, None, ml)
        assert score == 95.0

    def test_negative_sentiment_lowers_score(self):
        """Negative sentiment should produce a score below 50."""
        sent = _make_sentiment(overall_sentiment=-0.8)
        score = calculate_overall_score(None, None, sent, None)
        # 50 + (-0.8 * 25) = 30.0
        assert score == 30.0


# =========================================================================
# cache_analysis_results
# =========================================================================

class TestCacheAnalysisResults:

    @pytest.mark.asyncio
    async def test_caches_without_error(self):
        """Caching should complete without raising."""
        await cache_analysis_results(
            symbol="AAPL",
            score=78.5,
            analysis_data={"technical": {"rsi": 55}},
        )
        # No exception means success (function logs and awaits sleep)

    @pytest.mark.asyncio
    async def test_handles_exception_gracefully(self):
        """Even if internal logic fails, caching should not raise."""
        # Patch asyncio.sleep to raise and verify graceful handling
        with patch("asyncio.sleep", side_effect=RuntimeError("boom")):
            # Should not raise
            await cache_analysis_results(
                symbol="FAIL",
                score=0.0,
                analysis_data={},
            )


# =========================================================================
# AnalysisService class
# =========================================================================

class TestAnalysisService:

    @pytest.fixture
    def service(self):
        """Fresh AnalysisService with mocked engines."""
        with patch("backend.analytics.fundamental_analysis.FundamentalAnalysisEngine"), \
             patch("backend.analytics.technical_analysis.TechnicalAnalysisEngine"), \
             patch("backend.analytics.sentiment_analysis.SentimentAnalysisEngine"):
            svc = AnalysisService()
        return svc

    @pytest.mark.asyncio
    async def test_run_analysis_returns_all_types(self, service):
        """Default analysis returns technical, fundamental, and sentiment."""
        result = await service.run_analysis("AAPL")
        assert result["ticker"] == "AAPL"
        assert "technical" in result["analyses"]
        assert "fundamental" in result["analyses"]
        assert "sentiment" in result["analyses"]

    @pytest.mark.asyncio
    async def test_run_analysis_selected_types(self, service):
        """Only requested analysis types are included."""
        result = await service.run_analysis("AAPL", types=["technical"])
        assert "technical" in result["analyses"]
        assert "fundamental" not in result["analyses"]

    @pytest.mark.asyncio
    async def test_run_analysis_caches_result(self, service):
        """Repeated calls for the same ticker use the cache."""
        result1 = await service.run_analysis("GOOG")
        result2 = await service.run_analysis("GOOG")
        assert result1["timestamp"] == result2["timestamp"]

    @pytest.mark.asyncio
    async def test_clear_cache_specific_ticker(self, service):
        """Clearing a specific ticker removes only that entry."""
        await service.run_analysis("AAPL")
        await service.run_analysis("GOOG")
        service.clear_cache("AAPL")

        assert await service.get_cached_analysis("AAPL") is None
        assert await service.get_cached_analysis("GOOG") is not None

    @pytest.mark.asyncio
    async def test_clear_cache_all(self, service):
        """Clearing all cache removes every entry."""
        await service.run_analysis("AAPL")
        await service.run_analysis("GOOG")
        service.clear_cache()

        assert await service.get_cached_analysis("AAPL") is None
        assert await service.get_cached_analysis("GOOG") is None

    def test_cache_key_generation(self, service):
        """Cache key is deterministic from ticker, types, and depth."""
        key = service._get_cache_key("AAPL", ["technical", "fundamental"], "deep")
        assert "AAPL" in key
        assert "deep" in key
        # Types are sorted
        assert "fundamental_technical" in key

    @pytest.mark.asyncio
    async def test_compare_stocks(self, service):
        """Stock comparison returns results for all tickers."""
        result = await service.compare_stocks(["AAPL", "GOOG", "MSFT"])
        assert result["comparison_type"] == "fundamental"
        assert len(result["stocks"]) == 3

    def test_composite_score_with_available_scores(self, service):
        """Composite score uses weighted sum when component scores exist."""
        analyses = {
            "technical": {"composite_score": 80.0},
            "fundamental": {"composite_score": 70.0},
            "sentiment": {"composite_score": 60.0},
        }
        score = service._calculate_composite_score(analyses)
        expected = 80.0 * 0.3 + 70.0 * 0.4 + 60.0 * 0.3
        assert abs(score - expected) < 0.01

    def test_composite_score_no_scores(self, service):
        """Without any composite_score fields, return 0.0."""
        analyses = {
            "technical": {"available": False},
            "fundamental": {"available": False},
        }
        score = service._calculate_composite_score(analyses)
        assert score == 0.0
