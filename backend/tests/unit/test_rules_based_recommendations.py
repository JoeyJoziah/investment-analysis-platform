"""
Unit tests for the rules-based quantitative screen.

Covers:
    * stock_repository.get_bulk_latest_fundamentals usage (patched)
    * recommendation_service.generate_rules_based_recommendations

The screen is a transparent, deterministic momentum (60-day price return) +
valuation (P/E percentile) rank over REAL stored data. These tests verify that:
    (a) momentum / valuation / composite are computed correctly,
    (b) ranking and recommendation_type tiers are correct,
    (c) confidence is deterministic (two runs -> identical),
    (d) symbols with < 30 price rows are skipped,
    (e) an empty universe returns [] (no random, generate_sample_recommendation
        is NEVER called),
    (f) the disclosure text no longer claims machine learning for this path.

All repository access is mocked; no database or external services required.
"""

import pytest
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from backend.services.recommendation_service import (
    RecommendationService,
    RULES_BASED_ALGORITHM_TYPE,
    RULES_BASED_METHODOLOGY_DISCLOSURE,
    MOMENTUM_MIN_ROWS,
    MOMENTUM_WINDOW_DAYS,
)


# ---------------------------------------------------------------------------
# Fixtures / fixture data builders
# ---------------------------------------------------------------------------

@pytest.fixture
def service():
    """A RecommendationService with mocked engines (no heavy analytics init)."""
    import sys
    mod = sys.modules["backend.services.recommendation_service"]
    with patch.object(mod, "RecommendationEngine", return_value=MagicMock()), \
         patch.object(mod, "FundamentalAnalysisEngine", return_value=MagicMock()):
        yield RecommendationService()


def _make_stock(symbol, *, name=None, sector="Technology", market_cap=1_000_000_000):
    return SimpleNamespace(
        symbol=symbol,
        name=name or f"{symbol} Inc.",
        sector=sector,
        industry="Software",
        market_cap=market_cap,
    )


def _price_row(close, *, volume=1_000_000, the_date=None):
    """A minimal OHLC row matching PriceHistory attribute access."""
    return SimpleNamespace(
        open=close,
        high=close,
        low=close,
        close=close,
        volume=volume,
        date=the_date or datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _linear_history(start, step, n):
    """Chronological (oldest-first) history of n rows: close = start + i*step."""
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return [
        _price_row(start + i * step, the_date=base + timedelta(days=i))
        for i in range(n)
    ]


def _fundamentals(pe_ratio=None, peg_ratio=None, *, period_date=None,
                  pb_ratio=None, roe=None, net_margin=None,
                  debt_to_equity=None, revenue=None, eps=None):
    return SimpleNamespace(
        pe_ratio=pe_ratio,
        peg_ratio=peg_ratio,
        pb_ratio=pb_ratio,
        roe=roe,
        net_margin=net_margin,
        debt_to_equity=debt_to_equity,
        revenue=revenue,
        eps=eps,
        period_date=period_date or date(2025, 12, 31),
    )


def _mock_repos(stocks, price_map, fundamentals_map):
    """Build patched stock_repo / price_repo objects with AsyncMock methods."""
    stock_repo = MagicMock()
    stock_repo.get_top_stocks = AsyncMock(return_value=list(stocks))
    stock_repo.get_bulk_latest_fundamentals = AsyncMock(return_value=dict(fundamentals_map))

    price_repo = MagicMock()
    price_repo.get_bulk_price_history = AsyncMock(return_value=dict(price_map))
    return stock_repo, price_repo


async def _run(service, stocks, price_map, fundamentals_map, **kwargs):
    stock_repo, price_repo = _mock_repos(stocks, price_map, fundamentals_map)
    return await service.generate_rules_based_recommendations(
        stock_repo=stock_repo,
        price_repo=price_repo,
        **kwargs,
    )


# =========================================================================
# (a) momentum / valuation / composite computed correctly
# =========================================================================

class TestSignalComputation:

    def test_momentum_60_day_return(self, service):
        """Momentum = close[-1]/close[-60] - 1 over a 60-row history."""
        # 60 rows, close[-60] = 100, close[-1] = 100 + 59*1 = 159 -> 0.59
        history = _linear_history(start=100.0, step=1.0, n=MOMENTUM_WINDOW_DAYS)
        momentum = service._compute_momentum_return(history)
        expected = history[-1].close / history[-MOMENTUM_WINDOW_DAYS].close - 1.0
        assert momentum == pytest.approx(expected)

    def test_momentum_uses_longest_window_when_fewer_than_60(self, service):
        """With 40 rows (30<=n<60) use the longest available window (40)."""
        history = _linear_history(start=50.0, step=2.0, n=40)
        momentum = service._compute_momentum_return(history)
        expected = history[-1].close / history[0].close - 1.0
        assert momentum == pytest.approx(expected)

    def test_momentum_none_when_below_min_rows(self, service):
        """Fewer than 30 rows -> momentum cannot be computed (None)."""
        history = _linear_history(start=10.0, step=1.0, n=MOMENTUM_MIN_ROWS - 1)
        assert service._compute_momentum_return(history) is None

    def test_percentile_ranks_basic(self, service):
        """Higher raw value -> higher percentile; min=0, max=1 for distinct vals."""
        ranks = service._percentile_ranks([10.0, 20.0, 30.0])
        assert ranks == [0.0, 0.5, 1.0]

    def test_percentile_single_value_is_half(self, service):
        assert service._percentile_ranks([42.0]) == [0.5]

    def test_percentile_ties_share_midrank(self, service):
        """Tied values receive the same (average) percentile."""
        ranks = service._percentile_ranks([5.0, 5.0, 9.0])
        # The two tied 5.0 share midrank 0.5/2 -> 0.25 each; 9.0 -> 1.0
        assert ranks[0] == ranks[1]
        assert ranks[2] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_composite_blends_momentum_and_inverse_valuation(self, service):
        """When both signals exist composite = 0.5*mom_pct + 0.5*(1 - pe_pct)."""
        # Two symbols. WIN has higher momentum AND lower P/E (cheaper) -> best.
        stocks = [_make_stock("WIN"), _make_stock("LOSE")]
        price_map = {
            "WIN": _linear_history(start=100.0, step=2.0, n=MOMENTUM_WINDOW_DAYS),   # big +momentum
            "LOSE": _linear_history(start=200.0, step=0.1, n=MOMENTUM_WINDOW_DAYS),  # small +momentum
        }
        fundamentals_map = {
            "WIN": _fundamentals(pe_ratio=10.0),   # cheap
            "LOSE": _fundamentals(pe_ratio=40.0),  # expensive
        }
        recs = await _run(service, stocks, price_map, fundamentals_map, limit=10)

        assert [r["symbol"] for r in recs] == ["WIN", "LOSE"]
        win = recs[0]
        # WIN: momentum_pct = 1.0 (highest), pe_pct = 0.0 (cheapest) -> (1-0)=1
        # composite = 0.5*1 + 0.5*1 = 1.0
        assert win["technical_signals"]["momentum_percentile"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_symbol_without_fundamentals_scored_momentum_only(self, service):
        """A symbol with no fundamentals uses momentum percentile as composite."""
        stocks = [_make_stock("AAA"), _make_stock("BBB")]
        price_map = {
            "AAA": _linear_history(start=100.0, step=2.0, n=MOMENTUM_WINDOW_DAYS),
            "BBB": _linear_history(start=100.0, step=0.5, n=MOMENTUM_WINDOW_DAYS),
        }
        fundamentals_map = {}  # neither has fundamentals
        recs = await _run(service, stocks, price_map, fundamentals_map, limit=10)
        # Momentum-only: AAA higher momentum -> ranked first.
        assert recs[0]["symbol"] == "AAA"
        # No valuation percentile recorded in key factors text.
        assert any("momentum-only" in f for f in recs[0]["key_factors"])


# =========================================================================
# (b) ranking + recommendation_type tiers
# =========================================================================

class TestRankingAndTiers:

    @pytest.mark.asyncio
    async def test_ranked_by_composite_descending(self, service):
        """Output is ordered best-composite first."""
        # Five symbols with strictly increasing momentum, no fundamentals.
        stocks = [_make_stock(s) for s in ["S0", "S1", "S2", "S3", "S4"]]
        price_map = {
            f"S{i}": _linear_history(start=100.0, step=0.5 * (i + 1), n=MOMENTUM_WINDOW_DAYS)
            for i in range(5)
        }
        recs = await _run(service, stocks, price_map, {}, limit=10)
        symbols = [r["symbol"] for r in recs]
        assert symbols == ["S4", "S3", "S2", "S1", "S0"]

    @pytest.mark.asyncio
    async def test_recommendation_type_tiers(self, service):
        """Top 20% -> strong_buy, bottom 20% -> strong_sell, middle -> hold."""
        # 5 distinct composites -> percentiles 0.0, 0.25, 0.5, 0.75, 1.0
        stocks = [_make_stock(s) for s in ["S0", "S1", "S2", "S3", "S4"]]
        price_map = {
            f"S{i}": _linear_history(start=100.0, step=0.5 * (i + 1), n=MOMENTUM_WINDOW_DAYS)
            for i in range(5)
        }
        recs = await _run(service, stocks, price_map, {}, limit=10)
        by_symbol = {r["symbol"]: r["recommendation_type"] for r in recs}
        assert by_symbol["S4"] == "strong_buy"   # pct 1.0  >= 0.80
        assert by_symbol["S3"] == "buy"          # pct 0.75 in [0.60, 0.80)
        assert by_symbol["S2"] == "hold"         # pct 0.50 in [0.40, 0.60)
        assert by_symbol["S1"] == "sell"         # pct 0.25 in [0.20, 0.40)
        assert by_symbol["S0"] == "strong_sell"  # pct 0.0  <  0.20

    @pytest.mark.asyncio
    async def test_target_price_clamped_momentum(self, service):
        """target_price = current_price * (1 + clamp(momentum, -0.3, 0.3))."""
        # Build a +100% momentum so it clamps to +0.3.
        history = _linear_history(start=100.0, step=0.0, n=MOMENTUM_WINDOW_DAYS)
        # Manually force a large jump on the last close.
        history[-1] = _price_row(300.0, the_date=history[-1].date)
        stocks = [_make_stock("BIG")]
        price_map = {"BIG": history}
        recs = await _run(service, stocks, price_map, {}, limit=10)
        rec = recs[0]
        # momentum = 300/100 - 1 = 2.0 -> clamped to 0.3 -> target = 300 * 1.3
        assert rec["current_price"] == pytest.approx(300.0)
        assert rec["target_price"] == pytest.approx(round(300.0 * 1.3, 2))

    @pytest.mark.asyncio
    async def test_limit_respected(self, service):
        """Only ``limit`` recommendations are returned even with more candidates."""
        stocks = [_make_stock(f"S{i}") for i in range(10)]
        price_map = {
            f"S{i}": _linear_history(start=100.0, step=0.5 * (i + 1), n=MOMENTUM_WINDOW_DAYS)
            for i in range(10)
        }
        recs = await _run(service, stocks, price_map, {}, limit=3)
        assert len(recs) == 3


# =========================================================================
# (c) confidence is deterministic
# =========================================================================

class TestDeterminism:

    @pytest.mark.asyncio
    async def test_confidence_is_deterministic_across_runs(self, service):
        """Two identical runs produce identical confidence scores (no random)."""
        stocks = [_make_stock(s) for s in ["S0", "S1", "S2"]]
        price_map = {
            f"S{i}": _linear_history(start=100.0, step=0.7 * (i + 1), n=MOMENTUM_WINDOW_DAYS)
            for i in range(3)
        }
        fundamentals_map = {"S0": _fundamentals(pe_ratio=15.0, peg_ratio=1.2)}

        run1 = await _run(service, stocks, price_map, fundamentals_map, limit=10)
        run2 = await _run(service, stocks, price_map, fundamentals_map, limit=10)

        c1 = {r["symbol"]: r["confidence_score"] for r in run1}
        c2 = {r["symbol"]: r["confidence_score"] for r in run2}
        assert c1 == c2

    @pytest.mark.asyncio
    async def test_confidence_formula(self, service):
        """confidence = round(0.5 + 0.45 * composite, 4) for the top pick."""
        stocks = [_make_stock("ONLY")]
        price_map = {"ONLY": _linear_history(start=100.0, step=1.0, n=MOMENTUM_WINDOW_DAYS)}
        recs = await _run(service, stocks, price_map, {}, limit=10)
        # Single symbol -> momentum percentile 0.5 -> composite 0.5
        # confidence = 0.5 + 0.45*0.5 = 0.725
        assert recs[0]["confidence_score"] == pytest.approx(0.725)

    @pytest.mark.asyncio
    async def test_confidence_within_bounds(self, service):
        """Confidence stays within [0.5, 0.95] for composites in [0, 1]."""
        stocks = [_make_stock(f"S{i}") for i in range(4)]
        price_map = {
            f"S{i}": _linear_history(start=100.0, step=0.5 * (i + 1), n=MOMENTUM_WINDOW_DAYS)
            for i in range(4)
        }
        recs = await _run(service, stocks, price_map, {}, limit=10)
        for r in recs:
            assert 0.5 <= r["confidence_score"] <= 0.95


# =========================================================================
# (d) symbols with < 30 price rows are skipped
# =========================================================================

class TestInsufficientData:

    @pytest.mark.asyncio
    async def test_symbol_below_threshold_skipped(self, service):
        """A symbol with < 30 price rows must not appear in the output."""
        stocks = [_make_stock("GOOD"), _make_stock("SHORT")]
        price_map = {
            "GOOD": _linear_history(start=100.0, step=1.0, n=MOMENTUM_WINDOW_DAYS),
            "SHORT": _linear_history(start=100.0, step=1.0, n=MOMENTUM_MIN_ROWS - 5),
        }
        recs = await _run(service, stocks, price_map, {}, limit=10)
        symbols = [r["symbol"] for r in recs]
        assert "GOOD" in symbols
        assert "SHORT" not in symbols

    @pytest.mark.asyncio
    async def test_symbol_missing_prices_skipped(self, service):
        """A symbol with no price history at all is skipped."""
        stocks = [_make_stock("HAS"), _make_stock("NONE")]
        price_map = {
            "HAS": _linear_history(start=100.0, step=1.0, n=MOMENTUM_WINDOW_DAYS),
            # NONE absent from price_map entirely
        }
        recs = await _run(service, stocks, price_map, {}, limit=10)
        assert [r["symbol"] for r in recs] == ["HAS"]

    @pytest.mark.asyncio
    async def test_all_below_threshold_returns_empty(self, service):
        """If no symbol meets the row threshold the result is []."""
        stocks = [_make_stock("A"), _make_stock("B")]
        price_map = {
            "A": _linear_history(start=100.0, step=1.0, n=5),
            "B": _linear_history(start=100.0, step=1.0, n=10),
        }
        recs = await _run(service, stocks, price_map, {}, limit=10)
        assert recs == []


# =========================================================================
# (e) empty universe -> [] (no random, generate_sample_recommendation NOT called)
# =========================================================================

class TestGracefulEmpty:

    @pytest.mark.asyncio
    async def test_empty_universe_returns_empty_list(self, service):
        """No stocks at all -> [] (never a fabricated sample)."""
        recs = await _run(service, [], {}, {}, limit=10)
        assert recs == []

    @pytest.mark.asyncio
    async def test_generate_sample_recommendation_never_called(self, service):
        """The random sample generator must NOT be invoked on the rules path."""
        service.generate_sample_recommendation = MagicMock(
            side_effect=AssertionError("generate_sample_recommendation must not be called")
        )

        # Empty universe
        recs_empty = await _run(service, [], {}, {}, limit=10)
        assert recs_empty == []

        # Universe present but all below threshold
        stocks = [_make_stock("A")]
        price_map = {"A": _linear_history(start=100.0, step=1.0, n=5)}
        recs_short = await _run(service, stocks, price_map, {}, limit=10)
        assert recs_short == []

        # Healthy universe -> real recs, still no sample call
        stocks2 = [_make_stock("REAL")]
        price_map2 = {"REAL": _linear_history(start=100.0, step=1.0, n=MOMENTUM_WINDOW_DAYS)}
        recs_real = await _run(service, stocks2, price_map2, {}, limit=10)
        assert len(recs_real) == 1
        service.generate_sample_recommendation.assert_not_called()

    @pytest.mark.asyncio
    async def test_fewer_than_limit_returns_what_exists(self, service):
        """If fewer than `limit` qualify, return only the qualifying ones."""
        stocks = [_make_stock("ONE"), _make_stock("TWO")]
        price_map = {
            "ONE": _linear_history(start=100.0, step=1.0, n=MOMENTUM_WINDOW_DAYS),
            "TWO": _linear_history(start=100.0, step=1.0, n=MOMENTUM_WINDOW_DAYS),
        }
        recs = await _run(service, stocks, price_map, {}, limit=10)
        assert len(recs) == 2


# =========================================================================
# (f) disclosure no longer claims ML for this path
# =========================================================================

class TestHonestDisclosure:

    @pytest.mark.asyncio
    async def test_algorithm_type_is_rules_based(self, service):
        """The embedded disclosure must label the methodology rules-based."""
        stocks = [_make_stock("X")]
        price_map = {"X": _linear_history(start=100.0, step=1.0, n=MOMENTUM_WINDOW_DAYS)}
        recs = await _run(service, stocks, price_map, {}, limit=10)
        disclosure = recs[0]["sec_disclosure"]
        assert disclosure["methodology_disclosure"] == RULES_BASED_METHODOLOGY_DISCLOSURE
        assert RULES_BASED_ALGORITHM_TYPE == "rules-based quantitative screen"

    @pytest.mark.asyncio
    async def test_disclosure_does_not_claim_ml(self, service):
        """The disclosure text must not claim machine learning for this path."""
        stocks = [_make_stock("X")]
        price_map = {"X": _linear_history(start=100.0, step=1.0, n=MOMENTUM_WINDOW_DAYS)}
        recs = await _run(service, stocks, price_map, {}, limit=10)
        text = recs[0]["sec_disclosure"]["methodology_disclosure"].lower()
        assert "machine learning" not in text or "does not use machine learning" in text
        assert "ml-powered" not in text
        # Positively asserts the transparent methodology is described.
        assert "momentum" in text
        assert "p/e" in text or "price-to-earnings" in text

    @pytest.mark.asyncio
    async def test_reasoning_describes_screen_not_ml(self, service):
        """The reasoning string describes the rules-based screen, not ML."""
        stocks = [_make_stock("X")]
        price_map = {"X": _linear_history(start=100.0, step=1.0, n=MOMENTUM_WINDOW_DAYS)}
        recs = await _run(service, stocks, price_map, {}, limit=10)
        reasoning = recs[0]["reasoning"].lower()
        assert "rules-based" in reasoning
        assert "ml-powered" not in reasoning
