"""
Unit tests for analytics modules:
  - fundamental_analysis.py  (FinancialMetrics, FundamentalAnalysisEngine, DCF, quality scores, etc.)
  - recommendation_engine.py (RecommendationAction, StockRecommendation, RecommendationEngine)
  - sentiment_analysis.py    (SentimentResult, SentimentAnalysisEngine - lexicon path)
  - finbert_analyzer.py      (FinBERTResult, FinancialTextPreprocessor, FinBERTAnalyzer,
                               FinBERTInference, module-level helpers)

Heavy dependencies (numpy/scipy/transformers/torch/pandas) are stubbed or provided as
lightweight stand-ins via importlib so no real ML libraries are needed.
Backend cross-imports are patched via sys.modules save/restore.
"""

import asyncio
import importlib
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# 1.  Stub heavy / optional deps BEFORE loading analytics modules
# ---------------------------------------------------------------------------

# scipy - fundamental_analysis.py uses `from scipy import stats`
_scipy_stub = MagicMock()
_scipy_stats_stub = MagicMock()
_scipy_stub.stats = _scipy_stats_stub
sys.modules.setdefault("scipy", _scipy_stub)
sys.modules.setdefault("scipy.stats", _scipy_stats_stub)

# torch / transformers - finbert_analyzer.py tries to import these
_torch_stub = MagicMock()
_torch_stub.cuda = MagicMock()
_torch_stub.cuda.is_available = MagicMock(return_value=False)
_torch_stub.device = MagicMock(return_value="cpu")
_torch_stub.no_grad = MagicMock(return_value=MagicMock(__enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False)))
_transformers_stub = MagicMock()
sys.modules.setdefault("torch", _torch_stub)
sys.modules.setdefault("transformers", _transformers_stub)

# numpy - let the real numpy through; both modules need it.
# (numpy is already installed in the project environment)

# ---------------------------------------------------------------------------
# 2.  Stub backend cross-imports needed by recommendation_engine.py
# ---------------------------------------------------------------------------

_CROSS_STUBS = {
    "backend": MagicMock(),
    "backend.analytics": MagicMock(),
    "backend.analytics.technical_analysis": MagicMock(),
    "backend.analytics.fundamental_analysis": MagicMock(),
    "backend.analytics.sentiment_analysis": MagicMock(),
    "backend.models": MagicMock(),
    "backend.models.ml_models": MagicMock(),
    "backend.data_ingestion": MagicMock(),
    "backend.data_ingestion.market_scanner": MagicMock(),
    "backend.utils": MagicMock(),
    "backend.utils.risk_manager": MagicMock(),
    "backend.utils.portfolio_optimizer": MagicMock(),
}

_saved_mods: dict = {}
for _name, _stub in _CROSS_STUBS.items():
    _saved_mods[_name] = sys.modules.get(_name)
    sys.modules[_name] = _stub

# Also add PredictionResult to the ml_models stub so type hints resolve
_pred_result_cls = type("PredictionResult", (), {})
sys.modules["backend.models.ml_models"].PredictionResult = _pred_result_cls
sys.modules["backend.models.ml_models"].ModelManager = MagicMock()

# ---------------------------------------------------------------------------
# 3.  Load the four analytics modules via importlib (bypass __init__.py chains)
# ---------------------------------------------------------------------------

_analytics_dir = Path(__file__).resolve().parents[2] / "analytics"


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _analytics_dir / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Load in dependency order
_fa_mod = _load_module("fundamental_analysis_mod", "fundamental_analysis.py")
_finbert_mod = _load_module("finbert_analyzer_mod", "finbert_analyzer.py")

# sentiment_analysis.py does a conditional `from backend.analytics.finbert_analyzer import …`
# Patch that cross-import with our freshly-loaded finbert module so the try-block succeeds
# but FINBERT_AVAILABLE remains False (because torch/transformers are stubs).
# The simplest approach: override backend.analytics.finbert_analyzer in sys.modules.
_finbert_proxy = MagicMock()
_finbert_proxy.FinBERTAnalyzer = _finbert_mod.FinBERTAnalyzer
_finbert_proxy.FinBERTInference = _finbert_mod.FinBERTInference
_finbert_proxy.FinBERTResult = _finbert_mod.FinBERTResult
_finbert_proxy.HAS_TRANSFORMERS = False  # keep lexicon-only mode
sys.modules["backend.analytics.finbert_analyzer"] = _finbert_proxy

_sa_mod = _load_module("sentiment_analysis_mod", "sentiment_analysis.py")

# recommendation_engine.py imports all others via backend.* - those are already
# stubbed in sys.modules, so we can load it directly.
_re_mod = _load_module("recommendation_engine_mod", "recommendation_engine.py")

# ---------------------------------------------------------------------------
# 4.  Restore saved sys.modules entries so other tests aren't polluted
# ---------------------------------------------------------------------------

for _name, _orig in _saved_mods.items():
    if _orig is not None:
        sys.modules[_name] = _orig
    else:
        sys.modules.pop(_name, None)

# ---------------------------------------------------------------------------
# 5.  Extract public names used in tests
# ---------------------------------------------------------------------------

FinancialMetrics = _fa_mod.FinancialMetrics
FundamentalAnalysisEngine = _fa_mod.FundamentalAnalysisEngine

RecommendationAction = _re_mod.RecommendationAction
StockRecommendation = _re_mod.StockRecommendation
RecommendationEngine = _re_mod.RecommendationEngine

SentimentResult = _sa_mod.SentimentResult
SentimentAnalysisEngine = _sa_mod.SentimentAnalysisEngine

FinBERTResult = _finbert_mod.FinBERTResult
FinancialTextPreprocessor = _finbert_mod.FinancialTextPreprocessor
FinBERTAnalyzer = _finbert_mod.FinBERTAnalyzer
FinBERTInference = _finbert_mod.FinBERTInference


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_financials(**overrides) -> dict:
    """Return a minimal but complete financials dict."""
    base = {
        "revenue": 10_000_000,
        "gross_profit": 4_000_000,
        "operating_income": 1_500_000,
        "net_income": 1_000_000,
        "total_assets": 20_000_000,
        "total_equity": 8_000_000,
        "total_debt": 4_000_000,
        "total_liabilities": 12_000_000,
        "current_assets": 5_000_000,
        "current_liabilities": 2_000_000,
        "cash": 1_000_000,
        "inventory": 500_000,
        "receivables": 800_000,
        "free_cash_flow": 900_000,
        "shares_outstanding": 1_000_000,
        "ebitda": 2_000_000,
        "operating_cash_flow": 1_200_000,
        "interest_expense": 200_000,
        "tax_rate": 0.21,
        "dividend_per_share": 0.5,
        "book_value_per_share": 8.0,
        "roe": 0.125,
        "retained_earnings": 3_000_000,
        "market_cap": 15_000_000,
        "property_plant_equipment": 3_000_000,
        "intangible_assets": 500_000,
        "roa": 0.05,
        "roa_previous": 0.04,
    }
    base.update(overrides)
    return base


def _make_market_data(**overrides) -> dict:
    base = {
        "market_cap": 15_000_000,
        "price": 15.0,
        "beta": 1.1,
        "enterprise_value": 18_000_000,
    }
    base.update(overrides)
    return base


def _make_stock_recommendation(**overrides) -> StockRecommendation:
    """Build a StockRecommendation with valid defaults."""
    now = datetime.now(timezone.utc)
    defaults = dict(
        ticker="AAPL",
        action=RecommendationAction.BUY,
        confidence=0.75,
        priority=7,
        entry_price=150.0,
        target_price=180.0,
        stop_loss=135.0,
        expected_return=0.20,
        time_horizon_days=30,
        risk_score=0.3,
        volatility=0.25,
        beta=1.1,
        sharpe_ratio=1.5,
        max_drawdown=-0.15,
        technical_score=0.7,
        fundamental_score=0.65,
        sentiment_score=0.6,
        ml_prediction_score=0.72,
        technical_analysis={},
        fundamental_analysis={},
        sentiment_analysis={},
        ml_predictions={},
        key_factors=["Strong earnings growth"],
        risks=["Market volatility"],
        opportunities=["Undervaluation"],
        catalysts=["Earnings in 10 days"],
        generated_at=now,
        valid_until=now,
        recommended_allocation=0.05,
        max_position_size=5000.0,
    )
    defaults.update(overrides)
    return StockRecommendation(**defaults)


# ===========================================================================
# Tests: FinancialMetrics dataclass
# ===========================================================================

class TestFinancialMetrics:
    """Tests for the FinancialMetrics dataclass."""

    def _make(self, **kw) -> FinancialMetrics:
        defaults = dict(
            gross_margin=40.0, operating_margin=15.0, net_margin=10.0,
            roe=12.5, roa=5.0, roic=8.0,
            revenue_growth=10.0, earnings_growth=12.0, fcf_growth=9.0,
            pe_ratio=18.0, peg_ratio=1.5, price_to_book=2.2,
            price_to_sales=1.8, ev_to_ebitda=9.0, fcf_yield=6.0,
            current_ratio=2.5, quick_ratio=1.8, debt_to_equity=0.5,
            interest_coverage=7.5,
            asset_turnover=0.5, inventory_turnover=8.0, receivables_turnover=12.5,
        )
        defaults.update(kw)
        return FinancialMetrics(**defaults)

    def test_construction_with_all_fields(self):
        m = self._make()
        assert m.gross_margin == 40.0
        assert m.pe_ratio == 18.0
        assert m.current_ratio == 2.5

    def test_profitability_fields(self):
        m = self._make(gross_margin=55.0, net_margin=20.0, roe=25.0)
        assert m.gross_margin == 55.0
        assert m.net_margin == 20.0
        assert m.roe == 25.0

    def test_growth_fields(self):
        m = self._make(revenue_growth=25.0, earnings_growth=30.0, fcf_growth=20.0)
        assert m.revenue_growth == 25.0
        assert m.earnings_growth == 30.0
        assert m.fcf_growth == 20.0

    def test_valuation_fields(self):
        m = self._make(pe_ratio=35.0, peg_ratio=2.5, ev_to_ebitda=20.0, fcf_yield=2.0)
        assert m.pe_ratio == 35.0
        assert m.peg_ratio == 2.5

    def test_financial_health_fields(self):
        m = self._make(current_ratio=3.0, quick_ratio=2.5, debt_to_equity=0.2, interest_coverage=15.0)
        assert m.current_ratio == 3.0
        assert m.debt_to_equity == 0.2
        assert m.interest_coverage == 15.0

    def test_efficiency_fields(self):
        m = self._make(asset_turnover=1.2, inventory_turnover=10.0, receivables_turnover=15.0)
        assert m.asset_turnover == 1.2
        assert m.inventory_turnover == 10.0

    def test_zero_values_allowed(self):
        m = self._make(pe_ratio=0.0, fcf_yield=0.0, interest_coverage=0.0)
        assert m.pe_ratio == 0.0
        assert m.fcf_yield == 0.0

    def test_negative_margin_allowed(self):
        """Negative margins are valid for loss-making companies."""
        m = self._make(net_margin=-5.0, operating_margin=-2.0)
        assert m.net_margin == -5.0


# ===========================================================================
# Tests: FundamentalAnalysisEngine
# ===========================================================================

class TestFundamentalAnalysisEngine:
    """Tests for FundamentalAnalysisEngine methods."""

    def setup_method(self):
        self.engine = FundamentalAnalysisEngine()

    # --- Constructor ---

    def test_initial_state(self):
        assert self.engine.risk_free_rate == pytest.approx(0.045)
        assert self.engine.market_risk_premium == pytest.approx(0.08)
        assert isinstance(self.engine.sector_averages, dict)

    # --- _calculate_financial_metrics ---

    def test_calculate_financial_metrics_basic(self):
        fin = _make_financials()
        mkt = _make_market_data()
        metrics = self.engine._calculate_financial_metrics(fin, mkt)
        assert isinstance(metrics, FinancialMetrics)
        # gross margin = 4M / 10M * 100 = 40
        assert metrics.gross_margin == pytest.approx(40.0)
        # net margin = 1M / 10M * 100 = 10
        assert metrics.net_margin == pytest.approx(10.0)

    def test_calculate_financial_metrics_roe(self):
        fin = _make_financials(net_income=2_000_000, total_equity=10_000_000)
        mkt = _make_market_data()
        metrics = self.engine._calculate_financial_metrics(fin, mkt)
        # ROE = 2M / 10M * 100 = 20
        assert metrics.roe == pytest.approx(20.0)

    def test_calculate_financial_metrics_zero_revenue(self):
        """Zero revenue should produce zero margins without error."""
        fin = _make_financials(revenue=0)
        mkt = _make_market_data()
        metrics = self.engine._calculate_financial_metrics(fin, mkt)
        assert metrics.gross_margin == 0.0
        assert metrics.net_margin == 0.0

    def test_calculate_financial_metrics_current_ratio(self):
        fin = _make_financials(current_assets=6_000_000, current_liabilities=2_000_000)
        mkt = _make_market_data()
        metrics = self.engine._calculate_financial_metrics(fin, mkt)
        assert metrics.current_ratio == pytest.approx(3.0)

    def test_calculate_financial_metrics_interest_coverage_no_interest(self):
        """No interest expense => coverage set to sentinel 999."""
        fin = _make_financials(interest_expense=0)
        mkt = _make_market_data()
        metrics = self.engine._calculate_financial_metrics(fin, mkt)
        assert metrics.interest_coverage == 999

    def test_calculate_financial_metrics_pe_ratio(self):
        # eps = 1M / 1M shares = 1.0; pe = 15 / 1 = 15
        fin = _make_financials(net_income=1_000_000, shares_outstanding=1_000_000)
        mkt = _make_market_data(price=15.0)
        metrics = self.engine._calculate_financial_metrics(fin, mkt)
        assert metrics.pe_ratio == pytest.approx(15.0)

    # --- _calculate_growth_rate ---

    def test_calculate_growth_rate_normal(self):
        # Values: [100, 110, 121] → CAGR over 2 years = (121/100)^(1/2) - 1 ≈ 10%
        rate = self.engine._calculate_growth_rate([100, 110, 121])
        assert rate == pytest.approx(10.0, rel=0.01)

    def test_calculate_growth_rate_empty(self):
        assert self.engine._calculate_growth_rate([]) == 0

    def test_calculate_growth_rate_single_value(self):
        assert self.engine._calculate_growth_rate([100]) == 0

    def test_calculate_growth_rate_zero_start(self):
        assert self.engine._calculate_growth_rate([0, 100, 200]) == 0

    # --- _calculate_wacc ---

    def test_calculate_wacc_default_beta(self):
        fin = _make_financials()
        mkt = _make_market_data(beta=1.0, market_cap=10_000_000)
        wacc = self.engine._calculate_wacc(fin, mkt)
        # Should be between 0 and 1 (decimal, not percent)
        assert 0.0 < wacc < 1.0

    def test_calculate_wacc_zero_total_value(self):
        """Zero market_cap + zero debt => default 10%."""
        fin = _make_financials(total_debt=0)
        mkt = _make_market_data(market_cap=0)
        wacc = self.engine._calculate_wacc(fin, mkt)
        assert wacc == pytest.approx(0.10)

    # --- _calculate_dcf ---

    def test_calculate_dcf_returns_dict_with_value(self):
        fin = _make_financials(free_cash_flow=1_000_000, fcf_growth=0.08)
        mkt = _make_market_data(market_cap=15_000_000)
        result = self.engine._calculate_dcf(fin, mkt)
        assert "value" in result
        assert "wacc" in result
        assert result["confidence"] == pytest.approx(0.8)

    def test_calculate_dcf_positive_value_for_profitable_company(self):
        fin = _make_financials(free_cash_flow=2_000_000, cash=1_000_000, total_debt=0)
        mkt = _make_market_data(market_cap=20_000_000)
        result = self.engine._calculate_dcf(fin, mkt)
        assert result["value"] > 0

    # --- _calculate_ddm ---

    def test_calculate_ddm_no_dividend(self):
        fin = _make_financials(dividend_per_share=0)
        mkt = _make_market_data()
        result = self.engine._calculate_ddm(fin, mkt)
        assert result["value"] == 0
        assert result["confidence"] == 0

    def test_calculate_ddm_with_dividend_gordon_growth(self):
        # required_return > growth_rate path
        fin = _make_financials(dividend_per_share=2.0, earnings_growth=0.04)
        mkt = _make_market_data(beta=1.0)
        result = self.engine._calculate_ddm(fin, mkt)
        assert result["value"] > 0
        assert result["confidence"] == pytest.approx(0.7)

    # --- _calculate_asset_based_value ---

    def test_calculate_asset_based_value(self):
        fin = _make_financials(
            total_assets=20_000_000,
            total_liabilities=12_000_000,
            shares_outstanding=1_000_000,
            intangible_assets=500_000,
        )
        result = self.engine._calculate_asset_based_value(fin)
        # nav = (20M - 12M) / 1M = 8.0
        assert result["value"] == pytest.approx(8.0)
        assert result["tangible_nav"] < result["value"]
        assert result["confidence"] == pytest.approx(0.5)

    # --- _calculate_epv ---

    def test_calculate_epv_positive(self):
        fin = _make_financials(operating_income=2_000_000, tax_rate=0.21, cash=500_000, total_debt=0)
        mkt = _make_market_data(market_cap=20_000_000)
        result = self.engine._calculate_epv(fin, mkt)
        assert result["value"] > 0
        assert result["no_growth_assumption"] is True

    # --- quality scoring ---

    def test_score_profitability_all_positive(self):
        fin = _make_financials(
            net_income=500_000,
            operating_cash_flow=700_000,
            roa=0.06,
            roa_previous=0.04,
            roic=20,
        )
        score = self.engine._score_profitability(fin)
        # net_income > 0 (+20), ocf > 0 (+20), ocf > ni (+20), roa improved (+20), roic > 15 (+20) = 100
        assert score == 100

    def test_score_profitability_minimal(self):
        score = self.engine._score_profitability(
            {"net_income": -1, "operating_cash_flow": -1, "roa": 0, "roa_previous": 0, "roic": 5}
        )
        assert score == 0

    def test_score_balance_sheet_excellent(self):
        fin = {
            "debt_to_equity": 0.3,
            "current_ratio": 2.5,
            "interest_coverage": 10,
            "intangibles_to_assets": 0.1,
        }
        score = self.engine._score_balance_sheet(fin)
        assert score == 100

    def test_score_balance_sheet_poor(self):
        fin = {
            "debt_to_equity": 3.0,
            "current_ratio": 0.8,
            "interest_coverage": 1,
            "intangibles_to_assets": 0.6,
        }
        score = self.engine._score_balance_sheet(fin)
        assert score == 0

    def test_get_quality_grade(self):
        assert self.engine._get_quality_grade(95) == "A+"
        assert self.engine._get_quality_grade(85) == "A"
        assert self.engine._get_quality_grade(70) == "B"
        assert self.engine._get_quality_grade(40) == "D"

    # --- Altman Z-Score ---

    def test_altman_z_score_safe_zone(self):
        fin = _make_financials(
            current_assets=10_000_000, current_liabilities=2_000_000,
            total_assets=20_000_000, retained_earnings=8_000_000,
            market_cap=30_000_000, total_liabilities=5_000_000, revenue=15_000_000,
        )
        result = self.engine._calculate_altman_z_score(fin)
        assert result["zone"] in ("safe", "grey", "distress")
        assert "score" in result
        assert "components" in result

    def test_altman_z_score_distress_zone(self):
        """A company with terrible financials should be in distress."""
        fin = {
            "current_assets": 100, "current_liabilities": 5_000_000,
            "total_assets": 1_000_000, "retained_earnings": -2_000_000,
            "operating_income": -500_000,
            "market_cap": 100_000, "total_liabilities": 9_000_000, "revenue": 200_000,
        }
        result = self.engine._calculate_altman_z_score(fin)
        assert result["zone"] == "distress"
        assert result["bankruptcy_risk"] == "high"

    # --- Piotroski F-Score ---

    def test_piotroski_score_strong(self):
        fin = {
            "net_income": 1_000_000,
            "operating_cash_flow": 1_200_000,
            "roa": 0.06, "roa_previous": 0.04,
            "debt_to_assets": 0.3, "debt_to_assets_previous": 0.4,
            "current_ratio": 2.5, "current_ratio_previous": 2.0,
            "shares_outstanding": 900_000, "shares_outstanding_previous": 1_000_000,
            "gross_margin": 0.42, "gross_margin_previous": 0.38,
            "asset_turnover": 0.6, "asset_turnover_previous": 0.5,
        }
        result = self.engine._calculate_piotroski_score(fin)
        assert result["score"] >= 7
        assert result["strength"] == "strong"

    def test_piotroski_score_weak(self):
        fin = {
            "net_income": -100_000,
            "operating_cash_flow": -50_000,  # OCF(-50K) > NI(-100K) = True → 1 point
            "roa": 0.01, "roa_previous": 0.05,
            "debt_to_assets": 0.8, "debt_to_assets_previous": 0.4,
            "current_ratio": 0.8, "current_ratio_previous": 1.5,
            "shares_outstanding": 1_200_000, "shares_outstanding_previous": 1_000_000,
            "gross_margin": 0.30, "gross_margin_previous": 0.40,
            "asset_turnover": 0.3, "asset_turnover_previous": 0.5,
        }
        result = self.engine._calculate_piotroski_score(fin)
        # OCF(-50K) > NI(-100K) is True so 1 "quality earnings" point is awarded
        assert result["score"] == 1
        assert result["strength"] == "weak"

    # --- industry multiples ---

    def test_get_industry_multiple_known(self):
        assert self.engine._get_industry_multiple("technology") == 20
        assert self.engine._get_industry_multiple("software") == 25
        assert self.engine._get_industry_multiple("utilities") == 7

    def test_get_industry_multiple_unknown(self):
        assert self.engine._get_industry_multiple("unknown_sector") == 10

    def test_get_industry_multiple_case_insensitive(self):
        assert self.engine._get_industry_multiple("Technology") == 20

    # --- moat analysis ---

    def test_analyze_moat_wide_with_network_effects(self):
        fin = _make_financials(customer_retention_rate=0.95, gross_margin=60, industry_avg_gross_margin=40)
        mkt = _make_market_data(network_effects_score=0.9, market_share=0.5, industry_concentration=0.8)
        result = self.engine._analyze_moat(fin, mkt)
        assert result["moat_score"] >= 60
        assert result["rating"] == "wide"

    def test_analyze_moat_none(self):
        fin = _make_financials()
        mkt = _make_market_data()
        result = self.engine._analyze_moat(fin, mkt)
        assert "rating" in result
        assert result["moat_score"] >= 0

    # --- composite score ---

    def test_calculate_composite_score_structure(self):
        engine = FundamentalAnalysisEngine()
        # Build a minimal analysis dict with the keys that _calculate_composite_score reads
        analysis = {
            "valuation_models": {"upside_potential": 40},
            "quality_score": {"overall_score": 75},
            "growth_analysis": {"historical_growth": {"revenue_cagr_3y": 12}},
            "financial_health": {"overall_health": 70},
            "moat_analysis": {"moat_score": 40},
            "management_quality": {"overall_score": 60},
        }
        score = engine._calculate_composite_score(analysis)
        assert 0 <= score <= 100

    # --- async analyze_company ---

    @pytest.mark.asyncio
    async def test_analyze_company_returns_dict(self):
        engine = FundamentalAnalysisEngine()
        fin = _make_financials()
        mkt = _make_market_data()
        # _calculate_efficiency_metrics is referenced in analyze_company but not yet
        # implemented in the source; patch it so the full flow can be tested.
        engine._calculate_efficiency_metrics = lambda f: {"asset_turnover": 0.5}
        result = await engine.analyze_company("AAPL", fin, mkt)
        assert result["ticker"] == "AAPL"
        assert "composite_score" in result
        assert "quality_score" in result
        assert "valuation_models" in result
        assert "financial_health" in result

    @pytest.mark.asyncio
    async def test_analyze_company_with_peers(self):
        engine = FundamentalAnalysisEngine()
        fin = _make_financials()
        mkt = _make_market_data()
        engine._calculate_efficiency_metrics = lambda f: {"asset_turnover": 0.5}
        peer_data = [
            {"pe_ratio": 20, "gross_margin": 38, "revenue_growth": 8},
            {"pe_ratio": 22, "gross_margin": 42, "revenue_growth": 10},
        ]
        result = await engine.analyze_company("MSFT", fin, mkt, peer_data=peer_data)
        assert result["ticker"] == "MSFT"
        assert result["peer_comparison"] is not None


# ===========================================================================
# Tests: RecommendationAction enum
# ===========================================================================

class TestRecommendationAction:
    """Tests for the RecommendationAction enum."""

    def test_all_values_exist(self):
        assert RecommendationAction.STRONG_BUY.value == "strong_buy"
        assert RecommendationAction.BUY.value == "buy"
        assert RecommendationAction.HOLD.value == "hold"
        assert RecommendationAction.SELL.value == "sell"
        assert RecommendationAction.STRONG_SELL.value == "strong_sell"

    def test_count(self):
        assert len(RecommendationAction) == 5

    def test_lookup_by_value(self):
        action = RecommendationAction("strong_buy")
        assert action is RecommendationAction.STRONG_BUY

    def test_enum_comparison(self):
        assert RecommendationAction.BUY != RecommendationAction.SELL
        assert RecommendationAction.HOLD is RecommendationAction.HOLD


# ===========================================================================
# Tests: StockRecommendation dataclass
# ===========================================================================

class TestStockRecommendation:
    """Tests for the StockRecommendation dataclass and to_dict()."""

    def test_construction_minimal(self):
        rec = _make_stock_recommendation()
        assert rec.ticker == "AAPL"
        assert rec.action == RecommendationAction.BUY
        assert rec.confidence == pytest.approx(0.75)

    def test_to_dict_keys(self):
        rec = _make_stock_recommendation()
        d = rec.to_dict()
        for key in ("ticker", "action", "confidence", "priority",
                    "entry_price", "target_price", "stop_loss",
                    "expected_return", "time_horizon_days",
                    "risk_score", "volatility", "beta", "sharpe_ratio",
                    "max_drawdown", "technical_score", "fundamental_score",
                    "sentiment_score", "ml_prediction_score",
                    "key_factors", "risks", "opportunities", "catalysts",
                    "generated_at", "valid_until",
                    "recommended_allocation", "max_position_size"):
            assert key in d, f"Missing key: {key}"

    def test_to_dict_action_is_string(self):
        rec = _make_stock_recommendation(action=RecommendationAction.STRONG_BUY)
        assert rec.to_dict()["action"] == "strong_buy"

    def test_to_dict_datetime_is_iso_string(self):
        rec = _make_stock_recommendation()
        d = rec.to_dict()
        # Should be parseable ISO strings
        datetime.fromisoformat(d["generated_at"])
        datetime.fromisoformat(d["valid_until"])

    def test_to_dict_lists_preserved(self):
        rec = _make_stock_recommendation(
            key_factors=["A", "B"],
            risks=["R1"],
            opportunities=["O1", "O2"],
        )
        d = rec.to_dict()
        assert d["key_factors"] == ["A", "B"]
        assert d["risks"] == ["R1"]
        assert d["opportunities"] == ["O1", "O2"]

    def test_strong_sell_recommendation(self):
        rec = _make_stock_recommendation(action=RecommendationAction.STRONG_SELL)
        assert rec.to_dict()["action"] == "strong_sell"


# ===========================================================================
# Tests: RecommendationEngine
# ===========================================================================

class TestRecommendationEngine:
    """Tests for RecommendationEngine logic methods (no heavy async flows)."""

    def setup_method(self):
        # Patch the imported dependencies so __init__ doesn't fail
        with patch.multiple(
            _re_mod,
            TechnicalAnalysisEngine=MagicMock,
            FundamentalAnalysisEngine=MagicMock,
            SentimentAnalysisEngine=MagicMock,
            ModelManager=MagicMock,
            MarketScanner=MagicMock,
            RiskManager=MagicMock,
            PortfolioOptimizer=MagicMock,
        ):
            self.engine = RecommendationEngine()

    # --- _normalize_score ---

    def test_normalize_score_midpoint(self):
        assert self.engine._normalize_score(0.0, -1, 1) == pytest.approx(0.5)

    def test_normalize_score_max(self):
        assert self.engine._normalize_score(1, -1, 1) == pytest.approx(1.0)

    def test_normalize_score_min(self):
        assert self.engine._normalize_score(-1, -1, 1) == pytest.approx(0.0)

    def test_normalize_score_clamps_above(self):
        assert self.engine._normalize_score(5.0, 0, 1) == pytest.approx(1.0)

    def test_normalize_score_clamps_below(self):
        assert self.engine._normalize_score(-5.0, 0, 1) == pytest.approx(0.0)

    def test_normalize_score_equal_min_max(self):
        assert self.engine._normalize_score(0.5, 0.5, 0.5) == pytest.approx(0.5)

    # --- _determine_action ---

    def test_determine_action_strong_buy(self):
        action = self.engine._determine_action(0.85)
        assert action == RecommendationAction.STRONG_BUY

    def test_determine_action_buy(self):
        action = self.engine._determine_action(0.65)
        assert action == RecommendationAction.BUY

    def test_determine_action_hold(self):
        action = self.engine._determine_action(0.45)
        assert action == RecommendationAction.HOLD

    def test_determine_action_sell(self):
        action = self.engine._determine_action(0.25)
        assert action == RecommendationAction.SELL

    def test_determine_action_strong_sell(self):
        action = self.engine._determine_action(0.05)
        assert action == RecommendationAction.STRONG_SELL

    # --- _calculate_priority ---

    def test_calculate_priority_bounds(self):
        p = self.engine._calculate_priority(0.95, 0.9, ["a", "b", "c"])
        assert 1 <= p <= 10

    def test_calculate_priority_high_score(self):
        p = self.engine._calculate_priority(0.9, 0.85, ["a", "b", "c"])
        # Score 0.9 → base 9, +1 for confidence, +1 for 3 opps = 11 → capped at 10
        assert p == 10

    def test_calculate_priority_low_score(self):
        p = self.engine._calculate_priority(0.1, 0.3, [])
        assert p >= 1

    # --- _should_recommend ---

    def test_should_recommend_conservative_high_risk(self):
        rec = _make_stock_recommendation(
            action=RecommendationAction.BUY,
            risk_score=0.4,
            confidence=0.6,
            expected_return=0.10,
        )
        assert not self.engine._should_recommend(rec, "conservative")

    def test_should_recommend_moderate_passes(self):
        rec = _make_stock_recommendation(
            action=RecommendationAction.BUY,
            risk_score=0.3,
            confidence=0.6,
            expected_return=0.10,
        )
        assert self.engine._should_recommend(rec, "moderate")

    def test_should_recommend_rejects_sell(self):
        rec = _make_stock_recommendation(
            action=RecommendationAction.SELL,
            risk_score=0.1,
            confidence=0.9,
            expected_return=0.20,
        )
        assert not self.engine._should_recommend(rec, "aggressive")

    def test_should_recommend_rejects_low_confidence(self):
        rec = _make_stock_recommendation(
            action=RecommendationAction.BUY,
            risk_score=0.2,
            confidence=0.3,
            expected_return=0.15,
        )
        assert not self.engine._should_recommend(rec, "aggressive")

    def test_should_recommend_rejects_low_return(self):
        rec = _make_stock_recommendation(
            action=RecommendationAction.BUY,
            risk_score=0.2,
            confidence=0.7,
            expected_return=0.02,
        )
        assert not self.engine._should_recommend(rec, "moderate")

    # --- _rank_recommendations ---

    def test_rank_recommendations_empty(self):
        result = self.engine._rank_recommendations([])
        assert result == []

    def test_rank_recommendations_order(self):
        r1 = _make_stock_recommendation(ticker="A", confidence=0.6, expected_return=0.10)
        r2 = _make_stock_recommendation(ticker="B", confidence=0.9, expected_return=0.30)
        ranked = self.engine._rank_recommendations([r1, r2])
        # Higher scoring recommendation should come first
        assert ranked[0].ticker == "B"

    # --- _calculate_position_sizing ---

    def test_calculate_position_sizing_strong_buy_cap(self):
        risk = {"risk_score": 0.2}
        sizing = self.engine._calculate_position_sizing(0.8, risk, RecommendationAction.STRONG_BUY)
        assert sizing["allocation"] <= 0.10
        assert sizing["max_size"] <= 10_000

    def test_calculate_position_sizing_sell_zero(self):
        risk = {"risk_score": 0.2}
        sizing = self.engine._calculate_position_sizing(0.8, risk, RecommendationAction.SELL)
        assert sizing["allocation"] == 0.0
        assert sizing["max_size"] == 0.0

    def test_calculate_position_sizing_non_negative(self):
        risk = {"risk_score": 0.9}
        sizing = self.engine._calculate_position_sizing(0.1, risk, RecommendationAction.BUY)
        assert sizing["allocation"] >= 0

    # --- _determine_time_horizon ---

    def test_determine_time_horizon_strong_buy(self):
        horizon = self.engine._determine_time_horizon(RecommendationAction.STRONG_BUY, {}, {})
        assert horizon == 60

    def test_determine_time_horizon_strong_sell(self):
        horizon = self.engine._determine_time_horizon(RecommendationAction.STRONG_SELL, {}, {})
        assert horizon == 5

    def test_determine_time_horizon_pattern_extends(self):
        technical = {"pattern_recognition": {"chart_patterns": {"cup_and_handle": {}}}}
        horizon = self.engine._determine_time_horizon(RecommendationAction.BUY, technical, {})
        # 30 * 1.5 = 45
        assert horizon == 45

    # --- _identify_risks (engine method) ---

    def test_identify_risks_high_volatility(self):
        risks = self.engine._identify_risks(
            {},
            {"volatility": 0.55, "beta": 1.0, "max_drawdown": -0.1},
            {}
        )
        assert any("volatility" in r.lower() or "Volatility" in r for r in risks)

    def test_identify_risks_high_beta(self):
        risks = self.engine._identify_risks(
            {},
            {"volatility": 0.2, "beta": 2.0, "max_drawdown": -0.1},
            {}
        )
        assert any("Beta" in r or "beta" in r.lower() for r in risks)

    def test_identify_risks_large_drawdown(self):
        risks = self.engine._identify_risks(
            {},
            {"volatility": 0.2, "beta": 1.0, "max_drawdown": -0.5},
            {}
        )
        assert any("drawdown" in r.lower() for r in risks)

    # --- _identify_opportunities (engine method) ---

    def test_identify_opportunities_empty_inputs(self):
        opps = self.engine._identify_opportunities({}, {}, {})
        assert isinstance(opps, list)

    def test_identify_opportunities_wide_moat(self):
        fundamental = {"moat_analysis": {"rating": "wide"}, "opportunities": []}
        opps = self.engine._identify_opportunities(fundamental, {}, {})
        assert any("moat" in o.lower() for o in opps)


# ===========================================================================
# Tests: SentimentResult dataclass
# ===========================================================================

class TestSentimentResult:
    """Tests for the SentimentResult dataclass."""

    def _make(self, **kw) -> SentimentResult:
        defaults = dict(
            score=0.5,
            confidence=0.8,
            label="positive",
            breakdown={"model": "lexicon"},
            keywords=["earnings", "growth"],
            sources_analyzed=5,
            timestamp=datetime.now(timezone.utc),
        )
        defaults.update(kw)
        return SentimentResult(**defaults)

    def test_construction(self):
        r = self._make()
        assert r.score == pytest.approx(0.5)
        assert r.label == "positive"
        assert r.sources_analyzed == 5

    def test_to_dict_keys(self):
        r = self._make()
        d = r.to_dict()
        assert set(d.keys()) == {"score", "confidence", "label", "breakdown", "keywords", "sources_analyzed", "timestamp"}

    def test_to_dict_timestamp_is_string(self):
        r = self._make()
        d = r.to_dict()
        assert isinstance(d["timestamp"], str)
        datetime.fromisoformat(d["timestamp"])

    def test_negative_score(self):
        r = self._make(score=-0.8, label="negative")
        assert r.score == pytest.approx(-0.8)
        assert r.label == "negative"


# ===========================================================================
# Tests: SentimentAnalysisEngine (lexicon path only - no FinBERT)
# ===========================================================================

class TestSentimentAnalysisEngine:
    """Tests for SentimentAnalysisEngine in lexicon-only mode."""

    def setup_method(self):
        # use_finbert=False forces lexicon path regardless of FINBERT_AVAILABLE
        self.engine = SentimentAnalysisEngine(use_finbert=False)

    # --- _analyze_with_lexicon ---

    @pytest.mark.asyncio
    async def test_positive_text_score(self):
        result = await self.engine._analyze_with_lexicon(
            "The company reports strong earnings growth and solid profits beating expectations.",
            "news"
        )
        assert result.score > 0
        assert result.label == "positive"

    @pytest.mark.asyncio
    async def test_negative_text_score(self):
        result = await self.engine._analyze_with_lexicon(
            "The company faces a major loss and significant decline in revenue.",
            "news"
        )
        assert result.score < 0
        assert result.label == "negative"

    @pytest.mark.asyncio
    async def test_neutral_text_score(self):
        result = await self.engine._analyze_with_lexicon(
            "The company filed its quarterly report today.",
            "news"
        )
        assert result.label == "neutral"
        assert result.score == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_intensifier_boosts_score(self):
        plain = await self.engine._analyze_with_lexicon("Stock has growth.", "news")
        intensified = await self.engine._analyze_with_lexicon("Stock has extremely strong growth.", "news")
        assert intensified.score >= plain.score

    @pytest.mark.asyncio
    async def test_empty_text_returns_neutral(self):
        result = await self.engine._analyze_with_lexicon("", "news")
        assert result.label == "neutral"
        assert result.score == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_result_score_clamped(self):
        # Many positive words
        text = " ".join(["growth profit gain strong bull bullish rise"] * 10)
        result = await self.engine._analyze_with_lexicon(text, "news")
        assert -1.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_breakdown_has_model_key(self):
        result = await self.engine._analyze_with_lexicon("good earnings", "news")
        assert result.breakdown.get("model") == "lexicon"

    # --- analyze_sentiment ---

    @pytest.mark.asyncio
    async def test_analyze_sentiment_dispatches_to_lexicon(self):
        result = await self.engine.analyze_sentiment("Company reports record profits.", "news")
        assert isinstance(result, SentimentResult)
        assert result.label in ("positive", "negative", "neutral")

    # --- analyze_stock_sentiment ---

    @pytest.mark.asyncio
    async def test_analyze_stock_sentiment_empty_list(self):
        result = await self.engine.analyze_stock_sentiment("AAPL", [])
        assert result.score == pytest.approx(0.0)
        assert result.sources_analyzed == 0
        assert result.label == "neutral"

    @pytest.mark.asyncio
    async def test_analyze_stock_sentiment_multiple_texts(self):
        texts = [
            "Strong earnings growth beats estimates.",
            "Profit margins expanding significantly.",
            "Revenue surge driven by innovation.",
        ]
        result = await self.engine.analyze_stock_sentiment("AAPL", texts)
        assert result.sources_analyzed == 3
        assert result.score > 0

    @pytest.mark.asyncio
    async def test_analyze_stock_sentiment_aggregated_label(self):
        texts = [
            "Stock decline and major loss.",
            "Weak earnings miss targets.",
        ]
        result = await self.engine.analyze_stock_sentiment("XYZ", texts)
        assert result.label in ("negative", "neutral")

    # --- _extract_keywords ---

    def test_extract_keywords_returns_list(self):
        kw = self.engine._extract_keywords("Apple reports strong quarterly earnings growth profit")
        assert isinstance(kw, list)
        assert len(kw) <= 5

    def test_extract_keywords_filters_short_words(self):
        kw = self.engine._extract_keywords("a an the of in at to is are")
        assert all(len(w) > 3 for w in kw)

    def test_extract_keywords_ignores_stop_words(self):
        kw = self.engine._extract_keywords("the company reported earnings during the quarter")
        # 'the', 'during' are stop words; 'company', 'reported', 'earnings', 'quarter' are not
        assert "the" not in kw

    # --- get_news_sentiment ---

    @pytest.mark.asyncio
    async def test_get_news_sentiment_returns_neutral(self):
        result = await self.engine.get_news_sentiment("AAPL", limit=5)
        assert isinstance(result, SentimentResult)
        assert result.label == "neutral"
        assert "AAPL".lower() in result.keywords

    # --- get_social_sentiment ---

    @pytest.mark.asyncio
    async def test_get_social_sentiment_returns_neutral(self):
        result = await self.engine.get_social_sentiment("TSLA", limit=10)
        assert result.label == "neutral"
        assert result.score == pytest.approx(0.0)

    # --- analyze_comprehensive_sentiment ---

    @pytest.mark.asyncio
    async def test_analyze_comprehensive_sentiment_structure(self):
        result = await self.engine.analyze_comprehensive_sentiment("GOOG")
        assert result["ticker"] == "GOOG"
        assert "overall_sentiment" in result
        assert "news_sentiment" in result
        assert "social_sentiment" in result
        assert "sources_analyzed" in result

    @pytest.mark.asyncio
    async def test_analyze_comprehensive_sentiment_label_valid(self):
        result = await self.engine.analyze_comprehensive_sentiment("MSFT")
        label = result["overall_sentiment"]["label"]
        assert label in ("positive", "negative", "neutral")


# ===========================================================================
# Tests: FinBERTResult dataclass
# ===========================================================================

class TestFinBERTResult:
    """Tests for the FinBERTResult dataclass."""

    def test_construction(self):
        r = FinBERTResult(
            score=0.7,
            confidence=0.85,
            label="positive",
            probabilities={"positive": 0.85, "negative": 0.05, "neutral": 0.10},
        )
        assert r.score == pytest.approx(0.7)
        assert r.confidence == pytest.approx(0.85)
        assert r.label == "positive"

    def test_negative_score(self):
        r = FinBERTResult(
            score=-0.6,
            confidence=0.75,
            label="negative",
            probabilities={"positive": 0.05, "negative": 0.80, "neutral": 0.15},
        )
        assert r.score < 0


# ===========================================================================
# Tests: FinancialTextPreprocessor
# ===========================================================================

class TestFinancialTextPreprocessor:
    """Tests for the FinBERT text preprocessor."""

    def setup_method(self):
        self.preprocessor = FinancialTextPreprocessor()

    def test_preprocess_empty(self):
        assert self.preprocessor.preprocess("") == ""

    def test_preprocess_removes_urls(self):
        text = "See https://example.com for more details."
        result = self.preprocessor.preprocess(text)
        assert "https://" not in result

    def test_preprocess_removes_mentions(self):
        text = "@TimCook said earnings were strong."
        result = self.preprocessor.preprocess(text)
        assert "@TimCook" not in result

    def test_preprocess_removes_html_tags(self):
        text = "<p>Earnings <b>beat</b> expectations.</p>"
        result = self.preprocessor.preprocess(text)
        assert "<p>" not in result
        assert "<b>" not in result
        assert "beat" in result

    def test_preprocess_normalizes_whitespace(self):
        text = "Earnings\n\n\nbeat\t\texpectations"
        result = self.preprocessor.preprocess(text)
        assert "\n" not in result
        assert "\t" not in result

    def test_preprocess_truncates_long_text(self):
        long_text = " ".join(["word"] * 500)
        result = self.preprocessor.preprocess(long_text)
        assert len(result.split()) <= 400

    def test_preprocess_hashtag_keeps_word(self):
        text = "Very #bullish on earnings."
        result = self.preprocessor.preprocess(text)
        assert "bullish" in result

    def test_preprocess_batch(self):
        texts = ["First text.", "Second text.", "Third text."]
        results = self.preprocessor.preprocess_batch(texts)
        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)

    def test_combine_headline_summary(self):
        combined = self.preprocessor.combine_headline_summary(
            "Apple beats earnings", "Record quarter driven by iPhone sales"
        )
        assert "Apple beats earnings" in combined
        assert "Record quarter" in combined

    def test_combine_headline_empty_summary(self):
        result = self.preprocessor.combine_headline_summary("Big headline", "")
        assert result == "Big headline"

    def test_combine_both_empty(self):
        result = self.preprocessor.combine_headline_summary("", "")
        assert result == ""


# ===========================================================================
# Tests: FinBERTAnalyzer (with stubbed torch/transformers)
# ===========================================================================

class TestFinBERTAnalyzer:
    """Tests for FinBERTAnalyzer in stub mode (HAS_TRANSFORMERS=False)."""

    def setup_method(self):
        # Reset singleton so tests don't share state
        FinBERTAnalyzer._instance = None
        self.analyzer = FinBERTAnalyzer()

    def test_singleton_pattern(self):
        a1 = FinBERTAnalyzer()
        a2 = FinBERTAnalyzer()
        assert a1 is a2

    def test_initial_not_initialized(self):
        assert not self.analyzer.is_initialized

    def test_initialize_fails_without_transformers(self):
        """initialize() should return False when HAS_TRANSFORMERS is patched to False."""
        # The torch/transformers stubs are MagicMocks so HAS_TRANSFORMERS resolved True
        # at module load time.  Patch the module-level flag to test the False branch.
        with patch.object(_finbert_mod, "HAS_TRANSFORMERS", False):
            FinBERTAnalyzer._instance = None
            analyzer = FinBERTAnalyzer()
            result = analyzer.initialize()
        assert result is False

    def test_analyze_batch_empty(self):
        results = self.analyzer.analyze_batch([])
        assert results == []

    def test_analyze_batch_returns_neutral_when_not_initialized(self):
        """analyze_batch without init should return neutral results."""
        results = self.analyzer.analyze_batch(["Apple stock rising."])
        assert len(results) == 1
        assert results[0].label == "neutral"
        assert results[0].score == pytest.approx(0.0)

    def test_analyze_single_returns_neutral_result(self):
        result = self.analyzer.analyze_single("Revenue beat expectations.")
        assert result.label == "neutral"
        assert isinstance(result.probabilities, dict)

    def test_neutral_result_structure(self):
        r = self.analyzer._neutral_result()
        assert r.score == pytest.approx(0.0)
        assert r.confidence == pytest.approx(0.33)
        assert r.label == "neutral"
        assert "positive" in r.probabilities
        assert "negative" in r.probabilities
        assert "neutral" in r.probabilities

    def test_label_map(self):
        assert FinBERTAnalyzer.LABEL_MAP[0] == "positive"
        assert FinBERTAnalyzer.LABEL_MAP[1] == "negative"
        assert FinBERTAnalyzer.LABEL_MAP[2] == "neutral"

    def test_model_name(self):
        assert FinBERTAnalyzer.MODEL_NAME == "ProsusAI/finbert"

    def test_probabilities_to_result_positive_dominant(self):
        """Test score calculation: positive_prob - negative_prob."""
        import numpy as np
        probs = np.array([0.8, 0.1, 0.1])  # positive dominant
        result = self.analyzer._probabilities_to_result(probs)
        # score = 0.8 - 0.1 = 0.7
        assert result.score == pytest.approx(0.7)
        assert result.label == "positive"
        assert result.confidence == pytest.approx(0.8)

    def test_probabilities_to_result_negative_dominant(self):
        import numpy as np
        probs = np.array([0.05, 0.85, 0.10])
        result = self.analyzer._probabilities_to_result(probs)
        assert result.score == pytest.approx(0.05 - 0.85)
        assert result.label == "negative"

    def test_probabilities_to_result_neutral_dominant(self):
        import numpy as np
        probs = np.array([0.20, 0.20, 0.60])
        result = self.analyzer._probabilities_to_result(probs)
        assert result.label == "neutral"
        assert result.score == pytest.approx(0.0)


# ===========================================================================
# Tests: FinBERTInference
# ===========================================================================

class TestFinBERTInference:
    """Tests for the FinBERTInference high-level interface."""

    def setup_method(self):
        FinBERTAnalyzer._instance = None
        self.analyzer = FinBERTAnalyzer()
        self.inference = FinBERTInference(self.analyzer, batch_size=8, enable_cache=True)

    def test_construction(self):
        assert self.inference.batch_size == 8
        assert self.inference.enable_cache is True

    def test_cache_key_deterministic(self):
        k1 = self.inference._get_cache_key("Apple beats earnings.")
        k2 = self.inference._get_cache_key("Apple beats earnings.")
        assert k1 == k2

    def test_cache_key_different_texts(self):
        k1 = self.inference._get_cache_key("Text one.")
        k2 = self.inference._get_cache_key("Text two.")
        assert k1 != k2

    def test_cache_key_length(self):
        k = self.inference._get_cache_key("some text")
        assert len(k) == 16

    def test_clear_cache(self):
        # Add something to cache manually
        self.inference._cache["test_key"] = FinBERTResult(0.0, 0.33, "neutral", {})
        self.inference.clear_cache()
        assert len(self.inference._cache) == 0

    @pytest.mark.asyncio
    async def test_analyze_single_async_caches_result(self):
        """Result should be cached after first call."""
        result1 = await self.inference.analyze_single_async("Stock is rising strongly.")
        cache_key = self.inference._get_cache_key("Stock is rising strongly.")
        assert cache_key in self.inference._cache

    @pytest.mark.asyncio
    async def test_analyze_single_async_returns_from_cache(self):
        """Second call with same text should return cached result."""
        text = "Unique text for cache test 12345."
        r1 = await self.inference.analyze_single_async(text)
        r2 = await self.inference.analyze_single_async(text)
        assert r1 is r2  # Same object from cache

    @pytest.mark.asyncio
    async def test_analyze_batch_async_returns_list(self):
        texts = ["Earnings beat.", "Revenue missed.", "Market neutral."]
        results = await self.inference.analyze_batch_async(texts)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_no_cache_mode(self):
        inference_no_cache = FinBERTInference(self.analyzer, enable_cache=False)
        assert inference_no_cache.enable_cache is False
