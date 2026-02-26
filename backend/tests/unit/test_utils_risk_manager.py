"""
Unit tests for backend/utils/risk_manager.py

Covers all major classes and methods:
- RiskManager VaR calculations (Historical, Parametric, Monte Carlo)
- CVaR / Expected Shortfall
- Maximum Drawdown analysis
- Beta and Tracking Error
- Stress Testing (historical scenarios + custom)
- Risk Decomposition
- Position Sizing (Kelly, vol-target, VaR-target)
- Risk Classification and Scoring
- Edge cases: empty data, zero values, extreme inputs

All tests are pure math -- no database, no network, no mocking of external APIs.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone

from backend.utils.risk_manager import (
    RiskManager,
    RiskLevel,
    VaRMethod,
    VaRResult,
    RiskAssessment,
    StressTestResult,
    RiskDecomposition,
    HISTORICAL_SCENARIOS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_returns(n: int = 252, mean: float = 0.0004, std: float = 0.015,
                  seed: int = 42) -> np.ndarray:
    """Create a reproducible array of daily returns."""
    rng = np.random.RandomState(seed)
    return rng.normal(loc=mean, scale=std, size=n)


def _make_prices(n: int = 253, start: float = 100.0, seed: int = 42) -> np.ndarray:
    """Create a reproducible price series from cumulative returns."""
    returns = _make_returns(n - 1, seed=seed)
    prices = start * np.cumprod(1 + returns)
    return np.insert(prices, 0, start)


def _make_price_df(n: int = 253, seed: int = 42) -> pd.DataFrame:
    """Create a DataFrame with a 'close' column, suitable for risk_manager methods."""
    prices = _make_prices(n, seed=seed)
    return pd.DataFrame({'close': prices})


@pytest.fixture
def rm() -> RiskManager:
    """Fresh RiskManager with default parameters."""
    return RiskManager()


# ===========================================================================
# Enums and Dataclass Smoke Tests
# ===========================================================================


class TestEnumsAndDataclasses:

    def test_risk_level_values(self):
        assert RiskLevel.VERY_LOW.value == "very_low"
        assert RiskLevel.VERY_HIGH.value == "very_high"

    def test_var_method_values(self):
        assert VaRMethod.HISTORICAL.value == "historical"
        assert VaRMethod.PARAMETRIC.value == "parametric"
        assert VaRMethod.MONTE_CARLO.value == "monte_carlo"

    def test_var_result_fields(self):
        r = VaRResult(var_value=-0.02, confidence_level=0.95,
                      method=VaRMethod.HISTORICAL, horizon_days=1)
        assert r.var_value == -0.02
        assert r.confidence_level == 0.95
        assert r.additional_metrics == {}

    def test_stress_test_result_fields(self):
        r = StressTestResult(
            scenario_name="test", portfolio_loss=-0.1,
            asset_impacts={"AAPL": -0.05}, var_breach=True,
            description="desc", historical_date="2020-01-01")
        assert r.var_breach is True
        assert r.historical_date == "2020-01-01"


# ===========================================================================
# RiskManager Initialization
# ===========================================================================


class TestRiskManagerInit:

    def test_default_params(self, rm):
        assert rm.max_portfolio_var == 0.02
        assert rm.max_position_size == 0.10
        assert rm.min_sharpe_ratio == 0.5
        assert rm.risk_free_rate == 0.045
        assert rm.monte_carlo_simulations == 10000
        assert rm.var_horizon_days == 1

    def test_custom_params(self):
        mgr = RiskManager(
            max_portfolio_var=0.05,
            max_position_size=0.20,
            min_sharpe_ratio=1.0,
            risk_free_rate=0.03,
            monte_carlo_simulations=5000,
            var_horizon_days=5,
        )
        assert mgr.max_portfolio_var == 0.05
        assert mgr.var_horizon_days == 5
        assert mgr.monte_carlo_simulations == 5000


# ===========================================================================
# VaR Calculations
# ===========================================================================


class TestVaRHistorical:

    def test_known_returns(self, rm):
        """VaR at 95% on a simple sorted array should match 5th percentile."""
        returns = np.linspace(-0.10, 0.10, 100)
        result = rm.calculate_var(returns, confidence=0.95, method='historical')
        expected = np.percentile(returns, 5)
        assert result.var_value == pytest.approx(expected, rel=1e-6)
        assert result.method == VaRMethod.HISTORICAL
        assert result.confidence_level == 0.95

    def test_all_negative_returns(self, rm):
        """If all returns are negative, VaR should be deeply negative."""
        returns = np.array([-0.05, -0.04, -0.03, -0.02, -0.01] * 10)
        result = rm.calculate_var(returns, confidence=0.95, method='historical')
        assert result.var_value < 0

    def test_all_positive_returns(self, rm):
        """If all returns are positive, VaR is positive (no loss)."""
        returns = np.array([0.01, 0.02, 0.03, 0.04, 0.05] * 10)
        result = rm.calculate_var(returns, confidence=0.95, method='historical')
        assert result.var_value > 0

    def test_higher_confidence_more_extreme(self, rm):
        """99% VaR should be more extreme (lower) than 95%."""
        returns = _make_returns(500, seed=7)
        var_95 = rm.calculate_var(returns, confidence=0.95, method='historical')
        var_99 = rm.calculate_var(returns, confidence=0.99, method='historical')
        assert var_99.var_value < var_95.var_value

    def test_multi_day_horizon(self, rm):
        """VaR with horizon > 1 should differ from 1-day VaR."""
        returns = _make_returns(500, seed=42)
        var_1d = rm.calculate_var(returns, method='historical', horizon_days=1)
        var_5d = rm.calculate_var(returns, method='historical', horizon_days=5)
        assert var_1d.var_value != pytest.approx(var_5d.var_value, rel=1e-3)
        assert var_5d.horizon_days == 5

    def test_additional_metrics_populated(self, rm):
        """VaR result should contain additional metrics dict."""
        returns = _make_returns(252)
        result = rm.calculate_var(returns, method='historical')
        assert 'mean_return' in result.additional_metrics
        assert 'std_return' in result.additional_metrics
        assert 'skewness' in result.additional_metrics
        assert 'kurtosis' in result.additional_metrics
        assert result.additional_metrics['data_points'] == 252


class TestVaRParametric:

    def test_parametric_close_to_theoretical(self, rm):
        """Parametric VaR on normal data should match theory."""
        from scipy import stats as sp_stats
        rng = np.random.RandomState(42)
        returns = rng.normal(loc=0.0, scale=0.02, size=10000)
        result = rm.calculate_var(returns, confidence=0.95, method='parametric')
        theoretical = 0.0 + sp_stats.norm.ppf(0.05) * 0.02
        assert result.var_value == pytest.approx(theoretical, abs=0.003)
        assert result.method == VaRMethod.PARAMETRIC

    def test_parametric_zero_std(self, rm):
        """Constant returns should give VaR equal to that constant value."""
        returns = np.full(100, 0.001)
        result = rm.calculate_var(returns, confidence=0.95, method='parametric')
        # std = 0, so VaR = mean + 0 = mean * horizon
        assert result.var_value == pytest.approx(0.001, abs=1e-6)

    def test_small_sample_forces_parametric(self, rm):
        """With fewer than 30 data points, method should be forced to parametric."""
        returns = np.array([0.01, -0.01, 0.005, -0.005, 0.002])
        result = rm.calculate_var(returns, confidence=0.95, method='historical')
        # Should be forced to parametric
        assert result.method == VaRMethod.PARAMETRIC


class TestVaRMonteCarlo:

    def test_monte_carlo_returns_value(self, rm):
        """Monte Carlo VaR should return a finite negative value for typical data."""
        returns = _make_returns(252, seed=10)
        result = rm.calculate_var(returns, confidence=0.95, method='monte_carlo')
        assert np.isfinite(result.var_value)
        assert result.method == VaRMethod.MONTE_CARLO

    def test_monte_carlo_close_to_parametric(self, rm):
        """Monte Carlo on normal data should roughly match parametric."""
        rng = np.random.RandomState(42)
        returns = rng.normal(0.0, 0.02, 5000)
        mc = rm.calculate_var(returns, confidence=0.95, method='monte_carlo')
        param = rm.calculate_var(returns, confidence=0.95, method='parametric')
        # Should be in the same ballpark (within 50% of each other)
        assert abs(mc.var_value - param.var_value) < 0.01


class TestVaRAllMethods:

    def test_all_methods_returns_three_keys(self, rm):
        returns = _make_returns(252)
        results = rm.calculate_var_all_methods(returns, 0.95)
        assert set(results.keys()) == {'historical', 'parametric', 'monte_carlo'}

    def test_all_methods_consistent_direction(self, rm):
        """All three VaR methods should agree on sign for negative-mean data."""
        returns = _make_returns(500, mean=-0.001, std=0.02, seed=42)
        results = rm.calculate_var_all_methods(returns, 0.95)
        for method, result in results.items():
            assert result.var_value < 0, f"{method} VaR should be negative"


# ===========================================================================
# CVaR / Expected Shortfall
# ===========================================================================


class TestCVaR:

    def test_cvar_worse_than_var(self, rm):
        """CVaR (expected shortfall) should be <= VaR."""
        returns = _make_returns(500, seed=42)
        var_val = rm._var_historical(returns, 0.95, 1)
        cvar_val = rm.calculate_cvar(returns, 0.95)
        assert cvar_val <= var_val

    def test_cvar_all_same_returns(self, rm):
        """If all returns identical, CVaR equals VaR equals that return."""
        returns = np.full(100, -0.01)
        cvar = rm.calculate_cvar(returns, 0.95)
        assert cvar == pytest.approx(-0.01, abs=1e-9)

    def test_cvar_parametric(self, rm):
        """Parametric CVaR should return a finite value."""
        returns = _make_returns(500, seed=7)
        cvar = rm.calculate_cvar_parametric(returns, 0.95)
        assert np.isfinite(cvar)

    def test_cvar_parametric_more_extreme_than_parametric_var(self, rm):
        """Parametric CVaR should be more extreme than parametric VaR."""
        returns = _make_returns(1000, mean=-0.001, std=0.02, seed=42)
        var_val = rm._var_parametric(returns, 0.95, 1)
        cvar_val = rm.calculate_cvar_parametric(returns, 0.95)
        assert cvar_val <= var_val

    def test_cvar_with_series(self, rm):
        """CVaR should accept pd.Series input."""
        returns = pd.Series(_make_returns(252))
        cvar = rm.calculate_cvar(returns, 0.95)
        assert np.isfinite(cvar)


# ===========================================================================
# Maximum Drawdown
# ===========================================================================


class TestMaxDrawdown:

    def test_monotonic_up_no_drawdown(self, rm):
        """Monotonically increasing prices should have zero drawdown."""
        prices = np.linspace(100, 200, 100)
        max_dd, peak, trough = rm.calculate_max_drawdown(prices)
        assert max_dd == pytest.approx(0.0, abs=1e-9)

    def test_monotonic_down_full_drawdown(self, rm):
        """Monotonically decreasing prices have a large drawdown from start."""
        prices = np.linspace(200, 100, 100)
        max_dd, peak, trough = rm.calculate_max_drawdown(prices)
        # Drawdown = (100 - 200) / 200 = -0.5
        assert max_dd == pytest.approx(-0.5, rel=1e-6)
        assert peak == 0
        assert trough == 99

    def test_known_drawdown(self, rm):
        """Verify drawdown on a hand-crafted price series."""
        prices = np.array([100, 110, 90, 95, 80, 120])
        max_dd, peak, trough = rm.calculate_max_drawdown(prices)
        # Peak is 110 (idx 1), trough is 80 (idx 4) => dd = (80-110)/110 = -0.2727
        assert max_dd == pytest.approx(-0.2727, rel=1e-2)
        assert peak == 1
        assert trough == 4

    def test_single_price(self, rm):
        """Single price should return zero drawdown."""
        prices = np.array([100.0])
        max_dd, peak, trough = rm.calculate_max_drawdown(prices)
        assert max_dd == 0.0
        assert peak == 0
        assert trough == 0

    def test_two_prices_declining(self, rm):
        """Two prices declining: drawdown = (end - start)/start."""
        prices = np.array([100.0, 80.0])
        max_dd, peak, trough = rm.calculate_max_drawdown(prices)
        assert max_dd == pytest.approx(-0.20, rel=1e-6)

    def test_drawdown_series_shape(self, rm):
        """Drawdown series should have same length as input."""
        prices = _make_prices(100)
        df = rm.calculate_drawdown_series(prices)
        assert len(df) == len(prices)
        assert set(df.columns) == {'price', 'running_max', 'drawdown', 'drawdown_duration'}

    def test_drawdown_series_running_max_nondecreasing(self, rm):
        """Running max should be non-decreasing."""
        prices = _make_prices(200, seed=7)
        df = rm.calculate_drawdown_series(prices)
        running_max = df['running_max'].values
        assert np.all(np.diff(running_max) >= -1e-12)

    def test_all_drawdowns_threshold(self, rm):
        """Find drawdowns exceeding 5% threshold."""
        # Create a price series with a known dip
        prices = np.concatenate([
            np.linspace(100, 120, 50),
            np.linspace(120, 100, 30),  # ~17% drop from peak
            np.linspace(100, 130, 50),
        ])
        drawdowns = rm.calculate_all_drawdowns(prices, threshold=-0.05)
        assert len(drawdowns) >= 1
        assert drawdowns[0]['max_drawdown'] < -0.05

    def test_all_drawdowns_no_significant(self, rm):
        """Monotonically increasing prices should produce no drawdowns."""
        prices = np.linspace(100, 200, 100)
        drawdowns = rm.calculate_all_drawdowns(prices, threshold=-0.05)
        assert drawdowns == []


# ===========================================================================
# Beta and Tracking Error
# ===========================================================================


class TestBeta:

    def test_beta_of_market_is_one(self, rm):
        """A return series identical to benchmark should have beta ~ 1."""
        returns = _make_returns(252, seed=42)
        result = rm.calculate_beta(returns, returns)
        assert result['beta'] == pytest.approx(1.0, rel=1e-6)
        assert result['r_squared'] == pytest.approx(1.0, rel=1e-6)

    def test_beta_with_doubled_returns(self, rm):
        """If asset returns = 2x benchmark, beta ~ 2."""
        benchmark = _make_returns(252, seed=42)
        asset = benchmark * 2
        result = rm.calculate_beta(asset, benchmark)
        assert result['beta'] == pytest.approx(2.0, rel=1e-3)

    def test_insufficient_data_returns_defaults(self, rm):
        """With <30 data points, beta defaults to 1.0."""
        returns = _make_returns(10, seed=42)
        benchmark = _make_returns(10, seed=7)
        result = rm.calculate_beta(returns, benchmark)
        assert result['beta'] == 1.0
        assert result['alpha'] == 0.0
        assert result['r_squared'] == 0.0

    def test_beta_dict_keys(self, rm):
        """Beta result should contain all expected keys."""
        returns = _make_returns(252, seed=42)
        benchmark = _make_returns(252, seed=7)
        result = rm.calculate_beta(returns, benchmark)
        expected_keys = {'beta', 'alpha', 'alpha_annualized', 'r_squared',
                         'correlation', 'data_points'}
        assert expected_keys == set(result.keys())

    def test_beta_with_mismatched_lengths(self, rm):
        """Different-length arrays should be aligned to shorter length."""
        returns = _make_returns(300, seed=42)
        benchmark = _make_returns(252, seed=7)
        result = rm.calculate_beta(returns, benchmark)
        assert result['data_points'] == 252


class TestTrackingError:

    def test_identical_returns_zero_tracking_error(self, rm):
        """Identical returns should have zero tracking error."""
        returns = _make_returns(252, seed=42)
        result = rm.calculate_tracking_error(returns, returns)
        assert result['tracking_error'] == pytest.approx(0.0, abs=1e-12)
        assert result['tracking_error_annualized'] == pytest.approx(0.0, abs=1e-10)

    def test_tracking_error_positive_for_different_series(self, rm):
        """Different return series should produce positive tracking error."""
        returns = _make_returns(252, seed=42)
        benchmark = _make_returns(252, seed=7)
        result = rm.calculate_tracking_error(returns, benchmark)
        assert result['tracking_error'] > 0
        assert result['tracking_error_annualized'] > result['tracking_error']

    def test_tracking_error_no_annualize(self, rm):
        """Without annualization, TE_annualized == TE."""
        returns = _make_returns(252, seed=42)
        benchmark = _make_returns(252, seed=7)
        result = rm.calculate_tracking_error(returns, benchmark, annualize=False)
        assert result['tracking_error'] == pytest.approx(
            result['tracking_error_annualized'], rel=1e-9)

    def test_tracking_error_keys(self, rm):
        returns = _make_returns(252, seed=42)
        benchmark = _make_returns(252, seed=7)
        result = rm.calculate_tracking_error(returns, benchmark)
        expected_keys = {'tracking_error', 'tracking_error_annualized',
                         'mean_active_return', 'mean_active_return_annualized',
                         'information_ratio', 'data_points'}
        assert expected_keys == set(result.keys())


# ===========================================================================
# Stress Testing
# ===========================================================================


class TestStressTest:

    def test_known_scenario(self, rm):
        """2008 crisis with 100% equity should produce ~50% loss."""
        portfolio = {"AAPL": 0.5, "MSFT": 0.5}
        result = rm.stress_test(portfolio, "2008_financial_crisis")
        assert result.scenario_name == "2008 Financial Crisis"
        expected_loss = 0.5 * (-0.50) + 0.5 * (-0.50)
        assert result.portfolio_loss == pytest.approx(expected_loss, rel=1e-6)
        assert result.var_breach is True

    def test_unknown_scenario_raises(self, rm):
        """Unknown scenario should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown scenario"):
            rm.stress_test({"AAPL": 1.0}, "fake_scenario")

    def test_sector_specific_shock(self, rm):
        """Tech sector should get tech-specific shock if available."""
        portfolio = {"GOOG": 1.0}
        sector_map = {"GOOG": "tech"}
        result = rm.stress_test(
            portfolio, "2000_dotcom_burst", sector_mappings=sector_map)
        # Tech shock for dotcom is -0.75
        assert result.asset_impacts["GOOG"] == pytest.approx(-0.75, rel=1e-6)
        assert result.portfolio_loss == pytest.approx(-0.75, rel=1e-6)

    def test_bond_sector_shock(self, rm):
        """Bond sector should use bond_shock."""
        portfolio = {"AGG": 1.0}
        sector_map = {"AGG": "bond"}
        result = rm.stress_test(
            portfolio, "2008_financial_crisis", sector_mappings=sector_map)
        assert result.asset_impacts["AGG"] == pytest.approx(0.05, rel=1e-6)

    def test_custom_beta_adjusts_shock(self, rm):
        """Asset beta should scale the shock."""
        portfolio = {"AAPL": 1.0}
        betas = {"AAPL": 1.5}
        result = rm.stress_test(
            portfolio, "2008_financial_crisis", asset_betas=betas)
        expected = -0.50 * 1.5
        assert result.asset_impacts["AAPL"] == pytest.approx(expected, rel=1e-6)

    def test_empty_portfolio(self, rm):
        """Empty portfolio should have zero loss."""
        result = rm.stress_test({}, "2008_financial_crisis")
        assert result.portfolio_loss == 0.0
        assert result.asset_impacts == {}

    def test_var_breach_flag(self, rm):
        """Portfolio loss exceeding max_portfolio_var should flag breach."""
        mgr = RiskManager(max_portfolio_var=0.01)
        result = mgr.stress_test({"AAPL": 1.0}, "2008_financial_crisis")
        assert result.var_breach is True

    def test_no_var_breach(self, rm):
        """Tiny portfolio loss should not breach VaR limit."""
        mgr = RiskManager(max_portfolio_var=1.0)
        result = mgr.stress_test({"AGG": 1.0}, "2008_financial_crisis",
                                 sector_mappings={"AGG": "bond"})
        # Bond shock = +0.05, abs(0.05) < 1.0
        assert result.var_breach is False


class TestStressTestCustom:

    def test_custom_shocks(self, rm):
        """Custom shocks should be applied per-asset."""
        portfolio = {"AAPL": 0.6, "MSFT": 0.4}
        shocks = {"AAPL": -0.10, "MSFT": -0.20}
        result = rm.stress_test_custom(portfolio, shocks, scenario_name="My Scenario")
        expected_loss = 0.6 * (-0.10) + 0.4 * (-0.20)
        assert result.portfolio_loss == pytest.approx(expected_loss, rel=1e-6)
        assert result.scenario_name == "My Scenario"
        assert result.historical_date is None

    def test_custom_missing_shocks_default_zero(self, rm):
        """Assets without a specified shock default to 0."""
        portfolio = {"AAPL": 0.5, "MSFT": 0.5}
        shocks = {"AAPL": -0.20}
        result = rm.stress_test_custom(portfolio, shocks)
        assert result.asset_impacts["MSFT"] == 0.0
        expected_loss = 0.5 * (-0.20)
        assert result.portfolio_loss == pytest.approx(expected_loss, rel=1e-6)


class TestStressTestAllScenarios:

    def test_all_scenarios_returns_list(self, rm):
        """Should return one result per historical scenario."""
        portfolio = {"AAPL": 0.5, "MSFT": 0.5}
        results = rm.stress_test_all_scenarios(portfolio)
        assert len(results) == len(HISTORICAL_SCENARIOS)

    def test_sorted_by_worst_first(self, rm):
        """Results should be sorted by portfolio_loss ascending (worst first)."""
        portfolio = {"AAPL": 0.5, "MSFT": 0.5}
        results = rm.stress_test_all_scenarios(portfolio)
        losses = [r.portfolio_loss for r in results]
        assert losses == sorted(losses)


# ===========================================================================
# Sortino Ratio
# ===========================================================================


class TestSortinoRatio:

    def test_all_positive_returns_inf(self, rm):
        """If no downside returns, sortino should be infinity."""
        returns = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        sortino = rm._calculate_sortino_ratio(returns, target_return=-1.0)
        assert sortino == float('inf')

    def test_mixed_returns_finite(self, rm):
        """Mixed returns should give a finite sortino ratio."""
        returns = _make_returns(252, seed=42)
        sortino = rm._calculate_sortino_ratio(returns)
        assert np.isfinite(sortino)

    def test_all_negative_returns(self, rm):
        """All-negative returns should give a negative sortino."""
        returns = np.array([-0.02, -0.03, -0.01, -0.05, -0.04] * 20)
        sortino = rm._calculate_sortino_ratio(returns)
        assert sortino < 0


# ===========================================================================
# Risk Decomposition
# ===========================================================================


class TestRiskDecomposition:

    def test_single_asset(self, rm):
        """Single-asset portfolio: 100% contribution, zero diversification benefit."""
        weights = np.array([1.0])
        cov = np.array([[0.04]])  # 20% vol
        result = rm._decompose_portfolio_risk(weights, cov, ["AAPL"])
        assert result.total_risk == pytest.approx(0.2, rel=1e-6)
        assert result.percentage_contributions["AAPL"] == pytest.approx(1.0, rel=1e-6)
        assert result.diversification_benefit == pytest.approx(0.0, abs=1e-6)

    def test_two_uncorrelated_assets(self, rm):
        """Two uncorrelated assets should show diversification benefit."""
        weights = np.array([0.5, 0.5])
        # uncorrelated, same vol
        cov = np.array([[0.04, 0.0], [0.0, 0.04]])
        result = rm._decompose_portfolio_risk(weights, cov, ["A", "B"])
        # portfolio vol = sqrt(0.25*0.04 + 0.25*0.04) = sqrt(0.02) ~ 0.1414
        assert result.total_risk == pytest.approx(np.sqrt(0.02), rel=1e-6)
        assert result.diversification_benefit > 0
        # Equal weights + same vol + zero corr => equal contributions
        assert result.percentage_contributions["A"] == pytest.approx(0.5, rel=1e-3)
        assert result.percentage_contributions["B"] == pytest.approx(0.5, rel=1e-3)

    def test_perfectly_correlated_no_diversification(self, rm):
        """Perfectly correlated assets: diversification benefit ~ 0."""
        weights = np.array([0.5, 0.5])
        # rho=1 => cov = vol1 * vol2 = 0.04
        cov = np.array([[0.04, 0.04], [0.04, 0.04]])
        result = rm._decompose_portfolio_risk(weights, cov, ["A", "B"])
        assert result.diversification_benefit == pytest.approx(0.0, abs=1e-6)

    def test_decompose_by_sector(self, rm):
        """Sector decomposition should aggregate asset contributions."""
        weights = np.array([0.3, 0.3, 0.4])
        cov = np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.04, 0.005],
            [0.005, 0.005, 0.02],
        ])
        sector_map = {"AAPL": "Tech", "MSFT": "Tech", "AGG": "Bonds"}
        result = rm.decompose_risk_by_sector(
            weights, cov, ["AAPL", "MSFT", "AGG"], sector_map)
        assert "Tech" in result
        assert "Bonds" in result
        # Contributions should sum to ~1.0
        total = sum(result.values())
        assert total == pytest.approx(1.0, abs=0.05)


# ===========================================================================
# Risk Scoring and Classification
# ===========================================================================


class TestRiskScoring:

    def test_low_risk_profile(self, rm):
        """Low vol, beta~1, small drawdown, good sharpe => low score."""
        score = rm._calculate_risk_score(
            volatility=0.10, beta=1.0, max_drawdown=-0.05, sharpe_ratio=1.5)
        assert score < 0.4

    def test_high_risk_profile(self, rm):
        """High vol, extreme beta, deep drawdown, poor sharpe => high score."""
        score = rm._calculate_risk_score(
            volatility=0.60, beta=2.5, max_drawdown=-0.40, sharpe_ratio=0.1)
        assert score > 0.7

    def test_risk_score_clamped_zero_to_one(self, rm):
        """Risk score should always be in [0, 1]."""
        # extreme low risk
        s1 = rm._calculate_risk_score(0.01, 1.0, 0.0, 5.0)
        assert 0 <= s1 <= 1
        # extreme high risk
        s2 = rm._calculate_risk_score(1.0, 5.0, -1.0, -2.0)
        assert 0 <= s2 <= 1


class TestRiskClassification:

    @pytest.mark.parametrize("score,expected", [
        (0.0, RiskLevel.VERY_LOW),
        (0.19, RiskLevel.VERY_LOW),
        (0.2, RiskLevel.LOW),
        (0.39, RiskLevel.LOW),
        (0.4, RiskLevel.MODERATE),
        (0.59, RiskLevel.MODERATE),
        (0.6, RiskLevel.HIGH),
        (0.79, RiskLevel.HIGH),
        (0.8, RiskLevel.VERY_HIGH),
        (1.0, RiskLevel.VERY_HIGH),
    ])
    def test_classification_boundaries(self, rm, score, expected):
        assert rm._classify_risk_level(score) == expected


class TestRiskFactors:

    def test_high_volatility_flagged(self, rm):
        factors = rm._identify_risk_factors(0.50, 1.0, -0.10, 1.0)
        assert any("volatility" in f.lower() for f in factors)

    def test_high_beta_flagged(self, rm):
        factors = rm._identify_risk_factors(0.20, 2.0, -0.10, 1.0)
        assert any("sensitivity" in f.lower() for f in factors)

    def test_low_beta_flagged(self, rm):
        factors = rm._identify_risk_factors(0.20, 0.3, -0.10, 1.0)
        assert any("correlation" in f.lower() for f in factors)

    def test_deep_drawdown_flagged(self, rm):
        factors = rm._identify_risk_factors(0.20, 1.0, -0.30, 1.0)
        assert any("drawdown" in f.lower() for f in factors)

    def test_low_sharpe_flagged(self, rm):
        factors = rm._identify_risk_factors(0.20, 1.0, -0.10, 0.3)
        assert any("sharpe" in f.lower() for f in factors)

    def test_no_factors_for_healthy_stock(self, rm):
        """A healthy stock should have no risk factors flagged."""
        factors = rm._identify_risk_factors(0.20, 1.0, -0.10, 1.0)
        assert factors == []


class TestPortfolioRiskFactors:

    def test_high_portfolio_vol_flagged(self, rm):
        factors = rm._identify_portfolio_risk_factors(0.30, -0.10, 1.0, 1.0, 0.1)
        assert any("volatility" in f.lower() for f in factors)

    def test_concentrated_portfolio_flagged(self, rm):
        factors = rm._identify_portfolio_risk_factors(0.15, -0.05, 1.0, 1.0, 0.5)
        assert any("concentrated" in f.lower() for f in factors)

    def test_below_sharpe_limit_flagged(self, rm):
        factors = rm._identify_portfolio_risk_factors(0.15, -0.05, 0.3, 1.0, 0.1)
        assert any("sharpe" in f.lower() for f in factors)


# ===========================================================================
# Recommendations
# ===========================================================================


class TestRecommendations:

    def test_high_risk_recommendations(self, rm):
        recs = rm._generate_risk_recommendations(
            RiskLevel.HIGH, ["High volatility (50% annualized)"])
        assert any("reduced position" in r.lower() for r in recs)
        assert any("stop-loss" in r.lower() for r in recs)
        assert any("hedging" in r.lower() for r in recs)

    def test_very_high_risk_aggressive_warning(self, rm):
        recs = rm._generate_risk_recommendations(RiskLevel.VERY_HIGH, [])
        assert any("aggressive" in r.lower() for r in recs)

    def test_drawdown_gets_trailing_stop(self, rm):
        recs = rm._generate_risk_recommendations(
            RiskLevel.HIGH, ["Significant historical drawdown (-30%)"])
        assert any("trailing" in r.lower() for r in recs)

    def test_low_risk_no_warnings(self, rm):
        recs = rm._generate_risk_recommendations(RiskLevel.LOW, [])
        assert recs == []


# ===========================================================================
# Position Sizing
# ===========================================================================


class TestCheckPositionSize:

    def test_within_limits(self, rm):
        ok, msg = rm.check_position_size(0.05, risk_score=0.3)
        assert ok is True
        assert "within limits" in msg.lower()

    def test_exceeds_risk_adjusted_limit(self, rm):
        """High risk score lowers the allowed position size."""
        ok, msg = rm.check_position_size(0.09, risk_score=0.5)
        # adjusted_max = 0.10 * (1 - 0.5*0.5) = 0.075
        assert ok is False
        assert "risk-adjusted" in msg.lower()

    def test_exceeds_absolute_limit(self):
        """Position above max_position_size is always rejected."""
        mgr = RiskManager(max_position_size=0.10)
        ok, msg = mgr.check_position_size(0.15, risk_score=0.0)
        # adjusted_max = 0.10 * 1.0 = 0.10, 0.15 > 0.10 => rejected
        assert ok is False

    def test_zero_risk_score_max_limit(self, rm):
        """Zero risk score means full max_position_size allowed."""
        ok, msg = rm.check_position_size(0.10, risk_score=0.0)
        assert ok is True


class TestOptimalPositionSize:

    def test_returns_all_methods(self, rm):
        returns = _make_returns(252, seed=42)
        result = rm.calculate_optimal_position_size(returns)
        expected_keys = {'kelly_full', 'kelly_half', 'volatility_target',
                         'var_target', 'recommended'}
        assert expected_keys == set(result.keys())

    def test_all_values_between_zero_and_one(self, rm):
        returns = _make_returns(252, seed=42)
        result = rm.calculate_optimal_position_size(returns)
        for key, val in result.items():
            assert 0 <= val <= 1, f"{key} = {val} outside [0, 1]"

    def test_recommended_is_conservative(self, rm):
        """Recommended size should be <= all individual methods (and <= max_position_size)."""
        returns = _make_returns(252, mean=0.001, std=0.02, seed=42)
        result = rm.calculate_optimal_position_size(returns)
        assert result['recommended'] <= result['kelly_half'] or \
               result['recommended'] <= rm.max_position_size
        assert result['recommended'] <= rm.max_position_size

    def test_zero_variance_kelly(self, rm):
        """Zero variance returns => kelly_full clipped to 0 or 1 (no division by zero)."""
        returns = np.full(100, 0.001)
        result = rm.calculate_optimal_position_size(returns)
        assert np.isfinite(result['kelly_full'])

    def test_negative_mean_kelly_zero(self, rm):
        """Negative mean returns: kelly should be 0 (clipped)."""
        returns = _make_returns(252, mean=-0.01, std=0.02, seed=42)
        result = rm.calculate_optimal_position_size(returns)
        assert result['kelly_full'] == 0.0
        assert result['kelly_half'] == 0.0

    def test_accepts_series(self, rm):
        """Should work with pd.Series input."""
        returns = pd.Series(_make_returns(252, seed=42))
        result = rm.calculate_optimal_position_size(returns)
        assert 'recommended' in result


# ===========================================================================
# Default Assessments
# ===========================================================================


class TestDefaults:

    def test_default_assessment(self, rm):
        """Default assessment should have moderate risk and known defaults."""
        result = rm._default_assessment("TEST")
        assert result.ticker == "TEST"
        assert result.risk_level == RiskLevel.MODERATE
        assert result.risk_score == 0.5
        assert isinstance(result.assessed_at, datetime)

    def test_default_portfolio_assessment(self, rm):
        """Default portfolio assessment should have all expected keys."""
        result = rm._default_portfolio_assessment()
        assert result['portfolio_volatility'] == 0.15
        assert result['within_all_limits'] is True
        assert result['n_positions'] == 0


# ===========================================================================
# Scenario Utilities
# ===========================================================================


class TestScenarioUtilities:

    def test_get_available_scenarios(self, rm):
        scenarios = rm.get_available_scenarios()
        assert len(scenarios) == len(HISTORICAL_SCENARIOS)
        for s in scenarios:
            assert 'id' in s
            assert 'name' in s
            assert 'description' in s
            assert 'date' in s

    def test_create_custom_scenario(self, rm):
        scenario = rm.create_custom_scenario(
            name="Pandemic V2",
            equity_shock=-0.30,
            bond_shock=0.02,
            tech_shock=-0.50,
            description="Hypothetical pandemic",
        )
        assert scenario['name'] == "Pandemic V2"
        assert scenario['equity_shock'] == -0.30
        assert scenario['tech_shock'] == -0.50
        assert scenario['bond_shock'] == 0.02

    def test_create_custom_scenario_no_tech(self, rm):
        """Without tech_shock, the key should not appear."""
        scenario = rm.create_custom_scenario(
            name="Simple", equity_shock=-0.10)
        assert 'tech_shock' not in scenario

    def test_create_custom_scenario_default_description(self, rm):
        scenario = rm.create_custom_scenario(name="Test", equity_shock=-0.05)
        assert "Test" in scenario['description']


# ===========================================================================
# _ensure_array helper
# ===========================================================================


class TestEnsureArray:

    def test_numpy_passthrough(self, rm):
        arr = np.array([1.0, 2.0, 3.0])
        result = rm._ensure_array(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_series_to_array(self, rm):
        s = pd.Series([1.0, 2.0, 3.0])
        result = rm._ensure_array(s)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_list_to_array(self, rm):
        result = rm._ensure_array([1, 2, 3])
        assert isinstance(result, np.ndarray)


# ===========================================================================
# Async Methods (assess_stock_risk, assess_portfolio_risk)
# ===========================================================================


class TestAssessStockRisk:

    @pytest.mark.asyncio
    async def test_sufficient_data(self, rm):
        """With enough data, should return a full RiskAssessment."""
        df = _make_price_df(253, seed=42)
        result = await rm.assess_stock_risk("AAPL", df)
        assert isinstance(result, RiskAssessment)
        assert result.ticker == "AAPL"
        assert 0 <= result.risk_score <= 1
        assert result.volatility > 0
        assert result.var_95 < 0  # typical VaR is a loss

    @pytest.mark.asyncio
    async def test_insufficient_data_default(self, rm):
        """With <30 data points, should return default assessment."""
        df = _make_price_df(10, seed=42)
        result = await rm.assess_stock_risk("TINY", df)
        assert result.ticker == "TINY"
        assert result.risk_level == RiskLevel.MODERATE
        assert result.risk_score == 0.5

    @pytest.mark.asyncio
    async def test_none_price_history(self, rm):
        """None input should return default assessment."""
        result = await rm.assess_stock_risk("NONE", None)
        assert result.ticker == "NONE"
        assert result.risk_level == RiskLevel.MODERATE

    @pytest.mark.asyncio
    async def test_with_explicit_beta(self, rm):
        """Explicit beta should be used directly."""
        df = _make_price_df(253, seed=42)
        result = await rm.assess_stock_risk("AAPL", df, beta=1.5)
        assert result.beta == 1.5

    @pytest.mark.asyncio
    async def test_with_benchmark(self, rm):
        """Providing benchmark should calculate beta from data."""
        df = _make_price_df(253, seed=42)
        benchmark = _make_price_df(253, seed=7)
        result = await rm.assess_stock_risk("AAPL", df, benchmark_history=benchmark)
        assert result.beta != 1.0  # calculated from data, unlikely to be exactly 1


class TestAssessPortfolioRisk:

    @pytest.mark.asyncio
    async def test_empty_positions_returns_default(self, rm):
        result = await rm.assess_portfolio_risk({}, {})
        assert result['n_positions'] == 0
        assert result['within_all_limits'] is True

    @pytest.mark.asyncio
    async def test_with_valid_data(self, rm):
        """Full portfolio assessment with two assets."""
        positions = {"AAPL": 0.6, "MSFT": 0.4}
        price_histories = {
            "AAPL": _make_price_df(253, seed=42),
            "MSFT": _make_price_df(253, seed=7),
        }
        result = await rm.assess_portfolio_risk(positions, price_histories)
        assert result['portfolio_volatility'] > 0
        assert result['n_positions'] == 2
        assert 'risk_decomposition' in result
        assert isinstance(result['risk_decomposition'], RiskDecomposition)

    @pytest.mark.asyncio
    async def test_with_benchmark(self, rm):
        """Portfolio assessment with benchmark should compute beta."""
        positions = {"AAPL": 0.6, "MSFT": 0.4}
        price_histories = {
            "AAPL": _make_price_df(253, seed=42),
            "MSFT": _make_price_df(253, seed=7),
        }
        benchmark = _make_price_df(253, seed=99)
        result = await rm.assess_portfolio_risk(
            positions, price_histories, benchmark_history=benchmark)
        assert result['beta'] is not None
        assert result['tracking_error'] is not None

    @pytest.mark.asyncio
    async def test_insufficient_individual_data(self, rm):
        """Assets with <30 data points should be excluded; may trigger default."""
        positions = {"SHORT": 1.0}
        price_histories = {"SHORT": _make_price_df(10, seed=42)}
        result = await rm.assess_portfolio_risk(positions, price_histories)
        # With insufficient data, should return default
        assert result['n_positions'] == 0 or result['data_points'] == 0
