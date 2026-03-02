"""
Unit tests for financial analytics models.

Covers:
- DCFModel (backend/analytics/fundamental/valuation/dcf_model.py)
- VaRCalculator (backend/analytics/risk/calculators/var_calculator.py)
- BlackLittermanOptimizer (backend/analytics/portfolio/black_litterman.py)

All tests are pure math -- no mocking, no I/O, no database.
"""

import numpy as np
import pandas as pd
import pytest

from backend.analytics.fundamental.valuation.dcf_model import DCFModel, DCFResult
from backend.analytics.risk.calculators.var_calculator import (
    VaRCalculator,
    VaRMethod,
)
from backend.analytics.portfolio.black_litterman import (
    BlackLittermanOptimizer,
    BlackLittermanResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_returns_df(n_days: int = 252, seed: int = 42) -> pd.DataFrame:
    """Create a reproducible DataFrame of daily returns for 3 assets."""
    rng = np.random.RandomState(seed)
    data = rng.normal(loc=0.0004, scale=0.015, size=(n_days, 3))
    return pd.DataFrame(data, columns=["AAPL", "MSFT", "GOOG"])


def _market_caps() -> dict:
    return {"AAPL": 3_000_000, "MSFT": 2_500_000, "GOOG": 1_500_000}


# ===========================================================================
# DCFModel Tests
# ===========================================================================


class TestDCFModelBasic:
    """Basic intrinsic value calculations."""

    def test_known_inputs_produce_known_output(self):
        """A hand-verified DCF with explicit growth rates."""
        model = DCFModel(projection_years=3, terminal_growth_rate=0.02)
        result = model.calculate_intrinsic_value(
            free_cash_flow=100.0,
            growth_rates=[0.10, 0.08, 0.05],
            discount_rate=0.10,
            shares_outstanding=10.0,
            current_price=50.0,
        )
        # Projected FCFs: 110, 118.8, 124.74
        assert isinstance(result, DCFResult)
        assert result.intrinsic_value > 0
        assert len(result.free_cash_flows) == 3
        assert result.free_cash_flows[0] == pytest.approx(110.0, rel=1e-6)
        assert result.free_cash_flows[1] == pytest.approx(118.8, rel=1e-6)
        assert result.free_cash_flows[2] == pytest.approx(124.74, rel=1e-6)

    def test_zero_fcf_returns_zero_intrinsic_value(self):
        """Zero FCF means the company generates no cash -- value should be 0."""
        model = DCFModel()
        result = model.calculate_intrinsic_value(
            free_cash_flow=0.0,
            discount_rate=0.10,
            shares_outstanding=1.0,
        )
        assert result.intrinsic_value == pytest.approx(0.0, abs=1e-10)
        assert all(fcf == pytest.approx(0.0, abs=1e-10) for fcf in result.free_cash_flows)
        assert result.terminal_value == pytest.approx(0.0, abs=1e-10)

    def test_negative_fcf_handled_gracefully(self):
        """Companies with negative FCF should return a negative intrinsic value, not crash."""
        model = DCFModel(projection_years=3, terminal_growth_rate=0.02)
        result = model.calculate_intrinsic_value(
            free_cash_flow=-50.0,
            growth_rates=[0.10, 0.10, 0.10],
            discount_rate=0.10,
            shares_outstanding=1.0,
        )
        assert result.intrinsic_value < 0
        assert all(fcf < 0 for fcf in result.free_cash_flows)

    def test_result_dataclass_fields(self):
        """DCFResult must expose all documented fields."""
        model = DCFModel()
        result = model.calculate_intrinsic_value(free_cash_flow=100.0)
        for attr in ("intrinsic_value", "current_price", "upside_potential",
                      "margin_of_safety", "free_cash_flows", "terminal_value",
                      "discount_rate"):
            assert hasattr(result, attr), f"Missing attribute: {attr}"


class TestDCFModelUpside:
    """Upside potential and margin of safety."""

    def test_upside_potential_formula(self):
        """upside = (intrinsic - current) / current."""
        model = DCFModel(projection_years=3, terminal_growth_rate=0.02)
        result = model.calculate_intrinsic_value(
            free_cash_flow=100.0,
            growth_rates=[0.10, 0.08, 0.05],
            discount_rate=0.10,
            shares_outstanding=1.0,
            current_price=500.0,
        )
        expected_upside = (result.intrinsic_value - 500.0) / 500.0
        assert result.upside_potential == pytest.approx(expected_upside, rel=1e-9)

    def test_upside_potential_zero_price(self):
        """When current_price is 0, upside should default to 0.0."""
        model = DCFModel()
        result = model.calculate_intrinsic_value(
            free_cash_flow=100.0, current_price=0.0
        )
        assert result.upside_potential == 0.0
        assert result.margin_of_safety == 0.0

    def test_margin_of_safety_non_negative(self):
        """Margin of safety must never be negative (clamped to 0)."""
        model = DCFModel(projection_years=2, terminal_growth_rate=0.02)
        # Use a very high current price so intrinsic < current
        result = model.calculate_intrinsic_value(
            free_cash_flow=10.0,
            discount_rate=0.10,
            shares_outstanding=1.0,
            current_price=999_999.0,
        )
        assert result.margin_of_safety >= 0.0


class TestDCFModelSensitivity:
    """Discount rate, projection years, and sensitivity analysis."""

    def test_higher_discount_rate_reduces_intrinsic_value(self):
        """Increasing the discount rate should lower the present value."""
        model = DCFModel(projection_years=5, terminal_growth_rate=0.025)
        low_dr = model.calculate_intrinsic_value(
            free_cash_flow=100.0, discount_rate=0.08, shares_outstanding=1.0
        )
        high_dr = model.calculate_intrinsic_value(
            free_cash_flow=100.0, discount_rate=0.15, shares_outstanding=1.0
        )
        assert high_dr.intrinsic_value < low_dr.intrinsic_value

    def test_different_projection_years_produce_different_results(self):
        """More projection years should change the intrinsic value."""
        short = DCFModel(projection_years=3, terminal_growth_rate=0.025)
        long = DCFModel(projection_years=10, terminal_growth_rate=0.025)
        r_short = short.calculate_intrinsic_value(
            free_cash_flow=100.0, discount_rate=0.10
        )
        r_long = long.calculate_intrinsic_value(
            free_cash_flow=100.0, discount_rate=0.10
        )
        assert r_short.intrinsic_value != pytest.approx(r_long.intrinsic_value, rel=1e-3)

    def test_sensitivity_analysis_shape(self):
        """sensitivity_analysis returns a grid matching input dimensions."""
        model = DCFModel(projection_years=5)
        drs = [0.08, 0.10, 0.12]
        grs = [0.02, 0.03]
        results = model.sensitivity_analysis(
            free_cash_flow=100.0,
            discount_rates=drs,
            growth_rates=grs,
            shares_outstanding=1.0,
        )
        assert len(results["values"]) == len(drs)
        assert all(len(row) == len(grs) for row in results["values"])

    def test_sensitivity_analysis_monotonicity(self):
        """For a given growth rate, higher discount rate -> lower value."""
        model = DCFModel(projection_years=5)
        drs = [0.08, 0.10, 0.12, 0.15]
        grs = [0.025]
        results = model.sensitivity_analysis(
            free_cash_flow=100.0,
            discount_rates=drs,
            growth_rates=grs,
        )
        values = [row[0] for row in results["values"]]
        for i in range(len(values) - 1):
            assert values[i] > values[i + 1]


class TestDCFModelWACC:
    """WACC calculation."""

    def test_wacc_simple(self):
        """WACC = E/V * Re + D/V * Rd * (1-T)."""
        model = DCFModel()
        wacc = model.calculate_wacc(
            cost_of_equity=0.10,
            cost_of_debt=0.05,
            tax_rate=0.25,
            equity_weight=0.60,
            debt_weight=0.40,
        )
        expected = 0.60 * 0.10 + 0.40 * 0.05 * (1 - 0.25)
        assert wacc == pytest.approx(expected, rel=1e-9)

    def test_wacc_all_equity(self):
        """100% equity -> WACC equals cost of equity."""
        model = DCFModel()
        wacc = model.calculate_wacc(
            cost_of_equity=0.12,
            cost_of_debt=0.05,
            tax_rate=0.30,
            equity_weight=1.0,
            debt_weight=0.0,
        )
        assert wacc == pytest.approx(0.12, rel=1e-9)


# ===========================================================================
# VaR Calculator Tests
# ===========================================================================


class TestVaRHistorical:
    """Historical VaR."""

    def test_known_returns_array(self):
        """VaR at 95% on a simple sorted array should equal the 5th percentile."""
        returns = np.array(sorted(range(-50, 50)))  # -50 to 49
        calc = VaRCalculator(confidence_level=0.95)
        var = calc.calculate_historical_var(returns)
        expected = np.percentile(returns, 5)
        assert var == pytest.approx(expected, rel=1e-9)

    def test_empty_array_returns_zero(self):
        """Edge case: no data means no risk estimate."""
        calc = VaRCalculator()
        var = calc.calculate_historical_var(np.array([]))
        assert var == 0.0

    def test_single_element(self):
        """A single observation should return that value at any confidence."""
        calc = VaRCalculator(confidence_level=0.95)
        var = calc.calculate_historical_var(np.array([-0.03]))
        assert var == pytest.approx(-0.03, rel=1e-6)

    def test_confidence_99_more_extreme_than_95(self):
        """99% VaR should be a more extreme (lower) loss than 95% VaR."""
        rng = np.random.RandomState(7)
        returns = rng.normal(-0.001, 0.02, 1000)
        var_95 = VaRCalculator(confidence_level=0.95).calculate_historical_var(returns)
        var_99 = VaRCalculator(confidence_level=0.99).calculate_historical_var(returns)
        assert var_99 < var_95  # more extreme tail

    def test_all_positive_returns_var_positive(self):
        """If all returns are positive, VaR should be positive (no loss)."""
        returns = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        calc = VaRCalculator(confidence_level=0.95)
        var = calc.calculate_historical_var(returns)
        assert var > 0


class TestVaRParametric:
    """Parametric VaR (normal distribution)."""

    def test_parametric_var_with_normal_data(self):
        """Parametric VaR should be close to theoretical for normal data."""
        from scipy import stats

        rng = np.random.RandomState(42)
        returns = rng.normal(loc=0.0, scale=0.02, size=10_000)
        calc = VaRCalculator(confidence_level=0.95)
        var = calc.calculate_parametric_var(returns)
        # Theoretical: mean + z * std ~ 0 + (-1.645)*0.02 ~ -0.0329
        theoretical = 0.0 + stats.norm.ppf(0.05) * 0.02
        assert var == pytest.approx(theoretical, abs=0.003)

    def test_parametric_var_empty(self):
        """Empty returns should return 0."""
        calc = VaRCalculator()
        var = calc.calculate_parametric_var(np.array([]))
        assert var == 0.0


class TestVaRDispatch:
    """VaR method dispatch via calculate_var."""

    def test_dispatch_historical(self):
        """calculate_var with HISTORICAL matches calculate_historical_var."""
        returns = np.array([-0.05, -0.03, -0.01, 0.01, 0.03])
        calc = VaRCalculator(confidence_level=0.95)
        assert calc.calculate_var(returns, VaRMethod.HISTORICAL) == pytest.approx(
            calc.calculate_historical_var(returns)
        )

    def test_dispatch_parametric(self):
        """calculate_var with PARAMETRIC matches calculate_parametric_var."""
        rng = np.random.RandomState(99)
        returns = rng.normal(0.0, 0.02, 500)
        calc = VaRCalculator(confidence_level=0.95)
        assert calc.calculate_var(returns, VaRMethod.PARAMETRIC) == pytest.approx(
            calc.calculate_parametric_var(returns)
        )

    def test_dispatch_monte_carlo_produces_reasonable_var(self):
        """Monte Carlo VaR should be negative (a loss) for volatile returns."""
        rng = np.random.RandomState(42)
        returns = rng.normal(-0.001, 0.02, 500)
        calc = VaRCalculator(confidence_level=0.95)
        mc_var = calc.calculate_var(returns, VaRMethod.MONTE_CARLO, seed=42)
        # MC VaR should be negative (indicating a loss at 95% confidence)
        assert mc_var < 0
        # Should be in the same order of magnitude as historical VaR
        hist_var = calc.calculate_historical_var(returns)
        assert abs(mc_var) < abs(hist_var) * 5  # within 5x


class TestCVaR:
    """Conditional VaR (Expected Shortfall)."""

    def test_cvar_more_extreme_than_var(self):
        """CVaR (average tail loss) should be <= VaR."""
        rng = np.random.RandomState(42)
        returns = rng.normal(-0.001, 0.02, 1000)
        calc = VaRCalculator(confidence_level=0.95)
        var = calc.calculate_historical_var(returns)
        cvar = calc.calculate_cvar(returns)
        assert cvar <= var


# ===========================================================================
# Black-Litterman Tests
# ===========================================================================


class TestBlackLittermanEquilibrium:
    """Equilibrium returns and no-view baseline."""

    def test_equilibrium_returns_shape(self):
        """Implied returns should have one entry per asset."""
        optimizer = BlackLittermanOptimizer()
        returns_df = _make_returns_df()
        implied = optimizer.calculate_implied_returns(returns_df, _market_caps())
        assert set(implied.keys()) == {"AAPL", "MSFT", "GOOG"}

    def test_equilibrium_returns_positive(self):
        """With positive-drift data and positive market caps, implied returns should be positive."""
        optimizer = BlackLittermanOptimizer()
        returns_df = _make_returns_df()
        implied = optimizer.calculate_implied_returns(returns_df, _market_caps())
        for asset, ret in implied.items():
            assert ret > 0, f"{asset} implied return should be positive"

    def test_empty_views_uses_pure_equilibrium(self):
        """No views -> weights equal market-cap weights."""
        optimizer = BlackLittermanOptimizer()
        returns_df = _make_returns_df()
        caps = _market_caps()
        result = optimizer.optimize(returns_df, caps, views=None)

        total = sum(caps.values())
        for asset in ["AAPL", "MSFT", "GOOG"]:
            expected_weight = caps[asset] / total
            assert result.weights[asset] == pytest.approx(expected_weight, rel=1e-9)

    def test_empty_views_list_same_as_none(self):
        """An empty list of views should behave the same as None."""
        optimizer = BlackLittermanOptimizer()
        returns_df = _make_returns_df()
        result = optimizer.optimize(returns_df, _market_caps(), views=[])
        total = sum(_market_caps().values())
        for asset in ["AAPL", "MSFT", "GOOG"]:
            assert result.weights[asset] == pytest.approx(
                _market_caps()[asset] / total, rel=1e-9
            )


class TestBlackLittermanOptimize:
    """Optimization with views."""

    def test_weights_sum_to_one(self):
        """Absolute weights must sum to 1.0 after normalization."""
        optimizer = BlackLittermanOptimizer()
        returns_df = _make_returns_df()
        views = [{"asset": "AAPL", "expected_return": 0.15}]
        result = optimizer.optimize(
            returns_df, _market_caps(), views=views, view_confidences=[0.8]
        )
        weight_sum = sum(abs(w) for w in result.weights.values())
        assert weight_sum == pytest.approx(1.0, abs=1e-9)

    def test_result_dataclass_fields(self):
        """BlackLittermanResult must expose all documented fields."""
        optimizer = BlackLittermanOptimizer()
        returns_df = _make_returns_df()
        result = optimizer.optimize(returns_df, _market_caps())
        for attr in ("weights", "expected_returns", "posterior_covariance",
                      "implied_equilibrium_returns"):
            assert hasattr(result, attr), f"Missing attribute: {attr}"

    def test_bullish_view_increases_weight(self):
        """A strong bullish view on GOOG should tilt weight toward GOOG."""
        optimizer = BlackLittermanOptimizer()
        returns_df = _make_returns_df()
        caps = _market_caps()

        baseline = optimizer.optimize(returns_df, caps, views=None)
        bullish = optimizer.optimize(
            returns_df,
            caps,
            views=[{"asset": "GOOG", "expected_return": 0.30}],
            view_confidences=[0.95],
        )
        assert bullish.weights["GOOG"] > baseline.weights["GOOG"]

    def test_posterior_covariance_shape(self):
        """Posterior covariance matrix should be n_assets x n_assets."""
        optimizer = BlackLittermanOptimizer()
        returns_df = _make_returns_df()
        result = optimizer.optimize(returns_df, _market_caps())
        n = len(returns_df.columns)
        assert result.posterior_covariance.shape == (n, n)

    def test_no_view_weights_are_non_negative(self):
        """Without views, market-cap weights should all be non-negative (long-only)."""
        optimizer = BlackLittermanOptimizer()
        returns_df = _make_returns_df()
        result = optimizer.optimize(returns_df, _market_caps(), views=None)
        for asset, w in result.weights.items():
            assert w >= 0.0, f"{asset} weight should be non-negative, got {w}"
