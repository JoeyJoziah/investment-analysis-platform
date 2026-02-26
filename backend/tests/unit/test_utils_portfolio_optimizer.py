"""
Unit tests for backend/utils/portfolio_optimizer.py

Covers:
- PortfolioOptimizer initialization and configuration
- Portfolio return and volatility calculation (_portfolio_return, _portfolio_volatility)
- Sharpe ratio, variance, Sortino ratio helpers
- Constraint and bounds building (_build_constraints, _build_bounds)
- Initial weight generation
- Optimization methods: max_sharpe, min_variance, risk_parity
- Fallback optimization
- Efficient frontier generation
- Minimum variance and max Sharpe portfolio shortcuts
- Target return optimization
- Rebalancing calculation
- Tracking error constrained optimization
- Risk contribution and diversification ratio
- Portfolio metrics (VaR, CVaR, max drawdown, HHI)
- Edge cases: single asset, empty portfolio, extreme correlations

All tests are pure math -- no database, no network, no mocking of external services.
"""

import numpy as np
import pytest

from backend.utils.portfolio_optimizer import (
    OptimizationResult,
    PortfolioMetrics,
    PortfolioOptimizer,
)


# ---------------------------------------------------------------------------
# Helpers -- reproducible test data
# ---------------------------------------------------------------------------

def _simple_cov(n: int = 3, base_var: float = 0.04, corr: float = 0.3) -> np.ndarray:
    """Build a positive-definite covariance matrix with uniform correlation."""
    std = np.sqrt(base_var)
    cov = np.full((n, n), corr * base_var)
    np.fill_diagonal(cov, base_var)
    return cov


def _diagonal_cov(variances: list) -> np.ndarray:
    """Build a diagonal (uncorrelated) covariance matrix from per-asset variances."""
    return np.diag(variances)


def _make_returns_history(n_days: int = 252, n_assets: int = 3, seed: int = 42) -> np.ndarray:
    """Create reproducible daily returns matrix (n_days x n_assets)."""
    rng = np.random.RandomState(seed)
    return rng.normal(loc=0.0003, scale=0.015, size=(n_days, n_assets))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def optimizer():
    """Fresh PortfolioOptimizer with default settings."""
    return PortfolioOptimizer(risk_free_rate=0.045)


@pytest.fixture
def three_asset_data():
    """Three-asset test data: returns and covariance matrix."""
    expected_returns = np.array([0.10, 0.08, 0.06])
    cov_matrix = _simple_cov(n=3, base_var=0.04, corr=0.3)
    return expected_returns, cov_matrix


# ===========================================================================
# Dataclass sanity checks
# ===========================================================================


class TestDataclasses:

    def test_optimization_result_fields(self):
        """OptimizationResult should store all declared fields."""
        result = OptimizationResult(
            weights=np.array([0.5, 0.5]),
            expected_return=0.08,
            expected_volatility=0.15,
            sharpe_ratio=0.23,
            sortino_ratio=0.30,
            constraints_satisfied=True,
            optimization_method="max_sharpe",
            iterations=42,
            converged=True,
        )
        assert result.converged is True
        assert result.optimization_method == "max_sharpe"
        assert result.metadata == {}

    def test_portfolio_metrics_optional_fields(self):
        """PortfolioMetrics beta and tracking_error default to None."""
        pm = PortfolioMetrics(
            expected_return=0.10, volatility=0.15, sharpe_ratio=0.37,
            sortino_ratio=0.50, max_drawdown=-0.12, var_95=-0.02,
            cvar_95=-0.03, n_positions=5, max_position=0.25, hhi=0.22,
        )
        assert pm.beta is None
        assert pm.tracking_error is None


# ===========================================================================
# Initialization
# ===========================================================================


class TestInit:

    def test_default_values(self, optimizer):
        """Verify default constructor parameters."""
        assert optimizer.risk_free_rate == 0.045
        assert optimizer.default_method == "max_sharpe"
        assert optimizer.max_iterations == 1000
        assert optimizer.tolerance == 1e-10

    def test_custom_values(self):
        """Custom parameters should override defaults."""
        opt = PortfolioOptimizer(
            risk_free_rate=0.02,
            default_method="min_variance",
            max_iterations=500,
            tolerance=1e-8,
        )
        assert opt.risk_free_rate == 0.02
        assert opt.default_method == "min_variance"
        assert opt.max_iterations == 500
        assert opt.tolerance == 1e-8


# ===========================================================================
# Low-level math helpers
# ===========================================================================


class TestPortfolioReturn:

    def test_equal_weights(self, optimizer):
        """Equal weights produce the average expected return."""
        returns = np.array([0.10, 0.08, 0.06])
        weights = np.array([1/3, 1/3, 1/3])
        result = optimizer._portfolio_return(weights, returns)
        assert result == pytest.approx(0.08, abs=1e-10)

    def test_concentrated_in_one_asset(self, optimizer):
        """100% allocation in one asset returns that asset's expected return."""
        returns = np.array([0.10, 0.08, 0.06])
        weights = np.array([0.0, 1.0, 0.0])
        assert optimizer._portfolio_return(weights, returns) == pytest.approx(0.08)

    def test_zero_weights_zero_return(self, optimizer):
        """All-zero weights produce zero return."""
        returns = np.array([0.10, 0.08, 0.06])
        weights = np.zeros(3)
        assert optimizer._portfolio_return(weights, returns) == pytest.approx(0.0)


class TestPortfolioVolatility:

    def test_single_asset(self, optimizer):
        """Single-asset portfolio vol should equal that asset's std dev."""
        cov = np.array([[0.04]])
        weights = np.array([1.0])
        vol = optimizer._portfolio_volatility(weights, cov)
        assert vol == pytest.approx(0.20, abs=1e-10)

    def test_equal_weights_uncorrelated(self, optimizer):
        """Equal weights with uncorrelated assets -> diversification benefit."""
        cov = _diagonal_cov([0.04, 0.04, 0.04])
        weights = np.array([1/3, 1/3, 1/3])
        vol = optimizer._portfolio_volatility(weights, cov)
        # Vol = sqrt(1/3 * 0.04) = 0.2/sqrt(3)
        expected = 0.20 / np.sqrt(3)
        assert vol == pytest.approx(expected, abs=1e-10)

    def test_perfectly_correlated(self, optimizer):
        """Perfectly correlated assets provide no diversification benefit."""
        # Build a perfect-correlation cov matrix: std=0.2 for both
        std = 0.20
        cov = np.array([[std**2, std**2], [std**2, std**2]])
        weights = np.array([0.5, 0.5])
        vol = optimizer._portfolio_volatility(weights, cov)
        assert vol == pytest.approx(std, abs=1e-8)


class TestNegativeSharpeRatio:

    def test_positive_sharpe_returns_negative(self, optimizer):
        """When return > rf, negative Sharpe is negative."""
        returns = np.array([0.10, 0.08])
        cov = _diagonal_cov([0.04, 0.04])
        weights = np.array([0.5, 0.5])
        neg_sharpe = optimizer._negative_sharpe_ratio(weights, returns, cov)
        assert neg_sharpe < 0

    def test_near_zero_vol_returns_large_value(self, optimizer):
        """Near-zero volatility should return a large penalty value."""
        returns = np.array([0.10])
        cov = np.array([[1e-22]])  # vol = 1e-11 < 1e-10 threshold
        weights = np.array([1.0])
        neg_sharpe = optimizer._negative_sharpe_ratio(weights, returns, cov)
        assert neg_sharpe == pytest.approx(1e10)


class TestPortfolioVariance:

    def test_matches_vol_squared(self, optimizer):
        """Variance should equal volatility squared."""
        cov = _simple_cov(n=2, base_var=0.04, corr=0.3)
        weights = np.array([0.6, 0.4])
        var = optimizer._portfolio_variance(weights, cov)
        vol = optimizer._portfolio_volatility(weights, cov)
        assert var == pytest.approx(vol ** 2, abs=1e-12)


# ===========================================================================
# Constraint and bounds building
# ===========================================================================


class TestBuildBounds:

    def test_default_bounds(self, optimizer):
        """Default bounds are (0.0, 1.0) for each asset (no short selling)."""
        bounds = optimizer._build_bounds(3)
        assert bounds == [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

    def test_max_position_constraint(self, optimizer):
        """max_position should cap the upper bound."""
        bounds = optimizer._build_bounds(2, {"max_position": 0.4})
        assert bounds == [(0.0, 0.4), (0.0, 0.4)]

    def test_min_position_constraint(self, optimizer):
        """min_position should raise the lower bound."""
        bounds = optimizer._build_bounds(2, {"min_position": 0.05, "max_position": 0.6})
        assert bounds == [(0.05, 0.6), (0.05, 0.6)]

    def test_allow_short_selling(self, optimizer):
        """allow_short flips the lower bound to negative max_position."""
        bounds = optimizer._build_bounds(2, {"allow_short": True, "max_position": 0.5})
        assert bounds == [(-0.5, 0.5), (-0.5, 0.5)]

    def test_per_asset_bounds_override(self, optimizer):
        """asset_bounds list should override uniform bounds."""
        custom = [(0.0, 0.3), (0.1, 0.5)]
        bounds = optimizer._build_bounds(2, {"asset_bounds": custom})
        assert bounds == custom

    def test_per_asset_bounds_wrong_length_ignored(self, optimizer):
        """asset_bounds with wrong length should fall back to uniform bounds."""
        custom = [(0.0, 0.3)]  # length 1, but n_assets is 3
        bounds = optimizer._build_bounds(3, {"asset_bounds": custom})
        assert len(bounds) == 3
        # Should be uniform defaults, not the custom list
        assert bounds == [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]


class TestBuildConstraints:

    def test_weights_sum_to_one_constraint(self, optimizer):
        """There should always be a weights-sum-to-1 equality constraint."""
        constraints = optimizer._build_constraints(3)
        assert len(constraints) >= 1
        # Test the sum-to-1 constraint
        w = np.array([0.4, 0.3, 0.3])
        assert constraints[0]["fun"](w) == pytest.approx(0.0)

    def test_target_return_constraint(self, optimizer):
        """Target return constraint should bind the portfolio return."""
        er = np.array([0.10, 0.08, 0.06])
        constraints = optimizer._build_constraints(
            3, expected_returns=er, target_return=0.08,
        )
        # Second constraint is the target return
        assert len(constraints) == 2
        w_exact = np.array([0.0, 1.0, 0.0])  # 100% in 8% asset
        assert constraints[1]["fun"](w_exact) == pytest.approx(0.0)

    def test_max_volatility_constraint(self, optimizer):
        """Max volatility constraint should be satisfied when vol is low enough."""
        cov = _diagonal_cov([0.04, 0.04])
        constraints = optimizer._build_constraints(
            2, constraints={"max_volatility": 0.25}, cov_matrix=cov,
        )
        # Should have sum-to-1 + max_vol
        assert len(constraints) == 2
        w_equal = np.array([0.5, 0.5])
        # vol = sqrt(0.5^2 * 0.04 * 2) = sqrt(0.02) ~ 0.1414 < 0.25 -> positive
        assert constraints[1]["fun"](w_equal) > 0

    def test_sector_exposure_limits(self, optimizer):
        """Sector limit constraint should restrict allocation per sector."""
        constraints = optimizer._build_constraints(
            4,
            constraints={
                "sector_limits": {"tech": 0.5, "finance": 0.3},
                "sector_mapping": ["tech", "tech", "finance", "finance"],
            },
        )
        # sum-to-1 + 2 sector constraints
        assert len(constraints) == 3


class TestGetInitialWeights:

    def test_equal_weights_sum_to_one(self, optimizer):
        """Initial weights should sum to 1.0."""
        w = optimizer._get_initial_weights(5)
        assert w.sum() == pytest.approx(1.0)
        assert len(w) == 5

    def test_respects_max_position(self, optimizer):
        """When max_position < equal weight, clamp and renormalize."""
        w = optimizer._get_initial_weights(2, {"max_position": 0.3})
        # Each weight starts at 0.3 (< 0.5 equal weight), then normalized back to sum=1
        assert w.sum() == pytest.approx(1.0)


# ===========================================================================
# Optimization methods (async)
# ===========================================================================


class TestOptimizeMaxSharpe:

    @pytest.mark.asyncio
    async def test_weights_sum_to_one(self, optimizer, three_asset_data):
        """Optimized weights must sum to 1."""
        er, cov = three_asset_data
        weights = await optimizer.optimize(er, cov, method="max_sharpe")
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.asyncio
    async def test_no_negative_weights(self, optimizer, three_asset_data):
        """Without short-selling, all weights must be >= 0."""
        er, cov = three_asset_data
        weights = await optimizer.optimize(er, cov, method="max_sharpe")
        assert np.all(weights >= -1e-8)

    @pytest.mark.asyncio
    async def test_higher_return_asset_gets_more_weight(self, optimizer):
        """With identical risk, the higher-return asset should get more weight."""
        er = np.array([0.12, 0.06])
        cov = _diagonal_cov([0.04, 0.04])
        weights = await optimizer.optimize(er, cov, method="max_sharpe")
        assert weights[0] > weights[1]

    @pytest.mark.asyncio
    async def test_max_position_constraint_respected(self, optimizer):
        """max_position constraint should cap individual weights."""
        er = np.array([0.15, 0.05, 0.05])
        cov = _diagonal_cov([0.04, 0.04, 0.04])
        constraints = {"max_position": 0.5}
        weights = await optimizer.optimize(er, cov, constraints=constraints, method="max_sharpe")
        assert np.all(weights <= 0.5 + 1e-6)


class TestOptimizeMinVariance:

    @pytest.mark.asyncio
    async def test_weights_sum_to_one(self, optimizer, three_asset_data):
        """Min variance weights must sum to 1."""
        _, cov = three_asset_data
        er = np.array([0.10, 0.08, 0.06])
        weights = await optimizer.optimize(er, cov, method="min_variance")
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.asyncio
    async def test_lower_vol_asset_preferred(self, optimizer):
        """Asset with lower variance should get higher weight."""
        er = np.array([0.08, 0.08])
        cov = _diagonal_cov([0.01, 0.09])  # Asset 0 is much less volatile
        weights = await optimizer.optimize(er, cov, method="min_variance")
        assert weights[0] > weights[1]

    @pytest.mark.asyncio
    async def test_min_variance_less_volatile_than_equal_weight(self, optimizer):
        """Min variance portfolio should have <= vol of equal-weight portfolio."""
        er = np.array([0.10, 0.08, 0.06])
        cov = _simple_cov(n=3, base_var=0.04, corr=0.3)
        weights = await optimizer.optimize(er, cov, method="min_variance")
        min_var_vol = optimizer._portfolio_volatility(weights, cov)

        equal_weights = np.ones(3) / 3
        equal_vol = optimizer._portfolio_volatility(equal_weights, cov)

        assert min_var_vol <= equal_vol + 1e-6


class TestOptimizeRiskParity:

    @pytest.mark.asyncio
    async def test_weights_sum_to_one(self, optimizer, three_asset_data):
        """Risk parity weights must sum to 1."""
        er, cov = three_asset_data
        weights = await optimizer.optimize(er, cov, method="risk_parity")
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.asyncio
    async def test_equal_variance_gives_equal_weights(self, optimizer):
        """Uncorrelated assets with equal variance should get equal risk parity weights."""
        er = np.array([0.10, 0.08, 0.06])
        cov = _diagonal_cov([0.04, 0.04, 0.04])
        weights = await optimizer.optimize(er, cov, method="risk_parity")
        # All variances the same, uncorrelated -> equal weights
        expected = np.array([1/3, 1/3, 1/3])
        np.testing.assert_allclose(weights, expected, atol=0.02)

    @pytest.mark.asyncio
    async def test_risk_parity_no_negative_weights(self, optimizer, three_asset_data):
        """Risk parity should produce non-negative weights."""
        er, cov = three_asset_data
        weights = await optimizer.optimize(er, cov, method="risk_parity")
        assert np.all(weights >= -1e-8)


# ===========================================================================
# Empty and single-asset edge cases
# ===========================================================================


class TestEdgeCases:

    @pytest.mark.asyncio
    async def test_empty_portfolio(self, optimizer):
        """Empty expected_returns should return empty array."""
        er = np.array([])
        cov = np.array([]).reshape(0, 0)
        weights = await optimizer.optimize(er, cov)
        assert len(weights) == 0

    @pytest.mark.asyncio
    async def test_single_asset(self, optimizer):
        """Single asset must receive 100% weight."""
        er = np.array([0.10])
        cov = np.array([[0.04]])
        weights = await optimizer.optimize(er, cov, method="max_sharpe")
        assert weights[0] == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.asyncio
    async def test_cov_matrix_shape_mismatch_raises(self, optimizer):
        """Covariance matrix with wrong dimensions should raise ValueError."""
        er = np.array([0.10, 0.08])
        cov = np.array([[0.04]])  # 1x1, but 2 assets
        with pytest.raises(ValueError, match="doesn't match"):
            await optimizer.optimize(er, cov)

    @pytest.mark.asyncio
    async def test_unknown_method_falls_back_to_max_sharpe(self, optimizer, three_asset_data):
        """Unknown method name should default to max_sharpe."""
        er, cov = three_asset_data
        weights = await optimizer.optimize(er, cov, method="unknown_method")
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)


# ===========================================================================
# Fallback optimization
# ===========================================================================


class TestFallbackOptimization:

    def test_inverse_vol_weights_sum_to_one(self, optimizer, three_asset_data):
        """Fallback weights should sum to 1."""
        er, cov = three_asset_data
        weights = optimizer._fallback_optimization(er, cov)
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)

    def test_lower_vol_gets_more_weight(self, optimizer):
        """Asset with lower variance gets higher fallback weight (inverse vol)."""
        er = np.array([0.10, 0.08])
        cov = _diagonal_cov([0.01, 0.09])
        weights = optimizer._fallback_optimization(er, cov)
        assert weights[0] > weights[1]

    def test_negative_return_excluded(self, optimizer):
        """Assets with negative expected return are excluded from fallback."""
        er = np.array([0.10, -0.05])
        cov = _diagonal_cov([0.04, 0.04])
        weights = optimizer._fallback_optimization(er, cov)
        assert weights[0] == pytest.approx(1.0, abs=1e-6)
        assert weights[1] == pytest.approx(0.0, abs=1e-6)

    def test_all_negative_returns_equal_weight(self, optimizer):
        """If all returns are negative, fall back to equal weighting."""
        er = np.array([-0.05, -0.10])
        cov = _diagonal_cov([0.04, 0.04])
        weights = optimizer._fallback_optimization(er, cov)
        np.testing.assert_allclose(weights, [0.5, 0.5], atol=1e-6)

    def test_max_position_applied(self, optimizer):
        """Fallback should respect max_position constraint."""
        er = np.array([0.10, 0.08, 0.06])
        cov = _diagonal_cov([0.04, 0.04, 0.04])
        constraints = {"max_position": 0.4}
        weights = optimizer._fallback_optimization(er, cov, constraints)
        # After clipping and renormalization, all should be within bounds
        assert np.all(weights <= 0.4 + 1e-6)


# ===========================================================================
# Efficient frontier
# ===========================================================================


class TestEfficientFrontier:

    @pytest.mark.asyncio
    async def test_frontier_has_points(self, optimizer, three_asset_data):
        """Efficient frontier should produce at least some valid points."""
        er, cov = three_asset_data
        frontier = await optimizer.get_efficient_frontier(er, cov, n_points=10)
        assert len(frontier) > 0

    @pytest.mark.asyncio
    async def test_frontier_sorted_by_volatility(self, optimizer, three_asset_data):
        """Frontier should be sorted by increasing volatility."""
        er, cov = three_asset_data
        frontier = await optimizer.get_efficient_frontier(er, cov, n_points=20)
        vols = [p[0] for p in frontier]
        for i in range(len(vols) - 1):
            assert vols[i] <= vols[i + 1] + 1e-10

    @pytest.mark.asyncio
    async def test_frontier_weights_sum_to_one(self, optimizer, three_asset_data):
        """Each frontier portfolio's weights should sum to 1."""
        er, cov = three_asset_data
        frontier = await optimizer.get_efficient_frontier(er, cov, n_points=10)
        for vol, ret, weights in frontier:
            assert weights.sum() == pytest.approx(1.0, abs=0.05)

    @pytest.mark.asyncio
    async def test_frontier_return_increases_with_vol(self, optimizer, three_asset_data):
        """Higher-vol portfolios on the frontier should have higher returns (roughly)."""
        er, cov = three_asset_data
        frontier = await optimizer.get_efficient_frontier(er, cov, n_points=20)
        if len(frontier) >= 2:
            # Compare first and last
            assert frontier[-1][1] >= frontier[0][1] - 0.01


# ===========================================================================
# Minimum variance and max Sharpe shortcuts
# ===========================================================================


class TestMinimumVariancePortfolio:

    @pytest.mark.asyncio
    async def test_weights_sum_to_one(self, optimizer, three_asset_data):
        _, cov = three_asset_data
        weights = await optimizer.get_minimum_variance_portfolio(cov)
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.asyncio
    async def test_no_negative_weights(self, optimizer, three_asset_data):
        _, cov = three_asset_data
        weights = await optimizer.get_minimum_variance_portfolio(cov)
        assert np.all(weights >= -1e-8)


class TestMaxSharpePortfolio:

    @pytest.mark.asyncio
    async def test_delegates_to_optimize(self, optimizer, three_asset_data):
        """get_max_sharpe_portfolio should produce valid weights."""
        er, cov = three_asset_data
        weights = await optimizer.get_max_sharpe_portfolio(er, cov)
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)
        assert np.all(weights >= -1e-8)


# ===========================================================================
# Target return optimization
# ===========================================================================


class TestOptimizeForTargetReturn:

    @pytest.mark.asyncio
    async def test_achieves_target_return(self, optimizer, three_asset_data):
        """Optimized portfolio should achieve approximately the target return."""
        er, cov = three_asset_data
        target = 0.08
        weights = await optimizer.optimize_for_target_return(er, cov, target_return=target)
        actual_return = optimizer._portfolio_return(weights, er)
        assert actual_return == pytest.approx(target, abs=0.01)

    @pytest.mark.asyncio
    async def test_weights_sum_to_one(self, optimizer, three_asset_data):
        er, cov = three_asset_data
        weights = await optimizer.optimize_for_target_return(er, cov, target_return=0.07)
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)


# ===========================================================================
# Sortino ratio
# ===========================================================================


class TestSortinoRatio:

    def test_with_returns_history(self, optimizer):
        """Sortino should compute from historical downside returns."""
        weights = np.array([0.5, 0.5])
        er = np.array([0.10, 0.08])
        history = _make_returns_history(n_days=252, n_assets=2, seed=42)
        sortino = optimizer._calculate_sortino_ratio(weights, er, returns_history=history)
        assert isinstance(sortino, float)

    def test_no_history_returns_default(self, optimizer):
        """Without history, Sortino uses a default downside_std of 0.01."""
        weights = np.array([0.6, 0.4])
        er = np.array([0.10, 0.08])
        sortino = optimizer._calculate_sortino_ratio(weights, er)
        expected = (optimizer._portfolio_return(weights, er) - 0.045) / 0.01
        assert sortino == pytest.approx(expected, abs=1e-6)

    def test_with_downside_returns(self, optimizer):
        """Sortino should use provided downside returns directly."""
        weights = np.array([0.5, 0.5])
        er = np.array([0.10, 0.06])
        # Make downside returns all negative
        downside = np.array([[-0.02, -0.03], [-0.01, -0.04], [-0.03, -0.02]])
        sortino = optimizer._calculate_sortino_ratio(
            weights, er, downside_returns=downside,
        )
        assert isinstance(sortino, float)


# ===========================================================================
# Portfolio metrics
# ===========================================================================


class TestCalculatePortfolioMetrics:

    @pytest.mark.asyncio
    async def test_basic_metrics(self, optimizer, three_asset_data):
        """Should return all expected metric keys."""
        er, cov = three_asset_data
        weights = np.array([0.4, 0.3, 0.3])
        metrics = await optimizer.calculate_portfolio_metrics(weights, er, cov)
        expected_keys = {
            "expected_return", "volatility", "sharpe_ratio", "sortino_ratio",
            "var_95", "cvar_95", "max_drawdown", "n_positions",
            "max_position", "hhi",
        }
        assert expected_keys == set(metrics.keys())

    @pytest.mark.asyncio
    async def test_parametric_var(self, optimizer, three_asset_data):
        """Without history, VaR95 uses parametric approximation."""
        er, cov = three_asset_data
        weights = np.array([0.4, 0.3, 0.3])
        metrics = await optimizer.calculate_portfolio_metrics(weights, er, cov)
        port_ret = optimizer._portfolio_return(weights, er)
        port_vol = optimizer._portfolio_volatility(weights, cov)
        expected_var = port_ret - 1.645 * port_vol
        assert metrics["var_95"] == pytest.approx(expected_var, abs=1e-6)

    @pytest.mark.asyncio
    async def test_hhi_concentrated(self, optimizer, three_asset_data):
        """100% in one asset should have HHI = 1.0."""
        er, cov = three_asset_data
        weights = np.array([1.0, 0.0, 0.0])
        metrics = await optimizer.calculate_portfolio_metrics(weights, er, cov)
        assert metrics["hhi"] == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.asyncio
    async def test_hhi_equal_weights(self, optimizer, three_asset_data):
        """Equal weights should have HHI = 1/n."""
        er, cov = three_asset_data
        weights = np.ones(3) / 3
        metrics = await optimizer.calculate_portfolio_metrics(weights, er, cov)
        assert metrics["hhi"] == pytest.approx(1/3, abs=1e-6)

    @pytest.mark.asyncio
    async def test_n_positions(self, optimizer, three_asset_data):
        """n_positions counts assets with weight > 1%."""
        er, cov = three_asset_data
        weights = np.array([0.5, 0.5, 0.005])  # third below threshold
        metrics = await optimizer.calculate_portfolio_metrics(weights, er, cov)
        assert metrics["n_positions"] == 2

    @pytest.mark.asyncio
    async def test_with_history_max_drawdown(self, optimizer, three_asset_data):
        """When returns_history is provided, max_drawdown should be non-zero."""
        er, cov = three_asset_data
        weights = np.array([0.4, 0.3, 0.3])
        history = _make_returns_history(n_days=252, n_assets=3, seed=42)
        metrics = await optimizer.calculate_portfolio_metrics(weights, er, cov, returns_history=history)
        assert metrics["max_drawdown"] <= 0  # Drawdown is non-positive


# ===========================================================================
# Rebalancing
# ===========================================================================


class TestRebalancePortfolio:

    @pytest.mark.asyncio
    async def test_no_trade_when_already_balanced(self, optimizer):
        """Identical current and target weights produce no trades."""
        current = np.array([0.4, 0.3, 0.3])
        target = np.array([0.4, 0.3, 0.3])
        result = await optimizer.rebalance_portfolio(current, target)
        assert result["n_trades"] == 0
        assert result["turnover"] == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_full_rebalance(self, optimizer):
        """Complete position flip should generate trades."""
        current = np.array([1.0, 0.0, 0.0])
        target = np.array([0.0, 0.5, 0.5])
        result = await optimizer.rebalance_portfolio(current, target)
        assert result["n_trades"] > 0
        assert result["total_sell_value"] > 0
        assert result["total_buy_value"] > 0

    @pytest.mark.asyncio
    async def test_transaction_costs_calculated(self, optimizer):
        """Transaction costs should be proportional to trade volume."""
        current = np.array([0.6, 0.4])
        target = np.array([0.4, 0.6])
        result = await optimizer.rebalance_portfolio(current, target, transaction_cost=0.01)
        # Total trade = |0.2| + |0.2| = 0.4, cost = 0.4 * 0.01 = 0.004
        assert result["transaction_costs"] == pytest.approx(0.004, abs=1e-6)

    @pytest.mark.asyncio
    async def test_min_trade_size_filters_small_trades(self, optimizer):
        """Trades smaller than min_trade_size should be zeroed out."""
        current = np.array([0.33, 0.34, 0.33])
        target = np.array([0.333, 0.334, 0.333])
        result = await optimizer.rebalance_portfolio(
            current, target, min_trade_size=0.01,
        )
        assert result["n_trades"] == 0

    @pytest.mark.asyncio
    async def test_turnover_is_one_way(self, optimizer):
        """Turnover should be half the two-way turnover."""
        current = np.array([0.6, 0.4])
        target = np.array([0.4, 0.6])
        result = await optimizer.rebalance_portfolio(current, target)
        # Buy = 0.2, Sell = 0.2, Turnover = 0.2
        assert result["turnover"] == pytest.approx(0.2, abs=1e-6)

    @pytest.mark.asyncio
    async def test_net_trade_near_zero(self, optimizer):
        """Net trade should be approximately zero for a fully-invested rebalance."""
        current = np.array([0.5, 0.3, 0.2])
        target = np.array([0.3, 0.4, 0.3])
        result = await optimizer.rebalance_portfolio(current, target)
        assert result["net_trade"] == pytest.approx(0.0, abs=1e-6)


# ===========================================================================
# Tracking error constrained optimization
# ===========================================================================


class TestOptimizeWithTrackingError:

    @pytest.mark.asyncio
    async def test_weights_sum_to_one(self, optimizer, three_asset_data):
        er, cov = three_asset_data
        benchmark = np.array([0.4, 0.3, 0.3])
        weights = await optimizer.optimize_with_tracking_error(
            er, cov, benchmark, max_tracking_error=0.05,
        )
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.asyncio
    async def test_tracking_error_within_limit(self, optimizer, three_asset_data):
        """Resulting portfolio should have tracking error within the specified limit."""
        er, cov = three_asset_data
        benchmark = np.array([0.4, 0.3, 0.3])
        max_te = 0.05
        weights = await optimizer.optimize_with_tracking_error(
            er, cov, benchmark, max_tracking_error=max_te,
        )
        diff = weights - benchmark
        te = np.sqrt(np.dot(diff.T, np.dot(cov, diff)))
        assert te <= max_te + 0.01  # Small tolerance for optimizer precision

    @pytest.mark.asyncio
    async def test_zero_tracking_error_returns_benchmark(self, optimizer, three_asset_data):
        """With max_tracking_error=0, should stay at or near benchmark."""
        er, cov = three_asset_data
        benchmark = np.array([0.4, 0.3, 0.3])
        weights = await optimizer.optimize_with_tracking_error(
            er, cov, benchmark, max_tracking_error=0.0,
        )
        np.testing.assert_allclose(weights, benchmark, atol=0.02)


# ===========================================================================
# Risk contribution and diversification
# ===========================================================================


class TestRiskContribution:

    def test_risk_contributions_sum_to_portfolio_variance(self, optimizer, three_asset_data):
        """Risk contributions should sum to portfolio variance."""
        _, cov = three_asset_data
        weights = np.array([0.4, 0.3, 0.3])
        rc = optimizer.calculate_risk_contribution(weights, cov)
        port_var = optimizer._portfolio_variance(weights, cov)
        assert rc.sum() == pytest.approx(port_var, abs=1e-8)

    def test_zero_weights_zero_contribution(self, optimizer, three_asset_data):
        """Zero-weight asset should have zero risk contribution."""
        _, cov = three_asset_data
        weights = np.array([0.5, 0.5, 0.0])
        rc = optimizer.calculate_risk_contribution(weights, cov)
        assert rc[2] == pytest.approx(0.0, abs=1e-10)

    def test_zero_portfolio_variance(self, optimizer):
        """When portfolio variance is near zero, return zeros."""
        cov = np.zeros((2, 2))
        weights = np.array([0.5, 0.5])
        rc = optimizer.calculate_risk_contribution(weights, cov)
        np.testing.assert_allclose(rc, [0.0, 0.0])


class TestDiversificationRatio:

    def test_single_asset_ratio_is_one(self, optimizer):
        """A single asset has diversification ratio of 1.0."""
        cov = np.array([[0.04]])
        weights = np.array([1.0])
        dr = optimizer.calculate_diversification_ratio(weights, cov)
        assert dr == pytest.approx(1.0, abs=1e-6)

    def test_uncorrelated_assets_high_ratio(self, optimizer):
        """Uncorrelated assets should yield diversification ratio > 1."""
        cov = _diagonal_cov([0.04, 0.04, 0.04])
        weights = np.ones(3) / 3
        dr = optimizer.calculate_diversification_ratio(weights, cov)
        # DR = weighted_avg_vol / portfolio_vol
        # weighted_avg = 0.2, port_vol = 0.2/sqrt(3) -> DR = sqrt(3) ~ 1.73
        assert dr == pytest.approx(np.sqrt(3), abs=0.01)

    def test_perfectly_correlated_ratio_is_one(self, optimizer):
        """Perfectly correlated assets give diversification ratio of 1.0."""
        std = 0.20
        cov = np.array([[std**2, std**2], [std**2, std**2]])
        weights = np.array([0.5, 0.5])
        dr = optimizer.calculate_diversification_ratio(weights, cov)
        assert dr == pytest.approx(1.0, abs=0.01)

    def test_zero_vol_returns_one(self, optimizer):
        """When portfolio vol is near zero, return 1.0."""
        cov = np.zeros((2, 2))
        weights = np.array([0.5, 0.5])
        dr = optimizer.calculate_diversification_ratio(weights, cov)
        assert dr == pytest.approx(1.0)
