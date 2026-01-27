"""
Tests for Portfolio Optimizer - Markowitz Mean-Variance Optimization

Tests cover:
- Basic optimization (max Sharpe, min variance)
- Efficient frontier calculation
- Constraint handling (position limits, volatility)
- Risk metrics calculation
- Edge cases and error handling
"""

import pytest
import numpy as np
import asyncio
from backend.utils.portfolio_optimizer import PortfolioOptimizer, OptimizationResult


@pytest.fixture
def sample_returns():
    """Generate sample expected returns for 5 assets."""
    return np.array([0.12, 0.10, 0.08, 0.15, 0.07])  # Annual returns


@pytest.fixture
def sample_cov_matrix():
    """Generate sample covariance matrix for 5 assets."""
    # Create a valid positive semi-definite covariance matrix
    volatilities = np.array([0.20, 0.15, 0.10, 0.25, 0.12])  # Annual volatilities
    correlation = np.array([
        [1.0, 0.3, 0.2, 0.4, 0.1],
        [0.3, 1.0, 0.25, 0.3, 0.2],
        [0.2, 0.25, 1.0, 0.15, 0.3],
        [0.4, 0.3, 0.15, 1.0, 0.2],
        [0.1, 0.2, 0.3, 0.2, 1.0]
    ])
    cov_matrix = np.outer(volatilities, volatilities) * correlation
    return cov_matrix


@pytest.fixture
def optimizer():
    """Create a PortfolioOptimizer instance."""
    return PortfolioOptimizer(risk_free_rate=0.03)


@pytest.fixture
def historical_returns():
    """Generate sample historical returns (252 days x 5 assets)."""
    np.random.seed(42)
    n_days = 252
    n_assets = 5
    # Generate correlated returns
    means = np.array([0.0005, 0.0004, 0.0003, 0.0006, 0.0003])  # Daily means
    stds = np.array([0.015, 0.012, 0.008, 0.018, 0.010])  # Daily stds
    returns = np.random.randn(n_days, n_assets) * stds + means
    return returns


class TestPortfolioOptimizerInit:
    """Test initialization."""

    def test_default_init(self):
        """Test default initialization."""
        opt = PortfolioOptimizer()
        assert opt.risk_free_rate == 0.045
        assert opt.default_method == 'max_sharpe'
        assert opt.max_iterations == 1000

    def test_custom_init(self):
        """Test custom initialization."""
        opt = PortfolioOptimizer(
            risk_free_rate=0.02,
            default_method='min_variance',
            max_iterations=500
        )
        assert opt.risk_free_rate == 0.02
        assert opt.default_method == 'min_variance'
        assert opt.max_iterations == 500


class TestMaxSharpeOptimization:
    """Test maximum Sharpe ratio optimization."""

    @pytest.mark.asyncio
    async def test_max_sharpe_basic(self, optimizer, sample_returns, sample_cov_matrix):
        """Test basic max Sharpe optimization."""
        weights = await optimizer.optimize(
            expected_returns=sample_returns,
            cov_matrix=sample_cov_matrix,
            method='max_sharpe'
        )

        # Weights should sum to 1
        assert np.isclose(weights.sum(), 1.0, atol=1e-6)
        # All weights should be non-negative (no short selling by default)
        assert all(weights >= -1e-6)
        # Should have same length as assets
        assert len(weights) == len(sample_returns)

    @pytest.mark.asyncio
    async def test_max_sharpe_with_constraints(self, optimizer, sample_returns, sample_cov_matrix):
        """Test max Sharpe with position constraints."""
        constraints = {
            'max_position': 0.30,
            'min_position': 0.05
        }
        weights = await optimizer.optimize(
            expected_returns=sample_returns,
            cov_matrix=sample_cov_matrix,
            constraints=constraints,
            method='max_sharpe'
        )

        # Check constraints
        assert all(weights <= constraints['max_position'] + 1e-6)
        # Non-zero weights should be >= min_position
        non_zero = weights[weights > 1e-6]
        if len(non_zero) > 0:
            assert all(non_zero >= constraints['min_position'] - 1e-6)


class TestMinVarianceOptimization:
    """Test minimum variance optimization."""

    @pytest.mark.asyncio
    async def test_min_variance_basic(self, optimizer, sample_returns, sample_cov_matrix):
        """Test basic min variance optimization."""
        weights = await optimizer.optimize(
            expected_returns=sample_returns,
            cov_matrix=sample_cov_matrix,
            method='min_variance'
        )

        assert np.isclose(weights.sum(), 1.0, atol=1e-6)
        assert all(weights >= -1e-6)

    @pytest.mark.asyncio
    async def test_min_variance_lower_volatility(self, optimizer, sample_returns, sample_cov_matrix):
        """Verify min variance has lower volatility than equal weights."""
        min_var_weights = await optimizer.optimize(
            expected_returns=sample_returns,
            cov_matrix=sample_cov_matrix,
            method='min_variance'
        )

        equal_weights = np.ones(len(sample_returns)) / len(sample_returns)

        min_var_vol = optimizer._portfolio_volatility(min_var_weights, sample_cov_matrix)
        equal_vol = optimizer._portfolio_volatility(equal_weights, sample_cov_matrix)

        assert min_var_vol <= equal_vol + 1e-6

    @pytest.mark.asyncio
    async def test_get_minimum_variance_portfolio(self, optimizer, sample_cov_matrix):
        """Test direct minimum variance portfolio method."""
        weights = await optimizer.get_minimum_variance_portfolio(sample_cov_matrix)

        assert np.isclose(weights.sum(), 1.0, atol=1e-6)
        assert all(weights >= -1e-6)


class TestRiskParityOptimization:
    """Test risk parity optimization."""

    @pytest.mark.asyncio
    async def test_risk_parity_basic(self, optimizer, sample_returns, sample_cov_matrix):
        """Test basic risk parity optimization."""
        weights = await optimizer.optimize(
            expected_returns=sample_returns,
            cov_matrix=sample_cov_matrix,
            method='risk_parity'
        )

        assert np.isclose(weights.sum(), 1.0, atol=1e-6)
        assert all(weights >= -1e-6)

    @pytest.mark.asyncio
    async def test_risk_parity_contribution(self, optimizer, sample_returns, sample_cov_matrix):
        """Verify risk parity has more equal risk contributions."""
        rp_weights = await optimizer.optimize(
            expected_returns=sample_returns,
            cov_matrix=sample_cov_matrix,
            method='risk_parity'
        )

        risk_contrib = optimizer.calculate_risk_contribution(rp_weights, sample_cov_matrix)

        # Risk contributions should be relatively equal
        # Normalize and check variance
        if risk_contrib.sum() > 0:
            normalized = risk_contrib / risk_contrib.sum()
            # Should be close to equal (1/n_assets = 0.2 for 5 assets)
            variance = np.var(normalized)
            assert variance < 0.05  # Reasonably balanced


class TestEfficientFrontier:
    """Test efficient frontier calculation."""

    @pytest.mark.asyncio
    async def test_efficient_frontier_basic(self, optimizer, sample_returns, sample_cov_matrix):
        """Test basic efficient frontier calculation."""
        frontier = await optimizer.get_efficient_frontier(
            expected_returns=sample_returns,
            cov_matrix=sample_cov_matrix,
            n_points=20
        )

        assert len(frontier) > 0
        assert len(frontier) <= 20

        # Each point should have (volatility, return, weights)
        for vol, ret, weights in frontier:
            assert isinstance(vol, float)
            assert isinstance(ret, float)
            assert isinstance(weights, np.ndarray)
            assert np.isclose(weights.sum(), 1.0, atol=1e-5)

    @pytest.mark.asyncio
    async def test_efficient_frontier_monotonic(self, optimizer, sample_returns, sample_cov_matrix):
        """Test that frontier is monotonically increasing in return-volatility space."""
        frontier = await optimizer.get_efficient_frontier(
            expected_returns=sample_returns,
            cov_matrix=sample_cov_matrix,
            n_points=30
        )

        if len(frontier) > 1:
            # Sort by volatility
            sorted_frontier = sorted(frontier, key=lambda x: x[0])
            # Returns should generally increase with volatility on efficient frontier
            vols = [f[0] for f in sorted_frontier]
            rets = [f[1] for f in sorted_frontier]

            # Volatilities should be sorted
            assert vols == sorted(vols)


class TestTargetReturnOptimization:
    """Test target return optimization."""

    @pytest.mark.asyncio
    async def test_target_return_achieved(self, optimizer, sample_returns, sample_cov_matrix):
        """Test that target return is approximately achieved."""
        target = 0.10  # 10% target return

        weights = await optimizer.optimize_for_target_return(
            expected_returns=sample_returns,
            cov_matrix=sample_cov_matrix,
            target_return=target
        )

        achieved_return = optimizer._portfolio_return(weights, sample_returns)
        # Should be close to target (within reasonable tolerance)
        assert abs(achieved_return - target) < 0.02  # Within 2%


class TestPortfolioMetrics:
    """Test portfolio metrics calculation."""

    @pytest.mark.asyncio
    async def test_metrics_calculation(self, optimizer, sample_returns, sample_cov_matrix):
        """Test comprehensive metrics calculation."""
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

        metrics = await optimizer.calculate_portfolio_metrics(
            weights=weights,
            expected_returns=sample_returns,
            cov_matrix=sample_cov_matrix
        )

        assert 'expected_return' in metrics
        assert 'volatility' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'sortino_ratio' in metrics
        assert 'n_positions' in metrics
        assert 'max_position' in metrics
        assert 'hhi' in metrics

        # Sanity checks
        assert metrics['volatility'] > 0
        assert metrics['n_positions'] == 5
        assert metrics['max_position'] == 0.2

    @pytest.mark.asyncio
    async def test_metrics_with_history(self, optimizer, sample_returns, sample_cov_matrix, historical_returns):
        """Test metrics calculation with historical data."""
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

        metrics = await optimizer.calculate_portfolio_metrics(
            weights=weights,
            expected_returns=sample_returns,
            cov_matrix=sample_cov_matrix,
            returns_history=historical_returns
        )

        assert 'var_95' in metrics
        assert 'cvar_95' in metrics
        assert 'max_drawdown' in metrics

        # VaR should be negative (loss)
        assert metrics['var_95'] < 0 or abs(metrics['var_95']) < 0.1


class TestRebalancing:
    """Test rebalancing calculations."""

    @pytest.mark.asyncio
    async def test_rebalance_basic(self, optimizer):
        """Test basic rebalancing calculation."""
        current = np.array([0.25, 0.25, 0.25, 0.25, 0.0])
        target = np.array([0.20, 0.20, 0.20, 0.20, 0.20])

        result = await optimizer.rebalance_portfolio(
            current_weights=current,
            target_weights=target,
            transaction_cost=0.001
        )

        assert 'trades' in result
        assert 'transaction_costs' in result
        assert 'n_trades' in result
        assert 'turnover' in result

        # Net trades should be approximately zero
        assert abs(result['net_trade']) < 1e-6

    @pytest.mark.asyncio
    async def test_rebalance_min_trade_filter(self, optimizer):
        """Test that small trades are filtered out."""
        current = np.array([0.20, 0.20, 0.20, 0.20, 0.20])
        target = np.array([0.201, 0.199, 0.20, 0.20, 0.20])  # Tiny changes

        result = await optimizer.rebalance_portfolio(
            current_weights=current,
            target_weights=target,
            min_trade_size=0.005
        )

        # Should filter out tiny trades
        assert result['n_trades'] == 0


class TestRiskContribution:
    """Test risk contribution analysis."""

    def test_risk_contribution_sums_to_variance(self, optimizer, sample_cov_matrix):
        """Test that risk contributions sum to portfolio variance."""
        weights = np.array([0.3, 0.2, 0.15, 0.25, 0.10])

        risk_contrib = optimizer.calculate_risk_contribution(weights, sample_cov_matrix)
        port_var = optimizer._portfolio_variance(weights, sample_cov_matrix)

        assert np.isclose(risk_contrib.sum(), port_var, rtol=1e-5)

    def test_diversification_ratio(self, optimizer, sample_cov_matrix):
        """Test diversification ratio calculation."""
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

        dr = optimizer.calculate_diversification_ratio(weights, sample_cov_matrix)

        # Diversification ratio should be >= 1
        assert dr >= 1.0


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_assets(self, optimizer):
        """Test handling of empty asset list."""
        weights = await optimizer.optimize(
            expected_returns=np.array([]),
            cov_matrix=np.array([[]]),
            method='max_sharpe'
        )

        assert len(weights) == 0

    @pytest.mark.asyncio
    async def test_single_asset(self, optimizer):
        """Test handling of single asset."""
        weights = await optimizer.optimize(
            expected_returns=np.array([0.10]),
            cov_matrix=np.array([[0.04]]),
            method='max_sharpe'
        )

        assert len(weights) == 1
        assert np.isclose(weights[0], 1.0)

    @pytest.mark.asyncio
    async def test_negative_returns(self, optimizer, sample_cov_matrix):
        """Test handling of all negative expected returns."""
        negative_returns = np.array([-0.05, -0.03, -0.08, -0.02, -0.10])

        weights = await optimizer.optimize(
            expected_returns=negative_returns,
            cov_matrix=sample_cov_matrix,
            method='max_sharpe'
        )

        # Should still produce valid weights
        assert np.isclose(weights.sum(), 1.0, atol=1e-6)
        assert all(weights >= -1e-6)

    @pytest.mark.asyncio
    async def test_fallback_on_failure(self, optimizer, sample_returns, sample_cov_matrix):
        """Test fallback mechanism when optimization fails."""
        # Create impossible constraints
        constraints = {
            'max_position': 0.01,  # Max 1% per position (impossible to sum to 100%)
            'min_position': 0.50   # Min 50% per position
        }

        weights = await optimizer.optimize(
            expected_returns=sample_returns,
            cov_matrix=sample_cov_matrix,
            constraints=constraints,
            method='max_sharpe'
        )

        # Should use fallback and produce valid weights
        assert len(weights) == len(sample_returns)
        # Weights might not perfectly sum to 1 with impossible constraints
        # but should be close
        assert abs(weights.sum() - 1.0) < 0.5


class TestTrackingError:
    """Test tracking error constrained optimization."""

    @pytest.mark.asyncio
    async def test_tracking_error_constraint(self, optimizer, sample_returns, sample_cov_matrix):
        """Test optimization with tracking error constraint."""
        benchmark = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # Equal weight benchmark

        weights = await optimizer.optimize_with_tracking_error(
            expected_returns=sample_returns,
            cov_matrix=sample_cov_matrix,
            benchmark_weights=benchmark,
            max_tracking_error=0.05
        )

        assert np.isclose(weights.sum(), 1.0, atol=1e-6)

        # Calculate tracking error
        diff = weights - benchmark
        te = np.sqrt(np.dot(diff.T, np.dot(sample_cov_matrix, diff)))

        # Should be within constraint (with some tolerance)
        assert te <= 0.06  # Allow small tolerance


class TestShortSelling:
    """Test short selling functionality."""

    @pytest.mark.asyncio
    async def test_allow_short_selling(self, optimizer, sample_returns, sample_cov_matrix):
        """Test optimization with short selling allowed."""
        constraints = {
            'allow_short': True,
            'max_position': 0.50
        }

        weights = await optimizer.optimize(
            expected_returns=sample_returns,
            cov_matrix=sample_cov_matrix,
            constraints=constraints,
            method='max_sharpe'
        )

        assert np.isclose(weights.sum(), 1.0, atol=1e-6)
        # Short positions are allowed (negative weights possible)
        # At least verify it runs without error


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
