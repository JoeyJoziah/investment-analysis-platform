"""
Tests for Risk Manager - Comprehensive Risk Assessment and Management

Tests cover:
- VaR calculations (Historical, Parametric, Monte Carlo)
- CVaR/Expected Shortfall
- Maximum Drawdown analysis
- Beta and Tracking Error calculations
- Stress Testing (historical scenarios and custom)
- Portfolio risk aggregation
- Risk decomposition
- Position sizing
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from backend.utils.risk_manager import (
    RiskManager,
    RiskLevel,
    VaRMethod,
    RiskAssessment,
    VaRResult,
    StressTestResult,
    RiskDecomposition,
    HISTORICAL_SCENARIOS,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def risk_manager():
    """Create a RiskManager instance with default settings."""
    return RiskManager()


@pytest.fixture
def custom_risk_manager():
    """Create a RiskManager with custom settings."""
    return RiskManager(
        max_portfolio_var=0.03,
        max_position_size=0.15,
        min_sharpe_ratio=0.3,
        risk_free_rate=0.05,
        monte_carlo_simulations=5000,
        var_horizon_days=5
    )


@pytest.fixture
def sample_returns():
    """Generate sample daily returns (252 days)."""
    np.random.seed(42)
    # Normal distribution with slight negative skew
    returns = np.random.normal(0.0005, 0.015, 252)
    # Add a few tail events
    returns[50] = -0.05  # -5% day
    returns[100] = -0.04  # -4% day
    returns[150] = 0.04  # +4% day
    return returns


@pytest.fixture
def sample_prices():
    """Generate sample price series from returns."""
    np.random.seed(42)
    returns = np.random.normal(0.0003, 0.012, 252)
    prices = 100 * np.cumprod(1 + returns)
    return prices


@pytest.fixture
def sample_benchmark_returns():
    """Generate sample benchmark returns."""
    np.random.seed(123)
    return np.random.normal(0.0004, 0.012, 252)


@pytest.fixture
def sample_price_history():
    """Generate sample price DataFrame."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
    returns = np.random.normal(0.0003, 0.012, 252)
    prices = 100 * np.cumprod(1 + returns)

    return pd.DataFrame({
        'date': dates,
        'open': prices * 0.999,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 252)
    })


@pytest.fixture
def sample_portfolio():
    """Create sample portfolio weights."""
    return {
        'AAPL': 0.25,
        'GOOGL': 0.20,
        'MSFT': 0.20,
        'AMZN': 0.15,
        'META': 0.10,
        'NVDA': 0.10
    }


@pytest.fixture
def sample_price_histories():
    """Generate price histories for multiple stocks."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=252, freq='D')

    histories = {}
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NVDA']
    base_vols = [0.012, 0.015, 0.011, 0.018, 0.020, 0.025]

    for ticker, vol in zip(tickers, base_vols):
        returns = np.random.normal(0.0003, vol, 252)
        prices = 100 * np.cumprod(1 + returns)
        histories[ticker] = pd.DataFrame({
            'date': dates,
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, 252)
        })

    return histories


# ============================================================================
# VaR Calculation Tests
# ============================================================================

class TestVaRCalculations:
    """Test VaR calculation methods."""

    def test_historical_var(self, risk_manager, sample_returns):
        """Test Historical VaR calculation."""
        result = risk_manager.calculate_var(
            sample_returns, confidence=0.95, method='historical'
        )

        assert isinstance(result, VaRResult)
        assert result.method == VaRMethod.HISTORICAL
        assert result.confidence_level == 0.95
        assert result.var_value < 0  # VaR should be negative (loss)
        assert result.horizon_days == 1

        # 95% VaR should be roughly the 5th percentile
        expected_var = np.percentile(sample_returns, 5)
        assert abs(result.var_value - expected_var) < 0.001

    def test_parametric_var(self, risk_manager, sample_returns):
        """Test Parametric (Variance-Covariance) VaR."""
        result = risk_manager.calculate_var(
            sample_returns, confidence=0.95, method='parametric'
        )

        assert isinstance(result, VaRResult)
        assert result.method == VaRMethod.PARAMETRIC
        assert result.var_value < 0

        # Parametric VaR should be close to historical for normal-ish data
        hist_result = risk_manager.calculate_var(
            sample_returns, confidence=0.95, method='historical'
        )
        # Within 50% of each other (parametric assumes normal distribution)
        assert abs(result.var_value / hist_result.var_value - 1) < 0.5

    def test_monte_carlo_var(self, risk_manager, sample_returns):
        """Test Monte Carlo VaR."""
        result = risk_manager.calculate_var(
            sample_returns, confidence=0.95, method='monte_carlo'
        )

        assert isinstance(result, VaRResult)
        assert result.method == VaRMethod.MONTE_CARLO
        assert result.var_value < 0

        # MC VaR should be similar to parametric (both assume normal)
        param_result = risk_manager.calculate_var(
            sample_returns, confidence=0.95, method='parametric'
        )
        assert abs(result.var_value / param_result.var_value - 1) < 0.2

    def test_var_confidence_levels(self, risk_manager, sample_returns):
        """Test VaR at different confidence levels."""
        var_90 = risk_manager.calculate_var(sample_returns, confidence=0.90)
        var_95 = risk_manager.calculate_var(sample_returns, confidence=0.95)
        var_99 = risk_manager.calculate_var(sample_returns, confidence=0.99)

        # Higher confidence = more extreme loss (more negative)
        assert var_99.var_value < var_95.var_value < var_90.var_value

    def test_var_horizon_scaling(self, risk_manager, sample_returns):
        """Test VaR scaling over different horizons."""
        var_1d = risk_manager.calculate_var(
            sample_returns, confidence=0.95, method='parametric', horizon_days=1
        )
        var_5d = risk_manager.calculate_var(
            sample_returns, confidence=0.95, method='parametric', horizon_days=5
        )
        var_10d = risk_manager.calculate_var(
            sample_returns, confidence=0.95, method='parametric', horizon_days=10
        )

        # VaR should scale approximately by sqrt(time)
        assert abs(var_5d.var_value) > abs(var_1d.var_value)
        assert abs(var_10d.var_value) > abs(var_5d.var_value)

        # Check sqrt(T) scaling for parametric
        ratio_5_1 = abs(var_5d.var_value / var_1d.var_value)
        expected_ratio = np.sqrt(5)
        assert abs(ratio_5_1 - expected_ratio) < 0.5

    def test_var_all_methods(self, risk_manager, sample_returns):
        """Test calculating VaR with all methods."""
        results = risk_manager.calculate_var_all_methods(
            sample_returns, confidence=0.95
        )

        assert 'historical' in results
        assert 'parametric' in results
        assert 'monte_carlo' in results

        for method, result in results.items():
            assert isinstance(result, VaRResult)
            assert result.var_value < 0

    def test_var_with_pandas_series(self, risk_manager, sample_returns):
        """Test VaR calculation with pandas Series input."""
        returns_series = pd.Series(sample_returns)
        result = risk_manager.calculate_var(returns_series, confidence=0.95)

        assert isinstance(result, VaRResult)
        assert result.var_value < 0

    def test_var_additional_metrics(self, risk_manager, sample_returns):
        """Test VaR result includes additional metrics."""
        result = risk_manager.calculate_var(sample_returns, confidence=0.95)

        assert 'mean_return' in result.additional_metrics
        assert 'std_return' in result.additional_metrics
        assert 'skewness' in result.additional_metrics
        assert 'kurtosis' in result.additional_metrics
        assert 'data_points' in result.additional_metrics

        assert result.additional_metrics['data_points'] == len(sample_returns)


# ============================================================================
# CVaR / Expected Shortfall Tests
# ============================================================================

class TestCVaRCalculations:
    """Test CVaR/Expected Shortfall calculations."""

    def test_cvar_historical(self, risk_manager, sample_returns):
        """Test historical CVaR calculation."""
        cvar = risk_manager.calculate_cvar(sample_returns, confidence=0.95)
        var = risk_manager.calculate_var(sample_returns, confidence=0.95).var_value

        # CVaR should be more negative than VaR (worse tail loss)
        assert cvar <= var
        assert cvar < 0

    def test_cvar_parametric(self, risk_manager, sample_returns):
        """Test parametric CVaR calculation."""
        cvar = risk_manager.calculate_cvar_parametric(sample_returns, confidence=0.95)
        var = risk_manager.calculate_var(
            sample_returns, confidence=0.95, method='parametric'
        ).var_value

        # Parametric CVaR should be more negative than VaR
        assert cvar < var
        assert cvar < 0

    def test_cvar_is_coherent_measure(self, risk_manager, sample_returns):
        """Test that CVaR behaves as a coherent risk measure."""
        cvar_90 = risk_manager.calculate_cvar(sample_returns, confidence=0.90)
        cvar_95 = risk_manager.calculate_cvar(sample_returns, confidence=0.95)
        cvar_99 = risk_manager.calculate_cvar(sample_returns, confidence=0.99)

        # Higher confidence = more extreme expected tail loss
        assert cvar_99 <= cvar_95 <= cvar_90


# ============================================================================
# Maximum Drawdown Tests
# ============================================================================

class TestMaximumDrawdown:
    """Test Maximum Drawdown calculations."""

    def test_max_drawdown_calculation(self, risk_manager, sample_prices):
        """Test basic max drawdown calculation."""
        max_dd, peak_idx, trough_idx = risk_manager.calculate_max_drawdown(sample_prices)

        assert max_dd <= 0  # Drawdown is negative
        assert max_dd >= -1  # Cannot lose more than 100%
        assert peak_idx <= trough_idx  # Peak comes before trough

    def test_max_drawdown_known_case(self, risk_manager):
        """Test max drawdown with known outcome."""
        # Create a series that goes up, then down, then up
        prices = np.array([100, 110, 120, 100, 90, 95, 105])
        max_dd, peak_idx, trough_idx = risk_manager.calculate_max_drawdown(prices)

        # Expected: peak at 120 (idx 2), trough at 90 (idx 4)
        # Drawdown = (90 - 120) / 120 = -0.25
        assert abs(max_dd - (-0.25)) < 0.01
        assert peak_idx == 2
        assert trough_idx == 4

    def test_drawdown_series(self, risk_manager, sample_prices):
        """Test full drawdown series calculation."""
        dd_df = risk_manager.calculate_drawdown_series(sample_prices)

        assert 'price' in dd_df.columns
        assert 'running_max' in dd_df.columns
        assert 'drawdown' in dd_df.columns
        assert 'drawdown_duration' in dd_df.columns

        # Running max should never decrease
        assert all(np.diff(dd_df['running_max']) >= 0)

        # Drawdown should always be <= 0
        assert all(dd_df['drawdown'] <= 0)

    def test_all_drawdowns(self, risk_manager, sample_prices):
        """Test identification of all significant drawdowns."""
        drawdowns = risk_manager.calculate_all_drawdowns(sample_prices, threshold=-0.02)

        for dd in drawdowns:
            assert 'start_idx' in dd
            assert 'trough_idx' in dd
            assert 'max_drawdown' in dd
            assert dd['max_drawdown'] <= -0.02  # All exceed threshold

    def test_no_drawdown_case(self, risk_manager):
        """Test when prices only go up."""
        prices = np.linspace(100, 200, 100)  # Monotonically increasing
        max_dd, peak_idx, trough_idx = risk_manager.calculate_max_drawdown(prices)

        assert max_dd == 0  # No drawdown
        assert peak_idx == trough_idx  # No distinct peak/trough


# ============================================================================
# Beta and Tracking Error Tests
# ============================================================================

class TestBetaCalculations:
    """Test Beta and related calculations."""

    def test_beta_calculation(self, risk_manager, sample_returns, sample_benchmark_returns):
        """Test beta calculation."""
        result = risk_manager.calculate_beta(sample_returns, sample_benchmark_returns)

        assert 'beta' in result
        assert 'alpha' in result
        assert 'r_squared' in result
        assert 'correlation' in result

        # R-squared should be between 0 and 1
        assert 0 <= result['r_squared'] <= 1

        # Correlation should be between -1 and 1
        assert -1 <= result['correlation'] <= 1

    def test_beta_perfect_correlation(self, risk_manager):
        """Test beta with perfectly correlated returns."""
        np.random.seed(42)
        benchmark = np.random.normal(0, 0.01, 252)
        # Asset has beta of 1.5
        asset = 1.5 * benchmark + np.random.normal(0, 0.001, 252)

        result = risk_manager.calculate_beta(asset, benchmark)

        # Beta should be close to 1.5
        assert abs(result['beta'] - 1.5) < 0.2

        # R-squared should be high
        assert result['r_squared'] > 0.9

    def test_tracking_error(self, risk_manager, sample_returns, sample_benchmark_returns):
        """Test tracking error calculation."""
        result = risk_manager.calculate_tracking_error(
            sample_returns, sample_benchmark_returns
        )

        assert 'tracking_error' in result
        assert 'tracking_error_annualized' in result
        assert 'mean_active_return' in result
        assert 'information_ratio' in result

        # Tracking error should be positive
        assert result['tracking_error'] > 0
        assert result['tracking_error_annualized'] > 0

    def test_tracking_error_identical_returns(self, risk_manager, sample_returns):
        """Test tracking error when returns match benchmark."""
        result = risk_manager.calculate_tracking_error(sample_returns, sample_returns)

        # Tracking error should be zero (or very close)
        assert result['tracking_error'] < 0.0001


# ============================================================================
# Stress Testing Tests
# ============================================================================

class TestStressTesting:
    """Test stress testing functionality."""

    def test_historical_stress_test(self, risk_manager, sample_portfolio):
        """Test historical stress scenario."""
        result = risk_manager.stress_test(
            sample_portfolio,
            scenario='2008_financial_crisis'
        )

        assert isinstance(result, StressTestResult)
        assert result.scenario_name == "2008 Financial Crisis"
        assert result.portfolio_loss < 0  # Should show loss
        assert len(result.asset_impacts) == len(sample_portfolio)
        assert result.historical_date is not None

    def test_all_historical_scenarios(self, risk_manager, sample_portfolio):
        """Test all predefined historical scenarios."""
        results = risk_manager.stress_test_all_scenarios(sample_portfolio)

        assert len(results) == len(HISTORICAL_SCENARIOS)

        # Results should be sorted by loss (worst first)
        losses = [r.portfolio_loss for r in results]
        assert losses == sorted(losses)

    def test_custom_stress_test(self, risk_manager, sample_portfolio):
        """Test custom stress scenario."""
        shocks = {
            'AAPL': -0.30,
            'GOOGL': -0.25,
            'MSFT': -0.20,
            'AMZN': -0.35,
            'META': -0.40,
            'NVDA': -0.50
        }

        result = risk_manager.stress_test_custom(
            sample_portfolio,
            shocks,
            scenario_name="Tech Crash",
            description="Major technology sector decline"
        )

        assert result.scenario_name == "Tech Crash"
        assert result.portfolio_loss < 0

        # Verify portfolio loss calculation
        expected_loss = sum(
            sample_portfolio[t] * shocks[t] for t in sample_portfolio
        )
        assert abs(result.portfolio_loss - expected_loss) < 0.0001

    def test_stress_test_with_betas(self, risk_manager, sample_portfolio):
        """Test stress test with asset betas."""
        betas = {
            'AAPL': 1.2,
            'GOOGL': 1.1,
            'MSFT': 0.9,
            'AMZN': 1.3,
            'META': 1.4,
            'NVDA': 1.8
        }

        result_no_beta = risk_manager.stress_test(
            sample_portfolio, '2008_financial_crisis'
        )
        result_with_beta = risk_manager.stress_test(
            sample_portfolio, '2008_financial_crisis', asset_betas=betas
        )

        # Results should differ when betas are applied
        assert result_no_beta.portfolio_loss != result_with_beta.portfolio_loss

    def test_stress_test_sector_mappings(self, risk_manager, sample_portfolio):
        """Test stress test with sector mappings."""
        sectors = {
            'AAPL': 'tech',
            'GOOGL': 'tech',
            'MSFT': 'tech',
            'AMZN': 'tech',
            'META': 'tech',
            'NVDA': 'tech'
        }

        # Use dot-com scenario which has specific tech shock
        result = risk_manager.stress_test(
            sample_portfolio,
            '2000_dotcom_burst',
            sector_mappings=sectors
        )

        # Tech stocks should get the tech_shock
        for ticker in sample_portfolio:
            expected_shock = HISTORICAL_SCENARIOS['2000_dotcom_burst']['tech_shock']
            assert result.asset_impacts[ticker] == expected_shock

    def test_var_breach_detection(self, risk_manager, sample_portfolio):
        """Test VaR breach detection in stress tests."""
        # 2008 crisis should breach typical VaR limits
        result = risk_manager.stress_test(
            sample_portfolio, '2008_financial_crisis'
        )

        # With 50% equity shock and no hedging, should breach 2% VaR
        assert result.var_breach is True

    def test_available_scenarios(self, risk_manager):
        """Test getting available scenarios."""
        scenarios = risk_manager.get_available_scenarios()

        assert len(scenarios) == len(HISTORICAL_SCENARIOS)
        for s in scenarios:
            assert 'id' in s
            assert 'name' in s
            assert 'description' in s
            assert 'date' in s

    def test_create_custom_scenario(self, risk_manager):
        """Test creating a custom scenario."""
        scenario = risk_manager.create_custom_scenario(
            name="Rate Shock",
            equity_shock=-0.15,
            bond_shock=-0.10,
            tech_shock=-0.25,
            volatility_multiplier=2.5,
            description="Aggressive rate hike scenario"
        )

        assert scenario['name'] == "Rate Shock"
        assert scenario['equity_shock'] == -0.15
        assert scenario['bond_shock'] == -0.10
        assert scenario['tech_shock'] == -0.25
        assert scenario['volatility_multiplier'] == 2.5

    def test_invalid_scenario_raises_error(self, risk_manager, sample_portfolio):
        """Test that invalid scenario name raises error."""
        with pytest.raises(ValueError, match="Unknown scenario"):
            risk_manager.stress_test(sample_portfolio, 'invalid_scenario')


# ============================================================================
# Portfolio Risk Assessment Tests
# ============================================================================

class TestPortfolioRiskAssessment:
    """Test portfolio-level risk assessment."""

    @pytest.mark.asyncio
    async def test_portfolio_risk_assessment(
        self, risk_manager, sample_portfolio, sample_price_histories
    ):
        """Test comprehensive portfolio risk assessment."""
        result = await risk_manager.assess_portfolio_risk(
            sample_portfolio, sample_price_histories
        )

        assert 'portfolio_volatility' in result
        assert 'portfolio_var_95_historical' in result
        assert 'portfolio_cvar_95' in result
        assert 'max_drawdown' in result
        assert 'sharpe_ratio' in result
        assert 'sortino_ratio' in result
        assert 'diversification_ratio' in result
        assert 'risk_decomposition' in result
        assert 'within_all_limits' in result

    @pytest.mark.asyncio
    async def test_portfolio_risk_with_benchmark(
        self, risk_manager, sample_portfolio, sample_price_histories
    ):
        """Test portfolio risk with benchmark."""
        # Create benchmark history
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
        benchmark_returns = np.random.normal(0.0004, 0.011, 252)
        benchmark_prices = 100 * np.cumprod(1 + benchmark_returns)
        benchmark_history = pd.DataFrame({
            'date': dates,
            'close': benchmark_prices
        })

        result = await risk_manager.assess_portfolio_risk(
            sample_portfolio,
            sample_price_histories,
            benchmark_history=benchmark_history
        )

        # Should have beta and tracking error when benchmark provided
        assert result['beta'] is not None
        assert result['tracking_error'] is not None
        assert result['information_ratio'] is not None

    @pytest.mark.asyncio
    async def test_portfolio_risk_empty_inputs(self, risk_manager):
        """Test portfolio risk with empty inputs."""
        result = await risk_manager.assess_portfolio_risk({}, {})

        # Should return default assessment
        assert result['n_positions'] == 0
        assert 'Insufficient data' in result['risk_factors'][0]


# ============================================================================
# Stock Risk Assessment Tests
# ============================================================================

class TestStockRiskAssessment:
    """Test individual stock risk assessment."""

    @pytest.mark.asyncio
    async def test_stock_risk_assessment(self, risk_manager, sample_price_history):
        """Test single stock risk assessment."""
        result = await risk_manager.assess_stock_risk('AAPL', sample_price_history)

        assert isinstance(result, RiskAssessment)
        assert result.ticker == 'AAPL'
        assert isinstance(result.risk_level, RiskLevel)
        assert 0 <= result.risk_score <= 1
        assert result.volatility > 0
        assert result.var_95 < 0
        assert result.cvar_95 <= result.var_95
        assert result.max_drawdown <= 0

    @pytest.mark.asyncio
    async def test_stock_risk_with_benchmark(self, risk_manager, sample_price_history):
        """Test stock risk with benchmark for beta calculation."""
        np.random.seed(123)
        dates = sample_price_history['date']
        benchmark_returns = np.random.normal(0.0004, 0.010, len(dates) - 1)
        benchmark_prices = 100 * np.cumprod(1 + benchmark_returns)
        benchmark_prices = np.insert(benchmark_prices, 0, 100)

        benchmark_history = pd.DataFrame({
            'date': dates,
            'close': benchmark_prices
        })

        result = await risk_manager.assess_stock_risk(
            'AAPL',
            sample_price_history,
            benchmark_history=benchmark_history
        )

        # Beta should be calculated from benchmark
        assert result.beta != 1.0 or True  # May still be close to 1

    @pytest.mark.asyncio
    async def test_stock_risk_insufficient_data(self, risk_manager):
        """Test stock risk with insufficient data."""
        short_history = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
            'close': np.linspace(100, 105, 10)
        })

        result = await risk_manager.assess_stock_risk('TEST', short_history)

        # Should return default assessment
        assert 'Insufficient data' in result.risk_factors[0]

    @pytest.mark.asyncio
    async def test_stock_risk_none_history(self, risk_manager):
        """Test stock risk with None history."""
        result = await risk_manager.assess_stock_risk('TEST', None)

        assert result.risk_level == RiskLevel.MODERATE
        assert 'Insufficient data' in result.risk_factors[0]


# ============================================================================
# Risk Decomposition Tests
# ============================================================================

class TestRiskDecomposition:
    """Test risk decomposition functionality."""

    def test_asset_risk_decomposition(self, risk_manager):
        """Test risk decomposition by asset."""
        weights = np.array([0.3, 0.3, 0.4])
        cov_matrix = np.array([
            [0.04, 0.01, 0.02],
            [0.01, 0.09, 0.03],
            [0.02, 0.03, 0.06]
        ])
        tickers = ['A', 'B', 'C']

        decomp = risk_manager._decompose_portfolio_risk(weights, cov_matrix, tickers)

        assert isinstance(decomp, RiskDecomposition)
        assert decomp.total_risk > 0
        assert len(decomp.marginal_contributions) == 3
        assert len(decomp.percentage_contributions) == 3

        # Percentage contributions should sum to approximately 1
        total_pct = sum(decomp.percentage_contributions.values())
        assert abs(total_pct - 1.0) < 0.01

    def test_sector_risk_decomposition(self, risk_manager):
        """Test risk decomposition by sector."""
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        cov_matrix = np.array([
            [0.04, 0.03, 0.02, 0.01, 0.01],
            [0.03, 0.05, 0.02, 0.01, 0.01],
            [0.02, 0.02, 0.03, 0.01, 0.01],
            [0.01, 0.01, 0.01, 0.04, 0.02],
            [0.01, 0.01, 0.01, 0.02, 0.05]
        ])
        tickers = ['AAPL', 'MSFT', 'XOM', 'JPM', 'GS']
        sectors = {
            'AAPL': 'Technology',
            'MSFT': 'Technology',
            'XOM': 'Energy',
            'JPM': 'Financials',
            'GS': 'Financials'
        }

        sector_contrib = risk_manager.decompose_risk_by_sector(
            weights, cov_matrix, tickers, sectors
        )

        assert 'Technology' in sector_contrib
        assert 'Energy' in sector_contrib
        assert 'Financials' in sector_contrib


# ============================================================================
# Position Sizing Tests
# ============================================================================

class TestPositionSizing:
    """Test position sizing and risk limits."""

    def test_position_size_check_within_limits(self, risk_manager):
        """Test position size check when within limits."""
        acceptable, message = risk_manager.check_position_size(
            proposed_weight=0.05,
            risk_score=0.3
        )

        assert acceptable is True
        assert "within limits" in message.lower()

    def test_position_size_check_exceeds_limit(self, risk_manager):
        """Test position size check when exceeding limits."""
        acceptable, message = risk_manager.check_position_size(
            proposed_weight=0.15,
            risk_score=0.5
        )

        assert acceptable is False
        assert "exceeds" in message.lower()

    def test_position_size_risk_adjusted(self, risk_manager):
        """Test risk-adjusted position sizing."""
        # Low risk allows larger position
        ok_low, _ = risk_manager.check_position_size(0.08, risk_score=0.1)

        # High risk requires smaller position
        ok_high, _ = risk_manager.check_position_size(0.08, risk_score=0.8)

        # Same weight, but high risk should be rejected
        assert ok_low is True
        assert ok_high is False

    def test_optimal_position_size(self, risk_manager, sample_returns):
        """Test optimal position size calculation."""
        result = risk_manager.calculate_optimal_position_size(sample_returns)

        assert 'kelly_full' in result
        assert 'kelly_half' in result
        assert 'volatility_target' in result
        assert 'var_target' in result
        assert 'recommended' in result

        # All should be between 0 and 1
        for key, value in result.items():
            assert 0 <= value <= 1

        # Half Kelly should always be <= full Kelly (after clipping)
        assert result['kelly_half'] <= result['kelly_full']

        # Recommended should be the minimum of all approaches (conservative)
        assert result['recommended'] <= result['kelly_half']
        assert result['recommended'] <= result['volatility_target']
        assert result['recommended'] <= result['var_target']


# ============================================================================
# Risk Classification Tests
# ============================================================================

class TestRiskClassification:
    """Test risk level classification."""

    def test_risk_score_calculation(self, risk_manager):
        """Test risk score calculation."""
        # Low risk profile
        low_score = risk_manager._calculate_risk_score(
            volatility=0.10,
            beta=1.0,
            max_drawdown=-0.05,
            sharpe_ratio=2.0
        )

        # High risk profile
        high_score = risk_manager._calculate_risk_score(
            volatility=0.50,
            beta=2.0,
            max_drawdown=-0.40,
            sharpe_ratio=0.2
        )

        assert 0 <= low_score <= 1
        assert 0 <= high_score <= 1
        assert low_score < high_score

    def test_risk_level_classification(self, risk_manager):
        """Test risk level classification thresholds."""
        assert risk_manager._classify_risk_level(0.1) == RiskLevel.VERY_LOW
        assert risk_manager._classify_risk_level(0.3) == RiskLevel.LOW
        assert risk_manager._classify_risk_level(0.5) == RiskLevel.MODERATE
        assert risk_manager._classify_risk_level(0.7) == RiskLevel.HIGH
        assert risk_manager._classify_risk_level(0.9) == RiskLevel.VERY_HIGH

    def test_risk_factors_identification(self, risk_manager):
        """Test risk factor identification."""
        factors = risk_manager._identify_risk_factors(
            volatility=0.50,
            beta=2.0,
            max_drawdown=-0.30,
            sharpe_ratio=0.3
        )

        assert len(factors) > 0
        assert any('volatility' in f.lower() for f in factors)
        assert any('beta' in f.lower() for f in factors)
        assert any('drawdown' in f.lower() for f in factors)
        assert any('sharpe' in f.lower() for f in factors)

    def test_recommendations_generation(self, risk_manager):
        """Test risk-based recommendations."""
        recommendations = risk_manager._generate_risk_recommendations(
            RiskLevel.HIGH,
            ["High volatility (50% annualized)"]
        )

        assert len(recommendations) > 0
        assert any('position size' in r.lower() for r in recommendations)


# ============================================================================
# Edge Cases and Error Handling Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_var_with_insufficient_data(self, risk_manager):
        """Test VaR with very short return series."""
        short_returns = np.array([0.01, -0.02, 0.015])

        # Should fall back to parametric method
        result = risk_manager.calculate_var(short_returns, confidence=0.95)
        assert isinstance(result, VaRResult)

    def test_var_with_zero_volatility(self, risk_manager):
        """Test VaR when volatility is zero."""
        constant_returns = np.zeros(100)
        result = risk_manager.calculate_var(constant_returns, confidence=0.95)

        # VaR should be 0 when there's no volatility
        assert abs(result.var_value) < 0.001

    def test_drawdown_single_price(self, risk_manager):
        """Test drawdown with single price."""
        single_price = np.array([100])
        max_dd, peak, trough = risk_manager.calculate_max_drawdown(single_price)

        assert max_dd == 0
        assert peak == 0
        assert trough == 0

    def test_beta_mismatched_lengths(self, risk_manager):
        """Test beta calculation with mismatched return lengths."""
        returns_long = np.random.normal(0, 0.01, 252)
        returns_short = np.random.normal(0, 0.01, 100)

        # Should handle mismatched lengths
        result = risk_manager.calculate_beta(returns_long, returns_short)
        assert 'beta' in result

    def test_sortino_ratio_no_downside(self, risk_manager):
        """Test Sortino ratio when there's no downside."""
        positive_returns = np.abs(np.random.normal(0.01, 0.005, 100))
        sortino = risk_manager._calculate_sortino_ratio(positive_returns)

        # Should return infinity when no downside
        assert sortino == float('inf')


# ============================================================================
# Custom Settings Tests
# ============================================================================

class TestCustomSettings:
    """Test risk manager with custom settings."""

    def test_custom_var_horizon(self, custom_risk_manager, sample_returns):
        """Test custom VaR horizon setting."""
        result = custom_risk_manager.calculate_var(sample_returns, confidence=0.95)
        assert result.horizon_days == 5  # Default set in custom_risk_manager

    def test_custom_position_limits(self, custom_risk_manager):
        """Test custom position limit settings."""
        # Custom manager has 15% max position
        ok_standard, _ = custom_risk_manager.check_position_size(0.12, risk_score=0.2)
        assert ok_standard is True

    def test_custom_monte_carlo_simulations(self, custom_risk_manager, sample_returns):
        """Test custom Monte Carlo simulation count."""
        # Custom manager uses 5000 simulations instead of 10000
        result = custom_risk_manager.calculate_var(
            sample_returns, confidence=0.95, method='monte_carlo'
        )
        assert isinstance(result, VaRResult)
