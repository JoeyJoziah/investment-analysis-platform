"""
Comprehensive Risk Manager Implementation

This module provides a full-featured risk management system for the investment
analysis platform, including:
- Value at Risk (VaR) calculations (Historical, Parametric, Monte Carlo)
- Conditional Value at Risk (CVaR/Expected Shortfall)
- Maximum Drawdown analysis
- Beta and Tracking Error calculations
- Stress Testing (historical scenarios and custom shocks)
- Portfolio risk decomposition
- Risk limits and alerts
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from scipy import stats
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level classification."""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class VaRMethod(Enum):
    """VaR calculation methods."""
    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"


@dataclass
class RiskAssessment:
    """Result of a risk assessment."""
    ticker: str
    risk_level: RiskLevel
    risk_score: float  # 0-1
    volatility: float
    beta: float
    var_95: float
    cvar_95: float
    max_drawdown: float
    sharpe_ratio: float
    risk_factors: List[str]
    recommendations: List[str]
    assessed_at: datetime


@dataclass
class VaRResult:
    """Result of a VaR calculation."""
    var_value: float
    confidence_level: float
    method: VaRMethod
    horizon_days: int
    additional_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StressTestResult:
    """Result of a stress test."""
    scenario_name: str
    portfolio_loss: float
    asset_impacts: Dict[str, float]
    var_breach: bool
    description: str
    historical_date: Optional[str] = None


@dataclass
class RiskDecomposition:
    """Risk decomposition by asset or sector."""
    total_risk: float
    marginal_contributions: Dict[str, float]
    percentage_contributions: Dict[str, float]
    diversification_benefit: float


# Historical stress scenarios with typical asset class impacts
HISTORICAL_SCENARIOS = {
    "2008_financial_crisis": {
        "name": "2008 Financial Crisis",
        "description": "Global financial crisis triggered by subprime mortgage collapse",
        "date": "2008-09-15",
        "equity_shock": -0.50,
        "bond_shock": 0.05,
        "commodity_shock": -0.35,
        "volatility_multiplier": 3.0,
        "correlation_shock": 0.3,  # Correlations increase in crisis
    },
    "2020_covid_crash": {
        "name": "COVID-19 March 2020 Crash",
        "description": "Rapid market decline due to pandemic fears",
        "date": "2020-03-16",
        "equity_shock": -0.34,
        "bond_shock": 0.02,
        "commodity_shock": -0.40,
        "volatility_multiplier": 4.0,
        "correlation_shock": 0.4,
    },
    "2000_dotcom_burst": {
        "name": "Dot-Com Bubble Burst",
        "description": "Technology sector collapse",
        "date": "2000-03-10",
        "equity_shock": -0.40,
        "tech_shock": -0.75,
        "bond_shock": 0.08,
        "commodity_shock": -0.15,
        "volatility_multiplier": 2.5,
        "correlation_shock": 0.2,
    },
    "2011_european_debt": {
        "name": "European Debt Crisis",
        "description": "Sovereign debt crisis in Europe",
        "date": "2011-08-05",
        "equity_shock": -0.20,
        "bond_shock": -0.05,
        "commodity_shock": -0.15,
        "volatility_multiplier": 2.0,
        "correlation_shock": 0.25,
    },
    "1987_black_monday": {
        "name": "Black Monday 1987",
        "description": "Largest single-day stock market crash",
        "date": "1987-10-19",
        "equity_shock": -0.22,
        "bond_shock": 0.03,
        "commodity_shock": -0.10,
        "volatility_multiplier": 5.0,
        "correlation_shock": 0.5,
    },
    "2022_rate_hike": {
        "name": "2022 Rate Hike Cycle",
        "description": "Aggressive Fed rate hikes to combat inflation",
        "date": "2022-06-13",
        "equity_shock": -0.25,
        "bond_shock": -0.15,
        "commodity_shock": -0.10,
        "volatility_multiplier": 2.0,
        "correlation_shock": 0.3,
    },
}


class RiskManager:
    """
    Comprehensive risk management system.

    The RiskManager provides:
    - VaR calculations using multiple methods (Historical, Parametric, Monte Carlo)
    - CVaR/Expected Shortfall calculations
    - Maximum Drawdown analysis
    - Beta and Tracking Error calculations
    - Stress Testing with historical scenarios and custom shocks
    - Portfolio-level risk aggregation
    - Risk decomposition by asset and sector
    """

    def __init__(
        self,
        max_portfolio_var: float = 0.02,
        max_position_size: float = 0.10,
        min_sharpe_ratio: float = 0.5,
        risk_free_rate: float = 0.045,
        monte_carlo_simulations: int = 10000,
        var_horizon_days: int = 1
    ):
        """
        Initialize the risk manager.

        Args:
            max_portfolio_var: Maximum acceptable portfolio VaR (daily)
            max_position_size: Maximum position size as fraction of portfolio
            min_sharpe_ratio: Minimum acceptable Sharpe ratio
            risk_free_rate: Annual risk-free rate for calculations
            monte_carlo_simulations: Number of Monte Carlo simulations
            var_horizon_days: Default horizon for VaR calculations
        """
        self.max_portfolio_var = max_portfolio_var
        self.max_position_size = max_position_size
        self.min_sharpe_ratio = min_sharpe_ratio
        self.risk_free_rate = risk_free_rate
        self.monte_carlo_simulations = monte_carlo_simulations
        self.var_horizon_days = var_horizon_days
        logger.info("RiskManager initialized")

    # =========================================================================
    # VaR Calculations
    # =========================================================================

    def calculate_var(
        self,
        returns: Union[np.ndarray, pd.Series],
        confidence: float = 0.95,
        method: str = 'historical',
        horizon_days: Optional[int] = None
    ) -> VaRResult:
        """
        Calculate Value at Risk using specified method.

        Args:
            returns: Array or Series of returns (daily)
            confidence: Confidence level (e.g., 0.95 for 95% VaR)
            method: 'historical', 'parametric', or 'monte_carlo'
            horizon_days: Time horizon in days (default: self.var_horizon_days)

        Returns:
            VaRResult with VaR value and additional metrics
        """
        returns = self._ensure_array(returns)
        horizon_days = horizon_days or self.var_horizon_days

        if len(returns) < 30:
            logger.warning("Insufficient data for VaR calculation, using parametric method")
            method = 'parametric'

        method_enum = VaRMethod(method.lower())

        if method_enum == VaRMethod.HISTORICAL:
            var_value = self._var_historical(returns, confidence, horizon_days)
        elif method_enum == VaRMethod.PARAMETRIC:
            var_value = self._var_parametric(returns, confidence, horizon_days)
        elif method_enum == VaRMethod.MONTE_CARLO:
            var_value = self._var_monte_carlo(returns, confidence, horizon_days)
        else:
            raise ValueError(f"Unknown VaR method: {method}")

        # Additional metrics
        additional = {
            'mean_return': float(np.mean(returns)),
            'std_return': float(np.std(returns)),
            'skewness': float(stats.skew(returns)),
            'kurtosis': float(stats.kurtosis(returns)),
            'data_points': len(returns),
        }

        return VaRResult(
            var_value=var_value,
            confidence_level=confidence,
            method=method_enum,
            horizon_days=horizon_days,
            additional_metrics=additional
        )

    def _var_historical(
        self,
        returns: np.ndarray,
        confidence: float,
        horizon_days: int
    ) -> float:
        """
        Historical simulation VaR.

        Uses the empirical distribution of returns to estimate VaR.
        """
        # Scale returns to the specified horizon
        if horizon_days > 1:
            # Use overlapping window returns for multi-day horizon
            if len(returns) >= horizon_days:
                rolling_returns = pd.Series(returns).rolling(horizon_days).sum().dropna().values
            else:
                # Fallback: scale by square root of time
                rolling_returns = returns * np.sqrt(horizon_days)
        else:
            rolling_returns = returns

        # VaR is the (1 - confidence) percentile of returns (losses are negative)
        var_percentile = (1 - confidence) * 100
        var_value = np.percentile(rolling_returns, var_percentile)

        return float(var_value)

    def _var_parametric(
        self,
        returns: np.ndarray,
        confidence: float,
        horizon_days: int
    ) -> float:
        """
        Parametric (Variance-Covariance) VaR.

        Assumes returns are normally distributed.
        """
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        # Scale to horizon (square root of time rule)
        mean_scaled = mean_return * horizon_days
        std_scaled = std_return * np.sqrt(horizon_days)

        # Z-score for confidence level
        z_score = stats.norm.ppf(1 - confidence)

        # VaR = mean - z * std (negative value represents loss)
        var_value = mean_scaled + z_score * std_scaled

        return float(var_value)

    def _var_monte_carlo(
        self,
        returns: np.ndarray,
        confidence: float,
        horizon_days: int
    ) -> float:
        """
        Monte Carlo VaR.

        Simulates many possible return paths and estimates VaR from the distribution.
        """
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        # Simulate returns
        np.random.seed(42)  # For reproducibility
        simulated_returns = np.random.normal(
            mean_return * horizon_days,
            std_return * np.sqrt(horizon_days),
            self.monte_carlo_simulations
        )

        # VaR from simulated distribution
        var_percentile = (1 - confidence) * 100
        var_value = np.percentile(simulated_returns, var_percentile)

        return float(var_value)

    def calculate_var_all_methods(
        self,
        returns: Union[np.ndarray, pd.Series],
        confidence: float = 0.95,
        horizon_days: Optional[int] = None
    ) -> Dict[str, VaRResult]:
        """
        Calculate VaR using all three methods for comparison.

        Args:
            returns: Array or Series of returns
            confidence: Confidence level
            horizon_days: Time horizon

        Returns:
            Dictionary mapping method name to VaRResult
        """
        results = {}
        for method in ['historical', 'parametric', 'monte_carlo']:
            try:
                results[method] = self.calculate_var(
                    returns, confidence, method, horizon_days
                )
            except Exception as e:
                logger.error(f"Error calculating {method} VaR: {e}")

        return results

    # =========================================================================
    # CVaR / Expected Shortfall
    # =========================================================================

    def calculate_cvar(
        self,
        returns: Union[np.ndarray, pd.Series],
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).

        CVaR represents the expected loss given that the loss exceeds VaR.
        It is a coherent risk measure that accounts for tail risk.

        Args:
            returns: Array or Series of returns
            confidence: Confidence level

        Returns:
            CVaR value (negative indicates loss)
        """
        returns = self._ensure_array(returns)

        # Calculate VaR threshold
        var_threshold = self._var_historical(returns, confidence, 1)

        # CVaR is the mean of returns below VaR
        tail_returns = returns[returns <= var_threshold]

        if len(tail_returns) == 0:
            # No returns below VaR threshold, use VaR as CVaR
            return var_threshold

        cvar = np.mean(tail_returns)
        return float(cvar)

    def calculate_cvar_parametric(
        self,
        returns: Union[np.ndarray, pd.Series],
        confidence: float = 0.95
    ) -> float:
        """
        Calculate parametric CVaR assuming normal distribution.

        Args:
            returns: Array or Series of returns
            confidence: Confidence level

        Returns:
            CVaR value
        """
        returns = self._ensure_array(returns)
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        # For normal distribution, CVaR has a closed-form solution
        alpha = 1 - confidence
        z_alpha = stats.norm.ppf(alpha)
        pdf_z = stats.norm.pdf(z_alpha)

        cvar = mean_return - std_return * pdf_z / alpha

        return float(cvar)

    # =========================================================================
    # Maximum Drawdown
    # =========================================================================

    def calculate_max_drawdown(
        self,
        prices: Union[np.ndarray, pd.Series]
    ) -> Tuple[float, int, int]:
        """
        Calculate Maximum Drawdown and identify the drawdown period.

        Args:
            prices: Array or Series of prices

        Returns:
            Tuple of (max_drawdown, peak_index, trough_index)
        """
        prices = self._ensure_array(prices)

        if len(prices) < 2:
            return 0.0, 0, 0

        # Calculate running maximum
        running_max = np.maximum.accumulate(prices)

        # Calculate drawdown at each point
        drawdown = (prices - running_max) / running_max

        # Find maximum drawdown
        max_dd = np.min(drawdown)
        trough_idx = int(np.argmin(drawdown))

        # Find the peak before the trough
        peak_idx = int(np.argmax(prices[:trough_idx + 1]))

        return float(max_dd), peak_idx, trough_idx

    def calculate_drawdown_series(
        self,
        prices: Union[np.ndarray, pd.Series]
    ) -> pd.DataFrame:
        """
        Calculate the full drawdown series with details.

        Args:
            prices: Array or Series of prices

        Returns:
            DataFrame with drawdown metrics at each point
        """
        prices = self._ensure_array(prices)

        running_max = np.maximum.accumulate(prices)
        drawdown = (prices - running_max) / running_max

        # Calculate drawdown duration
        in_drawdown = drawdown < 0
        drawdown_start = np.zeros(len(prices), dtype=int)
        current_start = 0

        for i in range(len(prices)):
            if not in_drawdown[i]:
                current_start = i
            drawdown_start[i] = current_start

        drawdown_duration = np.arange(len(prices)) - drawdown_start

        return pd.DataFrame({
            'price': prices,
            'running_max': running_max,
            'drawdown': drawdown,
            'drawdown_duration': drawdown_duration
        })

    def calculate_all_drawdowns(
        self,
        prices: Union[np.ndarray, pd.Series],
        threshold: float = -0.05
    ) -> List[Dict[str, Any]]:
        """
        Identify all drawdown periods exceeding a threshold.

        Args:
            prices: Array or Series of prices
            threshold: Minimum drawdown to include (e.g., -0.05 for 5%)

        Returns:
            List of drawdown dictionaries with start, end, depth, and duration
        """
        prices = self._ensure_array(prices)
        dd_series = self.calculate_drawdown_series(prices)

        drawdowns = []
        in_dd = False
        dd_start = 0
        peak_value = prices[0]

        for i in range(len(prices)):
            if dd_series['drawdown'].iloc[i] < threshold and not in_dd:
                in_dd = True
                dd_start = i - 1 if i > 0 else 0
                peak_value = dd_series['running_max'].iloc[i]
            elif dd_series['drawdown'].iloc[i] >= 0 and in_dd:
                in_dd = False
                trough_idx = int(np.argmin(dd_series['drawdown'].iloc[dd_start:i])) + dd_start
                drawdowns.append({
                    'start_idx': dd_start,
                    'trough_idx': trough_idx,
                    'end_idx': i,
                    'peak_value': peak_value,
                    'trough_value': prices[trough_idx],
                    'max_drawdown': float(dd_series['drawdown'].iloc[trough_idx]),
                    'duration': i - dd_start,
                    'recovery_time': i - trough_idx
                })

        # Handle ongoing drawdown
        if in_dd:
            trough_idx = int(np.argmin(dd_series['drawdown'].iloc[dd_start:])) + dd_start
            drawdowns.append({
                'start_idx': dd_start,
                'trough_idx': trough_idx,
                'end_idx': len(prices) - 1,
                'peak_value': peak_value,
                'trough_value': prices[trough_idx],
                'max_drawdown': float(dd_series['drawdown'].iloc[trough_idx]),
                'duration': len(prices) - dd_start,
                'recovery_time': None  # Still in drawdown
            })

        return drawdowns

    # =========================================================================
    # Beta and Tracking Error
    # =========================================================================

    def calculate_beta(
        self,
        returns: Union[np.ndarray, pd.Series],
        benchmark_returns: Union[np.ndarray, pd.Series]
    ) -> Dict[str, float]:
        """
        Calculate Beta and related metrics.

        Beta measures sensitivity to market movements.

        Args:
            returns: Asset returns
            benchmark_returns: Benchmark (market) returns

        Returns:
            Dictionary with beta, alpha, r_squared, and correlation
        """
        returns = self._ensure_array(returns)
        benchmark_returns = self._ensure_array(benchmark_returns)

        # Align lengths
        min_len = min(len(returns), len(benchmark_returns))
        returns = returns[-min_len:]
        benchmark_returns = benchmark_returns[-min_len:]

        if len(returns) < 30:
            logger.warning("Insufficient data for beta calculation")
            return {
                'beta': 1.0,
                'alpha': 0.0,
                'r_squared': 0.0,
                'correlation': 0.0,
                'data_points': len(returns)
            }

        # Calculate covariance and variance
        cov_matrix = np.cov(returns, benchmark_returns)
        covariance = cov_matrix[0, 1]
        benchmark_variance = cov_matrix[1, 1]

        # Beta = Cov(r, rm) / Var(rm)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0

        # Alpha = mean(r) - beta * mean(rm)
        alpha = np.mean(returns) - beta * np.mean(benchmark_returns)

        # Annualize alpha (assuming daily returns)
        alpha_annualized = alpha * 252

        # R-squared
        correlation = np.corrcoef(returns, benchmark_returns)[0, 1]
        r_squared = correlation ** 2

        return {
            'beta': float(beta),
            'alpha': float(alpha),
            'alpha_annualized': float(alpha_annualized),
            'r_squared': float(r_squared),
            'correlation': float(correlation),
            'data_points': len(returns)
        }

    def calculate_tracking_error(
        self,
        returns: Union[np.ndarray, pd.Series],
        benchmark_returns: Union[np.ndarray, pd.Series],
        annualize: bool = True
    ) -> Dict[str, float]:
        """
        Calculate Tracking Error (deviation from benchmark).

        Args:
            returns: Portfolio/asset returns
            benchmark_returns: Benchmark returns
            annualize: Whether to annualize the tracking error

        Returns:
            Dictionary with tracking error metrics
        """
        returns = self._ensure_array(returns)
        benchmark_returns = self._ensure_array(benchmark_returns)

        # Align lengths
        min_len = min(len(returns), len(benchmark_returns))
        returns = returns[-min_len:]
        benchmark_returns = benchmark_returns[-min_len:]

        # Active returns (excess over benchmark)
        active_returns = returns - benchmark_returns

        # Tracking error is the std of active returns
        tracking_error = np.std(active_returns)

        # Annualize if requested (assuming daily returns)
        if annualize:
            tracking_error_annualized = tracking_error * np.sqrt(252)
        else:
            tracking_error_annualized = tracking_error

        # Information ratio
        mean_active_return = np.mean(active_returns)
        if annualize:
            mean_active_return_annualized = mean_active_return * 252
        else:
            mean_active_return_annualized = mean_active_return

        information_ratio = (
            mean_active_return_annualized / tracking_error_annualized
            if tracking_error_annualized > 0 else 0
        )

        return {
            'tracking_error': float(tracking_error),
            'tracking_error_annualized': float(tracking_error_annualized),
            'mean_active_return': float(mean_active_return),
            'mean_active_return_annualized': float(mean_active_return_annualized),
            'information_ratio': float(information_ratio),
            'data_points': len(returns)
        }

    # =========================================================================
    # Stress Testing
    # =========================================================================

    def stress_test(
        self,
        portfolio: Dict[str, float],
        scenario: str,
        asset_betas: Optional[Dict[str, float]] = None,
        sector_mappings: Optional[Dict[str, str]] = None
    ) -> StressTestResult:
        """
        Apply a historical stress scenario to a portfolio.

        Args:
            portfolio: Dictionary mapping tickers to weights
            scenario: Scenario name (e.g., '2008_financial_crisis')
            asset_betas: Dictionary mapping tickers to beta values
            sector_mappings: Dictionary mapping tickers to sectors

        Returns:
            StressTestResult with scenario impact details
        """
        if scenario not in HISTORICAL_SCENARIOS:
            available = ", ".join(HISTORICAL_SCENARIOS.keys())
            raise ValueError(f"Unknown scenario: {scenario}. Available: {available}")

        scenario_data = HISTORICAL_SCENARIOS[scenario]
        asset_betas = asset_betas or {}
        sector_mappings = sector_mappings or {}

        asset_impacts = {}
        portfolio_loss = 0.0

        for ticker, weight in portfolio.items():
            beta = asset_betas.get(ticker, 1.0)
            sector = sector_mappings.get(ticker, 'equity')

            # Determine base shock based on sector
            if sector.lower() == 'tech' and 'tech_shock' in scenario_data:
                base_shock = scenario_data['tech_shock']
            elif sector.lower() == 'bond':
                base_shock = scenario_data['bond_shock']
            elif sector.lower() == 'commodity':
                base_shock = scenario_data['commodity_shock']
            else:
                base_shock = scenario_data['equity_shock']

            # Adjust shock by beta
            adjusted_shock = base_shock * beta

            asset_impacts[ticker] = adjusted_shock
            portfolio_loss += weight * adjusted_shock

        # Check if VaR is breached
        var_breach = abs(portfolio_loss) > self.max_portfolio_var

        return StressTestResult(
            scenario_name=scenario_data['name'],
            portfolio_loss=float(portfolio_loss),
            asset_impacts=asset_impacts,
            var_breach=var_breach,
            description=scenario_data['description'],
            historical_date=scenario_data['date']
        )

    def stress_test_custom(
        self,
        portfolio: Dict[str, float],
        shocks: Dict[str, float],
        scenario_name: str = "Custom Scenario",
        description: str = "User-defined stress test"
    ) -> StressTestResult:
        """
        Apply custom shocks to a portfolio.

        Args:
            portfolio: Dictionary mapping tickers to weights
            shocks: Dictionary mapping tickers to shock values
            scenario_name: Name for the custom scenario
            description: Description of the scenario

        Returns:
            StressTestResult with custom scenario impact
        """
        asset_impacts = {}
        portfolio_loss = 0.0

        for ticker, weight in portfolio.items():
            shock = shocks.get(ticker, 0.0)
            asset_impacts[ticker] = shock
            portfolio_loss += weight * shock

        var_breach = abs(portfolio_loss) > self.max_portfolio_var

        return StressTestResult(
            scenario_name=scenario_name,
            portfolio_loss=float(portfolio_loss),
            asset_impacts=asset_impacts,
            var_breach=var_breach,
            description=description,
            historical_date=None
        )

    def stress_test_all_scenarios(
        self,
        portfolio: Dict[str, float],
        asset_betas: Optional[Dict[str, float]] = None,
        sector_mappings: Optional[Dict[str, str]] = None
    ) -> List[StressTestResult]:
        """
        Run all historical stress scenarios on a portfolio.

        Args:
            portfolio: Dictionary mapping tickers to weights
            asset_betas: Dictionary mapping tickers to beta values
            sector_mappings: Dictionary mapping tickers to sectors

        Returns:
            List of StressTestResult for all scenarios
        """
        results = []
        for scenario_name in HISTORICAL_SCENARIOS.keys():
            try:
                result = self.stress_test(
                    portfolio, scenario_name, asset_betas, sector_mappings
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error in stress test {scenario_name}: {e}")

        # Sort by portfolio loss (worst first)
        results.sort(key=lambda x: x.portfolio_loss)

        return results

    # =========================================================================
    # Portfolio Risk Aggregation
    # =========================================================================

    async def assess_portfolio_risk(
        self,
        positions: Dict[str, float],
        price_histories: Dict[str, pd.DataFrame],
        benchmark_history: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Assess portfolio-level risk with comprehensive metrics.

        Args:
            positions: Dictionary mapping tickers to position weights
            price_histories: Dictionary mapping tickers to price DataFrames
            benchmark_history: Optional benchmark price DataFrame

        Returns:
            Dictionary with comprehensive portfolio risk metrics
        """
        logger.info(f"Assessing portfolio risk for {len(positions)} positions")

        # Validate inputs
        if not positions or not price_histories:
            return self._default_portfolio_assessment()

        # Calculate individual returns
        returns_dict = {}
        for ticker, weight in positions.items():
            if ticker in price_histories and len(price_histories[ticker]) >= 30:
                prices = price_histories[ticker]['close'].values
                returns_dict[ticker] = np.diff(prices) / prices[:-1]

        if not returns_dict:
            return self._default_portfolio_assessment()

        # Create returns matrix
        tickers = list(returns_dict.keys())
        weights = np.array([positions.get(t, 0) for t in tickers])
        weights = weights / weights.sum()  # Normalize

        # Align returns to same length
        min_len = min(len(r) for r in returns_dict.values())
        returns_matrix = np.column_stack([
            returns_dict[t][-min_len:] for t in tickers
        ])

        # Portfolio returns
        portfolio_returns = returns_matrix @ weights

        # Covariance matrix
        cov_matrix = np.cov(returns_matrix.T)

        # Portfolio volatility
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
        portfolio_vol_annualized = portfolio_vol * np.sqrt(252)

        # VaR calculations
        var_results = self.calculate_var_all_methods(portfolio_returns, 0.95)

        # CVaR
        cvar_95 = self.calculate_cvar(portfolio_returns, 0.95)

        # Portfolio prices (normalized to 100)
        portfolio_prices = 100 * np.cumprod(1 + portfolio_returns)
        portfolio_prices = np.insert(portfolio_prices, 0, 100)

        # Maximum drawdown
        max_dd, peak_idx, trough_idx = self.calculate_max_drawdown(portfolio_prices)

        # Sharpe ratio
        excess_returns = portfolio_returns - self.risk_free_rate / 252
        sharpe_ratio = (
            np.mean(excess_returns) * 252 / portfolio_vol_annualized
            if portfolio_vol_annualized > 0 else 0
        )

        # Beta and tracking error (if benchmark provided)
        beta_metrics = {}
        tracking_metrics = {}
        if benchmark_history is not None and len(benchmark_history) >= min_len:
            benchmark_prices = benchmark_history['close'].values[-min_len - 1:]
            benchmark_returns = np.diff(benchmark_prices) / benchmark_prices[:-1]

            beta_metrics = self.calculate_beta(portfolio_returns, benchmark_returns)
            tracking_metrics = self.calculate_tracking_error(
                portfolio_returns, benchmark_returns
            )

        # Risk decomposition
        risk_decomp = self._decompose_portfolio_risk(weights, cov_matrix, tickers)

        # Diversification ratio
        individual_vols = np.sqrt(np.diag(cov_matrix))
        weighted_avg_vol = np.dot(weights, individual_vols)
        diversification_ratio = (
            weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1.0
        )

        # Concentration metrics
        hhi = np.sum(weights ** 2)  # Herfindahl-Hirschman Index
        effective_n = 1 / hhi if hhi > 0 else len(weights)

        # Check risk limits
        within_var_limit = abs(var_results['historical'].var_value) <= self.max_portfolio_var
        max_weight = np.max(weights)
        within_position_limit = max_weight <= self.max_position_size
        within_sharpe_limit = sharpe_ratio >= self.min_sharpe_ratio

        # Identify risk factors
        risk_factors = self._identify_portfolio_risk_factors(
            portfolio_vol_annualized,
            max_dd,
            sharpe_ratio,
            beta_metrics.get('beta', 1.0),
            hhi
        )

        return {
            'portfolio_volatility': float(portfolio_vol_annualized),
            'portfolio_var_95_historical': float(var_results.get('historical', VaRResult(0, 0.95, VaRMethod.HISTORICAL, 1)).var_value),
            'portfolio_var_95_parametric': float(var_results.get('parametric', VaRResult(0, 0.95, VaRMethod.PARAMETRIC, 1)).var_value),
            'portfolio_var_95_monte_carlo': float(var_results.get('monte_carlo', VaRResult(0, 0.95, VaRMethod.MONTE_CARLO, 1)).var_value),
            'portfolio_cvar_95': float(cvar_95),
            'max_drawdown': float(max_dd),
            'max_drawdown_peak_idx': peak_idx,
            'max_drawdown_trough_idx': trough_idx,
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(self._calculate_sortino_ratio(portfolio_returns)),
            'diversification_ratio': float(diversification_ratio),
            'effective_n_assets': float(effective_n),
            'concentration_hhi': float(hhi),
            'beta': beta_metrics.get('beta', None),
            'alpha_annualized': beta_metrics.get('alpha_annualized', None),
            'tracking_error': tracking_metrics.get('tracking_error_annualized', None),
            'information_ratio': tracking_metrics.get('information_ratio', None),
            'risk_decomposition': risk_decomp,
            'within_var_limit': within_var_limit,
            'within_position_limit': within_position_limit,
            'within_sharpe_limit': within_sharpe_limit,
            'within_all_limits': within_var_limit and within_position_limit and within_sharpe_limit,
            'risk_factors': risk_factors,
            'n_positions': len(positions),
            'data_points': min_len,
            'assessed_at': datetime.now(timezone.utc).isoformat()
        }

    def _calculate_sortino_ratio(
        self,
        returns: np.ndarray,
        target_return: float = 0.0
    ) -> float:
        """Calculate Sortino ratio (penalizes only downside volatility)."""
        excess_returns = returns - target_return
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0:
            return float('inf')

        downside_std = np.sqrt(np.mean(downside_returns ** 2))

        if downside_std == 0:
            return float('inf')

        mean_excess = np.mean(returns) - self.risk_free_rate / 252
        sortino = mean_excess * 252 / (downside_std * np.sqrt(252))

        return float(sortino)

    # =========================================================================
    # Risk Decomposition
    # =========================================================================

    def _decompose_portfolio_risk(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray,
        tickers: List[str]
    ) -> RiskDecomposition:
        """
        Decompose portfolio risk by asset contribution.

        Uses marginal contribution to risk (MCR) methodology.

        Args:
            weights: Asset weights
            cov_matrix: Covariance matrix
            tickers: Asset tickers

        Returns:
            RiskDecomposition with contribution details
        """
        # Total portfolio variance
        portfolio_variance = weights @ cov_matrix @ weights
        portfolio_volatility = np.sqrt(portfolio_variance)

        # Marginal contribution to risk
        # MCR_i = (Cov @ w)_i / portfolio_vol
        mcr = (cov_matrix @ weights) / portfolio_volatility

        # Component contribution to risk
        # CCR_i = w_i * MCR_i
        ccr = weights * mcr

        # Create dictionaries
        marginal_contributions = dict(zip(tickers, mcr.tolist()))
        percentage_contributions = dict(zip(
            tickers,
            (ccr / portfolio_volatility).tolist()
        ))

        # Diversification benefit
        individual_vols = np.sqrt(np.diag(cov_matrix))
        undiversified_risk = np.sum(weights * individual_vols)
        diversification_benefit = 1 - portfolio_volatility / undiversified_risk

        return RiskDecomposition(
            total_risk=float(portfolio_volatility),
            marginal_contributions=marginal_contributions,
            percentage_contributions=percentage_contributions,
            diversification_benefit=float(diversification_benefit)
        )

    def decompose_risk_by_sector(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray,
        tickers: List[str],
        sector_mappings: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Decompose portfolio risk by sector.

        Args:
            weights: Asset weights
            cov_matrix: Covariance matrix
            tickers: Asset tickers
            sector_mappings: Dictionary mapping tickers to sectors

        Returns:
            Dictionary mapping sectors to risk contributions
        """
        asset_decomp = self._decompose_portfolio_risk(weights, cov_matrix, tickers)

        sector_contributions = {}
        for ticker, pct_contrib in asset_decomp.percentage_contributions.items():
            sector = sector_mappings.get(ticker, 'Unknown')
            sector_contributions[sector] = (
                sector_contributions.get(sector, 0) + pct_contrib
            )

        return sector_contributions

    # =========================================================================
    # Individual Stock Risk Assessment
    # =========================================================================

    async def assess_stock_risk(
        self,
        ticker: str,
        price_history: pd.DataFrame,
        beta: Optional[float] = None,
        benchmark_history: Optional[pd.DataFrame] = None
    ) -> RiskAssessment:
        """
        Assess the risk of a single stock.

        Args:
            ticker: Stock ticker symbol
            price_history: DataFrame with at least 'close' column
            beta: Stock beta (will be calculated if not provided and benchmark given)
            benchmark_history: Optional benchmark price DataFrame for beta calculation

        Returns:
            RiskAssessment with comprehensive risk metrics
        """
        logger.info(f"Assessing risk for {ticker}")

        if price_history is None or len(price_history) < 30:
            logger.warning(f"Insufficient data for {ticker}, returning default assessment")
            return self._default_assessment(ticker)

        prices = price_history['close'].values
        returns = np.diff(prices) / prices[:-1]

        # Annualized volatility
        volatility = float(np.std(returns) * np.sqrt(252))

        # VaR (95% confidence)
        var_result = self.calculate_var(returns, 0.95, 'historical')
        var_95 = var_result.var_value

        # CVaR
        cvar_95 = self.calculate_cvar(returns, 0.95)

        # Maximum Drawdown
        max_dd, _, _ = self.calculate_max_drawdown(prices)

        # Calculate beta if benchmark provided
        if beta is None and benchmark_history is not None:
            benchmark_prices = benchmark_history['close'].values
            if len(benchmark_prices) >= len(prices):
                benchmark_returns = np.diff(benchmark_prices) / benchmark_prices[:-1]
                benchmark_returns = benchmark_returns[-len(returns):]
                beta_metrics = self.calculate_beta(returns, benchmark_returns)
                beta = beta_metrics['beta']

        beta = beta if beta is not None else 1.0

        # Sharpe Ratio
        excess_returns = returns - self.risk_free_rate / 252
        sharpe_ratio = (
            np.mean(excess_returns) * 252 / volatility
            if volatility > 0 else 0
        )

        # Calculate composite risk score
        risk_score = self._calculate_risk_score(volatility, beta, max_dd, sharpe_ratio)

        # Classify risk level
        risk_level = self._classify_risk_level(risk_score)

        # Identify risk factors
        risk_factors = self._identify_risk_factors(volatility, beta, max_dd, sharpe_ratio)

        # Generate recommendations
        recommendations = self._generate_risk_recommendations(risk_level, risk_factors)

        return RiskAssessment(
            ticker=ticker,
            risk_level=risk_level,
            risk_score=risk_score,
            volatility=volatility,
            beta=float(beta),
            var_95=float(var_95),
            cvar_95=float(cvar_95),
            max_drawdown=float(max_dd),
            sharpe_ratio=float(sharpe_ratio),
            risk_factors=risk_factors,
            recommendations=recommendations,
            assessed_at=datetime.now(timezone.utc)
        )

    # =========================================================================
    # Position Sizing and Risk Limits
    # =========================================================================

    def check_position_size(
        self,
        proposed_weight: float,
        risk_score: float
    ) -> Tuple[bool, str]:
        """
        Check if a proposed position size is within risk limits.

        Args:
            proposed_weight: Proposed position weight (0-1)
            risk_score: Risk score of the position (0-1)

        Returns:
            Tuple of (is_acceptable, reason)
        """
        # Adjust max position based on risk
        adjusted_max = self.max_position_size * (1 - risk_score * 0.5)

        if proposed_weight > adjusted_max:
            return (
                False,
                f"Position size {proposed_weight:.1%} exceeds risk-adjusted limit {adjusted_max:.1%}"
            )

        if proposed_weight > self.max_position_size:
            return (
                False,
                f"Position size {proposed_weight:.1%} exceeds maximum limit {self.max_position_size:.1%}"
            )

        return True, "Position size within limits"

    def calculate_optimal_position_size(
        self,
        returns: Union[np.ndarray, pd.Series],
        target_var: Optional[float] = None,
        method: str = 'kelly'
    ) -> Dict[str, float]:
        """
        Calculate optimal position size using various methods.

        Args:
            returns: Historical returns
            target_var: Target VaR for position (default: max_portfolio_var)
            method: 'kelly', 'volatility_target', or 'var_target'

        Returns:
            Dictionary with optimal size and details
        """
        returns = self._ensure_array(returns)
        target_var = target_var or self.max_portfolio_var

        results = {}

        # Kelly Criterion (full Kelly)
        mean_return = np.mean(returns)
        variance = np.var(returns)
        kelly_full = mean_return / variance if variance > 0 else 0
        kelly_half = kelly_full / 2  # Half Kelly is more conservative

        results['kelly_full'] = float(np.clip(kelly_full, 0, 1))
        results['kelly_half'] = float(np.clip(kelly_half, 0, 1))

        # Volatility-based sizing
        annual_vol = np.std(returns) * np.sqrt(252)
        target_vol = 0.15  # 15% target volatility
        vol_based_size = target_vol / annual_vol if annual_vol > 0 else 1

        results['volatility_target'] = float(np.clip(vol_based_size, 0, 1))

        # VaR-based sizing
        var_95 = abs(self._var_parametric(returns, 0.95, 1))
        var_based_size = target_var / var_95 if var_95 > 0 else 1

        results['var_target'] = float(np.clip(var_based_size, 0, 1))

        # Recommended size (conservative of all methods)
        results['recommended'] = float(np.clip(
            min(kelly_half, vol_based_size, var_based_size),
            0,
            self.max_position_size
        ))

        return results

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _ensure_array(self, data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Convert input to numpy array."""
        if isinstance(data, pd.Series):
            return data.values
        return np.asarray(data)

    def _calculate_risk_score(
        self,
        volatility: float,
        beta: float,
        max_drawdown: float,
        sharpe_ratio: float
    ) -> float:
        """Calculate composite risk score (0-1, higher is riskier)."""
        # Normalize components
        vol_score = min(volatility / 0.5, 1.0)  # 50% annual vol = max
        beta_score = min(abs(beta - 1) / 1.0, 1.0)  # Distance from market beta
        dd_score = min(abs(max_drawdown) / 0.3, 1.0)  # 30% drawdown = max
        sharpe_score = max(0, 1 - sharpe_ratio / 2)  # Higher Sharpe = lower risk

        # Weighted average
        risk_score = (
            0.35 * vol_score +
            0.15 * beta_score +
            0.25 * dd_score +
            0.25 * sharpe_score
        )

        return min(1.0, max(0.0, risk_score))

    def _classify_risk_level(self, risk_score: float) -> RiskLevel:
        """Classify risk level based on score."""
        if risk_score < 0.2:
            return RiskLevel.VERY_LOW
        elif risk_score < 0.4:
            return RiskLevel.LOW
        elif risk_score < 0.6:
            return RiskLevel.MODERATE
        elif risk_score < 0.8:
            return RiskLevel.HIGH
        else:
            return RiskLevel.VERY_HIGH

    def _identify_risk_factors(
        self,
        volatility: float,
        beta: float,
        max_drawdown: float,
        sharpe_ratio: float
    ) -> List[str]:
        """Identify specific risk factors."""
        factors = []

        if volatility > 0.4:
            factors.append(f"High volatility ({volatility:.0%} annualized)")
        if beta > 1.5:
            factors.append(f"High market sensitivity (beta: {beta:.2f})")
        if beta < 0.5:
            factors.append(f"Low correlation to market (beta: {beta:.2f})")
        if max_drawdown < -0.25:
            factors.append(f"Significant historical drawdown ({max_drawdown:.0%})")
        if sharpe_ratio < 0.5:
            factors.append(f"Poor risk-adjusted returns (Sharpe: {sharpe_ratio:.2f})")

        return factors

    def _identify_portfolio_risk_factors(
        self,
        volatility: float,
        max_drawdown: float,
        sharpe_ratio: float,
        beta: float,
        hhi: float
    ) -> List[str]:
        """Identify portfolio-level risk factors."""
        factors = []

        if volatility > 0.25:
            factors.append(f"Elevated portfolio volatility ({volatility:.0%} annualized)")
        if max_drawdown < -0.15:
            factors.append(f"Significant drawdown risk ({max_drawdown:.0%})")
        if sharpe_ratio < self.min_sharpe_ratio:
            factors.append(f"Below minimum Sharpe ratio ({sharpe_ratio:.2f} < {self.min_sharpe_ratio})")
        if beta > 1.2:
            factors.append(f"High market exposure (beta: {beta:.2f})")
        if hhi > 0.2:
            factors.append(f"Concentrated portfolio (HHI: {hhi:.2f})")

        return factors

    def _generate_risk_recommendations(
        self,
        risk_level: RiskLevel,
        risk_factors: List[str]
    ) -> List[str]:
        """Generate risk-based recommendations."""
        recommendations = []

        if risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]:
            recommendations.append("Consider reduced position size due to elevated risk")
            recommendations.append("Set tight stop-loss orders")
            recommendations.append("Monitor position closely")

        if risk_level == RiskLevel.VERY_HIGH:
            recommendations.append("This position is suitable only for aggressive portfolios")

        if any('volatility' in f.lower() for f in risk_factors):
            recommendations.append("Consider using options for hedging volatility")

        if any('drawdown' in f.lower() for f in risk_factors):
            recommendations.append("Implement trailing stop-loss strategy")

        if any('sharpe' in f.lower() for f in risk_factors):
            recommendations.append("Review the risk-return tradeoff for this position")

        return recommendations

    def _default_assessment(self, ticker: str) -> RiskAssessment:
        """Return default assessment when data is insufficient."""
        return RiskAssessment(
            ticker=ticker,
            risk_level=RiskLevel.MODERATE,
            risk_score=0.5,
            volatility=0.25,
            beta=1.0,
            var_95=-0.02,
            cvar_95=-0.03,
            max_drawdown=-0.15,
            sharpe_ratio=0.8,
            risk_factors=["Insufficient data for complete analysis"],
            recommendations=["Gather more historical data before making investment decisions"],
            assessed_at=datetime.now(timezone.utc)
        )

    def _default_portfolio_assessment(self) -> Dict[str, Any]:
        """Return default portfolio assessment."""
        return {
            'portfolio_volatility': 0.15,
            'portfolio_var_95_historical': -0.02,
            'portfolio_var_95_parametric': -0.02,
            'portfolio_var_95_monte_carlo': -0.02,
            'portfolio_cvar_95': -0.03,
            'max_drawdown': -0.10,
            'sharpe_ratio': 1.0,
            'sortino_ratio': 1.2,
            'diversification_ratio': 1.5,
            'effective_n_assets': 1.0,
            'concentration_hhi': 1.0,
            'beta': None,
            'alpha_annualized': None,
            'tracking_error': None,
            'information_ratio': None,
            'risk_decomposition': None,
            'within_var_limit': True,
            'within_position_limit': True,
            'within_sharpe_limit': True,
            'within_all_limits': True,
            'risk_factors': ['Insufficient data for complete analysis'],
            'n_positions': 0,
            'data_points': 0,
            'assessed_at': datetime.now(timezone.utc).isoformat()
        }

    # =========================================================================
    # Scenario Analysis Utilities
    # =========================================================================

    def get_available_scenarios(self) -> List[Dict[str, str]]:
        """Get list of available stress test scenarios."""
        return [
            {
                'id': scenario_id,
                'name': data['name'],
                'description': data['description'],
                'date': data['date']
            }
            for scenario_id, data in HISTORICAL_SCENARIOS.items()
        ]

    def create_custom_scenario(
        self,
        name: str,
        equity_shock: float,
        bond_shock: float = 0.0,
        commodity_shock: float = 0.0,
        tech_shock: Optional[float] = None,
        volatility_multiplier: float = 2.0,
        correlation_shock: float = 0.2,
        description: str = ""
    ) -> Dict[str, Any]:
        """
        Create a custom stress scenario.

        Args:
            name: Scenario name
            equity_shock: Shock to equity prices (e.g., -0.20 for 20% decline)
            bond_shock: Shock to bond prices
            commodity_shock: Shock to commodity prices
            tech_shock: Optional separate shock for tech stocks
            volatility_multiplier: How much volatility increases
            correlation_shock: How much correlations increase
            description: Scenario description

        Returns:
            Scenario dictionary that can be used with stress_test()
        """
        scenario = {
            'name': name,
            'description': description or f"Custom scenario: {name}",
            'date': datetime.now(timezone.utc).strftime('%Y-%m-%d'),
            'equity_shock': equity_shock,
            'bond_shock': bond_shock,
            'commodity_shock': commodity_shock,
            'volatility_multiplier': volatility_multiplier,
            'correlation_shock': correlation_shock,
        }

        if tech_shock is not None:
            scenario['tech_shock'] = tech_shock

        return scenario
