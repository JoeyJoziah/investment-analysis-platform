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

Sub-modules (extracted for maintainability):
- var_utils      -- VaR / CVaR calculation functions and VaRMethod / VaRResult
- risk_metrics   -- Drawdown, beta, tracking-error, Sharpe, scoring, decomposition
- risk_stress    -- Historical scenario catalog and stress-test functions

All public names from those sub-modules are re-exported here so that existing
``from backend.utils.risk_manager import ...`` statements continue to work
without any changes.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize  # kept for any future use; preserves original imports

# ---------------------------------------------------------------------------
# Sub-module re-exports (backward-compatibility)
# ---------------------------------------------------------------------------

from backend.utils.var_utils import (          # noqa: F401
    VaRMethod,
    VaRResult,
    var_historical,
    var_parametric,
    var_monte_carlo,
    calculate_var,
    calculate_var_all_methods,
    calculate_cvar,
    calculate_cvar_parametric,
)

from backend.utils.risk_metrics import (       # noqa: F401
    RiskDecomposition,
    calculate_max_drawdown,
    calculate_drawdown_series,
    calculate_all_drawdowns,
    calculate_beta,
    calculate_tracking_error,
    calculate_sortino_ratio,
    calculate_risk_score,
    identify_risk_factors,
    identify_portfolio_risk_factors,
    decompose_portfolio_risk,
    decompose_risk_by_sector,
)

from backend.utils.risk_stress import (        # noqa: F401
    StressTestResult,
    HISTORICAL_SCENARIOS,
    stress_test,
    stress_test_custom,
    stress_test_all_scenarios,
    get_available_scenarios,
    create_custom_scenario,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and dataclasses that live in this module
# ---------------------------------------------------------------------------

class RiskLevel(Enum):
    """Risk level classification."""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


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


# ---------------------------------------------------------------------------
# RiskManager class
# ---------------------------------------------------------------------------

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
    # VaR Calculations (delegate to var_utils)
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
        horizon_days = horizon_days or self.var_horizon_days
        return calculate_var(
            returns, confidence, method, horizon_days, self.monte_carlo_simulations
        )

    def _var_historical(
        self,
        returns: np.ndarray,
        confidence: float,
        horizon_days: int
    ) -> float:
        """Historical simulation VaR (internal helper, delegates to var_utils)."""
        return var_historical(returns, confidence, horizon_days)

    def _var_parametric(
        self,
        returns: np.ndarray,
        confidence: float,
        horizon_days: int
    ) -> float:
        """Parametric VaR (internal helper, delegates to var_utils)."""
        return var_parametric(returns, confidence, horizon_days)

    def _var_monte_carlo(
        self,
        returns: np.ndarray,
        confidence: float,
        horizon_days: int
    ) -> float:
        """Monte Carlo VaR (internal helper, delegates to var_utils)."""
        return var_monte_carlo(returns, confidence, horizon_days, self.monte_carlo_simulations)

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
        horizon_days = horizon_days or self.var_horizon_days
        return calculate_var_all_methods(
            returns, confidence, horizon_days, self.monte_carlo_simulations
        )

    # =========================================================================
    # CVaR / Expected Shortfall (delegate to var_utils)
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
        return calculate_cvar(returns, confidence)

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
        return calculate_cvar_parametric(returns, confidence)

    # =========================================================================
    # Maximum Drawdown (delegate to risk_metrics)
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
        return calculate_max_drawdown(prices)

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
        return calculate_drawdown_series(prices)

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
        return calculate_all_drawdowns(prices, threshold)

    # =========================================================================
    # Beta and Tracking Error (delegate to risk_metrics)
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
        return calculate_beta(returns, benchmark_returns)

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
        return calculate_tracking_error(returns, benchmark_returns, annualize)

    # =========================================================================
    # Stress Testing (delegate to risk_stress)
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
        return stress_test(
            portfolio, scenario, self.max_portfolio_var, asset_betas, sector_mappings
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
        return stress_test_custom(
            portfolio, shocks, self.max_portfolio_var, scenario_name, description
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
        return stress_test_all_scenarios(
            portfolio, self.max_portfolio_var, asset_betas, sector_mappings
        )

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
        return calculate_sortino_ratio(returns, self.risk_free_rate, target_return)

    # =========================================================================
    # Risk Decomposition (delegate to risk_metrics)
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
        return decompose_portfolio_risk(weights, cov_matrix, tickers)

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
        return decompose_risk_by_sector(weights, cov_matrix, tickers, sector_mappings)

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
        var_95 = abs(var_parametric(returns, 0.95, 1))
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
    # Scenario Analysis Utilities (delegate to risk_stress)
    # =========================================================================

    def get_available_scenarios(self) -> List[Dict[str, str]]:
        """Get list of available stress test scenarios."""
        return get_available_scenarios()

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
        return create_custom_scenario(
            name, equity_shock, bond_shock, commodity_shock,
            tech_shock, volatility_multiplier, correlation_shock, description
        )

    # =========================================================================
    # Private helper methods
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
        return calculate_risk_score(volatility, beta, max_drawdown, sharpe_ratio)

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
        return identify_risk_factors(volatility, beta, max_drawdown, sharpe_ratio)

    def _identify_portfolio_risk_factors(
        self,
        volatility: float,
        max_drawdown: float,
        sharpe_ratio: float,
        beta: float,
        hhi: float
    ) -> List[str]:
        """Identify portfolio-level risk factors."""
        return identify_portfolio_risk_factors(
            volatility, max_drawdown, sharpe_ratio, beta, hhi, self.min_sharpe_ratio
        )

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
