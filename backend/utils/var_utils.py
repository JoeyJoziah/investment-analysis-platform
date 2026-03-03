"""
VaR (Value at Risk) Calculation Utilities

Provides VaR and CVaR calculations using three methods:
- Historical simulation
- Parametric (Variance-Covariance)
- Monte Carlo simulation

Also includes Conditional Value at Risk (Expected Shortfall) calculations.
"""

import logging
from typing import Dict, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class VaRMethod(Enum):
    """VaR calculation methods."""
    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"


@dataclass
class VaRResult:
    """Result of a VaR calculation."""
    var_value: float
    confidence_level: float
    method: VaRMethod
    horizon_days: int
    additional_metrics: Dict[str, Any] = field(default_factory=dict)


def var_historical(
    returns: np.ndarray,
    confidence: float,
    horizon_days: int
) -> float:
    """
    Historical simulation VaR.

    Uses the empirical distribution of returns to estimate VaR.

    Args:
        returns: Array of daily returns
        confidence: Confidence level (e.g., 0.95)
        horizon_days: Time horizon in days

    Returns:
        VaR value (negative indicates loss)
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


def var_parametric(
    returns: np.ndarray,
    confidence: float,
    horizon_days: int
) -> float:
    """
    Parametric (Variance-Covariance) VaR.

    Assumes returns are normally distributed.

    Args:
        returns: Array of daily returns
        confidence: Confidence level (e.g., 0.95)
        horizon_days: Time horizon in days

    Returns:
        VaR value (negative indicates loss)
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


def var_monte_carlo(
    returns: np.ndarray,
    confidence: float,
    horizon_days: int,
    n_simulations: int = 10000
) -> float:
    """
    Monte Carlo VaR.

    Simulates many possible return paths and estimates VaR from the
    resulting distribution.

    Args:
        returns: Array of daily returns
        confidence: Confidence level (e.g., 0.95)
        horizon_days: Time horizon in days
        n_simulations: Number of Monte Carlo simulations

    Returns:
        VaR value (negative indicates loss)
    """
    mean_return = np.mean(returns)
    std_return = np.std(returns)

    # Simulate returns
    np.random.seed(42)  # For reproducibility
    simulated_returns = np.random.normal(
        mean_return * horizon_days,
        std_return * np.sqrt(horizon_days),
        n_simulations
    )

    # VaR from simulated distribution
    var_percentile = (1 - confidence) * 100
    var_value = np.percentile(simulated_returns, var_percentile)

    return float(var_value)


def calculate_var(
    returns: Union[np.ndarray, pd.Series],
    confidence: float = 0.95,
    method: str = 'historical',
    horizon_days: int = 1,
    n_simulations: int = 10000
) -> VaRResult:
    """
    Calculate Value at Risk using the specified method.

    Args:
        returns: Array or Series of returns (daily)
        confidence: Confidence level (e.g., 0.95 for 95% VaR)
        method: 'historical', 'parametric', or 'monte_carlo'
        horizon_days: Time horizon in days
        n_simulations: Number of Monte Carlo simulations (Monte Carlo only)

    Returns:
        VaRResult with VaR value and additional metrics
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    returns = np.asarray(returns)

    if len(returns) < 30:
        logger.warning("Insufficient data for VaR calculation, using parametric method")
        method = 'parametric'

    method_enum = VaRMethod(method.lower())

    if method_enum == VaRMethod.HISTORICAL:
        var_value = var_historical(returns, confidence, horizon_days)
    elif method_enum == VaRMethod.PARAMETRIC:
        var_value = var_parametric(returns, confidence, horizon_days)
    elif method_enum == VaRMethod.MONTE_CARLO:
        var_value = var_monte_carlo(returns, confidence, horizon_days, n_simulations)
    else:
        raise ValueError(f"Unknown VaR method: {method}")

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


def calculate_var_all_methods(
    returns: Union[np.ndarray, pd.Series],
    confidence: float = 0.95,
    horizon_days: int = 1,
    n_simulations: int = 10000
) -> Dict[str, VaRResult]:
    """
    Calculate VaR using all three methods for comparison.

    Args:
        returns: Array or Series of returns
        confidence: Confidence level
        horizon_days: Time horizon in days
        n_simulations: Number of Monte Carlo simulations

    Returns:
        Dictionary mapping method name to VaRResult
    """
    results = {}
    for method in ['historical', 'parametric', 'monte_carlo']:
        try:
            results[method] = calculate_var(
                returns, confidence, method, horizon_days, n_simulations
            )
        except Exception as e:
            logger.error(f"Error calculating {method} VaR: {e}")

    return results


def calculate_cvar(
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
    if isinstance(returns, pd.Series):
        returns = returns.values
    returns = np.asarray(returns)

    # Calculate VaR threshold
    var_threshold = var_historical(returns, confidence, 1)

    # CVaR is the mean of returns below VaR
    tail_returns = returns[returns <= var_threshold]

    if len(tail_returns) == 0:
        # No returns below VaR threshold, use VaR as CVaR
        return var_threshold

    cvar = np.mean(tail_returns)
    return float(cvar)


def calculate_cvar_parametric(
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
    if isinstance(returns, pd.Series):
        returns = returns.values
    returns = np.asarray(returns)

    mean_return = np.mean(returns)
    std_return = np.std(returns)

    # For normal distribution, CVaR has a closed-form solution
    alpha = 1 - confidence
    z_alpha = stats.norm.ppf(alpha)
    pdf_z = stats.norm.pdf(z_alpha)

    cvar = mean_return - std_return * pdf_z / alpha

    return float(cvar)
