"""
Risk Metrics Utilities

Provides standalone functions for computing common risk metrics:
- Maximum drawdown and drawdown series analysis
- Beta and alpha calculation relative to a benchmark
- Tracking error and information ratio
- Sharpe ratio and Sortino ratio
- Composite risk scoring and classification
- Portfolio risk factor identification

These functions are pure-math helpers with no external dependencies beyond
NumPy, Pandas, and SciPy.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RiskDecomposition:
    """Risk decomposition by asset or sector."""
    total_risk: float
    marginal_contributions: Dict[str, float]
    percentage_contributions: Dict[str, float]
    diversification_benefit: float


# ---------------------------------------------------------------------------
# Drawdown analysis
# ---------------------------------------------------------------------------

def calculate_max_drawdown(
    prices: Union[np.ndarray, pd.Series]
) -> Tuple[float, int, int]:
    """
    Calculate Maximum Drawdown and identify the drawdown period.

    Args:
        prices: Array or Series of prices

    Returns:
        Tuple of (max_drawdown, peak_index, trough_index)
    """
    if isinstance(prices, pd.Series):
        prices = prices.values
    prices = np.asarray(prices)

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
    prices: Union[np.ndarray, pd.Series]
) -> pd.DataFrame:
    """
    Calculate the full drawdown series with details.

    Args:
        prices: Array or Series of prices

    Returns:
        DataFrame with drawdown metrics at each point
    """
    if isinstance(prices, pd.Series):
        prices = prices.values
    prices = np.asarray(prices)

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
    if isinstance(prices, pd.Series):
        prices = prices.values
    prices = np.asarray(prices)

    dd_series = calculate_drawdown_series(prices)

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


# ---------------------------------------------------------------------------
# Beta, correlation, and tracking error
# ---------------------------------------------------------------------------

def calculate_beta(
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
    if isinstance(returns, pd.Series):
        returns = returns.values
    if isinstance(benchmark_returns, pd.Series):
        benchmark_returns = benchmark_returns.values
    returns = np.asarray(returns)
    benchmark_returns = np.asarray(benchmark_returns)

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
    if isinstance(returns, pd.Series):
        returns = returns.values
    if isinstance(benchmark_returns, pd.Series):
        benchmark_returns = benchmark_returns.values
    returns = np.asarray(returns)
    benchmark_returns = np.asarray(benchmark_returns)

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


# ---------------------------------------------------------------------------
# Sharpe and Sortino ratios
# ---------------------------------------------------------------------------

def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.045,
    target_return: float = 0.0
) -> float:
    """
    Calculate Sortino ratio (penalizes only downside volatility).

    Args:
        returns: Array of daily returns
        risk_free_rate: Annual risk-free rate
        target_return: Minimum acceptable return (daily)

    Returns:
        Annualized Sortino ratio
    """
    excess_returns = returns - target_return
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        return float('inf')

    downside_std = np.sqrt(np.mean(downside_returns ** 2))

    if downside_std == 0:
        return float('inf')

    mean_excess = np.mean(returns) - risk_free_rate / 252
    sortino = mean_excess * 252 / (downside_std * np.sqrt(252))

    return float(sortino)


# ---------------------------------------------------------------------------
# Risk scoring and classification
# ---------------------------------------------------------------------------

def calculate_risk_score(
    volatility: float,
    beta: float,
    max_drawdown: float,
    sharpe_ratio: float
) -> float:
    """
    Calculate composite risk score (0-1, higher is riskier).

    Args:
        volatility: Annualized volatility
        beta: Market beta
        max_drawdown: Maximum drawdown (negative value)
        sharpe_ratio: Sharpe ratio

    Returns:
        Risk score between 0 and 1
    """
    # Normalize components
    vol_score = min(volatility / 0.5, 1.0)           # 50% annual vol = max
    beta_score = min(abs(beta - 1) / 1.0, 1.0)       # Distance from market beta
    dd_score = min(abs(max_drawdown) / 0.3, 1.0)     # 30% drawdown = max
    sharpe_score = max(0, 1 - sharpe_ratio / 2)      # Higher Sharpe = lower risk

    # Weighted average
    risk_score = (
        0.35 * vol_score +
        0.15 * beta_score +
        0.25 * dd_score +
        0.25 * sharpe_score
    )

    return min(1.0, max(0.0, risk_score))


def identify_risk_factors(
    volatility: float,
    beta: float,
    max_drawdown: float,
    sharpe_ratio: float
) -> List[str]:
    """
    Identify specific risk factors for a single asset.

    Args:
        volatility: Annualized volatility
        beta: Market beta
        max_drawdown: Maximum drawdown (negative value)
        sharpe_ratio: Sharpe ratio

    Returns:
        List of human-readable risk factor descriptions
    """
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


def identify_portfolio_risk_factors(
    volatility: float,
    max_drawdown: float,
    sharpe_ratio: float,
    beta: float,
    hhi: float,
    min_sharpe_ratio: float = 0.5
) -> List[str]:
    """
    Identify portfolio-level risk factors.

    Args:
        volatility: Annualized portfolio volatility
        max_drawdown: Maximum portfolio drawdown (negative value)
        sharpe_ratio: Portfolio Sharpe ratio
        beta: Portfolio beta
        hhi: Herfindahl-Hirschman Index (concentration measure)
        min_sharpe_ratio: Minimum acceptable Sharpe ratio

    Returns:
        List of human-readable risk factor descriptions
    """
    factors = []

    if volatility > 0.25:
        factors.append(f"Elevated portfolio volatility ({volatility:.0%} annualized)")
    if max_drawdown < -0.15:
        factors.append(f"Significant drawdown risk ({max_drawdown:.0%})")
    if sharpe_ratio < min_sharpe_ratio:
        factors.append(
            f"Below minimum Sharpe ratio ({sharpe_ratio:.2f} < {min_sharpe_ratio})"
        )
    if beta > 1.2:
        factors.append(f"High market exposure (beta: {beta:.2f})")
    if hhi > 0.2:
        factors.append(f"Concentrated portfolio (HHI: {hhi:.2f})")

    return factors


# ---------------------------------------------------------------------------
# Portfolio risk decomposition
# ---------------------------------------------------------------------------

def decompose_portfolio_risk(
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
    asset_decomp = decompose_portfolio_risk(weights, cov_matrix, tickers)

    sector_contributions: Dict[str, float] = {}
    for ticker, pct_contrib in asset_decomp.percentage_contributions.items():
        sector = sector_mappings.get(ticker, 'Unknown')
        sector_contributions[sector] = (
            sector_contributions.get(sector, 0) + pct_contrib
        )

    return sector_contributions
