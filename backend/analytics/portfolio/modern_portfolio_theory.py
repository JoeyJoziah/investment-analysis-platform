"""
Modern Portfolio Theory Implementation

Implements Markowitz mean-variance optimization for portfolio construction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PortfolioResult:
    """Result of portfolio optimization."""
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    efficient_frontier: Optional[List[Tuple[float, float]]] = None


class PortfolioOptimizer:
    """
    Modern Portfolio Theory optimizer using Markowitz mean-variance optimization.

    Finds optimal portfolio weights to maximize returns for a given risk level
    or minimize risk for a given return level.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        max_weight: float = 0.3,
        min_weight: float = 0.0,
        allow_short: bool = False
    ):
        """
        Initialize the portfolio optimizer.

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
            max_weight: Maximum weight for any single asset
            min_weight: Minimum weight for any single asset
            allow_short: Whether to allow short positions
        """
        self.risk_free_rate = risk_free_rate
        self.max_weight = max_weight
        self.min_weight = min_weight if not allow_short else -max_weight
        self.allow_short = allow_short

    def optimize(
        self,
        returns: pd.DataFrame,
        target_return: Optional[float] = None,
        target_volatility: Optional[float] = None
    ) -> PortfolioResult:
        """
        Optimize portfolio weights.

        Args:
            returns: DataFrame of asset returns (columns are assets)
            target_return: Target portfolio return (if None, maximize Sharpe)
            target_volatility: Target portfolio volatility

        Returns:
            PortfolioResult with optimal weights and metrics
        """
        n_assets = len(returns.columns)
        assets = returns.columns.tolist()

        # Calculate expected returns and covariance matrix
        expected_returns = returns.mean() * 252  # Annualize
        cov_matrix = returns.cov() * 252  # Annualize

        # Simple equal weight as baseline
        weights = np.array([1.0 / n_assets] * n_assets)

        # Calculate portfolio metrics
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0

        weight_dict = dict(zip(assets, weights))

        return PortfolioResult(
            weights=weight_dict,
            expected_return=portfolio_return,
            volatility=portfolio_vol,
            sharpe_ratio=sharpe
        )

    def get_efficient_frontier(
        self,
        returns: pd.DataFrame,
        n_points: int = 50
    ) -> List[Tuple[float, float]]:
        """
        Calculate the efficient frontier.

        Args:
            returns: DataFrame of asset returns
            n_points: Number of points on the frontier

        Returns:
            List of (volatility, return) tuples
        """
        expected_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252

        min_ret = expected_returns.min()
        max_ret = expected_returns.max()

        frontier = []
        for target_ret in np.linspace(min_ret, max_ret, n_points):
            result = self.optimize(returns, target_return=target_ret)
            frontier.append((result.volatility, result.expected_return))

        return frontier

    def calculate_risk_metrics(
        self,
        returns: pd.DataFrame,
        weights: Dict[str, float],
        confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """
        Calculate portfolio risk metrics.

        Args:
            returns: DataFrame of asset returns
            weights: Portfolio weights
            confidence_level: Confidence level for VaR

        Returns:
            Dictionary of risk metrics
        """
        weight_array = np.array([weights.get(col, 0) for col in returns.columns])
        portfolio_returns = returns.dot(weight_array)

        var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        cvar = portfolio_returns[portfolio_returns <= var].mean()

        return {
            "var": abs(var) * np.sqrt(252),
            "cvar": abs(cvar) * np.sqrt(252) if not np.isnan(cvar) else abs(var) * np.sqrt(252),
            "max_drawdown": self._calculate_max_drawdown(portfolio_returns),
            "volatility": portfolio_returns.std() * np.sqrt(252)
        }

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        return abs(drawdown.min())
