"""
Black-Litterman Model Implementation

Implements the Black-Litterman asset allocation model that combines
investor views with market equilibrium.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BlackLittermanResult:
    """Result of Black-Litterman optimization."""
    weights: Dict[str, float]
    expected_returns: Dict[str, float]
    posterior_covariance: np.ndarray
    implied_equilibrium_returns: Dict[str, float]


class BlackLittermanOptimizer:
    """
    Black-Litterman asset allocation model.

    Combines market equilibrium returns with investor views to produce
    optimal portfolio weights.
    """

    def __init__(
        self,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        risk_free_rate: float = 0.02
    ):
        """
        Initialize the Black-Litterman optimizer.

        Args:
            risk_aversion: Market risk aversion coefficient
            tau: Scaling factor for uncertainty in prior
            risk_free_rate: Risk-free rate for calculations
        """
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.risk_free_rate = risk_free_rate

    def optimize(
        self,
        returns: pd.DataFrame,
        market_caps: Dict[str, float],
        views: Optional[List[Dict[str, Any]]] = None,
        view_confidences: Optional[List[float]] = None
    ) -> BlackLittermanResult:
        """
        Run Black-Litterman optimization.

        Args:
            returns: DataFrame of historical returns
            market_caps: Market capitalizations by asset
            views: List of investor views
            view_confidences: Confidence levels for each view (0-1)

        Returns:
            BlackLittermanResult with optimal weights and metrics
        """
        assets = returns.columns.tolist()
        n_assets = len(assets)

        # Calculate covariance matrix
        cov_matrix = returns.cov() * 252

        # Calculate market cap weights
        total_cap = sum(market_caps.values())
        market_weights = np.array([market_caps.get(a, 0) / total_cap for a in assets])

        # Calculate implied equilibrium returns (reverse optimization)
        implied_returns = self.risk_aversion * cov_matrix.values @ market_weights

        # If no views, return market weights
        if views is None or len(views) == 0:
            return BlackLittermanResult(
                weights=dict(zip(assets, market_weights)),
                expected_returns=dict(zip(assets, implied_returns)),
                posterior_covariance=cov_matrix.values,
                implied_equilibrium_returns=dict(zip(assets, implied_returns))
            )

        # Build view matrices (simplified)
        P = np.zeros((len(views), n_assets))
        Q = np.zeros(len(views))

        for i, view in enumerate(views):
            asset_idx = assets.index(view.get('asset', assets[0]))
            P[i, asset_idx] = 1
            Q[i] = view.get('expected_return', implied_returns[asset_idx])

        # View uncertainty (omega)
        if view_confidences:
            omega = np.diag([(1 - c) * 0.1 for c in view_confidences])
        else:
            omega = np.diag([0.05] * len(views))

        # Black-Litterman posterior
        tau_sigma = self.tau * cov_matrix.values
        M = np.linalg.inv(np.linalg.inv(tau_sigma) + P.T @ np.linalg.inv(omega) @ P)
        posterior_returns = M @ (np.linalg.inv(tau_sigma) @ implied_returns + P.T @ np.linalg.inv(omega) @ Q)

        # Calculate posterior weights
        posterior_weights = np.linalg.inv(self.risk_aversion * cov_matrix.values) @ posterior_returns

        # Normalize weights
        posterior_weights = posterior_weights / np.sum(np.abs(posterior_weights))

        return BlackLittermanResult(
            weights=dict(zip(assets, posterior_weights)),
            expected_returns=dict(zip(assets, posterior_returns)),
            posterior_covariance=M,
            implied_equilibrium_returns=dict(zip(assets, implied_returns))
        )

    def calculate_implied_returns(
        self,
        returns: pd.DataFrame,
        market_caps: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate implied equilibrium returns from market caps.

        Args:
            returns: Historical returns DataFrame
            market_caps: Market capitalizations

        Returns:
            Dictionary of implied equilibrium returns
        """
        assets = returns.columns.tolist()
        cov_matrix = returns.cov() * 252

        total_cap = sum(market_caps.values())
        market_weights = np.array([market_caps.get(a, 0) / total_cap for a in assets])

        implied_returns = self.risk_aversion * cov_matrix.values @ market_weights

        return dict(zip(assets, implied_returns))
