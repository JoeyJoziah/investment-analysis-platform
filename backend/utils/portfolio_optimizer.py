"""
Portfolio Optimizer - Markowitz Mean-Variance Optimization

This module provides a full implementation of the PortfolioOptimizer class
using scipy.optimize for mean-variance portfolio optimization (Markowitz).

Features:
- Efficient frontier calculation
- Minimum variance portfolio
- Maximum Sharpe ratio portfolio
- Target return optimization
- Constraint handling (position limits, sector exposure, no short-selling)
- Risk metrics: volatility, Sharpe ratio, Sortino ratio
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.optimize import minimize, OptimizeResult
from scipy import stats
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of portfolio optimization."""
    weights: np.ndarray
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    constraints_satisfied: bool
    optimization_method: str
    iterations: int
    converged: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioMetrics:
    """Comprehensive portfolio metrics."""
    expected_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    n_positions: int
    max_position: float
    hhi: float  # Herfindahl-Hirschman Index for concentration
    beta: Optional[float] = None
    tracking_error: Optional[float] = None


class PortfolioOptimizer:
    """
    Markowitz Mean-Variance Portfolio Optimizer.

    The PortfolioOptimizer is responsible for:
    - Finding optimal portfolio weights given expected returns and covariance
    - Handling various constraints (max position, min position, sector limits)
    - Supporting multiple optimization objectives (max Sharpe, min variance, etc.)
    - Generating efficient frontier
    - Calculating comprehensive risk metrics

    Uses scipy.optimize.minimize with SLSQP method for constrained optimization.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.045,
        default_method: str = 'max_sharpe',
        max_iterations: int = 1000,
        tolerance: float = 1e-10
    ):
        """
        Initialize the portfolio optimizer.

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
            default_method: Default optimization method ('max_sharpe', 'min_variance',
                          'target_return', 'risk_parity')
            max_iterations: Maximum iterations for optimization
            tolerance: Convergence tolerance
        """
        self.risk_free_rate = risk_free_rate
        self.default_method = default_method
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        logger.info(f"PortfolioOptimizer initialized - method={default_method}, rf={risk_free_rate:.2%}")

    def _portfolio_return(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray
    ) -> float:
        """Calculate expected portfolio return."""
        return np.dot(weights, expected_returns)

    def _portfolio_volatility(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray
    ) -> float:
        """Calculate portfolio volatility (standard deviation)."""
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    def _negative_sharpe_ratio(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray
    ) -> float:
        """Calculate negative Sharpe ratio (for minimization)."""
        port_return = self._portfolio_return(weights, expected_returns)
        port_vol = self._portfolio_volatility(weights, cov_matrix)

        if port_vol < 1e-10:
            return 1e10  # Avoid division by zero

        sharpe = (port_return - self.risk_free_rate) / port_vol
        return -sharpe  # Negative for minimization

    def _portfolio_variance(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray
    ) -> float:
        """Calculate portfolio variance (for minimization)."""
        return np.dot(weights.T, np.dot(cov_matrix, weights))

    def _build_constraints(
        self,
        n_assets: int,
        constraints: Optional[Dict[str, Any]] = None,
        expected_returns: Optional[np.ndarray] = None,
        cov_matrix: Optional[np.ndarray] = None,
        target_return: Optional[float] = None
    ) -> List[Dict]:
        """Build scipy constraint dictionaries."""
        constraints = constraints or {}
        scipy_constraints = []

        # Weights must sum to 1
        scipy_constraints.append({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1.0
        })

        # Target return constraint (for efficient frontier)
        if target_return is not None and expected_returns is not None:
            scipy_constraints.append({
                'type': 'eq',
                'fun': lambda w, er=expected_returns, tr=target_return: self._portfolio_return(w, er) - tr
            })

        # Maximum volatility constraint
        max_volatility = constraints.get('max_volatility')
        if max_volatility is not None and cov_matrix is not None:
            scipy_constraints.append({
                'type': 'ineq',
                'fun': lambda w, mv=max_volatility, cm=cov_matrix: mv - self._portfolio_volatility(w, cm)
            })

        # Minimum Sharpe ratio constraint
        min_sharpe = constraints.get('min_sharpe')
        if min_sharpe is not None and expected_returns is not None and cov_matrix is not None:
            scipy_constraints.append({
                'type': 'ineq',
                'fun': lambda w, ms=min_sharpe, er=expected_returns, cm=cov_matrix: (
                    -self._negative_sharpe_ratio(w, er, cm) - ms
                )
            })

        # Sector exposure limits
        sector_limits = constraints.get('sector_limits')
        sector_mapping = constraints.get('sector_mapping')
        if sector_limits is not None and sector_mapping is not None:
            for sector, max_exposure in sector_limits.items():
                sector_indices = [i for i, s in enumerate(sector_mapping) if s == sector]
                if sector_indices:
                    scipy_constraints.append({
                        'type': 'ineq',
                        'fun': lambda w, si=sector_indices, me=max_exposure: me - np.sum(w[si])
                    })

        return scipy_constraints

    def _build_bounds(
        self,
        n_assets: int,
        constraints: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[float, float]]:
        """Build bounds for each asset weight."""
        constraints = constraints or {}

        min_position = constraints.get('min_position', 0.0)
        max_position = constraints.get('max_position', 1.0)
        allow_short = constraints.get('allow_short', False)

        if allow_short:
            lower_bound = -max_position
        else:
            lower_bound = min_position

        # Handle per-asset bounds if provided
        asset_bounds = constraints.get('asset_bounds')
        if asset_bounds is not None and len(asset_bounds) == n_assets:
            return asset_bounds

        return [(lower_bound, max_position) for _ in range(n_assets)]

    def _get_initial_weights(
        self,
        n_assets: int,
        constraints: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """Generate initial weights for optimization."""
        constraints = constraints or {}
        min_position = constraints.get('min_position', 0.0)
        max_position = constraints.get('max_position', 1.0)

        # Start with equal weights, adjusted for constraints
        initial_weight = 1.0 / n_assets
        initial_weight = max(min_position, min(max_position, initial_weight))

        weights = np.array([initial_weight] * n_assets)

        # Normalize to sum to 1
        weights = weights / weights.sum()

        return weights

    async def optimize(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        constraints: Optional[Dict[str, Any]] = None,
        method: Optional[str] = None
    ) -> np.ndarray:
        """
        Optimize portfolio weights given expected returns and covariance matrix.

        Args:
            expected_returns: Array of expected returns for each asset
            cov_matrix: Covariance matrix of asset returns
            constraints: Dictionary of constraints:
                - max_volatility: Maximum portfolio volatility
                - min_sharpe: Minimum Sharpe ratio
                - max_position: Maximum weight for any single position
                - min_position: Minimum weight for any position (if included)
                - sector_limits: Dictionary of sector to max weight
                - sector_mapping: List mapping asset index to sector
                - allow_short: Whether to allow short selling
                - asset_bounds: List of (min, max) tuples per asset
            method: Optimization method ('max_sharpe', 'min_variance', 'risk_parity')

        Returns:
            Array of optimal weights for each asset
        """
        method = method or self.default_method
        n_assets = len(expected_returns)

        if n_assets == 0:
            logger.warning("No assets to optimize")
            return np.array([])

        logger.info(f"Optimizing portfolio with {n_assets} assets using {method} method")

        constraints = constraints or {}

        # Ensure inputs are numpy arrays
        expected_returns = np.asarray(expected_returns, dtype=np.float64)
        cov_matrix = np.asarray(cov_matrix, dtype=np.float64)

        # Validate covariance matrix
        if cov_matrix.shape != (n_assets, n_assets):
            raise ValueError(f"Covariance matrix shape {cov_matrix.shape} doesn't match {n_assets} assets")

        # Build bounds and constraints
        bounds = self._build_bounds(n_assets, constraints)
        scipy_constraints = self._build_constraints(
            n_assets, constraints, expected_returns, cov_matrix
        )
        initial_weights = self._get_initial_weights(n_assets, constraints)

        try:
            if method == 'max_sharpe':
                result = self._optimize_max_sharpe(
                    expected_returns, cov_matrix, bounds, scipy_constraints, initial_weights
                )
            elif method == 'min_variance':
                result = self._optimize_min_variance(
                    cov_matrix, bounds, scipy_constraints, initial_weights
                )
            elif method == 'risk_parity':
                result = self._optimize_risk_parity(
                    cov_matrix, bounds, initial_weights
                )
            else:
                # Default to max Sharpe
                result = self._optimize_max_sharpe(
                    expected_returns, cov_matrix, bounds, scipy_constraints, initial_weights
                )

            if result.success:
                weights = result.x
                # Clean up small weights (noise)
                weights = np.where(np.abs(weights) < 1e-6, 0, weights)
                # Normalize
                if weights.sum() > 0:
                    weights = weights / weights.sum()
                else:
                    weights = initial_weights

                logger.info(f"Optimization converged in {result.nit} iterations")
            else:
                logger.warning(f"Optimization did not converge: {result.message}")
                weights = self._fallback_optimization(
                    expected_returns, cov_matrix, constraints
                )

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            weights = self._fallback_optimization(
                expected_returns, cov_matrix, constraints
            )

        return weights

    def _optimize_max_sharpe(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        bounds: List[Tuple],
        constraints: List[Dict],
        initial_weights: np.ndarray
    ) -> OptimizeResult:
        """Optimize for maximum Sharpe ratio."""
        return minimize(
            self._negative_sharpe_ratio,
            initial_weights,
            args=(expected_returns, cov_matrix),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={
                'maxiter': self.max_iterations,
                'ftol': self.tolerance,
                'disp': False
            }
        )

    def _optimize_min_variance(
        self,
        cov_matrix: np.ndarray,
        bounds: List[Tuple],
        constraints: List[Dict],
        initial_weights: np.ndarray
    ) -> OptimizeResult:
        """Optimize for minimum variance."""
        return minimize(
            self._portfolio_variance,
            initial_weights,
            args=(cov_matrix,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={
                'maxiter': self.max_iterations,
                'ftol': self.tolerance,
                'disp': False
            }
        )

    def _optimize_risk_parity(
        self,
        cov_matrix: np.ndarray,
        bounds: List[Tuple],
        initial_weights: np.ndarray
    ) -> OptimizeResult:
        """
        Optimize for risk parity (equal risk contribution).

        Each asset contributes equally to portfolio risk.
        """
        n_assets = len(initial_weights)
        target_risk = 1.0 / n_assets

        def risk_parity_objective(weights):
            """Objective: minimize squared difference from equal risk contribution."""
            port_var = np.dot(weights.T, np.dot(cov_matrix, weights))
            if port_var < 1e-10:
                return 1e10

            # Marginal contribution to risk
            marginal_contrib = np.dot(cov_matrix, weights)
            # Risk contribution of each asset
            risk_contrib = weights * marginal_contrib / np.sqrt(port_var)
            # Normalized risk contribution
            risk_contrib_norm = risk_contrib / risk_contrib.sum() if risk_contrib.sum() > 0 else risk_contrib

            # Minimize squared difference from target
            return np.sum((risk_contrib_norm - target_risk) ** 2)

        # Sum to 1 constraint
        constraints_rp = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

        return minimize(
            risk_parity_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_rp,
            options={
                'maxiter': self.max_iterations,
                'ftol': self.tolerance,
                'disp': False
            }
        )

    def _fallback_optimization(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        constraints: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Fallback weight allocation when optimization fails.

        Uses inverse volatility weighting with constraint adjustments.
        """
        constraints = constraints or {}
        n_assets = len(expected_returns)
        max_position = constraints.get('max_position', 1.0)
        min_position = constraints.get('min_position', 0.0)

        # Inverse volatility weighting
        volatilities = np.sqrt(np.diag(cov_matrix))
        volatilities = np.maximum(volatilities, 1e-10)  # Avoid division by zero

        # Weight inversely proportional to volatility
        weights = 1.0 / volatilities

        # Only consider assets with positive expected returns
        positive_mask = expected_returns > 0
        if positive_mask.any():
            weights = weights * positive_mask

        # Normalize
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(n_assets) / n_assets

        # Apply position limits
        weights = np.clip(weights, min_position, max_position)

        # Renormalize
        if weights.sum() > 0:
            weights = weights / weights.sum()

        logger.info("Using fallback inverse-volatility weights")
        return weights

    async def get_efficient_frontier(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        n_points: int = 50,
        constraints: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[float, float, np.ndarray]]:
        """
        Calculate the efficient frontier.

        Args:
            expected_returns: Array of expected returns for each asset
            cov_matrix: Covariance matrix
            n_points: Number of points on the frontier
            constraints: Optional constraints to apply

        Returns:
            List of (volatility, return, weights) tuples representing the efficient frontier
        """
        logger.info(f"Calculating efficient frontier with {n_points} points")

        n_assets = len(expected_returns)
        constraints = constraints or {}

        # Ensure inputs are numpy arrays
        expected_returns = np.asarray(expected_returns, dtype=np.float64)
        cov_matrix = np.asarray(cov_matrix, dtype=np.float64)

        # Find min and max achievable returns
        min_var_weights = await self.get_minimum_variance_portfolio(cov_matrix, constraints)
        min_var_return = self._portfolio_return(min_var_weights, expected_returns)

        # Max return is achieved by concentrating in highest return asset (within constraints)
        max_position = constraints.get('max_position', 1.0)
        if max_position >= 1.0:
            max_return = expected_returns.max()
        else:
            # Approximate max return with position limits
            sorted_idx = np.argsort(expected_returns)[::-1]
            temp_weights = np.zeros(n_assets)
            remaining = 1.0
            for idx in sorted_idx:
                w = min(max_position, remaining)
                temp_weights[idx] = w
                remaining -= w
                if remaining <= 0:
                    break
            max_return = self._portfolio_return(temp_weights, expected_returns)

        # Build efficient frontier
        frontier = []
        target_returns = np.linspace(min_var_return, max_return * 0.99, n_points)

        bounds = self._build_bounds(n_assets, constraints)

        for target_ret in target_returns:
            try:
                # Build constraints with target return
                scipy_constraints = self._build_constraints(
                    n_assets, constraints, expected_returns, cov_matrix, target_return=target_ret
                )

                initial_weights = self._get_initial_weights(n_assets, constraints)

                # Minimize variance for given target return
                # Use slightly relaxed tolerance for frontier calculation
                result = minimize(
                    self._portfolio_variance,
                    initial_weights,
                    args=(cov_matrix,),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=scipy_constraints,
                    options={'maxiter': self.max_iterations, 'ftol': 1e-8, 'disp': False}
                )

                # Accept result even if not perfectly converged, as long as constraints are reasonably satisfied
                weights = result.x
                weights = np.where(np.abs(weights) < 1e-6, 0, weights)

                # Check if weights are valid (sum reasonably close to 1)
                weight_sum = weights.sum()
                if weight_sum > 0.9 and weight_sum < 1.1:
                    weights = weights / weight_sum

                    port_return = self._portfolio_return(weights, expected_returns)
                    port_vol = self._portfolio_volatility(weights, cov_matrix)

                    # Only add if return is reasonably close to target
                    if abs(port_return - target_ret) < 0.05:  # Within 5% of target
                        frontier.append((port_vol, port_return, weights.copy()))

            except Exception as e:
                logger.debug(f"Failed to compute frontier point for target return {target_ret:.4f}: {e}")
                continue

        # Sort by volatility
        frontier.sort(key=lambda x: x[0])

        logger.info(f"Computed {len(frontier)} points on efficient frontier")
        return frontier

    async def get_minimum_variance_portfolio(
        self,
        cov_matrix: np.ndarray,
        constraints: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Calculate the minimum variance portfolio.

        Args:
            cov_matrix: Covariance matrix
            constraints: Optional constraints

        Returns:
            Array of weights for minimum variance portfolio
        """
        n_assets = cov_matrix.shape[0]
        constraints = constraints or {}

        bounds = self._build_bounds(n_assets, constraints)
        scipy_constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        initial_weights = self._get_initial_weights(n_assets, constraints)

        result = minimize(
            self._portfolio_variance,
            initial_weights,
            args=(cov_matrix,),
            method='SLSQP',
            bounds=bounds,
            constraints=scipy_constraints,
            options={'maxiter': self.max_iterations, 'ftol': self.tolerance, 'disp': False}
        )

        if result.success:
            weights = result.x
            weights = np.where(np.abs(weights) < 1e-6, 0, weights)
            if weights.sum() > 0:
                weights = weights / weights.sum()
            return weights
        else:
            logger.warning("Minimum variance optimization did not converge")
            return initial_weights

    async def get_max_sharpe_portfolio(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        constraints: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Calculate the maximum Sharpe ratio portfolio (tangency portfolio).

        Args:
            expected_returns: Array of expected returns
            cov_matrix: Covariance matrix
            constraints: Optional constraints

        Returns:
            Array of weights for maximum Sharpe ratio portfolio
        """
        return await self.optimize(
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            constraints=constraints,
            method='max_sharpe'
        )

    async def optimize_for_target_return(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        target_return: float,
        constraints: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Optimize portfolio for a specific target return (minimize variance).

        Args:
            expected_returns: Array of expected returns
            cov_matrix: Covariance matrix
            target_return: Target portfolio return
            constraints: Optional constraints

        Returns:
            Array of optimal weights achieving target return with minimum variance
        """
        n_assets = len(expected_returns)
        constraints = constraints or {}

        bounds = self._build_bounds(n_assets, constraints)
        scipy_constraints = self._build_constraints(
            n_assets, constraints, expected_returns, cov_matrix, target_return=target_return
        )
        initial_weights = self._get_initial_weights(n_assets, constraints)

        result = minimize(
            self._portfolio_variance,
            initial_weights,
            args=(cov_matrix,),
            method='SLSQP',
            bounds=bounds,
            constraints=scipy_constraints,
            options={'maxiter': self.max_iterations, 'ftol': self.tolerance, 'disp': False}
        )

        if result.success:
            weights = result.x
            weights = np.where(np.abs(weights) < 1e-6, 0, weights)
            if weights.sum() > 0:
                weights = weights / weights.sum()
            return weights
        else:
            logger.warning(f"Target return optimization did not converge for target={target_return:.4f}")
            return await self.optimize(expected_returns, cov_matrix, constraints, method='max_sharpe')

    def _calculate_sortino_ratio(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        returns_history: Optional[np.ndarray] = None,
        downside_returns: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate Sortino ratio (risk-adjusted return using downside deviation).

        Args:
            weights: Portfolio weights
            expected_returns: Expected returns
            returns_history: Historical returns matrix (optional)
            downside_returns: Pre-computed downside returns (optional)

        Returns:
            Sortino ratio
        """
        port_return = self._portfolio_return(weights, expected_returns)

        if downside_returns is not None:
            # Use provided downside returns
            portfolio_downside = np.dot(downside_returns, weights)
            downside_std = np.std(portfolio_downside[portfolio_downside < 0])
        elif returns_history is not None:
            # Calculate from history
            portfolio_returns = np.dot(returns_history, weights)
            negative_returns = portfolio_returns[portfolio_returns < 0]
            downside_std = np.std(negative_returns) if len(negative_returns) > 0 else 0.01
        else:
            # Approximate: assume 60% of volatility is downside
            downside_std = 0.01  # Default small value

        if downside_std < 1e-10:
            return 0.0

        return (port_return - self.risk_free_rate) / downside_std

    async def calculate_portfolio_metrics(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        returns_history: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive portfolio metrics for given weights.

        Args:
            weights: Portfolio weights
            expected_returns: Expected returns
            cov_matrix: Covariance matrix
            returns_history: Optional historical returns for advanced metrics

        Returns:
            Dictionary of portfolio metrics
        """
        portfolio_return = self._portfolio_return(weights, expected_returns)
        portfolio_vol = self._portfolio_volatility(weights, cov_matrix)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0

        # Sortino ratio
        sortino_ratio = self._calculate_sortino_ratio(
            weights, expected_returns, returns_history
        )

        # Additional metrics if we have historical data
        var_95 = 0.0
        cvar_95 = 0.0
        max_drawdown = 0.0

        if returns_history is not None:
            portfolio_returns = np.dot(returns_history, weights)

            # Value at Risk (95%)
            var_95 = np.percentile(portfolio_returns, 5)

            # Conditional VaR (Expected Shortfall)
            cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean() if any(portfolio_returns <= var_95) else var_95

            # Max Drawdown
            cumulative = np.cumprod(1 + portfolio_returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = (cumulative - running_max) / running_max
            max_drawdown = np.min(drawdowns)
        else:
            # Parametric VaR approximation
            var_95 = portfolio_return - 1.645 * portfolio_vol
            cvar_95 = portfolio_return - 2.063 * portfolio_vol  # Approximation

        return {
            'expected_return': float(portfolio_return),
            'volatility': float(portfolio_vol),
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'var_95': float(var_95),
            'cvar_95': float(cvar_95),
            'max_drawdown': float(max_drawdown),
            'n_positions': int(np.sum(weights > 0.01)),
            'max_position': float(weights.max()),
            'hhi': float(np.sum(weights ** 2))  # Herfindahl-Hirschman Index
        }

    async def rebalance_portfolio(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        transaction_cost: float = 0.001,
        min_trade_size: float = 0.005
    ) -> Dict[str, Any]:
        """
        Calculate rebalancing trades and costs.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            transaction_cost: Transaction cost as fraction of trade value
            min_trade_size: Minimum trade size (weights below this are skipped)

        Returns:
            Dictionary with rebalancing details
        """
        trades = target_weights - current_weights

        # Filter out tiny trades
        trades = np.where(np.abs(trades) < min_trade_size, 0, trades)

        # Calculate trade metrics
        buy_trades = np.maximum(trades, 0)
        sell_trades = np.maximum(-trades, 0)

        total_buy = buy_trades.sum()
        total_sell = sell_trades.sum()
        turnover = (total_buy + total_sell) / 2  # One-way turnover

        total_cost = (total_buy + total_sell) * transaction_cost

        return {
            'trades': trades.tolist(),
            'buy_trades': buy_trades.tolist(),
            'sell_trades': sell_trades.tolist(),
            'total_buy_value': float(total_buy),
            'total_sell_value': float(total_sell),
            'turnover': float(turnover),
            'transaction_costs': float(total_cost),
            'n_trades': int(np.sum(np.abs(trades) > min_trade_size)),
            'net_trade': float(np.sum(trades))  # Should be ~0 for fully invested
        }

    async def optimize_with_tracking_error(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        benchmark_weights: np.ndarray,
        max_tracking_error: float = 0.05,
        constraints: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Optimize portfolio with tracking error constraint relative to benchmark.

        Args:
            expected_returns: Expected returns
            cov_matrix: Covariance matrix
            benchmark_weights: Benchmark portfolio weights
            max_tracking_error: Maximum allowed tracking error
            constraints: Additional constraints

        Returns:
            Optimal weights with tracking error constraint
        """
        n_assets = len(expected_returns)
        constraints = constraints or {}

        def tracking_error_constraint(weights):
            """Calculate tracking error relative to benchmark."""
            diff = weights - benchmark_weights
            te = np.sqrt(np.dot(diff.T, np.dot(cov_matrix, diff)))
            return max_tracking_error - te

        bounds = self._build_bounds(n_assets, constraints)
        scipy_constraints = self._build_constraints(n_assets, constraints, expected_returns, cov_matrix)
        scipy_constraints.append({'type': 'ineq', 'fun': tracking_error_constraint})

        initial_weights = benchmark_weights.copy()  # Start from benchmark

        result = minimize(
            self._negative_sharpe_ratio,
            initial_weights,
            args=(expected_returns, cov_matrix),
            method='SLSQP',
            bounds=bounds,
            constraints=scipy_constraints,
            options={'maxiter': self.max_iterations, 'ftol': self.tolerance, 'disp': False}
        )

        if result.success:
            weights = result.x
            weights = np.where(np.abs(weights) < 1e-6, 0, weights)
            if weights.sum() > 0:
                weights = weights / weights.sum()
            return weights
        else:
            logger.warning("Tracking error constrained optimization did not converge")
            return benchmark_weights

    def calculate_risk_contribution(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Calculate each asset's contribution to portfolio risk.

        Args:
            weights: Portfolio weights
            cov_matrix: Covariance matrix

        Returns:
            Array of risk contributions (sum to portfolio variance)
        """
        port_var = np.dot(weights.T, np.dot(cov_matrix, weights))
        if port_var < 1e-10:
            return np.zeros_like(weights)

        # Marginal contribution to risk
        marginal_contrib = np.dot(cov_matrix, weights)

        # Risk contribution = weight * marginal contribution
        risk_contrib = weights * marginal_contrib

        return risk_contrib

    def calculate_diversification_ratio(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray
    ) -> float:
        """
        Calculate portfolio diversification ratio.

        DR = weighted average volatility / portfolio volatility
        Higher ratio means better diversification.

        Args:
            weights: Portfolio weights
            cov_matrix: Covariance matrix

        Returns:
            Diversification ratio (>=1, higher is better)
        """
        # Individual volatilities
        vols = np.sqrt(np.diag(cov_matrix))

        # Weighted average volatility
        weighted_avg_vol = np.dot(weights, vols)

        # Portfolio volatility
        port_vol = self._portfolio_volatility(weights, cov_matrix)

        if port_vol < 1e-10:
            return 1.0

        return weighted_avg_vol / port_vol
