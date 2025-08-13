"""
Risk Parity and Equal Risk Contribution Optimization

Advanced risk parity strategies including:
- Equal Risk Contribution (ERC) optimization
- Risk budgeting with custom allocations
- Hierarchical Risk Parity (HRP)
- Nested Clustered Optimization (NCO)
- Dynamic risk parity with regime detection
- Factor-based risk parity
- Volatility targeting
- Maximum diversification

Focuses on risk-based portfolio construction rather than return optimization
"""

import logging
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy import optimize, cluster
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy
from sklearn.covariance import LedoitWolf
import cvxpy as cp

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class RiskParityOptimizer:
    """
    Advanced risk parity optimization with multiple strategies
    
    Features:
    - Equal Risk Contribution (ERC)
    - Risk budgeting with custom targets
    - Hierarchical Risk Parity (HRP)
    - Nested Clustered Optimization (NCO)
    - Dynamic risk parity
    - Factor-based risk parity
    - Volatility targeting strategies
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize risk parity optimizer
        
        Args:
            risk_free_rate: Risk-free rate for calculations
        """
        self.risk_free_rate = risk_free_rate
    
    def optimize_equal_risk_contribution(
        self,
        returns: pd.DataFrame,
        risk_budget: Optional[Dict] = None,
        bounds: Tuple[float, float] = (0.01, 0.40),
        method: str = 'analytical',
        max_iter: int = 1000,
        tolerance: float = 1e-8
    ) -> Dict:
        """
        Optimize for Equal Risk Contribution or custom risk budgets
        
        Args:
            returns: Historical returns DataFrame
            risk_budget: Dict of asset: target_risk_contribution (if None, equal weights)
            bounds: Weight bounds for each asset
            method: 'analytical' or 'numerical'
            max_iter: Maximum iterations for optimization
            tolerance: Convergence tolerance
            
        Returns:
            Optimization results
        """
        try:
            # Estimate covariance matrix
            cov_estimator = LedoitWolf()
            S = cov_estimator.fit(returns).covariance_ * 252
            n_assets = len(returns.columns)
            
            # Set risk budget
            if risk_budget is None:
                risk_budget_array = np.ones(n_assets) / n_assets  # Equal risk
            else:
                risk_budget_array = np.array([
                    risk_budget.get(asset, 1/n_assets) for asset in returns.columns
                ])
                risk_budget_array = risk_budget_array / risk_budget_array.sum()
            
            if method == 'analytical':
                weights = self._solve_erc_analytical(S, risk_budget_array, bounds, tolerance)
            else:
                weights = self._solve_erc_numerical(S, risk_budget_array, bounds, max_iter, tolerance)
            
            if weights is None:
                return {'error': 'ERC optimization failed'}
            
            # Calculate portfolio metrics
            portfolio_return = np.dot(weights, returns.mean() * 252)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(S, weights)))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol
            
            # Calculate risk contributions
            marginal_contrib = np.dot(S, weights) / portfolio_vol
            risk_contrib = weights * marginal_contrib
            risk_contrib_pct = risk_contrib / risk_contrib.sum()
            
            # Calculate diversification ratio
            weighted_vol = np.dot(weights, np.sqrt(np.diag(S)))
            diversification_ratio = weighted_vol / portfolio_vol
            
            return {
                'weights': dict(zip(returns.columns, weights)),
                'risk_contributions': dict(zip(returns.columns, risk_contrib_pct)),
                'target_risk_budget': dict(zip(returns.columns, risk_budget_array)),
                'risk_budget_deviation': dict(zip(returns.columns, risk_contrib_pct - risk_budget_array)),
                'portfolio_return': portfolio_return,
                'portfolio_volatility': portfolio_vol,
                'sharpe_ratio': sharpe_ratio,
                'diversification_ratio': diversification_ratio,
                'effective_number_of_bets': 1 / np.sum(risk_contrib_pct ** 2)
            }
            
        except Exception as e:
            logger.error(f"Error in ERC optimization: {e}")
            return {'error': str(e)}
    
    def _solve_erc_analytical(
        self,
        S: np.ndarray,
        risk_budget: np.ndarray,
        bounds: Tuple[float, float],
        tolerance: float
    ) -> Optional[np.ndarray]:
        """Solve ERC using analytical approach with Newton-Raphson"""
        try:
            n_assets = S.shape[0]
            
            # Initial guess (inverse volatility weights)
            vol = np.sqrt(np.diag(S))
            x = 1 / vol
            x = x / x.sum()
            
            # Ensure within bounds
            x = np.clip(x, bounds[0], bounds[1])
            x = x / x.sum()
            
            for iteration in range(1000):
                # Calculate portfolio volatility and derivatives
                portfolio_vol = np.sqrt(np.dot(x, np.dot(S, x)))
                
                if portfolio_vol == 0:
                    break
                
                # Risk contributions
                marginal_contrib = np.dot(S, x) / portfolio_vol
                risk_contrib = x * marginal_contrib
                risk_contrib_pct = risk_contrib / risk_contrib.sum()
                
                # Check convergence
                error = np.sum((risk_contrib_pct - risk_budget) ** 2)
                if error < tolerance:
                    break
                
                # Newton-Raphson update
                # Gradient of risk contribution deviation
                grad = self._erc_gradient(x, S, risk_budget)
                
                # Hessian approximation (simplified)
                hess = self._erc_hessian_approx(x, S)
                
                try:
                    # Newton step
                    delta = np.linalg.solve(hess, -grad)
                    
                    # Line search with constraints
                    alpha = 1.0
                    for _ in range(10):
                        x_new = x + alpha * delta
                        x_new = np.clip(x_new, bounds[0], bounds[1])
                        x_new = x_new / x_new.sum()
                        
                        # Check if improvement
                        new_risk_contrib = self._calculate_risk_contributions(x_new, S)
                        new_error = np.sum((new_risk_contrib - risk_budget) ** 2)
                        
                        if new_error < error:
                            x = x_new
                            break
                        
                        alpha *= 0.5
                    else:
                        # If line search fails, use small step
                        x = x - 0.01 * grad
                        x = np.clip(x, bounds[0], bounds[1])
                        x = x / x.sum()
                
                except np.linalg.LinAlgError:
                    # If Hessian is singular, use gradient descent
                    x = x - 0.01 * grad
                    x = np.clip(x, bounds[0], bounds[1])
                    x = x / x.sum()
            
            return x
            
        except Exception as e:
            logger.error(f"Error in analytical ERC solving: {e}")
            return None
    
    def _solve_erc_numerical(
        self,
        S: np.ndarray,
        risk_budget: np.ndarray,
        bounds: Tuple[float, float],
        max_iter: int,
        tolerance: float
    ) -> Optional[np.ndarray]:
        """Solve ERC using numerical optimization"""
        try:
            n_assets = S.shape[0]
            
            def objective(x):
                """Objective function: sum of squared deviations from target risk budget"""
                if np.sum(x) == 0:
                    return 1e6
                
                x = x / np.sum(x)  # Normalize
                risk_contrib = self._calculate_risk_contributions(x, S)
                return np.sum((risk_contrib - risk_budget) ** 2)
            
            def gradient(x):
                """Gradient of objective function"""
                return self._erc_gradient(x, S, risk_budget)
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
            ]
            
            # Bounds
            bounds_list = [(bounds[0], bounds[1]) for _ in range(n_assets)]
            
            # Initial guess
            x0 = 1 / np.sqrt(np.diag(S))
            x0 = x0 / x0.sum()
            
            # Optimize
            result = optimize.minimize(
                objective,
                x0,
                method='SLSQP',
                jac=gradient,
                bounds=bounds_list,
                constraints=constraints,
                options={'ftol': tolerance, 'maxiter': max_iter, 'disp': False}
            )
            
            if result.success:
                return result.x / result.x.sum()
            else:
                logger.warning(f"ERC optimization did not converge: {result.message}")
                return result.x / result.x.sum()  # Return best attempt
                
        except Exception as e:
            logger.error(f"Error in numerical ERC solving: {e}")
            return None
    
    def _erc_gradient(self, x: np.ndarray, S: np.ndarray, risk_budget: np.ndarray) -> np.ndarray:
        """Calculate gradient of ERC objective function"""
        n = len(x)
        portfolio_vol = np.sqrt(np.dot(x, np.dot(S, x)))
        
        if portfolio_vol == 0:
            return np.zeros(n)
        
        marginal_contrib = np.dot(S, x) / portfolio_vol
        risk_contrib = x * marginal_contrib
        total_risk = risk_contrib.sum()
        risk_contrib_pct = risk_contrib / total_risk
        
        # Gradient components
        grad = np.zeros(n)
        
        for i in range(n):
            # Partial derivative of risk contribution percentage w.r.t. weight i
            d_rc_i = (marginal_contrib[i] * total_risk + x[i] * np.dot(S[i, :], x) / portfolio_vol - 
                     risk_contrib[i]) / (total_risk ** 2)
            
            grad[i] = 2 * (risk_contrib_pct[i] - risk_budget[i]) * d_rc_i
        
        return grad
    
    def _erc_hessian_approx(self, x: np.ndarray, S: np.ndarray) -> np.ndarray:
        """Approximate Hessian for ERC optimization"""
        n = len(x)
        
        # Use identity matrix scaled by the covariance matrix as approximation
        hess = np.eye(n) * np.mean(np.diag(S))
        
        return hess
    
    def _calculate_risk_contributions(self, x: np.ndarray, S: np.ndarray) -> np.ndarray:
        """Calculate risk contributions as percentages"""
        portfolio_vol = np.sqrt(np.dot(x, np.dot(S, x)))
        
        if portfolio_vol == 0:
            return np.ones(len(x)) / len(x)
        
        marginal_contrib = np.dot(S, x) / portfolio_vol
        risk_contrib = x * marginal_contrib
        
        return risk_contrib / risk_contrib.sum()
    
    def optimize_hierarchical_risk_parity(
        self,
        returns: pd.DataFrame,
        linkage_method: str = 'single',
        distance_metric: str = 'correlation'
    ) -> Dict:
        """
        Hierarchical Risk Parity (HRP) optimization
        
        Args:
            returns: Historical returns DataFrame
            linkage_method: Hierarchical clustering linkage method
            distance_metric: Distance metric for clustering
            
        Returns:
            HRP optimization results
        """
        try:
            # Step 1: Tree Clustering
            corr_matrix = returns.corr()
            
            if distance_metric == 'correlation':
                distance_matrix = ((1 - corr_matrix) / 2) ** 0.5
            elif distance_metric == 'covariance':
                cov_matrix = returns.cov()
                distance_matrix = cov_matrix / np.sqrt(np.outer(np.diag(cov_matrix), np.diag(cov_matrix)))
                distance_matrix = ((1 - distance_matrix) / 2) ** 0.5
            else:
                raise ValueError(f"Unknown distance metric: {distance_metric}")
            
            # Convert to condensed distance matrix
            distance_condensed = squareform(distance_matrix, checks=False)
            
            # Perform hierarchical clustering
            linkage_matrix = hierarchy.linkage(distance_condensed, method=linkage_method)
            
            # Step 2: Quasi-Diagonalization
            sorted_indices = hierarchy.leaves_list(linkage_matrix)
            sorted_returns = returns.iloc[:, sorted_indices]
            
            # Step 3: Recursive Bisection
            cov_matrix = sorted_returns.cov().values * 252
            weights = self._hrp_recursive_bisection(cov_matrix)
            
            # Map back to original order
            original_weights = np.zeros(len(returns.columns))
            for i, idx in enumerate(sorted_indices):
                original_weights[idx] = weights[i]
            
            # Calculate portfolio metrics
            S = returns.cov().values * 252
            portfolio_return = np.dot(original_weights, returns.mean() * 252)
            portfolio_vol = np.sqrt(np.dot(original_weights, np.dot(S, original_weights)))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol
            
            # Calculate risk contributions
            marginal_contrib = np.dot(S, original_weights) / portfolio_vol
            risk_contrib = original_weights * marginal_contrib
            risk_contrib_pct = risk_contrib / risk_contrib.sum()
            
            # Calculate diversification metrics
            weighted_vol = np.dot(original_weights, np.sqrt(np.diag(S)))
            diversification_ratio = weighted_vol / portfolio_vol
            
            return {
                'weights': dict(zip(returns.columns, original_weights)),
                'risk_contributions': dict(zip(returns.columns, risk_contrib_pct)),
                'portfolio_return': portfolio_return,
                'portfolio_volatility': portfolio_vol,
                'sharpe_ratio': sharpe_ratio,
                'diversification_ratio': diversification_ratio,
                'clustering_order': [returns.columns[i] for i in sorted_indices],
                'linkage_matrix': linkage_matrix,
                'distance_matrix': distance_matrix
            }
            
        except Exception as e:
            logger.error(f"Error in HRP optimization: {e}")
            return {'error': str(e)}
    
    def _hrp_recursive_bisection(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Recursive bisection for HRP weight allocation"""
        n_assets = cov_matrix.shape[0]
        
        if n_assets == 1:
            return np.array([1.0])
        
        # Split into two clusters
        mid_point = n_assets // 2
        
        # Left cluster
        left_cov = cov_matrix[:mid_point, :mid_point]
        left_weights = self._hrp_recursive_bisection(left_cov)
        left_cluster_var = np.dot(left_weights, np.dot(left_cov, left_weights))
        
        # Right cluster
        right_cov = cov_matrix[mid_point:, mid_point:]
        right_weights = self._hrp_recursive_bisection(right_cov)
        right_cluster_var = np.dot(right_weights, np.dot(right_cov, right_weights))
        
        # Allocate weight between clusters (inverse variance weighting)
        total_var = left_cluster_var + right_cluster_var
        left_cluster_weight = right_cluster_var / total_var
        right_cluster_weight = left_cluster_var / total_var
        
        # Combine weights
        weights = np.zeros(n_assets)
        weights[:mid_point] = left_weights * left_cluster_weight
        weights[mid_point:] = right_weights * right_cluster_weight
        
        return weights
    
    def optimize_nested_clustered_optimization(
        self,
        returns: pd.DataFrame,
        max_clusters: int = 10,
        n_trials: int = 1000
    ) -> Dict:
        """
        Nested Clustered Optimization (NCO)
        
        Args:
            returns: Historical returns DataFrame
            max_clusters: Maximum number of clusters to consider
            n_trials: Number of Monte Carlo trials
            
        Returns:
            NCO optimization results
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            
            # Prepare correlation matrix
            corr_matrix = returns.corr()
            distance_matrix = np.sqrt((1 - corr_matrix) / 2)
            
            # Find optimal number of clusters
            best_k = 2
            best_silhouette = -1
            
            for k in range(2, min(max_clusters + 1, len(returns.columns))):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(distance_matrix)
                
                if len(np.unique(cluster_labels)) > 1:  # Need at least 2 clusters for silhouette
                    silhouette = silhouette_score(distance_matrix, cluster_labels)
                    
                    if silhouette > best_silhouette:
                        best_silhouette = silhouette
                        best_k = k
            
            # Perform final clustering
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(distance_matrix)
            
            # Step 1: Intra-cluster optimization (equal risk contribution within clusters)
            cluster_weights = {}
            intra_cluster_weights = {}
            
            for cluster_id in range(best_k):
                cluster_assets = [returns.columns[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
                
                if len(cluster_assets) > 1:
                    cluster_returns = returns[cluster_assets]
                    
                    # ERC optimization within cluster
                    erc_result = self.optimize_equal_risk_contribution(cluster_returns)
                    
                    if 'error' not in erc_result:
                        intra_cluster_weights[cluster_id] = erc_result['weights']
                        cluster_vol = erc_result['portfolio_volatility']
                    else:
                        # Fallback to equal weights
                        equal_weight = 1 / len(cluster_assets)
                        intra_cluster_weights[cluster_id] = {asset: equal_weight for asset in cluster_assets}
                        cluster_vol = np.sqrt(np.mean(np.diag(cluster_returns.cov() * 252)))
                else:
                    # Single asset cluster
                    intra_cluster_weights[cluster_id] = {cluster_assets[0]: 1.0}
                    cluster_vol = np.sqrt(returns[cluster_assets[0]].var() * 252)
                
                cluster_weights[cluster_id] = cluster_vol
            
            # Step 2: Inter-cluster optimization (inverse variance weighting)
            total_inv_var = sum(1 / vol for vol in cluster_weights.values())
            inter_cluster_weights = {cluster_id: (1 / vol) / total_inv_var 
                                   for cluster_id, vol in cluster_weights.items()}
            
            # Step 3: Combine weights
            final_weights = {}
            for cluster_id, cluster_weight in inter_cluster_weights.items():
                for asset, intra_weight in intra_cluster_weights[cluster_id].items():
                    final_weights[asset] = cluster_weight * intra_weight
            
            # Calculate portfolio metrics
            weights_array = np.array([final_weights[asset] for asset in returns.columns])
            S = returns.cov().values * 252
            
            portfolio_return = np.dot(weights_array, returns.mean() * 252)
            portfolio_vol = np.sqrt(np.dot(weights_array, np.dot(S, weights_array)))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol
            
            # Risk contributions
            marginal_contrib = np.dot(S, weights_array) / portfolio_vol
            risk_contrib = weights_array * marginal_contrib
            risk_contrib_pct = risk_contrib / risk_contrib.sum()
            
            return {
                'weights': final_weights,
                'risk_contributions': dict(zip(returns.columns, risk_contrib_pct)),
                'portfolio_return': portfolio_return,
                'portfolio_volatility': portfolio_vol,
                'sharpe_ratio': sharpe_ratio,
                'n_clusters': best_k,
                'cluster_labels': dict(zip(returns.columns, cluster_labels)),
                'inter_cluster_weights': inter_cluster_weights,
                'intra_cluster_weights': intra_cluster_weights,
                'silhouette_score': best_silhouette
            }
            
        except Exception as e:
            logger.error(f"Error in NCO optimization: {e}")
            return {'error': str(e)}
    
    def optimize_volatility_targeting(
        self,
        returns: pd.DataFrame,
        target_volatility: float = 0.10,
        base_strategy: str = 'equal_risk',
        lookback_window: int = 252
    ) -> Dict:
        """
        Volatility targeting strategy
        
        Args:
            returns: Historical returns DataFrame
            target_volatility: Target portfolio volatility (annual)
            base_strategy: Base strategy ('equal_risk', 'hrp', 'equal_weight')
            lookback_window: Lookback window for volatility estimation
            
        Returns:
            Volatility targeting results
        """
        try:
            # Get base strategy weights
            if base_strategy == 'equal_risk':
                base_result = self.optimize_equal_risk_contribution(returns)
            elif base_strategy == 'hrp':
                base_result = self.optimize_hierarchical_risk_parity(returns)
            elif base_strategy == 'equal_weight':
                n_assets = len(returns.columns)
                base_weights = np.ones(n_assets) / n_assets
                base_result = {'weights': dict(zip(returns.columns, base_weights))}
            else:
                raise ValueError(f"Unknown base strategy: {base_strategy}")
            
            if 'error' in base_result:
                return base_result
            
            base_weights = np.array(list(base_result['weights'].values()))
            
            # Calculate base strategy volatility
            S = returns.cov().values * 252
            base_volatility = np.sqrt(np.dot(base_weights, np.dot(S, base_weights)))
            
            # Volatility scaling factor
            vol_scale = target_volatility / base_volatility
            
            # If scaling would require leverage > 1.5 or < 0.5, cap it
            vol_scale = np.clip(vol_scale, 0.5, 1.5)
            
            # Scale weights
            scaled_weights = base_weights * vol_scale
            
            # Calculate cash allocation (1 - sum of scaled weights)
            cash_allocation = 1 - scaled_weights.sum()
            
            # Final weights including cash
            final_weights = dict(zip(returns.columns, scaled_weights))
            if abs(cash_allocation) > 0.01:  # Only include if material
                final_weights['CASH'] = cash_allocation
            
            # Calculate final portfolio metrics
            final_vol = base_volatility * vol_scale
            portfolio_return = np.dot(scaled_weights, returns.mean() * 252) + cash_allocation * self.risk_free_rate
            
            return {
                'weights': final_weights,
                'portfolio_return': portfolio_return,
                'portfolio_volatility': final_vol,
                'target_volatility': target_volatility,
                'base_volatility': base_volatility,
                'volatility_scale': vol_scale,
                'cash_allocation': cash_allocation,
                'base_strategy': base_strategy
            }
            
        except Exception as e:
            logger.error(f"Error in volatility targeting: {e}")
            return {'error': str(e)}
    
    def dynamic_risk_parity(
        self,
        returns: pd.DataFrame,
        regime_indicators: Optional[pd.Series] = None,
        rebalance_frequency: str = 'monthly',
        lookback_window: int = 252
    ) -> Dict:
        """
        Dynamic risk parity with regime detection
        
        Args:
            returns: Historical returns DataFrame
            regime_indicators: Series indicating market regimes (optional)
            rebalance_frequency: Rebalancing frequency
            lookback_window: Lookback window for optimization
            
        Returns:
            Dynamic risk parity results
        """
        try:
            # If no regime indicators provided, use simple volatility-based regimes
            if regime_indicators is None:
                rolling_vol = returns.rolling(21).std().mean(axis=1) * np.sqrt(252)
                vol_median = rolling_vol.median()
                regime_indicators = (rolling_vol > vol_median).astype(int)
            
            # Determine rebalancing dates
            if rebalance_frequency == 'monthly':
                rebalance_dates = returns.resample('M').last().index
            elif rebalance_frequency == 'quarterly':
                rebalance_dates = returns.resample('Q').last().index
            else:
                rebalance_dates = returns.resample('W').last().index
            
            portfolio_returns = []
            weights_history = []
            regime_history = []
            
            for i, rebal_date in enumerate(rebalance_dates[1:], 1):  # Skip first date
                # Get historical data
                end_idx = returns.index.get_loc(rebal_date)
                start_idx = max(0, end_idx - lookback_window)
                
                historical_returns = returns.iloc[start_idx:end_idx]
                historical_regimes = regime_indicators.iloc[start_idx:end_idx]
                
                if len(historical_returns) < 20:
                    continue
                
                # Get current regime
                current_regime = historical_regimes.iloc[-1]
                regime_history.append(current_regime)
                
                # Regime-specific risk budgeting
                if current_regime == 1:  # High volatility regime
                    # More defensive allocation
                    risk_budget = self._create_defensive_risk_budget(returns.columns)
                else:  # Low volatility regime
                    # Standard equal risk allocation
                    risk_budget = None
                
                # Optimize for current regime
                erc_result = self.optimize_equal_risk_contribution(
                    historical_returns,
                    risk_budget=risk_budget
                )
                
                if 'error' not in erc_result:
                    weights = list(erc_result['weights'].values())
                    weights_history.append(weights)
                    
                    # Calculate portfolio return for next period
                    if i < len(rebalance_dates):
                        next_date = rebalance_dates[i]
                        period_returns = returns.loc[rebal_date:next_date].iloc[1:]
                        
                        if not period_returns.empty:
                            period_portfolio_returns = period_returns.dot(weights)
                            portfolio_returns.extend(period_portfolio_returns)
            
            if portfolio_returns:
                portfolio_returns = pd.Series(portfolio_returns)
                
                # Calculate performance metrics by regime
                regime_performance = {}
                for regime in [0, 1]:
                    regime_mask = np.array(regime_history) == regime
                    if regime_mask.sum() > 0:
                        regime_weights = [w for i, w in enumerate(weights_history) if regime_mask[i]]
                        avg_weights = np.mean(regime_weights, axis=0) if regime_weights else None
                        
                        regime_performance[f'regime_{regime}'] = {
                            'periods': regime_mask.sum(),
                            'average_weights': dict(zip(returns.columns, avg_weights)) if avg_weights is not None else {}
                        }
                
                return {
                    'portfolio_returns': portfolio_returns,
                    'weights_history': weights_history,
                    'regime_history': regime_history,
                    'rebalance_dates': rebalance_dates[1:len(weights_history)+1],
                    'regime_performance': regime_performance,
                    'total_return': (1 + portfolio_returns).prod() - 1,
                    'annualized_return': portfolio_returns.mean() * 252,
                    'annualized_volatility': portfolio_returns.std() * np.sqrt(252),
                    'sharpe_ratio': portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
                }
            else:
                return {'error': 'No portfolio returns calculated'}
                
        except Exception as e:
            logger.error(f"Error in dynamic risk parity: {e}")
            return {'error': str(e)}
    
    def _create_defensive_risk_budget(self, asset_names: pd.Index) -> Dict:
        """Create defensive risk budget (simplified example)"""
        # This is a simplified example - in practice, this would be more sophisticated
        # based on asset characteristics, sectors, etc.
        
        defensive_assets = []  # Could identify bonds, utilities, etc.
        growth_assets = []     # Could identify tech, growth stocks, etc.
        
        risk_budget = {}
        n_assets = len(asset_names)
        
        # Equal allocation for now (can be enhanced)
        for asset in asset_names:
            risk_budget[asset] = 1 / n_assets
        
        return risk_budget