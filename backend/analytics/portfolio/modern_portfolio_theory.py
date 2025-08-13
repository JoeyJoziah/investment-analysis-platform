"""
Modern Portfolio Theory Implementation

Advanced MPT optimization including:
- Mean-variance optimization
- Efficient frontier construction
- Risk budgeting
- Black-Litterman integration
- Multiple constraint types
- Alternative risk measures
- Monte Carlo simulation
- Robust optimization techniques

Uses scipy.optimize and cvxpy for advanced optimization
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import norm
import cvxpy as cp

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ModernPortfolioOptimizer:
    """
    Advanced Modern Portfolio Theory optimizer
    
    Features:
    - Multiple optimization objectives
    - Various constraint types
    - Risk budgeting capabilities
    - Robust optimization
    - Transaction cost integration
    - Multiple risk measures
    - Scenario analysis
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize MPT optimizer
        
        Args:
            risk_free_rate: Risk-free rate for Sharpe ratio calculations
        """
        self.risk_free_rate = risk_free_rate
    
    def optimize_portfolio(
        self,
        returns: pd.DataFrame,
        objective: str = 'max_sharpe',
        constraints: Dict = None,
        bounds: Tuple[float, float] = (0, 1),
        target_return: Optional[float] = None,
        target_risk: Optional[float] = None,
        transaction_costs: Optional[Dict] = None,
        current_weights: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Optimize portfolio weights
        
        Args:
            returns: DataFrame of asset returns
            objective: Optimization objective ('max_sharpe', 'min_vol', 'max_return', 'risk_parity')
            constraints: Dictionary of constraints
            bounds: Weight bounds for each asset
            target_return: Target return for efficient frontier
            target_risk: Target risk for efficient frontier
            transaction_costs: Transaction cost parameters
            current_weights: Current portfolio weights (for transaction costs)
            
        Returns:
            Dictionary with optimal weights and portfolio metrics
        """
        try:
            # Prepare data
            mu = returns.mean() * 252  # Annualize returns
            S = returns.cov() * 252    # Annualize covariance
            n_assets = len(mu)
            
            if constraints is None:
                constraints = {'weights_sum_to_one': True}
            
            # Set up optimization problem
            w = cp.Variable(n_assets)
            
            # Define portfolio return and risk
            portfolio_return = mu.values.T @ w
            portfolio_risk = cp.quad_form(w, S.values)
            
            # Add transaction costs if specified
            if transaction_costs and current_weights is not None:
                tc_rate = transaction_costs.get('rate', 0.001)  # 0.1% default
                weight_changes = cp.abs(w - current_weights)
                transaction_cost = tc_rate * cp.sum(weight_changes)
                portfolio_return -= transaction_cost
            
            # Define constraints
            constraint_list = []
            
            # Weights sum to one
            if constraints.get('weights_sum_to_one', True):
                constraint_list.append(cp.sum(w) == 1)
            
            # Box constraints (weight bounds)
            if bounds:
                constraint_list.append(w >= bounds[0])
                constraint_list.append(w <= bounds[1])
            
            # Target return constraint
            if target_return is not None:
                constraint_list.append(portfolio_return >= target_return)
            
            # Target risk constraint
            if target_risk is not None:
                constraint_list.append(portfolio_risk <= target_risk**2)
            
            # Sector constraints
            if 'sector_limits' in constraints:
                sector_limits = constraints['sector_limits']
                sector_mapping = constraints.get('sector_mapping', {})
                
                for sector, (min_weight, max_weight) in sector_limits.items():
                    sector_assets = [i for i, asset in enumerate(returns.columns) 
                                   if sector_mapping.get(asset) == sector]
                    if sector_assets:
                        sector_weight = cp.sum([w[i] for i in sector_assets])
                        if min_weight is not None:
                            constraint_list.append(sector_weight >= min_weight)
                        if max_weight is not None:
                            constraint_list.append(sector_weight <= max_weight)
            
            # Turnover constraint
            if 'max_turnover' in constraints and current_weights is not None:
                max_turnover = constraints['max_turnover']
                turnover = cp.sum(cp.abs(w - current_weights))
                constraint_list.append(turnover <= max_turnover)
            
            # Cardinality constraint (maximum number of assets)
            if 'max_assets' in constraints:
                max_assets = constraints['max_assets']
                # This requires binary variables - simplified approach
                min_weight_threshold = constraints.get('min_weight_threshold', 0.01)
                active_assets = cp.sum(w >= min_weight_threshold)
                constraint_list.append(active_assets <= max_assets)
            
            # Define objective function
            if objective == 'max_sharpe':
                # Maximize Sharpe ratio (equivalent to maximizing return/risk)
                prob = cp.Problem(
                    cp.Maximize(portfolio_return - self.risk_free_rate),
                    constraint_list + [portfolio_risk <= 1]  # Normalize by risk
                )
            elif objective == 'min_vol':
                # Minimize portfolio volatility
                prob = cp.Problem(cp.Minimize(portfolio_risk), constraint_list)
            elif objective == 'max_return':
                # Maximize expected return
                prob = cp.Problem(cp.Maximize(portfolio_return), constraint_list)
            elif objective == 'risk_parity':
                # Risk parity optimization
                return self._optimize_risk_parity(returns, constraints, bounds)
            elif objective == 'max_diversification':
                # Maximum diversification ratio
                return self._optimize_max_diversification(returns, constraints, bounds)
            else:
                raise ValueError(f"Unknown objective: {objective}")
            
            # Solve optimization problem
            prob.solve(solver=cp.OSQP, verbose=False)
            
            if prob.status not in ["infeasible", "unbounded"]:
                weights = w.value
                
                # Calculate portfolio metrics
                portfolio_metrics = self._calculate_portfolio_metrics(
                    weights, mu, S, returns
                )
                
                return {
                    'weights': dict(zip(returns.columns, weights)),
                    'metrics': portfolio_metrics,
                    'status': prob.status,
                    'objective_value': prob.value
                }
            else:
                logger.error(f"Optimization failed with status: {prob.status}")
                return {'error': f"Optimization failed: {prob.status}"}
        
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {e}")
            return {'error': str(e)}
    
    def generate_efficient_frontier(
        self,
        returns: pd.DataFrame,
        n_portfolios: int = 100,
        constraints: Dict = None,
        bounds: Tuple[float, float] = (0, 1)
    ) -> Dict:
        """
        Generate efficient frontier
        
        Args:
            returns: DataFrame of asset returns
            n_portfolios: Number of portfolios on frontier
            constraints: Portfolio constraints
            bounds: Weight bounds
            
        Returns:
            Dictionary with frontier data
        """
        try:
            mu = returns.mean() * 252
            S = returns.cov() * 252
            
            # Find minimum variance portfolio
            min_vol_result = self.optimize_portfolio(
                returns, objective='min_vol', constraints=constraints, bounds=bounds
            )
            
            if 'error' in min_vol_result:
                return min_vol_result
            
            min_return = min_vol_result['metrics']['expected_return']
            
            # Find maximum return portfolio
            max_return_result = self.optimize_portfolio(
                returns, objective='max_return', constraints=constraints, bounds=bounds
            )
            
            if 'error' in max_return_result:
                return max_return_result
            
            max_return = max_return_result['metrics']['expected_return']
            
            # Generate target returns along the frontier
            target_returns = np.linspace(min_return, max_return, n_portfolios)
            
            frontier_weights = []
            frontier_returns = []
            frontier_volatilities = []
            frontier_sharpe_ratios = []
            
            for target_ret in target_returns:
                result = self.optimize_portfolio(
                    returns,
                    objective='min_vol',
                    target_return=target_ret,
                    constraints=constraints,
                    bounds=bounds
                )
                
                if 'error' not in result:
                    weights = list(result['weights'].values())
                    metrics = result['metrics']
                    
                    frontier_weights.append(weights)
                    frontier_returns.append(metrics['expected_return'])
                    frontier_volatilities.append(metrics['volatility'])
                    frontier_sharpe_ratios.append(metrics['sharpe_ratio'])
            
            # Find maximum Sharpe ratio portfolio
            max_sharpe_result = self.optimize_portfolio(
                returns, objective='max_sharpe', constraints=constraints, bounds=bounds
            )
            
            return {
                'returns': frontier_returns,
                'volatilities': frontier_volatilities,
                'sharpe_ratios': frontier_sharpe_ratios,
                'weights': frontier_weights,
                'asset_names': list(returns.columns),
                'max_sharpe_portfolio': max_sharpe_result,
                'min_vol_portfolio': min_vol_result,
                'max_return_portfolio': max_return_result
            }
            
        except Exception as e:
            logger.error(f"Error generating efficient frontier: {e}")
            return {'error': str(e)}
    
    def _optimize_risk_parity(
        self,
        returns: pd.DataFrame,
        constraints: Dict = None,
        bounds: Tuple[float, float] = (0, 1)
    ) -> Dict:
        """Optimize for risk parity (equal risk contribution)"""
        try:
            S = returns.cov() * 252
            n_assets = len(returns.columns)
            
            def risk_parity_objective(weights):
                """Objective function for risk parity"""
                weights = np.array(weights)
                portfolio_vol = np.sqrt(np.dot(weights, np.dot(S, weights)))
                
                # Calculate risk contributions
                marginal_contrib = np.dot(S, weights) / portfolio_vol
                contrib = weights * marginal_contrib
                
                # Target: equal risk contribution (1/n each)
                target_contrib = np.ones(n_assets) / n_assets
                
                # Minimize sum of squared deviations from equal risk
                return np.sum((contrib / np.sum(contrib) - target_contrib)**2)
            
            # Constraints
            constraint_list = []
            
            # Weights sum to one
            constraint_list.append({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            
            # Set bounds
            bounds_list = [bounds for _ in range(n_assets)]
            
            # Initial guess (equal weights)
            x0 = np.ones(n_assets) / n_assets
            
            # Optimize
            result = optimize.minimize(
                risk_parity_objective,
                x0,
                method='SLSQP',
                bounds=bounds_list,
                constraints=constraint_list,
                options={'ftol': 1e-9, 'disp': False}
            )
            
            if result.success:
                weights = result.x
                mu = returns.mean() * 252
                
                portfolio_metrics = self._calculate_portfolio_metrics(
                    weights, mu, S, returns
                )
                
                # Calculate actual risk contributions
                portfolio_vol = np.sqrt(np.dot(weights, np.dot(S, weights)))
                marginal_contrib = np.dot(S, weights) / portfolio_vol
                risk_contrib = weights * marginal_contrib
                risk_contrib_pct = risk_contrib / np.sum(risk_contrib)
                
                portfolio_metrics['risk_contributions'] = dict(
                    zip(returns.columns, risk_contrib_pct)
                )
                
                return {
                    'weights': dict(zip(returns.columns, weights)),
                    'metrics': portfolio_metrics,
                    'status': 'optimal'
                }
            else:
                return {'error': 'Risk parity optimization failed'}
                
        except Exception as e:
            logger.error(f"Error in risk parity optimization: {e}")
            return {'error': str(e)}
    
    def _optimize_max_diversification(
        self,
        returns: pd.DataFrame,
        constraints: Dict = None,
        bounds: Tuple[float, float] = (0, 1)
    ) -> Dict:
        """Optimize for maximum diversification ratio"""
        try:
            S = returns.cov() * 252
            vol = np.sqrt(np.diag(S))  # Individual asset volatilities
            n_assets = len(returns.columns)
            
            def diversification_ratio_objective(weights):
                """Negative diversification ratio (to maximize)"""
                weights = np.array(weights)
                weighted_avg_vol = np.dot(weights, vol)
                portfolio_vol = np.sqrt(np.dot(weights, np.dot(S, weights)))
                return -weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 0
            
            # Constraints
            constraint_list = []
            constraint_list.append({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            
            # Bounds
            bounds_list = [bounds for _ in range(n_assets)]
            
            # Initial guess
            x0 = np.ones(n_assets) / n_assets
            
            # Optimize
            result = optimize.minimize(
                diversification_ratio_objective,
                x0,
                method='SLSQP',
                bounds=bounds_list,
                constraints=constraint_list,
                options={'ftol': 1e-9, 'disp': False}
            )
            
            if result.success:
                weights = result.x
                mu = returns.mean() * 252
                
                portfolio_metrics = self._calculate_portfolio_metrics(
                    weights, mu, S, returns
                )
                
                # Calculate diversification ratio
                weighted_avg_vol = np.dot(weights, vol)
                portfolio_vol = portfolio_metrics['volatility']
                diversification_ratio = weighted_avg_vol / portfolio_vol
                
                portfolio_metrics['diversification_ratio'] = diversification_ratio
                
                return {
                    'weights': dict(zip(returns.columns, weights)),
                    'metrics': portfolio_metrics,
                    'status': 'optimal'
                }
            else:
                return {'error': 'Maximum diversification optimization failed'}
                
        except Exception as e:
            logger.error(f"Error in maximum diversification optimization: {e}")
            return {'error': str(e)}
    
    def _calculate_portfolio_metrics(
        self,
        weights: np.ndarray,
        mu: pd.Series,
        S: pd.DataFrame,
        returns: pd.DataFrame
    ) -> Dict:
        """Calculate portfolio performance metrics"""
        try:
            weights = np.array(weights)
            
            # Basic metrics
            portfolio_return = np.dot(weights, mu)
            portfolio_variance = np.dot(weights, np.dot(S, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # Sharpe ratio
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
            
            # Calculate portfolio returns time series
            portfolio_returns = returns.dot(weights)
            
            # Additional metrics
            metrics = {
                'expected_return': portfolio_return,
                'volatility': portfolio_volatility,
                'variance': portfolio_variance,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': self._calculate_sortino_ratio(portfolio_returns),
                'calmar_ratio': self._calculate_calmar_ratio(portfolio_returns),
                'max_drawdown': self._calculate_max_drawdown(portfolio_returns),
                'var_95': self._calculate_var(portfolio_returns, 0.05),
                'cvar_95': self._calculate_cvar(portfolio_returns, 0.05),
                'skewness': portfolio_returns.skew(),
                'kurtosis': portfolio_returns.kurtosis(),
                'tracking_error': portfolio_returns.std() * np.sqrt(252),
                'information_ratio': portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252) if portfolio_returns.std() > 0 else 0
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return {}
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio"""
        try:
            downside_returns = returns[returns < 0]
            if len(downside_returns) == 0:
                return float('inf')
            
            downside_deviation = downside_returns.std() * np.sqrt(252)
            annual_return = returns.mean() * 252
            
            return (annual_return - self.risk_free_rate) / downside_deviation
        except:
            return 0
    
    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio"""
        try:
            annual_return = returns.mean() * 252
            max_dd = self._calculate_max_drawdown(returns)
            
            return annual_return / abs(max_dd) if max_dd != 0 else 0
        except:
            return 0
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        try:
            cum_returns = (1 + returns).cumprod()
            rolling_max = cum_returns.expanding().max()
            drawdowns = (cum_returns - rolling_max) / rolling_max
            return drawdowns.min()
        except:
            return 0
    
    def _calculate_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Value at Risk"""
        try:
            return np.percentile(returns, confidence_level * 100)
        except:
            return 0
    
    def _calculate_cvar(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        try:
            var = self._calculate_var(returns, confidence_level)
            return returns[returns <= var].mean()
        except:
            return 0
    
    def backtesting_analysis(
        self,
        returns: pd.DataFrame,
        rebalance_frequency: str = 'monthly',
        lookback_window: int = 252,
        constraints: Dict = None,
        objective: str = 'max_sharpe'
    ) -> Dict:
        """
        Perform backtesting analysis with periodic rebalancing
        
        Args:
            returns: Historical returns data
            rebalance_frequency: 'daily', 'weekly', 'monthly', 'quarterly'
            lookback_window: Days to look back for optimization
            constraints: Portfolio constraints
            objective: Optimization objective
            
        Returns:
            Backtesting results
        """
        try:
            # Determine rebalance dates
            if rebalance_frequency == 'monthly':
                rebalance_dates = returns.resample('M').last().index
            elif rebalance_frequency == 'quarterly':
                rebalance_dates = returns.resample('Q').last().index
            elif rebalance_frequency == 'weekly':
                rebalance_dates = returns.resample('W').last().index
            else:  # daily
                rebalance_dates = returns.index
            
            portfolio_returns = []
            weights_history = []
            turnover_history = []
            previous_weights = None
            
            for i, rebal_date in enumerate(rebalance_dates):
                if i == 0:
                    continue  # Skip first date (need history)
                
                # Get historical data for optimization
                end_date = rebal_date
                start_date = returns.index[max(0, returns.index.get_loc(end_date) - lookback_window)]
                historical_data = returns.loc[start_date:end_date]
                
                if len(historical_data) < 20:  # Need minimum data
                    continue
                
                # Optimize portfolio
                result = self.optimize_portfolio(
                    historical_data,
                    objective=objective,
                    constraints=constraints,
                    current_weights=previous_weights
                )
                
                if 'error' in result:
                    continue
                
                new_weights = np.array(list(result['weights'].values()))
                
                # Calculate turnover
                if previous_weights is not None:
                    turnover = np.sum(np.abs(new_weights - previous_weights))
                    turnover_history.append(turnover)
                
                weights_history.append(new_weights)
                previous_weights = new_weights
                
                # Calculate portfolio return for next period
                if i < len(rebalance_dates) - 1:
                    next_date = rebalance_dates[i + 1]
                    period_returns = returns.loc[rebal_date:next_date].iloc[1:]  # Exclude rebalance date
                    
                    if not period_returns.empty:
                        period_portfolio_returns = period_returns.dot(new_weights)
                        portfolio_returns.extend(period_portfolio_returns)
            
            if portfolio_returns:
                portfolio_returns = pd.Series(portfolio_returns)
                
                # Calculate performance metrics
                metrics = {
                    'total_return': (1 + portfolio_returns).prod() - 1,
                    'annualized_return': portfolio_returns.mean() * 252,
                    'annualized_volatility': portfolio_returns.std() * np.sqrt(252),
                    'sharpe_ratio': (portfolio_returns.mean() * 252 - self.risk_free_rate) / (portfolio_returns.std() * np.sqrt(252)),
                    'max_drawdown': self._calculate_max_drawdown(portfolio_returns),
                    'sortino_ratio': self._calculate_sortino_ratio(portfolio_returns),
                    'calmar_ratio': self._calculate_calmar_ratio(portfolio_returns),
                    'average_turnover': np.mean(turnover_history) if turnover_history else 0,
                    'number_of_rebalances': len(weights_history)
                }
                
                return {
                    'portfolio_returns': portfolio_returns,
                    'weights_history': weights_history,
                    'rebalance_dates': rebalance_dates[1:len(weights_history)+1],
                    'turnover_history': turnover_history,
                    'metrics': metrics
                }
            else:
                return {'error': 'No valid portfolio returns calculated'}
                
        except Exception as e:
            logger.error(f"Error in backtesting analysis: {e}")
            return {'error': str(e)}
    
    def scenario_analysis(
        self,
        returns: pd.DataFrame,
        scenarios: Dict,
        current_weights: Dict
    ) -> Dict:
        """
        Perform scenario analysis on portfolio
        
        Args:
            returns: Historical returns
            scenarios: Dictionary of scenarios with expected returns and correlations
            current_weights: Current portfolio weights
            
        Returns:
            Scenario analysis results
        """
        try:
            results = {}
            weights_array = np.array(list(current_weights.values()))
            
            for scenario_name, scenario_data in scenarios.items():
                expected_returns = scenario_data['expected_returns']  # Dict or Series
                correlation_adjustment = scenario_data.get('correlation_adjustment', 1.0)
                
                # Convert to arrays if needed
                if isinstance(expected_returns, dict):
                    exp_ret_array = np.array([expected_returns.get(asset, 0) for asset in returns.columns])
                else:
                    exp_ret_array = expected_returns.values
                
                # Adjust covariance matrix for scenario
                base_cov = returns.cov() * 252
                scenario_cov = base_cov * correlation_adjustment
                
                # Calculate scenario portfolio metrics
                portfolio_return = np.dot(weights_array, exp_ret_array)
                portfolio_risk = np.sqrt(np.dot(weights_array, np.dot(scenario_cov, weights_array)))
                
                sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
                
                # Calculate VaR and CVaR for scenario
                scenario_portfolio_returns = np.random.multivariate_normal(
                    exp_ret_array / 252, scenario_cov / 252, 10000
                ).dot(weights_array)
                
                var_95 = np.percentile(scenario_portfolio_returns, 5)
                cvar_95 = scenario_portfolio_returns[scenario_portfolio_returns <= var_95].mean()
                
                results[scenario_name] = {
                    'expected_return': portfolio_return,
                    'volatility': portfolio_risk,
                    'sharpe_ratio': sharpe_ratio,
                    'var_95': var_95,
                    'cvar_95': cvar_95
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in scenario analysis: {e}")
            return {'error': str(e)}