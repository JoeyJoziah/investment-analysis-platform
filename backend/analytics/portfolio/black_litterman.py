"""
Black-Litterman Model Implementation

Advanced Black-Litterman portfolio optimization including:
- Classical Black-Litterman framework
- Bayesian approach to portfolio optimization
- View incorporation and confidence levels
- Alternative prior specifications
- Robust estimation techniques
- Multi-level view structures
- Dynamic view updating

Combines market equilibrium with investor views for superior optimization
"""

import logging
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.covariance import LedoitWolf, ShrunkCovariance

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class BlackLittermanModel:
    """
    Advanced Black-Litterman model implementation
    
    Features:
    - Classical BL framework
    - Multiple view types (absolute, relative)
    - Confidence scaling
    - Robust covariance estimation
    - Dynamic view updating
    - Alternative equilibrium methods
    - Bayesian inference
    """
    
    def __init__(
        self,
        risk_aversion: Optional[float] = None,
        market_cap_weights: Optional[Dict] = None,
        risk_free_rate: float = 0.02
    ):
        """
        Initialize Black-Litterman model
        
        Args:
            risk_aversion: Risk aversion parameter (if None, will be estimated)
            market_cap_weights: Market capitalization weights for equilibrium
            risk_free_rate: Risk-free rate
        """
        self.risk_aversion = risk_aversion
        self.market_cap_weights = market_cap_weights
        self.risk_free_rate = risk_free_rate
    
    def optimize(
        self,
        returns: pd.DataFrame,
        views: List[Dict],
        view_confidences: Optional[List[float]] = None,
        tau: float = 0.05,
        method: str = 'classic',
        shrinkage_method: str = 'ledoit_wolf'
    ) -> Dict:
        """
        Perform Black-Litterman optimization
        
        Args:
            returns: Historical returns DataFrame
            views: List of view dictionaries
            view_confidences: Confidence levels for each view (0-1)
            tau: Scaling factor for uncertainty of prior
            method: 'classic', 'alternative', or 'idzorek'
            shrinkage_method: Covariance shrinkage method
            
        Returns:
            Optimization results with posterior weights
        """
        try:
            # Prepare data
            n_assets = len(returns.columns)
            
            # Estimate covariance matrix with shrinkage
            if shrinkage_method == 'ledoit_wolf':
                cov_estimator = LedoitWolf()
                S = cov_estimator.fit(returns).covariance_ * 252
            elif shrinkage_method == 'shrunk':
                cov_estimator = ShrunkCovariance()
                S = cov_estimator.fit(returns).covariance_ * 252
            else:
                S = returns.cov().values * 252
            
            # Get equilibrium (prior) expected returns
            if self.market_cap_weights:
                # Use market cap weights as equilibrium
                w_market = np.array([self.market_cap_weights.get(asset, 1/n_assets) 
                                   for asset in returns.columns])
                w_market = w_market / w_market.sum()  # Normalize
            else:
                # Use equal weights or estimate from data
                w_market = np.ones(n_assets) / n_assets
            
            # Estimate risk aversion if not provided
            if self.risk_aversion is None:
                market_return = returns.mean().values * 252
                market_variance = np.dot(w_market, np.dot(S, w_market))
                excess_return = np.dot(w_market, market_return) - self.risk_free_rate
                self.risk_aversion = excess_return / market_variance if market_variance > 0 else 3.0
            
            # Calculate equilibrium expected returns (pi)
            pi = self.risk_aversion * np.dot(S, w_market)
            
            # Process views
            P, Q, omega = self._process_views(views, view_confidences, returns.columns, S, tau)
            
            if P is None or Q is None:
                return {'error': 'No valid views provided'}
            
            # Black-Litterman calculation
            if method == 'classic':
                mu_bl, S_bl = self._classic_bl(pi, S, P, Q, omega, tau)
            elif method == 'alternative':
                mu_bl, S_bl = self._alternative_bl(pi, S, P, Q, omega, tau)
            elif method == 'idzorek':
                mu_bl, S_bl = self._idzorek_bl(pi, S, P, Q, omega, tau, view_confidences)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Calculate optimal weights
            inv_S_bl = linalg.inv(S_bl)
            w_bl = np.dot(inv_S_bl, mu_bl) / self.risk_aversion
            
            # Ensure weights sum to 1
            w_bl = w_bl / w_bl.sum()
            
            # Calculate portfolio metrics
            portfolio_return = np.dot(w_bl, mu_bl)
            portfolio_variance = np.dot(w_bl, np.dot(S_bl, w_bl))
            portfolio_volatility = np.sqrt(portfolio_variance)
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            
            # Calculate weight changes from equilibrium
            weight_changes = w_bl - w_market
            
            return {
                'weights': dict(zip(returns.columns, w_bl)),
                'expected_returns': dict(zip(returns.columns, mu_bl)),
                'equilibrium_weights': dict(zip(returns.columns, w_market)),
                'weight_changes': dict(zip(returns.columns, weight_changes)),
                'portfolio_return': portfolio_return,
                'portfolio_volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'risk_aversion': self.risk_aversion,
                'tau': tau,
                'views_summary': self._summarize_views(views, P, Q),
                'posterior_covariance': S_bl,
                'prior_returns': dict(zip(returns.columns, pi))
            }
            
        except Exception as e:
            logger.error(f"Error in Black-Litterman optimization: {e}")
            return {'error': str(e)}
    
    def _process_views(
        self,
        views: List[Dict],
        view_confidences: Optional[List[float]],
        asset_names: pd.Index,
        S: np.ndarray,
        tau: float
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Process views into P (picking matrix) and Q (view returns)"""
        try:
            n_assets = len(asset_names)
            n_views = len(views)
            
            if n_views == 0:
                return None, None, None
            
            P = np.zeros((n_views, n_assets))
            Q = np.zeros(n_views)
            omega_diag = np.zeros(n_views)
            
            for i, view in enumerate(views):
                view_type = view.get('type', 'absolute')
                
                if view_type == 'absolute':
                    # Absolute view: Asset X will return Y%
                    asset = view['asset']
                    expected_return = view['expected_return']
                    
                    if asset in asset_names:
                        asset_idx = asset_names.get_loc(asset)
                        P[i, asset_idx] = 1.0
                        Q[i] = expected_return
                    else:
                        logger.warning(f"Asset {asset} not found in portfolio")
                        continue
                
                elif view_type == 'relative':
                    # Relative view: Asset X will outperform Asset Y by Z%
                    asset1 = view['asset1']
                    asset2 = view['asset2']
                    relative_return = view['relative_return']
                    
                    if asset1 in asset_names and asset2 in asset_names:
                        asset1_idx = asset_names.get_loc(asset1)
                        asset2_idx = asset_names.get_loc(asset2)
                        P[i, asset1_idx] = 1.0
                        P[i, asset2_idx] = -1.0
                        Q[i] = relative_return
                    else:
                        logger.warning(f"Assets {asset1} or {asset2} not found in portfolio")
                        continue
                
                elif view_type == 'portfolio':
                    # Portfolio view: A combination of assets will return X%
                    weights = view['weights']  # Dict of asset: weight
                    expected_return = view['expected_return']
                    
                    for asset, weight in weights.items():
                        if asset in asset_names:
                            asset_idx = asset_names.get_loc(asset)
                            P[i, asset_idx] = weight
                    
                    Q[i] = expected_return
                
                # Calculate uncertainty (Omega) for this view
                if view_confidences and i < len(view_confidences):
                    confidence = view_confidences[i]
                    # Higher confidence = lower uncertainty
                    view_variance = tau * np.dot(P[i, :], np.dot(S, P[i, :])) / confidence
                else:
                    # Default uncertainty
                    view_variance = tau * np.dot(P[i, :], np.dot(S, P[i, :]))
                
                omega_diag[i] = view_variance
            
            # Create omega matrix (diagonal matrix of view uncertainties)
            omega = np.diag(omega_diag)
            
            return P, Q, omega
            
        except Exception as e:
            logger.error(f"Error processing views: {e}")
            return None, None, None
    
    def _classic_bl(
        self,
        pi: np.ndarray,
        S: np.ndarray,
        P: np.ndarray,
        Q: np.ndarray,
        omega: np.ndarray,
        tau: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Classic Black-Litterman formulation"""
        try:
            # Prior covariance
            S_prior = tau * S
            
            # Posterior expected returns
            inv_S_prior = linalg.inv(S_prior)
            inv_omega = linalg.inv(omega)
            
            # BL formula for expected returns
            A = inv_S_prior + np.dot(P.T, np.dot(inv_omega, P))
            b = np.dot(inv_S_prior, pi) + np.dot(P.T, np.dot(inv_omega, Q))
            
            mu_bl = np.dot(linalg.inv(A), b)
            
            # Posterior covariance
            S_bl = linalg.inv(A)
            
            return mu_bl, S_bl
            
        except Exception as e:
            logger.error(f"Error in classic BL calculation: {e}")
            raise e
    
    def _alternative_bl(
        self,
        pi: np.ndarray,
        S: np.ndarray,
        P: np.ndarray,
        Q: np.ndarray,
        omega: np.ndarray,
        tau: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Alternative Black-Litterman formulation"""
        try:
            # This is the simpler form often used in practice
            inv_omega = linalg.inv(omega)
            
            # Calculate M matrix
            M1 = linalg.inv(tau * S)
            M2 = np.dot(P.T, np.dot(inv_omega, P))
            M = M1 + M2
            
            # Expected returns
            mu_bl = np.dot(linalg.inv(M), 
                          np.dot(M1, pi) + np.dot(P.T, np.dot(inv_omega, Q)))
            
            # Covariance
            S_bl = linalg.inv(M)
            
            return mu_bl, S_bl
            
        except Exception as e:
            logger.error(f"Error in alternative BL calculation: {e}")
            raise e
    
    def _idzorek_bl(
        self,
        pi: np.ndarray,
        S: np.ndarray,
        P: np.ndarray,
        Q: np.ndarray,
        omega: np.ndarray,
        tau: float,
        view_confidences: Optional[List[float]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Idzorek method with confidence-based omega"""
        try:
            # Idzorek's method for determining omega based on confidence levels
            if view_confidences:
                # Adjust omega based on confidence levels
                for i, confidence in enumerate(view_confidences):
                    if confidence > 0:
                        # Scale omega inversely with confidence
                        omega[i, i] = omega[i, i] / confidence
            
            # Use classic BL with adjusted omega
            return self._classic_bl(pi, S, P, Q, omega, tau)
            
        except Exception as e:
            logger.error(f"Error in Idzorek BL calculation: {e}")
            raise e
    
    def _summarize_views(
        self,
        views: List[Dict],
        P: np.ndarray,
        Q: np.ndarray
    ) -> List[Dict]:
        """Summarize views for output"""
        summary = []
        
        for i, view in enumerate(views):
            summary.append({
                'view_index': i,
                'type': view.get('type', 'absolute'),
                'description': self._describe_view(view),
                'expected_return': Q[i] if i < len(Q) else None,
                'picking_vector': P[i, :].tolist() if i < len(P) else None
            })
        
        return summary
    
    def _describe_view(self, view: Dict) -> str:
        """Create human-readable description of view"""
        view_type = view.get('type', 'absolute')
        
        if view_type == 'absolute':
            return f"{view['asset']} expected return: {view['expected_return']:.2%}"
        elif view_type == 'relative':
            return f"{view['asset1']} will outperform {view['asset2']} by {view['relative_return']:.2%}"
        elif view_type == 'portfolio':
            weights_str = ', '.join([f"{asset}: {weight:.1%}" for asset, weight in view['weights'].items()])
            return f"Portfolio ({weights_str}) expected return: {view['expected_return']:.2%}"
        else:
            return "Unknown view type"
    
    def sensitivity_analysis(
        self,
        returns: pd.DataFrame,
        views: List[Dict],
        view_confidences: Optional[List[float]] = None,
        tau_range: Tuple[float, float] = (0.01, 0.1),
        n_tau_values: int = 10
    ) -> Dict:
        """
        Perform sensitivity analysis on tau parameter
        
        Args:
            returns: Historical returns
            views: Investment views
            view_confidences: View confidence levels
            tau_range: Range of tau values to test
            n_tau_values: Number of tau values to test
            
        Returns:
            Sensitivity analysis results
        """
        try:
            tau_values = np.linspace(tau_range[0], tau_range[1], n_tau_values)
            
            results = {
                'tau_values': tau_values,
                'weights_evolution': {},
                'returns_evolution': {},
                'volatility_evolution': [],
                'sharpe_evolution': []
            }
            
            # Initialize weight evolution tracking
            for asset in returns.columns:
                results['weights_evolution'][asset] = []
                results['returns_evolution'][asset] = []
            
            for tau in tau_values:
                bl_result = self.optimize(
                    returns=returns,
                    views=views,
                    view_confidences=view_confidences,
                    tau=tau
                )
                
                if 'error' not in bl_result:
                    # Track weight evolution
                    for asset in returns.columns:
                        results['weights_evolution'][asset].append(
                            bl_result['weights'][asset]
                        )
                        results['returns_evolution'][asset].append(
                            bl_result['expected_returns'][asset]
                        )
                    
                    results['volatility_evolution'].append(bl_result['portfolio_volatility'])
                    results['sharpe_evolution'].append(bl_result['sharpe_ratio'])
                else:
                    # Fill with NaNs for failed optimizations
                    for asset in returns.columns:
                        results['weights_evolution'][asset].append(np.nan)
                        results['returns_evolution'][asset].append(np.nan)
                    
                    results['volatility_evolution'].append(np.nan)
                    results['sharpe_evolution'].append(np.nan)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in sensitivity analysis: {e}")
            return {'error': str(e)}
    
    def view_impact_analysis(
        self,
        returns: pd.DataFrame,
        base_views: List[Dict],
        view_confidences: Optional[List[float]] = None,
        tau: float = 0.05
    ) -> Dict:
        """
        Analyze the impact of individual views
        
        Args:
            returns: Historical returns
            base_views: Base set of views
            view_confidences: View confidence levels
            tau: Tau parameter
            
        Returns:
            View impact analysis
        """
        try:
            # Get baseline (no views) results
            baseline_result = self.optimize(
                returns=returns,
                views=[],
                tau=tau
            )
            
            if 'error' in baseline_result:
                return baseline_result
            
            # Get full views result
            full_result = self.optimize(
                returns=returns,
                views=base_views,
                view_confidences=view_confidences,
                tau=tau
            )
            
            if 'error' in full_result:
                return full_result
            
            # Analyze individual view impacts
            view_impacts = []
            
            for i, view in enumerate(base_views):
                # Create view set with single view
                single_view = [view]
                single_confidence = [view_confidences[i]] if view_confidences else None
                
                single_result = self.optimize(
                    returns=returns,
                    views=single_view,
                    view_confidences=single_confidence,
                    tau=tau
                )
                
                if 'error' not in single_result:
                    # Calculate weight changes from baseline
                    weight_impact = {}
                    for asset in returns.columns:
                        baseline_weight = baseline_result['weights'][asset]
                        single_weight = single_result['weights'][asset]
                        weight_impact[asset] = single_weight - baseline_weight
                    
                    view_impacts.append({
                        'view_index': i,
                        'view_description': self._describe_view(view),
                        'weight_impact': weight_impact,
                        'return_impact': single_result['portfolio_return'] - baseline_result['portfolio_return'],
                        'risk_impact': single_result['portfolio_volatility'] - baseline_result['portfolio_volatility'],
                        'sharpe_impact': single_result['sharpe_ratio'] - baseline_result['sharpe_ratio']
                    })
            
            return {
                'baseline_result': baseline_result,
                'full_result': full_result,
                'view_impacts': view_impacts,
                'total_weight_change': {
                    asset: full_result['weights'][asset] - baseline_result['weights'][asset]
                    for asset in returns.columns
                }
            }
            
        except Exception as e:
            logger.error(f"Error in view impact analysis: {e}")
            return {'error': str(e)}
    
    def dynamic_view_updating(
        self,
        returns: pd.DataFrame,
        initial_views: List[Dict],
        new_information: Dict,
        learning_rate: float = 0.1,
        tau: float = 0.05
    ) -> Dict:
        """
        Update views dynamically based on new information
        
        Args:
            returns: Historical returns
            initial_views: Initial investment views
            new_information: New market information
            learning_rate: Rate at which views are updated
            tau: Tau parameter
            
        Returns:
            Updated optimization results
        """
        try:
            updated_views = []
            
            for view in initial_views:
                updated_view = view.copy()
                
                # Update view based on new information
                if 'market_performance' in new_information:
                    market_perf = new_information['market_performance']
                    
                    if view['type'] == 'absolute':
                        asset = view['asset']
                        if asset in market_perf:
                            # Update expected return based on recent performance
                            recent_performance = market_perf[asset]
                            current_expectation = view['expected_return']
                            
                            # Bayesian updating
                            updated_expectation = (
                                (1 - learning_rate) * current_expectation + 
                                learning_rate * recent_performance
                            )
                            
                            updated_view['expected_return'] = updated_expectation
                
                updated_views.append(updated_view)
            
            # Run optimization with updated views
            result = self.optimize(
                returns=returns,
                views=updated_views,
                tau=tau
            )
            
            if 'error' not in result:
                result['view_updates'] = {
                    'original_views': initial_views,
                    'updated_views': updated_views,
                    'learning_rate': learning_rate
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in dynamic view updating: {e}")
            return {'error': str(e)}