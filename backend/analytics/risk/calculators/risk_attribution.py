"""
Risk Attribution Calculator

Provides portfolio risk decomposition tools:
  - Marginal risk contribution (MRC)
  - Component VaR
  - Systematic vs idiosyncratic risk decomposition
  - Factor-based risk attribution
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Any


class RiskAttributionCalculator:
    """Decompose portfolio risk into asset-level and factor-level contributions.

    All matrix operations use numpy.  The calculator is stateless; pass
    returns and weights into each method.
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Marginal Risk Contribution
    # ------------------------------------------------------------------

    def calculate_marginal_risk(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray,
    ) -> np.ndarray:
        """Calculate the marginal risk contribution of each asset.

        The marginal contribution to risk (MCR) of asset *i* is defined as
        the partial derivative of portfolio volatility with respect to the
        weight of asset *i*:

            MRC_i = (Sigma @ w)_i / sigma_p

        where Sigma is the covariance matrix, w is the weight vector, and
        sigma_p = sqrt(w^T Sigma w).

        Parameters
        ----------
        weights : np.ndarray
            1-D array of portfolio weights (length N).
        cov_matrix : np.ndarray
            N x N covariance matrix of asset returns.

        Returns
        -------
        np.ndarray
            1-D array of marginal risk contributions (length N).
        """
        weights = np.asarray(weights, dtype=float)
        cov_matrix = np.asarray(cov_matrix, dtype=float)

        portfolio_variance = float(weights.T @ cov_matrix @ weights)
        if portfolio_variance <= 0:
            return np.zeros_like(weights)

        sigma_p = np.sqrt(portfolio_variance)
        marginal_contrib = cov_matrix @ weights
        return marginal_contrib / sigma_p

    # ------------------------------------------------------------------
    # Component VaR
    # ------------------------------------------------------------------

    def calculate_component_var(
        self,
        weights: np.ndarray,
        returns: np.ndarray,
        confidence_level: float = 0.95,
        asset_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Calculate each asset's contribution to total portfolio VaR.

        Component VaR of asset *i*:

            CVaR_i = w_i * MRC_i * VaR_p / sigma_p

        By construction, sum of component VaRs equals total portfolio VaR.

        Parameters
        ----------
        weights : np.ndarray
            1-D array of portfolio weights (length N).
        returns : np.ndarray
            T x N matrix of historical asset returns.
        confidence_level : float
            Confidence level (default 0.95).
        asset_names : list of str, optional
            Human-readable asset labels.  Defaults to ``asset_0``, etc.

        Returns
        -------
        dict
            Mapping from asset name to its component VaR contribution.
        """
        weights = np.asarray(weights, dtype=float)
        returns = np.asarray(returns, dtype=float)

        n_assets = len(weights)
        if asset_names is None:
            asset_names = [f"asset_{i}" for i in range(n_assets)]

        # Handle single-asset portfolio
        if n_assets == 1:
            portfolio_returns = returns.flatten() * weights[0]
            if len(portfolio_returns) == 0:
                return {asset_names[0]: 0.0}
            alpha = (1 - confidence_level) * 100
            total_var = float(np.percentile(portfolio_returns, alpha))
            return {asset_names[0]: total_var}

        # Covariance matrix (use at least 2-D returns)
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)
        cov_matrix = np.cov(returns.T, ddof=1) if returns.shape[0] > 1 else np.zeros((n_assets, n_assets))

        # Ensure cov_matrix is 2-D even for single-asset edge case
        cov_matrix = np.atleast_2d(cov_matrix)

        portfolio_variance = float(weights.T @ cov_matrix @ weights)
        if portfolio_variance <= 0:
            return {name: 0.0 for name in asset_names}

        sigma_p = np.sqrt(portfolio_variance)

        # Portfolio-level VaR (historical on weighted returns)
        portfolio_returns = returns @ weights
        alpha = (1 - confidence_level) * 100
        total_var = float(np.percentile(portfolio_returns, alpha))

        # Marginal risk contributions
        mrc = self.calculate_marginal_risk(weights, cov_matrix)

        # Component VaR_i = w_i * MRC_i * (VaR_p / sigma_p)
        var_over_sigma = total_var / sigma_p if sigma_p != 0 else 0.0
        component_vars = weights * mrc * var_over_sigma

        return {
            name: float(cv) for name, cv in zip(asset_names, component_vars)
        }

    # ------------------------------------------------------------------
    # Risk Decomposition
    # ------------------------------------------------------------------

    def decompose_risk(
        self,
        weights: np.ndarray,
        returns: np.ndarray,
        factor_returns: Optional[np.ndarray] = None,
        factor_names: Optional[List[str]] = None,
        asset_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Decompose portfolio risk into systematic and idiosyncratic components.

        When *factor_returns* are provided, a multivariate OLS regression
        is used to decompose each asset's returns into factor exposures
        (betas) and residuals.  Portfolio systematic risk is then:

            sigma^2_sys = w^T B Sigma_f B^T w

        where B is the (N x K) beta matrix and Sigma_f is the (K x K)
        factor covariance matrix.

        When no factor data is available, a single-factor (market)
        decomposition is used with the equal-weighted portfolio as the
        market proxy:

            systematic_i  = beta_i^2 * sigma^2_market
            idiosyncratic = total_var - systematic

        Parameters
        ----------
        weights : np.ndarray
            1-D array of portfolio weights (length N).
        returns : np.ndarray
            T x N matrix of historical asset returns.
        factor_returns : np.ndarray, optional
            T x K matrix of factor returns.  If ``None``, a market-only
            decomposition is used.
        factor_names : list of str, optional
            Labels for the factors.
        asset_names : list of str, optional
            Labels for the assets.

        Returns
        -------
        dict
            Keys: portfolio_volatility, marginal_risk, component_risk,
            percent_contribution, systematic_risk, idiosyncratic_risk,
            systematic_pct, idiosyncratic_pct, factor_attribution (if
            factor data provided).
        """
        weights = np.asarray(weights, dtype=float)
        returns = np.asarray(returns, dtype=float)

        n_assets = len(weights)
        if asset_names is None:
            asset_names = [f"asset_{i}" for i in range(n_assets)]

        # Ensure 2-D returns
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)

        # Covariance matrix
        cov_matrix = (
            np.cov(returns.T, ddof=1)
            if returns.shape[0] > 1
            else np.zeros((n_assets, n_assets))
        )
        cov_matrix = np.atleast_2d(cov_matrix)

        portfolio_variance = float(weights.T @ cov_matrix @ weights)
        portfolio_volatility = np.sqrt(max(portfolio_variance, 0.0))

        # Marginal and component risk
        mrc = self.calculate_marginal_risk(weights, cov_matrix)
        component_risk = weights * mrc
        total_component = float(np.sum(component_risk))
        if total_component != 0:
            pct_contribution = (component_risk / total_component * 100).tolist()
        else:
            pct_contribution = [0.0] * n_assets

        # Factor-based decomposition
        if factor_returns is not None:
            factor_result = self._factor_decomposition(
                weights, returns, factor_returns, factor_names
            )
        else:
            factor_result = self._market_proxy_decomposition(
                weights, returns, cov_matrix
            )

        return {
            "portfolio_volatility": float(portfolio_volatility),
            "marginal_risk": mrc.tolist(),
            "component_risk": component_risk.tolist(),
            "percent_contribution": pct_contribution,
            **factor_result,
        }

    # ------------------------------------------------------------------
    # Internal: factor-based decomposition via OLS
    # ------------------------------------------------------------------

    def _factor_decomposition(
        self,
        weights: np.ndarray,
        returns: np.ndarray,
        factor_returns: np.ndarray,
        factor_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Regression-based systematic/idiosyncratic decomposition.

        For each asset, run OLS:  r_i = alpha_i + B_i @ f + epsilon_i

        Then:
            systematic_var  = w^T B Sigma_f B^T w
            idiosyncratic_var = w^T diag(sigma^2_eps) w
        """
        factor_returns = np.asarray(factor_returns, dtype=float)
        if factor_returns.ndim == 1:
            factor_returns = factor_returns.reshape(-1, 1)

        n_assets = returns.shape[1]
        n_factors = factor_returns.shape[1]

        if factor_names is None:
            factor_names = [f"factor_{k}" for k in range(n_factors)]

        # OLS for each asset: r_i ~ intercept + factor_returns
        betas = np.zeros((n_assets, n_factors))
        residual_variances = np.zeros(n_assets)

        X = np.column_stack([np.ones(factor_returns.shape[0]), factor_returns])

        for i in range(n_assets):
            y = returns[:, i]
            # Use least-squares
            coeffs, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
            betas[i, :] = coeffs[1:]  # skip intercept
            fitted = X @ coeffs
            eps = y - fitted
            residual_variances[i] = float(np.var(eps, ddof=n_factors + 1)) if len(eps) > n_factors + 1 else float(np.var(eps))

        # Factor covariance
        factor_cov = np.cov(factor_returns.T, ddof=1) if factor_returns.shape[0] > 1 else np.zeros((n_factors, n_factors))
        factor_cov = np.atleast_2d(factor_cov)

        # Systematic variance = w^T B Sigma_f B^T w
        Bw = betas.T @ weights  # (K,)
        systematic_variance = float(Bw.T @ factor_cov @ Bw)

        # Idiosyncratic variance = w^T diag(eps_var) w
        idiosyncratic_variance = float(
            weights.T @ np.diag(residual_variances) @ weights
        )

        total_variance = systematic_variance + idiosyncratic_variance
        total_variance = max(total_variance, 1e-20)

        systematic_risk = np.sqrt(max(systematic_variance, 0.0))
        idiosyncratic_risk = np.sqrt(max(idiosyncratic_variance, 0.0))

        # Per-factor attribution
        factor_attribution = {}
        for k, fname in enumerate(factor_names):
            # Contribution of factor k: (w^T beta_k)^2 * var(f_k)
            factor_loading = float(weights @ betas[:, k])
            factor_var = float(factor_cov[k, k])
            contribution = factor_loading ** 2 * factor_var
            factor_attribution[fname] = {
                "variance_contribution": contribution,
                "pct_of_total": contribution / total_variance * 100,
                "portfolio_beta": factor_loading,
            }

        return {
            "systematic_risk": float(systematic_risk),
            "idiosyncratic_risk": float(idiosyncratic_risk),
            "systematic_pct": systematic_variance / total_variance * 100,
            "idiosyncratic_pct": idiosyncratic_variance / total_variance * 100,
            "factor_attribution": factor_attribution,
            "betas": betas.tolist(),
        }

    # ------------------------------------------------------------------
    # Internal: single-factor (market-proxy) decomposition
    # ------------------------------------------------------------------

    def _market_proxy_decomposition(
        self,
        weights: np.ndarray,
        returns: np.ndarray,
        cov_matrix: np.ndarray,
    ) -> Dict[str, Any]:
        """Beta-based decomposition using equal-weighted portfolio as market proxy.

        For each asset:
            beta_i = cov(r_i, r_m) / var(r_m)
            systematic_var_i = beta_i^2 * var(r_m)
            idiosyncratic_var_i = var(r_i) - systematic_var_i

        Portfolio-level:
            systematic_var = sum_i sum_j w_i w_j beta_i beta_j var(r_m)
            idiosyncratic_var = sum_i w_i^2 * idiosyncratic_var_i
        """
        n_assets = returns.shape[1]

        # Market proxy: equal-weighted portfolio
        market_returns = np.mean(returns, axis=1)
        market_var = float(np.var(market_returns, ddof=1)) if returns.shape[0] > 1 else 0.0

        if market_var <= 0:
            portfolio_var = float(weights.T @ cov_matrix @ weights)
            return {
                "systematic_risk": 0.0,
                "idiosyncratic_risk": np.sqrt(max(portfolio_var, 0.0)),
                "systematic_pct": 0.0,
                "idiosyncratic_pct": 100.0,
                "factor_attribution": {},
                "betas": [0.0] * n_assets,
            }

        # Per-asset betas
        betas = np.zeros(n_assets)
        idio_variances = np.zeros(n_assets)

        for i in range(n_assets):
            asset_returns = returns[:, i]
            cov_with_market = float(np.cov(asset_returns, market_returns, ddof=1)[0, 1]) if returns.shape[0] > 1 else 0.0
            betas[i] = cov_with_market / market_var
            asset_var = float(np.var(asset_returns, ddof=1)) if returns.shape[0] > 1 else 0.0
            systematic_var_i = betas[i] ** 2 * market_var
            idio_variances[i] = max(asset_var - systematic_var_i, 0.0)

        # Portfolio-level
        # systematic: (w^T beta)^2 * var(market)
        portfolio_beta = float(weights @ betas)
        systematic_variance = portfolio_beta ** 2 * market_var

        # idiosyncratic: w^T diag(idio_var) w
        idiosyncratic_variance = float(
            weights.T @ np.diag(idio_variances) @ weights
        )

        total_variance = systematic_variance + idiosyncratic_variance
        total_variance = max(total_variance, 1e-20)

        systematic_risk = np.sqrt(max(systematic_variance, 0.0))
        idiosyncratic_risk = np.sqrt(max(idiosyncratic_variance, 0.0))

        return {
            "systematic_risk": float(systematic_risk),
            "idiosyncratic_risk": float(idiosyncratic_risk),
            "systematic_pct": systematic_variance / total_variance * 100,
            "idiosyncratic_pct": idiosyncratic_variance / total_variance * 100,
            "factor_attribution": {
                "market": {
                    "variance_contribution": systematic_variance,
                    "pct_of_total": systematic_variance / total_variance * 100,
                    "portfolio_beta": portfolio_beta,
                }
            },
            "betas": betas.tolist(),
        }
