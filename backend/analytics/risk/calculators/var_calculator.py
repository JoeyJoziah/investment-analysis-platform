"""
Value at Risk (VaR) Calculator

Provides multiple VaR estimation methods:
  - Historical simulation
  - Parametric (normal distribution)
  - Monte Carlo (geometric Brownian motion)
  - Conditional VaR (Expected Shortfall)
  - Kupiec back-testing
"""

import numpy as np
from scipy import stats
from typing import Optional, Dict, Any, List
from enum import Enum


class VaRMethod(str, Enum):
    """VaR calculation methods."""

    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"


class VaRCalculator:
    """Calculate Value at Risk using historical, parametric, or Monte Carlo methods.

    Parameters
    ----------
    confidence_level : float
        Confidence level for VaR (e.g. 0.95 for 95% VaR).
    """

    def __init__(self, confidence_level: float = 0.95) -> None:
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1 exclusive")
        self.confidence_level = confidence_level

    # ------------------------------------------------------------------
    # Historical VaR
    # ------------------------------------------------------------------

    def calculate_historical_var(
        self,
        returns: np.ndarray,
        horizon: int = 1,
    ) -> float:
        """Calculate VaR using historical simulation.

        The alpha-percentile of the empirical return distribution is used
        directly as the VaR estimate.  For multi-day horizons the single-day
        VaR is scaled by sqrt(horizon).

        Parameters
        ----------
        returns : np.ndarray
            1-D array of historical returns (simple or log).
        horizon : int
            Holding period in days (default 1).

        Returns
        -------
        float
            VaR estimate (negative value indicates a loss).
        """
        if len(returns) == 0:
            return 0.0
        percentile = (1 - self.confidence_level) * 100
        var_1d = float(np.percentile(returns, percentile))
        return var_1d * np.sqrt(horizon)

    # ------------------------------------------------------------------
    # Parametric VaR
    # ------------------------------------------------------------------

    def calculate_parametric_var(
        self,
        returns: np.ndarray,
        horizon: int = 1,
    ) -> float:
        """Calculate VaR assuming returns follow a normal distribution.

        VaR = mu + z_{alpha} * sigma

        where z_{alpha} is the inverse-CDF of the standard normal at the
        loss tail probability (1 - confidence_level).

        Parameters
        ----------
        returns : np.ndarray
            1-D array of historical returns.
        horizon : int
            Holding period in days (default 1).

        Returns
        -------
        float
            VaR estimate (negative value indicates a loss).
        """
        if len(returns) == 0:
            return 0.0
        mean = float(np.mean(returns))
        std = float(np.std(returns, ddof=0))
        z_score = stats.norm.ppf(1 - self.confidence_level)
        var_1d = mean + z_score * std
        return float(var_1d * np.sqrt(horizon))

    # ------------------------------------------------------------------
    # Monte Carlo VaR
    # ------------------------------------------------------------------

    def calculate_monte_carlo_var(
        self,
        returns: np.ndarray,
        horizon: int = 1,
        n_simulations: int = 10_000,
        seed: Optional[int] = None,
    ) -> float:
        """Calculate VaR via Monte Carlo simulation (geometric Brownian motion).

        For a portfolio return series the drift (mu) and volatility (sigma)
        are estimated from historical data.  N price paths of length
        *horizon* are generated using:

            S_t = S_0 * exp((mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z)

        The simulated terminal returns are sorted and the alpha-percentile
        is taken as the VaR estimate.

        Parameters
        ----------
        returns : np.ndarray
            1-D array of historical daily returns.
        horizon : int
            Holding period in trading days (default 1).
        n_simulations : int
            Number of Monte Carlo scenarios (default 10 000).
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        float
            VaR estimate (negative value indicates a loss).
        """
        if len(returns) == 0:
            return 0.0

        rng = np.random.RandomState(seed)

        mu = float(np.mean(returns))
        sigma = float(np.std(returns, ddof=0))

        if sigma == 0.0:
            return mu * horizon

        dt = 1.0  # each step = 1 trading day

        # Simulate terminal portfolio values after *horizon* days.
        # Using GBM: ln(S_T/S_0) = sum of daily log-returns
        # Each daily log-return ~ N((mu - sigma^2/2)*dt, sigma^2*dt)
        z = rng.standard_normal((n_simulations, horizon))
        daily_log_returns = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
        cumulative_returns = np.sum(daily_log_returns, axis=1)

        # Convert cumulative log-returns to simple returns
        simulated_returns = np.exp(cumulative_returns) - 1.0

        percentile = (1 - self.confidence_level) * 100
        return float(np.percentile(simulated_returns, percentile))

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    def calculate_var(
        self,
        returns: np.ndarray,
        method: VaRMethod = VaRMethod.HISTORICAL,
        horizon: int = 1,
        n_simulations: int = 10_000,
        seed: Optional[int] = None,
    ) -> float:
        """Calculate VaR using the specified method.

        Parameters
        ----------
        returns : np.ndarray
            1-D array of historical returns.
        method : VaRMethod
            Estimation method (HISTORICAL, PARAMETRIC, MONTE_CARLO).
        horizon : int
            Holding period in days.
        n_simulations : int
            Scenarios for Monte Carlo (ignored by other methods).
        seed : int, optional
            Random seed for Monte Carlo reproducibility.

        Returns
        -------
        float
            VaR estimate.
        """
        if method == VaRMethod.HISTORICAL:
            return self.calculate_historical_var(returns, horizon=horizon)
        elif method == VaRMethod.PARAMETRIC:
            return self.calculate_parametric_var(returns, horizon=horizon)
        elif method == VaRMethod.MONTE_CARLO:
            return self.calculate_monte_carlo_var(
                returns,
                horizon=horizon,
                n_simulations=n_simulations,
                seed=seed,
            )
        else:
            raise ValueError(f"Unknown VaR method: {method}")

    # ------------------------------------------------------------------
    # Conditional VaR (Expected Shortfall)
    # ------------------------------------------------------------------

    def calculate_cvar(
        self,
        returns: np.ndarray,
        method: VaRMethod = VaRMethod.HISTORICAL,
        horizon: int = 1,
    ) -> float:
        """Calculate Conditional VaR (Expected Shortfall).

        CVaR is the expected loss given that the loss exceeds the VaR
        threshold.  For historical simulation it is the mean of returns
        in the tail beyond the VaR quantile.  For the parametric method
        the closed-form formula under normality is used:

            CVaR = mu - sigma * phi(z_alpha) / (1 - confidence)

        Parameters
        ----------
        returns : np.ndarray
            1-D array of historical returns.
        method : VaRMethod
            Which VaR method to use for the threshold.
        horizon : int
            Holding period in days.

        Returns
        -------
        float
            CVaR estimate (negative value indicates a loss).
        """
        if len(returns) == 0:
            return 0.0

        if method == VaRMethod.PARAMETRIC:
            return self._parametric_cvar(returns, horizon)

        # Historical / Monte-Carlo: use empirical tail mean
        var = self.calculate_var(returns, method=method, horizon=horizon)
        if horizon > 1:
            # Scale returns for multi-day comparison
            scaled_returns = returns * np.sqrt(horizon)
        else:
            scaled_returns = returns
        tail = scaled_returns[scaled_returns <= var]
        if len(tail) == 0:
            return var
        return float(np.mean(tail))

    def _parametric_cvar(self, returns: np.ndarray, horizon: int = 1) -> float:
        """Closed-form CVaR under the normal distribution assumption.

        CVaR = mu - sigma * phi(Phi^{-1}(alpha)) / alpha

        where alpha = 1 - confidence_level, phi is the standard normal
        PDF, and Phi^{-1} is the inverse CDF.
        """
        mu = float(np.mean(returns))
        sigma = float(np.std(returns, ddof=0))
        alpha = 1 - self.confidence_level
        z_alpha = stats.norm.ppf(alpha)
        cvar_1d = mu - sigma * stats.norm.pdf(z_alpha) / alpha
        return float(cvar_1d * np.sqrt(horizon))

    # ------------------------------------------------------------------
    # Back-testing (Kupiec POF test)
    # ------------------------------------------------------------------

    def backtest_var(
        self,
        returns: np.ndarray,
        window_size: int = 252,
        method: VaRMethod = VaRMethod.HISTORICAL,
        significance_level: float = 0.05,
    ) -> Dict[str, Any]:
        """Back-test VaR model using the Kupiec Proportion-of-Failures test.

        A rolling window of *window_size* days is used to estimate VaR at
        each step.  The number of VaR exceptions (actual loss exceeding
        predicted VaR) is compared to the expected rate via a likelihood-ratio
        test.

        The Kupiec LR statistic is:

            LR = -2 * ln[ (1-p)^(n-x) * p^x ] + 2 * ln[ (1-x/n)^(n-x) * (x/n)^x ]

        Under H0 (model is correct), LR ~ chi-squared(1).

        Parameters
        ----------
        returns : np.ndarray
            Full history of daily returns.
        window_size : int
            Rolling estimation window (default 252 ~ 1 trading year).
        method : VaRMethod
            VaR estimation method used in back-test.
        significance_level : float
            Significance level for the Kupiec test (default 0.05).

        Returns
        -------
        dict
            Keys: violations, total_periods, violation_rate, expected_rate,
            kupiec_statistic, p_value, pass.
        """
        n = len(returns)
        if n <= window_size:
            return {
                "violations": 0,
                "total_periods": 0,
                "violation_rate": 0.0,
                "expected_rate": 1 - self.confidence_level,
                "kupiec_statistic": 0.0,
                "p_value": 1.0,
                "pass": True,
            }

        violations = 0
        total_periods = n - window_size

        for i in range(window_size, n):
            historical_window = returns[i - window_size : i]
            var = self.calculate_var(historical_window, method=method)
            if returns[i] < var:
                violations += 1

        violation_rate = violations / total_periods if total_periods > 0 else 0.0
        expected_rate = 1 - self.confidence_level

        # Kupiec likelihood-ratio statistic
        kupiec_stat, p_value = self._kupiec_test(
            violations, total_periods, expected_rate
        )

        return {
            "violations": violations,
            "total_periods": total_periods,
            "violation_rate": violation_rate,
            "expected_rate": expected_rate,
            "kupiec_statistic": kupiec_stat,
            "p_value": p_value,
            "pass": p_value > significance_level,
        }

    @staticmethod
    def _kupiec_test(
        exceptions: int, total: int, expected_rate: float
    ) -> tuple:
        """Compute the Kupiec POF likelihood-ratio statistic and p-value.

        Parameters
        ----------
        exceptions : int
            Number of VaR breaches.
        total : int
            Total out-of-sample observations.
        expected_rate : float
            Expected exception rate (1 - confidence_level).

        Returns
        -------
        tuple[float, float]
            (LR statistic, p-value from chi-squared(1)).
        """
        if total == 0:
            return 0.0, 1.0

        p = expected_rate
        x = exceptions
        n = total

        # Observed exception rate (clamp away from 0 and 1 for log safety)
        p_hat = x / n
        p_hat = np.clip(p_hat, 1e-10, 1 - 1e-10)
        p = np.clip(p, 1e-10, 1 - 1e-10)

        # Log-likelihood under H0 (model is correct)
        ll_h0 = (n - x) * np.log(1 - p) + x * np.log(p)
        # Log-likelihood under H1 (unrestricted)
        ll_h1 = (n - x) * np.log(1 - p_hat) + x * np.log(p_hat)

        lr_stat = -2 * (ll_h0 - ll_h1)
        lr_stat = max(lr_stat, 0.0)  # numerical guard

        p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
        return float(lr_stat), float(p_value)
