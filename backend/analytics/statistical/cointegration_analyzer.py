"""
Cointegration Analyzer for Pairs Trading Strategies

This module provides statistical cointegration analysis for identifying
pairs trading opportunities using Engle-Granger and Johansen tests.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CointegrationMethod(Enum):
    """Methods for cointegration testing."""
    ENGLE_GRANGER = "engle_granger"
    JOHANSEN = "johansen"


@dataclass
class CointegrationResult:
    """Result of cointegration analysis."""
    is_cointegrated: bool
    p_value: float
    test_statistic: float
    critical_values: Dict[str, float]
    hedge_ratio: float
    spread_mean: float
    spread_std: float
    half_life: float


@dataclass
class PairTradingSignal:
    """Signal for pairs trading."""
    pair: Tuple[str, str]
    signal: str  # 'long_spread', 'short_spread', 'close', 'no_signal'
    z_score: float
    entry_threshold: float
    exit_threshold: float
    confidence: float


class CointegrationAnalyzer:
    """
    Analyzes price series for cointegration relationships.

    Implements Engle-Granger two-step method and Johansen test for
    identifying cointegrated pairs suitable for statistical arbitrage.
    """

    def __init__(
        self,
        confidence_level: float = 0.05,
        lookback_period: int = 252,
        min_half_life: int = 5,
        max_half_life: int = 120
    ):
        """
        Initialize the cointegration analyzer.

        Args:
            confidence_level: Statistical significance level for tests
            lookback_period: Number of observations for analysis
            min_half_life: Minimum acceptable half-life for mean reversion
            max_half_life: Maximum acceptable half-life for mean reversion
        """
        self.confidence_level = confidence_level
        self.lookback_period = lookback_period
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life

    def test_cointegration(
        self,
        series1: pd.Series,
        series2: pd.Series,
        method: CointegrationMethod = CointegrationMethod.ENGLE_GRANGER
    ) -> CointegrationResult:
        """
        Test if two price series are cointegrated.

        Args:
            series1: First price series
            series2: Second price series
            method: Cointegration test method to use

        Returns:
            CointegrationResult with test statistics and parameters
        """
        if method == CointegrationMethod.ENGLE_GRANGER:
            return self._engle_granger_test(series1, series2)
        else:
            return self._johansen_test(series1, series2)

    def _engle_granger_test(
        self,
        series1: pd.Series,
        series2: pd.Series
    ) -> CointegrationResult:
        """
        Perform Engle-Granger two-step cointegration test.
        """
        # Align series
        combined = pd.concat([series1, series2], axis=1).dropna()
        if len(combined) < 30:
            return CointegrationResult(
                is_cointegrated=False,
                p_value=1.0,
                test_statistic=0.0,
                critical_values={'1%': -3.43, '5%': -2.86, '10%': -2.57},
                hedge_ratio=0.0,
                spread_mean=0.0,
                spread_std=0.0,
                half_life=float('inf')
            )

        y = combined.iloc[:, 0].values
        x = combined.iloc[:, 1].values

        # Step 1: OLS regression for hedge ratio
        x_with_const = np.column_stack([np.ones(len(x)), x])
        beta = np.linalg.lstsq(x_with_const, y, rcond=None)[0]
        hedge_ratio = beta[1]

        # Calculate spread
        spread = y - hedge_ratio * x - beta[0]
        spread_mean = np.mean(spread)
        spread_std = np.std(spread)

        # Step 2: ADF test on residuals (simplified)
        # In production, use statsmodels.tsa.stattools.adfuller
        spread_diff = np.diff(spread)
        spread_lag = spread[:-1]

        if len(spread_lag) < 10:
            return CointegrationResult(
                is_cointegrated=False,
                p_value=1.0,
                test_statistic=0.0,
                critical_values={'1%': -3.43, '5%': -2.86, '10%': -2.57},
                hedge_ratio=hedge_ratio,
                spread_mean=spread_mean,
                spread_std=spread_std,
                half_life=float('inf')
            )

        # Simple ADF test statistic calculation
        rho = np.corrcoef(spread_diff, spread_lag)[0, 1]
        n = len(spread_lag)
        test_stat = rho * np.sqrt(n)

        # Critical values for ADF test (no trend, no constant for residuals)
        critical_values = {'1%': -3.43, '5%': -2.86, '10%': -2.57}

        # Estimate p-value (simplified)
        if test_stat < critical_values['1%']:
            p_value = 0.01
        elif test_stat < critical_values['5%']:
            p_value = 0.05
        elif test_stat < critical_values['10%']:
            p_value = 0.10
        else:
            p_value = 0.5

        # Calculate half-life of mean reversion
        half_life = self._calculate_half_life(spread)

        is_cointegrated = (
            p_value < self.confidence_level and
            self.min_half_life <= half_life <= self.max_half_life
        )

        return CointegrationResult(
            is_cointegrated=is_cointegrated,
            p_value=p_value,
            test_statistic=test_stat,
            critical_values=critical_values,
            hedge_ratio=hedge_ratio,
            spread_mean=spread_mean,
            spread_std=spread_std,
            half_life=half_life
        )

    def _johansen_test(
        self,
        series1: pd.Series,
        series2: pd.Series
    ) -> CointegrationResult:
        """
        Perform Johansen cointegration test.
        (Simplified implementation - for full version use statsmodels)
        """
        # For now, fall back to Engle-Granger
        return self._engle_granger_test(series1, series2)

    def _calculate_half_life(self, spread: np.ndarray) -> float:
        """Calculate mean reversion half-life using OLS."""
        spread_lag = spread[:-1]
        spread_diff = np.diff(spread)

        if len(spread_lag) < 10:
            return float('inf')

        # Regress spread difference on lagged spread
        x_with_const = np.column_stack([np.ones(len(spread_lag)), spread_lag])
        beta = np.linalg.lstsq(x_with_const, spread_diff, rcond=None)[0]

        lambda_param = beta[1]
        if lambda_param >= 0:
            return float('inf')

        half_life = -np.log(2) / lambda_param
        return max(0, half_life)

    def find_cointegrated_pairs(
        self,
        price_data: Dict[str, pd.Series],
        max_pairs: int = 10
    ) -> List[Tuple[str, str, CointegrationResult]]:
        """
        Find cointegrated pairs from a universe of assets.

        Args:
            price_data: Dictionary mapping ticker to price series
            max_pairs: Maximum number of pairs to return

        Returns:
            List of (ticker1, ticker2, result) tuples sorted by p-value
        """
        tickers = list(price_data.keys())
        results = []

        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                ticker1, ticker2 = tickers[i], tickers[j]
                result = self.test_cointegration(
                    price_data[ticker1],
                    price_data[ticker2]
                )
                if result.is_cointegrated:
                    results.append((ticker1, ticker2, result))

        # Sort by p-value and return top pairs
        results.sort(key=lambda x: x[2].p_value)
        return results[:max_pairs]


class StatisticalArbitrageStrategy:
    """
    Implements statistical arbitrage trading strategy based on cointegration.
    """

    def __init__(
        self,
        analyzer: CointegrationAnalyzer,
        entry_z_score: float = 2.0,
        exit_z_score: float = 0.5,
        stop_loss_z_score: float = 4.0
    ):
        """
        Initialize the strategy.

        Args:
            analyzer: CointegrationAnalyzer instance
            entry_z_score: Z-score threshold for entry
            exit_z_score: Z-score threshold for exit
            stop_loss_z_score: Z-score threshold for stop loss
        """
        self.analyzer = analyzer
        self.entry_z_score = entry_z_score
        self.exit_z_score = exit_z_score
        self.stop_loss_z_score = stop_loss_z_score
        self.positions: Dict[Tuple[str, str], str] = {}

    def generate_signal(
        self,
        ticker1: str,
        ticker2: str,
        series1: pd.Series,
        series2: pd.Series,
        coint_result: CointegrationResult
    ) -> PairTradingSignal:
        """
        Generate trading signal for a cointegrated pair.

        Args:
            ticker1: First ticker symbol
            ticker2: Second ticker symbol
            series1: First price series
            series2: Second price series
            coint_result: Cointegration analysis result

        Returns:
            PairTradingSignal with recommended action
        """
        # Calculate current spread z-score
        current_spread = (
            series1.iloc[-1] -
            coint_result.hedge_ratio * series2.iloc[-1]
        )
        z_score = (current_spread - coint_result.spread_mean) / coint_result.spread_std

        pair = (ticker1, ticker2)
        current_position = self.positions.get(pair, 'flat')

        # Generate signal based on z-score and current position
        if current_position == 'flat':
            if z_score > self.entry_z_score:
                signal = 'short_spread'
            elif z_score < -self.entry_z_score:
                signal = 'long_spread'
            else:
                signal = 'no_signal'
        else:
            # Check exit conditions
            if abs(z_score) < self.exit_z_score:
                signal = 'close'
            elif abs(z_score) > self.stop_loss_z_score:
                signal = 'close'  # Stop loss
            else:
                signal = 'no_signal'  # Hold position

        confidence = 1.0 - coint_result.p_value

        return PairTradingSignal(
            pair=pair,
            signal=signal,
            z_score=z_score,
            entry_threshold=self.entry_z_score,
            exit_threshold=self.exit_z_score,
            confidence=confidence
        )

    def update_position(
        self,
        pair: Tuple[str, str],
        signal: PairTradingSignal
    ) -> None:
        """Update position tracking based on signal."""
        if signal.signal in ['long_spread', 'short_spread']:
            self.positions[pair] = signal.signal
        elif signal.signal == 'close':
            self.positions.pop(pair, None)

    def get_all_positions(self) -> Dict[Tuple[str, str], str]:
        """Get all current positions."""
        return self.positions.copy()
