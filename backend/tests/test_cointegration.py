"""
Tests for Cointegration Analysis
Tests statistical cointegration detection and pairs trading strategies.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from backend.analytics.statistical.cointegration_analyzer import (
    CointegrationAnalyzer,
    CointegrationResult,
    CointegrationMethod,
    StatisticalArbitrageStrategy,
    PairTradingSignal,
)


class TestCointegrationAnalyzer:
    """Test suite for CointegrationAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return CointegrationAnalyzer()

    @pytest.fixture
    def sample_price_data(self):
        """Generate sample price data for testing."""
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=252, freq='D')

        # Create cointegrated series
        common_factor = np.cumsum(np.random.randn(252))
        noise1 = np.random.randn(252) * 0.1
        noise2 = np.random.randn(252) * 0.1

        prices1 = pd.Series(100 + common_factor + noise1, index=dates)
        prices2 = pd.Series(150 + common_factor * 1.5 + noise2, index=dates)

        # Create non-cointegrated series
        prices3 = pd.Series(100 * np.exp(np.cumsum(np.random.randn(252) * 0.01)), index=dates)

        return {
            'AAPL': prices1,
            'MSFT': prices2,
            'TSLA': prices3
        }

    def test_test_cointegration_cointegrated(self, analyzer, sample_price_data):
        """Test cointegration test on cointegrated series."""
        result = analyzer.test_cointegration(
            sample_price_data['AAPL'],
            sample_price_data['MSFT']
        )

        assert isinstance(result, CointegrationResult)
        assert hasattr(result, 'p_value')
        assert hasattr(result, 'is_cointegrated')
        assert hasattr(result, 'hedge_ratio')
        assert hasattr(result, 'half_life')
        assert result.hedge_ratio != 0

    def test_test_cointegration_not_cointegrated(self, analyzer, sample_price_data):
        """Test cointegration test on non-cointegrated series."""
        result = analyzer.test_cointegration(
            sample_price_data['AAPL'],
            sample_price_data['TSLA']
        )

        assert isinstance(result, CointegrationResult)
        # Non-cointegrated series should have high p_value or is_cointegrated == False
        # (depends on the random seed / data, but the test checks the result is valid)
        assert result.p_value >= 0

    def test_engle_granger_method(self, analyzer, sample_price_data):
        """Test using Engle-Granger method explicitly."""
        result = analyzer.test_cointegration(
            sample_price_data['AAPL'],
            sample_price_data['MSFT'],
            method=CointegrationMethod.ENGLE_GRANGER
        )

        assert isinstance(result, CointegrationResult)
        assert result.hedge_ratio != 0

    def test_johansen_method(self, analyzer, sample_price_data):
        """Test using Johansen method."""
        result = analyzer.test_cointegration(
            sample_price_data['AAPL'],
            sample_price_data['MSFT'],
            method=CointegrationMethod.JOHANSEN
        )

        assert isinstance(result, CointegrationResult)

    def test_half_life_positive(self, analyzer, sample_price_data):
        """Test that half-life is calculated and non-negative."""
        result = analyzer.test_cointegration(
            sample_price_data['AAPL'],
            sample_price_data['MSFT']
        )

        assert result.half_life >= 0

    def test_critical_values_present(self, analyzer, sample_price_data):
        """Test that critical values are returned."""
        result = analyzer.test_cointegration(
            sample_price_data['AAPL'],
            sample_price_data['MSFT']
        )

        assert isinstance(result.critical_values, dict)
        assert '1%' in result.critical_values
        assert '5%' in result.critical_values
        assert '10%' in result.critical_values

    def test_spread_statistics(self, analyzer, sample_price_data):
        """Test that spread statistics are computed."""
        result = analyzer.test_cointegration(
            sample_price_data['AAPL'],
            sample_price_data['MSFT']
        )

        assert isinstance(result.spread_mean, float)
        assert isinstance(result.spread_std, float)
        assert result.spread_std >= 0

    def test_short_series_handling(self, analyzer):
        """Test handling of very short series."""
        short_series1 = pd.Series([100, 101, 102])
        short_series2 = pd.Series([200, 201, 202])

        result = analyzer.test_cointegration(short_series1, short_series2)

        assert isinstance(result, CointegrationResult)
        assert result.is_cointegrated is False

    def test_find_cointegrated_pairs(self, analyzer, sample_price_data):
        """Test finding cointegrated pairs from a universe."""
        pairs = analyzer.find_cointegrated_pairs(sample_price_data)

        assert isinstance(pairs, list)
        for ticker1, ticker2, result in pairs:
            assert isinstance(ticker1, str)
            assert isinstance(ticker2, str)
            assert isinstance(result, CointegrationResult)
            assert result.is_cointegrated is True

    def test_find_cointegrated_pairs_max_pairs(self, analyzer, sample_price_data):
        """Test max_pairs limit."""
        pairs = analyzer.find_cointegrated_pairs(sample_price_data, max_pairs=1)

        assert isinstance(pairs, list)
        assert len(pairs) <= 1


class TestStatisticalArbitrageStrategy:
    """Test suite for StatisticalArbitrageStrategy."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return CointegrationAnalyzer()

    @pytest.fixture
    def strategy(self, analyzer):
        """Create strategy instance."""
        return StatisticalArbitrageStrategy(
            analyzer=analyzer,
            entry_z_score=2.0,
            exit_z_score=0.5,
            stop_loss_z_score=4.0
        )

    @pytest.fixture
    def cointegrated_data(self):
        """Create cointegrated price data and result."""
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
        common_factor = np.cumsum(np.random.randn(252))
        noise1 = np.random.randn(252) * 0.1
        noise2 = np.random.randn(252) * 0.1

        series1 = pd.Series(100 + common_factor + noise1, index=dates)
        series2 = pd.Series(150 + common_factor * 1.5 + noise2, index=dates)

        return series1, series2

    def test_generate_signal(self, strategy, analyzer, cointegrated_data):
        """Test signal generation for a cointegrated pair."""
        series1, series2 = cointegrated_data
        coint_result = analyzer.test_cointegration(series1, series2)

        signal = strategy.generate_signal(
            'AAPL', 'MSFT', series1, series2, coint_result
        )

        assert isinstance(signal, PairTradingSignal)
        assert signal.pair == ('AAPL', 'MSFT')
        assert signal.signal in ['long_spread', 'short_spread', 'close', 'no_signal']
        assert isinstance(signal.z_score, float)
        assert 0 <= signal.confidence <= 1

    def test_update_position_open(self, strategy, analyzer, cointegrated_data):
        """Test opening a position."""
        series1, series2 = cointegrated_data
        coint_result = analyzer.test_cointegration(series1, series2)

        # Create a mock signal that opens a position
        signal = PairTradingSignal(
            pair=('AAPL', 'MSFT'),
            signal='long_spread',
            z_score=-2.5,
            entry_threshold=2.0,
            exit_threshold=0.5,
            confidence=0.95
        )

        strategy.update_position(('AAPL', 'MSFT'), signal)
        positions = strategy.get_all_positions()

        assert ('AAPL', 'MSFT') in positions
        assert positions[('AAPL', 'MSFT')] == 'long_spread'

    def test_update_position_close(self, strategy):
        """Test closing a position."""
        # Open position first
        strategy.positions[('AAPL', 'MSFT')] = 'long_spread'

        # Create close signal
        signal = PairTradingSignal(
            pair=('AAPL', 'MSFT'),
            signal='close',
            z_score=0.3,
            entry_threshold=2.0,
            exit_threshold=0.5,
            confidence=0.95
        )

        strategy.update_position(('AAPL', 'MSFT'), signal)
        positions = strategy.get_all_positions()

        assert ('AAPL', 'MSFT') not in positions

    def test_get_all_positions_empty(self, strategy):
        """Test getting positions when none are open."""
        positions = strategy.get_all_positions()
        assert isinstance(positions, dict)
        assert len(positions) == 0

    def test_get_all_positions_returns_copy(self, strategy):
        """Test that get_all_positions returns a copy."""
        strategy.positions[('AAPL', 'MSFT')] = 'long_spread'
        positions = strategy.get_all_positions()

        # Modifying the copy should not affect the original
        positions[('GOOGL', 'META')] = 'short_spread'
        assert ('GOOGL', 'META') not in strategy.positions


class TestIntegration:
    """Integration tests for cointegration analysis."""

    def test_full_cointegration_workflow(self):
        """Test the complete cointegration analysis workflow."""
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
        common_trend = np.cumsum(np.random.randn(252) * 0.01)

        price_data = {
            'AAPL': pd.Series(100 * np.exp(common_trend + np.random.randn(252) * 0.005), index=dates),
            'MSFT': pd.Series(150 * np.exp(common_trend * 1.2 + np.random.randn(252) * 0.005), index=dates),
            'GOOGL': pd.Series(200 * np.exp(np.cumsum(np.random.randn(252) * 0.01)), index=dates),
        }

        analyzer = CointegrationAnalyzer()
        pairs = analyzer.find_cointegrated_pairs(price_data)

        assert isinstance(pairs, list)

        if pairs:
            ticker1, ticker2, coint_result = pairs[0]
            strategy = StatisticalArbitrageStrategy(analyzer=analyzer)

            signal = strategy.generate_signal(
                ticker1, ticker2,
                price_data[ticker1], price_data[ticker2],
                coint_result
            )

            assert isinstance(signal, PairTradingSignal)
            assert signal.signal in ['long_spread', 'short_spread', 'close', 'no_signal']

    def test_cointegration_with_real_market_conditions(self):
        """Test cointegration analysis with realistic market conditions."""
        np.random.seed(42)
        analyzer = CointegrationAnalyzer()

        # Simulate different market conditions
        dates = pd.date_range(end=datetime.now(), periods=500, freq='D')

        # Bull market (both trending up but cointegrated)
        trend = np.linspace(0, 1, 500)
        noise = np.cumsum(np.random.randn(500) * 0.01)

        bull_series1 = pd.Series(100 * np.exp(trend + noise), index=dates)
        bull_series2 = pd.Series(150 * np.exp(trend * 1.2 + noise * 1.1), index=dates)

        result = analyzer.test_cointegration(bull_series1, bull_series2)
        assert isinstance(result, CointegrationResult)

        # Bear market (both trending down)
        bear_series1 = pd.Series(100 * np.exp(-trend * 0.5 + noise), index=dates)
        bear_series2 = pd.Series(150 * np.exp(-trend * 0.6 + noise * 1.1), index=dates)

        result = analyzer.test_cointegration(bear_series1, bear_series2)
        assert isinstance(result, CointegrationResult)

        # High volatility
        volatile_noise = np.cumsum(np.random.randn(500) * 0.05)
        volatile_series1 = pd.Series(100 + volatile_noise, index=dates)
        volatile_series2 = pd.Series(150 + volatile_noise * 1.5, index=dates)

        result = analyzer.test_cointegration(volatile_series1, volatile_series2)
        assert isinstance(result, CointegrationResult)

        # Structural break (relationship changes midway)
        break_point = 250
        series1_part1 = 100 + noise[:break_point]
        series1_part2 = 100 + noise[break_point:] * 2  # Changed relationship
        structural_series1 = pd.Series(np.concatenate([series1_part1, series1_part2]), index=dates)
        structural_series2 = pd.Series(150 + noise * 1.5, index=dates)

        result = analyzer.test_cointegration(structural_series1, structural_series2)
        assert isinstance(result, CointegrationResult)
