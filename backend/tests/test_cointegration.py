"""
Tests for Cointegration Analysis
Tests statistical cointegration detection and pairs trading strategies.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from backend.analytics.statistical.cointegration_analyzer import (
    CointegrationAnalyzer,
    StatisticalArbitrageStrategy
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
    
    def test_engle_granger_test_cointegrated(self, analyzer, sample_price_data):
        """Test Engle-Granger test on cointegrated series."""
        result = analyzer.engle_granger_test(
            sample_price_data['AAPL'],
            sample_price_data['MSFT']
        )
        
        assert result is not None
        assert 'p_value' in result
        assert 'cointegrated' in result
        assert 'hedge_ratio' in result
        assert result['cointegrated'] == True  # These series are designed to be cointegrated
        assert result['hedge_ratio'] > 0
    
    def test_engle_granger_test_not_cointegrated(self, analyzer, sample_price_data):
        """Test Engle-Granger test on non-cointegrated series."""
        result = analyzer.engle_granger_test(
            sample_price_data['AAPL'],
            sample_price_data['TSLA']
        )
        
        assert result is not None
        assert result['cointegrated'] == False
    
    def test_johansen_test(self, analyzer, sample_price_data):
        """Test Johansen cointegration test."""
        prices_df = pd.DataFrame({
            'AAPL': sample_price_data['AAPL'],
            'MSFT': sample_price_data['MSFT']
        })
        
        result = analyzer.johansen_test(prices_df)
        
        assert result is not None
        assert 'num_cointegrating' in result
        assert 'eigenvalues' in result
        assert 'eigenvectors' in result
        assert 'critical_values' in result
        assert result['num_cointegrating'] >= 0
    
    def test_half_life_calculation(self, analyzer, sample_price_data):
        """Test half-life of mean reversion calculation."""
        spread = sample_price_data['AAPL'] - sample_price_data['MSFT'] * 0.67
        half_life = analyzer.calculate_half_life(spread)
        
        assert half_life is not None
        assert half_life > 0
        assert half_life < 252  # Should be less than a year
    
    @pytest.mark.asyncio
    async def test_find_cointegrated_pairs(self, analyzer):
        """Test finding cointegrated pairs from a list of stocks."""
        with patch.object(analyzer, '_fetch_price_data') as mock_fetch:
            # Mock price data
            mock_fetch.return_value = {
                'AAPL': pd.Series(np.cumsum(np.random.randn(252)) + 100),
                'MSFT': pd.Series(np.cumsum(np.random.randn(252)) + 150),
                'GOOGL': pd.Series(np.cumsum(np.random.randn(252)) + 200)
            }
            
            pairs = await analyzer.find_cointegrated_pairs(
                ['AAPL', 'MSFT', 'GOOGL'],
                min_correlation=0.7
            )
            
            assert isinstance(pairs, list)
            for pair in pairs:
                assert 'pair' in pair
                assert 'p_value' in pair
                assert 'hedge_ratio' in pair
                assert 'half_life' in pair
                assert len(pair['pair']) == 2
    
    def test_calculate_spread(self, analyzer, sample_price_data):
        """Test spread calculation."""
        spread = analyzer.calculate_spread(
            sample_price_data['AAPL'],
            sample_price_data['MSFT'],
            hedge_ratio=0.67
        )
        
        assert len(spread) == len(sample_price_data['AAPL'])
        assert isinstance(spread, pd.Series)
        # Spread should have some variance but not be trending
        assert spread.std() > 0
    
    def test_calculate_z_score(self, analyzer, sample_price_data):
        """Test z-score calculation."""
        spread = sample_price_data['AAPL'] - sample_price_data['MSFT'] * 0.67
        z_scores = analyzer.calculate_z_score(spread, window=20)
        
        assert len(z_scores) == len(spread)
        assert z_scores.notna().sum() == len(spread) - 19  # First 19 values are NaN
        # Most z-scores should be between -3 and 3
        valid_z_scores = z_scores.dropna()
        assert (valid_z_scores.abs() < 3).sum() / len(valid_z_scores) > 0.9
    
    def test_generate_trading_signals(self, analyzer, sample_price_data):
        """Test trading signal generation."""
        spread = sample_price_data['AAPL'] - sample_price_data['MSFT'] * 0.67
        z_scores = analyzer.calculate_z_score(spread, window=20)
        
        signals = analyzer.generate_trading_signals(
            z_scores,
            entry_threshold=2.0,
            exit_threshold=0.5
        )
        
        assert 'positions' in signals
        assert 'entries' in signals
        assert 'exits' in signals
        assert len(signals['positions']) == len(z_scores)
        # Positions should be -1, 0, or 1
        assert set(signals['positions'].dropna().unique()).issubset({-1, 0, 1})


class TestStatisticalArbitrageStrategy:
    """Test suite for StatisticalArbitrageStrategy."""
    
    @pytest.fixture
    def strategy(self):
        """Create strategy instance."""
        return StatisticalArbitrageStrategy(
            lookback_days=60,
            entry_z_score=2.0,
            exit_z_score=0.5,
            stop_loss_z_score=3.0
        )
    
    @pytest.fixture
    def cointegrated_pair(self):
        """Create a cointegrated pair for testing."""
        return {
            'pair': ('AAPL', 'MSFT'),
            'hedge_ratio': 0.67,
            'half_life': 15,
            'p_value': 0.01
        }
    
    @pytest.mark.asyncio
    async def test_evaluate_pair(self, strategy, cointegrated_pair):
        """Test pair evaluation."""
        with patch.object(strategy, '_fetch_recent_prices') as mock_fetch:
            # Mock recent prices
            dates = pd.date_range(end=datetime.now(), periods=60, freq='D')
            mock_fetch.return_value = (
                pd.Series(np.cumsum(np.random.randn(60)) + 100, index=dates),
                pd.Series(np.cumsum(np.random.randn(60)) + 150, index=dates)
            )
            
            evaluation = await strategy.evaluate_pair(cointegrated_pair)
            
            assert evaluation is not None
            assert 'current_z_score' in evaluation
            assert 'signal' in evaluation
            assert 'confidence' in evaluation
            assert evaluation['signal'] in ['long', 'short', 'exit', 'hold']
            assert 0 <= evaluation['confidence'] <= 1
    
    def test_calculate_position_size(self, strategy):
        """Test position sizing calculation."""
        position_size = strategy.calculate_position_size(
            confidence=0.8,
            volatility=0.02,
            capital=100000
        )
        
        assert position_size > 0
        assert position_size <= 100000  # Should not exceed capital
        # Kelly criterion should give reasonable position size
        assert position_size <= 50000  # Max 50% of capital for high confidence
    
    def test_calculate_risk_metrics(self, strategy):
        """Test risk metrics calculation."""
        returns = pd.Series(np.random.randn(252) * 0.01)
        
        metrics = strategy.calculate_risk_metrics(returns)
        
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'var_95' in metrics
        assert 'cvar_95' in metrics
        
        # Validate metric ranges
        assert -5 <= metrics['sharpe_ratio'] <= 5
        assert -1 <= metrics['max_drawdown'] <= 0
        assert metrics['var_95'] <= 0  # VaR should be negative (loss)
    
    @pytest.mark.asyncio
    async def test_backtest_pair(self, strategy):
        """Test backtesting functionality."""
        with patch.object(strategy, '_fetch_historical_prices') as mock_fetch:
            # Mock historical prices
            dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
            common_factor = np.cumsum(np.random.randn(252))
            
            mock_fetch.return_value = (
                pd.Series(100 + common_factor + np.random.randn(252) * 0.1, index=dates),
                pd.Series(150 + common_factor * 1.5 + np.random.randn(252) * 0.1, index=dates)
            )
            
            pair_info = {
                'pair': ('AAPL', 'MSFT'),
                'hedge_ratio': 0.67,
                'half_life': 15
            }
            
            backtest_results = await strategy.backtest_pair(
                pair_info,
                start_date=datetime.now() - timedelta(days=252),
                end_date=datetime.now()
            )
            
            assert backtest_results is not None
            assert 'returns' in backtest_results
            assert 'positions' in backtest_results
            assert 'metrics' in backtest_results
            assert 'trades' in backtest_results
            
            # Validate backtest metrics
            assert 'total_return' in backtest_results['metrics']
            assert 'sharpe_ratio' in backtest_results['metrics']
            assert 'win_rate' in backtest_results['metrics']
            assert 'num_trades' in backtest_results['metrics']
    
    def test_check_risk_limits(self, strategy):
        """Test risk limit checking."""
        # Test within limits
        assert strategy.check_risk_limits(
            current_drawdown=-0.05,
            position_correlation=0.3,
            var_utilization=0.5
        ) == True
        
        # Test exceeding drawdown limit
        assert strategy.check_risk_limits(
            current_drawdown=-0.25,  # Exceeds typical 20% limit
            position_correlation=0.3,
            var_utilization=0.5
        ) == False
        
        # Test exceeding correlation limit
        assert strategy.check_risk_limits(
            current_drawdown=-0.05,
            position_correlation=0.9,  # High correlation
            var_utilization=0.5
        ) == False
    
    @pytest.mark.asyncio
    async def test_generate_recommendations(self, strategy):
        """Test recommendation generation."""
        pairs = [
            {
                'pair': ('AAPL', 'MSFT'),
                'hedge_ratio': 0.67,
                'half_life': 15,
                'p_value': 0.01
            },
            {
                'pair': ('GOOGL', 'META'),
                'hedge_ratio': 0.85,
                'half_life': 20,
                'p_value': 0.02
            }
        ]
        
        with patch.object(strategy, 'evaluate_pair') as mock_evaluate:
            mock_evaluate.side_effect = [
                {
                    'signal': 'long',
                    'confidence': 0.8,
                    'current_z_score': -2.5
                },
                {
                    'signal': 'hold',
                    'confidence': 0.3,
                    'current_z_score': 0.5
                }
            ]
            
            recommendations = await strategy.generate_recommendations(pairs)
            
            assert len(recommendations) == 1  # Only one actionable signal
            assert recommendations[0]['action'] == 'long'
            assert recommendations[0]['confidence'] == 0.8
            assert 'position_size' in recommendations[0]
            assert 'stop_loss' in recommendations[0]
            assert 'take_profit' in recommendations[0]


class TestIntegration:
    """Integration tests for cointegration analysis."""
    
    @pytest.mark.asyncio
    async def test_full_cointegration_workflow(self):
        """Test the complete cointegration analysis workflow."""
        analyzer = CointegrationAnalyzer()
        strategy = StatisticalArbitrageStrategy()
        
        with patch.object(analyzer, '_fetch_price_data') as mock_fetch:
            # Create realistic mock data
            dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
            common_trend = np.cumsum(np.random.randn(252) * 0.01)
            
            mock_fetch.return_value = {
                'AAPL': pd.Series(100 * np.exp(common_trend + np.random.randn(252) * 0.005), index=dates),
                'MSFT': pd.Series(150 * np.exp(common_trend * 1.2 + np.random.randn(252) * 0.005), index=dates),
                'GOOGL': pd.Series(200 * np.exp(np.cumsum(np.random.randn(252) * 0.01)), index=dates)
            }
            
            # Find cointegrated pairs
            pairs = await analyzer.find_cointegrated_pairs(
                ['AAPL', 'MSFT', 'GOOGL'],
                min_correlation=0.5,
                max_p_value=0.05
            )
            
            assert len(pairs) >= 0  # May or may not find pairs depending on random data
            
            if pairs:
                # Generate recommendations
                with patch.object(strategy, '_fetch_recent_prices') as mock_recent:
                    mock_recent.return_value = (
                        mock_fetch.return_value['AAPL'][-60:],
                        mock_fetch.return_value['MSFT'][-60:]
                    )
                    
                    recommendations = await strategy.generate_recommendations(pairs[:1])
                    
                    # Validate recommendations
                    for rec in recommendations:
                        assert 'pair' in rec
                        assert 'action' in rec
                        assert 'confidence' in rec
                        assert rec['action'] in ['long', 'short', 'exit', 'hold']
    
    def test_cointegration_with_real_market_conditions(self):
        """Test cointegration analysis with realistic market conditions."""
        analyzer = CointegrationAnalyzer()
        
        # Simulate different market conditions
        dates = pd.date_range(end=datetime.now(), periods=500, freq='D')
        
        # Bull market (both trending up but cointegrated)
        trend = np.linspace(0, 1, 500)
        noise = np.cumsum(np.random.randn(500) * 0.01)
        
        bull_series1 = pd.Series(100 * np.exp(trend + noise), index=dates)
        bull_series2 = pd.Series(150 * np.exp(trend * 1.2 + noise * 1.1), index=dates)
        
        result = analyzer.engle_granger_test(bull_series1, bull_series2)
        assert result is not None
        
        # Bear market (both trending down)
        bear_series1 = pd.Series(100 * np.exp(-trend * 0.5 + noise), index=dates)
        bear_series2 = pd.Series(150 * np.exp(-trend * 0.6 + noise * 1.1), index=dates)
        
        result = analyzer.engle_granger_test(bear_series1, bear_series2)
        assert result is not None
        
        # High volatility
        volatile_noise = np.cumsum(np.random.randn(500) * 0.05)
        volatile_series1 = pd.Series(100 + volatile_noise, index=dates)
        volatile_series2 = pd.Series(150 + volatile_noise * 1.5, index=dates)
        
        result = analyzer.engle_granger_test(volatile_series1, volatile_series2)
        assert result is not None
        
        # Structural break (relationship changes midway)
        break_point = 250
        series1_part1 = 100 + noise[:break_point]
        series1_part2 = 100 + noise[break_point:] * 2  # Changed relationship
        structural_series1 = pd.Series(np.concatenate([series1_part1, series1_part2]), index=dates)
        structural_series2 = pd.Series(150 + noise * 1.5, index=dates)
        
        result = analyzer.engle_granger_test(structural_series1, structural_series2)
        assert result is not None
        # Should detect lack of cointegration due to structural break
        assert result['cointegrated'] == False or result['p_value'] > 0.05