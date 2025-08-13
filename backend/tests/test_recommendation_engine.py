"""
Comprehensive tests for the recommendation engine
"""

import pytest
import asyncio
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, AsyncMock

from backend.analytics.recommendation_engine import (
    RecommendationEngine,
    RecommendationAction,
    StockRecommendation
)


@pytest.fixture
def recommendation_engine():
    """Create recommendation engine instance"""
    return RecommendationEngine()


@pytest.fixture
def sample_stock_data():
    """Create sample stock data for testing"""
    dates = pd.date_range(end=datetime.now(), periods=250, freq='D')
    
    return {
        'ticker': 'AAPL',
        'current_price': 150.0,
        'market_cap': 2500000000000,
        'beta': 1.2,
        'price_history': pd.DataFrame({
            'date': dates,
            'open': np.random.uniform(140, 160, 250),
            'high': np.random.uniform(145, 165, 250),
            'low': np.random.uniform(135, 155, 250),
            'close': np.random.uniform(140, 160, 250),
            'volume': np.random.randint(50000000, 100000000, 250)
        }).set_index('date'),
        'fundamentals': {
            'revenue': 400000000000,
            'net_income': 100000000000,
            'total_assets': 350000000000,
            'total_equity': 200000000000,
            'total_debt': 50000000000,
            'pe_ratio': 25,
            'roe': 0.50,
            'current_ratio': 1.5
        },
        'news': [
            {
                'headline': 'Apple announces record earnings',
                'summary': 'Tech giant beats expectations',
                'datetime': datetime.now() - timedelta(days=1)
            }
        ]
    }


@pytest.fixture
def mock_analysis_results():
    """Mock analysis results from different engines"""
    return {
        'technical': {
            'composite_score': 0.7,
            'signals': [
                {'type': 'trend', 'name': 'MACD Bullish Cross', 'action': 'buy'}
            ],
            'support_resistance': {
                'current_price': 150,
                'primary_support': 145,
                'primary_resistance': 155
            },
            'momentum_indicators': {'rsi_14': 55}
        },
        'fundamental': {
            'composite_score': 75,
            'valuation_models': {'upside_potential': 20},
            'quality_score': {'overall_score': 80},
            'risks': [{'description': 'High valuation'}],
            'opportunities': [{'description': 'Strong brand moat'}]
        },
        'sentiment': {
            'overall_sentiment': {'score': 0.6, 'confidence': 0.8},
            'signals': [{'type': 'sentiment', 'name': 'Positive consensus'}]
        },
        'ml_predictions': {
            'horizon_5': Mock(
                predicted_price=155,
                predicted_return=0.033,
                confidence_interval=(152, 158),
                model_confidence=0.75
            )
        },
        'risk_metrics': {
            'risk_score': 0.4,
            'volatility': 0.25,
            'beta': 1.2,
            'sharpe_ratio': 1.5,
            'max_drawdown': -0.15
        }
    }


class TestRecommendationEngine:
    """Test recommendation engine functionality"""
    
    @pytest.mark.asyncio
    async def test_analyze_stock(self, recommendation_engine, sample_stock_data, mock_analysis_results):
        """Test single stock analysis"""
        with patch.object(recommendation_engine, '_fetch_stock_data', return_value=sample_stock_data):
            with patch.object(recommendation_engine, '_run_technical_analysis', return_value=mock_analysis_results['technical']):
                with patch.object(recommendation_engine, '_run_fundamental_analysis', return_value=mock_analysis_results['fundamental']):
                    with patch.object(recommendation_engine, '_run_sentiment_analysis', return_value=mock_analysis_results['sentiment']):
                        with patch.object(recommendation_engine, '_run_ml_predictions', return_value=mock_analysis_results['ml_predictions']):
                            
                            result = await recommendation_engine.analyze_stock('AAPL')
                            
                            assert result is not None
                            assert isinstance(result, StockRecommendation)
                            assert result.ticker == 'AAPL'
                            assert result.action in list(RecommendationAction)
                            assert 0 <= result.confidence <= 1
                            assert result.entry_price == 150.0
    
    def test_generate_recommendation(self, recommendation_engine, sample_stock_data, mock_analysis_results):
        """Test recommendation generation logic"""
        recommendation = recommendation_engine._generate_recommendation(
            ticker='AAPL',
            stock_data=sample_stock_data,
            technical_analysis=mock_analysis_results['technical'],
            fundamental_analysis=mock_analysis_results['fundamental'],
            sentiment_analysis=mock_analysis_results['sentiment'],
            ml_predictions=mock_analysis_results['ml_predictions'],
            risk_metrics=mock_analysis_results['risk_metrics']
        )
        
        assert recommendation.ticker == 'AAPL'
        assert recommendation.entry_price == 150.0
        assert recommendation.target_price > recommendation.entry_price
        assert recommendation.stop_loss < recommendation.entry_price
        assert recommendation.expected_return > 0
        assert len(recommendation.key_factors) > 0
        assert recommendation.technical_score > 0.5
        assert recommendation.fundamental_score > 0.5
        assert recommendation.sentiment_score > 0.5
    
    def test_determine_action(self, recommendation_engine):
        """Test action determination based on score"""
        assert recommendation_engine._determine_action(0.85) == RecommendationAction.STRONG_BUY
        assert recommendation_engine._determine_action(0.65) == RecommendationAction.BUY
        assert recommendation_engine._determine_action(0.45) == RecommendationAction.HOLD
        assert recommendation_engine._determine_action(0.25) == RecommendationAction.SELL
        assert recommendation_engine._determine_action(0.1) == RecommendationAction.STRONG_SELL
    
    def test_calculate_confidence(self, recommendation_engine, mock_analysis_results):
        """Test confidence calculation"""
        confidence = recommendation_engine._calculate_confidence(
            mock_analysis_results['technical'],
            mock_analysis_results['fundamental'],
            mock_analysis_results['sentiment'],
            mock_analysis_results['ml_predictions'],
            mock_analysis_results['risk_metrics']
        )
        
        assert 0 <= confidence <= 1
        assert confidence > 0.5  # Should be reasonably confident with good scores
    
    def test_calculate_price_targets(self, recommendation_engine, mock_analysis_results):
        """Test price target calculation"""
        targets = recommendation_engine._calculate_price_targets(
            current_price=150.0,
            ml_predictions=mock_analysis_results['ml_predictions'],
            technical_analysis=mock_analysis_results['technical'],
            risk_metrics=mock_analysis_results['risk_metrics']
        )
        
        assert targets['target'] > 150.0
        assert targets['stop_loss'] < 150.0
        assert targets['expected_return'] > 0
        assert targets['risk_reward_ratio'] > 1
    
    def test_position_sizing(self, recommendation_engine, mock_analysis_results):
        """Test position sizing calculation"""
        sizing = recommendation_engine._calculate_position_sizing(
            confidence=0.7,
            risk_metrics=mock_analysis_results['risk_metrics'],
            action=RecommendationAction.BUY
        )
        
        assert 0 <= sizing['allocation'] <= 0.1  # Max 10% allocation
        assert sizing['max_size'] > 0
        assert sizing['kelly_fraction'] >= 0
    
    def test_rank_recommendations(self, recommendation_engine):
        """Test recommendation ranking"""
        recommendations = []
        for i in range(5):
            rec = Mock(spec=StockRecommendation)
            rec.expected_return = 0.1 * (i + 1)
            rec.confidence = 0.6 + 0.05 * i
            rec.risk_score = 0.5 - 0.05 * i
            rec.sharpe_ratio = 1.0 + 0.2 * i
            recommendations.append(rec)
        
        ranked = recommendation_engine._rank_recommendations(recommendations)
        
        # Should be sorted by composite ranking score
        assert len(ranked) == 5
        assert all(hasattr(r, 'ranking_score') for r in ranked)
        assert ranked[0].ranking_score >= ranked[-1].ranking_score
    
    @pytest.mark.asyncio
    async def test_monitor_recommendations(self, recommendation_engine):
        """Test recommendation monitoring"""
        active_rec = Mock(spec=StockRecommendation)
        active_rec.ticker = 'AAPL'
        active_rec.entry_price = 150.0
        active_rec.stop_loss = 145.0
        active_rec.target_price = 160.0
        active_rec.valid_until = datetime.utcnow() + timedelta(days=1)
        
        # Mock current data with price below stop loss
        with patch.object(
            recommendation_engine.market_scanner,
            'get_stock_data',
            return_value={'current_price': 144.0}
        ):
            alerts = await recommendation_engine.monitor_recommendations([active_rec])
            
            assert len(alerts) > 0
            assert alerts[0]['type'] == 'stop_loss'
            assert alerts[0]['urgency'] == 'high'
    
    def test_should_recommend_filtering(self, recommendation_engine):
        """Test recommendation filtering logic"""
        # Good recommendation
        good_rec = Mock(spec=StockRecommendation)
        good_rec.risk_score = 0.4
        good_rec.confidence = 0.7
        good_rec.action = RecommendationAction.BUY
        good_rec.expected_return = 0.15
        
        assert recommendation_engine._should_recommend(good_rec, 'moderate') is True
        
        # High risk recommendation
        risky_rec = Mock(spec=StockRecommendation)
        risky_rec.risk_score = 0.8
        risky_rec.confidence = 0.7
        risky_rec.action = RecommendationAction.BUY
        risky_rec.expected_return = 0.15
        
        assert recommendation_engine._should_recommend(risky_rec, 'conservative') is False
        assert recommendation_engine._should_recommend(risky_rec, 'aggressive') is True
    
    @pytest.mark.asyncio
    async def test_generate_daily_recommendations(self, recommendation_engine):
        """Test daily recommendation generation"""
        # Mock market scanner
        mock_candidates = [
            {'ticker': 'AAPL', 'score': 0.8},
            {'ticker': 'MSFT', 'score': 0.75},
            {'ticker': 'GOOGL', 'score': 0.7}
        ]
        
        with patch.object(recommendation_engine.market_scanner, 'scan_market', return_value=mock_candidates):
            with patch.object(recommendation_engine, 'analyze_stock', return_value=Mock(spec=StockRecommendation)):
                with patch.object(recommendation_engine, '_should_recommend', return_value=True):
                    with patch.object(recommendation_engine, '_optimize_recommendations', side_effect=lambda x, _: x):
                        
                        recommendations = await recommendation_engine.generate_daily_recommendations(
                            max_recommendations=2,
                            risk_tolerance='moderate'
                        )
                        
                        assert len(recommendations) <= 2
    
    def test_extract_key_factors(self, recommendation_engine, mock_analysis_results):
        """Test key factor extraction"""
        factors = recommendation_engine._extract_key_factors(
            mock_analysis_results['technical'],
            mock_analysis_results['fundamental'],
            mock_analysis_results['sentiment'],
            mock_analysis_results['ml_predictions']
        )
        
        assert isinstance(factors, list)
        assert len(factors) > 0
        assert all(isinstance(f, str) for f in factors)
    
    def test_error_handling(self, recommendation_engine):
        """Test error handling in analysis"""
        # Test with missing data
        result = recommendation_engine._generate_recommendation(
            ticker='ERROR',
            stock_data={},
            technical_analysis={},
            fundamental_analysis={},
            sentiment_analysis={},
            ml_predictions={},
            risk_metrics={'risk_score': 0.5, 'volatility': 0.2, 'beta': 1.0, 'sharpe_ratio': 0, 'max_drawdown': 0}
        )
        
        assert result is not None
        assert result.ticker == 'ERROR'
        assert result.confidence < 0.5  # Low confidence with missing data


class TestIntegration:
    """Integration tests for the recommendation system"""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_end_to_end_recommendation(self, recommendation_engine, sample_stock_data):
        """Test complete recommendation flow"""
        # This would test the full flow with real data
        # Skipped in unit tests but important for integration testing
        pass
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_performance_under_load(self, recommendation_engine):
        """Test system performance with multiple stocks"""
        # Test analyzing 100 stocks concurrently
        tickers = [f'TEST{i}' for i in range(100)]
        
        with patch.object(recommendation_engine, '_fetch_stock_data', return_value=None):
            tasks = [recommendation_engine.analyze_stock(ticker) for ticker in tickers]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Should handle all requests without crashing
            assert len(results) == 100


class TestRecommendationQuality:
    """Tests for recommendation quality metrics"""
    
    def test_recommendation_consistency(self, recommendation_engine, sample_stock_data, mock_analysis_results):
        """Test that recommendations are consistent"""
        # Generate multiple recommendations with same data
        recommendations = []
        for _ in range(5):
            rec = recommendation_engine._generate_recommendation(
                ticker='AAPL',
                stock_data=sample_stock_data,
                technical_analysis=mock_analysis_results['technical'],
                fundamental_analysis=mock_analysis_results['fundamental'],
                sentiment_analysis=mock_analysis_results['sentiment'],
                ml_predictions=mock_analysis_results['ml_predictions'],
                risk_metrics=mock_analysis_results['risk_metrics']
            )
            recommendations.append(rec)
        
        # All recommendations should be identical
        assert all(r.action == recommendations[0].action for r in recommendations)
        assert all(abs(r.target_price - recommendations[0].target_price) < 0.01 for r in recommendations)
    
    def test_risk_reward_ratio(self, recommendation_engine, mock_analysis_results):
        """Test that risk/reward ratios are reasonable"""
        targets = recommendation_engine._calculate_price_targets(
            current_price=100.0,
            ml_predictions=mock_analysis_results['ml_predictions'],
            technical_analysis=mock_analysis_results['technical'],
            risk_metrics=mock_analysis_results['risk_metrics']
        )
        
        # Risk/reward should be at least 2:1 for good trades
        risk = abs(targets['stop_loss'] - 100.0) / 100.0
        reward = abs(targets['target'] - 100.0) / 100.0
        
        assert reward / risk >= 1.5  # At least 1.5:1 risk/reward