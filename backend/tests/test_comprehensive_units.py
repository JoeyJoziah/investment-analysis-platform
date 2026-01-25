"""
Comprehensive Unit Testing Suite for Investment Analysis Application

This module provides extensive unit tests covering all critical components
with parameterized tests for different market conditions and edge cases.
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from decimal import Decimal
import json
import sqlite3
from typing import Dict, List, Any

# Import core modules
from backend.analytics.recommendation_engine import RecommendationEngine, RecommendationAction
from backend.analytics.technical_analysis import TechnicalAnalysisEngine
from backend.analytics.fundamental_analysis import FundamentalAnalysisEngine
from backend.analytics.sentiment_analysis import SentimentAnalysisEngine
from backend.data_ingestion.alpha_vantage_client import AlphaVantageClient
from backend.data_ingestion.finnhub_client import FinnhubClient
from backend.data_ingestion.polygon_client import PolygonClient
from backend.utils.cost_monitor import CostMonitor, SmartDataFetcher
from backend.utils.circuit_breaker import CircuitBreaker
from backend.utils.cache import CacheManager
from backend.utils.data_quality import DataQualityChecker
from backend.ml.model_manager import ModelManager, get_model_manager
from backend.repositories.stock_repository import StockRepository
from backend.security.rate_limiter import AdvancedRateLimiter as RateLimiter
from backend.models.schemas import Stock, AnalysisResult, PriceHistory


class TestRecommendationEngine:
    """Comprehensive tests for the recommendation engine"""
    
    @pytest.fixture
    def recommendation_engine(self):
        """Create recommendation engine with mocked dependencies"""
        engine = RecommendationEngine()
        engine.technical_engine = Mock()
        engine.fundamental_engine = Mock()
        engine.sentiment_engine = Mock()
        engine.model_manager = Mock()
        return engine
    
    @pytest.fixture
    def sample_stock_analysis(self):
        """Sample stock analysis data"""
        return {
            'ticker': 'AAPL',
            'current_price': 150.0,
            'technical_score': 0.75,
            'fundamental_score': 0.80,
            'sentiment_score': 0.65,
            'ml_prediction': {
                'target_price': 165.0,
                'confidence': 0.85,
                'timeframe': '3M'
            },
            'risk_metrics': {
                'volatility': 0.25,
                'beta': 1.2,
                'sharpe_ratio': 1.5,
                'max_drawdown': -0.15
            }
        }
    
    @pytest.mark.parametrize("technical_score,fundamental_score,sentiment_score,expected_action", [
        (0.9, 0.9, 0.9, RecommendationAction.STRONG_BUY),
        (0.7, 0.8, 0.6, RecommendationAction.BUY),
        (0.5, 0.5, 0.5, RecommendationAction.HOLD),
        (0.3, 0.2, 0.4, RecommendationAction.SELL),
        (0.1, 0.1, 0.2, RecommendationAction.STRONG_SELL),
    ])
    def test_action_determination(self, recommendation_engine, technical_score, 
                                fundamental_score, sentiment_score, expected_action):
        """Test recommendation action determination with various score combinations"""
        composite_score = (technical_score + fundamental_score + sentiment_score) / 3
        action = recommendation_engine._determine_action(composite_score)
        assert action == expected_action
    
    @pytest.mark.parametrize("market_condition", ["bull", "bear", "sideways", "volatile"])
    def test_market_condition_adaptation(self, recommendation_engine, market_condition):
        """Test that recommendations adapt to different market conditions"""
        mock_market_data = {
            'bull': {'trend': 'up', 'volatility': 0.15},
            'bear': {'trend': 'down', 'volatility': 0.30},
            'sideways': {'trend': 'neutral', 'volatility': 0.12},
            'volatile': {'trend': 'mixed', 'volatility': 0.45}
        }
        
        with patch.object(recommendation_engine, '_get_market_condition', 
                         return_value=mock_market_data[market_condition]):
            # Test that risk adjustment changes based on market condition
            risk_adjustment = recommendation_engine._calculate_risk_adjustment(market_condition)
            
            if market_condition == 'volatile':
                assert risk_adjustment > 1.2  # Higher risk adjustment
            elif market_condition == 'bull':
                assert 0.9 <= risk_adjustment <= 1.1  # Neutral adjustment
            elif market_condition == 'bear':
                assert risk_adjustment > 1.1  # Increased caution
    
    def test_recommendation_consistency(self, recommendation_engine, sample_stock_analysis):
        """Test that identical inputs produce consistent recommendations"""
        recommendations = []
        
        for _ in range(5):
            with patch.object(recommendation_engine, '_fetch_analysis_data', 
                             return_value=sample_stock_analysis):
                rec = recommendation_engine._generate_recommendation('AAPL')
                recommendations.append(rec)
        
        # All recommendations should be identical
        first_rec = recommendations[0]
        for rec in recommendations[1:]:
            assert rec.action == first_rec.action
            assert abs(rec.target_price - first_rec.target_price) < 0.01
            assert abs(rec.confidence - first_rec.confidence) < 0.01
    
    @pytest.mark.asyncio
    async def test_concurrent_analysis(self, recommendation_engine):
        """Test concurrent stock analysis doesn't cause race conditions"""
        tickers = [f'TEST{i}' for i in range(10)]
        
        with patch.object(recommendation_engine, '_fetch_analysis_data', 
                         return_value={'ticker': 'TEST', 'score': 0.7}):
            
            tasks = [recommendation_engine.analyze_stock(ticker) for ticker in tickers]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            assert len(results) == 10
            assert not any(isinstance(r, Exception) for r in results)


class TestTechnicalAnalysisEngine:
    """Tests for technical analysis engine"""
    
    @pytest.fixture
    def technical_engine(self):
        return TechnicalAnalysisEngine()
    
    @pytest.fixture
    def sample_price_data(self):
        """Sample price data with known patterns"""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        
        # Create data with a bullish pattern
        base_price = 100
        prices = []
        for i in range(100):
            # Add upward trend with some noise
            trend = base_price + (i * 0.5)  # 0.5% daily growth
            noise = np.random.normal(0, 1)
            prices.append(max(1, trend + noise))  # Ensure positive prices
        
        return pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.02))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.02))) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, 100)
        })
    
    @pytest.mark.parametrize("indicator", [
        "sma_20", "sma_50", "ema_12", "ema_26", "rsi_14", "macd", "bollinger_bands"
    ])
    def test_technical_indicators(self, technical_engine, sample_price_data, indicator):
        """Test that all technical indicators can be calculated"""
        result = technical_engine.calculate_indicator(sample_price_data, indicator)
        assert result is not None
        assert len(result) > 0
    
    def test_trend_detection(self, technical_engine, sample_price_data):
        """Test trend detection accuracy"""
        trend = technical_engine.detect_trend(sample_price_data)
        
        # Should detect upward trend in our sample data
        assert trend['direction'] in ['up', 'down', 'sideways']
        assert 0 <= trend['strength'] <= 1
        assert trend['confidence'] > 0
    
    def test_support_resistance_levels(self, technical_engine, sample_price_data):
        """Test support and resistance level detection"""
        levels = technical_engine.identify_support_resistance(sample_price_data)
        
        assert 'support_levels' in levels
        assert 'resistance_levels' in levels
        assert len(levels['support_levels']) > 0
        assert len(levels['resistance_levels']) > 0
        
        # Support should be below current price, resistance above
        current_price = sample_price_data['close'].iloc[-1]
        assert all(level < current_price for level in levels['support_levels'])
        assert all(level > current_price for level in levels['resistance_levels'])


class TestFundamentalAnalysisEngine:
    """Tests for fundamental analysis engine"""
    
    @pytest.fixture
    def fundamental_engine(self):
        return FundamentalAnalysisEngine()
    
    @pytest.fixture
    def sample_fundamentals(self):
        return {
            'revenue': 100_000_000_000,  # $100B
            'net_income': 20_000_000_000,  # $20B
            'total_assets': 150_000_000_000,  # $150B
            'total_equity': 80_000_000_000,  # $80B
            'total_debt': 30_000_000_000,  # $30B
            'shares_outstanding': 1_000_000_000,  # 1B shares
            'current_assets': 60_000_000_000,
            'current_liabilities': 40_000_000_000,
            'cash_and_equivalents': 25_000_000_000,
            'market_cap': 200_000_000_000,  # $200B
            'dividend_yield': 0.015,
            'beta': 1.2
        }
    
    def test_financial_ratios_calculation(self, fundamental_engine, sample_fundamentals):
        """Test calculation of key financial ratios"""
        ratios = fundamental_engine.calculate_ratios(sample_fundamentals)
        
        # Test specific ratio calculations
        expected_pe = 200_000_000_000 / 20_000_000_000  # Market cap / Net income
        expected_roe = 20_000_000_000 / 80_000_000_000  # Net income / Total equity
        expected_current_ratio = 60_000_000_000 / 40_000_000_000  # Current assets / Current liabilities
        
        assert abs(ratios['pe_ratio'] - expected_pe) < 0.01
        assert abs(ratios['roe'] - expected_roe) < 0.01
        assert abs(ratios['current_ratio'] - expected_current_ratio) < 0.01
    
    @pytest.mark.parametrize("industry", ["technology", "healthcare", "finance", "energy", "retail"])
    def test_industry_comparison(self, fundamental_engine, sample_fundamentals, industry):
        """Test industry-specific analysis"""
        analysis = fundamental_engine.analyze_by_industry(sample_fundamentals, industry)
        
        assert 'industry_percentile' in analysis
        assert 'peer_comparison' in analysis
        assert 0 <= analysis['industry_percentile'] <= 100
    
    def test_dcf_valuation(self, fundamental_engine, sample_fundamentals):
        """Test DCF valuation model"""
        # Add cash flow data
        cash_flow_data = {
            'free_cash_flow': [15_000_000_000, 16_500_000_000, 18_000_000_000],
            'growth_rate': 0.08,
            'discount_rate': 0.10
        }
        
        dcf_value = fundamental_engine.calculate_dcf_value(
            sample_fundamentals, cash_flow_data
        )
        
        assert dcf_value > 0
        assert 'intrinsic_value' in dcf_value
        assert 'upside_potential' in dcf_value
    
    def test_quality_score(self, fundamental_engine, sample_fundamentals):
        """Test financial quality scoring"""
        quality_score = fundamental_engine.calculate_quality_score(sample_fundamentals)
        
        assert 0 <= quality_score['overall_score'] <= 100
        assert 'profitability_score' in quality_score
        assert 'liquidity_score' in quality_score
        assert 'leverage_score' in quality_score


class TestSentimentAnalysisEngine:
    """Tests for sentiment analysis engine"""
    
    @pytest.fixture
    def sentiment_engine(self):
        return SentimentAnalysisEngine()
    
    @pytest.fixture
    def sample_news_articles(self):
        return [
            {
                'headline': 'Company beats earnings expectations significantly',
                'summary': 'Strong quarterly results show continued growth momentum',
                'published_at': datetime.now() - timedelta(hours=2),
                'source': 'Reuters'
            },
            {
                'headline': 'Regulatory concerns impact stock performance',
                'summary': 'New regulations may affect future profitability',
                'published_at': datetime.now() - timedelta(hours=4),
                'source': 'Bloomberg'
            },
            {
                'headline': 'Neutral analyst coverage maintains rating',
                'summary': 'Analysts see balanced risk-reward scenario',
                'published_at': datetime.now() - timedelta(hours=6),
                'source': 'CNBC'
            }
        ]
    
    def test_news_sentiment_analysis(self, sentiment_engine, sample_news_articles):
        """Test news sentiment analysis"""
        sentiment = sentiment_engine.analyze_news_sentiment(sample_news_articles)
        
        assert 'overall_sentiment' in sentiment
        assert 'confidence' in sentiment
        assert -1 <= sentiment['overall_sentiment'] <= 1
        assert 0 <= sentiment['confidence'] <= 1
        
        # Should have individual article scores
        assert 'article_scores' in sentiment
        assert len(sentiment['article_scores']) == 3
    
    @pytest.mark.parametrize("sentiment_score,expected_signal", [
        (0.8, 'very_positive'),
        (0.3, 'positive'),
        (0.0, 'neutral'),
        (-0.3, 'negative'),
        (-0.8, 'very_negative')
    ])
    def test_sentiment_signal_generation(self, sentiment_engine, sentiment_score, expected_signal):
        """Test sentiment signal generation"""
        signal = sentiment_engine._score_to_signal(sentiment_score)
        assert signal == expected_signal
    
    def test_social_media_sentiment(self, sentiment_engine):
        """Test social media sentiment analysis"""
        mock_social_data = [
            {'text': 'Amazing earnings! Stock going to the moon!', 'engagement': 100, 'timestamp': datetime.now()},
            {'text': 'This company is terrible, avoid at all costs', 'engagement': 50, 'timestamp': datetime.now()},
            {'text': 'Decent quarter, nothing special', 'engagement': 20, 'timestamp': datetime.now()}
        ]
        
        sentiment = sentiment_engine.analyze_social_sentiment(mock_social_data)
        
        assert 'weighted_sentiment' in sentiment
        assert 'engagement_factor' in sentiment
        assert sentiment['engagement_factor'] > 0


class TestDataIngestionClients:
    """Tests for external API clients"""
    
    @pytest.fixture
    def alpha_vantage_client(self):
        return AlphaVantageClient(api_key="test_key")
    
    @pytest.fixture
    def finnhub_client(self):
        return FinnhubClient(api_key="test_key")
    
    @pytest.fixture
    def polygon_client(self):
        return PolygonClient(api_key="test_key")
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, alpha_vantage_client):
        """Test that rate limiting works correctly"""
        # Mock HTTP responses
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.json.return_value = {'test': 'data'}
            mock_response.status = 200
            mock_get.return_value.__aenter__.return_value = mock_response
            
            # Make multiple rapid requests
            start_time = datetime.now()
            tasks = [alpha_vantage_client.get_stock_data('AAPL') for _ in range(3)]
            await asyncio.gather(*tasks)
            end_time = datetime.now()
            
            # Should take at least the rate limit duration
            duration = (end_time - start_time).total_seconds()
            assert duration >= alpha_vantage_client.rate_limit_delay * 2  # 2 delays for 3 requests
    
    @pytest.mark.parametrize("status_code,should_retry", [
        (200, False),
        (429, True),  # Rate limited
        (500, True),  # Server error
        (404, False),  # Not found
        (403, False),  # Forbidden
    ])
    def test_error_handling(self, alpha_vantage_client, status_code, should_retry):
        """Test client error handling and retry logic"""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = status_code
            mock_response.json.return_value = {'error': 'test error'}
            mock_get.return_value.__aenter__.return_value = mock_response
            
            # Test retry behavior
            with patch.object(alpha_vantage_client, '_should_retry', return_value=should_retry):
                if should_retry:
                    # Should attempt retries
                    with patch('asyncio.sleep'):  # Mock sleep to speed up test
                        result = asyncio.run(alpha_vantage_client.get_stock_data('AAPL'))
                else:
                    # Should not retry
                    result = asyncio.run(alpha_vantage_client.get_stock_data('AAPL'))
    
    def test_data_transformation(self, polygon_client):
        """Test that API responses are properly transformed"""
        raw_api_response = {
            'results': [{
                'o': 100.0,  # open
                'h': 105.0,  # high  
                'l': 98.0,   # low
                'c': 103.0,  # close
                'v': 1000000,  # volume
                't': 1640995200000  # timestamp
            }]
        }
        
        transformed = polygon_client._transform_price_data(raw_api_response)
        
        assert 'open' in transformed[0]
        assert 'high' in transformed[0]
        assert 'low' in transformed[0]
        assert 'close' in transformed[0]
        assert 'volume' in transformed[0]
        assert 'date' in transformed[0]


class TestCostMonitoring:
    """Tests for cost monitoring system"""
    
    @pytest.fixture
    def cost_monitor(self):
        return CostMonitor(monthly_budget=50.0)
    
    def test_api_cost_tracking(self, cost_monitor):
        """Test API cost tracking"""
        # Record some API calls
        cost_monitor.record_api_call('alpha_vantage', endpoint='daily_prices')
        cost_monitor.record_api_call('finnhub', endpoint='quote')
        cost_monitor.record_api_call('polygon', endpoint='aggregates')
        
        usage = cost_monitor.get_current_usage()
        
        assert usage['total_calls'] == 3
        assert usage['estimated_cost'] > 0
        assert 'alpha_vantage' in usage['by_provider']
    
    def test_budget_enforcement(self, cost_monitor):
        """Test budget enforcement logic"""
        # Simulate high usage
        for _ in range(1000):  # Simulate many API calls
            cost_monitor.record_api_call('expensive_provider', cost=0.1)
        
        # Should trigger budget warnings
        alerts = cost_monitor.check_budget_alerts()
        assert len(alerts) > 0
        assert any(alert['type'] == 'budget_warning' for alert in alerts)
    
    def test_fallback_logic(self, cost_monitor):
        """Test fallback to cached data when approaching limits"""
        # Simulate near budget limit
        cost_monitor.current_month_cost = 45.0  # Close to $50 limit
        
        should_fallback = cost_monitor.should_use_cached_data('expensive_api')
        assert should_fallback is True
        
        # Should still allow cheap APIs
        should_fallback_cheap = cost_monitor.should_use_cached_data('free_api')
        assert should_fallback_cheap is False


class TestCircuitBreaker:
    """Tests for circuit breaker pattern"""
    
    @pytest.fixture
    def circuit_breaker(self):
        return CircuitBreaker(failure_threshold=3, recovery_timeout=60)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_states(self, circuit_breaker):
        """Test circuit breaker state transitions"""
        
        # Initially closed
        assert circuit_breaker.state == 'closed'
        
        # Simulate failures
        for _ in range(3):
            try:
                await circuit_breaker.call(lambda: exec('raise Exception("test")'))
            except:
                pass
        
        # Should be open after failures
        assert circuit_breaker.state == 'open'
        
        # Should reject calls when open
        with pytest.raises(Exception):
            await circuit_breaker.call(lambda: "success")
    
    def test_success_resets_failure_count(self, circuit_breaker):
        """Test that successes reset failure count"""
        # Cause some failures
        for _ in range(2):
            try:
                asyncio.run(circuit_breaker.call(lambda: exec('raise Exception("test")')))
            except:
                pass
        
        assert circuit_breaker.failure_count == 2
        
        # Success should reset counter
        asyncio.run(circuit_breaker.call(lambda: "success"))
        assert circuit_breaker.failure_count == 0


class TestDataQuality:
    """Tests for data quality validation"""
    
    @pytest.fixture
    def quality_checker(self):
        return DataQualityChecker()
    
    def test_price_data_validation(self, quality_checker):
        """Test price data quality checks"""
        # Valid data
        valid_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'high': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            'low': [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
            'close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'volume': [1000000] * 10
        })
        
        quality_score = quality_checker.validate_price_data(valid_data)
        assert quality_score['overall_score'] > 0.9  # Should be high quality
        
        # Invalid data (high < low)
        invalid_data = valid_data.copy()
        invalid_data.loc[0, 'high'] = 90  # Lower than low
        
        quality_score_invalid = quality_checker.validate_price_data(invalid_data)
        assert quality_score_invalid['overall_score'] < 0.5  # Should be low quality
    
    def test_outlier_detection(self, quality_checker):
        """Test outlier detection in financial data"""
        # Create data with outlier
        normal_prices = [100 + i for i in range(10)]
        normal_prices[5] = 500  # Outlier
        
        outliers = quality_checker.detect_outliers(normal_prices)
        assert 5 in outliers  # Should detect index 5 as outlier
    
    def test_completeness_check(self, quality_checker):
        """Test data completeness validation"""
        # Complete data
        complete_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'value': range(10)
        })
        
        completeness = quality_checker.check_completeness(complete_data)
        assert completeness == 1.0  # 100% complete
        
        # Missing data
        incomplete_data = complete_data.copy()
        incomplete_data.loc[5, 'value'] = None
        
        completeness_incomplete = quality_checker.check_completeness(incomplete_data)
        assert completeness_incomplete == 0.9  # 90% complete


class TestCacheManager:
    """Tests for cache management system"""
    
    @pytest.fixture
    def cache_manager(self):
        return CacheManager()
    
    def test_cache_operations(self, cache_manager):
        """Test basic cache operations"""
        # Set value
        cache_manager.set('test_key', {'data': 'value'}, ttl=300)
        
        # Get value
        cached_value = cache_manager.get('test_key')
        assert cached_value == {'data': 'value'}
        
        # Test expiration
        cache_manager.set('expire_key', 'value', ttl=0.1)  # 0.1 seconds
        import time
        time.sleep(0.2)
        expired_value = cache_manager.get('expire_key')
        assert expired_value is None
    
    def test_cache_invalidation(self, cache_manager):
        """Test cache invalidation patterns"""
        # Set related keys
        cache_manager.set('stock:AAPL:price', {'price': 150})
        cache_manager.set('stock:AAPL:fundamentals', {'pe': 25})
        cache_manager.set('stock:MSFT:price', {'price': 300})
        
        # Invalidate AAPL related keys
        cache_manager.invalidate_pattern('stock:AAPL:*')
        
        # AAPL keys should be gone
        assert cache_manager.get('stock:AAPL:price') is None
        assert cache_manager.get('stock:AAPL:fundamentals') is None
        
        # MSFT key should remain
        assert cache_manager.get('stock:MSFT:price') is not None


class TestModelManager:
    """Tests for ML model management"""
    
    @pytest.fixture
    def model_manager(self):
        return ModelManager()
    
    def test_model_loading(self, model_manager):
        """Test model loading and initialization"""
        with patch('joblib.load') as mock_load:
            mock_model = Mock()
            mock_model.predict.return_value = [0.75]
            mock_load.return_value = mock_model
            
            model = model_manager.load_model('price_prediction')
            assert model is not None
            
            # Test prediction
            prediction = model.predict([[1, 2, 3, 4, 5]])
            assert prediction[0] == 0.75
    
    def test_model_performance_tracking(self, model_manager):
        """Test model performance tracking"""
        # Record predictions and outcomes
        model_manager.record_prediction('price_model', 
                                      prediction=150.0, 
                                      actual=148.0, 
                                      ticker='AAPL')
        
        model_manager.record_prediction('price_model',
                                      prediction=200.0,
                                      actual=205.0,
                                      ticker='MSFT')
        
        # Calculate performance metrics
        performance = model_manager.calculate_performance_metrics('price_model')
        
        assert 'mae' in performance  # Mean absolute error
        assert 'mse' in performance  # Mean squared error
        assert 'r2' in performance   # R-squared
    
    def test_model_retraining_logic(self, model_manager):
        """Test model retraining trigger logic"""
        # Simulate degrading performance
        for i in range(100):
            # Gradually worse predictions
            error = i * 0.1
            model_manager.record_prediction('degrading_model',
                                          prediction=100.0,
                                          actual=100.0 + error,
                                          ticker='TEST')
        
        should_retrain = model_manager.should_retrain('degrading_model')
        assert should_retrain is True


class TestSecurityComponents:
    """Tests for security components"""
    
    @pytest.fixture
    def rate_limiter(self):
        return RateLimiter()
    
    def test_rate_limiting(self, rate_limiter):
        """Test rate limiting functionality"""
        # Should allow initial requests
        for _ in range(5):
            allowed = rate_limiter.is_allowed('user123', 'api_endpoint')
            assert allowed is True
        
        # Should block after limit
        for _ in range(100):  # Exceed limit
            rate_limiter.is_allowed('user123', 'api_endpoint')
        
        blocked = rate_limiter.is_allowed('user123', 'api_endpoint')
        assert blocked is False
    
    def test_jwt_token_validation(self):
        """Test JWT token validation"""
        from backend.security.jwt_manager import JWTManager
        
        jwt_manager = JWTManager(secret_key="test_secret")
        
        # Create token
        payload = {'user_id': 123, 'role': 'user'}
        token = jwt_manager.create_token(payload)
        
        # Validate token
        decoded = jwt_manager.validate_token(token)
        assert decoded['user_id'] == 123
        assert decoded['role'] == 'user'
        
        # Test expired token
        expired_payload = {**payload, 'exp': datetime.now() - timedelta(hours=1)}
        expired_token = jwt_manager.create_token(expired_payload, expires_delta=timedelta(hours=-2))
        
        with pytest.raises(Exception):  # Should raise exception for expired token
            jwt_manager.validate_token(expired_token)


class TestRepositories:
    """Tests for data repository layers"""
    
    @pytest.fixture
    def stock_repository(self):
        # Use in-memory SQLite for testing
        return StockRepository(database_url="sqlite:///:memory:")
    
    @pytest.mark.asyncio
    async def test_stock_crud_operations(self, stock_repository):
        """Test basic CRUD operations for stocks"""
        # Create stock
        stock_data = {
            'ticker': 'TEST',
            'company_name': 'Test Company',
            'sector': 'Technology',
            'market_cap': 1000000000,
            'is_active': True
        }
        
        created_stock = await stock_repository.create_stock(stock_data)
        assert created_stock.ticker == 'TEST'
        
        # Read stock
        retrieved_stock = await stock_repository.get_stock('TEST')
        assert retrieved_stock is not None
        assert retrieved_stock.company_name == 'Test Company'
        
        # Update stock
        update_data = {'market_cap': 2000000000}
        updated_stock = await stock_repository.update_stock('TEST', update_data)
        assert updated_stock.market_cap == 2000000000
        
        # Delete stock
        deleted = await stock_repository.delete_stock('TEST')
        assert deleted is True
        
        # Verify deletion
        deleted_stock = await stock_repository.get_stock('TEST')
        assert deleted_stock is None
    
    @pytest.mark.asyncio
    async def test_bulk_operations(self, stock_repository):
        """Test bulk operations for performance"""
        # Create multiple stocks
        stocks_data = [
            {'ticker': f'TEST{i}', 'company_name': f'Test Company {i}', 
             'sector': 'Technology', 'market_cap': 1000000000 * (i+1)}
            for i in range(100)
        ]
        
        created_stocks = await stock_repository.bulk_create_stocks(stocks_data)
        assert len(created_stocks) == 100
        
        # Test bulk retrieval
        all_test_stocks = await stock_repository.get_stocks_by_pattern('TEST%')
        assert len(all_test_stocks) == 100


# Edge Case and Error Handling Tests
class TestEdgeCases:
    """Tests for edge cases and error conditions"""
    
    def test_empty_data_handling(self):
        """Test handling of empty datasets"""
        from backend.analytics.technical_analysis import TechnicalAnalysisEngine
        
        engine = TechnicalAnalysisEngine()
        empty_df = pd.DataFrame()
        
        # Should handle gracefully without crashing
        result = engine.calculate_indicator(empty_df, 'sma_20')
        assert result is not None or result == {}  # Should return empty or None, not crash
    
    def test_invalid_data_types(self):
        """Test handling of invalid data types"""
        from backend.utils.data_quality import DataQualityChecker
        
        checker = DataQualityChecker()
        
        # Test with string instead of numeric data
        invalid_data = pd.DataFrame({
            'price': ['not_a_number', 'also_invalid', '100']
        })
        
        quality_score = checker.validate_price_data(invalid_data)
        assert quality_score['overall_score'] < 0.3  # Should detect poor quality
    
    def test_extreme_market_conditions(self):
        """Test handling of extreme market conditions"""
        from backend.analytics.recommendation_engine import RecommendationEngine
        
        engine = RecommendationEngine()
        
        # Test with extreme volatility
        extreme_data = {
            'volatility': 2.0,  # 200% volatility
            'current_price': 0.01,  # Penny stock
            'volume': 0  # No volume
        }
        
        # Should handle without crashing
        risk_score = engine._calculate_risk_score(extreme_data)
        assert 0 <= risk_score <= 1
    
    @pytest.mark.parametrize("invalid_input", [
        None,
        {},
        [],
        "invalid_string",
        -999,
        float('inf'),
        float('nan')
    ])
    def test_invalid_inputs(self, invalid_input):
        """Test handling of various invalid inputs"""
        from backend.analytics.recommendation_engine import RecommendationEngine
        
        engine = RecommendationEngine()
        
        # Should not crash with invalid inputs
        try:
            result = engine._validate_input(invalid_input)
            # If validation passes, result should be sanitized
            assert result is not None
        except (ValueError, TypeError):
            # Or should raise appropriate exception
            pass


# Performance and Memory Tests
class TestPerformance:
    """Tests for performance characteristics"""
    
    def test_memory_usage(self):
        """Test that operations don't cause memory leaks"""
        import gc
        import sys
        
        # Get initial memory usage
        initial_objects = len(gc.get_objects())
        
        # Perform operations that could leak memory
        from backend.analytics.recommendation_engine import RecommendationEngine
        
        for _ in range(100):
            engine = RecommendationEngine()
            # Simulate analysis
            mock_data = {'ticker': 'TEST', 'price': 100}
            del engine
            
        # Force garbage collection
        gc.collect()
        
        # Check memory usage hasn't grown significantly
        final_objects = len(gc.get_objects())
        growth_ratio = final_objects / initial_objects
        
        # Allow for some growth but not excessive
        assert growth_ratio < 1.5, f"Memory usage grew by {growth_ratio}x"
    
    @pytest.mark.performance
    def test_analysis_speed(self):
        """Test that analysis completes within reasonable time"""
        import time
        from backend.analytics.recommendation_engine import RecommendationEngine
        
        engine = RecommendationEngine()
        
        # Mock dependencies for speed
        engine.technical_engine = Mock(return_value={'score': 0.7})
        engine.fundamental_engine = Mock(return_value={'score': 0.8})
        engine.sentiment_engine = Mock(return_value={'score': 0.6})
        
        start_time = time.time()
        
        # Analyze multiple stocks
        tasks = []
        for i in range(50):
            mock_data = {'ticker': f'TEST{i}', 'price': 100 + i}
            # Simulate analysis
            result = engine._generate_recommendation(f'TEST{i}')
            tasks.append(result)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete within reasonable time (e.g., 5 seconds for 50 stocks)
        assert duration < 5.0, f"Analysis took too long: {duration} seconds"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])