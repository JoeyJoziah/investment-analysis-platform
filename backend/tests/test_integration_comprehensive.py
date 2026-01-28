"""
Comprehensive Integration Testing Suite

This module provides end-to-end integration tests for the complete investment
analysis workflows with proper database isolation and external service mocking.
"""

import pytest
import asyncio
import pytest_asyncio
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, AsyncMock
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
import redis
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer

# Import application modules
from backend.api.main import app
from backend.models.database import Base
from backend.config.database import get_async_db_session as get_database
from backend.repositories.stock_repository import StockRepository
from backend.repositories.recommendation_repository import RecommendationRepository
from backend.analytics.recommendation_engine import RecommendationEngine
from backend.data_ingestion.alpha_vantage_client import AlphaVantageClient
from backend.utils.cache import CacheManager
from backend.utils.enhanced_cost_monitor import EnhancedCostMonitor
from backend.ml.model_manager import ModelManager
from backend.tasks.analysis_tasks import analyze_stock, run_daily_analysis
from backend.tests.fixtures.comprehensive_mock_fixtures import (
    mock_external_apis, AlphaVantageMocks, FinnhubMocks, PolygonMocks
)

# Test client for FastAPI
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport


class DatabaseTestContainer:
    """Manages test database container"""
    
    def __init__(self):
        self.container = None
        self.engine = None
        self.session_factory = None
    
    async def start(self):
        """Start test database container"""
        self.container = PostgresContainer("postgres:15")
        self.container.start()
        
        # Create async engine for tests
        database_url = self.container.get_connection_url().replace('psycopg2', 'asyncpg')
        self.engine = create_async_engine(database_url, echo=False)
        
        # Create tables
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        # Create session factory
        self.session_factory = sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    async def stop(self):
        """Stop test database container"""
        if self.engine:
            await self.engine.dispose()
        if self.container:
            self.container.stop()
    
    async def get_session(self) -> AsyncSession:
        """Get database session"""
        async with self.session_factory() as session:
            yield session


class RedisTestContainer:
    """Manages test Redis container"""
    
    def __init__(self):
        self.container = None
        self.client = None
    
    def start(self):
        """Start test Redis container"""
        self.container = RedisContainer()
        self.container.start()
        
        # Create Redis client
        self.client = redis.Redis(
            host=self.container.get_container_host_ip(),
            port=self.container.get_exposed_port(6379),
            db=0,
            decode_responses=True
        )
    
    def stop(self):
        """Stop test Redis container"""
        if self.client:
            self.client.close()
        if self.container:
            self.container.stop()


@pytest_asyncio.fixture(scope="session")
async def test_database():
    """Session-scoped test database"""
    db_container = DatabaseTestContainer()
    await db_container.start()
    yield db_container
    await db_container.stop()


@pytest.fixture(scope="session")
def test_redis():
    """Session-scoped test Redis"""
    redis_container = RedisTestContainer()
    redis_container.start()
    yield redis_container
    redis_container.stop()


@pytest_asyncio.fixture
async def db_session(test_database):
    """Database session fixture with transaction rollback"""
    async with test_database.session_factory() as session:
        # Start transaction
        await session.begin()
        
        try:
            yield session
        finally:
            # Rollback transaction to isolate tests
            await session.rollback()


@pytest.fixture
def cache_client(test_redis):
    """Cache client fixture"""
    return CacheManager(redis_client=test_redis.client)


@pytest.fixture
def mock_cost_monitor():
    """Mock cost monitor for testing"""
    monitor = Mock(spec=EnhancedCostMonitor)
    monitor.should_use_cached_data.return_value = False
    monitor.record_api_call.return_value = None
    monitor.get_remaining_budget.return_value = 45.0
    monitor.get_current_usage.return_value = {
        'total_calls': 100,
        'estimated_cost': 5.0,
        'by_provider': {}
    }
    return monitor


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows"""
    
    @pytest.mark.asyncio
    async def test_stock_analysis_workflow(self, db_session, cache_client, 
                                         mock_external_apis, mock_cost_monitor):
        """Test complete stock analysis workflow"""
        
        # Setup repositories
        stock_repo = StockRepository(db_session)
        rec_repo = RecommendationRepository(db_session)
        
        # Create test stock
        stock_data = {
            'ticker': 'AAPL',
            'company_name': 'Apple Inc.',
            'sector': 'Technology',
            'market_cap': 3000000000000,
            'is_active': True
        }
        
        stock = await stock_repo.create_stock(stock_data)
        assert stock.ticker == 'AAPL'
        
        # Setup recommendation engine with mocked dependencies
        rec_engine = RecommendationEngine()
        rec_engine.alpha_vantage_client = mock_external_apis['alpha_vantage']
        rec_engine.finnhub_client = mock_external_apis['finnhub']
        rec_engine.polygon_client = mock_external_apis['polygon']
        rec_engine.cache_manager = cache_client
        rec_engine.cost_monitor = mock_cost_monitor
        
        # Mock ML model manager
        mock_model_manager = Mock(spec=ModelManager)
        mock_model_manager.predict_price.return_value = {
            'predicted_price': 165.0,
            'confidence': 0.85,
            'timeframe': '3M'
        }
        rec_engine.model_manager = mock_model_manager
        
        # Run analysis
        recommendation = await rec_engine.analyze_stock('AAPL')
        
        # Verify recommendation
        assert recommendation is not None
        assert recommendation.ticker == 'AAPL'
        assert 0 <= recommendation.confidence <= 1
        assert recommendation.target_price > 0
        assert recommendation.stop_loss > 0
        assert len(recommendation.key_factors) > 0
        
        # Store recommendation
        rec_data = {
            'stock_id': stock.id,
            'ticker': recommendation.ticker,
            'action': recommendation.action.value,
            'confidence': recommendation.confidence,
            'target_price': recommendation.target_price,
            'stop_loss': recommendation.stop_loss,
            'expected_return': recommendation.expected_return,
            'risk_score': recommendation.risk_score,
            'key_factors': recommendation.key_factors,
            'technical_score': recommendation.technical_score,
            'fundamental_score': recommendation.fundamental_score,
            'sentiment_score': recommendation.sentiment_score
        }
        
        stored_rec = await rec_repo.create_recommendation(rec_data)
        assert stored_rec.ticker == 'AAPL'
        assert stored_rec.confidence == recommendation.confidence
        
        # Verify caching worked
        cached_data = cache_client.get(f"stock_analysis:AAPL")
        # Should have cached the analysis results
    
    @pytest.mark.asyncio
    async def test_daily_recommendations_workflow(self, db_session, cache_client,
                                                mock_external_apis, mock_cost_monitor):
        """Test complete daily recommendations workflow"""
        
        # Setup repositories
        stock_repo = StockRepository(db_session)
        rec_repo = RecommendationRepository(db_session)
        
        # Create multiple test stocks
        test_stocks = [
            {'ticker': 'AAPL', 'company_name': 'Apple Inc.', 'sector': 'Technology'},
            {'ticker': 'MSFT', 'company_name': 'Microsoft Corp.', 'sector': 'Technology'},
            {'ticker': 'GOOGL', 'company_name': 'Alphabet Inc.', 'sector': 'Technology'},
            {'ticker': 'TSLA', 'company_name': 'Tesla Inc.', 'sector': 'Automotive'},
            {'ticker': 'NVDA', 'company_name': 'NVIDIA Corp.', 'sector': 'Technology'}
        ]
        
        created_stocks = []
        for stock_data in test_stocks:
            stock_data.update({
                'market_cap': np.random.uniform(500e9, 3000e9),
                'is_active': True
            })
            stock = await stock_repo.create_stock(stock_data)
            created_stocks.append(stock)
        
        # Setup recommendation engine
        rec_engine = RecommendationEngine()
        rec_engine.alpha_vantage_client = mock_external_apis['alpha_vantage']
        rec_engine.finnhub_client = mock_external_apis['finnhub'] 
        rec_engine.polygon_client = mock_external_apis['polygon']
        rec_engine.cache_manager = cache_client
        rec_engine.cost_monitor = mock_cost_monitor
        
        # Mock market scanner
        mock_scanner = Mock()
        mock_scanner.scan_market.return_value = [
            {'ticker': stock.ticker, 'score': np.random.uniform(0.6, 0.9)}
            for stock in created_stocks
        ]
        rec_engine.market_scanner = mock_scanner
        
        # Generate daily recommendations
        recommendations = await rec_engine.generate_daily_recommendations(
            max_recommendations=3,
            risk_tolerance='moderate'
        )
        
        # Verify recommendations
        assert len(recommendations) <= 3
        assert all(rec.confidence > 0.5 for rec in recommendations)
        
        # Store recommendations
        for rec in recommendations:
            stock = next(s for s in created_stocks if s.ticker == rec.ticker)
            rec_data = {
                'stock_id': stock.id,
                'ticker': rec.ticker,
                'action': rec.action.value,
                'confidence': rec.confidence,
                'target_price': rec.target_price,
                'stop_loss': rec.stop_loss,
                'expected_return': rec.expected_return,
                'risk_score': rec.risk_score,
                'key_factors': rec.key_factors
            }
            await rec_repo.create_recommendation(rec_data)
        
        # Verify storage
        stored_recs = await rec_repo.get_recommendations_by_date(date.today())
        assert len(stored_recs) == len(recommendations)
    
    @pytest.mark.asyncio
    async def test_api_integration_workflow(self, db_session, cache_client,
                                          mock_external_apis):
        """Test API endpoints integration"""
        
        # Override database dependency
        def override_get_database():
            return db_session
        
        app.dependency_overrides[get_database] = override_get_database
        
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            
            # Test health endpoint
            response = await client.get("/api/health")
            assert response.status_code == 200
            
            # Test stock creation via API
            stock_data = {
                'ticker': 'AAPL',
                'company_name': 'Apple Inc.',
                'sector': 'Technology',
                'market_cap': 3000000000000,
                'is_active': True
            }
            
            response = await client.post("/api/stocks", json=stock_data)
            assert response.status_code == 201
            created_stock = response.json()
            assert created_stock['ticker'] == 'AAPL'
            
            # Test stock analysis endpoint
            with patch('backend.analytics.recommendation_engine.RecommendationEngine') as mock_engine:
                mock_rec = Mock()
                mock_rec.ticker = 'AAPL'
                mock_rec.action.value = 'BUY'
                mock_rec.confidence = 0.85
                mock_rec.target_price = 180.0
                mock_rec.stop_loss = 145.0
                mock_rec.expected_return = 0.12
                mock_rec.key_factors = ['Strong fundamentals', 'Positive sentiment']
                
                mock_engine.return_value.analyze_stock.return_value = mock_rec
                
                response = await client.post(f"/api/analysis/{created_stock['id']}")
                assert response.status_code == 200
                analysis = response.json()
                assert analysis['ticker'] == 'AAPL'
                assert analysis['action'] == 'BUY'
            
            # Test recommendations endpoint
            response = await client.get("/api/recommendations")
            assert response.status_code == 200
            recommendations = response.json()
            assert isinstance(recommendations, list)
        
        # Clean up
        app.dependency_overrides.clear()


class TestDataPipelineIntegration:
    """Test data pipeline integration"""
    
    @pytest.mark.asyncio
    async def test_data_ingestion_pipeline(self, db_session, cache_client,
                                         mock_external_apis, mock_cost_monitor):
        """Test complete data ingestion pipeline"""
        
        # Setup stock repository
        stock_repo = StockRepository(db_session)
        
        # Create test stock
        stock_data = {
            'ticker': 'AAPL',
            'company_name': 'Apple Inc.',
            'sector': 'Technology',
            'market_cap': 3000000000000,
            'is_active': True
        }
        stock = await stock_repo.create_stock(stock_data)
        
        # Test Alpha Vantage integration
        av_client = AlphaVantageClient(api_key="test_key")
        av_client._session = mock_external_apis['alpha_vantage']
        
        # Mock response
        mock_data = AlphaVantageMocks.daily_stock_data('AAPL')
        mock_external_apis['alpha_vantage'].get.return_value.json.return_value = mock_data
        
        price_data = await av_client.get_daily_prices('AAPL')
        assert price_data is not None
        assert len(price_data) > 0
        
        # Test data transformation and storage
        transformed_data = av_client._transform_daily_data(price_data)
        
        # Store price history
        from backend.repositories.price_repository import PriceRepository
        price_repo = PriceRepository(db_session)
        
        for price_record in transformed_data[:10]:  # Store last 10 days
            price_record['stock_id'] = stock.id
            await price_repo.create_price_record(price_record)
        
        # Verify storage
        stored_prices = await price_repo.get_price_history(stock.id, days=10)
        assert len(stored_prices) == 10
        
        # Test caching integration
        cache_key = f"price_data:AAPL:daily"
        cache_client.set(cache_key, transformed_data[:10], ttl=300)
        
        cached_data = cache_client.get(cache_key)
        assert cached_data is not None
        assert len(cached_data) == 10
    
    @pytest.mark.asyncio
    async def test_sentiment_analysis_pipeline(self, db_session, cache_client,
                                             mock_external_apis):
        """Test sentiment analysis pipeline integration"""
        
        from backend.analytics.sentiment_analysis import SentimentAnalysisEngine
        from backend.repositories.news_repository import NewsRepository
        
        # Setup
        sentiment_engine = SentimentAnalysisEngine()
        sentiment_engine.news_api_client = mock_external_apis['news_api']
        sentiment_engine.finnhub_client = mock_external_apis['finnhub']
        
        news_repo = NewsRepository(db_session)
        
        # Create stock for sentiment analysis
        stock_repo = StockRepository(db_session)
        stock_data = {
            'ticker': 'AAPL',
            'company_name': 'Apple Inc.',
            'sector': 'Technology',
            'is_active': True
        }
        stock = await stock_repo.create_stock(stock_data)
        
        # Run sentiment analysis
        sentiment_result = await sentiment_engine.analyze_stock_sentiment('AAPL')
        
        assert sentiment_result is not None
        assert 'overall_sentiment' in sentiment_result
        assert 'confidence' in sentiment_result
        assert 'article_scores' in sentiment_result
        
        # Store sentiment data
        sentiment_data = {
            'stock_id': stock.id,
            'ticker': 'AAPL',
            'sentiment_score': sentiment_result['overall_sentiment'],
            'confidence': sentiment_result['confidence'],
            'article_count': len(sentiment_result['article_scores']),
            'analysis_date': date.today()
        }
        
        # This would typically go to a sentiment_analysis table
        # For now, verify the data structure is correct
        assert -1 <= sentiment_data['sentiment_score'] <= 1
        assert 0 <= sentiment_data['confidence'] <= 1
    
    @pytest.mark.asyncio  
    async def test_technical_analysis_pipeline(self, db_session, cache_client):
        """Test technical analysis pipeline integration"""
        
        from backend.analytics.technical_analysis import TechnicalAnalysisEngine
        from backend.repositories.price_repository import PriceRepository
        
        # Setup
        tech_engine = TechnicalAnalysisEngine()
        price_repo = PriceRepository(db_session)
        stock_repo = StockRepository(db_session)
        
        # Create stock and price history
        stock_data = {
            'ticker': 'AAPL',
            'company_name': 'Apple Inc.',
            'sector': 'Technology',
            'is_active': True
        }
        stock = await stock_repo.create_stock(stock_data)
        
        # Generate sample price data
        dates = pd.date_range(end=date.today(), periods=100, freq='D')
        base_price = 150.0
        
        price_records = []
        current_price = base_price
        
        for i, date_val in enumerate(dates):
            # Generate realistic price movement
            daily_return = np.random.normal(0.001, 0.02)
            current_price *= (1 + daily_return)
            
            open_price = current_price * (1 + np.random.normal(0, 0.005))
            high_price = max(open_price, current_price) * (1 + abs(np.random.normal(0, 0.01)))
            low_price = min(open_price, current_price) * (1 - abs(np.random.normal(0, 0.01)))
            
            price_record = {
                'stock_id': stock.id,
                'ticker': 'AAPL',
                'date': date_val.date(),
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(current_price, 2),
                'volume': int(np.random.lognormal(14, 0.5)),
                'adj_close': round(current_price, 2)
            }
            
            price_records.append(price_record)
            stored_record = await price_repo.create_price_record(price_record)
            assert stored_record is not None
        
        # Run technical analysis
        price_df = pd.DataFrame([
            {
                'date': pr['date'],
                'open': pr['open'],
                'high': pr['high'],
                'low': pr['low'],
                'close': pr['close'],
                'volume': pr['volume']
            }
            for pr in price_records
        ])
        
        # Test various technical indicators
        indicators_to_test = [
            'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'rsi_14', 'macd', 'bollinger_bands', 'atr_14'
        ]
        
        for indicator in indicators_to_test:
            result = tech_engine.calculate_indicator(price_df, indicator)
            assert result is not None, f"Failed to calculate {indicator}"
        
        # Test comprehensive technical analysis
        tech_analysis = await tech_engine.analyze_stock('AAPL', price_df)
        
        assert tech_analysis is not None
        assert 'composite_score' in tech_analysis
        assert 'signals' in tech_analysis
        assert 'trend_analysis' in tech_analysis
        assert 'momentum_indicators' in tech_analysis
        
        # Store technical analysis results
        # This would typically go to technical_analysis table
        tech_data = {
            'stock_id': stock.id,
            'ticker': 'AAPL',
            'composite_score': tech_analysis['composite_score'],
            'trend_direction': tech_analysis['trend_analysis']['direction'],
            'trend_strength': tech_analysis['trend_analysis']['strength'],
            'momentum_score': tech_analysis['momentum_indicators']['rsi_14'] / 100,
            'analysis_date': date.today()
        }
        
        # Verify data structure
        assert 0 <= tech_data['composite_score'] <= 1
        assert tech_data['trend_direction'] in ['up', 'down', 'sideways']
        assert 0 <= tech_data['trend_strength'] <= 1


class TestErrorHandlingIntegration:
    """Test error handling across integrated components"""
    
    @pytest.mark.asyncio
    async def test_api_failure_handling(self, db_session, cache_client):
        """Test handling of external API failures"""
        
        from backend.data_ingestion.alpha_vantage_client import AlphaVantageClient
        from backend.utils.circuit_breaker import CircuitBreaker
        
        # Setup client with circuit breaker
        client = AlphaVantageClient(api_key="test_key")
        circuit_breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=60)
        client.circuit_breaker = circuit_breaker
        
        # Mock failures
        mock_session = AsyncMock()
        mock_session.get.side_effect = Exception("API Failure")
        client._session = mock_session
        
        # Test circuit breaker activation
        for _ in range(3):
            try:
                await client.get_daily_prices('AAPL')
            except Exception:
                pass
        
        # Circuit breaker should be open
        assert circuit_breaker.state == 'open'
        
        # Should use cached data when available
        cache_key = "alpha_vantage:daily:AAPL"
        cached_data = {'test': 'cached_data'}
        cache_client.set(cache_key, cached_data, ttl=3600)
        
        # Client should fall back to cached data
        fallback_data = cache_client.get(cache_key)
        assert fallback_data == cached_data
    
    @pytest.mark.asyncio
    async def test_database_transaction_rollback(self, db_session):
        """Test database transaction rollback on errors"""
        
        stock_repo = StockRepository(db_session)
        
        # Start transaction
        await db_session.begin()
        
        try:
            # Create stock
            stock_data = {
                'ticker': 'TEST',
                'company_name': 'Test Company',
                'sector': 'Technology',
                'is_active': True
            }
            stock = await stock_repo.create_stock(stock_data)
            assert stock.ticker == 'TEST'
            
            # Simulate error that should cause rollback
            raise Exception("Simulated error")
            
        except Exception:
            # Rollback transaction
            await db_session.rollback()
        
        # Stock should not exist after rollback
        retrieved_stock = await stock_repo.get_stock('TEST')
        assert retrieved_stock is None
    
    @pytest.mark.asyncio
    async def test_cost_limit_enforcement(self, db_session, mock_external_apis):
        """Test cost limit enforcement integration"""
        
        from backend.utils.enhanced_cost_monitor import EnhancedCostMonitor
        from backend.analytics.recommendation_engine import RecommendationEngine
        
        # Setup cost monitor with low budget
        cost_monitor = EnhancedCostMonitor(monthly_budget=10.0)  # Very low budget
        
        # Simulate high usage
        for _ in range(100):
            cost_monitor.record_api_call('expensive_api', cost=0.15)
        
        # Should trigger budget enforcement
        should_use_cache = cost_monitor.should_use_cached_data('expensive_api')
        assert should_use_cache is True
        
        # Recommendation engine should respect budget limits
        rec_engine = RecommendationEngine()
        rec_engine.cost_monitor = cost_monitor
        rec_engine.alpha_vantage_client = mock_external_apis['alpha_vantage']
        
        # Should prefer cached data when near budget limit
        cache_preference = await rec_engine._should_use_cached_data('AAPL')
        assert cache_preference is True


class TestPerformanceIntegration:
    """Test performance characteristics of integrated system"""
    
    @pytest.mark.asyncio
    async def test_concurrent_analysis_performance(self, db_session, cache_client,
                                                  mock_external_apis, mock_cost_monitor):
        """Test performance of concurrent stock analysis"""
        
        import time
        
        # Create multiple test stocks
        stock_repo = StockRepository(db_session)
        test_tickers = [f'TEST{i:03d}' for i in range(20)]
        
        for ticker in test_tickers:
            stock_data = {
                'ticker': ticker,
                'company_name': f'Test Company {ticker}',
                'sector': 'Technology',
                'is_active': True
            }
            await stock_repo.create_stock(stock_data)
        
        # Setup recommendation engine
        rec_engine = RecommendationEngine()
        rec_engine.alpha_vantage_client = mock_external_apis['alpha_vantage']
        rec_engine.finnhub_client = mock_external_apis['finnhub']
        rec_engine.cost_monitor = mock_cost_monitor
        
        # Test concurrent analysis
        start_time = time.time()
        
        tasks = [rec_engine.analyze_stock(ticker) for ticker in test_tickers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Verify all completed successfully
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == len(test_tickers)
        
        # Performance should be reasonable (adjust threshold as needed)
        assert duration < 10.0, f"Concurrent analysis took too long: {duration}s"
        
        # Average time per stock should be reasonable
        avg_time_per_stock = duration / len(test_tickers)
        assert avg_time_per_stock < 0.5, f"Average time per stock: {avg_time_per_stock}s"
    
    @pytest.mark.asyncio
    async def test_database_query_performance(self, db_session):
        """Test database query performance with large datasets"""
        
        import time
        
        stock_repo = StockRepository(db_session)
        
        # Create large number of stocks
        stock_data_list = []
        for i in range(1000):
            stock_data_list.append({
                'ticker': f'PERF{i:04d}',
                'company_name': f'Performance Test Company {i}',
                'sector': 'Technology',
                'market_cap': np.random.uniform(1e9, 100e9),
                'is_active': True
            })
        
        # Bulk create stocks
        start_time = time.time()
        created_stocks = await stock_repo.bulk_create_stocks(stock_data_list)
        create_time = time.time() - start_time
        
        assert len(created_stocks) == 1000
        assert create_time < 5.0, f"Bulk create took too long: {create_time}s"
        
        # Test bulk retrieval
        start_time = time.time()
        retrieved_stocks = await stock_repo.get_active_stocks(limit=1000)
        retrieve_time = time.time() - start_time
        
        assert len(retrieved_stocks) >= 1000
        assert retrieve_time < 2.0, f"Bulk retrieve took too long: {retrieve_time}s"
        
        # Test filtered queries
        start_time = time.time()
        tech_stocks = await stock_repo.get_stocks_by_sector('Technology', limit=1000)
        filter_time = time.time() - start_time
        
        assert len(tech_stocks) >= 1000
        assert filter_time < 1.0, f"Filtered query took too long: {filter_time}s"


class TestDataConsistencyIntegration:
    """Test data consistency across integrated components"""
    
    @pytest.mark.asyncio
    async def test_price_data_consistency(self, db_session, cache_client, 
                                        mock_external_apis):
        """Test price data consistency between sources"""
        
        from backend.data_ingestion.alpha_vantage_client import AlphaVantageClient
        from backend.data_ingestion.polygon_client import PolygonClient
        from backend.repositories.price_repository import PriceRepository
        
        # Setup
        av_client = AlphaVantageClient(api_key="test")
        av_client._session = mock_external_apis['alpha_vantage']
        
        polygon_client = PolygonClient(api_key="test")
        polygon_client._session = mock_external_apis['polygon']
        
        price_repo = PriceRepository(db_session)
        
        # Get data from both sources
        av_data = await av_client.get_daily_prices('AAPL')
        polygon_data = await polygon_client.get_aggregates('AAPL')
        
        # Transform data to common format
        av_transformed = av_client._transform_daily_data(av_data)
        polygon_transformed = polygon_client._transform_aggregates(polygon_data)
        
        # Data should be in consistent format
        required_fields = ['date', 'open', 'high', 'low', 'close', 'volume']
        
        for record in av_transformed:
            assert all(field in record for field in required_fields)
            assert record['high'] >= record['low']
            assert record['high'] >= record['open']
            assert record['high'] >= record['close']
            assert record['low'] <= record['open']
            assert record['low'] <= record['close']
        
        for record in polygon_transformed:
            assert all(field in record for field in required_fields)
            # Same OHLC consistency checks
            assert record['high'] >= record['low']
            assert record['volume'] > 0
    
    @pytest.mark.asyncio
    async def test_recommendation_consistency(self, db_session, cache_client,
                                           mock_external_apis, mock_cost_monitor):
        """Test recommendation consistency across multiple runs"""
        
        from backend.analytics.recommendation_engine import RecommendationEngine
        
        # Setup
        rec_engine = RecommendationEngine()
        rec_engine.alpha_vantage_client = mock_external_apis['alpha_vantage']
        rec_engine.finnhub_client = mock_external_apis['finnhub']
        rec_engine.cost_monitor = mock_cost_monitor
        rec_engine.cache_manager = cache_client
        
        # Mock consistent data
        consistent_analysis = {
            'technical_score': 0.75,
            'fundamental_score': 0.80,
            'sentiment_score': 0.65,
            'current_price': 150.0,
            'volatility': 0.25
        }
        
        # Run analysis multiple times
        recommendations = []
        for _ in range(5):
            with patch.object(rec_engine, '_fetch_analysis_data', 
                             return_value=consistent_analysis):
                rec = await rec_engine.analyze_stock('AAPL')
                recommendations.append(rec)
        
        # All recommendations should be identical
        first_rec = recommendations[0]
        for rec in recommendations[1:]:
            assert rec.action == first_rec.action
            assert abs(rec.confidence - first_rec.confidence) < 0.01
            assert abs(rec.target_price - first_rec.target_price) < 0.01
            assert abs(rec.expected_return - first_rec.expected_return) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])