"""
Data Pipeline Integration Tests for Investment Analysis Platform
Tests data loading, processing, caching, and real-time data pipeline components.
"""

import pytest
import asyncio
import json
from datetime import datetime, date, timedelta
from typing import Dict, Any, List
from unittest.mock import AsyncMock, patch, MagicMock
from sqlalchemy.ext.asyncio import AsyncSession
import pandas as pd
import numpy as np

from backend.tasks.data_pipeline import DataPipeline
from backend.data_ingestion.alpha_vantage_client import AlphaVantageClient
from backend.data_ingestion.finnhub_client import FinnhubClient
from backend.data_ingestion.polygon_client import PolygonClient
from backend.utils.comprehensive_cache import ComprehensiveCacheManager
from backend.utils.database_query_cache import QueryCache
from backend.tasks.celery_app import celery_app
from backend.repositories import stock_repository, price_repository
from backend.config.database import get_async_db_session


class TestDataPipelineIntegration:
    """Test complete data pipeline integration with external APIs and caching."""

    @pytest.fixture
    async def data_pipeline(self):
        """Create data pipeline instance."""
        pipeline = DataPipeline()
        await pipeline.initialize()
        return pipeline

    @pytest.fixture
    async def mock_db_session(self):
        """Create mock database session."""
        session = AsyncMock(spec=AsyncSession)
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        session.close = AsyncMock()
        return session

    @pytest.fixture
    def mock_cache_manager(self):
        """Create mock cache manager."""
        cache_manager = AsyncMock(spec=ComprehensiveCacheManager)
        cache_manager.get = AsyncMock()
        cache_manager.set = AsyncMock()
        cache_manager.invalidate = AsyncMock()
        cache_manager.get_stats = AsyncMock(return_value={
            "hits": 100,
            "misses": 20,
            "hit_rate": 0.83,
            "memory_usage": 1024 * 1024
        })
        return cache_manager

    @pytest.fixture
    def sample_stock_data(self):
        """Generate sample stock data."""
        return {
            "AAPL": {
                "price": 154.25,
                "change": 2.15,
                "change_percent": 1.41,
                "volume": 45123456,
                "market_cap": 3000000000000,
                "pe_ratio": 25.5,
                "timestamp": datetime.utcnow().isoformat()
            },
            "GOOGL": {
                "price": 2850.50,
                "change": -15.25,
                "change_percent": -0.53,
                "volume": 1234567,
                "market_cap": 2000000000000,
                "pe_ratio": 22.8,
                "timestamp": datetime.utcnow().isoformat()
            }
        }

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.database
    async def test_complete_data_ingestion_pipeline(self, data_pipeline, mock_db_session, sample_stock_data):
        """Test complete data ingestion from API to database."""
        
        with patch('backend.config.database.get_async_db_session', return_value=mock_db_session):
            with patch.object(data_pipeline, '_fetch_from_apis') as mock_fetch:
                mock_fetch.return_value = sample_stock_data
                
                with patch('backend.repositories.stock_repository.bulk_upsert_prices') as mock_upsert:
                    mock_upsert.return_value = None
                    
                    # Test data ingestion
                    symbols = ["AAPL", "GOOGL"]
                    result = await data_pipeline.ingest_stock_data(symbols)
                    
                    assert result["success"] is True
                    assert result["processed_count"] == 2
                    assert "AAPL" in result["data"]
                    assert "GOOGL" in result["data"]
                    
                    # Verify API was called
                    mock_fetch.assert_called_once_with(symbols)
                    
                    # Verify database write was attempted
                    mock_upsert.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.external_api
    async def test_api_client_integration(self):
        """Test API client integration with rate limiting and error handling."""
        
        # Test Alpha Vantage client
        av_client = AlphaVantageClient(api_key="test_key")
        
        with patch('httpx.AsyncClient.get') as mock_get:
            # Mock successful API response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "Global Quote": {
                    "01. symbol": "AAPL",
                    "02. open": "150.00",
                    "03. high": "155.00",
                    "04. low": "149.00",
                    "05. price": "154.25",
                    "06. volume": "45123456",
                    "07. latest trading day": "2024-01-15",
                    "08. previous close": "152.10",
                    "09. change": "2.15",
                    "10. change percent": "1.41%"
                }
            }
            mock_get.return_value = mock_response
            
            # Test data retrieval
            data = await av_client.get_quote("AAPL")
            
            assert data is not None
            assert data["symbol"] == "AAPL"
            assert data["price"] == 154.25
            assert data["change"] == 2.15
            
            # Verify rate limiting
            assert av_client.rate_limiter is not None

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.external_api
    async def test_api_rate_limiting_integration(self):
        """Test API rate limiting across multiple clients."""
        
        clients = [
            AlphaVantageClient(api_key="test_key"),
            FinnhubClient(api_key="test_key"),
            PolygonClient(api_key="test_key")
        ]
        
        for client in clients:
            # Test rate limit enforcement
            with patch.object(client.rate_limiter, 'acquire') as mock_acquire:
                mock_acquire.return_value = True
                
                with patch('httpx.AsyncClient.get') as mock_get:
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {"test": "data"}
                    mock_get.return_value = mock_response
                    
                    # Multiple rapid requests
                    tasks = []
                    for i in range(5):
                        if hasattr(client, 'get_quote'):
                            task = client.get_quote("AAPL")
                        elif hasattr(client, 'get_stock_price'):
                            task = client.get_stock_price("AAPL")
                        else:
                            continue
                        tasks.append(task)
                    
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Verify rate limiter was called
                    assert mock_acquire.call_count >= 5

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.cache
    async def test_cache_integration_pipeline(self, data_pipeline, mock_cache_manager, sample_stock_data):
        """Test cache integration in data pipeline."""
        
        with patch.object(data_pipeline, 'cache_manager', mock_cache_manager):
            # Test cache miss -> API call -> cache set
            mock_cache_manager.get.return_value = None
            
            with patch.object(data_pipeline, '_fetch_from_apis') as mock_fetch:
                mock_fetch.return_value = sample_stock_data
                
                symbols = ["AAPL"]
                result = await data_pipeline.get_cached_or_fetch(symbols)
                
                # Verify cache get was called
                mock_cache_manager.get.assert_called()
                
                # Verify API fetch was called (cache miss)
                mock_fetch.assert_called_once_with(symbols)
                
                # Verify cache set was called
                mock_cache_manager.set.assert_called()
                
                assert result["AAPL"]["price"] == 154.25
            
            # Test cache hit
            mock_cache_manager.get.return_value = json.dumps(sample_stock_data["AAPL"])
            
            result = await data_pipeline.get_cached_or_fetch(["AAPL"])
            
            # Verify no additional API call
            mock_fetch.assert_called_once()  # Still only one call from before
            
            assert result["AAPL"]["price"] == 154.25

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.database
    async def test_database_query_cache_integration(self, mock_db_session):
        """Test database query caching integration."""
        
        query_cache = QueryCache()
        
        # Mock database query result
        mock_result = [
            MagicMock(
                symbol="AAPL",
                name="Apple Inc.",
                sector="Technology",
                market_cap=3000000000000
            )
        ]
        
        with patch.object(stock_repository, 'search_stocks', return_value=mock_result) as mock_search:
            # First query - should hit database
            result1 = await query_cache.get_or_execute(
                "search_stocks_apple",
                stock_repository.search_stocks,
                query="apple",
                session=mock_db_session
            )
            
            # Second query - should use cache
            result2 = await query_cache.get_or_execute(
                "search_stocks_apple",
                stock_repository.search_stocks,
                query="apple",
                session=mock_db_session
            )
            
            # Verify database was only called once
            mock_search.assert_called_once()
            
            # Verify results are the same
            assert len(result1) == len(result2) == 1
            assert result1[0].symbol == result2[0].symbol == "AAPL"

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.performance
    async def test_data_pipeline_performance(self, data_pipeline, sample_stock_data):
        """Test data pipeline performance under load."""
        
        # Mock high-volume data
        large_dataset = {}
        symbols = [f"STOCK{i:04d}" for i in range(1000)]  # 1000 stocks
        
        for symbol in symbols:
            large_dataset[symbol] = {
                "price": np.random.uniform(10, 500),
                "change": np.random.uniform(-10, 10),
                "volume": np.random.randint(100000, 10000000),
                "timestamp": datetime.utcnow().isoformat()
            }
        
        with patch.object(data_pipeline, '_fetch_from_apis') as mock_fetch:
            mock_fetch.return_value = large_dataset
            
            with patch('backend.repositories.stock_repository.bulk_upsert_prices') as mock_upsert:
                mock_upsert.return_value = None
                
                # Measure processing time
                start_time = datetime.utcnow()
                result = await data_pipeline.process_batch(symbols, batch_size=100)
                end_time = datetime.utcnow()
                
                processing_time = (end_time - start_time).total_seconds()
                
                # Performance assertions
                assert processing_time < 30.0, f"Processing took {processing_time}s, should be under 30s"
                assert result["processed_count"] == 1000
                assert result["success"] is True
                
                # Verify batch processing was used
                expected_batches = len(symbols) // 100
                assert mock_upsert.call_count >= expected_batches

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.database
    async def test_data_validation_pipeline(self, data_pipeline, mock_db_session):
        """Test data validation in the pipeline."""
        
        # Test with invalid data
        invalid_data = {
            "INVALID": {
                "price": -100,  # Invalid negative price
                "change": "not_a_number",  # Invalid type
                "volume": None,  # Missing required field
                "timestamp": "invalid_date"
            },
            "VALID": {
                "price": 150.50,
                "change": 2.15,
                "volume": 1000000,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        with patch.object(data_pipeline, '_fetch_from_apis') as mock_fetch:
            mock_fetch.return_value = invalid_data
            
            with patch('backend.repositories.stock_repository.bulk_upsert_prices') as mock_upsert:
                mock_upsert.return_value = None
                
                result = await data_pipeline.ingest_stock_data(["INVALID", "VALID"])
                
                # Should process valid data and reject invalid
                assert result["success"] is True
                assert result["processed_count"] == 1  # Only VALID processed
                assert result["errors"]["INVALID"] is not None
                assert "VALID" in result["data"]

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_celery_task_integration(self, sample_stock_data):
        """Test Celery task integration for background processing."""
        
        with patch('backend.tasks.data_pipeline.DataPipeline') as mock_pipeline_class:
            mock_pipeline = AsyncMock()
            mock_pipeline.ingest_stock_data.return_value = {
                "success": True,
                "processed_count": 2,
                "data": sample_stock_data
            }
            mock_pipeline_class.return_value = mock_pipeline
            
            # Import task after patching
            from backend.tasks.data_tasks import update_stock_prices
            
            # Test task execution
            result = update_stock_prices.delay(["AAPL", "GOOGL"])
            
            # Verify task was queued (in test environment)
            assert result.task_id is not None
            
            # In real scenario, would test:
            # - Task execution
            # - Error handling
            # - Retry logic
            # - Task result storage

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.cache
    async def test_cache_invalidation_pipeline(self, data_pipeline, mock_cache_manager):
        """Test cache invalidation when new data arrives."""
        
        with patch.object(data_pipeline, 'cache_manager', mock_cache_manager):
            # Simulate new data arrival
            new_data = {
                "AAPL": {
                    "price": 155.00,  # Price changed
                    "change": 3.00,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
            with patch.object(data_pipeline, '_fetch_from_apis') as mock_fetch:
                mock_fetch.return_value = new_data
                
                # Process new data
                await data_pipeline.ingest_stock_data(["AAPL"])
                
                # Verify cache invalidation was triggered
                mock_cache_manager.invalidate.assert_called()
                
                # Verify pattern-based invalidation
                invalidation_calls = mock_cache_manager.invalidate.call_args_list
                patterns = [call[0][0] for call in invalidation_calls if call[0]]
                
                # Should invalidate related cache keys
                assert any("AAPL" in pattern for pattern in patterns)

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.database
    async def test_data_consistency_pipeline(self, data_pipeline, mock_db_session):
        """Test data consistency across pipeline stages."""
        
        # Test transactional consistency
        test_data = {
            "AAPL": {"price": 154.25, "volume": 1000000},
            "GOOGL": {"price": 2850.50, "volume": 500000}
        }
        
        with patch('backend.config.database.get_async_db_session', return_value=mock_db_session):
            # Simulate database error during processing
            with patch('backend.repositories.stock_repository.bulk_upsert_prices') as mock_upsert:
                mock_upsert.side_effect = Exception("Database error")
                
                result = await data_pipeline.ingest_stock_data(["AAPL", "GOOGL"])
                
                # Should handle error gracefully
                assert result["success"] is False
                assert "error" in result
                
                # Verify rollback was called
                mock_db_session.rollback.assert_called()

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.monitoring
    async def test_pipeline_monitoring_integration(self, data_pipeline):
        """Test pipeline monitoring and metrics collection."""
        
        with patch.object(data_pipeline, 'metrics_collector') as mock_metrics:
            mock_metrics.record_processing_time = MagicMock()
            mock_metrics.record_error = MagicMock()
            mock_metrics.record_success = MagicMock()
            
            # Test successful pipeline execution
            with patch.object(data_pipeline, '_fetch_from_apis') as mock_fetch:
                mock_fetch.return_value = {"AAPL": {"price": 150}}
                
                await data_pipeline.ingest_stock_data(["AAPL"])
                
                # Verify metrics were recorded
                mock_metrics.record_processing_time.assert_called()
                mock_metrics.record_success.assert_called()
            
            # Test error scenario
            with patch.object(data_pipeline, '_fetch_from_apis') as mock_fetch:
                mock_fetch.side_effect = Exception("API Error")
                
                result = await data_pipeline.ingest_stock_data(["AAPL"])
                
                # Verify error metrics were recorded
                mock_metrics.record_error.assert_called()
                assert result["success"] is False

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_real_time_data_pipeline(self, data_pipeline):
        """Test real-time data pipeline with streaming updates."""
        
        # Mock WebSocket or streaming connection
        with patch('backend.streaming.kafka_client.KafkaClient') as mock_kafka:
            mock_consumer = AsyncMock()
            mock_kafka.return_value.get_consumer.return_value = mock_consumer
            
            # Mock streaming messages
            streaming_messages = [
                {"symbol": "AAPL", "price": 154.25, "timestamp": datetime.utcnow().isoformat()},
                {"symbol": "GOOGL", "price": 2850.50, "timestamp": datetime.utcnow().isoformat()},
            ]
            
            mock_consumer.subscribe = AsyncMock()
            mock_consumer.__aiter__ = AsyncMock(return_value=iter(streaming_messages))
            
            # Test streaming pipeline
            processed_messages = []
            async for message in mock_consumer:
                processed_messages.append(message)
                
                # Process real-time update
                await data_pipeline.process_real_time_update(message)
            
            assert len(processed_messages) == 2
            assert processed_messages[0]["symbol"] == "AAPL"
            assert processed_messages[1]["symbol"] == "GOOGL"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])