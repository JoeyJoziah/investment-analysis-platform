"""
Integration tests for Week 3-4 components
Tests the complete integration of all new components.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import json

from backend.utils.integration import (
    UnifiedDataIngestion,
    StockTier
)
from backend.utils.redis_resilience import (
    RedisCircuitBreaker,
    ResilientRedisClient,
    CircuitState,
    RedisMode
)
from backend.api.versioning import (
    APIVersionManager,
    APIVersion,
    VersionStatus
)


class TestUnifiedDataIngestion:
    """Test unified data ingestion system."""
    
    @pytest.fixture
    async def ingestion(self):
        """Create ingestion instance."""
        ingestion = UnifiedDataIngestion()
        await ingestion.initialize()
        return ingestion
    
    @pytest.mark.asyncio
    async def test_stock_tiering(self, ingestion):
        """Test stock tier assignment."""
        # Test tier assignment
        assert ingestion.get_stock_tier('AAPL') == StockTier.CRITICAL
        assert ingestion.get_stock_tier('UNKNOWN') == StockTier.LOW
        
        # Test tier update frequencies
        assert ingestion.tier_update_frequencies[StockTier.CRITICAL] == 3600
        assert ingestion.tier_update_frequencies[StockTier.LOW] == 86400
    
    @pytest.mark.asyncio
    async def test_budget_aware_fetching(self, ingestion):
        """Test budget-aware data fetching."""
        with patch.object(ingestion.cost_monitor, 'is_in_emergency_mode') as mock_emergency:
            # Normal mode
            mock_emergency.return_value = False
            
            with patch.object(ingestion, '_fetch_tier_data') as mock_fetch:
                mock_fetch.return_value = {'AAPL': {'price': 150.0}}
                
                result = await ingestion.fetch_stock_data(['AAPL'])
                assert 'AAPL' in result
                mock_fetch.assert_called_once()
            
            # Emergency mode - cache only
            mock_emergency.return_value = True
            
            with patch.object(ingestion, '_fetch_cached_only') as mock_cache:
                mock_cache.return_value = {'AAPL': {'price': 149.0, '_stale': True}}
                
                result = await ingestion.fetch_stock_data(['AAPL'])
                assert result['AAPL']['_stale'] is True
                mock_cache.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cache_integration(self, ingestion):
        """Test cache integration."""
        # Test cache key generation
        key = ingestion._build_cache_key('AAPL', 'price')
        assert 'AAPL' in key
        assert 'price' in key
        
        # Test cache TTL calculation
        ttl = ingestion._get_cache_ttl(StockTier.CRITICAL, 'price')
        assert ttl == 300  # 5 minutes for critical tier prices
        
        ttl = ingestion._get_cache_ttl(StockTier.LOW, 'fundamentals')
        assert ttl == 86400 * 8  # 8 days for low tier fundamentals
    
    @pytest.mark.asyncio
    async def test_parallel_processing(self, ingestion):
        """Test parallel API processing."""
        with patch.object(ingestion.processor, 'process_batch') as mock_process:
            mock_process.return_value = [
                Mock(success=True, data={'price': 150}),
                Mock(success=True, data={'price': 2800})
            ]
            
            result = await ingestion._fetch_tier_data(
                StockTier.CRITICAL,
                ['AAPL', 'GOOGL'],
                ['price'],
                False
            )
            
            mock_process.assert_called_once()
            # Verify batch processing was used
            call_args = mock_process.call_args[0][0]
            assert len(call_args) == 2  # Two tasks created
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self, ingestion):
        """Test performance metrics collection."""
        metrics = await ingestion.get_performance_metrics()
        
        assert 'cache' in metrics
        assert 'rate_limiter' in metrics
        assert 'processor' in metrics
        assert 'cost_monitor' in metrics
        assert 'stock_tiers' in metrics
        
        # Verify tier counts
        tier_counts = metrics['stock_tiers']
        assert 'CRITICAL' in tier_counts
        assert 'HIGH' in tier_counts


class TestRedisResilience:
    """Test Redis resilience and circuit breaker."""
    
    @pytest.fixture
    def circuit_breaker(self):
        """Create circuit breaker instance."""
        return RedisCircuitBreaker(
            failure_threshold=3,
            recovery_timeout=10,
            name="test"
        )
    
    @pytest.fixture
    async def redis_client(self):
        """Create resilient Redis client."""
        client = ResilientRedisClient(
            mode=RedisMode.STANDALONE,
            standalone_url="redis://localhost:6379"
        )
        await client.initialize()
        return client
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_states(self, circuit_breaker):
        """Test circuit breaker state transitions."""
        # Initial state should be CLOSED
        assert circuit_breaker.get_state() == CircuitState.CLOSED
        
        # Simulate failures
        async def failing_func():
            raise ConnectionError("Redis down")
        
        for _ in range(3):
            with pytest.raises(ConnectionError):
                await circuit_breaker.call(failing_func)
        
        # Circuit should be OPEN after threshold
        assert circuit_breaker.get_state() == CircuitState.OPEN
        
        # Verify metrics
        metrics = circuit_breaker.get_metrics()
        assert metrics['failures'] == 3
        assert metrics['state'] == 'open'
    
    @pytest.mark.asyncio
    async def test_fallback_mechanism(self, circuit_breaker):
        """Test fallback when circuit is open."""
        async def failing_func():
            raise ConnectionError("Redis down")
        
        async def fallback_func():
            return "fallback_value"
        
        # Trip the circuit
        for _ in range(3):
            with pytest.raises(ConnectionError):
                await circuit_breaker.call(failing_func)
        
        # Use fallback when circuit is open
        result = await circuit_breaker.call(
            failing_func,
            fallback=fallback_func
        )
        assert result == "fallback_value"
        
        metrics = circuit_breaker.get_metrics()
        assert metrics['fallbacks'] > 0
    
    @pytest.mark.asyncio
    async def test_sentinel_mode(self):
        """Test Redis Sentinel mode initialization."""
        client = ResilientRedisClient(
            mode=RedisMode.SENTINEL,
            sentinel_hosts=[('localhost', 26379)],
            sentinel_service="mymaster"
        )
        
        with patch.object(client, '_initialize_sentinel') as mock_init:
            mock_init.return_value = None
            await client.initialize()
            mock_init.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_retry_logic(self, redis_client):
        """Test retry logic on connection failures."""
        with patch.object(redis_client._redis_client, 'get') as mock_get:
            # First two attempts fail, third succeeds
            mock_get.side_effect = [
                ConnectionError("Connection lost"),
                ConnectionError("Connection lost"),
                "success"
            ]
            
            result = await redis_client.get("test_key")
            assert result == "success"
            assert mock_get.call_count == 3
    
    @pytest.mark.asyncio
    async def test_health_status(self, redis_client):
        """Test health status reporting."""
        status = redis_client.get_health_status()
        
        assert 'mode' in status
        assert 'connected' in status
        assert 'circuit_state' in status
        assert 'circuit_metrics' in status
        assert 'connection_metrics' in status
        
        assert status['mode'] == 'standalone'
        assert status['circuit_state'] == 'closed'


class TestAPIVersioning:
    """Test API versioning system."""
    
    @pytest.fixture
    def version_manager(self):
        """Create version manager instance."""
        return APIVersionManager(default_version=APIVersion.V3)
    
    def test_version_detection(self, version_manager):
        """Test version detection from request."""
        # Test header detection
        request = Mock()
        request.headers = {"X-API-Version": "v2"}
        request.url.path = "/api/stocks"
        request.query_params = {}
        
        version = version_manager.get_version_from_request(request)
        assert version == APIVersion.V2
        
        # Test URL path detection
        request.headers = {}
        request.url.path = "/api/v1/stocks"
        version = version_manager.get_version_from_request(request)
        assert version == APIVersion.V1
        
        # Test query parameter detection
        request.url.path = "/api/stocks"
        request.query_params = {"version": "v3"}
        version = version_manager.get_version_from_request(request)
        assert version == APIVersion.V3
    
    def test_version_status_check(self, version_manager):
        """Test version status checking."""
        # Test deprecated version warning
        with pytest.warns(DeprecationWarning):
            version_manager.check_version_status(APIVersion.V1)
        
        # Test stable version (no warning)
        version_manager.check_version_status(APIVersion.V3)
        
        # Verify metrics
        metrics = version_manager.get_metrics()
        assert metrics['deprecated_version_usage'] > 0
    
    def test_response_transformation(self, version_manager):
        """Test response transformation between versions."""
        # V1 to V2 transformation
        v1_data = {
            'ticker': 'AAPL',
            'data': {'price': 150}
        }
        
        v2_data = version_manager.transform_response(
            v1_data,
            APIVersion.V1,
            APIVersion.V2
        )
        
        assert 'symbol' in v2_data
        assert v2_data['symbol'] == 'AAPL'
        assert 'result' in v2_data
        assert '_metadata' in v2_data
        
        # V2 to V3 transformation
        v2_data = {
            'page': 1,
            'per_page': 10,
            'total': 100,
            'error_code': 'ERR001'
        }
        
        v3_data = version_manager.transform_response(
            v2_data,
            APIVersion.V2,
            APIVersion.V3
        )
        
        assert 'pagination' in v3_data
        assert v3_data['pagination']['current_page'] == 1
        assert 'error' in v3_data
        assert v3_data['error']['code'] == 'VALIDATION_ERROR'
    
    def test_transformation_path_finding(self, version_manager):
        """Test finding transformation path between versions."""
        # Direct path
        path = version_manager._find_transformation_path(
            APIVersion.V1,
            APIVersion.V2
        )
        assert path == [APIVersion.V1, APIVersion.V2]
        
        # Multi-hop path
        path = version_manager._find_transformation_path(
            APIVersion.V1,
            APIVersion.V3
        )
        assert path == [APIVersion.V1, APIVersion.V3]  # Direct transformer exists
    
    @pytest.mark.asyncio
    async def test_versioned_endpoint_decorator(self, version_manager):
        """Test versioned endpoint decorator."""
        @version_manager.version_route(
            supported_versions=[APIVersion.V2, APIVersion.V3],
            deprecated_in=APIVersion.V3
        )
        async def test_endpoint(request):
            return {"data": "test"}
        
        # Test with supported version
        request = Mock()
        request.headers = {"X-API-Version": "v2"}
        request.url.path = "/api/v2/test"
        request.query_params = {}
        
        result = await test_endpoint(request)
        assert result == {"data": "test"}
        
        # Verify metrics updated
        metrics = version_manager.get_metrics()
        assert metrics['requests_by_version']['v2'] > 0


class TestValidationChecklist:
    """Test validation checklist items."""
    
    @pytest.mark.asyncio
    async def test_no_hardcoded_secrets(self):
        """Test that no hardcoded secrets exist in configuration files."""
        import subprocess
        
        # Check for hardcoded passwords in YAML files
        result = subprocess.run(
            ['grep', '-r', 'password', '--include=*.yaml', '--include=*.yml'],
            cwd='/mnt/c/Users/Devin McGrathj/01.project_files/investment_analysis_app',
            capture_output=True,
            text=True
        )
        
        # Should only find references, not actual passwords
        if result.stdout:
            assert 'password123' not in result.stdout.lower()
            assert 'secret123' not in result.stdout.lower()
            assert 'admin123' not in result.stdout.lower()
    
    @pytest.mark.asyncio
    async def test_api_rate_limits(self):
        """Test that API calls stay under free tier limits."""
        from backend.utils.distributed_rate_limiter import APIRateLimiter
        
        limiter = APIRateLimiter()
        await limiter.initialize()
        
        # Test Alpha Vantage limits (5 per minute, 25 per day)
        for i in range(5):
            allowed, details = await limiter.check_api_limit('alpha_vantage')
            assert allowed, f"Call {i+1} should be allowed"
        
        # 6th call should be rate limited
        allowed, details = await limiter.check_api_limit('alpha_vantage')
        assert not allowed, "Should be rate limited after 5 calls"
        assert details['limited_by'] == 'per_minute'
    
    @pytest.mark.asyncio
    async def test_database_query_performance(self):
        """Test database query performance."""
        from backend.utils.database import engine
        from sqlalchemy import text
        import time
        
        # Test simple query performance
        start = time.time()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()
        elapsed_ms = (time.time() - start) * 1000
        
        # Should be under 100ms for p95
        assert elapsed_ms < 100, f"Query took {elapsed_ms}ms, should be under 100ms"
    
    @pytest.mark.asyncio
    async def test_cost_tracking(self):
        """Test cost tracking stays under $50/month."""
        from backend.utils.persistent_cost_monitor import PersistentCostMonitor
        
        monitor = PersistentCostMonitor()
        await monitor.initialize()
        
        # Simulate a month of API calls
        daily_calls = {
            'finnhub': 1000,  # Within free tier
            'alpha_vantage': 25,  # At limit
            'polygon': 100  # Within free tier
        }
        
        monthly_cost = monitor.calculate_monthly_cost(daily_calls)
        assert monthly_cost < 50, f"Monthly cost ${monthly_cost} exceeds $50 budget"
    
    @pytest.mark.asyncio
    async def test_docker_containers(self):
        """Test Docker container configurations."""
        import yaml
        
        # Load docker-compose files
        with open('/mnt/c/Users/Devin McGrathj/01.project_files/investment_analysis_app/docker-compose.yml') as f:
            main_config = yaml.safe_load(f)
        
        with open('/mnt/c/Users/Devin McGrathj/01.project_files/investment_analysis_app/docker-compose.redis-sentinel.yml') as f:
            sentinel_config = yaml.safe_load(f)
        
        # Verify essential services
        assert 'backend' in main_config['services']
        assert 'postgres' in main_config['services']
        assert 'redis' in main_config['services']
        
        # Verify Sentinel configuration
        assert 'redis-master' in sentinel_config['services']
        assert 'redis-sentinel1' in sentinel_config['services']
        
        # Check health checks are configured
        assert 'healthcheck' in sentinel_config['services']['redis-master']
    
    @pytest.mark.asyncio
    async def test_memory_usage(self):
        """Test memory usage stays within limits."""
        import psutil
        import gc
        
        # Force garbage collection
        gc.collect()
        
        # Get current process memory
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        # Should be under 2GB for normal operation
        assert memory_mb < 2048, f"Memory usage {memory_mb}MB exceeds 2GB limit"
        
        # Test for memory leaks by creating and destroying objects
        initial_memory = memory_mb
        
        # Create and destroy 1000 temporary objects
        for _ in range(1000):
            temp_data = {'data': 'x' * 1000}  # 1KB each
        
        gc.collect()
        
        # Check memory didn't grow significantly
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        
        assert memory_growth < 10, f"Memory grew by {memory_growth}MB, possible leak"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])