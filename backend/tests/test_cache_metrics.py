"""
Tests for cache Prometheus metrics integration

Validates that cache operations correctly emit Prometheus metrics for:
- Cache hits/misses by layer and prefix
- Cache operation latency (get, set, delete)
- Cache size in bytes
- Cache evictions
"""

import asyncio
import pytest
import time
from unittest.mock import MagicMock, AsyncMock, patch
from prometheus_client import REGISTRY

from backend.utils.comprehensive_cache import (
    ComprehensiveCacheManager,
    CacheConfig,
    LRUCache,
    cache_hits_total,
    cache_misses_total,
    cache_latency_seconds,
    cache_size_bytes,
    cache_evictions_total,
)


@pytest.fixture
def cache_config():
    """Create test cache configuration"""
    return CacheConfig(
        l1_ttl=60,
        l2_ttl=300,
        l3_ttl=3600,
        l1_max_size=10,  # Small size to test evictions
        compression_threshold=1024,
        enable_warming=False,
    )


@pytest.fixture
def lru_cache():
    """Create test LRU cache"""
    return LRUCache(max_size=5)


@pytest.fixture
async def cache_manager(cache_config):
    """Create test cache manager with mocked dependencies"""
    manager = ComprehensiveCacheManager(config=cache_config)

    # Mock Redis client
    manager.redis_client = AsyncMock()
    manager.redis_client.get = AsyncMock(return_value=None)
    manager.redis_client.setex = AsyncMock()
    manager.redis_client.delete = AsyncMock()

    return manager


def get_metric_value(metric, labels):
    """Helper to get current metric value"""
    try:
        return metric.labels(**labels)._value.get()
    except Exception:
        return 0


def get_histogram_count(metric, labels):
    """Helper to get histogram sample count"""
    try:
        # For Histogram, we need to check the sample count
        # The histogram stores samples, not a single value
        metric_key = tuple(sorted(labels.items()))

        # Try to access the internal metrics storage
        if hasattr(metric, '_metrics'):
            if metric_key in metric._metrics:
                hist_metric = metric._metrics[metric_key]
                # Get count from the histogram's internal state
                if hasattr(hist_metric, '_sum'):
                    # Return the sum value which increases with observations
                    return hist_metric._sum._value.get()
                elif hasattr(hist_metric, '_count'):
                    return hist_metric._count._value.get()

        # Alternative: try direct label access
        labeled = metric.labels(**labels)
        if hasattr(labeled, '_sum'):
            return labeled._sum._value.get()

        return 0
    except Exception:
        return 0


class TestLRUCacheMetrics:
    """Test Prometheus metrics for LRU Cache (L1)"""

    def test_l1_get_records_latency(self, lru_cache):
        """Test that L1 get operations record latency"""
        # Set and get a value
        lru_cache.set('test_key', 'test_value', ttl=60)
        value, hit = lru_cache.get('test_key')

        assert hit is True
        assert value['value'] == 'test_value'

        # Check latency metric exists (histogram was observed)
        metric_key = ('get', 'l1')  # Format is (operation_value, layer_value)
        assert metric_key in cache_latency_seconds._metrics

    def test_l1_set_records_latency(self, lru_cache):
        """Test that L1 set operations record latency"""
        lru_cache.set('test_key', 'test_value', ttl=60)

        # Check latency metric exists (histogram was observed)
        metric_key = ('set', 'l1')  # Format is (operation_value, layer_value)
        assert metric_key in cache_latency_seconds._metrics

    def test_l1_delete_records_latency(self, lru_cache):
        """Test that L1 delete operations record latency"""
        lru_cache.set('test_key', 'test_value', ttl=60)

        deleted = lru_cache.delete('test_key')

        assert deleted is True

        # Check latency metric exists (histogram was observed)
        metric_key = ('delete', 'l1')  # Format is (operation_value, layer_value)
        assert metric_key in cache_latency_seconds._metrics

    def test_l1_evictions_counter(self, lru_cache):
        """Test that L1 cache evictions are counted"""
        initial_evictions = get_metric_value(
            cache_evictions_total,
            {'layer': 'l1'}
        )

        # Fill cache beyond capacity to trigger evictions
        for i in range(10):  # max_size is 5
            lru_cache.set(f'key_{i}', f'value_{i}', ttl=60)

        # Check evictions were recorded
        new_evictions = get_metric_value(
            cache_evictions_total,
            {'layer': 'l1'}
        )
        assert new_evictions > initial_evictions

    def test_l1_cleanup_expired_evictions(self, lru_cache):
        """Test that expired entry cleanup counts as evictions"""
        initial_evictions = get_metric_value(
            cache_evictions_total,
            {'layer': 'l1'}
        )

        # Add entries with very short TTL
        lru_cache.set('expire_soon', 'value', ttl=0.1)
        time.sleep(0.2)

        # Cleanup expired entries
        expired_count = lru_cache.cleanup_expired()

        assert expired_count > 0

        # Check evictions were recorded
        new_evictions = get_metric_value(
            cache_evictions_total,
            {'layer': 'l1'}
        )
        assert new_evictions >= initial_evictions + expired_count

    def test_l1_size_bytes_gauge(self, lru_cache):
        """Test that L1 cache size is tracked in bytes"""
        # Get initial size
        stats = lru_cache.get_stats()

        assert 'size_bytes' in stats
        assert stats['size_bytes'] >= 0

        # Add some data
        lru_cache.set('test_key', {'data': 'x' * 100}, ttl=60)

        # Get updated size
        new_stats = lru_cache.get_stats()
        assert new_stats['size_bytes'] > stats['size_bytes']


class TestCacheManagerMetrics:
    """Test Prometheus metrics for ComprehensiveCacheManager"""

    @pytest.mark.asyncio
    async def test_cache_hit_l1_increments_counter(self, cache_manager):
        """Test that L1 cache hits increment the hits counter"""
        # Pre-populate L1 cache
        cache_manager.l1_cache.set(
            'api:resp:test:abc123',
            {'data': 'test_value'},
            ttl=60
        )

        initial_hits = get_metric_value(
            cache_hits_total,
            {'layer': 'l1', 'prefix': 'api:resp'}
        )

        # Perform cache get that should hit L1
        result, source = await cache_manager.get(
            data_type='api_response',
            identifier='test:abc123'
        )

        assert source == 'l1'
        assert result == {'data': 'test_value'}

        # Check hit counter increased
        new_hits = get_metric_value(
            cache_hits_total,
            {'layer': 'l1', 'prefix': 'api:resp'}
        )
        assert new_hits > initial_hits

    @pytest.mark.asyncio
    async def test_cache_miss_l1_increments_counter(self, cache_manager):
        """Test that L1 cache misses increment the misses counter"""
        # Add an expired entry to L1 to trigger L1 miss
        import time as time_module
        cache_manager.l1_cache.set(
            'api:resp:test_key',
            {'data': 'expired'},
            ttl=0.001  # Very short TTL
        )
        time_module.sleep(0.01)  # Wait for expiration

        # Set up L2 to hit
        cache_manager.redis_client.get.return_value = b'{"data": "from_redis"}'

        initial_misses = get_metric_value(
            cache_misses_total,
            {'layer': 'l1', 'prefix': 'api:resp'}
        )

        # Perform cache get - should miss L1 (expired) and hit L2
        with patch.object(cache_manager, '_deserialize_data', return_value={'data': 'from_redis'}):
            result, source = await cache_manager.get(
                data_type='api_response',
                identifier='test_key'
            )

        assert source == 'l2'

        # Check miss counter increased
        new_misses = get_metric_value(
            cache_misses_total,
            {'layer': 'l1', 'prefix': 'api:resp'}
        )
        assert new_misses > initial_misses

    @pytest.mark.asyncio
    async def test_cache_hit_l2_increments_counter(self, cache_manager):
        """Test that L2 (Redis) cache hits increment the hits counter"""
        cache_manager.redis_client.get.return_value = b'{"data": "from_redis"}'

        initial_hits = get_metric_value(
            cache_hits_total,
            {'layer': 'l2', 'prefix': 'market'}
        )

        with patch.object(cache_manager, '_deserialize_data', return_value={'data': 'from_redis'}):
            result, source = await cache_manager.get(
                data_type='market_data',
                identifier='AAPL'
            )

        assert source == 'l2'

        # Check hit counter increased
        new_hits = get_metric_value(
            cache_hits_total,
            {'layer': 'l2', 'prefix': 'market'}
        )
        assert new_hits > initial_hits

    @pytest.mark.asyncio
    async def test_cache_miss_l2_increments_counter(self, cache_manager):
        """Test that L2 cache misses increment the misses counter"""
        cache_manager.redis_client.get.return_value = None

        initial_misses = get_metric_value(
            cache_misses_total,
            {'layer': 'l2', 'prefix': 'market'}
        )

        # Mock L3 to return data
        with patch.object(cache_manager, '_get_from_database', return_value={'data': 'from_db'}):
            result, source = await cache_manager.get(
                data_type='market_data',
                identifier='AAPL'
            )

        assert source == 'l3'

        # Check miss counter increased
        new_misses = get_metric_value(
            cache_misses_total,
            {'layer': 'l2', 'prefix': 'market'}
        )
        assert new_misses > initial_misses

    @pytest.mark.asyncio
    async def test_cache_hit_l3_increments_counter(self, cache_manager):
        """Test that L3 (database) cache hits increment the hits counter"""
        cache_manager.redis_client.get.return_value = None

        initial_hits = get_metric_value(
            cache_hits_total,
            {'layer': 'l3', 'prefix': 'analysis'}
        )

        with patch.object(cache_manager, '_get_from_database', return_value={'data': 'from_db'}):
            result, source = await cache_manager.get(
                data_type='analysis',
                identifier='AAPL:fundamental'
            )

        assert source == 'l3'

        # Check hit counter increased
        new_hits = get_metric_value(
            cache_hits_total,
            {'layer': 'l3', 'prefix': 'analysis'}
        )
        assert new_hits > initial_hits

    @pytest.mark.asyncio
    async def test_cache_miss_l3_increments_counter(self, cache_manager):
        """Test that L3 cache misses increment the misses counter"""
        cache_manager.redis_client.get.return_value = None

        initial_misses = get_metric_value(
            cache_misses_total,
            {'layer': 'l3', 'prefix': 'analysis'}
        )

        with patch.object(cache_manager, '_get_from_database', return_value=None):
            result, source = await cache_manager.get(
                data_type='analysis',
                identifier='AAPL:fundamental'
            )

        assert source == 'miss'

        # Check miss counter increased
        new_misses = get_metric_value(
            cache_misses_total,
            {'layer': 'l3', 'prefix': 'analysis'}
        )
        assert new_misses > initial_misses

    @pytest.mark.asyncio
    async def test_get_operation_records_latency(self, cache_manager):
        """Test that cache get operations record latency metrics"""
        cache_manager.l1_cache.set('api:resp:key', {'data': 'test'}, ttl=60)

        result, source = await cache_manager.get(
            data_type='api_response',
            identifier='key'
        )

        # Check latency metric exists (histogram was observed)
        metric_key = ('get', 'l1')  # Format is (operation_value, layer_value)
        assert metric_key in cache_latency_seconds._metrics

    @pytest.mark.asyncio
    async def test_set_operation_records_latency(self, cache_manager):
        """Test that cache set operations record latency metrics"""
        with patch.object(cache_manager, '_set_database', new_callable=AsyncMock):
            await cache_manager.set(
                data_type='market_data',
                identifier='AAPL',
                data={'price': 150.0}
            )

        # Check latency metrics exist (histograms were observed)
        l2_key = ('set', 'l2')  # Format is (operation_value, layer_value)
        l3_key = ('set', 'l3')

        assert l2_key in cache_latency_seconds._metrics
        assert l3_key in cache_latency_seconds._metrics

    @pytest.mark.asyncio
    async def test_delete_operation_records_latency(self, cache_manager):
        """Test that cache delete operations record latency metrics"""
        with patch.object(cache_manager, '_delete_from_database', new_callable=AsyncMock):
            await cache_manager.delete(
                data_type='market_data',
                identifier='AAPL'
            )

        # Check latency metrics exist (histograms were observed)
        l2_key = ('delete', 'l2')  # Format is (operation_value, layer_value)
        l3_key = ('delete', 'l3')

        assert l2_key in cache_latency_seconds._metrics
        assert l3_key in cache_latency_seconds._metrics

    @pytest.mark.asyncio
    async def test_different_prefixes_tracked_separately(self, cache_manager):
        """Test that different cache key prefixes are tracked separately"""
        # Pre-populate with different types
        cache_manager.l1_cache.set('market:AAPL', {'price': 150}, ttl=60)
        cache_manager.l1_cache.set('quote:AAPL', {'bid': 149.9}, ttl=60)

        initial_market_hits = get_metric_value(
            cache_hits_total,
            {'layer': 'l1', 'prefix': 'market'}
        )
        initial_quote_hits = get_metric_value(
            cache_hits_total,
            {'layer': 'l1', 'prefix': 'quote'}
        )

        # Get market data
        await cache_manager.get(data_type='market_data', identifier='AAPL')

        # Get real-time quote
        await cache_manager.get(data_type='real_time_quote', identifier='AAPL')

        # Check both counters increased independently
        new_market_hits = get_metric_value(
            cache_hits_total,
            {'layer': 'l1', 'prefix': 'market'}
        )
        new_quote_hits = get_metric_value(
            cache_hits_total,
            {'layer': 'l1', 'prefix': 'quote'}
        )

        assert new_market_hits > initial_market_hits
        assert new_quote_hits > initial_quote_hits

    @pytest.mark.asyncio
    async def test_cache_size_updates_on_set(self, cache_manager):
        """Test that cache size gauge updates when data is set"""
        # Get initial L1 size
        initial_stats = cache_manager.l1_cache.get_stats()
        initial_size = initial_stats.get('size_bytes', 0)

        # Set some data
        with patch.object(cache_manager, '_set_database', new_callable=AsyncMock):
            await cache_manager.set(
                data_type='market_data',
                identifier='AAPL',
                data={'price': 150.0, 'volume': 1000000}
            )

        # Check size increased
        new_stats = cache_manager.l1_cache.get_stats()
        new_size = new_stats.get('size_bytes', 0)

        assert new_size > initial_size


class TestMetricsAccuracy:
    """Test accuracy and consistency of cache metrics"""

    @pytest.mark.asyncio
    async def test_hit_miss_counters_are_accurate(self, cache_manager):
        """Test that hit and miss counters accurately reflect operations"""
        # Record initial values
        prefix = 'market'
        initial_l1_hits = get_metric_value(cache_hits_total, {'layer': 'l1', 'prefix': prefix})
        initial_l1_misses = get_metric_value(cache_misses_total, {'layer': 'l1', 'prefix': prefix})

        # Populate L1 with one key
        cache_manager.l1_cache.set('market:AAPL', {'price': 150}, ttl=60)

        # Hit: Get existing key
        await cache_manager.get(data_type='market_data', identifier='AAPL')

        # Miss: Get key that exists but is expired (this triggers L1 miss counter)
        import time as time_module
        cache_manager.l1_cache.set('market:GOOGL', {'price': 2800}, ttl=0.001)
        time_module.sleep(0.01)  # Wait for expiration

        cache_manager.redis_client.get.return_value = None
        with patch.object(cache_manager, '_get_from_database', return_value=None):
            await cache_manager.get(data_type='market_data', identifier='GOOGL')

        # Verify counters
        new_l1_hits = get_metric_value(cache_hits_total, {'layer': 'l1', 'prefix': prefix})
        new_l1_misses = get_metric_value(cache_misses_total, {'layer': 'l1', 'prefix': prefix})

        assert new_l1_hits == initial_l1_hits + 1
        assert new_l1_misses >= initial_l1_misses + 1  # At least 1 miss recorded

    @pytest.mark.asyncio
    async def test_latency_histogram_records_all_operations(self, cache_manager):
        """Test that latency histogram records all cache operations"""
        operations = ['get', 'set', 'delete']

        for operation in operations:
            if operation == 'get':
                cache_manager.l1_cache.set('api:resp:key', {'data': 'test'}, ttl=60)
                await cache_manager.get(data_type='api_response', identifier='key')
            elif operation == 'set':
                with patch.object(cache_manager, '_set_database', new_callable=AsyncMock):
                    await cache_manager.set(
                        data_type='api_response',
                        identifier='key',
                        data={'data': 'test'}
                    )
            elif operation == 'delete':
                await cache_manager.delete(data_type='api_response', identifier='key')

            # Check metric exists
            metric_key = (operation, 'l1')  # Format is (operation_value, layer_value)
            assert metric_key in cache_latency_seconds._metrics, f"Latency not recorded for {operation}"

    @pytest.mark.asyncio
    async def test_eviction_counter_accuracy(self, cache_manager):
        """Test that eviction counter accurately reflects evictions"""
        # Use small cache to force evictions
        small_cache = LRUCache(max_size=3)
        cache_manager.l1_cache = small_cache

        initial_evictions = get_metric_value(
            cache_evictions_total,
            {'layer': 'l1'}
        )

        # Add more items than capacity
        for i in range(5):
            small_cache.set(f'key_{i}', f'value_{i}', ttl=60)

        new_evictions = get_metric_value(
            cache_evictions_total,
            {'layer': 'l1'}
        )

        # Should have evicted 2 items (5 added - 3 capacity)
        assert new_evictions >= initial_evictions + 2
