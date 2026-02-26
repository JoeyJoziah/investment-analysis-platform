"""
Unit tests for backend/utils/comprehensive_cache.py

Pure unit tests covering:
- CacheConfig defaults and custom values
- CacheMetrics: hit_ratio, cost_savings, prefix tracking
- LRUCache: get/set, TTL, eviction, cleanup, stats
- ComprehensiveCacheManager: key generation, serialization, deserialization,
  compression, expiry checks, TTL policies
- cached() decorator logic

No Redis, no database, no network.
"""

import os
os.environ["TESTING"] = "True"
os.environ["DEBUG"] = "True"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

import gzip
import json
import time
import hashlib
import pytest
import numpy as np
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock, AsyncMock

from backend.utils.comprehensive_cache import (
    CacheConfig,
    CacheMetrics,
    LRUCache,
    ComprehensiveCacheManager,
)


# ===========================================================================
# CacheConfig Tests
# ===========================================================================

class TestCacheConfig:
    """Tests for CacheConfig dataclass defaults and custom values."""

    def test_default_l1_ttl(self):
        config = CacheConfig()
        assert config.l1_ttl == 300

    def test_default_l2_ttl(self):
        config = CacheConfig()
        assert config.l2_ttl == 3600

    def test_default_l3_ttl(self):
        config = CacheConfig()
        assert config.l3_ttl == 86400

    def test_default_l1_max_size(self):
        config = CacheConfig()
        assert config.l1_max_size == 1000

    def test_default_compression_threshold(self):
        config = CacheConfig()
        assert config.compression_threshold == 1024

    def test_default_enable_warming(self):
        config = CacheConfig()
        assert config.enable_warming is True

    def test_custom_values(self):
        config = CacheConfig(l1_ttl=60, l2_ttl=120, l3_ttl=240, l1_max_size=50)
        assert config.l1_ttl == 60
        assert config.l2_ttl == 120
        assert config.l3_ttl == 240
        assert config.l1_max_size == 50

    def test_invalidation_cascading_default(self):
        config = CacheConfig()
        assert config.invalidation_cascading is True


# ===========================================================================
# CacheMetrics Tests
# ===========================================================================

class TestCacheMetrics:
    """Tests for CacheMetrics properties and prefix tracking."""

    def test_initial_hit_ratio_is_zero(self):
        metrics = CacheMetrics()
        assert metrics.hit_ratio == 0.0

    def test_hit_ratio_calculation(self):
        metrics = CacheMetrics(
            l1_hits=5, l2_hits=3, l3_hits=2, total_requests=20
        )
        assert metrics.hit_ratio == 0.5  # (5+3+2)/20

    def test_hit_ratio_all_hits(self):
        metrics = CacheMetrics(l1_hits=10, total_requests=10)
        assert metrics.hit_ratio == 1.0

    def test_cost_savings_zero_api_calls(self):
        metrics = CacheMetrics()
        assert metrics.cost_savings == 0.0

    def test_cost_savings_calculation(self):
        metrics = CacheMetrics(api_calls_saved=100)
        assert metrics.cost_savings == pytest.approx(10.0)

    def test_prefix_hits_initialized_empty(self):
        metrics = CacheMetrics()
        assert metrics.prefix_hits == {}

    def test_prefix_misses_initialized_empty(self):
        metrics = CacheMetrics()
        assert metrics.prefix_misses == {}

    def test_track_prefix_hit_new_prefix(self):
        metrics = CacheMetrics()
        metrics.track_prefix_hit("api:resp")
        assert metrics.prefix_hits["api:resp"] == 1

    def test_track_prefix_hit_increment(self):
        metrics = CacheMetrics()
        metrics.track_prefix_hit("api:resp")
        metrics.track_prefix_hit("api:resp")
        metrics.track_prefix_hit("api:resp")
        assert metrics.prefix_hits["api:resp"] == 3

    def test_track_prefix_miss_new_prefix(self):
        metrics = CacheMetrics()
        metrics.track_prefix_miss("market")
        assert metrics.prefix_misses["market"] == 1

    def test_track_prefix_miss_increment(self):
        metrics = CacheMetrics()
        metrics.track_prefix_miss("market")
        metrics.track_prefix_miss("market")
        assert metrics.prefix_misses["market"] == 2

    def test_get_prefix_stats_empty(self):
        metrics = CacheMetrics()
        stats = metrics.get_prefix_stats()
        assert stats == {}

    def test_get_prefix_stats_with_data(self):
        metrics = CacheMetrics()
        metrics.track_prefix_hit("api:resp")
        metrics.track_prefix_hit("api:resp")
        metrics.track_prefix_miss("api:resp")

        stats = metrics.get_prefix_stats()
        assert "api:resp" in stats
        assert stats["api:resp"]["hits"] == 2
        assert stats["api:resp"]["misses"] == 1
        assert stats["api:resp"]["total"] == 3
        assert stats["api:resp"]["hit_rate"] == pytest.approx(2 / 3)

    def test_get_prefix_stats_miss_only_prefix(self):
        metrics = CacheMetrics()
        metrics.track_prefix_miss("quote")

        stats = metrics.get_prefix_stats()
        assert stats["quote"]["hits"] == 0
        assert stats["quote"]["misses"] == 1
        assert stats["quote"]["hit_rate"] == 0.0


# ===========================================================================
# LRUCache Tests
# ===========================================================================

class TestLRUCacheGet:
    """Tests for LRUCache.get()"""

    def test_get_missing_key_returns_none_false(self):
        cache = LRUCache(max_size=10)
        value, hit = cache.get("nonexistent")
        assert value is None
        assert hit is False

    def test_get_existing_key_returns_value_true(self):
        cache = LRUCache(max_size=10)
        cache.set("key1", "value1")
        value, hit = cache.get("key1")
        assert hit is True
        assert value is not None
        assert value["value"] == "value1"

    def test_get_moves_key_to_most_recent(self):
        cache = LRUCache(max_size=10)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)

        # Access "a" to make it most recently used
        cache.get("a")

        # Keys order should now be b, c, a
        keys = list(cache.cache.keys())
        assert keys[-1] == "a"

    def test_get_updates_access_time(self):
        cache = LRUCache(max_size=10)
        cache.set("key1", "val")
        time.sleep(0.01)
        old_time = cache.access_times.get("key1")
        cache.get("key1")
        new_time = cache.access_times.get("key1")
        assert new_time >= old_time


class TestLRUCacheSet:
    """Tests for LRUCache.set()"""

    def test_set_stores_value(self):
        cache = LRUCache(max_size=10)
        cache.set("key1", {"data": "test"})
        assert "key1" in cache.cache

    def test_set_with_ttl_stores_expiry(self):
        cache = LRUCache(max_size=10)
        cache.set("key1", "val", ttl=60)
        data = cache.cache["key1"]
        assert data["expires_at"] is not None
        assert data["expires_at"] > time.time()

    def test_set_without_ttl_no_expiry(self):
        cache = LRUCache(max_size=10)
        cache.set("key1", "val")
        data = cache.cache["key1"]
        assert data["expires_at"] is None

    def test_set_evicts_oldest_when_at_capacity(self):
        cache = LRUCache(max_size=3)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        # Cache is full, adding "d" should evict "a"
        cache.set("d", 4)
        assert "a" not in cache.cache
        assert "d" in cache.cache
        assert len(cache.cache) == 3

    def test_set_multiple_evictions(self):
        cache = LRUCache(max_size=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)  # evicts a
        cache.set("d", 4)  # evicts b
        assert "a" not in cache.cache
        assert "b" not in cache.cache
        assert "c" in cache.cache
        assert "d" in cache.cache

    def test_set_records_access_time(self):
        cache = LRUCache(max_size=10)
        before = time.time()
        cache.set("key1", "val")
        after = time.time()
        assert before <= cache.access_times["key1"] <= after

    def test_eviction_cleans_access_times(self):
        cache = LRUCache(max_size=1)
        cache.set("a", 1)
        assert "a" in cache.access_times
        cache.set("b", 2)  # evicts a
        assert "a" not in cache.access_times


class TestLRUCacheDelete:
    """Tests for LRUCache.delete()"""

    def test_delete_existing_key_returns_true(self):
        cache = LRUCache(max_size=10)
        cache.set("key1", "val")
        result = cache.delete("key1")
        assert result is True
        assert "key1" not in cache.cache

    def test_delete_missing_key_returns_false(self):
        cache = LRUCache(max_size=10)
        result = cache.delete("nonexistent")
        assert result is False

    def test_delete_cleans_access_time(self):
        cache = LRUCache(max_size=10)
        cache.set("key1", "val")
        cache.delete("key1")
        assert "key1" not in cache.access_times


class TestLRUCacheClear:
    """Tests for LRUCache.clear()"""

    def test_clear_empties_cache(self):
        cache = LRUCache(max_size=10)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert len(cache.cache) == 0

    def test_clear_empties_access_times(self):
        cache = LRUCache(max_size=10)
        cache.set("a", 1)
        cache.clear()
        assert len(cache.access_times) == 0


class TestLRUCacheCleanupExpired:
    """Tests for LRUCache.cleanup_expired()"""

    def test_cleanup_expired_removes_expired_entries(self):
        cache = LRUCache(max_size=10)
        # Manually inject an entry with an already-expired timestamp
        cache.cache["expired"] = {
            "value": "val",
            "expires_at": time.time() - 10,
        }
        cache.access_times["expired"] = time.time()
        cache.set("alive", "val", ttl=9999)

        removed = cache.cleanup_expired()
        assert removed == 1
        assert "expired" not in cache.cache
        assert "alive" in cache.cache

    def test_cleanup_no_expired_returns_zero(self):
        cache = LRUCache(max_size=10)
        cache.set("alive", "val", ttl=9999)
        removed = cache.cleanup_expired()
        assert removed == 0

    def test_cleanup_expired_no_ttl_entries_not_removed(self):
        cache = LRUCache(max_size=10)
        cache.set("no_ttl", "val")  # No TTL means no expiry
        removed = cache.cleanup_expired()
        assert removed == 0
        assert "no_ttl" in cache.cache


class TestLRUCacheGetStats:
    """Tests for LRUCache.get_stats()"""

    def test_stats_empty_cache(self):
        cache = LRUCache(max_size=100)
        stats = cache.get_stats()
        assert stats["size"] == 0
        assert stats["max_size"] == 100
        assert stats["utilization"] == 0.0
        assert stats["size_bytes"] == 0

    def test_stats_with_entries(self):
        cache = LRUCache(max_size=100)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        stats = cache.get_stats()
        assert stats["size"] == 2
        assert stats["utilization"] == pytest.approx(0.02)
        assert stats["size_bytes"] > 0


# ===========================================================================
# ComprehensiveCacheManager - Key Generation Tests
# ===========================================================================

class TestCacheManagerMakeKey:
    """Tests for ComprehensiveCacheManager._make_key()"""

    def test_make_key_known_prefix(self):
        mgr = ComprehensiveCacheManager()
        key = mgr._make_key("market_data", "AAPL")
        assert key == "market:AAPL"

    def test_make_key_unknown_type_uses_misc(self):
        mgr = ComprehensiveCacheManager()
        key = mgr._make_key("unknown_type", "some_id")
        assert key == "misc:some_id"

    def test_make_key_with_params_adds_hash(self):
        mgr = ComprehensiveCacheManager()
        params = {"interval": "daily", "period": "1y"}
        key = mgr._make_key("market_data", "AAPL", params)

        # Key should have format prefix:identifier:hash
        parts = key.split(":")
        assert len(parts) == 3
        assert parts[0] == "market"
        assert parts[1] == "AAPL"
        assert len(parts[2]) == 8  # MD5 truncated to 8 chars

    def test_make_key_params_are_deterministic(self):
        mgr = ComprehensiveCacheManager()
        params = {"b": 2, "a": 1}
        key1 = mgr._make_key("market_data", "AAPL", params)
        key2 = mgr._make_key("market_data", "AAPL", params)
        assert key1 == key2

    def test_make_key_different_params_produce_different_keys(self):
        mgr = ComprehensiveCacheManager()
        key1 = mgr._make_key("market_data", "AAPL", {"a": 1})
        key2 = mgr._make_key("market_data", "AAPL", {"a": 2})
        assert key1 != key2

    def test_make_key_param_order_does_not_matter(self):
        mgr = ComprehensiveCacheManager()
        key1 = mgr._make_key("market_data", "AAPL", {"a": 1, "b": 2})
        key2 = mgr._make_key("market_data", "AAPL", {"b": 2, "a": 1})
        assert key1 == key2


# ===========================================================================
# ComprehensiveCacheManager - Serialization Tests
# ===========================================================================

class TestCacheManagerSerialization:
    """Tests for _serialize_data / _deserialize_data and custom JSON handlers."""

    def test_serialize_simple_dict(self):
        mgr = ComprehensiveCacheManager()
        data = {"key": "value", "num": 42}
        serialized = mgr._serialize_data(data)
        assert isinstance(serialized, bytes)

    def test_roundtrip_simple_dict(self):
        mgr = ComprehensiveCacheManager()
        data = {"key": "value", "num": 42}
        serialized = mgr._serialize_data(data)
        result = mgr._deserialize_data(serialized)
        assert result == data

    def test_roundtrip_nested_dict(self):
        mgr = ComprehensiveCacheManager()
        data = {"level1": {"level2": {"value": [1, 2, 3]}}}
        serialized = mgr._serialize_data(data)
        result = mgr._deserialize_data(serialized)
        assert result == data

    def test_serialize_numpy_array(self):
        mgr = ComprehensiveCacheManager()
        data = {"prices": np.array([1.5, 2.5, 3.5])}
        serialized = mgr._serialize_data(data)
        result = mgr._deserialize_data(serialized)
        np.testing.assert_array_almost_equal(result["prices"], [1.5, 2.5, 3.5])

    def test_serialize_numpy_integer(self):
        mgr = ComprehensiveCacheManager()
        data = {"count": np.int64(42)}
        serialized = mgr._serialize_data(data)
        result = mgr._deserialize_data(serialized)
        assert result["count"] == 42

    def test_serialize_numpy_float(self):
        mgr = ComprehensiveCacheManager()
        data = {"ratio": np.float64(3.14)}
        serialized = mgr._serialize_data(data)
        result = mgr._deserialize_data(serialized)
        assert result["ratio"] == pytest.approx(3.14)

    def test_serialize_datetime(self):
        mgr = ComprehensiveCacheManager()
        dt = datetime(2025, 6, 15, 12, 30, 0, tzinfo=timezone.utc)
        data = {"timestamp": dt}
        serialized = mgr._serialize_data(data)
        result = mgr._deserialize_data(serialized)
        assert result["timestamp"] == dt

    def test_serialize_timedelta(self):
        mgr = ComprehensiveCacheManager()
        td = timedelta(hours=2, minutes=30)
        data = {"duration": td}
        serialized = mgr._serialize_data(data)
        result = mgr._deserialize_data(serialized)
        assert result["duration"] == td

    def test_serializer_raises_for_unhandled_type(self):
        mgr = ComprehensiveCacheManager()
        # A custom class without __dict__ would be tricky; use a lambda
        # which doesn't have __dict__ in the usual sense but will fail
        # json.dumps. Using set() which is not JSON-serializable.
        with pytest.raises(TypeError):
            mgr._json_serializer(set([1, 2, 3]))

    def test_compression_applied_for_large_data(self):
        mgr = ComprehensiveCacheManager(CacheConfig(compression_threshold=50))
        data = {"large": "x" * 200}
        serialized = mgr._serialize_data(data)
        assert serialized.startswith(b"gzip:")

    def test_compression_not_applied_for_small_data(self):
        mgr = ComprehensiveCacheManager(CacheConfig(compression_threshold=99999))
        data = {"small": "x"}
        serialized = mgr._serialize_data(data)
        assert not serialized.startswith(b"gzip:")

    def test_roundtrip_compressed_data(self):
        mgr = ComprehensiveCacheManager(CacheConfig(compression_threshold=50))
        data = {"large_key": "a" * 200, "number": 42}
        serialized = mgr._serialize_data(data)
        assert serialized.startswith(b"gzip:")
        result = mgr._deserialize_data(serialized)
        assert result == data

    def test_json_serializer_object_with_dict(self):
        mgr = ComprehensiveCacheManager()

        class Dummy:
            def __init__(self):
                self.x = 1
                self.y = "hello"

        obj = Dummy()
        result = mgr._json_serializer(obj)
        assert result["__object__"] is True
        assert result["type"] == "Dummy"
        assert result["data"]["x"] == 1
        assert result["data"]["y"] == "hello"

    def test_json_deserializer_object_returns_dict(self):
        mgr = ComprehensiveCacheManager()
        obj = {"__object__": True, "type": "Dummy", "data": {"x": 1}}
        result = mgr._json_deserializer(obj)
        assert result == {"x": 1}

    def test_json_deserializer_passthrough_normal_dict(self):
        mgr = ComprehensiveCacheManager()
        obj = {"name": "test", "value": 42}
        result = mgr._json_deserializer(obj)
        assert result == {"name": "test", "value": 42}


# ===========================================================================
# ComprehensiveCacheManager - Expiry Check Tests
# ===========================================================================

class TestCacheManagerIsExpired:
    """Tests for ComprehensiveCacheManager._is_expired()"""

    def test_not_expired(self):
        mgr = ComprehensiveCacheManager()
        data = {"expires_at": time.time() + 999}
        assert mgr._is_expired(data) is False

    def test_expired(self):
        mgr = ComprehensiveCacheManager()
        data = {"expires_at": time.time() - 1}
        assert mgr._is_expired(data) is True

    def test_no_expiry_not_expired(self):
        mgr = ComprehensiveCacheManager()
        data = {"expires_at": None}
        assert mgr._is_expired(data) is False

    def test_missing_expires_at_not_expired(self):
        mgr = ComprehensiveCacheManager()
        data = {}
        assert mgr._is_expired(data) is False


# ===========================================================================
# ComprehensiveCacheManager - TTL Policy Tests
# ===========================================================================

class TestCacheManagerTTLPolicies:
    """Tests for TTL policy resolution."""

    def test_known_data_type_has_policy(self):
        mgr = ComprehensiveCacheManager()
        assert "real_time_quote" in mgr.ttl_policies
        policy = mgr.ttl_policies["real_time_quote"]
        assert policy["l1"] == 60
        assert policy["l2"] == 300
        assert policy["l3"] == 900

    def test_company_overview_has_long_ttl(self):
        mgr = ComprehensiveCacheManager()
        policy = mgr.ttl_policies["company_overview"]
        assert policy["l1"] == 7200
        assert policy["l2"] == 43200
        assert policy["l3"] == 86400

    def test_all_known_prefixes_exist(self):
        mgr = ComprehensiveCacheManager()
        expected = [
            "api_response", "db_query", "computation", "market_data",
            "user_data", "analysis", "real_time_quote", "company_overview",
            "technical_indicators", "ml_predictions", "recommendations",
            "stock_list"
        ]
        for prefix in expected:
            assert prefix in mgr.key_prefixes, f"Missing key_prefix: {prefix}"


# ===========================================================================
# ComprehensiveCacheManager - L1 Integration Tests
# ===========================================================================

class TestCacheManagerL1Only:
    """Tests verifying L1 cache behavior through the manager (no Redis/DB)."""

    def test_custom_config_propagates_to_l1(self):
        config = CacheConfig(l1_max_size=5)
        mgr = ComprehensiveCacheManager(config)
        assert mgr.l1_cache.max_size == 5

    def test_metrics_default_to_zero(self):
        mgr = ComprehensiveCacheManager()
        assert mgr.metrics.total_requests == 0
        assert mgr.metrics.l1_hits == 0
        assert mgr.metrics.l2_hits == 0
        assert mgr.metrics.l3_hits == 0

    def test_warming_tasks_initially_empty(self):
        mgr = ComprehensiveCacheManager()
        assert mgr._warming_tasks == {}

    def test_redis_client_initially_none(self):
        mgr = ComprehensiveCacheManager()
        assert mgr.redis_client is None


# ===========================================================================
# ComprehensiveCacheManager - get() L1 hit path (async)
# ===========================================================================

class TestCacheManagerGetL1:
    """Tests for get() returning from L1 cache (mocked L2/L3)."""

    @pytest.mark.asyncio
    async def test_get_l1_hit_returns_data_and_source(self):
        mgr = ComprehensiveCacheManager()
        cache_key = mgr._make_key("market_data", "AAPL")

        # Manually put data in L1
        mgr.l1_cache.set(cache_key, {"price": 150.0}, ttl=9999)

        # Mock L3 to avoid DB calls
        mgr._get_from_database = AsyncMock(return_value=None)

        data, source = await mgr.get("market_data", "AAPL")
        assert source == "l1"
        assert data["price"] == 150.0

    @pytest.mark.asyncio
    async def test_get_l1_hit_increments_metrics(self):
        mgr = ComprehensiveCacheManager()
        cache_key = mgr._make_key("market_data", "AAPL")
        mgr.l1_cache.set(cache_key, {"price": 150.0}, ttl=9999)
        mgr._get_from_database = AsyncMock(return_value=None)

        await mgr.get("market_data", "AAPL")
        assert mgr.metrics.l1_hits == 1
        assert mgr.metrics.total_requests == 1

    @pytest.mark.asyncio
    async def test_get_expired_l1_falls_through(self):
        mgr = ComprehensiveCacheManager()
        cache_key = mgr._make_key("market_data", "AAPL")
        # Manually inject an expired entry in L1
        mgr.l1_cache.cache[cache_key] = {
            "value": {"price": 100.0},
            "expires_at": time.time() - 10,
        }
        mgr.l1_cache.access_times[cache_key] = time.time()

        mgr._get_from_database = AsyncMock(return_value=None)

        data, source = await mgr.get("market_data", "AAPL")
        assert source == "miss"
        assert data is None

    @pytest.mark.asyncio
    async def test_get_miss_all_layers_returns_none_miss(self):
        mgr = ComprehensiveCacheManager()
        mgr._get_from_database = AsyncMock(return_value=None)

        data, source = await mgr.get("market_data", "AAPL")
        assert source == "miss"
        assert data is None
        assert mgr.metrics.total_requests == 1

    @pytest.mark.asyncio
    async def test_get_with_sync_fallback_function(self):
        mgr = ComprehensiveCacheManager()
        mgr._get_from_database = AsyncMock(return_value=None)
        mgr._set_database = AsyncMock()

        def my_fallback():
            return {"price": 200.0}

        data, source = await mgr.get(
            "market_data", "AAPL", fallback_func=my_fallback
        )
        assert source == "computed"
        assert data["price"] == 200.0

    @pytest.mark.asyncio
    async def test_get_with_async_fallback_function(self):
        mgr = ComprehensiveCacheManager()
        mgr._get_from_database = AsyncMock(return_value=None)
        mgr._set_database = AsyncMock()

        async def my_fallback():
            return {"price": 300.0}

        data, source = await mgr.get(
            "market_data", "AAPL", fallback_func=my_fallback
        )
        assert source == "computed"
        assert data["price"] == 300.0

    @pytest.mark.asyncio
    async def test_get_fallback_exception_returns_miss(self):
        mgr = ComprehensiveCacheManager()
        mgr._get_from_database = AsyncMock(return_value=None)

        def bad_fallback():
            raise RuntimeError("API down")

        data, source = await mgr.get(
            "market_data", "AAPL", fallback_func=bad_fallback
        )
        assert source == "miss"
        assert data is None

    @pytest.mark.asyncio
    async def test_get_fallback_returning_none_returns_miss(self):
        mgr = ComprehensiveCacheManager()
        mgr._get_from_database = AsyncMock(return_value=None)
        mgr._set_database = AsyncMock()

        def null_fallback():
            return None

        data, source = await mgr.get(
            "market_data", "AAPL", fallback_func=null_fallback
        )
        assert source == "miss"
        assert data is None


# ===========================================================================
# ComprehensiveCacheManager - set() Tests (async)
# ===========================================================================

class TestCacheManagerSet:
    """Tests for set() populating cache layers."""

    @pytest.mark.asyncio
    async def test_set_populates_l1(self):
        mgr = ComprehensiveCacheManager()
        mgr._set_database = AsyncMock()

        await mgr.set("market_data", "AAPL", {"price": 150.0})
        cache_key = mgr._make_key("market_data", "AAPL")
        value, hit = mgr.l1_cache.get(cache_key)
        assert hit is True

    @pytest.mark.asyncio
    async def test_set_updates_storage_bytes_metric(self):
        mgr = ComprehensiveCacheManager()
        mgr._set_database = AsyncMock()

        await mgr.set("market_data", "AAPL", {"price": 150.0})
        assert mgr.metrics.cache_storage_bytes > 0

    @pytest.mark.asyncio
    async def test_set_with_custom_ttl(self):
        mgr = ComprehensiveCacheManager()
        mgr._set_database = AsyncMock()

        custom = {"l1": 10, "l2": 20, "l3": 30}
        await mgr.set("market_data", "AAPL", {"price": 150.0}, custom_ttl=custom)

        # L1 should be populated with TTL
        cache_key = mgr._make_key("market_data", "AAPL")
        value, hit = mgr.l1_cache.get(cache_key)
        assert hit is True
        assert value["expires_at"] is not None


# ===========================================================================
# ComprehensiveCacheManager - delete() Tests (async)
# ===========================================================================

class TestCacheManagerDelete:
    """Tests for delete() removing from all layers."""

    @pytest.mark.asyncio
    async def test_delete_removes_from_l1(self):
        mgr = ComprehensiveCacheManager()
        mgr._set_database = AsyncMock()
        mgr._delete_from_database = AsyncMock()

        await mgr.set("market_data", "AAPL", {"price": 150.0})
        await mgr.delete("market_data", "AAPL")

        cache_key = mgr._make_key("market_data", "AAPL")
        value, hit = mgr.l1_cache.get(cache_key)
        assert hit is False


# ===========================================================================
# ComprehensiveCacheManager - warm_cache() Tests (async)
# ===========================================================================

class TestCacheManagerWarmCache:
    """Tests for warm_cache() behavior."""

    @pytest.mark.asyncio
    async def test_warm_cache_disabled_does_nothing(self):
        config = CacheConfig(enable_warming=False)
        mgr = ComprehensiveCacheManager(config)

        await mgr.warm_cache([{"data_type": "market_data", "identifier": "AAPL"}])
        assert len(mgr._warming_tasks) == 0

    @pytest.mark.asyncio
    async def test_warm_cache_deduplicates_tasks(self):
        config = CacheConfig(enable_warming=True)
        mgr = ComprehensiveCacheManager(config)
        mgr._exists_in_any_layer = AsyncMock(return_value=True)

        specs = [{"data_type": "market_data", "identifier": "AAPL"}]
        await mgr.warm_cache(specs, priority=1)
        first_count = len(mgr._warming_tasks)

        # Same specs + priority should not create a new task
        await mgr.warm_cache(specs, priority=1)
        assert len(mgr._warming_tasks) == first_count
