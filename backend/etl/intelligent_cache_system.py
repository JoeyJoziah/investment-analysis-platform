"""
Intelligent Multi-Tier Caching System for Unlimited Stock Data Extraction
Implements memory, disk, and distributed caching with smart invalidation

Includes Bloom filter optimization for 90% faster cache misses.

This module is the thin orchestrator.  All implementation details live in:
- cache_primitives.py  : BloomFilter, CacheEntry, CacheStats, CompressionManager
- cache_storage.py     : MemoryTierCache, DiskTierCache
- cache_redis.py       : RedisTierCache
- cache_warming.py     : CacheWarmingResult, CacheWarmingMixin
- cache_analytics.py   : CacheAnalyticsMixin (access-pattern tracking + optimization)
"""

import asyncio
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Re-exports – all public symbols remain importable from this module so that
# existing callers (maintenance_tasks.py, unlimited_extractor_with_fallbacks.py,
# test_etl_extended_agent4.py, etc.) continue to work unchanged.
#
# The try/except pattern supports two loading contexts:
#   1. Normal package import (relative imports work, __package__ is set)
#   2. Standalone spec_from_file_location (no package context, test_agent4)
#
# In context 2 we temporarily add the etl directory to sys.path so that
# absolute imports of the sibling modules resolve correctly.
# ---------------------------------------------------------------------------
_etl_dir = os.path.dirname(os.path.abspath(__file__))

try:
    from .cache_primitives import (  # noqa: F401
        BloomFilter,
        CacheEntry,
        CacheStats,
        CompressionManager,
    )
    from .cache_storage import (  # noqa: F401
        DiskTierCache,
        MemoryTierCache,
    )
    from .cache_redis import RedisTierCache  # noqa: F401
    from .cache_warming import (  # noqa: F401
        CacheWarmingMixin,
        CacheWarmingResult,
    )
    from .cache_analytics import CacheAnalyticsMixin  # noqa: F401
except ImportError:
    # Standalone loading (no package context) – add etl dir to sys.path temporarily
    if _etl_dir not in sys.path:
        sys.path.insert(0, _etl_dir)
    from cache_primitives import (  # noqa: F401  # type: ignore[no-redef]
        BloomFilter,
        CacheEntry,
        CacheStats,
        CompressionManager,
    )
    from cache_storage import (  # noqa: F401  # type: ignore[no-redef]
        DiskTierCache,
        MemoryTierCache,
    )
    from cache_redis import RedisTierCache  # noqa: F401  # type: ignore[no-redef]
    from cache_warming import (  # noqa: F401  # type: ignore[no-redef]
        CacheWarmingMixin,
        CacheWarmingResult,
    )
    from cache_analytics import CacheAnalyticsMixin  # noqa: F401  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


class IntelligentCacheManager(CacheAnalyticsMixin, CacheWarmingMixin):
    """
    Multi-tier intelligent cache manager with automatic optimization.

    Features:
    - 3-tier caching: Memory (L1) -> Disk (L2) -> Redis (L3)
    - Bloom filter for 90% faster cache misses
    - Automatic tier promotion for hot keys
    - Access pattern analytics
    - Persistence across restarts
    - Cache warming for market open
    """

    def __init__(self,
                 cache_dir: str = "/tmp/intelligent_cache",
                 memory_size_mb: int = 256,
                 disk_size_mb: int = 2048,
                 redis_url: Optional[str] = None,
                 enable_analytics: bool = True,
                 bloom_filter_expected_items: int = 100000,
                 bloom_filter_fp_rate: float = 0.01):

        self.cache_dir = cache_dir
        self.enable_analytics = enable_analytics

        os.makedirs(cache_dir, exist_ok=True)

        # Initialize Bloom filter for fast negative lookups (~90% faster misses)
        bloom_persistence_path = os.path.join(cache_dir, 'bloom_filter.bin')
        self.bloom_filter = BloomFilter(
            expected_items=bloom_filter_expected_items,
            false_positive_rate=bloom_filter_fp_rate,
            persistence_path=bloom_persistence_path
        )

        self.bloom_filter_bypasses = 0  # Times full lookup was skipped via bloom filter

        # Initialize cache tiers
        self.memory_cache = MemoryTierCache(memory_size_mb, ttl_hours=1)
        self.disk_cache = DiskTierCache(cache_dir, disk_size_mb, ttl_hours=24)
        self.redis_cache = RedisTierCache(redis_url or "redis://localhost:6379", ttl_hours=48)

        # Cache statistics
        self.stats = CacheStats()

        # Access patterns for optimization
        self.access_patterns: Dict[str, Any] = {}
        self.optimization_interval = 3600  # 1 hour
        self.last_optimization = time.time()

        self._start_background_tasks()

        logger.info(
            "Initialized IntelligentCacheManager with 3-tier architecture + Bloom filter "
            f"(expected_items={bloom_filter_expected_items}, fp_rate={bloom_filter_fp_rate:.2%})"
        )

    async def get(self, key: str, category: str = "default") -> Optional[Any]:
        """
        Get item from cache with intelligent tier selection.

        Uses Bloom filter for fast negative lookups:
        - Bloom filter False -> key DEFINITELY not in cache, skip disk/Redis (~1ms)
        - Bloom filter True  -> proceed with full tier lookup (~10ms)
        """
        start_time = time.time()
        self.stats.total_requests += 1

        if self.enable_analytics:
            self._track_access(key, category)

        # L1: memory cache
        data = self.memory_cache.get(key)
        if data is not None:
            self.stats.hits += 1
            self.stats.memory_hits += 1
            await self._record_hit(key, 'memory', time.time() - start_time)
            return data

        # Bloom filter fast path: skip disk/Redis for definite misses
        if not self.bloom_filter.might_contain(key):
            self.bloom_filter_bypasses += 1
            self.stats.misses += 1
            await self._record_miss(key, category, bloom_filter_bypass=True)
            logger.debug(f"Bloom filter bypass for {key} ({time.time() - start_time:.4f}s)")
            return None

        # L2: disk cache
        data = self.disk_cache.get(key)
        if data is not None:
            self.stats.hits += 1
            self.stats.disk_hits += 1
            self.memory_cache.set(key, data)
            await self._record_hit(key, 'disk', time.time() - start_time)
            return data

        # L3: Redis cache
        if self.redis_cache.is_available():
            data = self.redis_cache.get(key)
            if data is not None:
                self.stats.hits += 1
                self.stats.redis_hits += 1
                self.disk_cache.set(key, data)
                self.memory_cache.set(key, data)
                await self._record_hit(key, 'redis', time.time() - start_time)
                return data

        # Definite miss (bloom filter false positive or true miss)
        self.stats.misses += 1
        await self._record_miss(key, category, bloom_filter_bypass=False)
        return None

    async def set(self, key: str, data: Any, category: str = "default",
                  ttl_hours: Optional[int] = None) -> bool:
        """
        Set item in appropriate cache tiers and add key to Bloom filter.
        """
        if data is None:
            return False

        success = False
        data_size = self._estimate_data_size(data)
        access_frequency = self._get_access_frequency(key)

        # L1: store hot data or small items in memory
        if access_frequency > 0.1 or data_size < 10240:
            if self.memory_cache.set(key, data, ttl_hours and ttl_hours * 3600):
                success = True

        # L2: disk storage for medium-term persistence
        if self.disk_cache.set(key, data, ttl_hours):
            success = True

        # L3: Redis for distributed/shared access
        if self.redis_cache.is_available() and (access_frequency > 0.05 or category == "shared"):
            self.redis_cache.set(key, data, ttl_hours and ttl_hours * 3600)

        # Register key in Bloom filter so future gets skip expensive lookups
        if success:
            self.bloom_filter.add(key)

        if self.enable_analytics:
            self._track_write(key, category, data_size)

        return success

    async def delete(self, key: str) -> bool:
        """Delete item from all cache tiers."""
        results = [
            self.memory_cache.delete(key),
            self.disk_cache.delete(key),
        ]
        if self.redis_cache.is_available():
            results.append(self.redis_cache.delete(key))
        return any(results)

    async def bulk_get(self, keys: List[str], category: str = "default") -> Dict[str, Any]:
        """Get multiple items in parallel."""
        values = await asyncio.gather(
            *[self.get(k, category) for k in keys],
            return_exceptions=True
        )
        return {
            k: v for k, v in zip(keys, values)
            if v is not None and not isinstance(v, Exception)
        }

    async def bulk_set(self, items: Dict[str, Any], category: str = "default",
                       ttl_hours: Optional[int] = None) -> Dict[str, bool]:
        """Set multiple items in parallel."""
        successes = await asyncio.gather(
            *[self.set(k, v, category, ttl_hours) for k, v in items.items()],
            return_exceptions=True
        )
        return {
            k: (s if not isinstance(s, Exception) else False)
            for k, s in zip(items.keys(), successes)
        }

    def get_comprehensive_stats(self) -> Dict:
        """Get comprehensive cache statistics including Bloom filter metrics."""
        memory_stats = self.memory_cache.get_stats()
        disk_stats = self.disk_cache.get_stats()
        redis_stats = self.redis_cache.get_stats()
        bloom_stats = self.bloom_filter.get_stats()
        analytics = self.get_analytics_summary()

        total_entries = memory_stats.get('entries', 0) + disk_stats.get('entries', 0)
        if redis_stats.get('available'):
            total_entries += redis_stats.get('entries', 0)

        bloom_bypass_rate = (
            self.bloom_filter_bypasses / max(self.stats.misses, 1)
            if self.stats.misses > 0 else 0
        )

        return {
            'overview': {
                'total_requests': self.stats.total_requests,
                'hit_rate': self.stats.hit_rate,
                'total_entries': total_entries,
                'memory_hit_rate': self.stats.memory_hit_rate
            },
            'performance': {
                'hits': self.stats.hits,
                'misses': self.stats.misses,
                'memory_hits': self.stats.memory_hits,
                'disk_hits': self.stats.disk_hits,
                'redis_hits': self.stats.redis_hits
            },
            'bloom_filter': {
                'enabled': True,
                'size_bytes': bloom_stats['size_bytes'],
                'items_tracked': bloom_stats['items_added'],
                'checks_performed': bloom_stats['checks_performed'],
                'true_negatives': bloom_stats['true_negatives'],
                'true_negative_rate': bloom_stats['true_negative_rate'],
                'estimated_fp_rate': bloom_stats['estimated_fp_rate'],
                'target_fp_rate': bloom_stats['target_fp_rate'],
                'fill_ratio': bloom_stats['fill_ratio'],
                'bypasses': self.bloom_filter_bypasses,
                'bypass_rate': bloom_bypass_rate,
                'capacity_remaining': bloom_stats['capacity_remaining']
            },
            'tiers': {
                'memory': memory_stats,
                'disk': disk_stats,
                'redis': redis_stats
            },
            'analytics': analytics
        }

    async def clear_all(self) -> bool:
        """Clear all cache tiers and Bloom filter."""
        try:
            self.memory_cache = MemoryTierCache(
                self.memory_cache.max_size_bytes // (1024 * 1024),
                self.memory_cache.ttl_seconds // 3600
            )

            if os.path.exists(self.cache_dir):
                for file in os.listdir(self.cache_dir):
                    if file.endswith('.cache'):
                        os.unlink(os.path.join(self.cache_dir, file))
                self.disk_cache._init_index_db()

            if self.redis_cache.is_available():
                keys = self.redis_cache.redis_client.keys("cache:*")
                if keys:
                    self.redis_cache.redis_client.delete(*keys)

            self.bloom_filter.clear()
            self.bloom_filter_bypasses = 0
            self.stats = CacheStats()
            self.access_patterns.clear()

            logger.info("All cache tiers and Bloom filter cleared")
            return True

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False


if __name__ == "__main__":
    # Quick smoke-test: set/get a value and print comprehensive stats.
    import logging as _logging
    from datetime import datetime

    _logging.basicConfig(level=_logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    async def _smoke_test():
        cache = IntelligentCacheManager(
            cache_dir="/tmp/test_cache",
            memory_size_mb=64,
            disk_size_mb=256,
        )
        test_data = {'ticker': 'AAPL', 'price': 150.25, 'timestamp': datetime.now().isoformat()}
        await cache.set('AAPL:stock_data', test_data, 'stocks')
        result = await cache.get('AAPL:stock_data', 'stocks')
        assert result == test_data, "Round-trip mismatch"
        stats = cache.get_comprehensive_stats()
        logger.info("Stats: %s", stats['overview'])
        logger.info("Bloom filter: %s", stats['bloom_filter'])
        logger.info("Smoke test passed.")

    asyncio.run(_smoke_test())
