"""
Cache analytics mixin for IntelligentCacheManager.

Provides access-pattern tracking, hit/miss recording, and background
optimization of cache tier promotion/eviction based on observed traffic.

Requires the host class to expose:
    self.enable_analytics  - bool flag
    self.access_patterns   - dict of key -> pattern dict
    self.memory_cache      - MemoryTierCache instance
    self.disk_cache        - DiskTierCache instance
    self.last_optimization - float timestamp
    self.optimization_interval - int seconds
"""

import logging
import threading
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)


class CacheAnalyticsMixin:
    """
    Mixin that adds access-pattern analytics and background optimization
    to IntelligentCacheManager.
    """

    def _estimate_data_size(self, data: Any) -> int:
        """
        Estimate size of data in bytes.
        SECURITY: Uses JSON for size estimation - no pickle to prevent code execution.
        """
        import json
        try:
            if isinstance(data, str):
                return len(data.encode('utf-8'))
            elif isinstance(data, (dict, list)):
                return len(json.dumps(data, default=str).encode('utf-8'))
            elif hasattr(data, '__dict__'):
                return len(json.dumps(data.__dict__, default=str).encode('utf-8'))
            else:
                return len(str(data).encode('utf-8'))
        except Exception:
            return 1024  # Default estimate

    def _get_access_frequency(self, key: str) -> float:
        """Get access frequency for key from analytics (accesses per hour)."""
        if not self.enable_analytics or key not in self.access_patterns:
            return 0.0

        pattern = self.access_patterns[key]
        total_accesses = pattern.get('reads', 0) + pattern.get('writes', 0)
        time_window = time.time() - pattern.get('first_seen', time.time())

        return total_accesses / max(time_window / 3600, 1)

    def _track_access(self, key: str, category: str):
        """Track read access patterns for optimization."""
        now = time.time()

        if key not in self.access_patterns:
            self.access_patterns[key] = {
                'first_seen': now,
                'last_accessed': now,
                'reads': 0,
                'writes': 0,
                'category': category
            }

        pattern = self.access_patterns[key]
        pattern['reads'] += 1
        pattern['last_accessed'] = now

    def _track_write(self, key: str, category: str, size: int):
        """Track write patterns and rolling average data size."""
        now = time.time()

        if key not in self.access_patterns:
            self.access_patterns[key] = {
                'first_seen': now,
                'last_accessed': now,
                'reads': 0,
                'writes': 0,
                'category': category,
                'avg_size': size
            }
        else:
            pattern = self.access_patterns[key]
            pattern['writes'] += 1
            pattern['last_accessed'] = now
            if 'avg_size' in pattern:
                pattern['avg_size'] = (pattern['avg_size'] + size) / 2
            else:
                pattern['avg_size'] = size

    async def _record_hit(self, key: str, tier: str, response_time: float):
        """Record cache hit for analytics."""
        if self.enable_analytics:
            logger.debug(f"Cache hit: {key} from {tier} in {response_time*1000:.1f}ms")

    async def _record_miss(self, key: str, category: str, bloom_filter_bypass: bool = False):
        """Record cache miss for analytics."""
        if self.enable_analytics:
            bypass_info = " (bloom filter bypass)" if bloom_filter_bypass else ""
            logger.debug(f"Cache miss: {key} (category: {category}){bypass_info}")

    def _start_background_tasks(self):
        """Start background optimization tasks."""

        def cleanup_task():
            while True:
                try:
                    self.disk_cache._cleanup_expired()

                    if time.time() - self.last_optimization > self.optimization_interval:
                        self._optimize_cache_strategy()
                        self.last_optimization = time.time()

                    self.bloom_filter.save_to_disk()

                    time.sleep(300)  # Blocking sleep OK: runs in dedicated daemon thread

                except Exception as e:
                    logger.error(f"Background cleanup task error: {e}")
                    time.sleep(60)  # Blocking sleep OK: runs in dedicated daemon thread

        cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
        cleanup_thread.start()

    def _optimize_cache_strategy(self):
        """Optimize caching strategy based on access patterns."""
        if not self.enable_analytics:
            return

        logger.info("Running cache optimization...")

        try:
            hot_keys = []
            cold_keys = []

            for key in self.access_patterns:
                frequency = self._get_access_frequency(key)

                if frequency > 0.5:  # More than 30 accesses per hour
                    hot_keys.append((key, frequency))
                elif frequency < 0.01:  # Less than 1 access per 100 hours
                    cold_keys.append(key)

            # Promote hot keys to memory
            for key, freq in hot_keys[:50]:  # Top 50 hot keys
                data = self.disk_cache.get(key)
                if data and not self.memory_cache.get(key):
                    self.memory_cache.set(key, data)
                    logger.debug(f"Promoted hot key {key} to memory (freq: {freq:.2f})")

            # Evict cold keys from memory
            for key in cold_keys:
                if self.memory_cache.delete(key):
                    logger.debug(f"Evicted cold key {key} from memory")

            logger.info(
                f"Cache optimization complete: {len(hot_keys)} hot keys, "
                f"{len(cold_keys)} cold keys"
            )

        except Exception as e:
            logger.error(f"Cache optimization error: {e}")

    def get_analytics_summary(self) -> Dict:
        """Return a summary of current access-pattern analytics."""
        if not self.enable_analytics:
            return {'enabled': False, 'tracked_keys': 0, 'hot_keys': 0}

        hot_key_count = sum(
            1 for k in self.access_patterns
            if self._get_access_frequency(k) > 0.5
        )

        return {
            'enabled': True,
            'tracked_keys': len(self.access_patterns),
            'hot_keys': hot_key_count
        }
