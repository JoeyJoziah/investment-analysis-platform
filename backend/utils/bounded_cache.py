"""
Bounded Cache Implementation
Provides thread-safe, size-limited caching with LRU eviction.
"""

import threading
import time
from typing import Any, Dict, Optional, Tuple
from collections import OrderedDict
import logging
import sys

logger = logging.getLogger(__name__)


class BoundedLRUCache:
    """
    Thread-safe, bounded LRU cache with TTL support.
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        """
        Initialize bounded LRU cache.
        
        Args:
            max_size: Maximum number of items in cache
            default_ttl: Default TTL in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        
        # Use OrderedDict for LRU ordering
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._lock = threading.RLock()
        
        # Metrics
        self._metrics = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size_bytes': 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            if key not in self._cache:
                self._metrics['misses'] += 1
                return None
            
            value, expiry = self._cache[key]
            
            # Check if expired
            if time.time() > expiry:
                del self._cache[key]
                self._metrics['misses'] += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._metrics['hits'] += 1
            return value
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> None:
        """
        Set value in cache with TTL.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        with self._lock:
            ttl = ttl or self.default_ttl
            expiry = time.time() + ttl
            
            # If key exists, move to end
            if key in self._cache:
                self._cache.move_to_end(key)
            
            # Add/update value
            self._cache[key] = (value, expiry)
            
            # Update size estimate
            self._update_size_estimate(key, value)
            
            # Evict if over capacity
            while len(self._cache) > self.max_size:
                self._evict_oldest()
    
    def _evict_oldest(self) -> None:
        """Evict oldest (least recently used) item."""
        if self._cache:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            self._metrics['evictions'] += 1
            logger.debug(f"Evicted cache key: {oldest_key}")
    
    def _update_size_estimate(self, key: str, value: Any) -> None:
        """Update estimated cache size in bytes."""
        try:
            # Rough estimate of size
            key_size = sys.getsizeof(key)
            value_size = sys.getsizeof(value)
            self._metrics['size_bytes'] = sum(
                sys.getsizeof(k) + sys.getsizeof(v[0])
                for k, v in self._cache.items()
            )
        except Exception as e:
            logger.debug(f"Could not estimate cache size: {e}")
    
    def delete(self, key: str) -> bool:
        """
        Delete key from cache.
        
        Args:
            key: Cache key
        
        Returns:
            True if key was deleted
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cached items."""
        with self._lock:
            self._cache.clear()
            self._metrics['evictions'] += len(self._cache)
    
    def cleanup_expired(self) -> int:
        """
        Remove expired entries.
        
        Returns:
            Number of entries removed
        """
        with self._lock:
            current_time = time.time()
            expired_keys = [
                k for k, (_, expiry) in self._cache.items()
                if current_time > expiry
            ]
            
            for key in expired_keys:
                del self._cache[key]
            
            return len(expired_keys)
    
    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get cache metrics."""
        with self._lock:
            metrics = self._metrics.copy()
            metrics['current_size'] = len(self._cache)
            metrics['max_size'] = self.max_size
            
            # Calculate hit rate
            total = metrics['hits'] + metrics['misses']
            metrics['hit_rate'] = (
                (metrics['hits'] / total * 100) if total > 0 else 0
            )
            
            return metrics
    
    def get_all_keys(self) -> list:
        """Get all cache keys (for debugging)."""
        with self._lock:
            return list(self._cache.keys())


class BoundedFallbackCache(BoundedLRUCache):
    """
    Specialized bounded cache for circuit breaker fallback.
    """
    
    def __init__(self, max_size: int = 500, default_ttl: int = 600):
        """
        Initialize fallback cache with conservative limits.
        
        Args:
            max_size: Maximum items (smaller for fallback)
            default_ttl: Default TTL (longer for fallback)
        """
        super().__init__(max_size, default_ttl)
        
        # Additional fallback-specific settings
        self.emergency_ttl = 3600  # 1 hour during emergencies
        self.stale_ttl = 86400  # 24 hours for stale data
    
    def set_emergency(
        self,
        key: str,
        value: Any
    ) -> None:
        """
        Set value with emergency TTL.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        self.set(key, value, ttl=self.emergency_ttl)
    
    def set_stale(
        self,
        key: str,
        value: Any
    ) -> None:
        """
        Set stale value with extended TTL.
        
        Args:
            key: Cache key
            value: Stale value to cache
        """
        # Mark as stale
        if isinstance(value, dict):
            value['_stale'] = True
            value['_stale_timestamp'] = time.time()
        
        self.set(key, value, ttl=self.stale_ttl)
    
    def get_with_staleness(
        self,
        key: str,
        max_stale_seconds: int = 3600
    ) -> Tuple[Optional[Any], bool]:
        """
        Get value with staleness check.
        
        Args:
            key: Cache key
            max_stale_seconds: Maximum acceptable staleness
        
        Returns:
            Tuple of (value, is_stale)
        """
        value = self.get(key)
        
        if value is None:
            return None, False
        
        # Check if marked as stale
        if isinstance(value, dict) and value.get('_stale'):
            stale_timestamp = value.get('_stale_timestamp', 0)
            stale_age = time.time() - stale_timestamp
            
            if stale_age > max_stale_seconds:
                # Too stale, don't use
                return None, False
            
            return value, True
        
        return value, False


# Global bounded cache instances - optimized for 6000+ stocks
bounded_cache = BoundedLRUCache(max_size=50000, default_ttl=300)  # Increased for 6000+ stocks
fallback_cache = BoundedFallbackCache(max_size=5000, default_ttl=600)  # Increased fallback capacity