"""
Inference caching system for the ML pipeline optimization module.

Extracted from pipeline_optimization.py.  Contains InferenceCache only.
Import via the original path (backend.ml.pipeline_optimization) or directly from here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import numpy as np

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]
    _PANDAS_AVAILABLE = False

logger = logging.getLogger(__name__)


class InferenceCache:
    """Intelligent caching system for model predictions"""

    def __init__(
        self,
        max_size: int = 10000,
        ttl_seconds: int = 3600,
        redis_url: Optional[str] = None,
    ) -> None:
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds

        # In-memory cache
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, datetime] = {}
        self.cache_lock = threading.Lock()

        # Redis cache (optional)
        self.redis_client = None
        if redis_url:
            try:
                import redis
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
                logger.info("Redis cache initialized")
            except Exception as exc:
                logger.warning(f"Redis cache initialization failed: {exc}")

        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get(self, cache_key: str) -> Optional[Any]:
        """Get cached prediction"""

        with self.cache_lock:
            if cache_key in self.cache:
                cache_entry = self.cache[cache_key]
                if (
                    datetime.now(timezone.utc) - cache_entry['timestamp']
                    < timedelta(seconds=self.ttl_seconds)
                ):
                    self.access_times[cache_key] = datetime.now(timezone.utc)
                    self.hits += 1
                    return cache_entry['data']
                else:
                    del self.cache[cache_key]
                    del self.access_times[cache_key]

        if self.redis_client:
            try:
                cached_data = self.redis_client.get(f"prediction:{cache_key}")
                if cached_data:
                    data = self._deserialize_cache_data(cached_data)
                    self._store_in_memory(cache_key, data)
                    self.hits += 1
                    return data
            except Exception as exc:
                logger.error(f"Error reading from Redis cache: {exc}")

        self.misses += 1
        return None

    def set(self, cache_key: str, data: Any) -> None:
        """Store prediction in cache"""

        self._store_in_memory(cache_key, data)

        if self.redis_client:
            try:
                serialized_data = self._serialize_cache_data(data)
                self.redis_client.setex(
                    f"prediction:{cache_key}",
                    self.ttl_seconds,
                    serialized_data,
                )
            except Exception as exc:
                logger.error(f"Error storing in Redis cache: {exc}")

    def generate_cache_key(
        self,
        model_name: str,
        input_data: np.ndarray,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate cache key for input"""

        input_hash = hashlib.md5(input_data.tobytes()).hexdigest()
        param_str = json.dumps(parameters or {}, sort_keys=True)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()
        return f"{model_name}:{input_hash}:{param_hash}"

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""

        total_requests = self.hits + self.misses
        hit_ratio = self.hits / total_requests if total_requests > 0 else 0

        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_ratio': hit_ratio,
            'evictions': self.evictions,
            'cache_size': len(self.cache),
            'max_size': self.max_size,
        }

    def clear(self) -> None:
        """Clear all caches"""

        with self.cache_lock:
            self.cache.clear()
            self.access_times.clear()

        if self.redis_client:
            try:
                keys = self.redis_client.keys("prediction:*")
                if keys:
                    self.redis_client.delete(*keys)
            except Exception as exc:
                logger.error(f"Error clearing Redis cache: {exc}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _store_in_memory(self, cache_key: str, data: Any) -> None:
        """Store data in in-memory cache"""

        with self.cache_lock:
            if len(self.cache) >= self.max_size:
                self._evict_lru()

            self.cache[cache_key] = {
                'data': data,
                'timestamp': datetime.now(timezone.utc),
            }
            self.access_times[cache_key] = datetime.now(timezone.utc)

    def _evict_lru(self) -> None:
        """Evict least recently used entry"""

        if not self.access_times:
            return

        lru_key = min(self.access_times.items(), key=lambda x: x[1])[0]
        del self.cache[lru_key]
        del self.access_times[lru_key]
        self.evictions += 1

    def _serialize_cache_data(self, data: Any) -> str:
        """
        Safely serialize data to JSON for Redis storage.

        SECURITY: Uses JSON instead of pickle to prevent arbitrary code execution.
        Handles numpy arrays and datetime objects.
        """

        def json_serializer(obj: Any) -> Any:
            if isinstance(obj, np.ndarray):
                return {'__numpy__': True, 'data': obj.tolist(), 'dtype': str(obj.dtype)}
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, datetime):
                return {'__datetime__': True, 'value': obj.isoformat()}
            elif _PANDAS_AVAILABLE and isinstance(obj, pd.DataFrame):
                return {
                    '__dataframe__': True,
                    'data': obj.to_dict(orient='records'),
                    'columns': list(obj.columns),
                }
            elif _PANDAS_AVAILABLE and isinstance(obj, pd.Series):
                return {'__series__': True, 'data': obj.tolist(), 'index': list(obj.index)}
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        return json.dumps(data, default=json_serializer)

    def _deserialize_cache_data(self, data: bytes) -> Any:
        """
        Safely deserialize JSON data from Redis.

        SECURITY: Uses JSON instead of pickle to prevent arbitrary code execution.
        Restores numpy arrays and datetime objects.
        """

        def object_hook(obj: Dict[str, Any]) -> Any:
            if obj.get('__numpy__'):
                return np.array(obj['data'], dtype=obj['dtype'])
            elif obj.get('__datetime__'):
                return datetime.fromisoformat(obj['value'])
            elif obj.get('__dataframe__') and _PANDAS_AVAILABLE:
                return pd.DataFrame(obj['data'], columns=obj['columns'])
            elif obj.get('__series__') and _PANDAS_AVAILABLE:
                return pd.Series(obj['data'], index=obj['index'])
            return obj

        return json.loads(data.decode('utf-8'), object_hook=object_hook)


__all__ = ["InferenceCache"]
