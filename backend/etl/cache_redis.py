"""
Redis cache tier implementation: RedisTierCache (L3).

Distributed Redis storage for shared caching across multiple processes/nodes.
Compression is handled via CompressionManager from cache_primitives.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

import redis

try:
    from .cache_primitives import CompressionManager
except ImportError:
    import os as _os, sys as _sys
    _here = _os.path.dirname(_os.path.abspath(__file__))
    if _here not in _sys.path:
        _sys.path.insert(0, _here)
    from cache_primitives import CompressionManager  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


class RedisTierCache:
    """L3 Cache - Distributed Redis storage for shared caching"""

    def __init__(self, redis_url: str = "redis://localhost:6379", ttl_hours: int = 48):
        self.ttl_seconds = ttl_hours * 3600
        self.redis_client = None

        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=False)
            # Test connection
            self.redis_client.ping()
            logger.info(f"Initialized Redis cache: {redis_url}, TTL: {ttl_hours}h")

        except Exception as e:
            logger.warning(f"Redis cache not available: {e}")
            self.redis_client = None

    def is_available(self) -> bool:
        """Check if Redis is available"""
        return self.redis_client is not None

    def get(self, key: str) -> Optional[Any]:
        """Get item from Redis cache"""
        if not self.redis_client:
            return None

        try:
            # Get compressed data
            compressed_data = self.redis_client.get(f"cache:{key}")
            if not compressed_data:
                return None

            # Decompress
            data = CompressionManager.decompress_data(compressed_data, 'gzip')

            # Update access count
            self.redis_client.incr(f"cache:{key}:hits")

            return data

        except Exception as e:
            logger.warning(f"Redis cache get error for {key}: {e}")
            return None

    def set(self, key: str, data: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Set item in Redis cache"""
        if not self.redis_client:
            return False

        try:
            # Compress data
            compressed_data, _ = CompressionManager.compress_data(data, 'gzip')

            # Store with TTL
            ttl = ttl_seconds or self.ttl_seconds

            pipe = self.redis_client.pipeline()
            pipe.setex(f"cache:{key}", ttl, compressed_data)
            pipe.setex(f"cache:{key}:created", ttl, datetime.now().isoformat())
            pipe.execute()

            return True

        except Exception as e:
            logger.warning(f"Redis cache set error for {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete item from Redis cache"""
        if not self.redis_client:
            return False

        try:
            pipe = self.redis_client.pipeline()
            pipe.delete(f"cache:{key}")
            pipe.delete(f"cache:{key}:hits")
            pipe.delete(f"cache:{key}:created")
            results = pipe.execute()

            return any(results)

        except Exception as e:
            logger.warning(f"Redis cache delete error for {key}: {e}")
            return False

    def get_stats(self) -> Dict:
        """Get Redis cache statistics"""
        if not self.redis_client:
            return {'available': False}

        try:
            info = self.redis_client.info()

            # Count our cache keys
            cache_keys = self.redis_client.keys("cache:*")
            cache_entries = len(
                [k for k in cache_keys if not k.endswith((b':hits', b':created'))]
            )

            return {
                'available': True,
                'entries': cache_entries,
                'memory_usage_mb': info.get('used_memory', 0) / (1024 * 1024),
                'connected_clients': info.get('connected_clients', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'hit_rate': info.get('keyspace_hits', 0) /
                            max(
                                info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0),
                                1
                            )
            }

        except Exception as e:
            logger.warning(f"Error getting Redis stats: {e}")
            return {'available': False, 'error': str(e)}
