"""
Cache Manager - Provides a unified interface for caching operations.

This module wraps the comprehensive cache implementation and provides
a simpler interface for common caching operations.
"""

import logging
from typing import Any, Optional, Dict
from datetime import timedelta

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Simple cache manager that wraps Redis or provides in-memory fallback.
    
    Provides a clean interface for caching operations used throughout
    the application, particularly by the analytics agents.
    """
    
    def __init__(self, redis_client=None, default_ttl: int = 3600):
        """
        Initialize the cache manager.
        
        Args:
            redis_client: Optional Redis client instance
            default_ttl: Default TTL in seconds for cached items
        """
        self.redis_client = redis_client
        self.default_ttl = default_ttl
        self._local_cache: Dict[str, Dict[str, Any]] = {}
        
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        try:
            if self.redis_client:
                import json
                data = await self.redis_client.get(key)
                if data:
                    return json.loads(data)
            else:
                cached = self._local_cache.get(key)
                if cached:
                    import time
                    if cached.get("expires_at", 0) > time.time():
                        return cached.get("value")
                    else:
                        del self._local_cache[key]
        except Exception as e:
            logger.warning(f"Cache get error for {key}: {e}")
        return None
        
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in cache."""
        ttl = ttl or self.default_ttl
        try:
            if self.redis_client:
                import json
                await self.redis_client.setex(key, ttl, json.dumps(value))
            else:
                import time
                self._local_cache[key] = {
                    "value": value,
                    "expires_at": time.time() + ttl
                }
            return True
        except Exception as e:
            logger.warning(f"Cache set error for {key}: {e}")
            return False
            
    async def delete(self, key: str) -> bool:
        """Delete a value from cache."""
        try:
            if self.redis_client:
                await self.redis_client.delete(key)
            else:
                self._local_cache.pop(key, None)
            return True
        except Exception as e:
            logger.warning(f"Cache delete error for {key}: {e}")
            return False
            
    async def get_stale(self, key: str, max_age: int = 86400) -> Optional[Any]:
        """Get a value from cache even if expired (for fallback)."""
        try:
            if self.redis_client:
                import json
                data = await self.redis_client.get(f"{key}:backup")
                if data:
                    return json.loads(data)
            else:
                cached = self._local_cache.get(key)
                if cached:
                    import time
                    created_at = cached.get("expires_at", 0) - self.default_ttl
                    if created_at + max_age > time.time():
                        return cached.get("value")
        except Exception as e:
            logger.warning(f"Cache get_stale error for {key}: {e}")
        return None
        
    async def exists(self, key: str) -> bool:
        """Check if a key exists in cache."""
        try:
            if self.redis_client:
                return await self.redis_client.exists(key)
            else:
                cached = self._local_cache.get(key)
                if cached:
                    import time
                    if cached.get("expires_at", 0) > time.time():
                        return True
                    else:
                        del self._local_cache[key]
        except Exception as e:
            logger.warning(f"Cache exists error for {key}: {e}")
        return False


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


async def get_cache_manager() -> CacheManager:
    """Get or create the global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        try:
            from backend.utils.cache import get_redis
            redis_client = await get_redis()
            _cache_manager = CacheManager(redis_client=redis_client)
        except Exception as e:
            logger.warning(f"Failed to initialize Redis cache manager: {e}")
            _cache_manager = CacheManager()
    return _cache_manager


def get_cache_manager_sync() -> CacheManager:
    """Get the cache manager instance synchronously."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager
