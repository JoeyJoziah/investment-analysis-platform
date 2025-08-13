"""
Cache utilities for Redis operations
"""

import redis.asyncio as redis
from typing import Optional, Any
import json
import logging
from backend.config.settings import settings

logger = logging.getLogger(__name__)

# Global Redis connection pool
_redis_pool: Optional[redis.ConnectionPool] = None
_redis_client: Optional[redis.Redis] = None


async def get_redis() -> redis.Redis:
    """
    Get Redis client instance with connection pooling
    """
    global _redis_pool, _redis_client
    
    if _redis_client is None:
        # Parse Redis URL to extract password if present
        redis_url = settings.REDIS_URL
        
        # Create connection pool
        _redis_pool = redis.ConnectionPool.from_url(
            redis_url,
            max_connections=50,
            decode_responses=True
        )
        
        # Create Redis client
        _redis_client = redis.Redis(connection_pool=_redis_pool)
        
        # Test connection
        try:
            await _redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    return _redis_client


async def close_redis():
    """
    Close Redis connection pool
    """
    global _redis_pool, _redis_client
    
    if _redis_client:
        await _redis_client.close()
        _redis_client = None
    
    if _redis_pool:
        await _redis_pool.disconnect()
        _redis_pool = None


class CacheManager:
    """
    High-level cache operations with JSON serialization
    """
    
    def __init__(self, prefix: str = ""):
        self.prefix = prefix
        self.redis = None
    
    async def _get_redis(self):
        if self.redis is None:
            self.redis = await get_redis()
        return self.redis
    
    def _make_key(self, key: str) -> str:
        """Create namespaced cache key"""
        if self.prefix:
            return f"{self.prefix}:{key}"
        return key
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        redis = await self._get_redis()
        value = await redis.get(self._make_key(key))
        
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache with optional TTL"""
        redis = await self._get_redis()
        
        if not isinstance(value, str):
            value = json.dumps(value)
        
        key = self._make_key(key)
        
        if ttl:
            await redis.setex(key, ttl, value)
        else:
            await redis.set(key, value)
    
    async def delete(self, key: str):
        """Delete key from cache"""
        redis = await self._get_redis()
        await redis.delete(self._make_key(key))
    
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        redis = await self._get_redis()
        return await redis.exists(self._make_key(key)) > 0
    
    async def expire(self, key: str, ttl: int):
        """Set expiration time for key"""
        redis = await self._get_redis()
        await redis.expire(self._make_key(key), ttl)
    
    async def get_many(self, keys: list) -> dict:
        """Get multiple values at once"""
        redis = await self._get_redis()
        
        # Make namespaced keys
        namespaced_keys = [self._make_key(k) for k in keys]
        
        # Get values
        values = await redis.mget(namespaced_keys)
        
        # Build result dict
        result = {}
        for key, value in zip(keys, values):
            if value:
                try:
                    result[key] = json.loads(value)
                except json.JSONDecodeError:
                    result[key] = value
            else:
                result[key] = None
        
        return result
    
    async def set_many(self, data: dict, ttl: Optional[int] = None):
        """Set multiple values at once"""
        redis = await self._get_redis()
        
        # Prepare data
        to_set = {}
        for key, value in data.items():
            if not isinstance(value, str):
                value = json.dumps(value)
            to_set[self._make_key(key)] = value
        
        # Set all at once
        await redis.mset(to_set)
        
        # Set expiration if needed
        if ttl:
            for key in to_set:
                await redis.expire(key, ttl)
    
    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment counter"""
        redis = await self._get_redis()
        return await redis.incrby(self._make_key(key), amount)
    
    async def decrement(self, key: str, amount: int = 1) -> int:
        """Decrement counter"""
        redis = await self._get_redis()
        return await redis.decrby(self._make_key(key), amount)


# Specialized cache managers
stock_cache = CacheManager("stocks")
market_cache = CacheManager("market")
analysis_cache = CacheManager("analysis")
user_cache = CacheManager("users")


async def init_cache():
    """Initialize cache connection"""
    try:
        redis_client = await get_redis()
        await redis_client.ping()
        logger.info("Cache initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize cache: {e}")
        raise


def get_redis_client():
    """Get synchronous Redis client for Celery tasks"""
    import redis
    return redis.from_url(settings.REDIS_URL, decode_responses=True)