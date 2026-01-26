"""
Cache utilities for Redis operations
"""

import redis.asyncio as redis
from typing import Optional, Any
import json
import logging
from backend.config.settings import settings

# Re-export get_cache_manager for backward compatibility
from backend.utils.cache_manager import get_cache_manager

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


def cache_with_ttl(ttl: int = 300, prefix: str = None, skip_args: int = 0):
    """
    Cache decorator for async functions with Redis backend.

    Caches function results in Redis with specified TTL. Falls back to
    direct function execution if Redis is unavailable.

    Args:
        ttl: Time-to-live in seconds for cached values (default: 300 = 5 minutes)
        prefix: Optional cache key prefix (default: function module + name)
        skip_args: Number of positional args to skip in key generation
                   (useful for skipping 'self', 'request', 'db' params)

    Example:
        @cache_with_ttl(ttl=300)
        async def get_stock_analysis(ticker: str) -> dict:
            ...

    Note: Objects like db sessions and request objects are automatically
    excluded from cache key generation.
    """
    import functools
    import inspect
    import hashlib

    # Types to exclude from cache key generation (non-serializable or request-specific)
    EXCLUDED_TYPES = (
        'AsyncSession', 'Session', 'Request', 'Response', 'WebSocket',
        'BackgroundTasks', 'User', 'HTTPConnection'
    )

    def _serialize_arg(arg: Any) -> str:
        """Serialize an argument for cache key generation."""
        from datetime import date, datetime
        from enum import Enum

        if arg is None:
            return "None"
        if isinstance(arg, (str, int, float, bool)):
            return str(arg)
        if isinstance(arg, (datetime, date)):
            return arg.isoformat()
        if isinstance(arg, Enum):
            return str(arg.value)
        if isinstance(arg, (list, tuple)):
            return f"[{','.join(_serialize_arg(a) for a in arg)}]"
        if isinstance(arg, dict):
            sorted_items = sorted(arg.items(), key=lambda x: str(x[0]))
            return f"{{{','.join(f'{k}:{_serialize_arg(v)}' for k, v in sorted_items)}}}"
        if hasattr(arg, 'model_dump'):  # Pydantic v2
            return _serialize_arg(arg.model_dump())
        if hasattr(arg, 'dict'):  # Pydantic v1
            return _serialize_arg(arg.dict())
        # For other objects, use class name (they won't be part of cache key logic)
        return f"<{type(arg).__name__}>"

    def _should_include_arg(arg: Any) -> bool:
        """Check if argument should be included in cache key."""
        type_name = type(arg).__name__
        return type_name not in EXCLUDED_TYPES

    def _generate_cache_key(func_name: str, args: tuple, kwargs: dict, skip: int) -> str:
        """Generate a deterministic cache key from function name and arguments."""
        key_parts = [func_name]

        # Process positional args (skip first N as specified)
        for arg in args[skip:]:
            if _should_include_arg(arg):
                key_parts.append(_serialize_arg(arg))

        # Process keyword args (sorted for consistency)
        for k, v in sorted(kwargs.items()):
            if _should_include_arg(v):
                key_parts.append(f"{k}={_serialize_arg(v)}")

        # Create hash for long keys to keep them manageable
        key_string = "|".join(key_parts)
        if len(key_string) > 200:
            key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]
            return f"{func_name}:{key_hash}"

        return key_string.replace(" ", "_")

    def _convert_to_serializable(obj: Any) -> Any:
        """Convert an object to a JSON-serializable format."""
        from datetime import date, datetime
        from enum import Enum
        from decimal import Decimal

        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, Enum):
            return obj.value
        if hasattr(obj, 'model_dump'):  # Pydantic v2
            return obj.model_dump(mode='json')
        if hasattr(obj, 'dict'):  # Pydantic v1
            return obj.dict()
        if isinstance(obj, (list, tuple)):
            return [_convert_to_serializable(item) for item in obj]
        if isinstance(obj, dict):
            return {k: _convert_to_serializable(v) for k, v in obj.items()}
        # Fallback: try to convert to string
        return str(obj)

    def _serialize_result(result: Any) -> str:
        """Serialize a result for storage in Redis cache."""
        serializable = _convert_to_serializable(result)
        return json.dumps(serializable)

    def decorator(func):
        # Determine the cache key prefix
        func_prefix = prefix or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"cache:{func_prefix}:{_generate_cache_key(func.__name__, args, kwargs, skip_args)}"

            try:
                # Try to get from cache
                redis_client = await get_redis()
                cached_value = await redis_client.get(cache_key)

                if cached_value is not None:
                    logger.debug(f"Cache HIT for key: {cache_key}")
                    try:
                        return json.loads(cached_value)
                    except json.JSONDecodeError:
                        # Return raw value if not JSON
                        return cached_value

                logger.debug(f"Cache MISS for key: {cache_key}")

            except Exception as e:
                # Redis unavailable - proceed without cache
                logger.warning(f"Redis cache lookup failed: {e}")
                result = await func(*args, **kwargs)
                return result

            # Execute the function
            result = await func(*args, **kwargs)

            # Store result in cache
            try:
                serialized = _serialize_result(result)
                await redis_client.setex(cache_key, ttl, serialized)
                logger.debug(f"Cached result for key: {cache_key} (TTL: {ttl}s)")

            except (TypeError, ValueError) as e:
                logger.warning(f"Failed to serialize result for caching: {e}")
            except Exception as e:
                logger.warning(f"Failed to store in cache: {e}")

            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, just execute without caching
            # (Redis operations are async, would need different approach)
            logger.debug(f"Sync function {func.__name__} called - cache skipped")
            return func(*args, **kwargs)

        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def get_cache_key(*args, **kwargs) -> str:
    """
    Generate a cache key from function arguments
    """
    import hashlib
    
    # Simple cache key generation
    key_parts = []
    
    # Add positional args
    for arg in args:
        if isinstance(arg, (str, int, float, bool)):
            key_parts.append(str(arg))
        elif hasattr(arg, '__dict__'):
            key_parts.append(str(arg.__class__.__name__))
    
    # Add keyword args
    for k, v in sorted(kwargs.items()):
        if isinstance(v, (str, int, float, bool)):
            key_parts.append(f"{k}:{v}")
    
    # Create hash using SHA256 for security
    cache_string = "|".join(key_parts)
    return hashlib.sha256(cache_string.encode()).hexdigest()[:16]


# Create a default cache instance for backward compatibility
# This allows imports like: from backend.utils.cache import enhanced_cache
class _EnhancedCache:
    """Simple cache wrapper for backward compatibility."""

    def __init__(self):
        self._cache_manager = None

    async def get(self, key: str) -> Any:
        """Get value from cache."""
        if self._cache_manager is None:
            self._cache_manager = await get_cache_manager()
        return await self._cache_manager.get(key)

    async def set(self, key: str, value: Any, ttl: int = 300):
        """Set value in cache."""
        if self._cache_manager is None:
            self._cache_manager = await get_cache_manager()
        await self._cache_manager.set(key, value, ttl)

    async def delete(self, key: str):
        """Delete value from cache."""
        if self._cache_manager is None:
            self._cache_manager = await get_cache_manager()
        await self._cache_manager.delete(key)


enhanced_cache = _EnhancedCache()