"""
Query Result Caching System
Provides intelligent caching for expensive database queries with automatic invalidation.
"""

import hashlib
import json
import pickle
import time
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from datetime import datetime, timedelta
from functools import wraps
import asyncio
from enum import Enum

import redis.asyncio as aioredis
import redis
from sqlalchemy import event, inspect
from sqlalchemy.orm import Session
from sqlalchemy.sql import Select
import pandas as pd
import numpy as np

import logging

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache invalidation strategies."""
    TTL = "ttl"  # Time-based expiration
    LRU = "lru"  # Least recently used
    LFU = "lfu"  # Least frequently used
    WRITE_THROUGH = "write_through"  # Update cache on write
    WRITE_BEHIND = "write_behind"  # Async cache update


class SerializationFormat(Enum):
    """Serialization formats for cached data."""
    JSON = "json"
    PICKLE = "pickle"
    MSGPACK = "msgpack"
    PARQUET = "parquet"  # For DataFrames


class QueryResultCache:
    """
    Intelligent query result caching with automatic invalidation.
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        default_ttl: int = 300,  # 5 minutes
        max_cache_size_mb: int = 100,
        strategy: CacheStrategy = CacheStrategy.TTL,
        key_prefix: str = "query_cache"
    ):
        """
        Initialize query result cache.
        
        Args:
            redis_url: Redis connection URL
            default_ttl: Default TTL in seconds
            max_cache_size_mb: Maximum cache size in MB
            strategy: Cache invalidation strategy
            key_prefix: Prefix for cache keys
        """
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.max_cache_size_mb = max_cache_size_mb
        self.strategy = strategy
        self.key_prefix = key_prefix
        
        self._redis_client: Optional[aioredis.Redis] = None
        self._sync_redis_client: Optional[redis.Redis] = None
        
        # Cache statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'errors': 0,
            'total_size_bytes': 0
        }
        
        # Invalidation rules
        self._invalidation_rules: Dict[str, List[Callable]] = {}
        
        # Query patterns for intelligent caching
        self._query_patterns: Dict[str, Dict[str, Any]] = {}
    
    async def initialize(self) -> None:
        """Initialize Redis connection."""
        try:
            self._redis_client = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=False,  # We'll handle encoding
                socket_keepalive=True,
                socket_connect_timeout=5,
                max_connections=50
            )
            
            self._sync_redis_client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=False,
                socket_keepalive=True,
                socket_connect_timeout=5
            )
            
            await self._redis_client.ping()
            logger.info("Query cache initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize query cache: {e}")
            raise
    
    def _generate_cache_key(
        self,
        query: Union[str, Select],
        params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a unique cache key for a query."""
        # Convert SQLAlchemy query to string if needed
        if hasattr(query, '__class__') and query.__class__.__name__ == 'Select':
            query_str = str(query.compile(compile_kwargs={"literal_binds": True}))
        else:
            query_str = str(query)
        
        # Include parameters in key
        if params:
            param_str = json.dumps(params, sort_keys=True)
            query_str = f"{query_str}:{param_str}"
        
        # Generate hash for key
        key_hash = hashlib.sha256(query_str.encode()).hexdigest()[:16]
        
        return f"{self.key_prefix}:{key_hash}"
    
    async def get(
        self,
        query: Union[str, Select],
        params: Optional[Dict[str, Any]] = None,
        deserializer: Optional[Callable] = None
    ) -> Optional[Any]:
        """
        Get cached query result.
        
        Args:
            query: SQL query or SQLAlchemy query object
            params: Query parameters
            deserializer: Custom deserializer function
        
        Returns:
            Cached result or None if not found
        """
        if not self._redis_client:
            await self.initialize()
        
        key = self._generate_cache_key(query, params)
        
        try:
            # Get from cache
            cached_data = await self._redis_client.get(key)
            
            if cached_data:
                self._stats['hits'] += 1
                
                # Update access time for LRU
                if self.strategy == CacheStrategy.LRU:
                    await self._redis_client.zadd(
                        f"{self.key_prefix}:lru",
                        {key: time.time()}
                    )
                
                # Update frequency for LFU
                elif self.strategy == CacheStrategy.LFU:
                    await self._redis_client.zincrby(
                        f"{self.key_prefix}:lfu",
                        1,
                        key
                    )
                
                # Deserialize
                if deserializer:
                    return deserializer(cached_data)
                else:
                    return self._deserialize(cached_data)
            else:
                self._stats['misses'] += 1
                return None
                
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self._stats['errors'] += 1
            return None
    
    async def set(
        self,
        query: Union[str, Select],
        result: Any,
        params: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
        serializer: Optional[Callable] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """
        Cache query result.
        
        Args:
            query: SQL query or SQLAlchemy query object
            result: Query result to cache
            params: Query parameters
            ttl: Time to live in seconds
            serializer: Custom serializer function
            tags: Tags for cache invalidation
        
        Returns:
            True if cached successfully
        """
        if not self._redis_client:
            await self.initialize()
        
        key = self._generate_cache_key(query, params)
        ttl = ttl or self.default_ttl
        
        try:
            # Serialize data
            if serializer:
                serialized_data = serializer(result)
            else:
                serialized_data = self._serialize(result)
            
            # Check size limit
            data_size = len(serialized_data)
            if data_size > self.max_cache_size_mb * 1024 * 1024:
                logger.warning(f"Data too large to cache: {data_size} bytes")
                return False
            
            # Store in cache
            await self._redis_client.setex(key, ttl, serialized_data)
            
            # Update metadata
            if self.strategy == CacheStrategy.LRU:
                await self._redis_client.zadd(
                    f"{self.key_prefix}:lru",
                    {key: time.time()}
                )
            elif self.strategy == CacheStrategy.LFU:
                await self._redis_client.zadd(
                    f"{self.key_prefix}:lfu",
                    {key: 1}
                )
            
            # Store tags for invalidation
            if tags:
                for tag in tags:
                    await self._redis_client.sadd(f"{self.key_prefix}:tag:{tag}", key)
                    await self._redis_client.expire(f"{self.key_prefix}:tag:{tag}", ttl)
            
            # Update stats
            self._stats['total_size_bytes'] += data_size
            
            # Enforce size limit
            await self._enforce_size_limit()
            
            return True
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            self._stats['errors'] += 1
            return False
    
    async def invalidate(
        self,
        query: Optional[Union[str, Select]] = None,
        params: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        pattern: Optional[str] = None
    ) -> int:
        """
        Invalidate cached results.
        
        Args:
            query: Specific query to invalidate
            params: Query parameters
            tags: Invalidate all queries with these tags
            pattern: Pattern to match keys
        
        Returns:
            Number of keys invalidated
        """
        if not self._redis_client:
            await self.initialize()
        
        invalidated = 0
        
        try:
            if query:
                # Invalidate specific query
                key = self._generate_cache_key(query, params)
                result = await self._redis_client.delete(key)
                invalidated += result
            
            if tags:
                # Invalidate by tags
                for tag in tags:
                    tag_key = f"{self.key_prefix}:tag:{tag}"
                    keys = await self._redis_client.smembers(tag_key)
                    if keys:
                        result = await self._redis_client.delete(*keys)
                        invalidated += result
                    await self._redis_client.delete(tag_key)
            
            if pattern:
                # Invalidate by pattern
                keys = await self._redis_client.keys(f"{self.key_prefix}:{pattern}")
                if keys:
                    result = await self._redis_client.delete(*keys)
                    invalidated += result
            
            logger.info(f"Invalidated {invalidated} cache entries")
            return invalidated
            
        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")
            return 0
    
    async def _enforce_size_limit(self) -> None:
        """Enforce maximum cache size."""
        if self.strategy == CacheStrategy.LRU:
            # Remove least recently used
            while self._stats['total_size_bytes'] > self.max_cache_size_mb * 1024 * 1024:
                oldest = await self._redis_client.zrange(
                    f"{self.key_prefix}:lru",
                    0, 0
                )
                if oldest:
                    key = oldest[0]
                    data = await self._redis_client.get(key)
                    if data:
                        self._stats['total_size_bytes'] -= len(data)
                    await self._redis_client.delete(key)
                    await self._redis_client.zrem(f"{self.key_prefix}:lru", key)
                    self._stats['evictions'] += 1
                else:
                    break
        
        elif self.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            while self._stats['total_size_bytes'] > self.max_cache_size_mb * 1024 * 1024:
                least_used = await self._redis_client.zrange(
                    f"{self.key_prefix}:lfu",
                    0, 0
                )
                if least_used:
                    key = least_used[0]
                    data = await self._redis_client.get(key)
                    if data:
                        self._stats['total_size_bytes'] -= len(data)
                    await self._redis_client.delete(key)
                    await self._redis_client.zrem(f"{self.key_prefix}:lfu", key)
                    self._stats['evictions'] += 1
                else:
                    break
    
    def _serialize(self, data: Any) -> bytes:
        """Serialize data for caching."""
        if isinstance(data, pd.DataFrame):
            # Use parquet for DataFrames
            return data.to_parquet()
        elif isinstance(data, (list, dict)):
            # Use JSON for simple structures
            return json.dumps(data, default=str).encode()
        else:
            # Add trusted marker for pickle serialization
            trusted_marker = b'CACHE_V1_TRUSTED'
            pickled_data = pickle.dumps(data)
            return trusted_marker + pickled_data
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize cached data with security checks."""
        try:
            # Try JSON first (safest)
            return json.loads(data.decode())
        except:
            try:
                # Try msgpack (safer than pickle)
                import msgpack
                return msgpack.unpackb(data, raw=False, strict_map_key=False)
            except:
                try:
                    # Try parquet for DataFrames
                    import io
                    return pd.read_parquet(io.BytesIO(data))
                except:
                    # Only use pickle for trusted data with verification
                    if self._is_trusted_source(data):
                        try:
                            # Use restricted unpickler for safety
                            import io
                            import pickle
                            
                            # Remove trusted marker before unpickling
                            actual_data = data[16:] if len(data) > 16 else data
                            
                            class RestrictedUnpickler(pickle.Unpickler):
                                def find_class(self, module, name):
                                    # Only allow safe modules
                                    ALLOWED_MODULES = {
                                        'pandas', 'numpy', 'datetime', 
                                        'collections', 'builtins'
                                    }
                                    if module.split('.')[0] not in ALLOWED_MODULES:
                                        raise pickle.UnpicklingError(f"Unsafe module: {module}")
                                    return super().find_class(module, name)
                            
                            return RestrictedUnpickler(io.BytesIO(actual_data)).load()
                        except Exception as e:
                            logger.error(f"Failed to deserialize with restricted pickle: {e}")
                            return None
                    else:
                        logger.warning("Untrusted data source, skipping pickle deserialization")
                        return None
    
    def _is_trusted_source(self, data: bytes) -> bool:
        """Check if data is from a trusted source."""
        # Check for a trusted signature/marker at the beginning of data
        # In production, this would verify a cryptographic signature
        try:
            # Look for our internal marker (first 16 bytes)
            if len(data) > 16:
                marker = data[:16]
                # Check if it matches our internal cache marker
                expected_marker = b'CACHE_V1_TRUSTED'
                return marker == expected_marker
        except:
            pass
        return False
    
    def cache_query(
        self,
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None,
        key_params: Optional[List[str]] = None
    ):
        """
        Decorator for caching function results.
        
        Args:
            ttl: Time to live in seconds
            tags: Tags for cache invalidation
            key_params: Parameters to include in cache key
        
        Usage:
            @cache_query(ttl=600, tags=['stocks'])
            def get_stock_data(symbol: str):
                return expensive_query(symbol)
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Generate cache key
                if key_params:
                    params = {k: kwargs.get(k) for k in key_params}
                else:
                    params = kwargs
                
                cache_key = f"{func.__name__}:{json.dumps(params, sort_keys=True)}"
                
                # Try cache first
                cached = await self.get(cache_key, params)
                if cached is not None:
                    return cached
                
                # Execute function
                result = await func(*args, **kwargs)
                
                # Cache result
                await self.set(cache_key, result, params, ttl, tags=tags)
                
                return result
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # For sync functions, use sync Redis client
                if key_params:
                    params = {k: kwargs.get(k) for k in key_params}
                else:
                    params = kwargs
                
                cache_key = f"{func.__name__}:{json.dumps(params, sort_keys=True)}"
                key = self._generate_cache_key(cache_key, params)
                
                # Try cache first
                if self._sync_redis_client:
                    cached = self._sync_redis_client.get(key)
                    if cached:
                        self._stats['hits'] += 1
                        return self._deserialize(cached)
                
                self._stats['misses'] += 1
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Cache result
                if self._sync_redis_client:
                    serialized = self._serialize(result)
                    self._sync_redis_client.setex(key, ttl or self.default_ttl, serialized)
                
                return result
            
            # Return appropriate wrapper
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self._stats.copy()
        
        # Calculate hit rate
        total_requests = stats['hits'] + stats['misses']
        if total_requests > 0:
            stats['hit_rate'] = (stats['hits'] / total_requests) * 100
        else:
            stats['hit_rate'] = 0
        
        # Convert size to MB
        stats['total_size_mb'] = stats['total_size_bytes'] / (1024 * 1024)
        
        return stats
    
    async def clear(self) -> int:
        """Clear all cached data."""
        if not self._redis_client:
            await self.initialize()
        
        keys = await self._redis_client.keys(f"{self.key_prefix}:*")
        if keys:
            result = await self._redis_client.delete(*keys)
            self._stats = {
                'hits': 0,
                'misses': 0,
                'evictions': 0,
                'errors': 0,
                'total_size_bytes': 0
            }
            return result
        return 0
    
    async def close(self) -> None:
        """Close Redis connections."""
        if self._redis_client:
            await self._redis_client.close()
        if self._sync_redis_client:
            self._sync_redis_client.close()


class SmartQueryCache(QueryResultCache):
    """
    Enhanced query cache with intelligent prefetching and pattern learning.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._access_patterns: Dict[str, List[datetime]] = {}
        self._prefetch_queue: asyncio.Queue = asyncio.Queue()
    
    async def learn_access_pattern(
        self,
        query: str,
        access_time: Optional[datetime] = None
    ) -> None:
        """Learn query access patterns for intelligent prefetching."""
        access_time = access_time or datetime.now()
        
        if query not in self._access_patterns:
            self._access_patterns[query] = []
        
        self._access_patterns[query].append(access_time)
        
        # Keep only last 100 accesses
        if len(self._access_patterns[query]) > 100:
            self._access_patterns[query] = self._access_patterns[query][-100:]
        
        # Analyze pattern
        await self._analyze_pattern(query)
    
    async def _analyze_pattern(self, query: str) -> None:
        """Analyze access pattern and schedule prefetching if needed."""
        accesses = self._access_patterns.get(query, [])
        
        if len(accesses) < 10:
            return
        
        # Calculate average interval
        intervals = []
        for i in range(1, len(accesses)):
            interval = (accesses[i] - accesses[i-1]).total_seconds()
            intervals.append(interval)
        
        avg_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        # If access pattern is regular (low std deviation)
        if std_interval < avg_interval * 0.3:
            # Schedule prefetch
            next_access = accesses[-1] + timedelta(seconds=avg_interval)
            await self._schedule_prefetch(query, next_access)
    
    async def _schedule_prefetch(
        self,
        query: str,
        scheduled_time: datetime
    ) -> None:
        """Schedule query prefetching."""
        wait_time = (scheduled_time - datetime.now()).total_seconds()
        
        if wait_time > 0:
            await asyncio.sleep(wait_time)
            await self._prefetch_queue.put(query)
    
    async def prefetch_worker(self) -> None:
        """Worker to process prefetch queue."""
        while True:
            try:
                query = await self._prefetch_queue.get()
                # Execute query and cache result
                # This would need actual query execution logic
                logger.info(f"Prefetching query: {query}")
            except Exception as e:
                logger.error(f"Prefetch error: {e}")


# Global cache instance
query_cache = QueryResultCache()
smart_cache = SmartQueryCache()