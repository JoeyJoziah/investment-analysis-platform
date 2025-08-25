"""
Comprehensive Multi-Layer Caching System for Investment Analysis Platform

This module implements a 3-layer caching architecture:
- L1: In-Memory Cache (fastest, smallest capacity)
- L2: Redis Cache (medium speed, medium capacity) 
- L3: Database Cache (slowest, largest capacity)

Designed to optimize API costs and stay under $50/month budget.
"""

import asyncio
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from threading import RLock
from functools import wraps
import hashlib
import pickle
import gzip

import redis.asyncio as redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.settings import settings
from backend.utils.cache import get_redis
from backend.config.database import get_async_db_session

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for cache behavior"""
    l1_ttl: int = 300  # 5 minutes for L1 cache
    l2_ttl: int = 3600  # 1 hour for L2 cache  
    l3_ttl: int = 86400  # 24 hours for L3 cache
    l1_max_size: int = 1000  # Max items in L1 cache
    compression_threshold: int = 1024  # Compress data > 1KB
    enable_warming: bool = True
    warming_batch_size: int = 50
    invalidation_cascading: bool = True


@dataclass
class CacheMetrics:
    """Cache performance metrics"""
    l1_hits: int = 0
    l1_misses: int = 0
    l2_hits: int = 0
    l2_misses: int = 0
    l3_hits: int = 0
    l3_misses: int = 0
    api_calls_saved: int = 0
    total_requests: int = 0
    cache_storage_bytes: int = 0
    
    @property
    def hit_ratio(self) -> float:
        total = self.total_requests
        if total == 0:
            return 0.0
        hits = self.l1_hits + self.l2_hits + self.l3_hits
        return hits / total
    
    @property
    def cost_savings(self) -> float:
        # Estimate cost savings based on API calls avoided
        # Alpha Vantage: $50/month for premium, we're using free tier
        # Estimate $0.10 per API call if we had to pay
        return self.api_calls_saved * 0.10


class LRUCache:
    """Thread-safe LRU cache for L1 layer"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
        self.lock = RLock()
        self.access_times: Dict[str, float] = {}
    
    def get(self, key: str) -> Tuple[Optional[Any], bool]:
        """Get item from cache, returns (value, hit)"""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                self.access_times[key] = time.time()
                return value, True
            return None, False
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set item in cache"""
        with self.lock:
            # Remove oldest items if at capacity
            while len(self.cache) >= self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                self.access_times.pop(oldest_key, None)
            
            # Add new item
            self.cache[key] = {
                'value': value,
                'expires_at': time.time() + ttl if ttl else None
            }
            self.access_times[key] = time.time()
    
    def delete(self, key: str) -> bool:
        """Delete item from cache"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                self.access_times.pop(key, None)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all items"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
    
    def cleanup_expired(self) -> int:
        """Remove expired items, returns count removed"""
        current_time = time.time()
        expired_keys = []
        
        with self.lock:
            for key, data in self.cache.items():
                if data.get('expires_at') and data['expires_at'] < current_time:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache[key]
                self.access_times.pop(key, None)
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'utilization': len(self.cache) / self.max_size
            }


class ComprehensiveCacheManager:
    """
    Multi-layer cache manager with intelligent caching strategies
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.l1_cache = LRUCache(self.config.l1_max_size)
        self.redis_client: Optional[redis.Redis] = None
        self.metrics = CacheMetrics()
        self._warming_tasks: Dict[str, asyncio.Task] = {}
        self._last_cleanup = time.time()
        
        # Cache key prefixes for different data types
        self.key_prefixes = {
            'api_response': 'api:resp',
            'db_query': 'db:query',
            'computation': 'comp',
            'market_data': 'market',
            'user_data': 'user',
            'analysis': 'analysis'
        }
    
    async def initialize(self):
        """Initialize cache connections"""
        try:
            self.redis_client = await get_redis()
            logger.info("ComprehensiveCacheManager initialized successfully")
            
            # Start periodic cleanup task
            asyncio.create_task(self._periodic_cleanup())
            
        except Exception as e:
            logger.error(f"Failed to initialize cache manager: {e}")
            raise
    
    def _make_key(self, data_type: str, identifier: str, params: Optional[Dict] = None) -> str:
        """Create standardized cache key"""
        prefix = self.key_prefixes.get(data_type, 'misc')
        
        if params:
            # Create deterministic hash of parameters
            param_str = json.dumps(params, sort_keys=True)
            param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
            return f"{prefix}:{identifier}:{param_hash}"
        
        return f"{prefix}:{identifier}"
    
    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data with optional compression"""
        serialized = pickle.dumps(data)
        
        if len(serialized) > self.config.compression_threshold:
            serialized = gzip.compress(serialized)
            return b'gzip:' + serialized
        
        return serialized
    
    def _deserialize_data(self, data: bytes) -> Any:
        """Deserialize data with compression support"""
        if data.startswith(b'gzip:'):
            data = gzip.decompress(data[5:])
        
        return pickle.loads(data)
    
    async def get(
        self,
        data_type: str,
        identifier: str,
        params: Optional[Dict] = None,
        fallback_func: Optional[Callable] = None
    ) -> Tuple[Optional[Any], str]:
        """
        Get data from cache with fallback to computation
        Returns (data, source) where source is 'l1', 'l2', 'l3', or 'computed'
        """
        self.metrics.total_requests += 1
        cache_key = self._make_key(data_type, identifier, params)
        
        # L1 Cache Check (Memory)
        l1_data, l1_hit = self.l1_cache.get(cache_key)
        if l1_hit and l1_data and not self._is_expired(l1_data):
            self.metrics.l1_hits += 1
            logger.debug(f"L1 cache hit: {cache_key}")
            return l1_data['value'], 'l1'
        
        if l1_hit:
            self.metrics.l1_misses += 1
        
        # L2 Cache Check (Redis)
        if self.redis_client:
            try:
                l2_data = await self.redis_client.get(cache_key)
                if l2_data:
                    self.metrics.l2_hits += 1
                    data = self._deserialize_data(l2_data)
                    
                    # Populate L1 cache
                    self.l1_cache.set(cache_key, data, self.config.l1_ttl)
                    
                    logger.debug(f"L2 cache hit: {cache_key}")
                    return data, 'l2'
                else:
                    self.metrics.l2_misses += 1
            except Exception as e:
                logger.warning(f"L2 cache error for {cache_key}: {e}")
                self.metrics.l2_misses += 1
        
        # L3 Cache Check (Database)
        l3_data = await self._get_from_database(cache_key)
        if l3_data:
            self.metrics.l3_hits += 1
            
            # Populate L1 and L2 caches
            self.l1_cache.set(cache_key, l3_data, self.config.l1_ttl)
            if self.redis_client:
                await self._set_redis(cache_key, l3_data, self.config.l2_ttl)
            
            logger.debug(f"L3 cache hit: {cache_key}")
            return l3_data, 'l3'
        else:
            self.metrics.l3_misses += 1
        
        # Fallback to computation
        if fallback_func:
            try:
                computed_data = await fallback_func() if asyncio.iscoroutinefunction(fallback_func) else fallback_func()
                if computed_data is not None:
                    # Store in all cache layers
                    await self.set(data_type, identifier, computed_data, params)
                    self.metrics.api_calls_saved += 1
                    logger.debug(f"Data computed and cached: {cache_key}")
                    return computed_data, 'computed'
            except Exception as e:
                logger.error(f"Fallback function failed for {cache_key}: {e}")
        
        return None, 'miss'
    
    async def set(
        self,
        data_type: str,
        identifier: str,
        data: Any,
        params: Optional[Dict] = None,
        custom_ttl: Optional[Dict[str, int]] = None
    ):
        """Set data in all cache layers"""
        cache_key = self._make_key(data_type, identifier, params)
        
        # Use custom TTL or defaults
        ttls = custom_ttl or {
            'l1': self.config.l1_ttl,
            'l2': self.config.l2_ttl,
            'l3': self.config.l3_ttl
        }
        
        # L1 Cache (Memory)
        self.l1_cache.set(cache_key, data, ttls['l1'])
        
        # L2 Cache (Redis)
        if self.redis_client:
            await self._set_redis(cache_key, data, ttls['l2'])
        
        # L3 Cache (Database)
        await self._set_database(cache_key, data, ttls['l3'])
        
        # Update metrics
        data_size = len(self._serialize_data(data))
        self.metrics.cache_storage_bytes += data_size
        
        logger.debug(f"Data cached across all layers: {cache_key}")
    
    async def delete(self, data_type: str, identifier: str, params: Optional[Dict] = None):
        """Delete from all cache layers"""
        cache_key = self._make_key(data_type, identifier, params)
        
        # L1 Cache
        self.l1_cache.delete(cache_key)
        
        # L2 Cache
        if self.redis_client:
            await self.redis_client.delete(cache_key)
        
        # L3 Cache
        await self._delete_from_database(cache_key)
        
        logger.debug(f"Cache entry deleted from all layers: {cache_key}")
    
    async def invalidate_pattern(self, pattern: str):
        """Invalidate all keys matching pattern"""
        if self.redis_client:
            # Get matching keys from Redis
            keys = await self.redis_client.keys(pattern)
            if keys:
                # Delete from L2
                await self.redis_client.delete(*keys)
                
                # Delete from L1
                for key in keys:
                    self.l1_cache.delete(key)
                
                # Delete from L3
                for key in keys:
                    await self._delete_from_database(key)
                
                logger.info(f"Invalidated {len(keys)} cache entries matching pattern: {pattern}")
    
    async def warm_cache(
        self,
        data_specs: List[Dict[str, Any]],
        priority: int = 1
    ):
        """Warm cache with frequently accessed data"""
        if not self.config.enable_warming:
            return
        
        task_key = f"warming_{priority}_{hash(str(data_specs))}"
        
        if task_key in self._warming_tasks:
            return  # Already warming
        
        async def _warm_data():
            try:
                for spec in data_specs:
                    data_type = spec['data_type']
                    identifier = spec['identifier']
                    params = spec.get('params')
                    fallback_func = spec.get('fallback_func')
                    
                    # Check if data exists
                    cache_key = self._make_key(data_type, identifier, params)
                    if not await self._exists_in_any_layer(cache_key):
                        if fallback_func:
                            try:
                                data = await fallback_func() if asyncio.iscoroutinefunction(fallback_func) else fallback_func()
                                if data is not None:
                                    await self.set(data_type, identifier, data, params)
                                    logger.debug(f"Cache warmed: {cache_key}")
                            except Exception as e:
                                logger.warning(f"Cache warming failed for {cache_key}: {e}")
                    
                    # Rate limiting for cache warming
                    await asyncio.sleep(0.1)
                
                logger.info(f"Cache warming completed for {len(data_specs)} items")
                
            finally:
                self._warming_tasks.pop(task_key, None)
        
        self._warming_tasks[task_key] = asyncio.create_task(_warm_data())
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive cache metrics"""
        l1_stats = self.l1_cache.get_stats()
        
        redis_stats = {}
        if self.redis_client:
            try:
                info = await self.redis_client.info('memory')
                redis_stats = {
                    'used_memory': info.get('used_memory', 0),
                    'used_memory_human': info.get('used_memory_human', '0B'),
                    'keyspace_hits': info.get('keyspace_hits', 0),
                    'keyspace_misses': info.get('keyspace_misses', 0)
                }
            except Exception as e:
                logger.warning(f"Could not get Redis stats: {e}")
        
        return {
            'cache_metrics': {
                'l1_hits': self.metrics.l1_hits,
                'l1_misses': self.metrics.l1_misses,
                'l2_hits': self.metrics.l2_hits,
                'l2_misses': self.metrics.l2_misses,
                'l3_hits': self.metrics.l3_hits,
                'l3_misses': self.metrics.l3_misses,
                'total_requests': self.metrics.total_requests,
                'hit_ratio': self.metrics.hit_ratio,
                'api_calls_saved': self.metrics.api_calls_saved,
                'estimated_cost_savings': self.metrics.cost_savings
            },
            'l1_cache_stats': l1_stats,
            'l2_cache_stats': redis_stats,
            'active_warming_tasks': len(self._warming_tasks),
            'storage_bytes': self.metrics.cache_storage_bytes
        }
    
    # Private helper methods
    
    def _is_expired(self, cached_data: Dict) -> bool:
        """Check if cached data is expired"""
        expires_at = cached_data.get('expires_at')
        return expires_at is not None and expires_at < time.time()
    
    async def _set_redis(self, key: str, data: Any, ttl: int):
        """Set data in Redis with error handling"""
        try:
            serialized_data = self._serialize_data(data)
            await self.redis_client.setex(key, ttl, serialized_data)
        except Exception as e:
            logger.warning(f"Failed to set Redis cache for {key}: {e}")
    
    async def _get_from_database(self, cache_key: str) -> Optional[Any]:
        """Get cached data from database"""
        try:
            async with get_async_db_session() as db:
                result = await db.execute(
                    text("""
                        SELECT data, expires_at 
                        FROM cache_storage 
                        WHERE cache_key = :key 
                        AND (expires_at IS NULL OR expires_at > NOW())
                    """),
                    {"key": cache_key}
                )
                row = result.fetchone()
                if row:
                    return self._deserialize_data(row[0])
        except Exception as e:
            logger.debug(f"L3 cache miss for {cache_key}: {e}")
        
        return None
    
    async def _set_database(self, cache_key: str, data: Any, ttl: int):
        """Set data in database cache"""
        try:
            serialized_data = self._serialize_data(data)
            expires_at = datetime.utcnow() + timedelta(seconds=ttl)
            
            async with get_async_db_session() as db:
                await db.execute(
                    text("""
                        INSERT INTO cache_storage (cache_key, data, expires_at, created_at)
                        VALUES (:key, :data, :expires_at, NOW())
                        ON CONFLICT (cache_key) 
                        DO UPDATE SET 
                            data = EXCLUDED.data,
                            expires_at = EXCLUDED.expires_at,
                            updated_at = NOW()
                    """),
                    {
                        "key": cache_key,
                        "data": serialized_data,
                        "expires_at": expires_at
                    }
                )
                await db.commit()
        except Exception as e:
            logger.warning(f"Failed to set database cache for {cache_key}: {e}")
    
    async def _delete_from_database(self, cache_key: str):
        """Delete cached data from database"""
        try:
            async with get_async_db_session() as db:
                await db.execute(
                    text("DELETE FROM cache_storage WHERE cache_key = :key"),
                    {"key": cache_key}
                )
                await db.commit()
        except Exception as e:
            logger.warning(f"Failed to delete from database cache {cache_key}: {e}")
    
    async def _exists_in_any_layer(self, cache_key: str) -> bool:
        """Check if key exists in any cache layer"""
        # L1 check
        _, l1_hit = self.l1_cache.get(cache_key)
        if l1_hit:
            return True
        
        # L2 check
        if self.redis_client:
            try:
                exists = await self.redis_client.exists(cache_key)
                if exists:
                    return True
            except Exception:
                pass
        
        # L3 check
        data = await self._get_from_database(cache_key)
        return data is not None
    
    async def _periodic_cleanup(self):
        """Periodic cleanup task"""
        while True:
            try:
                # Clean expired L1 entries
                expired_count = self.l1_cache.cleanup_expired()
                if expired_count > 0:
                    logger.debug(f"Cleaned {expired_count} expired L1 cache entries")
                
                # Clean expired database entries (every hour)
                current_time = time.time()
                if current_time - self._last_cleanup > 3600:
                    await self._cleanup_database()
                    self._last_cleanup = current_time
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
                await asyncio.sleep(300)
    
    async def _cleanup_database(self):
        """Clean expired entries from database cache"""
        try:
            async with get_async_db_session() as db:
                result = await db.execute(
                    text("DELETE FROM cache_storage WHERE expires_at < NOW()")
                )
                await db.commit()
                logger.info(f"Cleaned {result.rowcount} expired database cache entries")
        except Exception as e:
            logger.warning(f"Failed to cleanup database cache: {e}")


# Global cache manager instance
_cache_manager: Optional[ComprehensiveCacheManager] = None


async def get_cache_manager() -> ComprehensiveCacheManager:
    """Get global cache manager instance"""
    global _cache_manager
    
    if _cache_manager is None:
        _cache_manager = ComprehensiveCacheManager()
        await _cache_manager.initialize()
    
    return _cache_manager


def cached(
    data_type: str,
    ttl_config: Optional[Dict[str, int]] = None,
    cache_key_func: Optional[Callable] = None
):
    """
    Decorator for caching function results
    
    Args:
        data_type: Type of data being cached
        ttl_config: Custom TTL configuration for each layer
        cache_key_func: Function to generate custom cache key
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_manager = await get_cache_manager()
            
            # Generate cache key
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = ":".join(key_parts)
            
            # Try to get from cache
            result, source = await cache_manager.get(
                data_type=data_type,
                identifier=cache_key,
                fallback_func=lambda: func(*args, **kwargs)
            )
            
            return result
        
        return wrapper
    return decorator