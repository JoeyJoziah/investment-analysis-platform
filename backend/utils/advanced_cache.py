"""Advanced caching strategies for optimal performance"""

import asyncio
import hashlib
import json
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Union

import redis.asyncio as redis
from pydantic import BaseModel

from backend.config import settings
from backend.utils.monitoring import metrics

# Cache TTL configurations
CACHE_TTLS = {
    # Historical data - cache for days
    "historical_prices": 86400 * 7,  # 7 days
    "fundamentals": 86400 * 1,  # 1 day
    "financial_statements": 86400 * 30,  # 30 days
    
    # Real-time data - cache for minutes
    "live_quotes": 60,  # 1 minute
    "market_status": 300,  # 5 minutes
    "option_chains": 900,  # 15 minutes
    
    # Analysis results - cache for hours
    "technical_analysis": 3600,  # 1 hour
    "sentiment_analysis": 1800,  # 30 minutes
    "ml_predictions": 7200,  # 2 hours
    "recommendations": 1800,  # 30 minutes
    
    # Company info - cache for long periods
    "company_profile": 86400 * 7,  # 7 days
    "stock_list": 86400 * 1,  # 1 day
    "exchange_info": 86400 * 7,  # 7 days
    
    # User data - cache briefly
    "user_portfolio": 300,  # 5 minutes
    "user_preferences": 1800,  # 30 minutes
    "watchlists": 600,  # 10 minutes
}


class CacheConfig(BaseModel):
    """Cache configuration"""
    ttl: int
    compress: bool = False
    serialize_json: bool = True
    cache_null: bool = False
    tags: List[str] = []


class CacheEntry(BaseModel):
    """Cache entry metadata"""
    key: str
    value: Any
    created_at: datetime
    ttl: int
    hits: int = 0
    tags: List[str] = []
    size_bytes: int = 0


class MultiLevelCache:
    """Multi-level caching system with L1 (memory) and L2 (Redis)"""
    
    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        l1_max_size: int = 5000,
        l1_max_memory_mb: int = 256
    ):
        self.redis_client = redis_client
        self.l1_cache: Dict[str, CacheEntry] = {}
        self.l1_max_size = l1_max_size
        self.l1_max_memory_bytes = l1_max_memory_mb * 1024 * 1024
        self.l1_current_memory = 0
        self._lock = asyncio.Lock()
        
    async def get(
        self,
        key: str,
        default: Any = None,
        cache_type: str = "default"
    ) -> Any:
        """
        Get value from cache (L1 first, then L2)
        
        Args:
            key: Cache key
            default: Default value if not found
            cache_type: Type of cache for metrics
            
        Returns:
            Cached value or default
        """
        # Try L1 cache first
        l1_value = await self._get_l1(key)
        if l1_value is not None:
            metrics.cache_hits_total.labels(
                cache_type=cache_type,
                operation="get"
            ).inc()
            return l1_value
            
        # Try L2 cache (Redis)
        if self.redis_client:
            l2_value = await self._get_l2(key)
            if l2_value is not None:
                # Store in L1 for faster access
                await self._set_l1(key, l2_value, ttl=CACHE_TTLS.get(cache_type, 300))
                metrics.cache_hits_total.labels(
                    cache_type=cache_type,
                    operation="get"
                ).inc()
                return l2_value
                
        # Cache miss
        metrics.cache_misses_total.labels(
            cache_type=cache_type,
            operation="get"
        ).inc()
        return default
        
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        cache_type: str = "default",
        config: Optional[CacheConfig] = None
    ) -> bool:
        """
        Set value in cache (both L1 and L2)
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            cache_type: Type of cache
            config: Cache configuration
            
        Returns:
            Success status
        """
        if config:
            ttl = config.ttl
            
        # Set default TTL
        if ttl is None:
            ttl = CACHE_TTLS.get(cache_type, 300)
            
        # Don't cache null values unless configured
        if value is None and not (config and config.cache_null):
            return False
            
        try:
            # Set in L1 cache
            await self._set_l1(key, value, ttl)
            
            # Set in L2 cache (Redis)
            if self.redis_client:
                await self._set_l2(key, value, ttl, config)
                
            return True
            
        except Exception as e:
            logger.error(f"Cache set failed: {e}")
            return False
            
    async def delete(self, key: str) -> bool:
        """Delete key from both cache levels"""
        try:
            # Delete from L1
            async with self._lock:
                if key in self.l1_cache:
                    entry = self.l1_cache[key]
                    self.l1_current_memory -= entry.size_bytes
                    del self.l1_cache[key]
                    
            # Delete from L2
            if self.redis_client:
                await self.redis_client.delete(key)
                
            return True
            
        except Exception as e:
            logger.error(f"Cache delete failed: {e}")
            return False
            
    async def clear(self, pattern: Optional[str] = None) -> int:
        """Clear cache entries matching pattern"""
        count = 0
        
        try:
            if pattern:
                # Clear matching keys
                if self.redis_client:
                    keys = await self.redis_client.keys(pattern)
                    if keys:
                        await self.redis_client.delete(*keys)
                        count += len(keys)
                        
                # Clear from L1
                async with self._lock:
                    import fnmatch
                    keys_to_delete = [
                        k for k in self.l1_cache.keys()
                        if fnmatch.fnmatch(k, pattern)
                    ]
                    for key in keys_to_delete:
                        entry = self.l1_cache[key]
                        self.l1_current_memory -= entry.size_bytes
                        del self.l1_cache[key]
                        count += 1
            else:
                # Clear all
                if self.redis_client:
                    await self.redis_client.flushdb()
                    
                async with self._lock:
                    count = len(self.l1_cache)
                    self.l1_cache.clear()
                    self.l1_current_memory = 0
                    
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
            
        return count
        
    async def _get_l1(self, key: str) -> Any:
        """Get from L1 cache"""
        async with self._lock:
            if key not in self.l1_cache:
                return None
                
            entry = self.l1_cache[key]
            
            # Check expiry
            if datetime.utcnow() > entry.created_at + timedelta(seconds=entry.ttl):
                # Expired, remove
                self.l1_current_memory -= entry.size_bytes
                del self.l1_cache[key]
                return None
                
            # Update hit count
            entry.hits += 1
            return entry.value
            
    async def _set_l1(self, key: str, value: Any, ttl: int):
        """Set in L1 cache"""
        # Serialize to estimate size
        serialized = json.dumps(value, default=str) if not isinstance(value, str) else value
        size_bytes = len(serialized.encode('utf-8'))
        
        async with self._lock:
            # Check if we need to evict
            await self._evict_l1_if_needed(size_bytes)
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.utcnow(),
                ttl=ttl,
                size_bytes=size_bytes
            )
            
            # Remove old entry if exists
            if key in self.l1_cache:
                old_entry = self.l1_cache[key]
                self.l1_current_memory -= old_entry.size_bytes
                
            # Add new entry
            self.l1_cache[key] = entry
            self.l1_current_memory += size_bytes
            
            # Update metrics
            metrics.cache_size_bytes.labels(cache_type="l1").set(self.l1_current_memory)
            
    async def _get_l2(self, key: str) -> Any:
        """Get from L2 cache (Redis)"""
        try:
            data = await self.redis_client.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.error(f"L2 cache get failed: {e}")
        return None
        
    async def _set_l2(
        self,
        key: str,
        value: Any,
        ttl: int,
        config: Optional[CacheConfig] = None
    ):
        """Set in L2 cache (Redis)"""
        try:
            # Serialize value
            if config and not config.serialize_json:
                data = str(value)
            else:
                data = json.dumps(value, default=str)
                
            # Compress if configured
            if config and config.compress:
                import gzip
                data = gzip.compress(data.encode()).decode('latin1')
                
            # Set with TTL
            await self.redis_client.setex(key, ttl, data)
            
            # Add tags if configured
            if config and config.tags:
                for tag in config.tags:
                    await self.redis_client.sadd(f"cache_tag:{tag}", key)
                    await self.redis_client.expire(f"cache_tag:{tag}", ttl)
                    
        except Exception as e:
            logger.error(f"L2 cache set failed: {e}")
            
    async def _evict_l1_if_needed(self, new_size: int):
        """Evict L1 cache entries if needed"""
        # Check size limit
        if len(self.l1_cache) >= self.l1_max_size:
            await self._evict_l1_lru()
            
        # Check memory limit
        if self.l1_current_memory + new_size >= self.l1_max_memory_bytes:
            await self._evict_l1_by_memory()
            
    async def _evict_l1_lru(self):
        """Evict least recently used entry"""
        if not self.l1_cache:
            return
            
        # Find LRU entry
        lru_key = min(self.l1_cache.keys(), key=lambda k: self.l1_cache[k].hits)
        entry = self.l1_cache[lru_key]
        
        # Remove entry
        self.l1_current_memory -= entry.size_bytes
        del self.l1_cache[lru_key]
        
    async def _evict_l1_by_memory(self):
        """Evict entries to free memory"""
        # Sort by hits (ascending) and size (descending)
        sorted_keys = sorted(
            self.l1_cache.keys(),
            key=lambda k: (self.l1_cache[k].hits, -self.l1_cache[k].size_bytes)
        )
        
        # Remove 25% of entries
        evict_count = max(1, len(sorted_keys) // 4)
        
        for key in sorted_keys[:evict_count]:
            entry = self.l1_cache[key]
            self.l1_current_memory -= entry.size_bytes
            del self.l1_cache[key]


class CacheManager:
    """High-level cache management"""
    
    def __init__(self, cache: MultiLevelCache):
        self.cache = cache
        self.warming_tasks: Dict[str, asyncio.Task] = {}
        
    def cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        # Create deterministic key
        key_parts = [str(arg) for arg in args]
        key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
        key_string = ":".join(key_parts)
        
        # Hash for consistent length
        return hashlib.md5(key_string.encode()).hexdigest()
        
    def cached(
        self,
        cache_type: str = "default",
        ttl: Optional[int] = None,
        key_func: Optional[Callable] = None
    ):
        """Decorator for caching function results"""
        def decorator(func):
            async def async_wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = f"{func.__name__}:{self.cache_key(*args, **kwargs)}"
                    
                # Try to get from cache
                result = await self.cache.get(cache_key, cache_type=cache_type)
                if result is not None:
                    return result
                    
                # Execute function
                result = await func(*args, **kwargs)
                
                # Cache result
                await self.cache.set(
                    cache_key,
                    result,
                    ttl=ttl,
                    cache_type=cache_type
                )
                
                return result
                
            def sync_wrapper(*args, **kwargs):
                # For sync functions
                import asyncio
                return asyncio.run(async_wrapper(*args, **kwargs))
                
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
            
        return decorator
        
    async def warm_cache(
        self,
        warm_func: Callable,
        keys: List[str],
        cache_type: str = "default"
    ):
        """Warm cache with data"""
        async def warm_single(key: str):
            try:
                value = await warm_func(key)
                if value is not None:
                    await self.cache.set(key, value, cache_type=cache_type)
            except Exception as e:
                logger.error(f"Cache warming failed for {key}: {e}")
                
        # Warm in parallel
        tasks = [warm_single(key) for key in keys]
        await asyncio.gather(*tasks, return_exceptions=True)
        
    async def invalidate_by_tags(self, tags: List[str]) -> int:
        """Invalidate cache entries by tags"""
        count = 0
        
        if not self.cache.redis_client:
            return count
            
        try:
            for tag in tags:
                tag_key = f"cache_tag:{tag}"
                keys = await self.cache.redis_client.smembers(tag_key)
                
                if keys:
                    # Delete cached entries
                    await self.cache.redis_client.delete(*keys)
                    count += len(keys)
                    
                    # Delete tag set
                    await self.cache.redis_client.delete(tag_key)
                    
        except Exception as e:
            logger.error(f"Tag invalidation failed: {e}")
            
        return count
        
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            "l1_cache": {
                "entries": len(self.cache.l1_cache),
                "memory_bytes": self.cache.l1_current_memory,
                "max_entries": self.cache.l1_max_size,
                "max_memory_bytes": self.cache.l1_max_memory_bytes
            }
        }
        
        if self.cache.redis_client:
            try:
                info = await self.cache.redis_client.info()
                stats["l2_cache"] = {
                    "memory_bytes": info.get("used_memory", 0),
                    "keys": info.get("db0", {}).get("keys", 0),
                    "hits": info.get("keyspace_hits", 0),
                    "misses": info.get("keyspace_misses", 0)
                }
            except:
                stats["l2_cache"] = {"status": "unavailable"}
                
        return stats


# Initialize cache system
redis_client = None
if hasattr(settings, 'REDIS_URL'):
    redis_client = redis.from_url(settings.REDIS_URL)

cache = MultiLevelCache(redis_client)
cache_manager = CacheManager(cache)

# Import logger after cache is initialized
import logging
logger = logging.getLogger(__name__)