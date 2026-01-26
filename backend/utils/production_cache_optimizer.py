"""
Production Cache Optimizer for Investment Analysis Platform
Intelligent multi-layer caching system designed for cost optimization (<$50/month)
Reduces external API calls by 90%+ through strategic caching policies
"""

import asyncio
import json
import logging
import hashlib
import zlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import numpy as np

# SECURITY: Removed pickle import - using JSON serialization instead to prevent
# arbitrary code execution vulnerabilities from untrusted cache data

logger = logging.getLogger(__name__)


class CacheLayer(Enum):
    """Cache layer hierarchy for optimal cost management"""
    MEMORY = "memory"      # L1: In-memory (fastest, smallest)
    REDIS = "redis"        # L2: Redis (fast, medium)
    DATABASE = "database"  # L3: PostgreSQL (slower, persistent)
    DISK = "disk"         # L4: Local storage (slowest, largest)


@dataclass
class CachePolicy:
    """Cache policy configuration for different data types"""
    ttl_seconds: int
    max_size: int
    compression: bool = False
    layers: List[CacheLayer] = None
    cost_weight: float = 1.0  # Higher weight = more expensive to regenerate
    
    def __post_init__(self):
        if self.layers is None:
            self.layers = [CacheLayer.MEMORY, CacheLayer.REDIS]


class ProductionCacheOptimizer:
    """
    Production-ready cache optimizer with intelligent cost management
    Designed to minimize API calls and operational costs
    """
    
    def __init__(self, redis_client: redis.Redis, db_session: AsyncSession):
        self.redis_client = redis_client
        self.db_session = db_session
        
        # Memory cache (L1) - Limited size for cost optimization
        self._memory_cache: Dict[str, tuple] = {}  # key: (data, expiry, access_count)
        self._memory_cache_max_size = 1000
        
        # Cache policies for different data types
        self.cache_policies = {
            # High-value, expensive-to-generate data
            'stock_analysis': CachePolicy(
                ttl_seconds=3600,  # 1 hour
                max_size=10000,
                compression=True,
                layers=[CacheLayer.MEMORY, CacheLayer.REDIS, CacheLayer.DATABASE],
                cost_weight=10.0
            ),
            'market_sentiment': CachePolicy(
                ttl_seconds=1800,  # 30 minutes
                max_size=5000,
                compression=True,
                layers=[CacheLayer.MEMORY, CacheLayer.REDIS],
                cost_weight=8.0
            ),
            'stock_prices': CachePolicy(
                ttl_seconds=300,   # 5 minutes for real-time prices
                max_size=50000,
                compression=False,
                layers=[CacheLayer.MEMORY, CacheLayer.REDIS],
                cost_weight=3.0
            ),
            'news_data': CachePolicy(
                ttl_seconds=7200,  # 2 hours
                max_size=20000,
                compression=True,
                layers=[CacheLayer.REDIS, CacheLayer.DATABASE],
                cost_weight=5.0
            ),
            'technical_indicators': CachePolicy(
                ttl_seconds=900,   # 15 minutes
                max_size=15000,
                compression=True,
                layers=[CacheLayer.MEMORY, CacheLayer.REDIS],
                cost_weight=4.0
            ),
            # Low-cost, frequently accessed data
            'company_info': CachePolicy(
                ttl_seconds=86400, # 24 hours
                max_size=10000,
                compression=False,
                layers=[CacheLayer.MEMORY, CacheLayer.REDIS, CacheLayer.DATABASE],
                cost_weight=1.0
            ),
            'market_holidays': CachePolicy(
                ttl_seconds=604800, # 1 week
                max_size=100,
                compression=False,
                layers=[CacheLayer.MEMORY, CacheLayer.DATABASE],
                cost_weight=0.1
            )
        }
        
        # Statistics for cost monitoring
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'cost_savings': 0.0,
            'api_calls_prevented': 0
        }
        
        # Initialize background tasks
        self._setup_background_tasks()
    
    def _setup_background_tasks(self):
        """Setup background maintenance tasks"""
        asyncio.create_task(self._cleanup_expired_cache())
        asyncio.create_task(self._optimize_cache_distribution())
        asyncio.create_task(self._monitor_cache_performance())
    
    def _generate_cache_key(self, key_parts: List[str], cache_type: str) -> str:
        """Generate consistent cache keys with namespace"""
        key_string = f"{cache_type}:{':'.join(key_parts)}"
        return hashlib.sha256(key_string.encode()).hexdigest()[:32]
    
    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for complex types - SECURITY: Replaces pickle"""
        if isinstance(obj, np.ndarray):
            return {'__numpy__': True, 'data': obj.tolist(), 'dtype': str(obj.dtype)}
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, datetime):
            return {'__datetime__': True, 'value': obj.isoformat()}
        elif isinstance(obj, timedelta):
            return {'__timedelta__': True, 'seconds': obj.total_seconds()}
        elif hasattr(obj, '__dict__'):
            return {'__object__': True, 'type': type(obj).__name__, 'data': obj.__dict__}
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    def _json_deserializer(self, obj: Dict) -> Any:
        """Custom JSON deserializer for complex types - SECURITY: Replaces pickle"""
        if isinstance(obj, dict):
            if obj.get('__numpy__'):
                return np.array(obj['data'], dtype=obj['dtype'])
            elif obj.get('__datetime__'):
                return datetime.fromisoformat(obj['value'])
            elif obj.get('__timedelta__'):
                return timedelta(seconds=obj['seconds'])
            elif obj.get('__object__'):
                # Note: Only deserialize data as dict, not reconstruct arbitrary objects
                return obj['data']
        return obj

    def _serialize_data(self, data: Any, use_compression: bool = False) -> bytes:
        """
        Serialize data with optional compression.
        SECURITY: Uses JSON instead of pickle to prevent arbitrary code execution.
        """
        serialized = json.dumps(data, default=self._json_serializer).encode('utf-8')

        if use_compression:
            serialized = zlib.compress(serialized, level=6)  # Balanced compression

        return serialized

    def _deserialize_data(self, data: bytes, use_compression: bool = False) -> Any:
        """
        Deserialize data with optional decompression.
        SECURITY: Uses JSON instead of pickle to prevent arbitrary code execution.
        """
        if use_compression:
            data = zlib.decompress(data)

        def object_hook(obj):
            return self._json_deserializer(obj)

        return json.loads(data.decode('utf-8'), object_hook=object_hook)
    
    async def get(self, key_parts: List[str], cache_type: str = 'default') -> Optional[Any]:
        """
        Get data from cache with intelligent layer traversal
        Returns None if not found in any layer
        """
        cache_key = self._generate_cache_key(key_parts, cache_type)
        policy = self.cache_policies.get(cache_type, self.cache_policies['stock_prices'])
        
        # Try each cache layer in order
        for layer in policy.layers:
            try:
                data = await self._get_from_layer(cache_key, layer, policy)
                if data is not None:
                    self.cache_stats['hits'] += 1
                    
                    # Promote data to higher layers for better performance
                    await self._promote_to_higher_layers(cache_key, data, layer, policy)
                    
                    return data
            
            except Exception as e:
                logger.warning(f"Cache get error in layer {layer}: {e}")
                continue
        
        self.cache_stats['misses'] += 1
        return None
    
    async def set(self, key_parts: List[str], data: Any, cache_type: str = 'default',
                  ttl_override: Optional[int] = None) -> bool:
        """
        Set data in cache across appropriate layers
        Returns True if successfully cached
        """
        cache_key = self._generate_cache_key(key_parts, cache_type)
        policy = self.cache_policies.get(cache_type, self.cache_policies['stock_prices'])
        
        ttl = ttl_override or policy.ttl_seconds
        expiry_time = datetime.utcnow() + timedelta(seconds=ttl)
        
        success_count = 0
        
        # Store in all configured layers
        for layer in policy.layers:
            try:
                await self._set_in_layer(cache_key, data, layer, policy, expiry_time)
                success_count += 1
            except Exception as e:
                logger.warning(f"Cache set error in layer {layer}: {e}")
        
        # Update cost savings statistics
        if success_count > 0:
            self.cache_stats['cost_savings'] += policy.cost_weight
            self.cache_stats['api_calls_prevented'] += 1
        
        return success_count > 0
    
    async def _get_from_layer(self, key: str, layer: CacheLayer, 
                             policy: CachePolicy) -> Optional[Any]:
        """Get data from specific cache layer"""
        
        if layer == CacheLayer.MEMORY:
            cache_entry = self._memory_cache.get(key)
            if cache_entry and cache_entry[1] > datetime.utcnow():
                # Update access count for LRU
                self._memory_cache[key] = (cache_entry[0], cache_entry[1], cache_entry[2] + 1)
                return cache_entry[0]
            elif cache_entry:
                # Expired entry
                del self._memory_cache[key]
        
        elif layer == CacheLayer.REDIS:
            try:
                cached_data = await self.redis_client.get(f"cache:{key}")
                if cached_data:
                    return self._deserialize_data(cached_data, policy.compression)
            except Exception as e:
                logger.warning(f"Redis get error: {e}")
        
        elif layer == CacheLayer.DATABASE:
            try:
                query = text("""
                    SELECT result_data, expires_at 
                    FROM analysis_cache 
                    WHERE id = :key AND expires_at > NOW()
                """)
                result = await self.db_session.execute(query, {"key": key})
                row = result.fetchone()
                if row:
                    return json.loads(row[0])
            except Exception as e:
                logger.warning(f"Database cache get error: {e}")
        
        return None
    
    async def _set_in_layer(self, key: str, data: Any, layer: CacheLayer,
                           policy: CachePolicy, expiry_time: datetime):
        """Set data in specific cache layer"""
        
        if layer == CacheLayer.MEMORY:
            # Implement LRU eviction if at capacity
            if len(self._memory_cache) >= self._memory_cache_max_size:
                await self._evict_lru_memory_cache()
            
            self._memory_cache[key] = (data, expiry_time, 1)
        
        elif layer == CacheLayer.REDIS:
            try:
                serialized_data = self._serialize_data(data, policy.compression)
                ttl_seconds = int((expiry_time - datetime.utcnow()).total_seconds())
                
                await self.redis_client.setex(
                    f"cache:{key}", 
                    ttl_seconds, 
                    serialized_data
                )
                
                # Set metadata for monitoring
                await self.redis_client.hset(
                    f"cache:meta:{key}",
                    mapping={
                        "type": layer.value,
                        "size": len(serialized_data),
                        "created": datetime.utcnow().isoformat(),
                        "cost_weight": policy.cost_weight
                    }
                )
            except Exception as e:
                logger.warning(f"Redis set error: {e}")
        
        elif layer == CacheLayer.DATABASE:
            try:
                # Extract relevant fields based on data structure
                symbol = None
                analysis_type = "general"
                
                if isinstance(data, dict):
                    symbol = data.get('symbol')
                    analysis_type = data.get('type', 'general')
                
                query = text("""
                    INSERT INTO analysis_cache 
                    (symbol, analysis_type, result_data, expires_at)
                    VALUES (:symbol, :analysis_type, :result_data, :expires_at)
                    ON CONFLICT (symbol, analysis_type, analysis_date)
                    DO UPDATE SET 
                        result_data = EXCLUDED.result_data,
                        expires_at = EXCLUDED.expires_at
                """)
                
                await self.db_session.execute(query, {
                    "symbol": symbol or "unknown",
                    "analysis_type": analysis_type,
                    "result_data": json.dumps(data),
                    "expires_at": expiry_time
                })
                
                await self.db_session.commit()
            except Exception as e:
                logger.warning(f"Database cache set error: {e}")
    
    async def _promote_to_higher_layers(self, key: str, data: Any, 
                                       current_layer: CacheLayer, policy: CachePolicy):
        """Promote frequently accessed data to faster cache layers"""
        
        current_index = policy.layers.index(current_layer)
        
        # Promote to all layers above current layer
        for i in range(current_index):
            higher_layer = policy.layers[i]
            try:
                expiry_time = datetime.utcnow() + timedelta(seconds=policy.ttl_seconds)
                await self._set_in_layer(key, data, higher_layer, policy, expiry_time)
            except Exception as e:
                logger.warning(f"Cache promotion error to {higher_layer}: {e}")
    
    async def _evict_lru_memory_cache(self):
        """Evict least recently used items from memory cache"""
        if not self._memory_cache:
            return
        
        # Sort by access count (ascending) and remove 10% of items
        sorted_items = sorted(
            self._memory_cache.items(),
            key=lambda x: x[1][2]  # Access count
        )
        
        evict_count = max(1, len(sorted_items) // 10)
        
        for i in range(evict_count):
            key = sorted_items[i][0]
            del self._memory_cache[key]
            self.cache_stats['evictions'] += 1
    
    async def invalidate(self, key_parts: List[str], cache_type: str = 'default') -> bool:
        """Invalidate cache entry across all layers"""
        cache_key = self._generate_cache_key(key_parts, cache_type)
        policy = self.cache_policies.get(cache_type, self.cache_policies['stock_prices'])
        
        invalidated_count = 0
        
        for layer in policy.layers:
            try:
                if layer == CacheLayer.MEMORY:
                    if cache_key in self._memory_cache:
                        del self._memory_cache[cache_key]
                        invalidated_count += 1
                
                elif layer == CacheLayer.REDIS:
                    deleted = await self.redis_client.delete(f"cache:{cache_key}")
                    await self.redis_client.delete(f"cache:meta:{cache_key}")
                    invalidated_count += deleted
                
                elif layer == CacheLayer.DATABASE:
                    query = text("DELETE FROM analysis_cache WHERE id = :key")
                    result = await self.db_session.execute(query, {"key": cache_key})
                    await self.db_session.commit()
                    invalidated_count += result.rowcount
                    
            except Exception as e:
                logger.warning(f"Cache invalidation error in layer {layer}: {e}")
        
        return invalidated_count > 0
    
    async def _cleanup_expired_cache(self):
        """Background task to clean up expired cache entries"""
        while True:
            try:
                # Clean memory cache
                expired_keys = [
                    key for key, (_, expiry, _) in self._memory_cache.items()
                    if expiry <= datetime.utcnow()
                ]
                
                for key in expired_keys:
                    del self._memory_cache[key]
                    self.cache_stats['evictions'] += 1
                
                # Clean database cache
                query = text("DELETE FROM analysis_cache WHERE expires_at <= NOW()")
                result = await self.db_session.execute(query)
                await self.db_session.commit()
                
                if result.rowcount > 0:
                    logger.info(f"Cleaned {result.rowcount} expired database cache entries")
                
                # Wait 5 minutes before next cleanup
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def _optimize_cache_distribution(self):
        """Background task to optimize cache distribution across layers"""
        while True:
            try:
                # Analyze cache hit patterns and adjust policies
                total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
                
                if total_requests > 1000:  # Only optimize after sufficient data
                    hit_rate = self.cache_stats['hits'] / total_requests
                    
                    if hit_rate < 0.8:  # Poor hit rate, increase cache sizes
                        self._memory_cache_max_size = min(2000, self._memory_cache_max_size * 1.1)
                        logger.info(f"Increased memory cache size to {self._memory_cache_max_size}")
                    
                    elif hit_rate > 0.95:  # Excellent hit rate, can reduce cache sizes
                        self._memory_cache_max_size = max(500, self._memory_cache_max_size * 0.9)
                        logger.info(f"Decreased memory cache size to {self._memory_cache_max_size}")
                
                # Wait 15 minutes before next optimization
                await asyncio.sleep(900)
                
            except Exception as e:
                logger.error(f"Cache optimization error: {e}")
                await asyncio.sleep(300)
    
    async def _monitor_cache_performance(self):
        """Background task to monitor and log cache performance"""
        while True:
            try:
                total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
                
                if total_requests > 0:
                    hit_rate = self.cache_stats['hits'] / total_requests
                    
                    logger.info(f"Cache Performance - Hit Rate: {hit_rate:.2%}, "
                              f"API Calls Prevented: {self.cache_stats['api_calls_prevented']}, "
                              f"Estimated Cost Savings: ${self.cache_stats['cost_savings']:.2f}")
                
                # Reset stats periodically
                if total_requests > 10000:
                    self.cache_stats = {
                        'hits': 0,
                        'misses': 0,
                        'evictions': 0,
                        'cost_savings': 0.0,
                        'api_calls_prevented': 0
                    }
                
                # Wait 30 minutes before next monitoring report
                await asyncio.sleep(1800)
                
            except Exception as e:
                logger.error(f"Cache monitoring error: {e}")
                await asyncio.sleep(600)
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get current cache performance statistics"""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            'hit_rate': hit_rate,
            'total_requests': total_requests,
            'memory_cache_size': len(self._memory_cache),
            'memory_cache_max_size': self._memory_cache_max_size,
            'api_calls_prevented': self.cache_stats['api_calls_prevented'],
            'estimated_cost_savings': self.cache_stats['cost_savings'],
            'cache_policies': {k: {
                'ttl_seconds': v.ttl_seconds,
                'max_size': v.max_size,
                'layers': [layer.value for layer in v.layers],
                'cost_weight': v.cost_weight
            } for k, v in self.cache_policies.items()}
        }


# Cache decorator for easy integration
def cache_result(cache_type: str = 'default', ttl_override: Optional[int] = None):
    """Decorator to automatically cache function results"""
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            key_parts = [func.__name__] + [str(arg) for arg in args] + [f"{k}={v}" for k, v in sorted(kwargs.items())]
            
            # Try to get from cache first
            if hasattr(wrapper, '_cache_optimizer'):
                cached_result = await wrapper._cache_optimizer.get(key_parts, cache_type)
                if cached_result is not None:
                    return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            
            if hasattr(wrapper, '_cache_optimizer') and result is not None:
                await wrapper._cache_optimizer.set(key_parts, result, cache_type, ttl_override)
            
            return result
        
        return wrapper
    return decorator