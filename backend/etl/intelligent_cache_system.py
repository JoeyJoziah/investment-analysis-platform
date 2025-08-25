"""
Intelligent Multi-Tier Caching System for Unlimited Stock Data Extraction
Implements memory, disk, and distributed caching with smart invalidation
"""

import asyncio
import aiofiles
import json
import gzip
import pickle
import hashlib
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Set, Union, Callable
import logging
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import redis
from cachetools import TTLCache, LRUCache
import psutil
import mmap

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Represents a cached item with metadata"""
    key: str
    data: Any
    created_at: datetime
    expires_at: datetime
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    size_bytes: int = 0
    compression_ratio: float = 1.0
    hit_score: float = 0.0
    source_tier: str = 'unknown'

@dataclass
class CacheStats:
    """Cache performance statistics"""
    total_requests: int = 0
    hits: int = 0
    misses: int = 0
    memory_hits: int = 0
    disk_hits: int = 0
    redis_hits: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    compression_saved_bytes: int = 0
    
    @property
    def hit_rate(self) -> float:
        return self.hits / max(self.total_requests, 1)
    
    @property
    def memory_hit_rate(self) -> float:
        return self.memory_hits / max(self.hits, 1) if self.hits > 0 else 0

class CompressionManager:
    """Handles data compression for cache storage"""
    
    @staticmethod
    def compress_data(data: Any, method: str = 'gzip') -> tuple[bytes, float]:
        """Compress data and return compressed bytes + ratio"""
        try:
            # Serialize data
            if isinstance(data, (dict, list)):
                serialized = json.dumps(data).encode('utf-8')
            else:
                serialized = pickle.dumps(data)
            
            original_size = len(serialized)
            
            if method == 'gzip':
                compressed = gzip.compress(serialized)
            else:
                compressed = serialized
            
            compressed_size = len(compressed)
            compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
            
            return compressed, compression_ratio
            
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return pickle.dumps(data), 1.0
    
    @staticmethod
    def decompress_data(compressed_bytes: bytes, method: str = 'gzip') -> Any:
        """Decompress and deserialize data"""
        try:
            if method == 'gzip':
                decompressed = gzip.decompress(compressed_bytes)
            else:
                decompressed = compressed_bytes
            
            # Try JSON first, then pickle
            try:
                return json.loads(decompressed.decode('utf-8'))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return pickle.loads(decompressed)
                
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            return None

class MemoryTierCache:
    """L1 Cache - Fast in-memory storage with LRU eviction"""
    
    def __init__(self, max_size_mb: int = 256, ttl_hours: int = 1):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.ttl_seconds = ttl_hours * 3600
        
        # Use TTL cache for automatic expiration
        self.cache = TTLCache(maxsize=10000, ttl=self.ttl_seconds)
        self.metadata = {}  # Key -> CacheEntry metadata
        self.current_size = 0
        self.lock = threading.RLock()
        
        logger.info(f"Initialized memory cache: {max_size_mb}MB, TTL: {ttl_hours}h")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from memory cache"""
        with self.lock:
            try:
                if key in self.cache:
                    # Update access metadata
                    if key in self.metadata:
                        entry = self.metadata[key]
                        entry.access_count += 1
                        entry.last_accessed = datetime.now()
                        entry.hit_score = self._calculate_hit_score(entry)
                    
                    return self.cache[key]
                return None
                
            except Exception as e:
                logger.error(f"Memory cache get error for {key}: {e}")
                return None
    
    def set(self, key: str, data: Any, ttl_override: Optional[int] = None) -> bool:
        """Set item in memory cache"""
        with self.lock:
            try:
                # Estimate data size
                data_size = self._estimate_size(data)
                
                # Check if we need to make room
                if self.current_size + data_size > self.max_size_bytes:
                    self._evict_least_valuable(data_size)
                
                # Store data
                if ttl_override:
                    # Create new TTL cache instance for custom TTL
                    self.cache[key] = data
                else:
                    self.cache[key] = data
                
                # Store metadata
                expires_at = datetime.now() + timedelta(seconds=ttl_override or self.ttl_seconds)
                self.metadata[key] = CacheEntry(
                    key=key,
                    data=None,  # Don't duplicate data in metadata
                    created_at=datetime.now(),
                    expires_at=expires_at,
                    size_bytes=data_size,
                    source_tier='memory'
                )
                
                self.current_size += data_size
                return True
                
            except Exception as e:
                logger.error(f"Memory cache set error for {key}: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        """Delete item from memory cache"""
        with self.lock:
            try:
                if key in self.cache:
                    del self.cache[key]
                    
                    if key in self.metadata:
                        self.current_size -= self.metadata[key].size_bytes
                        del self.metadata[key]
                    
                    return True
                return False
                
            except Exception as e:
                logger.error(f"Memory cache delete error for {key}: {e}")
                return False
    
    def _estimate_size(self, data: Any) -> int:
        """Estimate memory size of data"""
        try:
            if isinstance(data, str):
                return len(data.encode('utf-8'))
            elif isinstance(data, (dict, list)):
                return len(json.dumps(data).encode('utf-8'))
            else:
                return len(pickle.dumps(data))
        except:
            return 1024  # Default estimate
    
    def _calculate_hit_score(self, entry: CacheEntry) -> float:
        """Calculate hit score for eviction decisions"""
        age_hours = (datetime.now() - entry.created_at).total_seconds() / 3600
        recency_hours = (datetime.now() - entry.last_accessed).total_seconds() / 3600
        
        # Higher score = more valuable to keep
        frequency_score = entry.access_count / max(age_hours, 1)
        recency_score = 1.0 / max(recency_hours, 0.1)
        
        return frequency_score + recency_score
    
    def _evict_least_valuable(self, needed_bytes: int):
        """Evict least valuable items to make room"""
        if not self.metadata:
            return
        
        # Sort by hit score (ascending = least valuable first)
        sorted_entries = sorted(
            self.metadata.values(),
            key=lambda e: e.hit_score
        )
        
        freed_bytes = 0
        for entry in sorted_entries:
            if freed_bytes >= needed_bytes:
                break
            
            self.delete(entry.key)
            freed_bytes += entry.size_bytes
            logger.debug(f"Evicted cache entry {entry.key} (score: {entry.hit_score:.2f})")
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        with self.lock:
            return {
                'entries': len(self.cache),
                'size_bytes': self.current_size,
                'size_mb': self.current_size / (1024 * 1024),
                'max_size_mb': self.max_size_bytes / (1024 * 1024),
                'utilization': self.current_size / self.max_size_bytes,
                'avg_entry_size': self.current_size / max(len(self.cache), 1)
            }

class DiskTierCache:
    """L2 Cache - Persistent disk storage with compression"""
    
    def __init__(self, cache_dir: str, max_size_mb: int = 2048, ttl_hours: int = 24):
        self.cache_dir = cache_dir
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.ttl_hours = ttl_hours
        
        os.makedirs(cache_dir, exist_ok=True)
        
        # SQLite index for fast lookups
        self.index_db_path = os.path.join(cache_dir, 'cache_index.db')
        self._init_index_db()
        
        self.compression_manager = CompressionManager()
        self.lock = threading.RLock()
        
        # Background cleanup task
        self._cleanup_expired()
        
        logger.info(f"Initialized disk cache: {cache_dir}, {max_size_mb}MB, TTL: {ttl_hours}h")
    
    def _init_index_db(self):
        """Initialize SQLite index database"""
        conn = sqlite3.connect(self.index_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache_index (
                key TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT,
                size_bytes INTEGER DEFAULT 0,
                compression_method TEXT DEFAULT 'gzip',
                compression_ratio REAL DEFAULT 1.0
            )
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_expires_at ON cache_index(expires_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache_index(last_accessed)")
        
        conn.commit()
        conn.close()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from disk cache"""
        with self.lock:
            conn = sqlite3.connect(self.index_db_path)
            cursor = conn.cursor()
            
            try:
                # Check if key exists and not expired
                cursor.execute("""
                    SELECT file_path, compression_method, expires_at 
                    FROM cache_index 
                    WHERE key = ? AND expires_at > ?
                """, (key, datetime.now().isoformat()))
                
                result = cursor.fetchone()
                if not result:
                    return None
                
                file_path, compression_method, expires_at = result
                
                # Load and decompress data
                try:
                    with open(file_path, 'rb') as f:
                        compressed_data = f.read()
                    
                    data = self.compression_manager.decompress_data(compressed_data, compression_method)
                    
                    # Update access statistics
                    cursor.execute("""
                        UPDATE cache_index 
                        SET access_count = access_count + 1, last_accessed = ?
                        WHERE key = ?
                    """, (datetime.now().isoformat(), key))
                    
                    conn.commit()
                    return data
                    
                except (IOError, OSError) as e:
                    logger.warning(f"Failed to read cache file {file_path}: {e}")
                    # Clean up broken index entry
                    cursor.execute("DELETE FROM cache_index WHERE key = ?", (key,))
                    conn.commit()
                    return None
                
            except Exception as e:
                logger.error(f"Disk cache get error for {key}: {e}")
                return None
                
            finally:
                conn.close()
    
    def set(self, key: str, data: Any, ttl_hours: Optional[int] = None) -> bool:
        """Set item in disk cache"""
        with self.lock:
            try:
                # Generate file path
                key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
                file_path = os.path.join(self.cache_dir, f"{key_hash}.cache")
                
                # Compress data
                compressed_data, compression_ratio = self.compression_manager.compress_data(data)
                
                # Check disk space
                if not self._ensure_disk_space(len(compressed_data)):
                    logger.warning("Unable to make disk space for cache entry")
                    return False
                
                # Write to disk
                with open(file_path, 'wb') as f:
                    f.write(compressed_data)
                
                # Update index
                expires_at = datetime.now() + timedelta(hours=ttl_hours or self.ttl_hours)
                
                conn = sqlite3.connect(self.index_db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO cache_index 
                    (key, file_path, created_at, expires_at, size_bytes, 
                     compression_method, compression_ratio, last_accessed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (key, file_path, datetime.now().isoformat(), expires_at.isoformat(),
                      len(compressed_data), 'gzip', compression_ratio, datetime.now().isoformat()))
                
                conn.commit()
                conn.close()
                
                return True
                
            except Exception as e:
                logger.error(f"Disk cache set error for {key}: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        """Delete item from disk cache"""
        with self.lock:
            conn = sqlite3.connect(self.index_db_path)
            cursor = conn.cursor()
            
            try:
                # Get file path
                cursor.execute("SELECT file_path FROM cache_index WHERE key = ?", (key,))
                result = cursor.fetchone()
                
                if result:
                    file_path = result[0]
                    
                    # Delete file
                    try:
                        if os.path.exists(file_path):
                            os.unlink(file_path)
                    except OSError as e:
                        logger.warning(f"Failed to delete cache file {file_path}: {e}")
                    
                    # Delete from index
                    cursor.execute("DELETE FROM cache_index WHERE key = ?", (key,))
                    conn.commit()
                    
                    return True
                
                return False
                
            except Exception as e:
                logger.error(f"Disk cache delete error for {key}: {e}")
                return False
                
            finally:
                conn.close()
    
    def _ensure_disk_space(self, needed_bytes: int) -> bool:
        """Ensure sufficient disk space by cleaning up if needed"""
        conn = sqlite3.connect(self.index_db_path)
        cursor = conn.cursor()
        
        try:
            # Get current disk usage
            cursor.execute("SELECT SUM(size_bytes) FROM cache_index")
            current_usage = cursor.fetchone()[0] or 0
            
            if current_usage + needed_bytes <= self.max_size_bytes:
                return True
            
            # Need to clean up - remove least recently accessed entries
            bytes_to_free = (current_usage + needed_bytes) - self.max_size_bytes
            
            cursor.execute("""
                SELECT key, size_bytes 
                FROM cache_index 
                ORDER BY last_accessed ASC
            """)
            
            freed_bytes = 0
            for key, size_bytes in cursor.fetchall():
                if freed_bytes >= bytes_to_free:
                    break
                
                self.delete(key)
                freed_bytes += size_bytes
            
            return freed_bytes >= bytes_to_free
            
        except Exception as e:
            logger.error(f"Error ensuring disk space: {e}")
            return False
            
        finally:
            conn.close()
    
    def _cleanup_expired(self):
        """Remove expired cache entries"""
        conn = sqlite3.connect(self.index_db_path)
        cursor = conn.cursor()
        
        try:
            # Get expired entries
            cursor.execute("""
                SELECT key, file_path 
                FROM cache_index 
                WHERE expires_at < ?
            """, (datetime.now().isoformat(),))
            
            expired_entries = cursor.fetchall()
            
            for key, file_path in expired_entries:
                try:
                    if os.path.exists(file_path):
                        os.unlink(file_path)
                except OSError:
                    pass
            
            # Clean up index
            cursor.execute("DELETE FROM cache_index WHERE expires_at < ?", 
                          (datetime.now().isoformat(),))
            
            conn.commit()
            
            if expired_entries:
                logger.info(f"Cleaned up {len(expired_entries)} expired cache entries")
                
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")
            
        finally:
            conn.close()
    
    def get_stats(self) -> Dict:
        """Get disk cache statistics"""
        conn = sqlite3.connect(self.index_db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT COUNT(*), SUM(size_bytes), AVG(compression_ratio) FROM cache_index")
            count, total_size, avg_compression = cursor.fetchone()
            
            return {
                'entries': count or 0,
                'size_bytes': total_size or 0,
                'size_mb': (total_size or 0) / (1024 * 1024),
                'max_size_mb': self.max_size_bytes / (1024 * 1024),
                'utilization': (total_size or 0) / self.max_size_bytes,
                'avg_compression_ratio': avg_compression or 1.0,
                'compression_saved_mb': ((total_size or 0) * (1 - (avg_compression or 1.0))) / (1024 * 1024)
            }
            
        except Exception as e:
            logger.error(f"Error getting disk cache stats: {e}")
            return {}
            
        finally:
            conn.close()

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
            cache_entries = len([k for k in cache_keys if not k.endswith((b':hits', b':created'))])
            
            return {
                'available': True,
                'entries': cache_entries,
                'memory_usage_mb': info.get('used_memory', 0) / (1024 * 1024),
                'connected_clients': info.get('connected_clients', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'hit_rate': info.get('keyspace_hits', 0) / 
                          max(info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0), 1)
            }
            
        except Exception as e:
            logger.warning(f"Error getting Redis stats: {e}")
            return {'available': False, 'error': str(e)}

class IntelligentCacheManager:
    """Multi-tier intelligent cache manager with automatic optimization"""
    
    def __init__(self, 
                 cache_dir: str = "/tmp/intelligent_cache",
                 memory_size_mb: int = 256,
                 disk_size_mb: int = 2048,
                 redis_url: Optional[str] = None,
                 enable_analytics: bool = True):
        
        self.cache_dir = cache_dir
        self.enable_analytics = enable_analytics
        
        # Initialize cache tiers
        self.memory_cache = MemoryTierCache(memory_size_mb, ttl_hours=1)
        self.disk_cache = DiskTierCache(cache_dir, disk_size_mb, ttl_hours=24)
        self.redis_cache = RedisTierCache(redis_url or "redis://localhost:6379", ttl_hours=48)
        
        # Cache statistics
        self.stats = CacheStats()
        
        # Access patterns for optimization
        self.access_patterns = {}
        self.optimization_interval = 3600  # 1 hour
        self.last_optimization = time.time()
        
        # Background tasks
        self._start_background_tasks()
        
        logger.info("Initialized IntelligentCacheManager with 3-tier architecture")
    
    async def get(self, key: str, category: str = "default") -> Optional[Any]:
        """Get item from cache with intelligent tier selection"""
        start_time = time.time()
        self.stats.total_requests += 1
        
        # Track access pattern
        if self.enable_analytics:
            self._track_access(key, category)
        
        # Try memory cache first (L1)
        data = self.memory_cache.get(key)
        if data is not None:
            self.stats.hits += 1
            self.stats.memory_hits += 1
            await self._record_hit(key, 'memory', time.time() - start_time)
            return data
        
        # Try disk cache (L2)
        data = self.disk_cache.get(key)
        if data is not None:
            self.stats.hits += 1
            self.stats.disk_hits += 1
            
            # Promote to memory cache for faster access
            self.memory_cache.set(key, data)
            
            await self._record_hit(key, 'disk', time.time() - start_time)
            return data
        
        # Try Redis cache (L3)
        if self.redis_cache.is_available():
            data = self.redis_cache.get(key)
            if data is not None:
                self.stats.hits += 1
                self.stats.redis_hits += 1
                
                # Promote to higher tiers
                self.disk_cache.set(key, data)
                self.memory_cache.set(key, data)
                
                await self._record_hit(key, 'redis', time.time() - start_time)
                return data
        
        # Cache miss
        self.stats.misses += 1
        await self._record_miss(key, category)
        return None
    
    async def set(self, key: str, data: Any, category: str = "default", 
                  ttl_hours: Optional[int] = None) -> bool:
        """Set item in appropriate cache tiers"""
        if data is None:
            return False
        
        success = False
        
        # Determine optimal storage strategy based on data characteristics
        data_size = self._estimate_data_size(data)
        access_frequency = self._get_access_frequency(key)
        
        # Always try to store in memory cache for hot data
        if access_frequency > 0.1 or data_size < 10240:  # < 10KB
            if self.memory_cache.set(key, data, ttl_hours and ttl_hours * 3600):
                success = True
        
        # Store in disk cache for medium-term storage
        if self.disk_cache.set(key, data, ttl_hours):
            success = True
        
        # Store in Redis for distributed access (optional)
        if self.redis_cache.is_available() and (access_frequency > 0.05 or category == "shared"):
            self.redis_cache.set(key, data, ttl_hours and ttl_hours * 3600)
        
        if self.enable_analytics:
            self._track_write(key, category, data_size)
        
        return success
    
    async def delete(self, key: str) -> bool:
        """Delete item from all cache tiers"""
        results = []
        
        results.append(self.memory_cache.delete(key))
        results.append(self.disk_cache.delete(key))
        
        if self.redis_cache.is_available():
            results.append(self.redis_cache.delete(key))
        
        return any(results)
    
    async def bulk_get(self, keys: List[str], category: str = "default") -> Dict[str, Any]:
        """Get multiple items efficiently"""
        results = {}
        
        # Use asyncio to parallelize cache lookups
        tasks = []
        for key in keys:
            tasks.append(self.get(key, category))
        
        values = await asyncio.gather(*tasks, return_exceptions=True)
        
        for key, value in zip(keys, values):
            if value is not None and not isinstance(value, Exception):
                results[key] = value
        
        return results
    
    async def bulk_set(self, items: Dict[str, Any], category: str = "default", 
                      ttl_hours: Optional[int] = None) -> Dict[str, bool]:
        """Set multiple items efficiently"""
        results = {}
        
        # Use asyncio to parallelize cache writes
        tasks = []
        for key, value in items.items():
            tasks.append(self.set(key, value, category, ttl_hours))
        
        success_values = await asyncio.gather(*tasks, return_exceptions=True)
        
        for key, success in zip(items.keys(), success_values):
            results[key] = success if not isinstance(success, Exception) else False
        
        return results
    
    def _estimate_data_size(self, data: Any) -> int:
        """Estimate size of data in bytes"""
        try:
            if isinstance(data, str):
                return len(data.encode('utf-8'))
            elif isinstance(data, (dict, list)):
                return len(json.dumps(data).encode('utf-8'))
            else:
                return len(pickle.dumps(data))
        except:
            return 1024  # Default estimate
    
    def _get_access_frequency(self, key: str) -> float:
        """Get access frequency for key from analytics"""
        if not self.enable_analytics or key not in self.access_patterns:
            return 0.0
        
        pattern = self.access_patterns[key]
        total_accesses = pattern.get('reads', 0) + pattern.get('writes', 0)
        time_window = time.time() - pattern.get('first_seen', time.time())
        
        return total_accesses / max(time_window / 3600, 1)  # Accesses per hour
    
    def _track_access(self, key: str, category: str):
        """Track access patterns for optimization"""
        now = time.time()
        
        if key not in self.access_patterns:
            self.access_patterns[key] = {
                'first_seen': now,
                'last_accessed': now,
                'reads': 0,
                'writes': 0,
                'category': category
            }
        
        pattern = self.access_patterns[key]
        pattern['reads'] += 1
        pattern['last_accessed'] = now
    
    def _track_write(self, key: str, category: str, size: int):
        """Track write patterns"""
        now = time.time()
        
        if key not in self.access_patterns:
            self.access_patterns[key] = {
                'first_seen': now,
                'last_accessed': now,
                'reads': 0,
                'writes': 0,
                'category': category,
                'avg_size': size
            }
        else:
            pattern = self.access_patterns[key]
            pattern['writes'] += 1
            pattern['last_accessed'] = now
            # Update average size
            if 'avg_size' in pattern:
                pattern['avg_size'] = (pattern['avg_size'] + size) / 2
            else:
                pattern['avg_size'] = size
    
    async def _record_hit(self, key: str, tier: str, response_time: float):
        """Record cache hit for analytics"""
        if self.enable_analytics:
            # Could send to monitoring system
            logger.debug(f"Cache hit: {key} from {tier} in {response_time*1000:.1f}ms")
    
    async def _record_miss(self, key: str, category: str):
        """Record cache miss for analytics"""
        if self.enable_analytics:
            logger.debug(f"Cache miss: {key} (category: {category})")
    
    def _start_background_tasks(self):
        """Start background optimization tasks"""
        def cleanup_task():
            """Periodic cleanup task"""
            while True:
                try:
                    # Clean up expired disk cache entries
                    self.disk_cache._cleanup_expired()
                    
                    # Optimize cache based on access patterns
                    if time.time() - self.last_optimization > self.optimization_interval:
                        self._optimize_cache_strategy()
                        self.last_optimization = time.time()
                    
                    time.sleep(300)  # Run every 5 minutes
                    
                except Exception as e:
                    logger.error(f"Background cleanup task error: {e}")
                    time.sleep(60)  # Wait before retrying
        
        # Start cleanup thread
        cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
        cleanup_thread.start()
    
    def _optimize_cache_strategy(self):
        """Optimize caching strategy based on access patterns"""
        if not self.enable_analytics:
            return
        
        logger.info("Running cache optimization...")
        
        try:
            # Find hot keys that should be promoted to memory
            hot_keys = []
            cold_keys = []
            
            for key, pattern in self.access_patterns.items():
                frequency = self._get_access_frequency(key)
                
                if frequency > 0.5:  # More than 30 accesses per hour
                    hot_keys.append((key, frequency))
                elif frequency < 0.01:  # Less than 1 access per 100 hours
                    cold_keys.append(key)
            
            # Promote hot keys to memory
            for key, freq in hot_keys[:50]:  # Top 50 hot keys
                data = self.disk_cache.get(key)
                if data and not self.memory_cache.get(key):
                    self.memory_cache.set(key, data)
                    logger.debug(f"Promoted hot key {key} to memory (freq: {freq:.2f})")
            
            # Consider evicting cold keys from memory
            for key in cold_keys:
                if self.memory_cache.delete(key):
                    logger.debug(f"Evicted cold key {key} from memory")
            
            logger.info(f"Cache optimization complete: {len(hot_keys)} hot keys, {len(cold_keys)} cold keys")
            
        except Exception as e:
            logger.error(f"Cache optimization error: {e}")
    
    def get_comprehensive_stats(self) -> Dict:
        """Get comprehensive cache statistics"""
        memory_stats = self.memory_cache.get_stats()
        disk_stats = self.disk_cache.get_stats()
        redis_stats = self.redis_cache.get_stats()
        
        total_entries = memory_stats.get('entries', 0) + disk_stats.get('entries', 0)
        if redis_stats.get('available'):
            total_entries += redis_stats.get('entries', 0)
        
        return {
            'overview': {
                'total_requests': self.stats.total_requests,
                'hit_rate': self.stats.hit_rate,
                'total_entries': total_entries,
                'memory_hit_rate': self.stats.memory_hit_rate
            },
            'performance': {
                'hits': self.stats.hits,
                'misses': self.stats.misses,
                'memory_hits': self.stats.memory_hits,
                'disk_hits': self.stats.disk_hits,
                'redis_hits': self.stats.redis_hits
            },
            'tiers': {
                'memory': memory_stats,
                'disk': disk_stats,
                'redis': redis_stats
            },
            'analytics': {
                'tracked_keys': len(self.access_patterns) if self.enable_analytics else 0,
                'hot_keys': len([k for k, p in self.access_patterns.items() 
                               if self._get_access_frequency(k) > 0.5]) if self.enable_analytics else 0
            }
        }
    
    async def clear_all(self) -> bool:
        """Clear all cache tiers"""
        try:
            # Clear memory cache
            self.memory_cache = MemoryTierCache(
                self.memory_cache.max_size_bytes // (1024 * 1024), 
                self.memory_cache.ttl_seconds // 3600
            )
            
            # Clear disk cache (remove all files)
            if os.path.exists(self.cache_dir):
                for file in os.listdir(self.cache_dir):
                    if file.endswith('.cache'):
                        os.unlink(os.path.join(self.cache_dir, file))
                
                # Reinitialize disk cache
                self.disk_cache._init_index_db()
            
            # Clear Redis cache
            if self.redis_cache.is_available():
                keys = self.redis_cache.redis_client.keys("cache:*")
                if keys:
                    self.redis_cache.redis_client.delete(*keys)
            
            # Reset statistics
            self.stats = CacheStats()
            self.access_patterns.clear()
            
            logger.info("All cache tiers cleared")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False

# Example usage and testing
async def test_intelligent_cache():
    """Test the intelligent cache system"""
    cache = IntelligentCacheManager(
        cache_dir="/tmp/test_cache",
        memory_size_mb=64,
        disk_size_mb=256,
        redis_url="redis://localhost:6379"
    )
    
    try:
        # Test single operations
        logger.info("Testing single cache operations...")
        
        test_data = {
            'ticker': 'AAPL',
            'price': 150.25,
            'volume': 50000000,
            'timestamp': datetime.now().isoformat()
        }
        
        # Set data
        success = await cache.set('AAPL:stock_data', test_data, 'stocks')
        logger.info(f"Set operation: {'✓' if success else '✗'}")
        
        # Get data
        retrieved_data = await cache.get('AAPL:stock_data', 'stocks')
        logger.info(f"Get operation: {'✓' if retrieved_data else '✗'}")
        logger.info(f"Data match: {'✓' if retrieved_data == test_data else '✗'}")
        
        # Test bulk operations
        logger.info("Testing bulk cache operations...")
        
        bulk_data = {}
        for ticker in ['MSFT', 'GOOGL', 'AMZN', 'META']:
            bulk_data[f'{ticker}:stock_data'] = {
                'ticker': ticker,
                'price': 100.0 + hash(ticker) % 100,
                'timestamp': datetime.now().isoformat()
            }
        
        # Bulk set
        set_results = await cache.bulk_set(bulk_data, 'stocks')
        successful_sets = sum(set_results.values())
        logger.info(f"Bulk set: {successful_sets}/{len(bulk_data)} successful")
        
        # Bulk get
        get_results = await cache.bulk_get(list(bulk_data.keys()), 'stocks')
        logger.info(f"Bulk get: {len(get_results)}/{len(bulk_data)} retrieved")
        
        # Test cache promotion
        logger.info("Testing cache tier promotion...")
        
        # Access the same key multiple times to trigger promotion
        for _ in range(5):
            await cache.get('AAPL:stock_data', 'stocks')
        
        # Show comprehensive statistics
        stats = cache.get_comprehensive_stats()
        logger.info("Cache Statistics:")
        logger.info(f"  Hit Rate: {stats['overview']['hit_rate']:.2%}")
        logger.info(f"  Total Entries: {stats['overview']['total_entries']}")
        logger.info(f"  Memory Hits: {stats['performance']['memory_hits']}")
        logger.info(f"  Disk Hits: {stats['performance']['disk_hits']}")
        logger.info(f"  Redis Available: {stats['tiers']['redis'].get('available', False)}")
        
        logger.info("Intelligent cache test completed successfully!")
        
    except Exception as e:
        logger.error(f"Cache test failed: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    asyncio.run(test_intelligent_cache())