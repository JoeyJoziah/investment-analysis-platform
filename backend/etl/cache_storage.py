"""
Cache storage tier implementations: MemoryTierCache and DiskTierCache.

- MemoryTierCache (L1): Fast in-memory storage with LRU eviction
- DiskTierCache   (L2): Persistent disk storage with compression and SQLite index

The distributed Redis tier lives in cache_redis.py (RedisTierCache).
"""

import hashlib
import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from cachetools import TTLCache

try:
    from .cache_primitives import CacheEntry, CompressionManager
except ImportError:
    import os as _os, sys as _sys
    _here = _os.path.dirname(_os.path.abspath(__file__))
    if _here not in _sys.path:
        _sys.path.insert(0, _here)
    from cache_primitives import CacheEntry, CompressionManager  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


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
        """
        Estimate memory size of data.
        SECURITY: Uses JSON for size estimation - no pickle to prevent code execution.
        """
        try:
            if isinstance(data, str):
                return len(data.encode('utf-8'))
            elif isinstance(data, (dict, list)):
                return len(json.dumps(data, default=str).encode('utf-8'))
            elif hasattr(data, '__dict__'):
                return len(json.dumps(data.__dict__, default=str).encode('utf-8'))
            else:
                return len(str(data).encode('utf-8'))
        except Exception:
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
                cursor.execute("""
                    SELECT file_path, compression_method, expires_at
                    FROM cache_index
                    WHERE key = ? AND expires_at > ?
                """, (key, datetime.now().isoformat()))

                result = cursor.fetchone()
                if not result:
                    return None

                file_path, compression_method, _ = result

                try:
                    with open(file_path, 'rb') as f:
                        compressed_data = f.read()

                    data = self.compression_manager.decompress_data(compressed_data, compression_method)

                    cursor.execute("""
                        UPDATE cache_index
                        SET access_count = access_count + 1, last_accessed = ?
                        WHERE key = ?
                    """, (datetime.now().isoformat(), key))

                    conn.commit()
                    return data

                except (IOError, OSError) as e:
                    logger.warning(f"Failed to read cache file {file_path}: {e}")
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
                key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
                file_path = os.path.join(self.cache_dir, f"{key_hash}.cache")

                compressed_data, compression_ratio = self.compression_manager.compress_data(data)

                if not self._ensure_disk_space(len(compressed_data)):
                    logger.warning("Unable to make disk space for cache entry")
                    return False

                with open(file_path, 'wb') as f:
                    f.write(compressed_data)

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
                cursor.execute("SELECT file_path FROM cache_index WHERE key = ?", (key,))
                result = cursor.fetchone()

                if result:
                    file_path = result[0]

                    try:
                        if os.path.exists(file_path):
                            os.unlink(file_path)
                    except OSError as e:
                        logger.warning(f"Failed to delete cache file {file_path}: {e}")

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
            cursor.execute("SELECT SUM(size_bytes) FROM cache_index")
            current_usage = cursor.fetchone()[0] or 0

            if current_usage + needed_bytes <= self.max_size_bytes:
                return True

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
                'compression_saved_mb': (
                    (total_size or 0) * (1 - (avg_compression or 1.0)) / (1024 * 1024)
                )
            }

        except Exception as e:
            logger.error(f"Error getting disk cache stats: {e}")
            return {}

        finally:
            conn.close()
