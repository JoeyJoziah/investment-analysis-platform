"""
Cache primitive types: BloomFilter, CacheEntry, CacheStats, CompressionManager.

These are the foundational building blocks used by all cache tiers.
"""

import gzip
import hashlib
import json
import math
import os
import struct
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class BloomFilter:
    """
    Space-efficient probabilistic data structure for fast negative lookups.

    Returns False if key is DEFINITELY NOT in set (no false negatives).
    Returns True if key MIGHT be in set (possible false positives).

    Used to avoid expensive Redis/disk lookups for keys that don't exist.
    Target: 90% faster cache misses (10ms -> 1ms).
    """

    def __init__(
        self,
        expected_items: int = 100000,
        false_positive_rate: float = 0.01,
        persistence_path: Optional[str] = None
    ):
        """
        Initialize Bloom filter with optimal size for expected items.

        Args:
            expected_items: Expected number of unique keys
            false_positive_rate: Target false positive rate (0.01 = 1%)
            persistence_path: Optional file path for persistence
        """
        self.expected_items = expected_items
        self.false_positive_rate = false_positive_rate
        self.persistence_path = persistence_path

        # Calculate optimal filter size (bits) and hash count
        # m = -n * ln(p) / (ln(2)^2)
        self.size = self._optimal_size(expected_items, false_positive_rate)
        # k = (m/n) * ln(2)
        self.hash_count = self._optimal_hash_count(self.size, expected_items)

        # Initialize bit array (using bytearray for efficiency)
        self.byte_size = (self.size + 7) // 8
        self.bit_array = bytearray(self.byte_size)

        # Track statistics
        self.items_added = 0
        self.checks_performed = 0
        self.true_negatives = 0  # Definite misses (filter returned False)

        self._lock = threading.Lock()

        # Load persisted state if available
        if persistence_path and os.path.exists(persistence_path):
            self._load_from_disk()

        logger.info(
            f"BloomFilter initialized: size={self.size} bits, "
            f"hash_count={self.hash_count}, target_fp_rate={false_positive_rate:.2%}"
        )

    @staticmethod
    def _optimal_size(n: int, p: float) -> int:
        """Calculate optimal bit array size for n items with false positive rate p."""
        if n <= 0:
            return 1024
        if p <= 0:
            p = 0.001
        m = -n * math.log(p) / (math.log(2) ** 2)
        return max(int(m), 1024)  # Minimum 1024 bits

    @staticmethod
    def _optimal_hash_count(m: int, n: int) -> int:
        """Calculate optimal number of hash functions."""
        if n <= 0:
            return 3
        k = (m / n) * math.log(2)
        return max(int(k), 1)  # At least 1 hash function

    def _get_hash_values(self, key: str) -> List[int]:
        """
        Generate k hash values for a key using double hashing technique.

        Uses two independent hash functions to generate k values:
        h_i(x) = (h1(x) + i * h2(x)) mod m

        This is computationally cheaper than k independent hashes.
        """
        # Primary hash (SHA-256, first 8 bytes as int)
        key_bytes = key.encode('utf-8')
        sha_hash = hashlib.sha256(key_bytes).digest()
        h1 = struct.unpack('<Q', sha_hash[:8])[0]

        # Secondary hash (MD5, first 8 bytes as int)
        md5_hash = hashlib.md5(key_bytes).digest()
        h2 = struct.unpack('<Q', md5_hash[:8])[0]

        # Generate k hash values using double hashing
        hashes = []
        for i in range(self.hash_count):
            combined = (h1 + i * h2) % self.size
            hashes.append(combined)

        return hashes

    def add(self, key: str) -> None:
        """
        Add a key to the Bloom filter.

        This should be called whenever a key is added to the cache.
        """
        with self._lock:
            for bit_index in self._get_hash_values(key):
                byte_index = bit_index // 8
                bit_offset = bit_index % 8
                self.bit_array[byte_index] |= (1 << bit_offset)

            self.items_added += 1

    def might_contain(self, key: str) -> bool:
        """
        Check if key might be in the filter.

        Returns:
            False: Key is DEFINITELY NOT in the set (no false negatives)
            True: Key MIGHT be in the set (possible false positive)
        """
        with self._lock:
            self.checks_performed += 1

            for bit_index in self._get_hash_values(key):
                byte_index = bit_index // 8
                bit_offset = bit_index % 8

                if not (self.bit_array[byte_index] & (1 << bit_offset)):
                    # Bit is 0 - key is definitely not present
                    self.true_negatives += 1
                    return False

            # All bits are set - key might be present
            return True

    def __contains__(self, key: str) -> bool:
        """Allow 'in' operator usage: if key in bloom_filter."""
        return self.might_contain(key)

    def clear(self) -> None:
        """Clear the Bloom filter."""
        with self._lock:
            self.bit_array = bytearray(self.byte_size)
            self.items_added = 0
            self.checks_performed = 0
            self.true_negatives = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get Bloom filter statistics."""
        with self._lock:
            # Estimate current false positive rate
            # p' = (1 - e^(-kn/m))^k
            if self.items_added > 0:
                exp_term = math.exp(-self.hash_count * self.items_added / self.size)
                estimated_fp_rate = (1 - exp_term) ** self.hash_count
            else:
                estimated_fp_rate = 0.0

            # Calculate fill ratio (percentage of bits set to 1)
            bits_set = sum(bin(byte).count('1') for byte in self.bit_array)
            fill_ratio = bits_set / self.size if self.size > 0 else 0

            return {
                'size_bits': self.size,
                'size_bytes': self.byte_size,
                'hash_count': self.hash_count,
                'items_added': self.items_added,
                'checks_performed': self.checks_performed,
                'true_negatives': self.true_negatives,
                'true_negative_rate': (
                    self.true_negatives / max(self.checks_performed, 1)
                ),
                'target_fp_rate': self.false_positive_rate,
                'estimated_fp_rate': estimated_fp_rate,
                'fill_ratio': fill_ratio,
                'capacity_remaining': max(0, self.expected_items - self.items_added)
            }

    def save_to_disk(self) -> bool:
        """
        Persist Bloom filter to disk for recovery across restarts.

        Returns:
            True if save successful, False otherwise
        """
        if not self.persistence_path:
            return False

        try:
            with self._lock:
                # Create header with metadata
                header = {
                    'version': 1,
                    'size': self.size,
                    'hash_count': self.hash_count,
                    'expected_items': self.expected_items,
                    'false_positive_rate': self.false_positive_rate,
                    'items_added': self.items_added,
                    'saved_at': datetime.now().isoformat()
                }

                # Write header + bit array
                with open(self.persistence_path, 'wb') as f:
                    header_bytes = json.dumps(header).encode('utf-8')
                    # Write header length (4 bytes) + header + bit array
                    f.write(struct.pack('<I', len(header_bytes)))
                    f.write(header_bytes)
                    f.write(self.bit_array)

                logger.debug(f"BloomFilter saved to {self.persistence_path}")
                return True

        except Exception as e:
            logger.error(f"Failed to save BloomFilter: {e}")
            return False

    def _load_from_disk(self) -> bool:
        """
        Load Bloom filter state from disk.

        Returns:
            True if load successful, False otherwise
        """
        if not self.persistence_path or not os.path.exists(self.persistence_path):
            return False

        try:
            with open(self.persistence_path, 'rb') as f:
                # Read header length
                header_len_bytes = f.read(4)
                if len(header_len_bytes) < 4:
                    return False

                header_len = struct.unpack('<I', header_len_bytes)[0]

                # Read and parse header
                header_bytes = f.read(header_len)
                header = json.loads(header_bytes.decode('utf-8'))

                # Validate compatibility
                if header.get('version') != 1:
                    logger.warning("BloomFilter version mismatch, reinitializing")
                    return False

                if (header.get('size') != self.size or
                        header.get('hash_count') != self.hash_count):
                    logger.warning("BloomFilter config changed, reinitializing")
                    return False

                # Read bit array
                self.bit_array = bytearray(f.read())
                self.items_added = header.get('items_added', 0)

                logger.info(
                    f"BloomFilter loaded from {self.persistence_path}: "
                    f"{self.items_added} items"
                )
                return True

        except Exception as e:
            logger.error(f"Failed to load BloomFilter: {e}")
            return False


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
    def compress_data(data: Any, method: str = 'gzip') -> tuple:
        """
        Compress data and return compressed bytes + ratio.

        SECURITY: Uses JSON serialization only (no pickle).
        Data must be JSON-serializable.
        """
        try:
            # SECURITY: Use JSON only - no pickle to prevent arbitrary code execution
            serialized = json.dumps(data, default=str).encode('utf-8')

            original_size = len(serialized)

            if method == 'gzip':
                compressed = gzip.compress(serialized)
            else:
                compressed = serialized

            compressed_size = len(compressed)
            compression_ratio = compressed_size / original_size if original_size > 0 else 1.0

            return compressed, compression_ratio

        except Exception as e:
            logger.warning(f"Compression failed (data must be JSON-serializable): {e}")
            # Return JSON-serialized error marker instead of pickle
            error_data = json.dumps({"__error__": str(e)}).encode('utf-8')
            return error_data, 1.0

    @staticmethod
    def decompress_data(compressed_bytes: bytes, method: str = 'gzip') -> Any:
        """
        Decompress and deserialize data.

        SECURITY: Uses JSON deserialization only (no pickle).
        """
        try:
            if method == 'gzip':
                decompressed = gzip.decompress(compressed_bytes)
            else:
                decompressed = compressed_bytes

            # SECURITY: JSON only - do NOT use pickle fallback
            return json.loads(decompressed.decode('utf-8'))

        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            # SECURITY: Do NOT fall back to pickle - it allows arbitrary code execution
            logger.error(f"Decompression failed - data not JSON compatible: {e}")
            return None

        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            return None
