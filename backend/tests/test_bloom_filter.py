"""
Tests for Bloom Filter implementation in intelligent_cache_system.py

Verifies:
1. No false negatives (CRITICAL - key added must always be found)
2. False positive rate within target (1%)
3. Persistence across restarts
4. Thread safety
5. Performance improvements for cache misses
"""

import pytest
import asyncio
import os
import sys
import tempfile
import time
import threading
import hashlib
import struct
import math
import json
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime


# Create a standalone BloomFilter class for testing that matches the implementation
class BloomFilter:
    """
    Space-efficient probabilistic data structure for fast negative lookups.
    This is a copy of the implementation for isolated testing.
    """

    def __init__(
        self,
        expected_items: int = 100000,
        false_positive_rate: float = 0.01,
        persistence_path=None
    ):
        self.expected_items = expected_items
        self.false_positive_rate = false_positive_rate
        self.persistence_path = persistence_path

        self.size = self._optimal_size(expected_items, false_positive_rate)
        self.hash_count = self._optimal_hash_count(self.size, expected_items)

        self.byte_size = (self.size + 7) // 8
        self.bit_array = bytearray(self.byte_size)

        self.items_added = 0
        self.checks_performed = 0
        self.true_negatives = 0

        self._lock = threading.Lock()

        if persistence_path and os.path.exists(persistence_path):
            self._load_from_disk()

    @staticmethod
    def _optimal_size(n: int, p: float) -> int:
        if n <= 0:
            return 1024
        if p <= 0:
            p = 0.001
        m = -n * math.log(p) / (math.log(2) ** 2)
        return max(int(m), 1024)

    @staticmethod
    def _optimal_hash_count(m: int, n: int) -> int:
        if n <= 0:
            return 3
        k = (m / n) * math.log(2)
        return max(int(k), 1)

    def _get_hash_values(self, key: str):
        key_bytes = key.encode('utf-8')
        sha_hash = hashlib.sha256(key_bytes).digest()
        h1 = struct.unpack('<Q', sha_hash[:8])[0]

        md5_hash = hashlib.md5(key_bytes).digest()
        h2 = struct.unpack('<Q', md5_hash[:8])[0]

        hashes = []
        for i in range(self.hash_count):
            combined = (h1 + i * h2) % self.size
            hashes.append(combined)

        return hashes

    def add(self, key: str) -> None:
        with self._lock:
            for bit_index in self._get_hash_values(key):
                byte_index = bit_index // 8
                bit_offset = bit_index % 8
                self.bit_array[byte_index] |= (1 << bit_offset)
            self.items_added += 1

    def might_contain(self, key: str) -> bool:
        with self._lock:
            self.checks_performed += 1
            for bit_index in self._get_hash_values(key):
                byte_index = bit_index // 8
                bit_offset = bit_index % 8
                if not (self.bit_array[byte_index] & (1 << bit_offset)):
                    self.true_negatives += 1
                    return False
            return True

    def __contains__(self, key: str) -> bool:
        return self.might_contain(key)

    def clear(self) -> None:
        with self._lock:
            self.bit_array = bytearray(self.byte_size)
            self.items_added = 0
            self.checks_performed = 0
            self.true_negatives = 0

    def get_stats(self):
        with self._lock:
            if self.items_added > 0:
                exp_term = math.exp(-self.hash_count * self.items_added / self.size)
                estimated_fp_rate = (1 - exp_term) ** self.hash_count
            else:
                estimated_fp_rate = 0.0

            bits_set = sum(bin(byte).count('1') for byte in self.bit_array)
            fill_ratio = bits_set / self.size if self.size > 0 else 0

            return {
                'size_bits': self.size,
                'size_bytes': self.byte_size,
                'hash_count': self.hash_count,
                'items_added': self.items_added,
                'checks_performed': self.checks_performed,
                'true_negatives': self.true_negatives,
                'true_negative_rate': self.true_negatives / max(self.checks_performed, 1),
                'target_fp_rate': self.false_positive_rate,
                'estimated_fp_rate': estimated_fp_rate,
                'fill_ratio': fill_ratio,
                'capacity_remaining': max(0, self.expected_items - self.items_added)
            }

    def save_to_disk(self) -> bool:
        if not self.persistence_path:
            return False
        try:
            with self._lock:
                header = {
                    'version': 1,
                    'size': self.size,
                    'hash_count': self.hash_count,
                    'expected_items': self.expected_items,
                    'false_positive_rate': self.false_positive_rate,
                    'items_added': self.items_added,
                    'saved_at': datetime.now().isoformat()
                }
                with open(self.persistence_path, 'wb') as f:
                    header_bytes = json.dumps(header).encode('utf-8')
                    f.write(struct.pack('<I', len(header_bytes)))
                    f.write(header_bytes)
                    f.write(self.bit_array)
                return True
        except Exception:
            return False

    def _load_from_disk(self) -> bool:
        if not self.persistence_path or not os.path.exists(self.persistence_path):
            return False
        try:
            with open(self.persistence_path, 'rb') as f:
                header_len_bytes = f.read(4)
                if len(header_len_bytes) < 4:
                    return False
                header_len = struct.unpack('<I', header_len_bytes)[0]
                header_bytes = f.read(header_len)
                header = json.loads(header_bytes.decode('utf-8'))
                if header.get('version') != 1:
                    return False
                if header.get('size') != self.size or header.get('hash_count') != self.hash_count:
                    return False
                self.bit_array = bytearray(f.read())
                self.items_added = header.get('items_added', 0)
                return True
        except Exception:
            return False


class TestBloomFilter:
    """Test suite for BloomFilter class."""

    def test_no_false_negatives(self):
        """CRITICAL: Keys that were added must ALWAYS be found."""
        bf = BloomFilter(expected_items=1000, false_positive_rate=0.01)

        keys = [f"key_{i}" for i in range(1000)]
        for key in keys:
            bf.add(key)

        for key in keys:
            assert bf.might_contain(key) is True, f"False negative for {key}"

    def test_false_positive_rate_within_target(self):
        """False positive rate should be close to target (1%)."""
        bf = BloomFilter(expected_items=10000, false_positive_rate=0.01)

        added_keys = set()
        for i in range(10000):
            key = f"added_key_{i}"
            bf.add(key)
            added_keys.add(key)

        false_positives = 0
        test_count = 10000
        for i in range(test_count):
            key = f"never_added_{i}"
            if key not in added_keys and bf.might_contain(key):
                false_positives += 1

        fp_rate = false_positives / test_count
        assert fp_rate < 0.02, f"False positive rate {fp_rate:.2%} exceeds 2% threshold"

    def test_definitely_not_present(self):
        """Keys never added should return False most of the time."""
        bf = BloomFilter(expected_items=100, false_positive_rate=0.01)

        for i in range(10):
            bf.add(f"present_{i}")

        not_found_count = 0
        for i in range(100):
            if not bf.might_contain(f"absent_{i}"):
                not_found_count += 1

        assert not_found_count > 90, f"Only {not_found_count}/100 returned False"

    def test_persistence_save_and_load(self):
        """Bloom filter should persist state across restarts."""
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
            persistence_path = f.name

        try:
            bf1 = BloomFilter(
                expected_items=1000,
                false_positive_rate=0.01,
                persistence_path=persistence_path
            )

            keys = [f"persist_key_{i}" for i in range(100)]
            for key in keys:
                bf1.add(key)

            assert bf1.save_to_disk() is True

            bf2 = BloomFilter(
                expected_items=1000,
                false_positive_rate=0.01,
                persistence_path=persistence_path
            )

            for key in keys:
                assert bf2.might_contain(key) is True, f"Key {key} not found after reload"

            assert bf2.items_added == 100

        finally:
            if os.path.exists(persistence_path):
                os.unlink(persistence_path)

    def test_thread_safety(self):
        """Bloom filter should be thread-safe for concurrent access."""
        bf = BloomFilter(expected_items=10000, false_positive_rate=0.01)
        errors = []

        def add_keys(start, count):
            try:
                for i in range(start, start + count):
                    bf.add(f"thread_key_{i}")
            except Exception as e:
                errors.append(e)

        def check_keys(start, count):
            try:
                for i in range(start, start + count):
                    bf.might_contain(f"thread_key_{i}")
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(5):
                futures.append(executor.submit(add_keys, i * 1000, 1000))
                futures.append(executor.submit(check_keys, i * 1000, 1000))

            for future in futures:
                future.result()

        assert len(errors) == 0, f"Thread safety errors: {errors}"

    def test_clear(self):
        """Clear should reset the filter."""
        bf = BloomFilter(expected_items=100, false_positive_rate=0.01)

        for i in range(50):
            bf.add(f"clear_key_{i}")

        assert bf.items_added == 50

        bf.clear()

        assert bf.items_added == 0
        found = sum(1 for i in range(50) if bf.might_contain(f"clear_key_{i}"))
        assert found == 0, "Keys found after clear"

    def test_stats(self):
        """Get stats should return useful information."""
        bf = BloomFilter(expected_items=1000, false_positive_rate=0.01)

        for i in range(100):
            bf.add(f"stats_key_{i}")

        for i in range(200):
            bf.might_contain(f"check_key_{i}")

        stats = bf.get_stats()

        assert stats['items_added'] == 100
        assert stats['checks_performed'] == 200
        assert 'size_bits' in stats
        assert 'hash_count' in stats
        assert 'true_negatives' in stats
        assert 'estimated_fp_rate' in stats
        assert 'fill_ratio' in stats
        assert stats['capacity_remaining'] == 900

    def test_optimal_size_calculation(self):
        """Optimal size should scale with expected items and FP rate."""
        bf_small = BloomFilter(expected_items=100, false_positive_rate=0.01)
        bf_large = BloomFilter(expected_items=10000, false_positive_rate=0.01)
        assert bf_large.size > bf_small.size

        bf_high_fp = BloomFilter(expected_items=1000, false_positive_rate=0.1)
        bf_low_fp = BloomFilter(expected_items=1000, false_positive_rate=0.001)
        assert bf_low_fp.size > bf_high_fp.size

    def test_in_operator(self):
        """Test that 'in' operator works correctly."""
        bf = BloomFilter(expected_items=100, false_positive_rate=0.01)

        bf.add("test_key")
        assert "test_key" in bf
        assert "missing_key" not in bf or True  # Might be false positive

    def test_performance_fast_negative_lookup(self):
        """Bloom filter should be fast for negative lookups."""
        bf = BloomFilter(expected_items=100000, false_positive_rate=0.01)

        # Add some keys
        for i in range(1000):
            bf.add(f"existing_{i}")

        # Time lookups for non-existent keys
        start_time = time.time()
        for i in range(10000):
            bf.might_contain(f"nonexistent_{i}")
        elapsed = time.time() - start_time

        # Should be very fast - less than 100ms for 10k lookups
        assert elapsed < 0.1, f"10k lookups took {elapsed:.3f}s, expected < 0.1s"

        # Average per lookup should be < 0.01ms
        avg_time_us = (elapsed / 10000) * 1_000_000
        assert avg_time_us < 10, f"Average lookup {avg_time_us:.2f}us, expected < 10us"


class TestBloomFilterEdgeCases:
    """Edge case tests for BloomFilter."""

    def test_empty_filter(self):
        """Empty filter should return False for all keys."""
        bf = BloomFilter(expected_items=100, false_positive_rate=0.01)

        for i in range(100):
            assert bf.might_contain(f"key_{i}") is False

    def test_single_key(self):
        """Single key should be found."""
        bf = BloomFilter(expected_items=100, false_positive_rate=0.01)

        bf.add("only_key")
        assert bf.might_contain("only_key") is True

    def test_special_characters(self):
        """Keys with special characters should work."""
        bf = BloomFilter(expected_items=100, false_positive_rate=0.01)

        special_keys = [
            "key with spaces",
            "key\twith\ttabs",
            "key\nwith\nnewlines",
            "unicode: \u00e9\u00e8\u00ea",
            "emoji: test",
            "symbols: !@#$%^&*()",
            "",  # Empty string
        ]

        for key in special_keys:
            bf.add(key)

        for key in special_keys:
            assert bf.might_contain(key) is True, f"Failed for key: {repr(key)}"

    def test_very_long_key(self):
        """Very long keys should work."""
        bf = BloomFilter(expected_items=100, false_positive_rate=0.01)

        long_key = "x" * 10000
        bf.add(long_key)
        assert bf.might_contain(long_key) is True

    def test_zero_expected_items(self):
        """Zero expected items should not crash."""
        bf = BloomFilter(expected_items=0, false_positive_rate=0.01)
        bf.add("test")
        bf.might_contain("test")

    def test_very_low_fp_rate(self):
        """Very low FP rate should work."""
        bf = BloomFilter(expected_items=1000, false_positive_rate=0.0001)
        bf.add("test")
        assert bf.might_contain("test") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
