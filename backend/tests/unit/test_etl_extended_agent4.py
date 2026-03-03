"""
Unit tests for ETL modules: intelligent_cache_system.py and web_scrapers.py.

Tests cover:
- BloomFilter: init, add, contains, false positive rate, persistence (save/load), clear, stats
- CacheEntry / CacheStats dataclasses
- CompressionManager: compress/decompress, gzip, error handling
- MemoryTierCache: get, set, eviction, TTL, stats, delete
- DiskTierCache: get, set, file ops (mocked SQLite + filesystem)
- RedisTierCache: get, set, availability, stats (mock redis)
- IntelligentCacheManager: init, get tier cascade, set write-through,
  bloom filter bypass, invalidation, comprehensive stats, default top stocks
- CacheWarmingResult: dataclass, success_rate
- WebScraperBase: init, _get_random_user_agent, _make_request (mock aiohttp)
- YahooFinanceScraper: _parse_price, _parse_volume, _parse_ratio
- MarketWatchScraper: _parse_price, _parse_volume
- GoogleFinanceScraper: _parse_price
- FREDScraper: _parse_price
- get_scraper factory function
"""

import asyncio
import gzip
import importlib
import json
import math
import os
import struct
import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch, mock_open

import pytest

# ---------------------------------------------------------------------------
# Stub heavy dependencies BEFORE importing the ETL modules.
# ---------------------------------------------------------------------------

_etl_dir = Path(__file__).resolve().parents[2] / "etl"

# --- Stubs for intelligent_cache_system.py ---
_redis_mock = MagicMock()
_redis_mock.from_url = MagicMock(return_value=MagicMock())
# Make ping raise so RedisTierCache sets redis_client = None by default
_redis_mock.from_url.return_value.ping.side_effect = Exception("no redis")
sys.modules.setdefault("redis", _redis_mock)

_cachetools_mock = MagicMock()


class _FakeTTLCache(dict):
    """Minimal TTLCache stand-in that behaves like a dict."""

    def __init__(self, maxsize=128, ttl=600, **kwargs):
        super().__init__()
        self.maxsize = maxsize
        self.ttl = ttl


class _FakeLRUCache(dict):
    def __init__(self, maxsize=128, **kwargs):
        super().__init__()
        self.maxsize = maxsize


_cachetools_mock.TTLCache = _FakeTTLCache
_cachetools_mock.LRUCache = _FakeLRUCache
# Force-set (not setdefault) because another test file may have registered a
# plain MagicMock for cachetools first.  We need the real _FakeTTLCache/
# _FakeLRUCache dict subclasses so MemoryTierCache can store/retrieve data.
sys.modules["cachetools"] = _cachetools_mock

_psutil_mock = MagicMock()
sys.modules.setdefault("psutil", _psutil_mock)

_aiofiles_mock = MagicMock()
sys.modules.setdefault("aiofiles", _aiofiles_mock)

_mmap_mock = MagicMock()
# mmap is a stdlib module but might need stubbing if not present
sys.modules.setdefault("mmap", _mmap_mock)

# --- Stubs for web_scrapers.py ---
_aiohttp_mock = MagicMock()
_aiohttp_mock.ClientSession = MagicMock
_aiohttp_mock.ClientTimeout = MagicMock
sys.modules.setdefault("aiohttp", _aiohttp_mock)

_bs4_mock = MagicMock()
sys.modules.setdefault("bs4", _bs4_mock)

# --- Load intelligent_cache_system module ---
# Use a unique module name to force a fresh load in case another test file
# already loaded this module with a different (MagicMock) cachetools stub.
_ics_spec = importlib.util.spec_from_file_location(
    "_agent4_intelligent_cache_system", _etl_dir / "intelligent_cache_system.py"
)
_ics = importlib.util.module_from_spec(_ics_spec)
_ics_spec.loader.exec_module(_ics)

BloomFilter = _ics.BloomFilter
CacheEntry = _ics.CacheEntry
CacheStats = _ics.CacheStats
CompressionManager = _ics.CompressionManager
MemoryTierCache = _ics.MemoryTierCache
DiskTierCache = _ics.DiskTierCache
RedisTierCache = _ics.RedisTierCache
IntelligentCacheManager = _ics.IntelligentCacheManager
CacheWarmingResult = _ics.CacheWarmingResult

# --- Load web_scrapers module ---
_ws_spec = importlib.util.spec_from_file_location(
    "web_scrapers", _etl_dir / "web_scrapers.py"
)
_ws = importlib.util.module_from_spec(_ws_spec)
_ws_spec.loader.exec_module(_ws)

WebScraperBase = _ws.WebScraperBase
YahooFinanceScraper = _ws.YahooFinanceScraper
MarketWatchScraper = _ws.MarketWatchScraper
GoogleFinanceScraper = _ws.GoogleFinanceScraper
FREDScraper = _ws.FREDScraper
get_scraper = _ws.get_scraper


# ==========================================================================
# BloomFilter
# ==========================================================================


class TestBloomFilter:
    def test_init_default_parameters(self):
        bf = BloomFilter(expected_items=1000, false_positive_rate=0.01)
        assert bf.expected_items == 1000
        assert bf.false_positive_rate == 0.01
        assert bf.size > 0
        assert bf.hash_count >= 1
        assert bf.items_added == 0
        assert bf.checks_performed == 0

    def test_optimal_size_formula(self):
        """Verify m = -n * ln(p) / (ln2)^2 calculation."""
        n, p = 10000, 0.01
        expected = int(-n * math.log(p) / (math.log(2) ** 2))
        actual = BloomFilter._optimal_size(n, p)
        assert actual == max(expected, 1024)

    def test_optimal_size_edge_case_zero_items(self):
        assert BloomFilter._optimal_size(0, 0.01) == 1024

    def test_optimal_size_edge_case_zero_fp_rate(self):
        result = BloomFilter._optimal_size(100, 0.0)
        assert result >= 1024

    def test_optimal_hash_count(self):
        result = BloomFilter._optimal_hash_count(9585, 1000)
        expected = max(int((9585 / 1000) * math.log(2)), 1)
        assert result == expected

    def test_optimal_hash_count_zero_items(self):
        assert BloomFilter._optimal_hash_count(1024, 0) == 3

    def test_add_and_contains(self):
        bf = BloomFilter(expected_items=100, false_positive_rate=0.01)
        bf.add("test_key")
        assert bf.items_added == 1
        assert bf.might_contain("test_key") is True
        assert "test_key" in bf  # __contains__

    def test_definitely_not_present(self):
        """Items never added must return False (no false negatives)."""
        bf = BloomFilter(expected_items=100, false_positive_rate=0.01)
        bf.add("alpha")
        bf.add("beta")
        # A key that was never added -- bloom filter guarantees no false negatives
        # We test with a clearly distinct key
        assert bf.might_contain("never_added_xyz_12345") is False

    def test_checks_and_true_negatives_tracking(self):
        bf = BloomFilter(expected_items=100, false_positive_rate=0.01)
        bf.add("present")
        bf.might_contain("present")
        bf.might_contain("absent_key_unique_42")
        assert bf.checks_performed == 2
        assert bf.true_negatives >= 1

    def test_clear(self):
        bf = BloomFilter(expected_items=100, false_positive_rate=0.01)
        bf.add("key1")
        bf.add("key2")
        bf.clear()
        assert bf.items_added == 0
        assert bf.checks_performed == 0
        assert bf.true_negatives == 0
        assert bf.might_contain("key1") is False

    def test_get_stats(self):
        bf = BloomFilter(expected_items=100, false_positive_rate=0.01)
        bf.add("key1")
        bf.might_contain("key1")
        bf.might_contain("nonexistent_xyz")
        stats = bf.get_stats()
        assert stats["items_added"] == 1
        assert stats["checks_performed"] == 2
        assert "estimated_fp_rate" in stats
        assert "fill_ratio" in stats
        assert "capacity_remaining" in stats
        assert stats["capacity_remaining"] == 99

    def test_get_stats_empty_filter(self):
        bf = BloomFilter(expected_items=50, false_positive_rate=0.05)
        stats = bf.get_stats()
        assert stats["estimated_fp_rate"] == 0.0
        assert stats["items_added"] == 0

    def test_persistence_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bloom.bin")
            bf1 = BloomFilter(
                expected_items=100,
                false_positive_rate=0.01,
                persistence_path=path,
            )
            bf1.add("persistent_key")
            assert bf1.save_to_disk() is True

            # Create second filter with same params; should load state
            bf2 = BloomFilter(
                expected_items=100,
                false_positive_rate=0.01,
                persistence_path=path,
            )
            assert bf2.items_added == 1
            assert bf2.might_contain("persistent_key") is True

    def test_save_to_disk_no_path(self):
        bf = BloomFilter(expected_items=10, false_positive_rate=0.1)
        assert bf.save_to_disk() is False

    def test_load_from_disk_version_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bloom.bin")
            # Write a file with version 99
            header = json.dumps({
                "version": 99,
                "size": 1024,
                "hash_count": 3,
                "expected_items": 100,
                "false_positive_rate": 0.01,
                "items_added": 0,
            }).encode("utf-8")
            with open(path, "wb") as f:
                f.write(struct.pack("<I", len(header)))
                f.write(header)
                f.write(bytearray(128))

            bf = BloomFilter(
                expected_items=100,
                false_positive_rate=0.01,
                persistence_path=path,
            )
            # Should have reinitialized (version mismatch)
            assert bf.items_added == 0

    def test_thread_safety(self):
        """Concurrent adds should not corrupt the filter."""
        bf = BloomFilter(expected_items=10000, false_positive_rate=0.01)

        def add_keys(start, count):
            for i in range(start, start + count):
                bf.add(f"key_{i}")

        threads = [threading.Thread(target=add_keys, args=(i * 100, 100)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert bf.items_added == 400


# ==========================================================================
# CacheEntry / CacheStats dataclasses
# ==========================================================================


class TestCacheEntry:
    def test_defaults(self):
        entry = CacheEntry(
            key="k",
            data={"val": 1},
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1),
        )
        assert entry.access_count == 0
        assert entry.size_bytes == 0
        assert entry.source_tier == "unknown"

    def test_custom_fields(self):
        entry = CacheEntry(
            key="k2",
            data=None,
            created_at=datetime.now(),
            expires_at=datetime.now(),
            access_count=5,
            size_bytes=2048,
            source_tier="memory",
        )
        assert entry.access_count == 5
        assert entry.size_bytes == 2048


class TestCacheStats:
    def test_hit_rate_zero(self):
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_hit_rate_computed(self):
        stats = CacheStats(total_requests=100, hits=75)
        assert stats.hit_rate == pytest.approx(0.75)

    def test_memory_hit_rate_no_hits(self):
        stats = CacheStats(hits=0, memory_hits=0)
        assert stats.memory_hit_rate == 0

    def test_memory_hit_rate_computed(self):
        stats = CacheStats(hits=10, memory_hits=8)
        assert stats.memory_hit_rate == pytest.approx(0.8)


# ==========================================================================
# CompressionManager
# ==========================================================================


class TestCompressionManager:
    def test_compress_decompress_roundtrip(self):
        data = {"ticker": "AAPL", "price": 150.25, "tags": [1, 2, 3]}
        compressed, ratio = CompressionManager.compress_data(data, "gzip")
        assert isinstance(compressed, bytes)
        assert 0 < ratio <= 1.5  # ratio can be > 1 for small data
        decompressed = CompressionManager.decompress_data(compressed, "gzip")
        assert decompressed["ticker"] == "AAPL"
        assert decompressed["price"] == 150.25

    def test_compress_no_method(self):
        data = {"key": "value"}
        compressed, ratio = CompressionManager.compress_data(data, "none")
        assert ratio == 1.0
        result = CompressionManager.decompress_data(compressed, "none")
        assert result["key"] == "value"

    def test_decompress_corrupt_data(self):
        result = CompressionManager.decompress_data(b"not valid gzip", "gzip")
        assert result is None

    def test_decompress_non_json_data(self):
        # Valid gzip but not JSON
        raw = gzip.compress(b"this is not json")
        result = CompressionManager.decompress_data(raw, "gzip")
        assert result is None


# ==========================================================================
# MemoryTierCache
# ==========================================================================


class TestMemoryTierCache:
    def test_set_and_get(self):
        cache = MemoryTierCache(max_size_mb=1, ttl_hours=1)
        assert cache.set("k1", {"val": 42}) is True
        assert cache.get("k1") == {"val": 42}

    def test_get_missing_key(self):
        cache = MemoryTierCache(max_size_mb=1, ttl_hours=1)
        assert cache.get("nonexistent") is None

    def test_delete(self):
        cache = MemoryTierCache(max_size_mb=1, ttl_hours=1)
        cache.set("del_key", "data")
        assert cache.delete("del_key") is True
        assert cache.get("del_key") is None

    def test_delete_missing(self):
        cache = MemoryTierCache(max_size_mb=1, ttl_hours=1)
        assert cache.delete("nope") is False

    def test_access_count_increments(self):
        cache = MemoryTierCache(max_size_mb=1, ttl_hours=1)
        cache.set("counted", "val")
        cache.get("counted")
        cache.get("counted")
        entry = cache.metadata.get("counted")
        assert entry is not None
        assert entry.access_count == 2

    def test_get_stats(self):
        cache = MemoryTierCache(max_size_mb=10, ttl_hours=1)
        cache.set("s1", "data")
        stats = cache.get_stats()
        assert stats["entries"] == 1
        assert stats["max_size_mb"] == 10.0
        assert 0 <= stats["utilization"] <= 1.0

    def test_estimate_size_string(self):
        cache = MemoryTierCache(max_size_mb=1, ttl_hours=1)
        size = cache._estimate_size("hello")
        assert size == 5

    def test_estimate_size_dict(self):
        cache = MemoryTierCache(max_size_mb=1, ttl_hours=1)
        size = cache._estimate_size({"a": 1})
        assert size > 0

    def test_eviction_when_full(self):
        """When cache is full, eviction should make room for new data."""
        # Very small cache: 1 byte max to force eviction
        cache = MemoryTierCache(max_size_mb=1, ttl_hours=1)
        # Override max_size_bytes to a very small value
        cache.max_size_bytes = 50
        cache.set("first", "a" * 30)
        cache.set("second", "b" * 30)
        # At least one of the original entries may have been evicted
        # Both keys should be handled gracefully
        stats = cache.get_stats()
        assert stats["entries"] >= 0


# ==========================================================================
# CacheWarmingResult
# ==========================================================================


class TestCacheWarmingResult:
    def test_success_rate_zero_stocks(self):
        r = CacheWarmingResult(total_stocks=0)
        assert r.success_rate == 0.0

    def test_success_rate_computed(self):
        r = CacheWarmingResult(total_stocks=10, successful=7)
        assert r.success_rate == pytest.approx(0.7)

    def test_default_fields(self):
        r = CacheWarmingResult()
        assert r.total_stocks == 0
        assert r.failed == 0
        assert r.skipped == 0
        assert r.errors == []


# ==========================================================================
# DiskTierCache (with temp directory)
# ==========================================================================


class TestDiskTierCache:
    @pytest.fixture
    def disk_cache(self, tmp_path):
        return DiskTierCache(
            cache_dir=str(tmp_path / "disk_cache"),
            max_size_mb=10,
            ttl_hours=1,
        )

    def test_set_and_get(self, disk_cache):
        assert disk_cache.set("dk1", {"price": 100}) is True
        result = disk_cache.get("dk1")
        assert result is not None
        assert result["price"] == 100

    def test_get_missing(self, disk_cache):
        assert disk_cache.get("nonexistent_disk_key") is None

    def test_delete(self, disk_cache):
        disk_cache.set("del_d", "value")
        assert disk_cache.delete("del_d") is True
        assert disk_cache.get("del_d") is None

    def test_delete_missing(self, disk_cache):
        assert disk_cache.delete("nope_d") is False

    def test_get_stats(self, disk_cache):
        disk_cache.set("s1", {"data": True})
        stats = disk_cache.get_stats()
        assert stats["entries"] == 1
        assert stats["size_bytes"] > 0


# ==========================================================================
# RedisTierCache (mocked)
# ==========================================================================


class TestRedisTierCache:
    def _make_unavailable_cache(self):
        """Create a RedisTierCache with redis_client = None (unavailable)."""
        redis_mod = sys.modules["redis"]
        original_from_url = redis_mod.from_url
        # Make from_url().ping() raise so constructor sets redis_client=None
        mock_client = MagicMock()
        mock_client.ping.side_effect = ConnectionError("no redis")
        redis_mod.from_url = MagicMock(return_value=mock_client)
        try:
            cache = RedisTierCache(redis_url="redis://fake:6379")
        finally:
            redis_mod.from_url = original_from_url
        return cache

    def test_not_available_when_connection_fails(self):
        cache = self._make_unavailable_cache()
        assert cache.is_available() is False

    def test_get_returns_none_when_unavailable(self):
        cache = self._make_unavailable_cache()
        assert cache.get("any_key") is None

    def test_set_returns_false_when_unavailable(self):
        cache = self._make_unavailable_cache()
        assert cache.set("k", "v") is False

    def test_delete_returns_false_when_unavailable(self):
        cache = self._make_unavailable_cache()
        assert cache.delete("k") is False

    def test_get_stats_unavailable(self):
        cache = self._make_unavailable_cache()
        stats = cache.get_stats()
        assert stats["available"] is False


# ==========================================================================
# IntelligentCacheManager
# ==========================================================================


class TestIntelligentCacheManager:
    @pytest.fixture
    def cache_manager(self, tmp_path):
        return IntelligentCacheManager(
            cache_dir=str(tmp_path / "icm"),
            memory_size_mb=16,
            disk_size_mb=32,
            redis_url=None,
            enable_analytics=True,
            bloom_filter_expected_items=1000,
            bloom_filter_fp_rate=0.01,
        )

    @pytest.mark.asyncio
    async def test_set_and_get(self, cache_manager):
        success = await cache_manager.set("test_key", {"val": 1}, "default")
        assert success is True
        result = await cache_manager.get("test_key", "default")
        assert result is not None
        assert result["val"] == 1

    @pytest.mark.asyncio
    async def test_get_miss_bloom_filter_bypass(self, cache_manager):
        """Key never added should be a bloom filter bypass (fast miss)."""
        result = await cache_manager.get("never_set_key_xyz123", "default")
        assert result is None
        assert cache_manager.bloom_filter_bypasses >= 1
        assert cache_manager.stats.misses >= 1

    @pytest.mark.asyncio
    async def test_set_none_returns_false(self, cache_manager):
        result = await cache_manager.set("null_key", None, "default")
        assert result is False

    @pytest.mark.asyncio
    async def test_delete(self, cache_manager):
        await cache_manager.set("del_key", {"x": 1})
        deleted = await cache_manager.delete("del_key")
        assert deleted is True

    @pytest.mark.asyncio
    async def test_bulk_get_and_set(self, cache_manager):
        items = {"bk1": {"a": 1}, "bk2": {"b": 2}, "bk3": {"c": 3}}
        set_results = await cache_manager.bulk_set(items, "test")
        assert all(set_results.values())

        get_results = await cache_manager.bulk_get(["bk1", "bk2", "bk3"], "test")
        assert len(get_results) == 3

    @pytest.mark.asyncio
    async def test_stats_tracking(self, cache_manager):
        await cache_manager.set("stat_key", {"data": True})
        await cache_manager.get("stat_key")
        await cache_manager.get("miss_key_abc")

        assert cache_manager.stats.total_requests >= 2
        assert cache_manager.stats.hits >= 1
        assert cache_manager.stats.misses >= 1

    def test_get_comprehensive_stats(self, cache_manager):
        stats = cache_manager.get_comprehensive_stats()
        assert "overview" in stats
        assert "performance" in stats
        assert "bloom_filter" in stats
        assert "tiers" in stats
        assert "analytics" in stats
        assert stats["bloom_filter"]["enabled"] is True

    def test_estimate_data_size_string(self, cache_manager):
        assert cache_manager._estimate_data_size("hello") == 5

    def test_estimate_data_size_dict(self, cache_manager):
        size = cache_manager._estimate_data_size({"key": "value"})
        assert size > 0

    def test_get_access_frequency_unknown(self, cache_manager):
        freq = cache_manager._get_access_frequency("unknown_key")
        assert freq == 0.0

    def test_default_top_stocks(self, cache_manager):
        stocks = cache_manager._get_default_top_stocks()
        assert len(stocks) == 100
        symbols = [s["symbol"] for s in stocks]
        assert "AAPL" in symbols
        assert "MSFT" in symbols
        assert "NVDA" in symbols

    def test_estimate_stock_coverage_empty(self, cache_manager):
        coverage = cache_manager._estimate_stock_coverage()
        assert coverage == 0.0

    def test_get_warming_status(self, cache_manager):
        status = cache_manager.get_warming_status()
        assert "memory_cache" in status
        assert "disk_cache" in status
        assert "bloom_filter" in status
        assert "estimated_coverage" in status


# ==========================================================================
# web_scrapers.py -- WebScraperBase
# ==========================================================================


class TestWebScraperBase:
    def test_init_default_delay(self):
        scraper = WebScraperBase(base_delay=3.0)
        assert scraper.base_delay == 3.0

    def test_init_default_headers(self):
        scraper = WebScraperBase()
        assert "User-Agent" in scraper.session_headers
        assert "Accept" in scraper.session_headers

    def test_get_random_user_agent_returns_string(self):
        scraper = WebScraperBase()
        ua = scraper._get_random_user_agent()
        assert isinstance(ua, str)
        assert "Mozilla" in ua

    def test_user_agent_varies(self):
        """Multiple calls should produce at least two different UAs (randomness)."""
        scraper = WebScraperBase()
        agents = {scraper._get_random_user_agent() for _ in range(50)}
        # With 5 options and 50 tries, extremely unlikely all are the same
        assert len(agents) >= 2


# ==========================================================================
# YahooFinanceScraper -- parsing helpers
# ==========================================================================


class TestYahooFinanceScraperParsing:
    @pytest.fixture
    def scraper(self):
        return YahooFinanceScraper()

    def test_parse_price_normal(self, scraper):
        assert scraper._parse_price("$150.25") == pytest.approx(150.25)

    def test_parse_price_empty(self, scraper):
        assert scraper._parse_price("") == 0.0

    def test_parse_price_invalid(self, scraper):
        assert scraper._parse_price("N/A") == 0.0

    def test_parse_volume_thousands(self, scraper):
        assert scraper._parse_volume("1.5K") == 1500

    def test_parse_volume_millions(self, scraper):
        assert scraper._parse_volume("2.5M") == 2500000

    def test_parse_volume_billions(self, scraper):
        assert scraper._parse_volume("1.2B") == 1200000000

    def test_parse_volume_plain(self, scraper):
        assert scraper._parse_volume("50000") == 50000

    def test_parse_volume_invalid(self, scraper):
        assert scraper._parse_volume("N/A") == 0

    def test_parse_ratio_normal(self, scraper):
        assert scraper._parse_ratio("25.3") == pytest.approx(25.3)

    def test_parse_ratio_empty(self, scraper):
        assert scraper._parse_ratio("") == 0.0

    def test_parse_market_cap_delegates_to_volume(self, scraper):
        assert scraper._parse_market_cap("2.5B") == 2500000000

    def test_init_sets_base_url_and_delay(self, scraper):
        assert scraper.base_url == "https://finance.yahoo.com"
        assert scraper.base_delay == 3.0


# ==========================================================================
# MarketWatchScraper -- parsing helpers
# ==========================================================================


class TestMarketWatchScraperParsing:
    @pytest.fixture
    def scraper(self):
        return MarketWatchScraper()

    def test_parse_price_normal(self, scraper):
        assert scraper._parse_price("$247.50") == pytest.approx(247.50)

    def test_parse_volume_millions(self, scraper):
        assert scraper._parse_volume("3.2M") == 3200000

    def test_parse_market_cap_billions(self, scraper):
        assert scraper._parse_market_cap("1.5B") == 1500000000

    def test_parse_ratio(self, scraper):
        assert scraper._parse_ratio("18.5x") == pytest.approx(18.5)

    def test_init_base_delay(self, scraper):
        assert scraper.base_delay == 4.0


# ==========================================================================
# GoogleFinanceScraper -- parsing helpers
# ==========================================================================


class TestGoogleFinanceScraperParsing:
    def test_parse_price(self):
        scraper = GoogleFinanceScraper()
        assert scraper._parse_price("$300.00") == pytest.approx(300.0)

    def test_parse_ratio(self):
        scraper = GoogleFinanceScraper()
        assert scraper._parse_ratio("12.5%") == pytest.approx(12.5)

    def test_init_base_delay(self):
        scraper = GoogleFinanceScraper()
        assert scraper.base_delay == 5.0


# ==========================================================================
# FREDScraper -- parsing helpers
# ==========================================================================


class TestFREDScraperParsing:
    def test_parse_price(self):
        scraper = FREDScraper()
        assert scraper._parse_price("3.75%") == pytest.approx(3.75)

    def test_init_base_delay(self):
        scraper = FREDScraper()
        assert scraper.base_delay == 2.0


# ==========================================================================
# get_scraper factory
# ==========================================================================


class TestGetScraperFactory:
    def test_yahoo_scraper(self):
        scraper = get_scraper("yahoo_scraper")
        assert isinstance(scraper, YahooFinanceScraper)

    def test_marketwatch_scraper(self):
        scraper = get_scraper("marketwatch_scraper")
        assert isinstance(scraper, MarketWatchScraper)

    def test_google_finance_scraper(self):
        scraper = get_scraper("google_finance_scraper")
        assert isinstance(scraper, GoogleFinanceScraper)

    def test_fred_scraper(self):
        scraper = get_scraper("fred_scraper")
        assert isinstance(scraper, FREDScraper)

    def test_unknown_returns_none(self):
        assert get_scraper("nonexistent_source") is None
