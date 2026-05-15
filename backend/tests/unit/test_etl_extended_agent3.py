"""
Unit tests for ETL extended modules:
- multi_source_extractor.py: ExtractionResult, SourcePriority, IntelligentSourceRouter,
  MultiSourceStockExtractor, extract_stocks_data
- unlimited_data_extractor.py: StockData, ExtractionResult, UnlimitedDataExtractor,
  BulkDataDownloader
- unlimited_extractor_with_fallbacks.py: SourceHealth, HealthMonitor, FallbackStrategy,
  FallbackManager, UnlimitedStockDataExtractor

Uses importlib.util.spec_from_file_location to bypass backend.etl.__init__.py
(which pulls in selenium and other heavy deps).
"""

import asyncio
import importlib
import importlib.util
import sys
import time
import types
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock, PropertyMock

import pytest

# ---------------------------------------------------------------------------
# Stub ALL heavy dependencies BEFORE loading any ETL modules.
# ---------------------------------------------------------------------------

_etl_dir = Path(__file__).resolve().parents[2] / "etl"

# --- Stub external packages that are imported at module level ---
_stubs = {}
for mod_name in [
    "yfinance",
    "aiohttp",
    "bs4",
    "bs4.BeautifulSoup",
    "selenium",
    "selenium.webdriver",
    "selenium.webdriver.chrome",
    "selenium.webdriver.chrome.options",
    "selenium.webdriver.common",
    "selenium.webdriver.common.by",
    "selenium.webdriver.support",
    "selenium.webdriver.support.ui",
    "selenium.webdriver.support.expected_conditions",
    "tenacity",
    "dotenv",
    "redis",
    "cachetools",
    "psutil",
    "aiofiles",
    "requests",
]:
    mock = MagicMock()
    _stubs[mod_name] = mock
    sys.modules.setdefault(mod_name, mock)

# tenacity decorators must be no-ops
_tenacity = sys.modules["tenacity"]
_tenacity.retry = lambda *a, **kw: (lambda fn: fn)
_tenacity.stop_after_attempt = MagicMock()
_tenacity.wait_exponential = MagicMock()

# dotenv.load_dotenv must be callable
sys.modules["dotenv"].load_dotenv = lambda *a, **kw: None

# bs4.BeautifulSoup must be available as a class
_bs4 = sys.modules["bs4"]
_bs4.BeautifulSoup = MagicMock

# selenium sub-modules wiring
_sel = sys.modules["selenium"]
_sel_wd = sys.modules["selenium.webdriver"]
_sel_opts = sys.modules["selenium.webdriver.chrome.options"]
_sel_opts.Options = MagicMock
_sel_wd.Chrome = MagicMock
_sel_by = sys.modules["selenium.webdriver.common.by"]
_sel_by.By = MagicMock()
_sel_ui = sys.modules["selenium.webdriver.support.ui"]
_sel_ui.WebDriverWait = MagicMock
_sel_ec = sys.modules["selenium.webdriver.support.expected_conditions"]
_sel_ec.EC = MagicMock()

# aiohttp stubs
_aiohttp = sys.modules["aiohttp"]
_aiohttp.ClientSession = MagicMock
_aiohttp.ClientTimeout = MagicMock
_aiohttp.TCPConnector = MagicMock

# ---------------------------------------------------------------------------
# 1. Load rate_limiting (dependency of multi_source_extractor)
# ---------------------------------------------------------------------------
_rl_spec = importlib.util.spec_from_file_location("rate_limiting", _etl_dir / "rate_limiting.py")
_rl = importlib.util.module_from_spec(_rl_spec)
sys.modules["rate_limiting"] = _rl
_rl_spec.loader.exec_module(_rl)

# ---------------------------------------------------------------------------
# 2. Load web_scrapers (dependency of multi_source_extractor)
# ---------------------------------------------------------------------------
_ws_spec = importlib.util.spec_from_file_location("web_scrapers", _etl_dir / "web_scrapers.py")
_ws = importlib.util.module_from_spec(_ws_spec)
sys.modules["web_scrapers"] = _ws
_ws_spec.loader.exec_module(_ws)

# ---------------------------------------------------------------------------
# 3. Load multi_source_extractor
# ---------------------------------------------------------------------------
# Patch relative imports: the module uses `from .web_scrapers import ...` and
# `from .rate_limiting import ...`. Since we loaded the modules standalone,
# we register them under the package path the relative imports expect.
# Create a fake package so relative imports resolve.
_fake_pkg = types.ModuleType("_etl_pkg")
_fake_pkg.__path__ = [str(_etl_dir)]
_fake_pkg.__package__ = "_etl_pkg"
sys.modules["_etl_pkg"] = _fake_pkg
sys.modules["_etl_pkg.web_scrapers"] = _ws
sys.modules["_etl_pkg.rate_limiting"] = _rl

# F-05-005: the extractor modules now do ``from .types import
# ExtractionResult``. Load the real types module under the synthetic
# package so relative imports resolve.
_types_spec = importlib.util.spec_from_file_location(
    "_etl_pkg.types",
    _etl_dir / "types.py",
    submodule_search_locations=[],
)
_types = importlib.util.module_from_spec(_types_spec)
_types.__package__ = "_etl_pkg"
sys.modules["_etl_pkg.types"] = _types
_types_spec.loader.exec_module(_types)

_mse_spec = importlib.util.spec_from_file_location(
    "_etl_pkg.multi_source_extractor",
    _etl_dir / "multi_source_extractor.py",
    submodule_search_locations=[],
)
_mse = importlib.util.module_from_spec(_mse_spec)
_mse.__package__ = "_etl_pkg"
sys.modules["_etl_pkg.multi_source_extractor"] = _mse
_mse_spec.loader.exec_module(_mse)

MSE_ExtractionResult = _mse.ExtractionResult
SourcePriority = _mse.SourcePriority
IntelligentSourceRouter = _mse.IntelligentSourceRouter
MultiSourceStockExtractor = _mse.MultiSourceStockExtractor
extract_stocks_data = _mse.extract_stocks_data

# ---------------------------------------------------------------------------
# 4. Load unlimited_data_extractor
# ---------------------------------------------------------------------------
# F-05-005: unlimited_data_extractor now does ``from .types import
# ExtractionResult``. Load under the synthetic ``_etl_pkg`` so the
# relative import resolves to the same stub already registered above.
_ude_spec = importlib.util.spec_from_file_location(
    "_etl_pkg.unlimited_data_extractor",
    _etl_dir / "unlimited_data_extractor.py",
    submodule_search_locations=[],
)
_ude = importlib.util.module_from_spec(_ude_spec)
_ude.__package__ = "_etl_pkg"
sys.modules["_etl_pkg.unlimited_data_extractor"] = _ude
sys.modules["unlimited_data_extractor"] = _ude  # legacy alias retained
_ude_spec.loader.exec_module(_ude)

StockData = _ude.StockData
UDE_ExtractionResult = _ude.ExtractionResult
UnlimitedDataExtractor = _ude.UnlimitedDataExtractor
BulkDataDownloader = _ude.BulkDataDownloader

# ---------------------------------------------------------------------------
# 5. Load stubs for unlimited_extractor_with_fallbacks deps
# ---------------------------------------------------------------------------
# intelligent_cache_system
_ics_mock = MagicMock()
sys.modules["_etl_pkg.intelligent_cache_system"] = _ics_mock
_ics_mock.IntelligentCacheManager = MagicMock

# concurrent_processor -- we already loaded the real one earlier in test_etl_modules
_cp_spec = importlib.util.spec_from_file_location("concurrent_processor", _etl_dir / "concurrent_processor.py")
_cp = importlib.util.module_from_spec(_cp_spec)
_cp_spec.loader.exec_module(_cp)
sys.modules["_etl_pkg.concurrent_processor"] = _cp

# data_validation_pipeline
_dvp_spec = importlib.util.spec_from_file_location("data_validation_pipeline", _etl_dir / "data_validation_pipeline.py")
_dvp = importlib.util.module_from_spec(_dvp_spec)
_dvp_spec.loader.exec_module(_dvp)
sys.modules["_etl_pkg.data_validation_pipeline"] = _dvp

# Register unlimited_data_extractor under the fake package too
sys.modules["_etl_pkg.unlimited_data_extractor"] = _ude

# ---------------------------------------------------------------------------
# 6. Load unlimited_extractor_with_fallbacks
# ---------------------------------------------------------------------------
_uewf_spec = importlib.util.spec_from_file_location(
    "_etl_pkg.unlimited_extractor_with_fallbacks",
    _etl_dir / "unlimited_extractor_with_fallbacks.py",
    submodule_search_locations=[],
)
_uewf = importlib.util.module_from_spec(_uewf_spec)
_uewf.__package__ = "_etl_pkg"
sys.modules["_etl_pkg.unlimited_extractor_with_fallbacks"] = _uewf
_uewf_spec.loader.exec_module(_uewf)

SourceHealth = _uewf.SourceHealth
FallbackStrategy = _uewf.FallbackStrategy
HealthMonitor = _uewf.HealthMonitor
FallbackManager = _uewf.FallbackManager
UEWF_UnlimitedStockDataExtractor = _uewf.UnlimitedStockDataExtractor

RequestPriority = _rl.RequestPriority


# ==========================================================================
# multi_source_extractor.py
# ==========================================================================


class TestMSEExtractionResult:
    """ExtractionResult dataclass from multi_source_extractor."""

    def test_default_timestamp(self):
        result = MSE_ExtractionResult(ticker="AAPL", success=True)
        assert result.timestamp is not None
        assert isinstance(result.timestamp, datetime)

    def test_explicit_fields(self):
        ts = datetime(2025, 6, 1)
        result = MSE_ExtractionResult(
            ticker="MSFT", success=False, data={"k": "v"},
            source="yfinance", error="timeout", timestamp=ts,
        )
        assert result.ticker == "MSFT"
        assert result.success is False
        assert result.data == {"k": "v"}
        assert result.source == "yfinance"
        assert result.error == "timeout"
        assert result.timestamp == ts

    def test_optional_defaults_none(self):
        result = MSE_ExtractionResult(ticker="X", success=True)
        assert result.data is None
        assert result.source is None
        assert result.error is None


class TestSourcePriority:
    def test_fields(self):
        sp = SourcePriority(name="yf", priority=2, success_rate=0.85)
        assert sp.name == "yf"
        assert sp.priority == 2
        assert sp.success_rate == 0.85
        assert sp.last_success is None
        assert sp.consecutive_failures == 0
        assert sp.enabled is True

    def test_disabled_flag(self):
        sp = SourcePriority(name="x", priority=1, success_rate=0.5, enabled=False)
        assert sp.enabled is False


class TestIntelligentSourceRouter:
    def test_default_sources(self):
        router = IntelligentSourceRouter()
        assert "yahoo_scraper" in router.source_priorities
        assert "yfinance" in router.source_priorities
        assert len(router.source_priorities) == 7

    def test_get_optimal_sources_default_limit(self):
        router = IntelligentSourceRouter()
        sources = router.get_optimal_sources("AAPL")
        assert len(sources) <= 3
        assert isinstance(sources, list)

    def test_get_optimal_sources_excludes_disabled(self):
        router = IntelligentSourceRouter()
        router.source_priorities["yahoo_scraper"].enabled = False
        sources = router.get_optimal_sources("AAPL", max_sources=10)
        assert "yahoo_scraper" not in sources

    def test_record_success_updates_stats(self):
        router = IntelligentSourceRouter()
        original_rate = router.source_priorities["yfinance"].success_rate
        router.record_success("yfinance")
        assert router.source_priorities["yfinance"].success_rate >= original_rate
        assert router.source_priorities["yfinance"].consecutive_failures == 0
        assert router.source_priorities["yfinance"].last_success is not None

    def test_record_failure_degrades_source(self):
        router = IntelligentSourceRouter()
        original_rate = router.source_priorities["yfinance"].success_rate
        router.record_failure("yfinance")
        assert router.source_priorities["yfinance"].success_rate < original_rate
        assert router.source_priorities["yfinance"].consecutive_failures == 1

    def test_consecutive_failures_disable_source(self):
        router = IntelligentSourceRouter()
        for _ in range(5):
            router.record_failure("yfinance")
        assert router.source_priorities["yfinance"].enabled is False

    def test_reset_source_re_enables(self):
        router = IntelligentSourceRouter()
        for _ in range(5):
            router.record_failure("yfinance")
        assert router.source_priorities["yfinance"].enabled is False
        router.reset_source("yfinance")
        assert router.source_priorities["yfinance"].enabled is True
        assert router.source_priorities["yfinance"].consecutive_failures == 0

    def test_record_on_unknown_source_is_noop(self):
        router = IntelligentSourceRouter()
        router.record_success("nonexistent_source")
        router.record_failure("nonexistent_source")
        router.reset_source("nonexistent_source")
        # should not raise

    def test_recent_success_boosts_score(self):
        router = IntelligentSourceRouter()
        # Give polygon a recent success
        router.source_priorities["polygon"].last_success = datetime.now()
        router.source_priorities["polygon"].success_rate = 0.95
        sources = router.get_optimal_sources("AAPL", max_sources=7)
        # polygon should be near the top now
        assert "polygon" in sources


class TestMultiSourceStockExtractor:
    @pytest.fixture
    def extractor(self, tmp_path):
        with patch.dict("os.environ", {}, clear=False):
            ext = MultiSourceStockExtractor(cache_dir=str(tmp_path), max_concurrent=5)
            return ext

    def test_init_creates_cache_dir(self, tmp_path):
        cache_dir = str(tmp_path / "test_cache")
        with patch.dict("os.environ", {}, clear=False):
            ext = MultiSourceStockExtractor(cache_dir=cache_dir)
            import os
            assert os.path.isdir(cache_dir)

    def test_init_sets_defaults(self, extractor):
        assert extractor.max_concurrent == 5
        assert isinstance(extractor.router, IntelligentSourceRouter)
        assert extractor.cache_expiry_hours == 4

    def test_set_high_priority_tickers(self, extractor):
        extractor.set_high_priority_tickers({"AAPL", "MSFT"})
        assert "AAPL" in extractor.high_priority_tickers
        assert "MSFT" in extractor.high_priority_tickers

    def test_set_critical_tickers(self, extractor):
        extractor.set_critical_tickers({"TSLA"})
        assert "TSLA" in extractor.critical_tickers

    def test_get_priority_for_ticker_critical(self, extractor):
        extractor.set_critical_tickers({"AAPL"})
        p = extractor._get_priority_for_ticker("AAPL")
        assert p == RequestPriority.CRITICAL

    def test_get_priority_for_ticker_high(self, extractor):
        extractor.set_high_priority_tickers({"MSFT"})
        p = extractor._get_priority_for_ticker("MSFT")
        assert p == RequestPriority.HIGH

    def test_get_priority_for_ticker_normal(self, extractor):
        p = extractor._get_priority_for_ticker("XYZ")
        assert p == RequestPriority.NORMAL

    def test_can_make_request_delegates_to_manager(self, extractor):
        # Just confirm it doesn't crash and returns bool
        result = extractor._can_make_request("yfinance")
        assert isinstance(result, bool)

    def test_get_rate_limit_stats_structure(self, extractor):
        stats = extractor.get_rate_limit_stats()
        assert "sources" in stats
        assert "summary" in stats

    def test_reset_failed_sources(self, extractor):
        for _ in range(5):
            extractor.router.record_failure("yfinance")
        assert extractor.router.source_priorities["yfinance"].enabled is False
        extractor.reset_failed_sources()
        assert extractor.router.source_priorities["yfinance"].enabled is True


# ==========================================================================
# unlimited_data_extractor.py
# ==========================================================================


class TestStockData:
    def test_basic_construction(self):
        sd = StockData(ticker="AAPL")
        assert sd.ticker == "AAPL"
        assert sd.price == 0.0
        assert sd.volume == 0
        assert sd.market_cap == 0.0
        assert sd.pe_ratio == 0.0
        assert sd.extra == {}
        assert sd.source == ""

    def test_timestamp_default(self):
        sd = StockData(ticker="AAPL")
        assert isinstance(sd.timestamp, datetime)

    def test_explicit_timestamp(self):
        ts = datetime(2025, 1, 15)
        sd = StockData(ticker="GOOG", timestamp=ts)
        assert sd.timestamp == ts

    def test_kwargs_populate_fields(self):
        sd = StockData(ticker="MSFT", price=350.0, volume=1000000, source="test_src")
        assert sd.price == 350.0
        assert sd.volume == 1000000
        assert sd.source == "test_src"

    def test_extra_kwargs_stored(self):
        sd = StockData(ticker="TSLA", custom_field="hello", another=42)
        assert sd.extra["custom_field"] == "hello"
        assert sd.extra["another"] == 42


class TestUDEExtractionResult:
    def test_default_timestamp(self):
        r = UDE_ExtractionResult(ticker="AAPL", success=True)
        assert r.timestamp is not None

    def test_explicit_fields(self):
        r = UDE_ExtractionResult(
            ticker="MSFT", success=False, data={"x": 1},
            source="yahoo", error="bad"
        )
        assert r.ticker == "MSFT"
        assert r.success is False
        assert r.source == "yahoo"
        assert r.error == "bad"
        assert r.data == {"x": 1}


class TestUnlimitedDataExtractor:
    def test_init_defaults(self):
        ext = UnlimitedDataExtractor()
        assert ext.cache == {}
        assert ext.cache_ttl == 300
        assert ext.session is None
        assert ext.driver is None

    def test_parse_market_cap_billions(self):
        ext = UnlimitedDataExtractor()
        result = ext._parse_market_cap("2.5B")
        assert result == 2.5e9

    def test_parse_market_cap_millions(self):
        ext = UnlimitedDataExtractor()
        result = ext._parse_market_cap("150M")
        assert result == 150e6

    def test_parse_market_cap_trillions(self):
        ext = UnlimitedDataExtractor()
        result = ext._parse_market_cap("3.1T")
        assert result == 3.1e12

    def test_parse_market_cap_thousands(self):
        ext = UnlimitedDataExtractor()
        result = ext._parse_market_cap("500K")
        assert result == 500e3

    def test_parse_market_cap_plain_number(self):
        ext = UnlimitedDataExtractor()
        result = ext._parse_market_cap("1234567")
        assert result == 1234567.0

    def test_parse_market_cap_invalid(self):
        ext = UnlimitedDataExtractor()
        result = ext._parse_market_cap("N/A")
        assert result == 0

    def test_parse_volume_millions(self):
        ext = UnlimitedDataExtractor()
        result = ext._parse_volume("5.2M")
        assert result == 5200000

    def test_parse_volume_thousands(self):
        ext = UnlimitedDataExtractor()
        result = ext._parse_volume("100K")
        assert result == 100000

    def test_parse_volume_plain(self):
        ext = UnlimitedDataExtractor()
        result = ext._parse_volume("5000")
        assert result == 5000

    def test_parse_volume_invalid(self):
        ext = UnlimitedDataExtractor()
        result = ext._parse_volume("bad")
        assert result == 0

    def test_parse_recent_filings_normal(self):
        ext = UnlimitedDataExtractor()
        filings = {
            "form": ["10-K", "10-Q", "8-K"],
            "filingDate": ["2025-01-01", "2024-10-01", "2024-07-01"],
            "accessionNumber": ["001", "002", "003"],
        }
        result = ext._parse_recent_filings(filings)
        assert len(result) == 3
        assert result[0]["form"] == "10-K"
        assert result[1]["filing_date"] == "2024-10-01"

    def test_parse_recent_filings_empty(self):
        ext = UnlimitedDataExtractor()
        result = ext._parse_recent_filings({})
        assert result == []

    def test_parse_recent_filings_max_10(self):
        ext = UnlimitedDataExtractor()
        filings = {
            "form": [f"form-{i}" for i in range(20)],
            "filingDate": [f"2025-01-{i+1:02d}" for i in range(20)],
            "accessionNumber": [f"acc-{i}" for i in range(20)],
        }
        result = ext._parse_recent_filings(filings)
        assert len(result) == 10

    def test_placeholder_aliases_are_none(self):
        assert _ude.YahooFinanceWebScraper is None
        assert _ude.SECEdgarExtractor is None
        assert _ude.IEXCloudFreeExtractor is None


class TestBulkDataDownloader:
    @pytest.mark.asyncio
    async def test_download_yahoo_bulk_data_returns_dataframe(self):
        import pandas as pd
        result = await BulkDataDownloader.download_yahoo_bulk_data(datetime.now())
        assert isinstance(result, pd.DataFrame)
        assert result.empty


# ==========================================================================
# unlimited_extractor_with_fallbacks.py
# ==========================================================================


class TestSourceHealth:
    def test_defaults(self):
        sh = SourceHealth(source_name="yahoo")
        assert sh.is_available is True
        assert sh.consecutive_failures == 0
        assert sh.success_count == 0
        assert sh.failure_count == 0

    def test_success_rate_no_attempts(self):
        sh = SourceHealth(source_name="yahoo")
        assert sh.success_rate == 0.0

    def test_success_rate_computed(self):
        sh = SourceHealth(source_name="yahoo", success_count=7, failure_count=3)
        assert sh.success_rate == pytest.approx(0.7)

    def test_is_healthy_default(self):
        sh = SourceHealth(source_name="yahoo")
        assert sh.is_healthy is True

    def test_is_healthy_false_when_unavailable(self):
        sh = SourceHealth(source_name="yahoo", is_available=False)
        assert sh.is_healthy is False

    def test_is_healthy_false_high_failures(self):
        sh = SourceHealth(source_name="yahoo", consecutive_failures=5)
        assert sh.is_healthy is False

    def test_is_healthy_false_high_error_rate(self):
        sh = SourceHealth(source_name="yahoo", error_rate=0.6)
        assert sh.is_healthy is False

    def test_is_healthy_true_edge(self):
        sh = SourceHealth(
            source_name="yahoo",
            is_available=True,
            consecutive_failures=4,
            error_rate=0.49,
        )
        assert sh.is_healthy is True


class TestFallbackStrategy:
    def test_defaults(self):
        fs = FallbackStrategy(name="test", priority=1, sources=["a", "b"])
        assert fs.max_attempts_per_source == 3
        assert fs.timeout_seconds == 30
        assert fs.min_success_rate == 0.3
        assert fs.enabled is True
        assert fs.description == ""


class TestHealthMonitor:
    def test_init_known_sources(self):
        hm = HealthMonitor(check_interval_minutes=1)
        assert "yahoo_scraper" in hm.source_health
        assert "sec_edgar" in hm.source_health
        assert len(hm.source_health) == 6

    def test_record_success(self):
        hm = HealthMonitor()
        hm.record_success("yahoo_scraper", response_time_ms=100.0)
        health = hm.source_health["yahoo_scraper"]
        assert health.success_count == 1
        assert health.consecutive_failures == 0
        assert health.avg_response_time_ms == 100.0
        assert health.last_success is not None

    def test_record_success_unknown_source_creates_entry(self):
        hm = HealthMonitor()
        hm.record_success("brand_new_source")
        assert "brand_new_source" in hm.source_health
        assert hm.source_health["brand_new_source"].success_count == 1

    def test_record_failure(self):
        hm = HealthMonitor()
        hm.record_failure("yahoo_scraper", "timeout")
        health = hm.source_health["yahoo_scraper"]
        assert health.failure_count == 1
        assert health.consecutive_failures == 1

    def test_record_failure_disables_after_5(self):
        hm = HealthMonitor()
        for _ in range(5):
            hm.record_failure("yahoo_scraper")
        assert hm.source_health["yahoo_scraper"].is_available is False

    def test_record_success_recovers_source(self):
        hm = HealthMonitor()
        for _ in range(5):
            hm.record_failure("yahoo_scraper")
        assert hm.source_health["yahoo_scraper"].is_available is False
        hm.record_success("yahoo_scraper")
        assert hm.source_health["yahoo_scraper"].is_available is True

    def test_get_healthy_sources(self):
        hm = HealthMonitor()
        healthy = hm.get_healthy_sources()
        assert len(healthy) == 6  # all start healthy

    def test_get_healthy_sources_after_failure(self):
        hm = HealthMonitor()
        for _ in range(5):
            hm.record_failure("yahoo_scraper")
        healthy = hm.get_healthy_sources()
        assert "yahoo_scraper" not in healthy

    def test_get_source_health(self):
        hm = HealthMonitor()
        h = hm.get_source_health("sec_edgar")
        assert h is not None
        assert h.source_name == "sec_edgar"

    def test_get_source_health_missing(self):
        hm = HealthMonitor()
        h = hm.get_source_health("nonexistent")
        assert h is None

    def test_get_health_summary_structure(self):
        hm = HealthMonitor()
        summary = hm.get_health_summary()
        assert "total_sources" in summary
        assert "healthy_sources" in summary
        assert "unhealthy_sources" in summary
        assert "health_rate" in summary
        assert "source_details" in summary
        assert summary["total_sources"] == 6
        assert summary["healthy_sources"] == 6

    def test_start_stop_monitoring(self):
        hm = HealthMonitor(check_interval_minutes=999)
        # Patch _monitoring_loop to avoid long-running thread
        with patch.object(hm, "_monitoring_loop"):
            hm.start_monitoring()
            assert hm.monitoring is True
            hm.stop_monitoring()
            assert hm.monitoring is False

    def test_start_monitoring_idempotent(self):
        hm = HealthMonitor(check_interval_minutes=999)
        with patch.object(hm, "_monitoring_loop"):
            hm.start_monitoring()
            hm.start_monitoring()  # should not create second thread
            assert hm.monitoring is True
            hm.stop_monitoring()

    def test_add_health_callback(self):
        hm = HealthMonitor()
        cb = MagicMock()
        hm.add_health_callback(cb)
        assert cb in hm.health_callbacks

    def test_avg_response_time_update(self):
        hm = HealthMonitor()
        hm.record_success("yahoo_scraper", response_time_ms=100.0)
        hm.record_success("yahoo_scraper", response_time_ms=200.0)
        avg = hm.source_health["yahoo_scraper"].avg_response_time_ms
        assert avg == pytest.approx(150.0)


class TestFallbackManager:
    def test_init_strategies(self):
        hm = HealthMonitor()
        fm = FallbackManager(hm)
        assert len(fm.strategies) == 4

    def test_strategy_names(self):
        hm = HealthMonitor()
        fm = FallbackManager(hm)
        names = [s.name for s in fm.strategies]
        assert "primary_scraping" in names
        assert "selenium_fallback" in names
        assert "official_data" in names
        assert "comprehensive_retry" in names

    def test_get_strategy_stats_empty(self):
        hm = HealthMonitor()
        fm = FallbackManager(hm)
        stats = fm.get_strategy_stats()
        assert stats == {}

    @pytest.mark.asyncio
    async def test_extract_with_fallback_all_fail(self):
        hm = HealthMonitor()
        fm = FallbackManager(hm)

        async def failing_extractor(ticker, source):
            return UDE_ExtractionResult(
                ticker=ticker, success=False, error="fail", source=source
            )

        # Patch asyncio.sleep in the fallback module to avoid inter-attempt delays
        with patch.object(_uewf.asyncio, "sleep", new_callable=AsyncMock):
            result = await fm.extract_with_fallback("AAPL", failing_extractor)
        assert result.success is False
        assert "All fallback strategies failed" in result.error

    @pytest.mark.asyncio
    async def test_extract_with_fallback_first_strategy_succeeds(self):
        hm = HealthMonitor()
        fm = FallbackManager(hm)

        async def succeeding_extractor(ticker, source):
            return UDE_ExtractionResult(
                ticker=ticker, success=True, data={"price": 150}, source=source
            )

        result = await fm.extract_with_fallback("AAPL", succeeding_extractor)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_extract_with_fallback_records_stats(self):
        hm = HealthMonitor()
        fm = FallbackManager(hm)

        async def succeeding_extractor(ticker, source):
            return UDE_ExtractionResult(
                ticker=ticker, success=True, data={"price": 150}, source=source
            )

        await fm.extract_with_fallback("AAPL", succeeding_extractor)
        stats = fm.get_strategy_stats()
        assert len(stats) > 0
        first_name = list(stats.keys())[0]
        assert stats[first_name]["successes"] >= 1
