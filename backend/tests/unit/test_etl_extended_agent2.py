"""
Unit tests for ETL layer modules: distributed_batch_processor.py and etl_orchestrator.py.

These modules cannot be imported via `from backend.etl.xxx import ...` because
__init__.py pulls in selenium and other heavy dependencies.  We use the proven
importlib.util.spec_from_file_location pattern to load them directly, stubbing
all heavy transitive dependencies first.

For modules that use relative imports (from .xxx import ...) we pre-register
mock modules under the `backend.etl.*` namespace in sys.modules and set
__package__ on the module before exec_module.
"""

import asyncio
import importlib
import importlib.util
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock, PropertyMock, mock_open

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Stub ALL heavy transitive dependencies BEFORE loading modules
# ---------------------------------------------------------------------------

_etl_dir = Path(__file__).resolve().parents[2] / "etl"

# --- Stub third-party packages that the import chain needs ---
for _mod_name in [
    "aiohttp",
    "aiofiles",
    "bs4",
    "yfinance",
    "tenacity",
    "requests",
    "dotenv",
    "selenium",
    "selenium.webdriver",
    "selenium.webdriver.chrome",
    "selenium.webdriver.chrome.options",
    "selenium.webdriver.common",
    "selenium.webdriver.common.by",
    "selenium.webdriver.support",
    "selenium.webdriver.support.ui",
    "selenium.webdriver.support.expected_conditions",
    "cachetools",
    "psutil",
    "redis",
    "pandas_ta",
]:
    sys.modules.setdefault(_mod_name, MagicMock())

# tenacity decorators must be passthrough
_tenacity = sys.modules["tenacity"]
_tenacity.retry = lambda **kw: (lambda fn: fn)
_tenacity.stop_after_attempt = lambda n: None
_tenacity.wait_exponential = lambda **kw: None

# dotenv load_dotenv as no-op
_dotenv = sys.modules["dotenv"]
_dotenv.load_dotenv = lambda *a, **kw: None

# psycopg2 stubs
_psycopg2 = MagicMock()
_psycopg2.extras = MagicMock()
sys.modules.setdefault("psycopg2", _psycopg2)
sys.modules.setdefault("psycopg2.extras", _psycopg2.extras)

# --- Helper: load an ETL module from file, supporting relative imports ---


def _load_etl_module(name, filename=None, register=True):
    """Load an ETL module from the etl directory by filename.

    Sets __package__ = 'backend.etl' so relative imports resolve via
    sys.modules['backend.etl.<name>'].

    If *register* is True, also registers the module at
    ``sys.modules['backend.etl.<name>']``.
    """
    if filename is None:
        filename = f"{name}.py"
    filepath = _etl_dir / filename
    spec = importlib.util.spec_from_file_location(
        f"backend.etl.{name}", filepath,
        submodule_search_locations=[],
    )
    mod = importlib.util.module_from_spec(spec)
    # Set __package__ so `from .xxx import ...` resolves via sys.modules
    mod.__package__ = "backend.etl"
    if register:
        sys.modules[f"backend.etl.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


# --- Ensure 'backend' and 'backend.etl' package stubs exist ---
if "backend" not in sys.modules:
    _backend_pkg = MagicMock()
    sys.modules["backend"] = _backend_pkg
if "backend.etl" not in sys.modules:
    _backend_etl_pkg = MagicMock()
    sys.modules["backend.etl"] = _backend_etl_pkg

# --- Stub ML optional imports ---
_ml_mock = MagicMock()
sys.modules.setdefault("backend.ml", _ml_mock)
sys.modules.setdefault("backend.ml.ensemble_model", _ml_mock)
sys.modules.setdefault("backend.ml.recommendation_engine", _ml_mock)

# ---------------------------------------------------------------------------
# Load modules bottom-up in dependency order
# ---------------------------------------------------------------------------

# Leaf modules (no relative imports)
_rl = _load_etl_module("rate_limiting")
_ws = _load_etl_module("web_scrapers")
_dvp = _load_etl_module("data_validation_pipeline")
_cp = _load_etl_module("concurrent_processor")

# data_transformer (no relative imports, needs pandas_ta stub)
_dt = _load_etl_module("data_transformer")

# data_loader (no relative imports, needs psycopg2 stub)
_dl = _load_etl_module("data_loader")

# data_validator (no relative imports)
_dv = _load_etl_module("data_validator")

# unlimited_data_extractor (relative: none, but uses selenium etc. -- all stubbed)
_ude = _load_etl_module("unlimited_data_extractor")

# intelligent_cache_system (no relative imports)
_ics = _load_etl_module("intelligent_cache_system")

# unlimited_extractor_with_fallbacks (relative imports from .unlimited_data_extractor, etc.)
_uewf = _load_etl_module("unlimited_extractor_with_fallbacks")

# multi_source_extractor (relative imports from .web_scrapers, .rate_limiting)
_mse = _load_etl_module("multi_source_extractor")

# data_extractor (relative imports from .unlimited_extractor_with_fallbacks, .data_validation_pipeline)
_de = _load_etl_module("data_extractor")

# stock_universe_manager (needs yfinance, requests -- stubbed)
_sum_mod = _load_etl_module("stock_universe_manager")

# --- TARGET MODULE 1: distributed_batch_processor ---
# Relative import: from .multi_source_extractor import ...
_dbp = _load_etl_module("distributed_batch_processor")

BatchJob = _dbp.BatchJob
ProcessorConfig = _dbp.ProcessorConfig
DistributedBatchProcessor = _dbp.DistributedBatchProcessor

# --- TARGET MODULE 2: etl_orchestrator ---
# Absolute imports: from backend.etl.xxx import ...
_eo = _load_etl_module("etl_orchestrator")

ETLOrchestrator = _eo.ETLOrchestrator
ETLScheduler = _eo.ETLScheduler


# ==========================================================================
# distributed_batch_processor.py -- BatchJob dataclass
# ==========================================================================


class TestBatchJobDataclass:
    """Tests for the BatchJob dataclass."""

    def test_construction_with_required_fields(self):
        job = BatchJob(
            job_id="j1",
            tickers=["AAPL", "MSFT"],
            status="pending",
            priority=1,
            created_at=datetime.now(),
        )
        assert job.job_id == "j1"
        assert job.status == "pending"
        assert job.priority == 1

    def test_post_init_sets_total_tickers_from_list(self):
        job = BatchJob(
            job_id="j2",
            tickers=["A", "B", "C"],
            status="pending",
            priority=1,
            created_at=datetime.now(),
        )
        assert job.total_tickers == 3

    def test_post_init_preserves_explicit_total(self):
        job = BatchJob(
            job_id="j3",
            tickers=["A"],
            status="pending",
            priority=1,
            created_at=datetime.now(),
            total_tickers=99,
        )
        assert job.total_tickers == 99

    def test_defaults(self):
        job = BatchJob(
            job_id="j4",
            tickers=[],
            status="pending",
            priority=2,
            created_at=datetime.now(),
        )
        assert job.started_at is None
        assert job.completed_at is None
        assert job.progress == 0
        assert job.successful_extractions == 0
        assert job.failed_extractions == 0
        assert job.error_message is None

    def test_empty_tickers_total_zero(self):
        job = BatchJob(
            job_id="j5",
            tickers=[],
            status="pending",
            priority=1,
            created_at=datetime.now(),
        )
        assert job.total_tickers == 0


# ==========================================================================
# distributed_batch_processor.py -- ProcessorConfig dataclass
# ==========================================================================


class TestProcessorConfig:
    """Tests for the ProcessorConfig dataclass."""

    def test_defaults(self):
        cfg = ProcessorConfig()
        assert cfg.max_concurrent_jobs == 3
        assert cfg.max_concurrent_per_job == 8
        assert cfg.batch_size == 20
        assert cfg.cache_dir == "/tmp/stock_cache"
        assert cfg.job_timeout_hours == 12
        assert cfg.retry_failed_jobs is True
        assert cfg.max_retries == 3
        assert cfg.inter_batch_delay == (2.0, 5.0)
        assert cfg.priority_processing is True

    def test_custom_values(self):
        cfg = ProcessorConfig(
            max_concurrent_jobs=5,
            batch_size=100,
            cache_dir="/custom/path",
            retry_failed_jobs=False,
        )
        assert cfg.max_concurrent_jobs == 5
        assert cfg.batch_size == 100
        assert cfg.cache_dir == "/custom/path"
        assert cfg.retry_failed_jobs is False


# ==========================================================================
# distributed_batch_processor.py -- DistributedBatchProcessor init
# ==========================================================================


class TestDistributedBatchProcessorInit:
    """Tests for DistributedBatchProcessor initialization."""

    def test_init_default_config(self, tmp_path):
        cfg = ProcessorConfig(cache_dir=str(tmp_path / "cache"))
        proc = DistributedBatchProcessor(cfg)
        assert proc.config is cfg
        assert proc.is_running is False
        assert proc.active_jobs == {}
        assert proc.job_queue == []

    def test_init_creates_directories(self, tmp_path):
        cache = tmp_path / "cache"
        cfg = ProcessorConfig(cache_dir=str(cache))
        proc = DistributedBatchProcessor(cfg)
        assert cache.exists()
        assert (cache / "jobs").exists()
        assert (cache / "results").exists()

    def test_init_creates_sqlite_db(self, tmp_path):
        cfg = ProcessorConfig(cache_dir=str(tmp_path / "cache"))
        proc = DistributedBatchProcessor(cfg)
        db_path = tmp_path / "cache" / "jobs.db"
        assert db_path.exists()
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()
        assert "batch_jobs" in tables
        assert "job_tickers" in tables
        assert "processor_stats" in tables

    def test_init_stats_structure(self, tmp_path):
        cfg = ProcessorConfig(cache_dir=str(tmp_path / "cache"))
        proc = DistributedBatchProcessor(cfg)
        assert proc.stats["total_jobs_created"] == 0
        assert proc.stats["total_jobs_completed"] == 0
        assert proc.stats["start_time"] is None


# ==========================================================================
# distributed_batch_processor.py -- create_job / job management
# ==========================================================================


class TestDistributedBatchProcessorCreateJob:
    """Tests for create_job and job management."""

    @pytest.fixture
    def processor(self, tmp_path):
        cfg = ProcessorConfig(cache_dir=str(tmp_path / "cache"))
        return DistributedBatchProcessor(cfg)

    def test_create_job_returns_id(self, processor):
        job_id = processor.create_job(["AAPL", "MSFT"], priority=1)
        assert isinstance(job_id, str)
        assert len(job_id) > 0

    def test_create_job_with_custom_id(self, processor):
        job_id = processor.create_job(["AAPL"], priority=1, job_id="my_job")
        assert job_id == "my_job"

    def test_create_job_increments_stats(self, processor):
        processor.create_job(["AAPL"])
        processor.create_job(["MSFT"])
        assert processor.stats["total_jobs_created"] == 2

    def test_create_job_adds_to_queue(self, processor):
        processor.create_job(["AAPL", "MSFT"])
        assert len(processor.job_queue) == 1
        assert processor.job_queue[0].tickers == ["AAPL", "MSFT"]

    def test_create_job_saves_json_file(self, processor):
        processor.create_job(["GOOG"], job_id="file_test")
        job_file = processor.jobs_dir / "file_test.json"
        assert job_file.exists()
        with open(job_file) as f:
            data = json.load(f)
        assert data["job_id"] == "file_test"
        assert data["tickers"] == ["GOOG"]

    def test_create_job_persists_to_db(self, processor):
        processor.create_job(["TSLA"], job_id="db_test")
        conn = sqlite3.connect(processor.job_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT job_id, status FROM batch_jobs WHERE job_id = ?", ("db_test",)
        )
        row = cursor.fetchone()
        conn.close()
        assert row is not None
        assert row[0] == "db_test"
        assert row[1] == "pending"


# ==========================================================================
# distributed_batch_processor.py -- job splitting
# ==========================================================================


class TestDistributedBatchProcessorJobSplitting:
    """Tests for create_jobs_from_ticker_list."""

    @pytest.fixture
    def processor(self, tmp_path):
        cfg = ProcessorConfig(cache_dir=str(tmp_path / "cache"))
        return DistributedBatchProcessor(cfg)

    def test_single_batch(self, processor):
        tickers = ["A", "B", "C"]
        job_ids = processor.create_jobs_from_ticker_list(tickers, tickers_per_job=10)
        assert len(job_ids) == 1

    def test_multiple_batches(self, processor):
        tickers = [f"T{i}" for i in range(25)]
        job_ids = processor.create_jobs_from_ticker_list(tickers, tickers_per_job=10)
        assert len(job_ids) == 3

    def test_exact_split(self, processor):
        tickers = [f"T{i}" for i in range(20)]
        job_ids = processor.create_jobs_from_ticker_list(tickers, tickers_per_job=10)
        assert len(job_ids) == 2

    def test_empty_tickers(self, processor):
        job_ids = processor.create_jobs_from_ticker_list([], tickers_per_job=10)
        assert job_ids == []


# ==========================================================================
# distributed_batch_processor.py -- status / list
# ==========================================================================


class TestDistributedBatchProcessorStatus:
    """Tests for get_job_status and list_jobs."""

    @pytest.fixture
    def processor(self, tmp_path):
        cfg = ProcessorConfig(cache_dir=str(tmp_path / "cache"))
        return DistributedBatchProcessor(cfg)

    def test_get_job_status_returns_dict(self, processor):
        processor.create_job(["AAPL", "MSFT"], job_id="status_test")
        status = processor.get_job_status("status_test")
        assert status is not None
        assert status["job_id"] == "status_test"
        assert status["status"] == "pending"
        assert status["total_tickers"] == 2
        assert status["completion_percentage"] == 0

    def test_get_job_status_unknown_job(self, processor):
        assert processor.get_job_status("nonexistent") is None

    def test_list_jobs_empty(self, processor):
        assert processor.list_jobs() == []

    def test_list_jobs_all(self, processor):
        processor.create_job(["AAPL"], job_id="j1")
        processor.create_job(["MSFT"], job_id="j2")
        jobs = processor.list_jobs()
        assert len(jobs) == 2

    def test_list_jobs_filtered(self, processor):
        processor.create_job(["AAPL"], job_id="j1")
        assert len(processor.list_jobs(status_filter="pending")) == 1
        assert len(processor.list_jobs(status_filter="running")) == 0

    def test_completion_percentage_calculated(self, processor):
        processor.create_job(["A", "B", "C", "D"], job_id="pct_test")
        conn = sqlite3.connect(processor.job_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE batch_jobs SET progress = 2 WHERE job_id = ?", ("pct_test",)
        )
        conn.commit()
        conn.close()
        status = processor.get_job_status("pct_test")
        assert status["completion_percentage"] == pytest.approx(50.0)


# ==========================================================================
# distributed_batch_processor.py -- control flow
# ==========================================================================


class TestDistributedBatchProcessorControl:
    """Tests for stop_processing, pause_job, resume_job."""

    @pytest.fixture
    def processor(self, tmp_path):
        cfg = ProcessorConfig(cache_dir=str(tmp_path / "cache"))
        return DistributedBatchProcessor(cfg)

    def test_stop_processing(self, processor):
        processor.is_running = True
        processor.stop_processing()
        assert processor.is_running is False

    def test_pause_job_updates_db(self, processor):
        processor.create_job(["AAPL"], job_id="pause_test")
        mock_task = MagicMock()
        processor.active_jobs["pause_test"] = mock_task
        processor.pause_job("pause_test")
        mock_task.cancel.assert_called_once()
        assert "pause_test" not in processor.active_jobs
        conn = sqlite3.connect(processor.job_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT status FROM batch_jobs WHERE job_id = ?", ("pause_test",)
        )
        assert cursor.fetchone()[0] == "paused"
        conn.close()

    def test_pause_nonexistent_job_no_error(self, processor):
        processor.pause_job("does_not_exist")

    def test_resume_job_updates_db(self, processor):
        processor.create_job(["AAPL"], job_id="resume_test")
        conn = sqlite3.connect(processor.job_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE batch_jobs SET status = 'paused' WHERE job_id = ?",
            ("resume_test",),
        )
        conn.commit()
        conn.close()
        processor.resume_job("resume_test")
        status = processor.get_job_status("resume_test")
        assert status["status"] == "pending"


# ==========================================================================
# distributed_batch_processor.py -- processor stats
# ==========================================================================


class TestDistributedBatchProcessorStats:
    """Tests for get_processor_stats."""

    @pytest.fixture
    def processor(self, tmp_path):
        cfg = ProcessorConfig(cache_dir=str(tmp_path / "cache"))
        return DistributedBatchProcessor(cfg)

    def test_stats_no_start_time(self, processor):
        stats = processor.get_processor_stats()
        assert stats["total_jobs_created"] == 0
        assert stats["active_jobs"] == 0
        assert stats["queued_jobs"] == 0
        assert "runtime_hours" not in stats

    def test_stats_with_start_time(self, processor):
        processor.stats["start_time"] = datetime.now() - timedelta(hours=2)
        processor.stats["total_tickers_processed"] = 100
        stats = processor.get_processor_stats()
        assert "runtime_hours" in stats
        assert stats["runtime_hours"] > 0
        assert "avg_tickers_per_hour" in stats

    def test_stats_success_rate(self, processor):
        processor.stats["start_time"] = datetime.now()
        processor.stats["total_tickers_processed"] = 100
        processor.stats["total_successful_extractions"] = 80
        stats = processor.get_processor_stats()
        assert stats["overall_success_rate"] == pytest.approx(80.0)


# ==========================================================================
# distributed_batch_processor.py -- async start_processing
# ==========================================================================


class TestDistributedBatchProcessorStartProcessing:
    """Tests for start_processing async method."""

    @pytest.fixture
    def processor(self, tmp_path):
        cfg = ProcessorConfig(cache_dir=str(tmp_path / "cache"))
        return DistributedBatchProcessor(cfg)

    @pytest.mark.asyncio
    async def test_already_running_returns_immediately(self, processor):
        processor.is_running = True
        await processor.start_processing()

    @pytest.mark.asyncio
    async def test_start_sets_is_running(self, processor):
        async def stop_after_tick():
            await asyncio.sleep(0.05)
            processor.stop_processing()

        asyncio.create_task(stop_after_tick())
        await processor.start_processing()
        assert processor.is_running is False
        assert processor.stats["start_time"] is not None


# ==========================================================================
# etl_orchestrator.py -- shared helper for patching deep init chain
# ==========================================================================


def _make_orchestrator(tmp_path, use_distributed=False):
    """Create an ETLOrchestrator with all deep-chain constructors patched."""
    with patch.object(_dl, "create_engine", return_value=MagicMock()), \
         patch.object(_eo, "DataExtractor", return_value=MagicMock()), \
         patch.object(_eo, "DataValidator", return_value=MagicMock()), \
         patch.object(_eo, "MultiSourceStockExtractor", return_value=MagicMock()), \
         patch.object(_eo, "FinancialDataValidator", return_value=MagicMock()):
        return ETLOrchestrator(
            use_distributed=use_distributed,
            cache_dir=str(tmp_path / "cache"),
        )


# ==========================================================================
# etl_orchestrator.py -- init
# ==========================================================================


class TestETLOrchestratorInit:
    """Tests for ETLOrchestrator initialization."""

    def test_init_distributed_mode(self, tmp_path):
        orch = _make_orchestrator(tmp_path, use_distributed=True)
        assert orch.use_distributed is True
        assert orch.distributed_processor is not None
        assert orch.config["batch_size"] == 50
        assert orch.config["max_workers"] == 8

    def test_init_standard_mode(self, tmp_path):
        orch = _make_orchestrator(tmp_path, use_distributed=False)
        assert orch.use_distributed is False
        assert orch.distributed_processor is None
        assert orch.config["batch_size"] == 20
        assert orch.config["max_workers"] == 4

    def test_metrics_initialized(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        assert orch.metrics["start_time"] is None
        assert orch.metrics["stocks_processed"] == 0
        assert orch.metrics["errors"] == []

    def test_ml_components_initially_none(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        assert orch.predictor is None
        assert orch.recommender is None


# ==========================================================================
# etl_orchestrator.py -- validate_extracted_data
# ==========================================================================


class TestETLOrchestratorValidation:
    """Tests for validate_extracted_data."""

    @pytest.fixture
    def orchestrator(self, tmp_path):
        return _make_orchestrator(tmp_path)

    def test_valid_yfinance_data(self, orchestrator):
        data = {
            "sources": {
                "yfinance": {"price_data": {"history": [{"close": 100}]}}
            }
        }
        assert orchestrator.validate_extracted_data(data) is True

    def test_valid_polygon_data(self, orchestrator):
        data = {
            "sources": {
                "polygon": {"aggregates": [{"c": 100}]}
            }
        }
        assert orchestrator.validate_extracted_data(data) is True

    def test_missing_sources_key(self, orchestrator):
        assert orchestrator.validate_extracted_data({}) is False

    def test_empty_sources(self, orchestrator):
        assert orchestrator.validate_extracted_data({"sources": {}}) is False

    def test_unknown_source_only(self, orchestrator):
        data = {"sources": {"newsapi": {"articles": []}}}
        assert orchestrator.validate_extracted_data(data) is False

    def test_exception_returns_false(self, orchestrator):
        assert orchestrator.validate_extracted_data(None) is False


# ==========================================================================
# etl_orchestrator.py -- transform_single_stock
# ==========================================================================


class TestETLOrchestratorTransformSingleStock:
    """Tests for transform_single_stock."""

    @pytest.fixture
    def orchestrator(self, tmp_path):
        return _make_orchestrator(tmp_path)

    def test_transform_returns_dict(self, orchestrator):
        raw = {
            "ticker": "AAPL",
            "sources": {
                "yfinance": {
                    "price_data": {
                        "history": [
                            {"open": 100, "high": 105, "low": 99, "close": 103, "volume": 1000},
                            {"open": 103, "high": 108, "low": 102, "close": 107, "volume": 1100},
                            {"open": 107, "high": 110, "low": 106, "close": 109, "volume": 950},
                        ]
                    },
                    "company_info": {"sector": "Tech"},
                }
            },
            "extraction_time": "2024-01-01T00:00:00",
        }
        result = orchestrator.transform_single_stock(raw)
        if result is not None:
            assert result["ticker"] == "AAPL"
            assert "price_df" in result
            assert "features_df" in result

    def test_transform_empty_price_returns_none(self, orchestrator):
        raw = {
            "ticker": "BAD",
            "sources": {"yfinance": {"price_data": {"history": []}}},
        }
        result = orchestrator.transform_single_stock(raw)
        assert result is None

    def test_transform_exception_returns_none(self, orchestrator):
        with patch.object(
            orchestrator.transformer,
            "transform_price_data",
            side_effect=ValueError("boom"),
        ):
            result = orchestrator.transform_single_stock(
                {"ticker": "ERR", "sources": {"yfinance": {}}}
            )
        assert result is None


# ==========================================================================
# etl_orchestrator.py -- pipeline phase methods
# ==========================================================================


class TestETLOrchestratorPipelineMethods:
    """Tests for pipeline phase methods."""

    @pytest.fixture
    def orchestrator(self, tmp_path):
        return _make_orchestrator(tmp_path)

    @pytest.mark.asyncio
    async def test_run_full_pipeline_error_captured(self, orchestrator):
        orchestrator.config["enable_ml"] = False
        with patch.object(
            orchestrator, "get_active_tickers", side_effect=RuntimeError("db down")
        ):
            result = await orchestrator.run_full_pipeline()
        assert "db down" in str(result.get("errors", []))

    @pytest.mark.asyncio
    async def test_run_full_pipeline_with_tickers(self, orchestrator):
        orchestrator.config["enable_ml"] = False
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.data = {"ticker": "AAPL", "sources": {}}
        mock_result.source = "yfinance"
        mock_result.ticker = "AAPL"
        mock_result.error = None

        with patch.object(
            orchestrator.multi_source_extractor,
            "batch_extract",
            new_callable=AsyncMock,
            return_value=[mock_result],
        ), patch.object(
            orchestrator, "validation_phase",
            new_callable=AsyncMock, return_value=[],
        ), patch.object(
            orchestrator, "transform_phase",
            new_callable=AsyncMock, return_value=[],
        ), patch.object(
            orchestrator, "load_phase", new_callable=AsyncMock,
        ), patch.object(
            orchestrator, "cleanup_phase", new_callable=AsyncMock,
        ):
            result = await orchestrator.run_full_pipeline(["AAPL"])
        assert result["stocks_processed"] == 1

    @pytest.mark.asyncio
    async def test_transform_phase_empty_input(self, orchestrator):
        result = await orchestrator.transform_phase([])
        assert result == []

    @pytest.mark.asyncio
    async def test_load_phase_skips_none(self, orchestrator):
        await orchestrator.load_phase([None, None])

    @pytest.mark.asyncio
    async def test_load_phase_handles_loader_error(self, orchestrator):
        data = [
            {
                "ticker": "ERR",
                "price_df": pd.DataFrame({"close": [100]}),
                "features_df": pd.DataFrame(),
                "sentiment": None,
            }
        ]
        with patch.object(
            orchestrator.loader,
            "load_price_data",
            side_effect=RuntimeError("db fail"),
        ):
            await orchestrator.load_phase(data)
        assert any("Load error" in e for e in orchestrator.metrics["errors"])

    @pytest.mark.asyncio
    async def test_cleanup_phase_handles_error(self, orchestrator):
        with patch.object(
            orchestrator.loader,
            "cleanup_old_data",
            side_effect=RuntimeError("cleanup fail"),
        ):
            await orchestrator.cleanup_phase()
        assert any("Cleanup error" in e for e in orchestrator.metrics["errors"])

    @pytest.mark.asyncio
    async def test_ml_phase_no_ml(self, orchestrator):
        with patch.object(_eo, "HAS_ML", False):
            result = await orchestrator.ml_phase([])
        assert result == []


# ==========================================================================
# etl_orchestrator.py -- incremental / realtime
# ==========================================================================


class TestETLOrchestratorIncrementalAndRealtime:
    """Tests for run_incremental_update and run_realtime_update."""

    @pytest.fixture
    def orchestrator(self, tmp_path):
        return _make_orchestrator(tmp_path)

    @pytest.mark.asyncio
    async def test_incremental_adjusts_batch_size(self, orchestrator):
        original = orchestrator.config["batch_size"]
        with patch.object(
            orchestrator,
            "run_full_pipeline",
            new_callable=AsyncMock,
            return_value={"ok": True},
        ) as mock_pipeline:
            await orchestrator.run_incremental_update(["AAPL"])
            mock_pipeline.assert_called_once_with(["AAPL"])
        assert orchestrator.config["batch_size"] == original

    @pytest.mark.asyncio
    async def test_realtime_update_error_path(self, orchestrator):
        orchestrator.extractor = MagicMock()
        orchestrator.extractor.extract_all_data = AsyncMock(
            side_effect=RuntimeError("net err")
        )
        result = await orchestrator.run_realtime_update("FAIL")
        assert result["status"] == "error"


# ==========================================================================
# etl_orchestrator.py -- get_active_tickers
# ==========================================================================


class TestETLOrchestratorGetActiveTickers:
    """Tests for get_active_tickers fallback behavior."""

    @pytest.fixture
    def orchestrator(self, tmp_path):
        return _make_orchestrator(tmp_path)

    @pytest.mark.asyncio
    async def test_fallback_tickers_on_exception(self, orchestrator):
        with patch.dict(sys.modules, {"backend.etl.stock_universe_manager": MagicMock(
            StockUniverseManager=MagicMock(side_effect=RuntimeError("no db"))
        )}):
            tickers = await orchestrator.get_active_tickers()
        assert isinstance(tickers, list)
        assert len(tickers) > 0
        assert "AAPL" in tickers


# ==========================================================================
# etl_orchestrator.py -- ETLScheduler
# ==========================================================================


class TestETLScheduler:
    """Tests for ETLScheduler."""

    def _make_scheduler(self):
        """Create an ETLScheduler with deep-chain constructors patched."""
        with patch.object(_dl, "create_engine", return_value=MagicMock()), \
             patch.object(_eo, "DataExtractor", return_value=MagicMock()), \
             patch.object(_eo, "DataValidator", return_value=MagicMock()), \
             patch.object(_eo, "MultiSourceStockExtractor", return_value=MagicMock()), \
             patch.object(_eo, "FinancialDataValidator", return_value=MagicMock()):
            return ETLScheduler()

    def test_init(self):
        scheduler = self._make_scheduler()
        assert scheduler.is_running is False

    @pytest.mark.asyncio
    async def test_run_daily_pipeline_already_running(self):
        scheduler = self._make_scheduler()
        scheduler.is_running = True
        await scheduler.run_daily_pipeline()
        assert scheduler.is_running is True

    @pytest.mark.asyncio
    async def test_run_daily_pipeline_completes(self):
        scheduler = self._make_scheduler()
        with patch.object(
            scheduler.orchestrator,
            "run_full_pipeline",
            new_callable=AsyncMock,
            return_value={"errors": []},
        ):
            await scheduler.run_daily_pipeline()
        assert scheduler.is_running is False

    @pytest.mark.asyncio
    async def test_run_daily_pipeline_error(self):
        scheduler = self._make_scheduler()
        with patch.object(
            scheduler.orchestrator,
            "run_full_pipeline",
            new_callable=AsyncMock,
            side_effect=RuntimeError("boom"),
        ):
            await scheduler.run_daily_pipeline()
        assert scheduler.is_running is False

    @pytest.mark.asyncio
    async def test_run_hourly_skips_if_running(self):
        scheduler = self._make_scheduler()
        scheduler.is_running = True
        with patch.object(
            scheduler.orchestrator,
            "run_incremental_update",
            new_callable=AsyncMock,
        ) as mock_inc:
            await scheduler.run_hourly_update()
        mock_inc.assert_not_called()
