"""
Unit tests for backend/utils/resilient_pipeline.py

Tests cover:
- PipelineStage, FailureMode, RecoveryMode enums
- PipelineTask dataclass construction and defaults
- PipelineMetrics dataclass
- TaskExecutor: execute, metrics, fallback, sync/async callables
- DataQualityValidator: register, validate, quality metrics, error paths
- ResilientPipeline:
  - register_executor, register_data_validator
  - add_task (including duplicate rejection, dependency handling)
  - _classify_failure for all FailureMode variants
  - _determine_recovery_strategy (normal + critical override)
  - _calculate_retry_delay (exponential, linear, jitter, cap)
  - _generate_cache_key determinism
  - _get_cached_result (valid, expired, stale_ok)
  - _cache_result with eviction
  - _mark_task_completed and _mark_task_failed state transitions
  - _are_dependencies_satisfied
  - _handle_task_failure retry/skip/halt/fallback paths
  - _process_task: cache hit, missing executor, low quality data, success
  - get_health_status (idle, healthy, degraded, paused, warning)
  - get_task_status (found / not found)
  - _update_metrics aggregation
  - start/stop/pause/resume lifecycle
  - _check_dependent_tasks unblocking
"""

import asyncio
import hashlib
import json
import time
from collections import deque
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.utils.resilient_pipeline import (
    DataQualityValidator,
    FailureMode,
    PipelineMetrics,
    PipelineStage,
    PipelineTask,
    RecoveryMode,
    ResilientPipeline,
    TaskExecutor,
)
from backend.utils.exceptions import (
    AuthenticationException,
    ConfigurationException,
    DataQualityException,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(
    task_id: str = "t1",
    stage: PipelineStage = PipelineStage.PENDING,
    data: object = None,
    metadata: dict = None,
    retry_count: int = 0,
    max_retries: int = 3,
    priority: int = 1,
    dependencies: list = None,
) -> PipelineTask:
    return PipelineTask(
        task_id=task_id,
        stage=stage,
        data=data or {"symbol": "AAPL"},
        metadata=metadata or {"stage": "fetch"},
        created_at=datetime.now(),
        started_at=None,
        completed_at=None,
        retry_count=retry_count,
        max_retries=max_retries,
        last_error=None,
        priority=priority,
        dependencies=dependencies,
    )


def _make_pipeline(**kwargs) -> ResilientPipeline:
    defaults = dict(
        name="test_pipeline",
        max_concurrent_tasks=2,
        enable_checkpointing=False,
        enable_caching=True,
        cache_ttl=3600,
    )
    defaults.update(kwargs)
    return ResilientPipeline(**defaults)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------

class TestPipelineStageEnum:
    def test_all_values(self):
        expected = {"pending", "processing", "completed", "failed", "retrying", "skipped", "cached"}
        assert {s.value for s in PipelineStage} == expected

    def test_member_count(self):
        assert len(PipelineStage) == 7


class TestFailureModeEnum:
    def test_all_values(self):
        expected = {"transient", "permanent", "rate_limited", "data_corrupt", "resource_exhausted"}
        assert {f.value for f in FailureMode} == expected


class TestRecoveryModeEnum:
    def test_all_values(self):
        expected = {
            "retry_exponential", "retry_linear", "fallback_cache",
            "skip_and_continue", "halt_pipeline", "alternative_source",
        }
        assert {r.value for r in RecoveryMode} == expected


# ---------------------------------------------------------------------------
# PipelineTask dataclass
# ---------------------------------------------------------------------------

class TestPipelineTask:
    def test_defaults_dependencies_to_empty_list(self):
        task = _make_task(dependencies=None)
        assert task.dependencies == []

    def test_explicit_dependencies_preserved(self):
        task = _make_task(dependencies=["dep1", "dep2"])
        assert task.dependencies == ["dep1", "dep2"]

    def test_default_priority(self):
        task = _make_task()
        assert task.priority == 1

    def test_initial_timestamps(self):
        task = _make_task()
        assert task.started_at is None
        assert task.completed_at is None
        assert task.last_error is None


# ---------------------------------------------------------------------------
# PipelineMetrics dataclass
# ---------------------------------------------------------------------------

class TestPipelineMetrics:
    def test_construction(self):
        m = PipelineMetrics(
            total_tasks=100,
            completed_tasks=80,
            failed_tasks=5,
            retrying_tasks=3,
            pending_tasks=12,
            average_processing_time=0.5,
            success_rate=0.8,
            throughput_per_minute=10.0,
            error_rate=0.05,
            cache_hit_rate=0.2,
            last_updated=datetime.now(),
        )
        assert m.total_tasks == 100
        assert m.success_rate == 0.8


# ---------------------------------------------------------------------------
# TaskExecutor tests
# ---------------------------------------------------------------------------

class TestTaskExecutor:
    @pytest.mark.asyncio
    async def test_execute_async_function(self):
        async def processor(data):
            return {"result": data["symbol"]}

        executor = TaskExecutor(name="fetch", executor_func=processor)
        task = _make_task()
        result = await executor.execute(task)
        assert result == {"result": "AAPL"}
        assert executor.success_count == 1
        assert executor.failure_count == 0

    @pytest.mark.asyncio
    async def test_execute_sync_function(self):
        def processor(data):
            return data["symbol"].lower()

        executor = TaskExecutor(name="sync", executor_func=processor)
        task = _make_task()
        result = await executor.execute(task)
        assert result == "aapl"

    @pytest.mark.asyncio
    async def test_execute_failure_increments_count(self):
        async def bad_func(data):
            raise RuntimeError("boom")

        executor = TaskExecutor(name="bad", executor_func=bad_func)
        task = _make_task()
        with pytest.raises(RuntimeError, match="boom"):
            await executor.execute(task)
        assert executor.failure_count == 1
        assert executor.success_count == 0

    def test_get_metrics_no_executions(self):
        executor = TaskExecutor(name="idle", executor_func=lambda d: d)
        metrics = executor.get_metrics()
        assert metrics["executions"] == 0
        assert metrics["success_rate"] == 0
        assert metrics["avg_execution_time_ms"] == 0
        assert metrics["circuit_breaker"] is None

    @pytest.mark.asyncio
    async def test_get_metrics_after_executions(self):
        async def processor(data):
            return "ok"

        executor = TaskExecutor(name="m", executor_func=processor)
        await executor.execute(_make_task(task_id="t1"))
        await executor.execute(_make_task(task_id="t2"))
        metrics = executor.get_metrics()
        assert metrics["executions"] == 2
        assert metrics["successes"] == 2
        assert metrics["success_rate"] == 1.0
        assert metrics["avg_execution_time_ms"] >= 0

    @pytest.mark.asyncio
    async def test_fallback_returns_none(self):
        executor = TaskExecutor(name="fb", executor_func=lambda d: d)
        result = await executor._fallback_execution(_make_task())
        assert result is None


# ---------------------------------------------------------------------------
# DataQualityValidator tests
# ---------------------------------------------------------------------------

class TestDataQualityValidator:
    @pytest.mark.asyncio
    async def test_validate_unknown_type_passes(self):
        validator = DataQualityValidator()
        result = await validator.validate_data({"x": 1}, "unknown_type")
        assert result["is_valid"] is True
        assert result["quality_score"] == 1.0

    @pytest.mark.asyncio
    async def test_register_and_validate_sync(self):
        validator = DataQualityValidator()
        validator.register_validator("stock", lambda data: {
            "is_valid": data.get("price", 0) > 0,
            "quality_score": 0.9 if data.get("price", 0) > 0 else 0.1,
            "errors": [] if data.get("price", 0) > 0 else ["negative price"],
        })
        result = await validator.validate_data({"price": 100}, "stock")
        assert result["is_valid"] is True
        assert result["quality_score"] == 0.9

    @pytest.mark.asyncio
    async def test_register_and_validate_async(self):
        validator = DataQualityValidator()

        async def async_validator(data):
            return {"is_valid": True, "quality_score": 0.95, "errors": []}

        validator.register_validator("async_type", async_validator)
        result = await validator.validate_data({}, "async_type")
        assert result["quality_score"] == 0.95

    @pytest.mark.asyncio
    async def test_validator_exception_marks_invalid(self):
        validator = DataQualityValidator()
        validator.register_validator("bad", lambda d: (_ for _ in ()).throw(ValueError("oops")))
        result = await validator.validate_data({}, "bad")
        assert result["is_valid"] is False
        assert result["quality_score"] == 0.0
        assert any("Validation failed" in e for e in result["errors"])

    def test_quality_metrics_empty(self):
        validator = DataQualityValidator()
        metrics = validator.get_quality_metrics()
        assert metrics == {"message": "No validation data available"}

    @pytest.mark.asyncio
    async def test_quality_metrics_after_validations(self):
        validator = DataQualityValidator()
        validator.register_validator("ok", lambda d: {"is_valid": True, "quality_score": 0.9, "errors": []})
        validator.register_validator("bad", lambda d: {"is_valid": False, "quality_score": 0.3, "errors": ["err"]})

        await validator.validate_data({}, "ok")
        await validator.validate_data({}, "ok")
        await validator.validate_data({}, "bad")

        metrics = validator.get_quality_metrics()
        assert metrics["total_validations"] == 3
        assert metrics["recent_errors_1h"] == 1
        assert 0 < metrics["average_quality_score"] < 1.0


# ---------------------------------------------------------------------------
# ResilientPipeline: failure classification
# ---------------------------------------------------------------------------

class TestClassifyFailure:
    def setup_method(self):
        self.pipeline = _make_pipeline()

    def test_rate_limit_keyword(self):
        err = RuntimeError("Rate limit exceeded for API")
        assert self.pipeline._classify_failure(err) == FailureMode.RATE_LIMITED

    def test_throttle_keyword(self):
        err = RuntimeError("Request was throttled")
        assert self.pipeline._classify_failure(err) == FailureMode.RATE_LIMITED

    def test_timeout_keyword(self):
        err = RuntimeError("Connection timeout after 30s")
        assert self.pipeline._classify_failure(err) == FailureMode.TRANSIENT

    def test_connection_keyword(self):
        err = RuntimeError("Connection refused by server")
        assert self.pipeline._classify_failure(err) == FailureMode.TRANSIENT

    def test_data_quality_keyword(self):
        err = RuntimeError("Data quality check failed")
        assert self.pipeline._classify_failure(err) == FailureMode.DATA_CORRUPT

    def test_validation_keyword(self):
        err = RuntimeError("Validation error on field X")
        assert self.pipeline._classify_failure(err) == FailureMode.DATA_CORRUPT

    def test_memory_keyword(self):
        err = RuntimeError("Out of memory")
        assert self.pipeline._classify_failure(err) == FailureMode.RESOURCE_EXHAUSTED

    def test_resource_keyword(self):
        err = RuntimeError("Resource exhausted")
        assert self.pipeline._classify_failure(err) == FailureMode.RESOURCE_EXHAUSTED

    def test_authentication_exception_type(self):
        err = AuthenticationException("bad token")
        assert self.pipeline._classify_failure(err) == FailureMode.PERMANENT

    def test_configuration_exception_type(self):
        err = ConfigurationException("missing key")
        assert self.pipeline._classify_failure(err) == FailureMode.PERMANENT

    def test_unknown_error_defaults_transient(self):
        err = RuntimeError("something unexpected")
        assert self.pipeline._classify_failure(err) == FailureMode.TRANSIENT


# ---------------------------------------------------------------------------
# ResilientPipeline: recovery strategy
# ---------------------------------------------------------------------------

class TestDetermineRecoveryStrategy:
    def setup_method(self):
        self.pipeline = _make_pipeline()

    def test_transient_uses_exponential_retry(self):
        task = _make_task()
        assert self.pipeline._determine_recovery_strategy(FailureMode.TRANSIENT, task) == RecoveryMode.RETRY_EXPONENTIAL

    def test_rate_limited_uses_linear_retry(self):
        task = _make_task()
        assert self.pipeline._determine_recovery_strategy(FailureMode.RATE_LIMITED, task) == RecoveryMode.RETRY_LINEAR

    def test_data_corrupt_skips(self):
        task = _make_task()
        assert self.pipeline._determine_recovery_strategy(FailureMode.DATA_CORRUPT, task) == RecoveryMode.SKIP_AND_CONTINUE

    def test_resource_exhausted_linear_retry(self):
        task = _make_task()
        assert self.pipeline._determine_recovery_strategy(FailureMode.RESOURCE_EXHAUSTED, task) == RecoveryMode.RETRY_LINEAR

    def test_permanent_skips(self):
        task = _make_task()
        assert self.pipeline._determine_recovery_strategy(FailureMode.PERMANENT, task) == RecoveryMode.SKIP_AND_CONTINUE

    def test_critical_permanent_halts_pipeline(self):
        task = _make_task(metadata={"stage": "fetch", "critical": True})
        assert self.pipeline._determine_recovery_strategy(FailureMode.PERMANENT, task) == RecoveryMode.HALT_PIPELINE


# ---------------------------------------------------------------------------
# ResilientPipeline: retry delay calculation
# ---------------------------------------------------------------------------

class TestCalculateRetryDelay:
    def setup_method(self):
        self.pipeline = _make_pipeline()

    def test_transient_exponential_backoff(self):
        # retry_count=1: base=1.0 * 2^1 = 2.0, then jitter [0.8, 1.2]
        delay = self.pipeline._calculate_retry_delay(FailureMode.TRANSIENT, 1)
        assert 1.5 <= delay <= 2.5  # 2.0 * [0.8, 1.2]

    def test_rate_limited_linear_backoff(self):
        # retry_count=2: base=60.0 * 2 = 120.0, then jitter
        delay = self.pipeline._calculate_retry_delay(FailureMode.RATE_LIMITED, 2)
        assert 90 <= delay <= 150  # 120 * [0.8, 1.2]

    def test_delay_capped_at_300(self):
        # Large retry count should not exceed 300
        delay = self.pipeline._calculate_retry_delay(FailureMode.TRANSIENT, 20)
        assert delay <= 300.0

    def test_resource_exhausted_exponential(self):
        # base=30, retry_count=1: 30 * 2^1 = 60, jitter [0.8, 1.2]
        delay = self.pipeline._calculate_retry_delay(FailureMode.RESOURCE_EXHAUSTED, 1)
        assert 45 <= delay <= 75


# ---------------------------------------------------------------------------
# ResilientPipeline: cache key generation
# ---------------------------------------------------------------------------

class TestCacheKey:
    def setup_method(self):
        self.pipeline = _make_pipeline()

    def test_deterministic_for_same_input(self):
        task = _make_task()
        key1 = self.pipeline._generate_cache_key(task)
        key2 = self.pipeline._generate_cache_key(task)
        assert key1 == key2

    def test_different_for_different_data(self):
        t1 = _make_task(data={"symbol": "AAPL"})
        t2 = _make_task(data={"symbol": "MSFT"})
        assert self.pipeline._generate_cache_key(t1) != self.pipeline._generate_cache_key(t2)

    def test_returns_hex_string(self):
        task = _make_task()
        key = self.pipeline._generate_cache_key(task)
        assert isinstance(key, str)
        assert len(key) == 64  # SHA-256 hex digest


# ---------------------------------------------------------------------------
# ResilientPipeline: caching
# ---------------------------------------------------------------------------

class TestCaching:
    @pytest.mark.asyncio
    async def test_cache_hit(self):
        pipeline = _make_pipeline()
        await pipeline._cache_result("abc123", {"price": 150})
        result = await pipeline._get_cached_result("abc123")
        assert result == {"price": 150}

    @pytest.mark.asyncio
    async def test_cache_miss(self):
        pipeline = _make_pipeline()
        result = await pipeline._get_cached_result("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_expired(self):
        pipeline = _make_pipeline(cache_ttl=1)
        await pipeline._cache_result("key1", "data")
        # Manually backdate the timestamp
        pipeline.cache_timestamps["key1"] = datetime.now() - timedelta(seconds=10)
        result = await pipeline._get_cached_result("key1")
        assert result is None
        # Entry should be evicted
        assert "key1" not in pipeline.result_cache

    @pytest.mark.asyncio
    async def test_stale_ok_returns_expired_data(self):
        pipeline = _make_pipeline(cache_ttl=1)
        await pipeline._cache_result("key1", "stale_data")
        pipeline.cache_timestamps["key1"] = datetime.now() - timedelta(seconds=10)
        result = await pipeline._get_cached_result("key1", stale_ok=True)
        assert result == "stale_data"

    @pytest.mark.asyncio
    async def test_cache_eviction_on_overflow(self):
        pipeline = _make_pipeline()
        # Fill cache beyond 10000
        for i in range(10005):
            pipeline.result_cache[f"key_{i}"] = f"val_{i}"
            pipeline.cache_timestamps[f"key_{i}"] = datetime.now() - timedelta(seconds=10005 - i)
        await pipeline._cache_result("final", "value")
        # Should have evicted ~10% of oldest entries
        assert len(pipeline.result_cache) < 10006


# ---------------------------------------------------------------------------
# ResilientPipeline: add_task
# ---------------------------------------------------------------------------

class TestAddTask:
    @pytest.mark.asyncio
    async def test_add_task_success(self):
        pipeline = _make_pipeline()
        task = await pipeline.add_task("t1", {"symbol": "AAPL"}, "fetch")
        assert task.task_id == "t1"
        assert task.stage == PipelineStage.PENDING
        assert "t1" in pipeline.tasks

    @pytest.mark.asyncio
    async def test_add_duplicate_task_raises(self):
        pipeline = _make_pipeline()
        await pipeline.add_task("t1", {"symbol": "AAPL"}, "fetch")
        with pytest.raises(ValueError, match="already exists"):
            await pipeline.add_task("t1", {"symbol": "MSFT"}, "fetch")

    @pytest.mark.asyncio
    async def test_task_with_satisfied_dependencies_queued(self):
        pipeline = _make_pipeline()
        # First complete a dependency
        dep_task = _make_task(task_id="dep1", stage=PipelineStage.COMPLETED)
        pipeline.completed_tasks["dep1"] = dep_task
        task = await pipeline.add_task("t1", {}, "fetch", dependencies=["dep1"])
        assert task in pipeline.task_queue

    @pytest.mark.asyncio
    async def test_task_with_unsatisfied_dependencies_not_queued(self):
        pipeline = _make_pipeline()
        task = await pipeline.add_task("t1", {}, "fetch", dependencies=["dep_missing"])
        assert task not in pipeline.task_queue


# ---------------------------------------------------------------------------
# ResilientPipeline: dependency satisfaction
# ---------------------------------------------------------------------------

class TestDependencySatisfaction:
    @pytest.mark.asyncio
    async def test_no_dependencies_satisfied(self):
        pipeline = _make_pipeline()
        task = _make_task(dependencies=[])
        assert await pipeline._are_dependencies_satisfied(task) is True

    @pytest.mark.asyncio
    async def test_all_dependencies_completed(self):
        pipeline = _make_pipeline()
        pipeline.completed_tasks["dep1"] = _make_task(task_id="dep1")
        pipeline.completed_tasks["dep2"] = _make_task(task_id="dep2")
        task = _make_task(dependencies=["dep1", "dep2"])
        assert await pipeline._are_dependencies_satisfied(task) is True

    @pytest.mark.asyncio
    async def test_partial_dependencies_not_satisfied(self):
        pipeline = _make_pipeline()
        pipeline.completed_tasks["dep1"] = _make_task(task_id="dep1")
        task = _make_task(dependencies=["dep1", "dep2"])
        assert await pipeline._are_dependencies_satisfied(task) is False


# ---------------------------------------------------------------------------
# ResilientPipeline: mark completed / failed
# ---------------------------------------------------------------------------

class TestMarkTaskCompletedFailed:
    @pytest.mark.asyncio
    async def test_mark_completed(self):
        pipeline = _make_pipeline()
        task = _make_task(task_id="t1")
        pipeline.tasks["t1"] = task
        pipeline.processing_tasks["t1"] = task
        await pipeline._mark_task_completed(task, {"price": 150})
        assert task.stage == PipelineStage.COMPLETED
        assert task.completed_at is not None
        assert "t1" in pipeline.completed_tasks
        assert "t1" not in pipeline.processing_tasks
        assert task.metadata["result"] == {"price": 150}

    @pytest.mark.asyncio
    async def test_mark_completed_from_cache(self):
        pipeline = _make_pipeline()
        task = _make_task(task_id="t1")
        pipeline.tasks["t1"] = task
        await pipeline._mark_task_completed(task, "cached_val", from_cache=True)
        assert task.stage == PipelineStage.CACHED

    @pytest.mark.asyncio
    async def test_mark_failed(self):
        pipeline = _make_pipeline()
        task = _make_task(task_id="t1")
        pipeline.tasks["t1"] = task
        pipeline.processing_tasks["t1"] = task
        await pipeline._mark_task_failed(task, "Something broke")
        assert task.stage == PipelineStage.FAILED
        assert task.last_error == "Something broke"
        assert "t1" in pipeline.failed_tasks
        assert "t1" not in pipeline.processing_tasks


# ---------------------------------------------------------------------------
# ResilientPipeline: check_dependent_tasks
# ---------------------------------------------------------------------------

class TestCheckDependentTasks:
    @pytest.mark.asyncio
    async def test_unblocks_waiting_task(self):
        pipeline = _make_pipeline()
        dep_task = _make_task(task_id="dep1", stage=PipelineStage.COMPLETED)
        pipeline.completed_tasks["dep1"] = dep_task

        waiting = _make_task(task_id="waiter", dependencies=["dep1"])
        waiting.stage = PipelineStage.PENDING
        pipeline.tasks["waiter"] = waiting

        await pipeline._check_dependent_tasks("dep1")
        assert waiting in pipeline.task_queue


# ---------------------------------------------------------------------------
# ResilientPipeline: handle_task_failure
# ---------------------------------------------------------------------------

class TestHandleTaskFailure:
    @pytest.mark.asyncio
    async def test_skip_on_data_corrupt(self):
        pipeline = _make_pipeline()
        task = _make_task(task_id="t1", max_retries=3)
        pipeline.tasks["t1"] = task
        pipeline.processing_tasks["t1"] = task

        err = RuntimeError("Data quality bad")
        await pipeline._handle_task_failure(task, err)
        # DATA_CORRUPT => SKIP_AND_CONTINUE => mark failed immediately
        assert task.stage == PipelineStage.FAILED

    @pytest.mark.asyncio
    async def test_halt_pipeline_on_critical_permanent(self):
        pipeline = _make_pipeline()
        task = _make_task(
            task_id="t1",
            metadata={"stage": "fetch", "critical": True},
            max_retries=3,
        )
        pipeline.tasks["t1"] = task
        pipeline.processing_tasks["t1"] = task

        err = AuthenticationException("bad creds")
        await pipeline._handle_task_failure(task, err)
        assert pipeline.is_paused is True
        assert task.stage == PipelineStage.FAILED

    @pytest.mark.asyncio
    async def test_max_retries_exceeded_marks_failed(self):
        pipeline = _make_pipeline()
        task = _make_task(task_id="t1", retry_count=2, max_retries=3)
        pipeline.tasks["t1"] = task
        pipeline.processing_tasks["t1"] = task

        err = RuntimeError("Connection timeout")
        await pipeline._handle_task_failure(task, err)
        # retry_count becomes 3, equals max_retries => fail
        assert task.stage == PipelineStage.FAILED

    @pytest.mark.asyncio
    async def test_retry_scheduled_on_transient(self):
        pipeline = _make_pipeline()
        task = _make_task(task_id="t1", retry_count=0, max_retries=3)
        pipeline.tasks["t1"] = task
        pipeline.processing_tasks["t1"] = task

        with patch.object(pipeline, "_schedule_retry", new_callable=AsyncMock) as mock_retry:
            # Use a plain asyncio.Task mock via create_task patch
            with patch("asyncio.create_task") as mock_create:
                err = RuntimeError("Connection timeout")
                await pipeline._handle_task_failure(task, err)
                assert task.stage == PipelineStage.RETRYING
                mock_create.assert_called_once()


# ---------------------------------------------------------------------------
# ResilientPipeline: _process_task
# ---------------------------------------------------------------------------

class TestProcessTask:
    @pytest.mark.asyncio
    async def test_missing_executor_marks_failed(self):
        pipeline = _make_pipeline()
        task = _make_task(task_id="t1", metadata={"stage": "nonexistent"})
        pipeline.tasks["t1"] = task
        # Patch the decorator so it just calls through
        with patch("backend.utils.resilient_pipeline.with_error_handling", lambda **kw: lambda fn: fn):
            await pipeline._process_task(task, "worker_0")
        assert "t1" in pipeline.failed_tasks

    @pytest.mark.asyncio
    async def test_cache_hit_skips_execution(self):
        pipeline = _make_pipeline()
        task = _make_task(task_id="t1")
        pipeline.tasks["t1"] = task

        # Register executor that should NOT be called
        called = False
        async def executor_func(data):
            nonlocal called
            called = True
            return "fresh"

        pipeline.register_executor("fetch", executor_func)

        # Pre-populate cache
        cache_key = pipeline._generate_cache_key(task)
        await pipeline._cache_result(cache_key, "cached_value")

        await pipeline._process_task(task, "worker_0")
        assert called is False
        assert task.stage == PipelineStage.CACHED

    @pytest.mark.asyncio
    async def test_successful_execution_completes_task(self):
        pipeline = _make_pipeline()
        task = _make_task(task_id="t1")
        pipeline.tasks["t1"] = task

        async def executor_func(data):
            return {"price": 150}

        pipeline.register_executor("fetch", executor_func)
        await pipeline._process_task(task, "worker_0")
        assert task.stage == PipelineStage.COMPLETED
        assert task.metadata["result"] == {"price": 150}

    @pytest.mark.asyncio
    async def test_executor_failure_triggers_handle_failure(self):
        pipeline = _make_pipeline()
        task = _make_task(task_id="t1", max_retries=1)
        pipeline.tasks["t1"] = task
        pipeline.processing_tasks["t1"] = task

        async def bad_executor(data):
            raise RuntimeError("connection timeout")

        pipeline.register_executor("fetch", bad_executor)
        await pipeline._process_task(task, "worker_0")
        # retry_count should have incremented
        assert task.retry_count >= 1

    @pytest.mark.asyncio
    async def test_low_quality_data_raises_exception(self):
        pipeline = _make_pipeline()
        task = _make_task(task_id="t1", metadata={"stage": "fetch", "data_type": "stock"})
        pipeline.tasks["t1"] = task

        # Register validator that returns very low quality
        pipeline.register_data_validator("stock", lambda d: {
            "is_valid": False,
            "quality_score": 0.1,
            "errors": ["corrupt data"],
        })

        async def executor_func(data):
            return "ok"

        pipeline.register_executor("fetch", executor_func)
        await pipeline._process_task(task, "worker_0")
        # Should have failed due to DataQualityException
        assert task.retry_count >= 1 or "t1" in pipeline.failed_tasks


# ---------------------------------------------------------------------------
# ResilientPipeline: health status
# ---------------------------------------------------------------------------

class TestHealthStatus:
    def test_idle_when_no_tasks(self):
        pipeline = _make_pipeline()
        status = pipeline.get_health_status()
        assert status["status"] == "idle"
        assert status["is_running"] is False

    @pytest.mark.asyncio
    async def test_healthy_with_low_error_rate(self):
        pipeline = _make_pipeline()
        await pipeline.add_task("t1", {}, "fetch")
        pipeline.pipeline_metrics.error_rate = 0.05
        status = pipeline.get_health_status()
        assert status["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_warning_with_moderate_error_rate(self):
        pipeline = _make_pipeline()
        await pipeline.add_task("t1", {}, "fetch")
        pipeline.pipeline_metrics.error_rate = 0.15
        status = pipeline.get_health_status()
        assert status["status"] == "warning"

    @pytest.mark.asyncio
    async def test_degraded_with_high_error_rate(self):
        pipeline = _make_pipeline()
        await pipeline.add_task("t1", {}, "fetch")
        pipeline.pipeline_metrics.error_rate = 0.25
        status = pipeline.get_health_status()
        assert status["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_paused_status(self):
        pipeline = _make_pipeline()
        await pipeline.add_task("t1", {}, "fetch")
        pipeline.is_paused = True
        status = pipeline.get_health_status()
        assert status["status"] == "paused"

    def test_health_includes_executor_metrics(self):
        pipeline = _make_pipeline()
        pipeline.register_executor("fetch", lambda d: d)
        status = pipeline.get_health_status()
        assert "fetch" in status["executor_metrics"]

    def test_health_includes_data_quality_metrics(self):
        pipeline = _make_pipeline()
        status = pipeline.get_health_status()
        assert "data_quality_metrics" in status


# ---------------------------------------------------------------------------
# ResilientPipeline: get_task_status
# ---------------------------------------------------------------------------

class TestGetTaskStatus:
    @pytest.mark.asyncio
    async def test_existing_task(self):
        pipeline = _make_pipeline()
        await pipeline.add_task("t1", {"sym": "AAPL"}, "fetch", priority=2)
        status = await pipeline.get_task_status("t1")
        assert status is not None
        assert status["task_id"] == "t1"
        assert status["stage"] == "pending"
        assert status["priority"] == 2

    @pytest.mark.asyncio
    async def test_nonexistent_task(self):
        pipeline = _make_pipeline()
        status = await pipeline.get_task_status("missing")
        assert status is None


# ---------------------------------------------------------------------------
# ResilientPipeline: update_metrics
# ---------------------------------------------------------------------------

class TestUpdateMetrics:
    @pytest.mark.asyncio
    async def test_metrics_reflect_task_counts(self):
        pipeline = _make_pipeline()
        t1 = _make_task(task_id="t1")
        t2 = _make_task(task_id="t2", stage=PipelineStage.COMPLETED)
        t2.started_at = datetime.now() - timedelta(seconds=2)
        t2.completed_at = datetime.now()

        pipeline.tasks["t1"] = t1
        pipeline.tasks["t2"] = t2
        pipeline.completed_tasks["t2"] = t2
        pipeline.task_queue.append(t1)

        await pipeline._update_metrics()
        m = pipeline.pipeline_metrics
        assert m.total_tasks == 2
        assert m.completed_tasks == 1
        assert m.pending_tasks == 1
        assert m.success_rate == 0.5

    @pytest.mark.asyncio
    async def test_metrics_cache_hit_rate(self):
        pipeline = _make_pipeline()
        t1 = _make_task(task_id="t1", stage=PipelineStage.CACHED)
        t1.completed_at = datetime.now()
        t2 = _make_task(task_id="t2", stage=PipelineStage.COMPLETED)
        t2.completed_at = datetime.now()

        pipeline.tasks["t1"] = t1
        pipeline.tasks["t2"] = t2
        pipeline.completed_tasks["t1"] = t1
        pipeline.completed_tasks["t2"] = t2

        await pipeline._update_metrics()
        assert pipeline.pipeline_metrics.cache_hit_rate == 0.5


# ---------------------------------------------------------------------------
# ResilientPipeline: start / stop / pause / resume
# ---------------------------------------------------------------------------

class TestLifecycle:
    @pytest.mark.asyncio
    async def test_start_sets_running(self):
        pipeline = _make_pipeline()
        with patch.object(pipeline, "_load_checkpoint", new_callable=AsyncMock):
            # Start creates tasks, we need to cancel them quickly
            await pipeline.start()
            assert pipeline.is_running is True
            assert len(pipeline.worker_tasks) > 0
            await pipeline.stop()
            assert pipeline.is_running is False

    @pytest.mark.asyncio
    async def test_start_when_already_running_is_noop(self):
        pipeline = _make_pipeline()
        pipeline.is_running = True
        # Should return immediately without creating workers
        await pipeline.start()
        assert pipeline.worker_tasks == []

    @pytest.mark.asyncio
    async def test_stop_when_not_running_is_noop(self):
        pipeline = _make_pipeline()
        await pipeline.stop()  # should not raise

    @pytest.mark.asyncio
    async def test_pause_and_resume(self):
        pipeline = _make_pipeline()
        await pipeline.pause()
        assert pipeline.is_paused is True
        await pipeline.resume()
        assert pipeline.is_paused is False


# ---------------------------------------------------------------------------
# ResilientPipeline: schedule_retry
# ---------------------------------------------------------------------------

class TestScheduleRetry:
    @pytest.mark.asyncio
    async def test_schedule_retry_requeues_task(self):
        pipeline = _make_pipeline()
        task = _make_task(task_id="t1", stage=PipelineStage.RETRYING)
        pipeline.processing_tasks["t1"] = task
        # Use very short delay
        await pipeline._schedule_retry(task, 0.01)
        assert task.stage == PipelineStage.PENDING
        assert task in pipeline.task_queue
        assert "t1" not in pipeline.processing_tasks


# ---------------------------------------------------------------------------
# ResilientPipeline: register_executor
# ---------------------------------------------------------------------------

class TestRegisterExecutor:
    def test_registers_successfully(self):
        pipeline = _make_pipeline()
        pipeline.register_executor("fetch", lambda d: d, max_retries=5)
        assert "fetch" in pipeline.executors
        assert pipeline.executors["fetch"].name == "fetch"
        assert pipeline.executors["fetch"].max_retries == 5

    def test_register_multiple_executors(self):
        pipeline = _make_pipeline()
        pipeline.register_executor("fetch", lambda d: d)
        pipeline.register_executor("transform", lambda d: d)
        assert len(pipeline.executors) == 2


# ---------------------------------------------------------------------------
# ResilientPipeline: _get_next_task priority ordering
# ---------------------------------------------------------------------------

class TestGetNextTask:
    @pytest.mark.asyncio
    async def test_returns_highest_priority_first(self):
        pipeline = _make_pipeline()
        t_low = _make_task(task_id="low", priority=5)
        t_high = _make_task(task_id="high", priority=1)
        pipeline.task_queue.append(t_low)
        pipeline.task_queue.append(t_high)
        result = await pipeline._get_next_task()
        assert result.task_id == "high"

    @pytest.mark.asyncio
    async def test_returns_none_when_empty(self):
        pipeline = _make_pipeline()
        result = await pipeline._get_next_task()
        assert result is None

    @pytest.mark.asyncio
    async def test_skips_task_with_unsatisfied_deps(self):
        pipeline = _make_pipeline()
        t_blocked = _make_task(task_id="blocked", priority=1, dependencies=["missing"])
        t_ready = _make_task(task_id="ready", priority=5)
        pipeline.task_queue.append(t_blocked)
        pipeline.task_queue.append(t_ready)
        result = await pipeline._get_next_task()
        assert result.task_id == "ready"
