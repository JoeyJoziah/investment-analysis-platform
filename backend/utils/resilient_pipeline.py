"""
Resilient Data Pipeline with Fault Tolerance and Recovery
Advanced pipeline management for processing 6,000+ stocks with comprehensive error handling
"""

import asyncio
import time
import json
import hashlib
# SECURITY: Removed pickle import - using JSON for state serialization
from typing import Any, Dict, List, Optional, Callable, Union, TypeVar, Generic
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
from collections import deque, defaultdict
import threading
import logging
from pathlib import Path
import aiofiles
import random

from .enhanced_error_handling import with_error_handling, ErrorSeverity, ErrorCategory
from .advanced_circuit_breaker import EnhancedCircuitBreaker, AdaptiveThresholds
from .exceptions import *

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class PipelineStage(Enum):
    """Pipeline processing stages"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    SKIPPED = "skipped"
    CACHED = "cached"


class FailureMode(Enum):
    """Types of pipeline failures"""
    TRANSIENT = "transient"      # Temporary failures, retry possible
    PERMANENT = "permanent"      # Persistent failures, manual intervention needed
    RATE_LIMITED = "rate_limited"  # API rate limiting
    DATA_CORRUPT = "data_corrupt"  # Bad input data
    RESOURCE_EXHAUSTED = "resource_exhausted"  # System resources depleted


class RecoveryMode(Enum):
    """Recovery strategies for failed pipeline stages"""
    RETRY_EXPONENTIAL = "retry_exponential"
    RETRY_LINEAR = "retry_linear"
    FALLBACK_CACHE = "fallback_cache"
    SKIP_AND_CONTINUE = "skip_and_continue"
    HALT_PIPELINE = "halt_pipeline"
    ALTERNATIVE_SOURCE = "alternative_source"


@dataclass
class PipelineTask:
    """Individual task within the pipeline"""
    task_id: str
    stage: PipelineStage
    data: Any
    metadata: Dict[str, Any]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    retry_count: int
    max_retries: int
    last_error: Optional[str]
    priority: int = 1  # 1=highest, 5=lowest
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class PipelineMetrics:
    """Pipeline performance and health metrics"""
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    retrying_tasks: int
    pending_tasks: int
    average_processing_time: float
    success_rate: float
    throughput_per_minute: float
    error_rate: float
    cache_hit_rate: float
    last_updated: datetime


class TaskExecutor(Generic[T, R]):
    """Fault-tolerant task executor with circuit breaker protection"""
    
    def __init__(
        self,
        name: str,
        executor_func: Callable[[T], Union[R, Callable]],
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: bool = True,
        circuit_breaker_config: Optional[Dict] = None
    ):
        self.name = name
        self.executor_func = executor_func
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        
        # Circuit breaker protection
        if circuit_breaker_config:
            thresholds = AdaptiveThresholds(**circuit_breaker_config)
            self.circuit_breaker = EnhancedCircuitBreaker(
                name=f"executor_{name}",
                base_thresholds=thresholds,
                fallback_func=self._fallback_execution
            )
        else:
            self.circuit_breaker = None
        
        # Metrics
        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.total_execution_time = 0.0
        self._lock = threading.RLock()
    
    async def execute(self, task: PipelineTask) -> R:
        """Execute task with fault tolerance"""
        start_time = time.time()
        
        try:
            with self._lock:
                self.execution_count += 1
            
            if self.circuit_breaker:
                result = await self.circuit_breaker.call(
                    self._safe_execute,
                    task
                )
            else:
                result = await self._safe_execute(task)
            
            execution_time = time.time() - start_time
            
            with self._lock:
                self.success_count += 1
                self.total_execution_time += execution_time
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            with self._lock:
                self.failure_count += 1
                self.total_execution_time += execution_time
            
            logger.error(f"Task execution failed: {task.task_id}, error: {e}")
            raise e
    
    async def _safe_execute(self, task: PipelineTask) -> R:
        """Safely execute the task function"""
        if asyncio.iscoroutinefunction(self.executor_func):
            return await self.executor_func(task.data)
        else:
            return self.executor_func(task.data)
    
    async def _fallback_execution(self, task: PipelineTask) -> Optional[R]:
        """Fallback execution when circuit breaker is open"""
        logger.warning(f"Using fallback execution for task: {task.task_id}")
        # Could return cached data or simplified processing
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get executor performance metrics"""
        with self._lock:
            avg_execution_time = (
                self.total_execution_time / self.execution_count
                if self.execution_count > 0 else 0
            )
            
            success_rate = (
                self.success_count / self.execution_count
                if self.execution_count > 0 else 0
            )
            
            return {
                'name': self.name,
                'executions': self.execution_count,
                'successes': self.success_count,
                'failures': self.failure_count,
                'success_rate': round(success_rate, 3),
                'avg_execution_time_ms': round(avg_execution_time * 1000, 2),
                'circuit_breaker': (
                    self.circuit_breaker.get_comprehensive_metrics()
                    if self.circuit_breaker else None
                )
            }


class DataQualityValidator:
    """Validates data quality and handles corrupted data gracefully"""
    
    def __init__(self):
        self.validation_rules: Dict[str, Callable] = {}
        self.quality_scores: deque = deque(maxlen=1000)
        self.validation_errors: deque = deque(maxlen=1000)
    
    def register_validator(self, data_type: str, validator_func: Callable):
        """Register data validation function"""
        self.validation_rules[data_type] = validator_func
    
    async def validate_data(self, data: Any, data_type: str) -> Dict[str, Any]:
        """
        Validate data quality and return quality metrics
        
        Returns:
            Dict with validation results and quality score
        """
        validation_result = {
            'is_valid': True,
            'quality_score': 1.0,
            'errors': [],
            'warnings': [],
            'corrected_data': data
        }
        
        if data_type in self.validation_rules:
            try:
                validator = self.validation_rules[data_type]
                if asyncio.iscoroutinefunction(validator):
                    result = await validator(data)
                else:
                    result = validator(data)
                
                validation_result.update(result)
                
            except Exception as e:
                validation_result['is_valid'] = False
                validation_result['quality_score'] = 0.0
                validation_result['errors'].append(f"Validation failed: {e}")
        
        # Store quality metrics
        self.quality_scores.append(validation_result['quality_score'])
        if not validation_result['is_valid']:
            self.validation_errors.append({
                'timestamp': datetime.now(),
                'data_type': data_type,
                'errors': validation_result['errors']
            })
        
        return validation_result
    
    def get_quality_metrics(self) -> Dict[str, Any]:
        """Get data quality metrics"""
        if not self.quality_scores:
            return {'message': 'No validation data available'}
        
        avg_quality = sum(self.quality_scores) / len(self.quality_scores)
        recent_errors = [
            e for e in self.validation_errors
            if (datetime.now() - e['timestamp']).seconds < 3600
        ]
        
        return {
            'average_quality_score': round(avg_quality, 3),
            'total_validations': len(self.quality_scores),
            'recent_errors_1h': len(recent_errors),
            'validation_success_rate': round(
                len([s for s in self.quality_scores if s >= 0.8]) / len(self.quality_scores), 3
            )
        }


class ResilientPipeline:
    """
    Resilient data pipeline with comprehensive fault tolerance
    Designed to handle 6,000+ stocks with graceful degradation
    """
    
    def __init__(
        self,
        name: str,
        max_concurrent_tasks: int = 10,
        enable_checkpointing: bool = True,
        checkpoint_interval: int = 100,
        enable_caching: bool = True,
        cache_ttl: int = 3600
    ):
        self.name = name
        self.max_concurrent_tasks = max_concurrent_tasks
        self.enable_checkpointing = enable_checkpointing
        self.checkpoint_interval = checkpoint_interval
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl
        
        # Task management
        self.tasks: Dict[str, PipelineTask] = {}
        self.task_queue: deque = deque()
        self.processing_tasks: Dict[str, PipelineTask] = {}
        self.completed_tasks: Dict[str, PipelineTask] = {}
        self.failed_tasks: Dict[str, PipelineTask] = {}
        
        # Executors
        self.executors: Dict[str, TaskExecutor] = {}
        
        # Components
        self.data_validator = DataQualityValidator()
        
        # Caching
        self.result_cache: Dict[str, Any] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        
        # Metrics and monitoring
        self.pipeline_metrics = PipelineMetrics(
            total_tasks=0,
            completed_tasks=0,
            failed_tasks=0,
            retrying_tasks=0,
            pending_tasks=0,
            average_processing_time=0.0,
            success_rate=0.0,
            throughput_per_minute=0.0,
            error_rate=0.0,
            cache_hit_rate=0.0,
            last_updated=datetime.now()
        )
        
        # State management
        self.is_running = False
        self.is_paused = False
        self.worker_tasks: List[asyncio.Task] = []
        self.checkpoint_data: Dict = {}
        
        # Thread safety
        self._lock = asyncio.Lock()
        self._metrics_lock = threading.RLock()
    
    def register_executor(
        self,
        stage_name: str,
        executor_func: Callable,
        max_retries: int = 3,
        circuit_breaker_config: Optional[Dict] = None
    ):
        """Register task executor for a pipeline stage"""
        self.executors[stage_name] = TaskExecutor(
            name=stage_name,
            executor_func=executor_func,
            max_retries=max_retries,
            circuit_breaker_config=circuit_breaker_config
        )
        
        logger.info(f"Registered executor for stage: {stage_name}")
    
    def register_data_validator(self, data_type: str, validator_func: Callable):
        """Register data validation function"""
        self.data_validator.register_validator(data_type, validator_func)
    
    async def add_task(
        self,
        task_id: str,
        data: Any,
        stage_name: str,
        priority: int = 1,
        max_retries: int = 3,
        dependencies: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> PipelineTask:
        """Add task to pipeline"""
        async with self._lock:
            if task_id in self.tasks:
                raise ValueError(f"Task {task_id} already exists")
            
            task = PipelineTask(
                task_id=task_id,
                stage=PipelineStage.PENDING,
                data=data,
                metadata=metadata or {'stage': stage_name},
                created_at=datetime.now(),
                started_at=None,
                completed_at=None,
                retry_count=0,
                max_retries=max_retries,
                last_error=None,
                priority=priority,
                dependencies=dependencies or []
            )
            
            self.tasks[task_id] = task
            
            # Check if ready to queue (all dependencies satisfied)
            if await self._are_dependencies_satisfied(task):
                self.task_queue.append(task)
            
            await self._update_metrics()
            
            logger.debug(f"Added task: {task_id} with priority {priority}")
            return task
    
    async def add_batch_tasks(
        self,
        tasks_data: List[Dict[str, Any]],
        stage_name: str,
        batch_size: int = 100
    ) -> List[str]:
        """Add multiple tasks in batches for efficiency"""
        task_ids = []
        
        for i in range(0, len(tasks_data), batch_size):
            batch = tasks_data[i:i + batch_size]
            
            for task_data in batch:
                task_id = task_data.get('task_id', str(uuid.uuid4()))
                
                await self.add_task(
                    task_id=task_id,
                    data=task_data['data'],
                    stage_name=stage_name,
                    priority=task_data.get('priority', 1),
                    max_retries=task_data.get('max_retries', 3),
                    dependencies=task_data.get('dependencies'),
                    metadata=task_data.get('metadata')
                )
                
                task_ids.append(task_id)
            
            # Small delay between batches to prevent overwhelming
            await asyncio.sleep(0.01)
        
        logger.info(f"Added {len(task_ids)} tasks in batches of {batch_size}")
        return task_ids
    
    async def start(self) -> None:
        """Start the pipeline processing"""
        if self.is_running:
            logger.warning("Pipeline is already running")
            return
        
        self.is_running = True
        self.is_paused = False
        
        # Load checkpoint if available
        await self._load_checkpoint()
        
        # Start worker tasks
        self.worker_tasks = [
            asyncio.create_task(self._worker(f"worker_{i}"))
            for i in range(self.max_concurrent_tasks)
        ]
        
        # Start monitoring task
        self.worker_tasks.append(
            asyncio.create_task(self._monitor_pipeline())
        )
        
        # Start checkpointing task
        if self.enable_checkpointing:
            self.worker_tasks.append(
                asyncio.create_task(self._checkpoint_task())
            )
        
        logger.info(f"Started pipeline '{self.name}' with {self.max_concurrent_tasks} workers")
    
    async def stop(self) -> None:
        """Stop the pipeline gracefully"""
        if not self.is_running:
            return
        
        logger.info(f"Stopping pipeline '{self.name}'...")
        self.is_running = False
        
        # Cancel all worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        # Save final checkpoint
        if self.enable_checkpointing:
            await self._save_checkpoint()
        
        logger.info(f"Pipeline '{self.name}' stopped")
    
    async def pause(self) -> None:
        """Pause pipeline processing"""
        self.is_paused = True
        logger.info(f"Pipeline '{self.name}' paused")
    
    async def resume(self) -> None:
        """Resume pipeline processing"""
        self.is_paused = False
        logger.info(f"Pipeline '{self.name}' resumed")
    
    async def _worker(self, worker_name: str) -> None:
        """Worker task to process pipeline items"""
        logger.info(f"Worker {worker_name} started")
        
        while self.is_running:
            try:
                if self.is_paused:
                    await asyncio.sleep(1)
                    continue
                
                # Get next task
                task = await self._get_next_task()
                if not task:
                    await asyncio.sleep(0.1)
                    continue
                
                # Process task
                await self._process_task(task, worker_name)
                
            except asyncio.CancelledError:
                logger.info(f"Worker {worker_name} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
                await asyncio.sleep(1)
        
        logger.info(f"Worker {worker_name} stopped")
    
    async def _get_next_task(self) -> Optional[PipelineTask]:
        """Get next task from queue with priority ordering"""
        async with self._lock:
            if not self.task_queue:
                return None
            
            # Sort by priority (lower number = higher priority)
            sorted_tasks = sorted(self.task_queue, key=lambda t: (t.priority, t.created_at))
            
            for task in sorted_tasks:
                if await self._are_dependencies_satisfied(task):
                    self.task_queue.remove(task)
                    return task
            
            return None
    
    async def _are_dependencies_satisfied(self, task: PipelineTask) -> bool:
        """Check if all task dependencies are satisfied"""
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
        return True
    
    @with_error_handling(service="resilient_pipeline", operation="process_task")
    async def _process_task(self, task: PipelineTask, worker_name: str) -> None:
        """Process a single task with comprehensive error handling"""
        stage_name = task.metadata.get('stage')
        if not stage_name or stage_name not in self.executors:
            await self._mark_task_failed(task, f"No executor found for stage: {stage_name}")
            return
        
        # Check cache first
        if self.enable_caching:
            cache_key = self._generate_cache_key(task)
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                await self._mark_task_completed(task, cached_result, from_cache=True)
                return
        
        # Mark task as processing
        async with self._lock:
            task.stage = PipelineStage.PROCESSING
            task.started_at = datetime.now()
            self.processing_tasks[task.task_id] = task
        
        executor = self.executors[stage_name]
        
        try:
            # Validate input data
            data_type = task.metadata.get('data_type', 'generic')
            validation_result = await self.data_validator.validate_data(task.data, data_type)
            
            if not validation_result['is_valid']:
                if validation_result['quality_score'] < 0.5:
                    raise DataQualityException(
                        f"Data quality too low: {validation_result['quality_score']}",
                        data_source=stage_name,
                        quality_score=validation_result['quality_score']
                    )
                else:
                    # Use corrected data if quality is acceptable
                    task.data = validation_result['corrected_data']
                    logger.warning(f"Using corrected data for task {task.task_id}")
            
            # Execute task
            result = await executor.execute(task)
            
            # Cache result
            if self.enable_caching and result:
                cache_key = self._generate_cache_key(task)
                await self._cache_result(cache_key, result)
            
            # Mark task as completed
            await self._mark_task_completed(task, result)
            
            logger.debug(f"Task {task.task_id} completed by {worker_name}")
            
        except Exception as e:
            # Handle task failure
            await self._handle_task_failure(task, e)
    
    async def _handle_task_failure(self, task: PipelineTask, error: Exception) -> None:
        """Handle task failure with intelligent retry logic"""
        task.retry_count += 1
        task.last_error = str(error)
        
        # Determine failure mode and recovery strategy
        failure_mode = self._classify_failure(error)
        recovery_mode = self._determine_recovery_strategy(failure_mode, task)
        
        logger.warning(
            f"Task {task.task_id} failed (attempt {task.retry_count}/{task.max_retries}): "
            f"{error}, failure_mode={failure_mode.value}, recovery={recovery_mode.value}"
        )
        
        if recovery_mode == RecoveryMode.HALT_PIPELINE:
            logger.critical(f"Halting pipeline due to critical failure in task {task.task_id}")
            await self.pause()
            await self._mark_task_failed(task, f"Critical failure: {error}")
            return
        
        elif recovery_mode == RecoveryMode.SKIP_AND_CONTINUE:
            await self._mark_task_failed(task, f"Skipping task due to: {error}")
            return
        
        elif task.retry_count >= task.max_retries:
            if recovery_mode == RecoveryMode.FALLBACK_CACHE:
                # Try to use cached data
                cache_key = self._generate_cache_key(task)
                cached_result = await self._get_cached_result(cache_key, stale_ok=True)
                if cached_result:
                    logger.info(f"Using stale cache for failed task {task.task_id}")
                    await self._mark_task_completed(task, cached_result, from_cache=True)
                    return
            
            await self._mark_task_failed(task, f"Max retries exceeded: {error}")
            return
        
        # Prepare for retry
        task.stage = PipelineStage.RETRYING
        
        # Calculate retry delay based on failure mode and attempt
        retry_delay = self._calculate_retry_delay(failure_mode, task.retry_count)
        
        logger.info(f"Retrying task {task.task_id} in {retry_delay} seconds")
        
        # Schedule retry
        asyncio.create_task(self._schedule_retry(task, retry_delay))
    
    def _classify_failure(self, error: Exception) -> FailureMode:
        """Classify failure type for appropriate handling"""
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        if 'rate limit' in error_str or 'throttle' in error_str:
            return FailureMode.RATE_LIMITED
        elif 'timeout' in error_str or 'connection' in error_str:
            return FailureMode.TRANSIENT
        elif 'data quality' in error_str or 'validation' in error_str:
            return FailureMode.DATA_CORRUPT
        elif 'memory' in error_str or 'resource' in error_str:
            return FailureMode.RESOURCE_EXHAUSTED
        elif error_type in ['AuthenticationException', 'ConfigurationException']:
            return FailureMode.PERMANENT
        else:
            return FailureMode.TRANSIENT
    
    def _determine_recovery_strategy(self, failure_mode: FailureMode, task: PipelineTask) -> RecoveryMode:
        """Determine appropriate recovery strategy"""
        critical_task = task.metadata.get('critical', False)
        
        strategy_map = {
            FailureMode.TRANSIENT: RecoveryMode.RETRY_EXPONENTIAL,
            FailureMode.RATE_LIMITED: RecoveryMode.RETRY_LINEAR,
            FailureMode.DATA_CORRUPT: RecoveryMode.SKIP_AND_CONTINUE,
            FailureMode.RESOURCE_EXHAUSTED: RecoveryMode.RETRY_LINEAR,
            FailureMode.PERMANENT: RecoveryMode.SKIP_AND_CONTINUE
        }
        
        base_strategy = strategy_map[failure_mode]
        
        # Override for critical tasks
        if critical_task and failure_mode == FailureMode.PERMANENT:
            return RecoveryMode.HALT_PIPELINE
        
        return base_strategy
    
    def _calculate_retry_delay(self, failure_mode: FailureMode, retry_count: int) -> float:
        """Calculate retry delay based on failure mode and attempt"""
        base_delays = {
            FailureMode.TRANSIENT: 1.0,
            FailureMode.RATE_LIMITED: 60.0,  # Longer delay for rate limits
            FailureMode.DATA_CORRUPT: 5.0,
            FailureMode.RESOURCE_EXHAUSTED: 30.0,
            FailureMode.PERMANENT: 300.0
        }
        
        base_delay = base_delays.get(failure_mode, 1.0)
        
        if failure_mode == FailureMode.RATE_LIMITED:
            # Linear backoff for rate limits
            delay = base_delay * retry_count
        else:
            # Exponential backoff for other failures
            delay = base_delay * (2 ** retry_count)
        
        # Add jitter to prevent thundering herd
        jitter = random.uniform(0.8, 1.2)
        
        return min(delay * jitter, 300.0)  # Cap at 5 minutes
    
    async def _schedule_retry(self, task: PipelineTask, delay: float) -> None:
        """Schedule task retry after delay"""
        await asyncio.sleep(delay)
        
        async with self._lock:
            if task.task_id in self.processing_tasks:
                del self.processing_tasks[task.task_id]
            
            task.stage = PipelineStage.PENDING
            self.task_queue.append(task)
    
    async def _mark_task_completed(
        self,
        task: PipelineTask,
        result: Any,
        from_cache: bool = False
    ) -> None:
        """Mark task as completed"""
        async with self._lock:
            task.stage = PipelineStage.CACHED if from_cache else PipelineStage.COMPLETED
            task.completed_at = datetime.now()
            
            # Move to completed tasks
            if task.task_id in self.processing_tasks:
                del self.processing_tasks[task.task_id]
            
            self.completed_tasks[task.task_id] = task
            
            # Store result in task metadata
            task.metadata['result'] = result
            
            # Check for dependent tasks that can now be queued
            await self._check_dependent_tasks(task.task_id)
        
        await self._update_metrics()
    
    async def _mark_task_failed(self, task: PipelineTask, error_message: str) -> None:
        """Mark task as failed"""
        async with self._lock:
            task.stage = PipelineStage.FAILED
            task.last_error = error_message
            
            # Move to failed tasks
            if task.task_id in self.processing_tasks:
                del self.processing_tasks[task.task_id]
            
            self.failed_tasks[task.task_id] = task
        
        await self._update_metrics()
    
    async def _check_dependent_tasks(self, completed_task_id: str) -> None:
        """Check if any pending tasks can now be queued"""
        tasks_to_queue = []
        
        for task in list(self.tasks.values()):
            if (task.stage == PipelineStage.PENDING and 
                completed_task_id in task.dependencies and
                task not in self.task_queue):
                
                if await self._are_dependencies_satisfied(task):
                    tasks_to_queue.append(task)
        
        for task in tasks_to_queue:
            self.task_queue.append(task)
    
    def _generate_cache_key(self, task: PipelineTask) -> str:
        """Generate cache key for task result"""
        key_data = {
            'stage': task.metadata.get('stage'),
            'data_hash': hashlib.md5(str(task.data).encode()).hexdigest(),
            'version': task.metadata.get('version', '1.0')
        }
        
        return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    async def _get_cached_result(self, cache_key: str, stale_ok: bool = False) -> Optional[Any]:
        """Get cached result if available and valid"""
        if cache_key not in self.result_cache:
            return None
        
        cached_at = self.cache_timestamps.get(cache_key)
        if not cached_at:
            return None
        
        # Check if cache is still valid
        age = (datetime.now() - cached_at).total_seconds()
        if not stale_ok and age > self.cache_ttl:
            # Remove expired cache
            del self.result_cache[cache_key]
            del self.cache_timestamps[cache_key]
            return None
        
        return self.result_cache[cache_key]
    
    async def _cache_result(self, cache_key: str, result: Any) -> None:
        """Cache task result"""
        self.result_cache[cache_key] = result
        self.cache_timestamps[cache_key] = datetime.now()
        
        # Limit cache size (simple LRU-like behavior)
        if len(self.result_cache) > 10000:
            # Remove oldest 10% of cache
            sorted_items = sorted(
                self.cache_timestamps.items(),
                key=lambda x: x[1]
            )
            
            to_remove = sorted_items[:len(sorted_items) // 10]
            for cache_key, _ in to_remove:
                self.result_cache.pop(cache_key, None)
                self.cache_timestamps.pop(cache_key, None)
    
    async def _update_metrics(self) -> None:
        """Update pipeline performance metrics"""
        with self._metrics_lock:
            total_tasks = len(self.tasks)
            completed_tasks = len(self.completed_tasks)
            failed_tasks = len(self.failed_tasks)
            retrying_tasks = len([t for t in self.tasks.values() if t.stage == PipelineStage.RETRYING])
            pending_tasks = len(self.task_queue)
            
            # Calculate processing times
            completed_with_times = [
                t for t in self.completed_tasks.values()
                if t.started_at and t.completed_at
            ]
            
            avg_processing_time = 0.0
            if completed_with_times:
                processing_times = [
                    (t.completed_at - t.started_at).total_seconds()
                    for t in completed_with_times
                ]
                avg_processing_time = sum(processing_times) / len(processing_times)
            
            # Calculate rates
            success_rate = completed_tasks / total_tasks if total_tasks > 0 else 0.0
            error_rate = failed_tasks / total_tasks if total_tasks > 0 else 0.0
            
            # Calculate cache hit rate
            cached_tasks = len([t for t in self.completed_tasks.values() if t.stage == PipelineStage.CACHED])
            cache_hit_rate = cached_tasks / completed_tasks if completed_tasks > 0 else 0.0
            
            # Update metrics
            self.pipeline_metrics = PipelineMetrics(
                total_tasks=total_tasks,
                completed_tasks=completed_tasks,
                failed_tasks=failed_tasks,
                retrying_tasks=retrying_tasks,
                pending_tasks=pending_tasks,
                average_processing_time=avg_processing_time,
                success_rate=success_rate,
                throughput_per_minute=0.0,  # Will be calculated by monitor
                error_rate=error_rate,
                cache_hit_rate=cache_hit_rate,
                last_updated=datetime.now()
            )
    
    async def _monitor_pipeline(self) -> None:
        """Monitor pipeline health and performance"""
        last_completed_count = 0
        last_check_time = datetime.now()
        
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                current_time = datetime.now()
                current_completed = len(self.completed_tasks)
                
                # Calculate throughput
                time_diff = (current_time - last_check_time).total_seconds() / 60.0
                completed_diff = current_completed - last_completed_count
                throughput = completed_diff / time_diff if time_diff > 0 else 0.0
                
                # Update throughput metric
                with self._metrics_lock:
                    self.pipeline_metrics.throughput_per_minute = throughput
                
                # Log health status
                health_status = self.get_health_status()
                logger.info(f"Pipeline health: {health_status['status']}, throughput: {throughput:.1f}/min")
                
                # Check for concerning patterns
                if health_status['error_rate'] > 0.1:  # >10% error rate
                    logger.warning(f"High error rate detected: {health_status['error_rate']:.1%}")
                
                if health_status['pending_tasks'] > 1000:
                    logger.warning(f"Large task backlog: {health_status['pending_tasks']} pending tasks")
                
                last_completed_count = current_completed
                last_check_time = current_time
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Pipeline monitoring error: {e}")
    
    async def _checkpoint_task(self) -> None:
        """Periodic checkpointing task"""
        while self.is_running:
            try:
                await asyncio.sleep(self.checkpoint_interval)
                await self._save_checkpoint()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Checkpoint error: {e}")
    
    async def _save_checkpoint(self) -> None:
        """Save pipeline state to checkpoint"""
        try:
            checkpoint_dir = Path("data/pipeline_checkpoints")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint_data = {
                'pipeline_name': self.name,
                'timestamp': datetime.now().isoformat(),
                'tasks': {
                    task_id: {
                        'stage': task.stage.value,
                        'retry_count': task.retry_count,
                        'last_error': task.last_error,
                        'metadata': task.metadata
                    }
                    for task_id, task in self.tasks.items()
                },
                'metrics': asdict(self.pipeline_metrics)
            }
            
            checkpoint_file = checkpoint_dir / f"{self.name}_checkpoint.json"
            
            async with aiofiles.open(checkpoint_file, 'w') as f:
                await f.write(json.dumps(checkpoint_data, indent=2, default=str))
            
            logger.debug(f"Saved checkpoint with {len(self.tasks)} tasks")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    async def _load_checkpoint(self) -> None:
        """Load pipeline state from checkpoint"""
        try:
            checkpoint_file = Path(f"data/pipeline_checkpoints/{self.name}_checkpoint.json")
            
            if not checkpoint_file.exists():
                return
            
            async with aiofiles.open(checkpoint_file, 'r') as f:
                content = await f.read()
                checkpoint_data = json.loads(content)
            
            # Restore task states (but not data - that would need to be re-added)
            restored_count = 0
            for task_id, task_state in checkpoint_data.get('tasks', {}).items():
                if task_id in self.tasks:
                    task = self.tasks[task_id]
                    task.stage = PipelineStage(task_state['stage'])
                    task.retry_count = task_state['retry_count']
                    task.last_error = task_state['last_error']
                    restored_count += 1
            
            logger.info(f"Restored {restored_count} task states from checkpoint")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current pipeline health status"""
        total_tasks = len(self.tasks)
        
        if total_tasks == 0:
            status = "idle"
        elif self.is_paused:
            status = "paused"
        elif self.pipeline_metrics.error_rate > 0.2:
            status = "degraded"
        elif self.pipeline_metrics.error_rate > 0.1:
            status = "warning"
        else:
            status = "healthy"
        
        return {
            'pipeline_name': self.name,
            'status': status,
            'is_running': self.is_running,
            'is_paused': self.is_paused,
            'total_tasks': total_tasks,
            'pending_tasks': len(self.task_queue),
            'processing_tasks': len(self.processing_tasks),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'success_rate': self.pipeline_metrics.success_rate,
            'error_rate': self.pipeline_metrics.error_rate,
            'cache_hit_rate': self.pipeline_metrics.cache_hit_rate,
            'throughput_per_minute': self.pipeline_metrics.throughput_per_minute,
            'average_processing_time_seconds': self.pipeline_metrics.average_processing_time,
            'executor_metrics': {
                name: executor.get_metrics()
                for name, executor in self.executors.items()
            },
            'data_quality_metrics': self.data_validator.get_quality_metrics(),
            'last_updated': self.pipeline_metrics.last_updated.isoformat()
        }
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed status of a specific task"""
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        
        return {
            'task_id': task.task_id,
            'stage': task.stage.value,
            'created_at': task.created_at.isoformat(),
            'started_at': task.started_at.isoformat() if task.started_at else None,
            'completed_at': task.completed_at.isoformat() if task.completed_at else None,
            'retry_count': task.retry_count,
            'max_retries': task.max_retries,
            'last_error': task.last_error,
            'priority': task.priority,
            'dependencies': task.dependencies,
            'metadata': task.metadata
        }