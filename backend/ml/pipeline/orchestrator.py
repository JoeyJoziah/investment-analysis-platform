"""
ML Pipeline Orchestrator - Handles scheduling, execution, and automated retraining

Integrates with Claude Flow V3 for:
- Neural pattern training coordination
- Swarm agent task delegation
- Memory-based learning persistence
- Hook-based event handling
"""

import asyncio
import logging
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import json
from pathlib import Path
import schedule
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import uuid

from .base import ModelPipeline, PipelineConfig, PipelineResult, PipelineStatus, ModelArtifact
from .registry import ModelRegistry
from .monitoring import ModelMonitor, PerformanceMetrics

# Claude Flow integration imports (lazy loaded)
try:
    from .memory_sync import get_memory_adapter, ClaudeFlowMemoryAdapter
    from .task_bridge import get_task_bridge, TaskBridge, TaskStatus
    CLAUDE_FLOW_AVAILABLE = True
except ImportError:
    CLAUDE_FLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


class TriggerType(Enum):
    """Types of retraining triggers"""
    SCHEDULED = "scheduled"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_DRIFT = "data_drift"
    MANUAL = "manual"
    NEW_DATA_THRESHOLD = "new_data_threshold"
    ERROR_RATE = "error_rate"
    CONCEPT_DRIFT = "concept_drift"


class ScheduleFrequency(Enum):
    """Training schedule frequencies"""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    CUSTOM = "custom"


@dataclass
class TrainingSchedule:
    """Configuration for training schedule"""
    frequency: ScheduleFrequency
    time_of_day: str = "02:00"  # 2 AM default
    day_of_week: Optional[int] = None  # 0=Monday, 6=Sunday
    day_of_month: Optional[int] = None
    custom_cron: Optional[str] = None
    enabled: bool = True
    max_concurrent_trainings: int = 2
    
    def get_next_run_time(self) -> datetime:
        """Calculate next scheduled run time"""
        now = datetime.utcnow()
        
        if self.frequency == ScheduleFrequency.HOURLY:
            return now + timedelta(hours=1)
        elif self.frequency == ScheduleFrequency.DAILY:
            # Parse time_of_day (HH:MM format)
            hour, minute = map(int, self.time_of_day.split(':'))
            next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)
            return next_run
        elif self.frequency == ScheduleFrequency.WEEKLY:
            # Calculate next occurrence of specified day
            days_ahead = (self.day_of_week - now.weekday()) % 7
            if days_ahead == 0 and now.time() > datetime.strptime(self.time_of_day, "%H:%M").time():
                days_ahead = 7
            return now + timedelta(days=days_ahead)
        elif self.frequency == ScheduleFrequency.MONTHLY:
            # Next occurrence of specified day of month
            if now.day > self.day_of_month:
                # Move to next month
                if now.month == 12:
                    next_run = now.replace(year=now.year + 1, month=1, day=self.day_of_month)
                else:
                    next_run = now.replace(month=now.month + 1, day=self.day_of_month)
            else:
                next_run = now.replace(day=self.day_of_month)
            
            hour, minute = map(int, self.time_of_day.split(':'))
            return next_run.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        return now + timedelta(days=1)  # Default fallback


@dataclass
class RetrainingTrigger:
    """Configuration for automated retraining triggers"""
    trigger_type: TriggerType
    enabled: bool = True
    
    # Performance thresholds
    min_accuracy: float = 0.8
    max_error_rate: float = 0.1
    performance_window_hours: int = 24
    
    # Data drift thresholds
    drift_threshold: float = 0.3
    drift_detection_method: str = "kolmogorov_smirnov"
    
    # New data thresholds
    min_new_samples: int = 1000
    new_data_window_days: int = 7
    
    # Alert configuration
    alert_on_trigger: bool = True
    cooldown_hours: int = 24  # Minimum time between retrainings
    
    def should_trigger(self, metrics: Dict[str, Any]) -> bool:
        """Check if retraining should be triggered"""
        if not self.enabled:
            return False
        
        if self.trigger_type == TriggerType.PERFORMANCE_DEGRADATION:
            current_accuracy = metrics.get("accuracy", 1.0)
            current_error_rate = metrics.get("error_rate", 0.0)
            return (current_accuracy < self.min_accuracy or 
                   current_error_rate > self.max_error_rate)
        
        elif self.trigger_type == TriggerType.DATA_DRIFT:
            drift_score = metrics.get("drift_score", 0.0)
            return drift_score > self.drift_threshold
        
        elif self.trigger_type == TriggerType.NEW_DATA_THRESHOLD:
            new_samples = metrics.get("new_samples", 0)
            return new_samples >= self.min_new_samples
        
        elif self.trigger_type == TriggerType.ERROR_RATE:
            error_rate = metrics.get("prediction_error_rate", 0.0)
            return error_rate > self.max_error_rate
        
        return False


@dataclass
class OrchestratorConfig:
    """Configuration for ML Orchestrator"""
    name: str = "ml_orchestrator"
    
    # Execution settings
    max_concurrent_pipelines: int = 3
    executor_type: str = "thread"  # "thread" or "process"
    num_workers: int = 4
    
    # Scheduling
    enable_scheduling: bool = True
    default_schedule: TrainingSchedule = field(default_factory=lambda: TrainingSchedule(
        frequency=ScheduleFrequency.DAILY,
        time_of_day="02:00"
    ))
    
    # Retraining
    enable_auto_retraining: bool = True
    retraining_triggers: List[RetrainingTrigger] = field(default_factory=lambda: [
        RetrainingTrigger(
            trigger_type=TriggerType.PERFORMANCE_DEGRADATION,
            min_accuracy=0.8
        ),
        RetrainingTrigger(
            trigger_type=TriggerType.DATA_DRIFT,
            drift_threshold=0.3
        )
    ])
    
    # Resource management
    max_memory_gb: float = 16.0
    max_training_hours: float = 24.0
    cost_limit_daily_usd: float = 10.0
    
    # Storage
    models_path: str = "/app/ml_models"
    logs_path: str = "/app/ml_logs"
    checkpoint_interval_minutes: int = 30
    
    # Monitoring
    enable_monitoring: bool = True
    metrics_collection_interval_seconds: int = 60
    
    # Failure handling
    max_retries: int = 3
    retry_delay_seconds: int = 300
    failure_alert_threshold: int = 2


class MLOrchestrator:
    """
    Main orchestrator for ML pipelines
    Handles scheduling, execution, monitoring, and automated retraining
    """
    
    def __init__(self, config: OrchestratorConfig = None):
        self.config = config or OrchestratorConfig()
        self.registry = ModelRegistry()
        self.monitor = ModelMonitor()
        
        # Pipeline management
        self.active_pipelines: Dict[str, ModelPipeline] = {}
        self.pipeline_history: List[PipelineResult] = []
        self.scheduled_jobs: Dict[str, schedule.Job] = {}
        
        # Execution management
        if self.config.executor_type == "thread":
            self.executor = ThreadPoolExecutor(max_workers=self.config.num_workers)
        else:
            self.executor = ProcessPoolExecutor(max_workers=self.config.num_workers)
        
        # Retraining state
        self.last_retraining: Dict[str, datetime] = {}
        self.retraining_queue: List[Dict] = []
        
        # Monitoring
        self.performance_history: Dict[str, List[PerformanceMetrics]] = {}
        
        # Background tasks
        self.scheduler_thread = None
        self.monitor_thread = None
        self.running = False
        
        # Create necessary directories
        Path(self.config.models_path).mkdir(parents=True, exist_ok=True)
        Path(self.config.logs_path).mkdir(parents=True, exist_ok=True)
    
    async def start(self):
        """Start the orchestrator"""
        self.running = True
        logger.info("Starting ML Orchestrator...")
        
        # Start scheduler thread
        if self.config.enable_scheduling:
            self.scheduler_thread = threading.Thread(target=self._run_scheduler)
            self.scheduler_thread.daemon = True
            self.scheduler_thread.start()
            logger.info("Scheduler started")
        
        # Start monitoring thread
        if self.config.enable_monitoring:
            self.monitor_thread = threading.Thread(target=self._run_monitor)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            logger.info("Monitor started")
        
        # Load existing models from registry
        await self.registry.load_registry()
        
        logger.info("ML Orchestrator started successfully")
    
    async def stop(self):
        """Stop the orchestrator"""
        logger.info("Stopping ML Orchestrator...")
        self.running = False
        
        # Cancel active pipelines
        for pipeline_id in list(self.active_pipelines.keys()):
            await self.cancel_pipeline(pipeline_id)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Wait for threads to finish
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("ML Orchestrator stopped")
    
    async def submit_pipeline(
        self,
        pipeline: ModelPipeline,
        trigger_type: TriggerType = TriggerType.MANUAL
    ) -> str:
        """Submit a pipeline for execution"""
        pipeline_id = str(uuid.uuid4())
        
        # Check resource limits
        if len(self.active_pipelines) >= self.config.max_concurrent_pipelines:
            logger.warning(f"Maximum concurrent pipelines reached ({self.config.max_concurrent_pipelines})")
            self.retraining_queue.append({
                "pipeline": pipeline,
                "pipeline_id": pipeline_id,
                "trigger_type": trigger_type,
                "submitted_at": datetime.utcnow()
            })
            return pipeline_id
        
        # Start pipeline execution
        self.active_pipelines[pipeline_id] = pipeline
        
        # Execute asynchronously
        asyncio.create_task(self._execute_pipeline(pipeline_id, pipeline, trigger_type))
        
        logger.info(f"Pipeline {pipeline_id} submitted (trigger: {trigger_type.value})")
        return pipeline_id
    
    async def _execute_pipeline(
        self,
        pipeline_id: str,
        pipeline: ModelPipeline,
        trigger_type: TriggerType
    ):
        """Execute a pipeline with monitoring and error handling"""
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Executing pipeline {pipeline_id}")
            
            # Validate configuration
            if not await pipeline.validate_config():
                raise ValueError("Invalid pipeline configuration")
            
            # Execute pipeline
            result = await pipeline.execute()
            
            # Store result
            self.pipeline_history.append(result)
            
            # Register model if successful
            if result.status == PipelineStatus.COMPLETED and result.model_artifact:
                await self.registry.register_model(
                    result.model_artifact,
                    metadata={
                        "trigger_type": trigger_type.value,
                        "pipeline_id": pipeline_id,
                        "execution_time": (datetime.utcnow() - start_time).total_seconds()
                    }
                )
                
                # Update last retraining time
                model_name = result.model_artifact.name
                self.last_retraining[model_name] = datetime.utcnow()
                
                logger.info(f"Pipeline {pipeline_id} completed successfully")
            else:
                logger.error(f"Pipeline {pipeline_id} failed: {result.error_message}")
                
                # Handle failure
                await self._handle_pipeline_failure(pipeline_id, result)
        
        except Exception as e:
            logger.error(f"Pipeline {pipeline_id} execution error: {e}")
            
            # Create failure result
            result = PipelineResult(
                pipeline_id=pipeline_id,
                status=PipelineStatus.FAILED,
                start_time=start_time,
                end_time=datetime.utcnow(),
                error_message=str(e)
            )
            self.pipeline_history.append(result)
        
        finally:
            # Remove from active pipelines
            if pipeline_id in self.active_pipelines:
                del self.active_pipelines[pipeline_id]
            
            # Process queue if available
            await self._process_queue()
    
    async def _handle_pipeline_failure(self, pipeline_id: str, result: PipelineResult):
        """Handle pipeline failure with retry logic"""
        pipeline = self.active_pipelines.get(pipeline_id)
        if not pipeline:
            return
        
        # Check retry count
        retry_count = getattr(pipeline, '_retry_count', 0)
        
        if retry_count < self.config.max_retries:
            logger.info(f"Retrying pipeline {pipeline_id} (attempt {retry_count + 1})")
            
            # Update retry count
            pipeline._retry_count = retry_count + 1
            
            # Schedule retry
            await asyncio.sleep(self.config.retry_delay_seconds)
            await self.submit_pipeline(pipeline, TriggerType.MANUAL)
        else:
            logger.error(f"Pipeline {pipeline_id} failed after {self.config.max_retries} retries")
            
            # Send alert if threshold reached
            if retry_count >= self.config.failure_alert_threshold:
                await self._send_alert(
                    f"Pipeline {pipeline_id} failed repeatedly",
                    result.error_message
                )
    
    async def _process_queue(self):
        """Process queued pipelines"""
        if not self.retraining_queue:
            return
        
        while (self.retraining_queue and 
               len(self.active_pipelines) < self.config.max_concurrent_pipelines):
            
            # Get next item from queue
            item = self.retraining_queue.pop(0)
            
            # Submit pipeline
            pipeline_id = item["pipeline_id"]
            self.active_pipelines[pipeline_id] = item["pipeline"]
            
            asyncio.create_task(self._execute_pipeline(
                pipeline_id,
                item["pipeline"],
                item["trigger_type"]
            ))
    
    def _run_scheduler(self):
        """Background thread for scheduled training"""
        while self.running:
            try:
                schedule.run_pending()
                asyncio.run(self._check_retraining_triggers())
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
            
            # Sleep for a short interval
            threading.Event().wait(10)
    
    def _run_monitor(self):
        """Background thread for monitoring"""
        while self.running:
            try:
                asyncio.run(self._collect_metrics())
            except Exception as e:
                logger.error(f"Monitor error: {e}")
            
            # Sleep for monitoring interval
            threading.Event().wait(self.config.metrics_collection_interval_seconds)
    
    async def _check_retraining_triggers(self):
        """Check if any retraining triggers are met"""
        if not self.config.enable_auto_retraining:
            return
        
        # Get current models from registry
        models = await self.registry.get_active_models()
        
        for model_metadata in models:
            model_name = model_metadata["name"]
            
            # Check cooldown period
            last_training = self.last_retraining.get(model_name)
            if last_training:
                hours_since_training = (datetime.utcnow() - last_training).total_seconds() / 3600
                if hours_since_training < 24:  # Default 24 hour cooldown
                    continue
            
            # Get model metrics
            metrics = await self.monitor.get_model_metrics(model_name)
            
            # Check each trigger
            for trigger in self.config.retraining_triggers:
                if trigger.should_trigger(metrics):
                    logger.info(f"Retraining trigger activated for {model_name}: {trigger.trigger_type.value}")
                    
                    # Create retraining pipeline
                    pipeline = await self._create_retraining_pipeline(model_name, model_metadata)
                    
                    # Submit for execution
                    await self.submit_pipeline(pipeline, trigger.trigger_type)
                    
                    # Only trigger once per model
                    break
    
    async def _create_retraining_pipeline(
        self,
        model_name: str,
        model_metadata: Dict
    ) -> ModelPipeline:
        """Create a retraining pipeline for an existing model"""
        # Load original configuration
        config_path = Path(model_metadata.get("config_path", ""))
        if config_path.exists():
            with open(config_path, 'r') as f:
                original_config = json.load(f)
        else:
            original_config = {}
        
        # Create new pipeline config with updated version
        config = PipelineConfig(
            name=model_name,
            version=self._increment_version(model_metadata.get("version", "1.0.0")),
            model_type=model_metadata.get("model_type", "regression"),
            data_source=original_config.get("data_source", ""),
            feature_columns=original_config.get("feature_columns", []),
            target_column=original_config.get("target_column", ""),
            hyperparameters=original_config.get("hyperparameters", {})
        )
        
        # Import the appropriate pipeline class dynamically
        # This would be based on the model type
        from .implementations import create_pipeline
        pipeline = create_pipeline(config)
        
        return pipeline
    
    def _increment_version(self, version: str) -> str:
        """Increment model version string"""
        parts = version.split('.')
        if len(parts) == 3:
            # Increment patch version
            parts[2] = str(int(parts[2]) + 1)
        else:
            parts = ["1", "0", "1"]
        return '.'.join(parts)
    
    async def _collect_metrics(self):
        """Collect metrics from deployed models"""
        try:
            # Get deployed models
            models = await self.registry.get_deployed_models()
            
            for model in models:
                model_name = model["name"]
                endpoint = model.get("deployment_endpoint")
                
                if endpoint:
                    # Collect performance metrics
                    metrics = await self.monitor.collect_metrics(model_name, endpoint)
                    
                    # Store in history
                    if model_name not in self.performance_history:
                        self.performance_history[model_name] = []
                    
                    self.performance_history[model_name].append(metrics)
                    
                    # Keep only recent history (last 7 days)
                    cutoff_time = datetime.utcnow() - timedelta(days=7)
                    self.performance_history[model_name] = [
                        m for m in self.performance_history[model_name]
                        if m.timestamp > cutoff_time
                    ]
        
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
    
    async def _send_alert(self, subject: str, message: str):
        """Send alert notification"""
        logger.warning(f"ALERT: {subject} - {message}")
        # In production, this would send to monitoring system, email, Slack, etc.
    
    async def schedule_training(
        self,
        pipeline: ModelPipeline,
        schedule: TrainingSchedule
    ) -> str:
        """Schedule a pipeline for regular training"""
        job_id = str(uuid.uuid4())
        
        if schedule.frequency == ScheduleFrequency.HOURLY:
            job = schedule.every().hour.do(
                lambda: asyncio.run(self.submit_pipeline(pipeline, TriggerType.SCHEDULED))
            )
        elif schedule.frequency == ScheduleFrequency.DAILY:
            job = schedule.every().day.at(schedule.time_of_day).do(
                lambda: asyncio.run(self.submit_pipeline(pipeline, TriggerType.SCHEDULED))
            )
        elif schedule.frequency == ScheduleFrequency.WEEKLY:
            job = schedule.every().week.do(
                lambda: asyncio.run(self.submit_pipeline(pipeline, TriggerType.SCHEDULED))
            )
        else:
            logger.warning(f"Unsupported schedule frequency: {schedule.frequency}")
            return None
        
        self.scheduled_jobs[job_id] = job
        logger.info(f"Scheduled pipeline {pipeline.config.name} with job ID {job_id}")
        
        return job_id
    
    async def cancel_pipeline(self, pipeline_id: str) -> bool:
        """Cancel a running pipeline"""
        if pipeline_id in self.active_pipelines:
            # Mark as cancelled in history
            for result in self.pipeline_history:
                if result.pipeline_id == pipeline_id:
                    result.status = PipelineStatus.CANCELLED
                    result.end_time = datetime.utcnow()
                    break
            
            # Remove from active pipelines
            del self.active_pipelines[pipeline_id]
            
            logger.info(f"Pipeline {pipeline_id} cancelled")
            return True
        
        return False
    
    def get_status(self) -> Dict:
        """Get orchestrator status"""
        return {
            "running": self.running,
            "active_pipelines": len(self.active_pipelines),
            "queued_pipelines": len(self.retraining_queue),
            "scheduled_jobs": len(self.scheduled_jobs),
            "total_executions": len(self.pipeline_history),
            "recent_failures": sum(
                1 for r in self.pipeline_history[-10:]
                if r.status == PipelineStatus.FAILED
            ),
            "models_monitored": len(self.performance_history),
            "claude_flow_integration": CLAUDE_FLOW_AVAILABLE,
            "config": {
                "max_concurrent": self.config.max_concurrent_pipelines,
                "auto_retraining": self.config.enable_auto_retraining,
                "scheduling": self.config.enable_scheduling
            }
        }

    # =========================================================================
    # CLAUDE FLOW V3 INTEGRATION - Hook Listeners
    # =========================================================================

    async def handle_pre_task_hook(self, task_description: str) -> Dict:
        """
        Handle Claude Flow pre-task hook
        Called before a task starts to get routing recommendations
        """
        if not CLAUDE_FLOW_AVAILABLE:
            return {"status": "skipped", "reason": "claude_flow_not_available"}

        try:
            # Get memory adapter for pattern lookup
            memory = await get_memory_adapter()

            # Search for similar past tasks
            similar_tasks = await memory.search(
                query=task_description,
                namespace="training-jobs",
                limit=5
            )

            # Determine optimal model/approach based on history
            recommendation = {
                "task_description": task_description,
                "similar_tasks_found": len(similar_tasks),
                "recommended_approach": "standard",
                "estimated_duration": None,
                "patterns_applied": []
            }

            if similar_tasks:
                # Analyze past performance
                successes = [t for t in similar_tasks if t.get("value", {}).get("status") == "completed"]
                if successes:
                    recommendation["recommended_approach"] = "learned"
                    recommendation["patterns_applied"] = [t["key"] for t in successes[:3]]

            # Store pre-task event
            await memory.store(
                key=f"pre_task_{datetime.utcnow().timestamp()}",
                value={
                    "task": task_description,
                    "recommendation": recommendation,
                    "timestamp": datetime.utcnow().isoformat()
                },
                namespace="pipeline-state"
            )

            logger.info(f"Pre-task hook: {task_description[:50]}...")
            return recommendation

        except Exception as e:
            logger.error(f"Pre-task hook error: {e}")
            return {"status": "error", "error": str(e)}

    async def handle_post_task_hook(
        self,
        task_id: str,
        success: bool,
        result: Optional[Dict] = None,
        store_results: bool = True
    ) -> Dict:
        """
        Handle Claude Flow post-task hook
        Called after a task completes to record outcome for learning
        """
        if not CLAUDE_FLOW_AVAILABLE:
            return {"status": "skipped", "reason": "claude_flow_not_available"}

        try:
            memory = await get_memory_adapter()

            # Record task outcome
            outcome = {
                "task_id": task_id,
                "success": success,
                "result": result,
                "completed_at": datetime.utcnow().isoformat()
            }

            if store_results:
                await memory.sync_training_job(task_id, outcome)

            # If successful, extract patterns for learning
            if success and result:
                await self._extract_learning_patterns(task_id, result)

            # Trigger neural training update if significant
            if success:
                await self._trigger_neural_update(task_id, result)

            logger.info(f"Post-task hook: {task_id} (success={success})")
            return {"status": "recorded", "task_id": task_id}

        except Exception as e:
            logger.error(f"Post-task hook error: {e}")
            return {"status": "error", "error": str(e)}

    async def handle_post_edit_hook(
        self,
        file_path: str,
        success: bool = True,
        train_neural: bool = False
    ) -> Dict:
        """
        Handle Claude Flow post-edit hook
        Called after file edits to potentially trigger retraining
        """
        if not CLAUDE_FLOW_AVAILABLE:
            return {"status": "skipped", "reason": "claude_flow_not_available"}

        try:
            # Check if edit affects ML pipeline
            ml_related = any(p in file_path for p in [
                "backend/ml/",
                "data_pipelines/",
                "models/",
                ".py"
            ])

            if not ml_related:
                return {"status": "skipped", "reason": "not_ml_related"}

            memory = await get_memory_adapter()

            # Record edit event
            await memory.store(
                key=f"edit_{Path(file_path).name}_{datetime.utcnow().timestamp()}",
                value={
                    "file": file_path,
                    "success": success,
                    "train_neural": train_neural,
                    "timestamp": datetime.utcnow().isoformat()
                },
                namespace="ml-patterns"
            )

            # Trigger neural training if requested
            if train_neural:
                result = await self._run_neural_training("post-edit")
                return {"status": "neural_training_triggered", "result": result}

            return {"status": "recorded", "file": file_path}

        except Exception as e:
            logger.error(f"Post-edit hook error: {e}")
            return {"status": "error", "error": str(e)}

    async def _extract_learning_patterns(
        self,
        task_id: str,
        result: Dict
    ):
        """Extract learning patterns from successful task execution"""
        try:
            memory = await get_memory_adapter()

            # Extract key metrics
            patterns = {
                "task_id": task_id,
                "execution_time": result.get("execution_time"),
                "model_type": result.get("model_type"),
                "accuracy": result.get("metrics", {}).get("accuracy"),
                "hyperparameters": result.get("hyperparameters"),
                "extracted_at": datetime.utcnow().isoformat()
            }

            # Store pattern
            await memory.store(
                key=f"pattern_{task_id}",
                value=patterns,
                namespace="ml-patterns",
                metadata={"type": "execution_pattern"}
            )

        except Exception as e:
            logger.warning(f"Pattern extraction failed: {e}")

    async def _trigger_neural_update(
        self,
        task_id: str,
        result: Optional[Dict] = None
    ):
        """Trigger neural pattern training update via CLI"""
        try:
            # Run neural training via claude-flow CLI
            cmd = [
                "npx", "@claude-flow/cli@latest", "hooks",
                "post-task", "--task-id", task_id, "--success", "true"
            ]

            if result:
                cmd.extend(["--store-results", "true"])

            # Run async
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.wait()

        except Exception as e:
            logger.warning(f"Neural update trigger failed: {e}")

    async def _run_neural_training(self, trigger: str) -> Dict:
        """Run neural pattern training via CLI"""
        try:
            cmd = [
                "npx", "@claude-flow/cli@latest", "neural",
                "train", "--pattern-type", "coordination", "--epochs", "10"
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            return {
                "trigger": trigger,
                "success": process.returncode == 0,
                "output": stdout.decode() if stdout else None
            }

        except Exception as e:
            logger.error(f"Neural training failed: {e}")
            return {"trigger": trigger, "success": False, "error": str(e)}

    async def sync_with_claude_flow_memory(self) -> Dict:
        """Sync pipeline state with Claude Flow memory system"""
        if not CLAUDE_FLOW_AVAILABLE:
            return {"status": "skipped", "reason": "claude_flow_not_available"}

        try:
            memory = await get_memory_adapter()

            # Sync model registry
            result = await memory.sync_registry_to_memory()

            # Store pipeline status
            status = self.get_status()
            await memory.store(
                key="orchestrator_status",
                value=status,
                namespace="pipeline-state"
            )

            logger.info(f"Synced with Claude Flow: {result.entries_synced} entries")
            return {
                "status": "synced",
                "entries_synced": result.entries_synced,
                "errors": result.errors
            }

        except Exception as e:
            logger.error(f"Memory sync failed: {e}")
            return {"status": "error", "error": str(e)}