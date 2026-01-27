"""
Task Bridge - Adapter between Claude Code Task format and Celery Task format
Enables coordination between Claude Flow swarm agents and ML pipeline workers
"""

import os
import json
import uuid
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Unified task status across systems"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class UnifiedTask:
    """
    Unified task format that bridges Claude Code Task tool and Celery tasks
    """
    # Common fields
    task_id: str
    status: TaskStatus
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    # Claude Code Task fields
    subject: Optional[str] = None
    description: Optional[str] = None
    owner: Optional[str] = None
    blocks: List[str] = field(default_factory=list)
    blocked_by: List[str] = field(default_factory=list)

    # Celery Task fields
    celery_task_id: Optional[str] = None
    func_name: Optional[str] = None
    args: tuple = field(default_factory=tuple)
    kwargs: Dict = field(default_factory=dict)
    queue: str = "default"
    retry_count: int = 0
    max_retries: int = 3

    # Result
    result: Optional[Any] = None
    error_message: Optional[str] = None

    # Metadata for coordination
    source: str = "unknown"  # "claude_code" or "celery"
    swarm_id: Optional[str] = None
    agent_type: Optional[str] = None

    def to_claude_code_format(self) -> Dict:
        """Convert to Claude Code Task tool format"""
        return {
            "id": self.task_id,
            "subject": self.subject or self.func_name or "ML Task",
            "description": self.description or f"Execute {self.func_name}",
            "status": self.status.value,
            "owner": self.owner or self.agent_type,
            "blocks": self.blocks,
            "blockedBy": self.blocked_by,
            "metadata": {
                "celery_task_id": self.celery_task_id,
                "source": self.source,
                "swarm_id": self.swarm_id,
                "created_at": self.created_at.isoformat()
            }
        }

    def to_celery_format(self) -> Dict:
        """Convert to Celery task format"""
        return {
            "task_id": self.celery_task_id or self.task_id,
            "name": self.func_name,
            "args": list(self.args),
            "kwargs": self.kwargs,
            "queue": self.queue,
            "retries": self.retry_count,
            "max_retries": self.max_retries,
            "status": self.status.value.upper(),
            "result": self.result,
            "error": self.error_message
        }

    @classmethod
    def from_claude_code(cls, task_data: Dict) -> "UnifiedTask":
        """Create UnifiedTask from Claude Code Task tool format"""
        metadata = task_data.get("metadata", {})
        return cls(
            task_id=task_data.get("id", str(uuid.uuid4())),
            status=TaskStatus(task_data.get("status", "pending")),
            subject=task_data.get("subject"),
            description=task_data.get("description"),
            owner=task_data.get("owner"),
            blocks=task_data.get("blocks", []),
            blocked_by=task_data.get("blockedBy", []),
            celery_task_id=metadata.get("celery_task_id"),
            swarm_id=metadata.get("swarm_id"),
            source="claude_code"
        )

    @classmethod
    def from_celery(cls, task_data: Dict) -> "UnifiedTask":
        """Create UnifiedTask from Celery task format"""
        status_map = {
            "PENDING": TaskStatus.PENDING,
            "STARTED": TaskStatus.IN_PROGRESS,
            "SUCCESS": TaskStatus.COMPLETED,
            "FAILURE": TaskStatus.FAILED,
            "REVOKED": TaskStatus.CANCELLED
        }

        return cls(
            task_id=str(uuid.uuid4()),
            celery_task_id=task_data.get("task_id"),
            status=status_map.get(task_data.get("status", "PENDING"), TaskStatus.PENDING),
            func_name=task_data.get("name"),
            args=tuple(task_data.get("args", [])),
            kwargs=task_data.get("kwargs", {}),
            queue=task_data.get("queue", "default"),
            retry_count=task_data.get("retries", 0),
            max_retries=task_data.get("max_retries", 3),
            result=task_data.get("result"),
            error_message=task_data.get("error"),
            source="celery"
        )


class TaskBridge:
    """
    Bridge between Claude Code Task tool and Celery for ML pipeline coordination
    """

    def __init__(self):
        self.tasks: Dict[str, UnifiedTask] = {}
        self.celery_to_unified: Dict[str, str] = {}  # celery_id -> unified_id mapping
        self._callbacks: Dict[str, List[Callable]] = {}
        self._state_file = Path(".claude-flow/task_bridge_state.json")

    async def initialize(self):
        """Initialize the bridge and load persisted state"""
        try:
            if self._state_file.exists():
                with open(self._state_file, 'r') as f:
                    state = json.load(f)

                for task_data in state.get("tasks", []):
                    task = UnifiedTask(
                        task_id=task_data["task_id"],
                        status=TaskStatus(task_data["status"]),
                        subject=task_data.get("subject"),
                        description=task_data.get("description"),
                        celery_task_id=task_data.get("celery_task_id"),
                        source=task_data.get("source", "unknown")
                    )
                    self.tasks[task.task_id] = task

                    if task.celery_task_id:
                        self.celery_to_unified[task.celery_task_id] = task.task_id

                logger.info(f"Loaded {len(self.tasks)} tasks from state")

        except Exception as e:
            logger.warning(f"Could not load task bridge state: {e}")

    async def save_state(self):
        """Persist task state"""
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)

            state = {
                "tasks": [
                    {
                        "task_id": t.task_id,
                        "status": t.status.value,
                        "subject": t.subject,
                        "description": t.description,
                        "celery_task_id": t.celery_task_id,
                        "source": t.source
                    }
                    for t in self.tasks.values()
                ],
                "updated_at": datetime.utcnow().isoformat()
            }

            with open(self._state_file, 'w') as f:
                json.dump(state, f, indent=2)

        except Exception as e:
            logger.error(f"Could not save task bridge state: {e}")

    async def create_ml_task(
        self,
        func_name: str,
        args: tuple = (),
        kwargs: Optional[Dict] = None,
        subject: Optional[str] = None,
        description: Optional[str] = None,
        queue: str = "ml_pipeline",
        agent_type: Optional[str] = None,
        swarm_id: Optional[str] = None
    ) -> UnifiedTask:
        """
        Create a unified task for ML pipeline execution
        Can be tracked by both Claude Code and Celery
        """
        task = UnifiedTask(
            task_id=str(uuid.uuid4()),
            status=TaskStatus.PENDING,
            subject=subject or f"ML: {func_name}",
            description=description or f"Execute ML task: {func_name}",
            func_name=func_name,
            args=args,
            kwargs=kwargs or {},
            queue=queue,
            agent_type=agent_type,
            swarm_id=swarm_id,
            source="claude_code"
        )

        self.tasks[task.task_id] = task
        await self.save_state()

        logger.info(f"Created ML task: {task.task_id} ({func_name})")
        return task

    async def submit_to_celery(
        self,
        task: UnifiedTask,
        celery_app: Any = None
    ) -> str:
        """Submit a unified task to Celery for execution"""
        try:
            if celery_app is None:
                # Import Celery app if not provided
                from backend.tasks.celery_app import celery_app as app
                celery_app = app

            # Get the Celery task
            celery_task = celery_app.tasks.get(task.func_name)
            if not celery_task:
                raise ValueError(f"Celery task not found: {task.func_name}")

            # Submit to Celery
            result = celery_task.apply_async(
                args=task.args,
                kwargs=task.kwargs,
                queue=task.queue
            )

            # Update unified task
            task.celery_task_id = result.id
            task.status = TaskStatus.IN_PROGRESS
            task.updated_at = datetime.utcnow()

            # Track mapping
            self.celery_to_unified[result.id] = task.task_id

            await self.save_state()

            logger.info(f"Submitted to Celery: {task.task_id} -> {result.id}")
            return result.id

        except Exception as e:
            logger.error(f"Failed to submit to Celery: {e}")
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            await self.save_state()
            raise

    async def update_from_celery(
        self,
        celery_task_id: str,
        status: str,
        result: Any = None,
        error: Optional[str] = None
    ):
        """Update unified task from Celery callback"""
        unified_id = self.celery_to_unified.get(celery_task_id)
        if not unified_id:
            logger.warning(f"Unknown Celery task: {celery_task_id}")
            return

        task = self.tasks.get(unified_id)
        if not task:
            return

        # Map Celery status
        status_map = {
            "PENDING": TaskStatus.PENDING,
            "STARTED": TaskStatus.IN_PROGRESS,
            "SUCCESS": TaskStatus.COMPLETED,
            "FAILURE": TaskStatus.FAILED,
            "REVOKED": TaskStatus.CANCELLED
        }

        task.status = status_map.get(status, task.status)
        task.result = result
        task.error_message = error
        task.updated_at = datetime.utcnow()

        await self.save_state()

        # Trigger callbacks
        await self._trigger_callbacks(task)

        logger.info(f"Updated from Celery: {unified_id} -> {status}")

    async def get_task(self, task_id: str) -> Optional[UnifiedTask]:
        """Get a task by unified ID or Celery ID"""
        # Try unified ID first
        if task_id in self.tasks:
            return self.tasks[task_id]

        # Try Celery ID
        unified_id = self.celery_to_unified.get(task_id)
        if unified_id:
            return self.tasks.get(unified_id)

        return None

    async def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        source: Optional[str] = None,
        swarm_id: Optional[str] = None
    ) -> List[UnifiedTask]:
        """List tasks with optional filters"""
        tasks = list(self.tasks.values())

        if status:
            tasks = [t for t in tasks if t.status == status]
        if source:
            tasks = [t for t in tasks if t.source == source]
        if swarm_id:
            tasks = [t for t in tasks if t.swarm_id == swarm_id]

        return sorted(tasks, key=lambda t: t.created_at, reverse=True)

    def register_callback(
        self,
        task_id: str,
        callback: Callable[[UnifiedTask], None]
    ):
        """Register a callback for task completion"""
        if task_id not in self._callbacks:
            self._callbacks[task_id] = []
        self._callbacks[task_id].append(callback)

    async def _trigger_callbacks(self, task: UnifiedTask):
        """Trigger registered callbacks for a task"""
        callbacks = self._callbacks.get(task.task_id, [])
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(task)
                else:
                    callback(task)
            except Exception as e:
                logger.error(f"Callback error for {task.task_id}: {e}")

    def get_status_summary(self) -> Dict:
        """Get summary of task statuses"""
        summary = {status.value: 0 for status in TaskStatus}
        for task in self.tasks.values():
            summary[task.status.value] += 1

        return {
            "total": len(self.tasks),
            "by_status": summary,
            "by_source": {
                "claude_code": len([t for t in self.tasks.values() if t.source == "claude_code"]),
                "celery": len([t for t in self.tasks.values() if t.source == "celery"])
            }
        }


# Singleton instance
_task_bridge: Optional[TaskBridge] = None


async def get_task_bridge() -> TaskBridge:
    """Get or create the task bridge singleton"""
    global _task_bridge
    if _task_bridge is None:
        _task_bridge = TaskBridge()
        await _task_bridge.initialize()
    return _task_bridge


# Celery signal handlers for integration
def setup_celery_signals(celery_app: Any):
    """Setup Celery signal handlers for task bridge integration"""
    from celery import signals

    @signals.task_prerun.connect
    def task_prerun_handler(task_id, task, *args, **kwargs):
        asyncio.create_task(
            get_task_bridge().then(
                lambda bridge: bridge.update_from_celery(task_id, "STARTED")
            )
        )

    @signals.task_success.connect
    def task_success_handler(sender, result, *args, **kwargs):
        asyncio.create_task(
            get_task_bridge().then(
                lambda bridge: bridge.update_from_celery(
                    sender.request.id, "SUCCESS", result=result
                )
            )
        )

    @signals.task_failure.connect
    def task_failure_handler(sender, exception, *args, **kwargs):
        asyncio.create_task(
            get_task_bridge().then(
                lambda bridge: bridge.update_from_celery(
                    sender.request.id, "FAILURE", error=str(exception)
                )
            )
        )

    logger.info("Celery signals configured for task bridge")
