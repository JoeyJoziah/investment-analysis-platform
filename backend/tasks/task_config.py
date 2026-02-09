"""
Celery task configuration and optimization

This module provides centralized configuration for Celery tasks including:
- Priority queue routing
- Retry configuration with exponential backoff
- Task result caching
- Rate limiting per task type
- Task monitoring utilities
"""
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import logging
import json
from enum import IntEnum

try:
    from celery import Task
    from celery.exceptions import MaxRetriesExceededError
    from kombu import Queue, Exchange
except ImportError:
    Task = object
    MaxRetriesExceededError = Exception
    Queue = None
    Exchange = None

try:
    from redis import Redis
except ImportError:
    Redis = None

try:
    from backend.utils.cache import get_redis_client
except ImportError:
    get_redis_client = None

logger = logging.getLogger(__name__)


# Priority Levels
class TaskPriority(IntEnum):
    """Task priority levels for queue routing"""
    CRITICAL = 10  # User-facing actions, alerts
    HIGH = 7       # Real-time data updates
    NORMAL = 5     # Regular analysis tasks
    LOW = 3        # Batch processing, cleanup
    BACKGROUND = 1 # Maintenance, reports


# Queue Configuration
if Queue is not None and Exchange is not None:
    CELERY_TASK_QUEUES = (
        Queue('critical', Exchange('critical'), routing_key='critical', priority=10),
        Queue('high', Exchange('high'), routing_key='high', priority=7),
        Queue('default', Exchange('default'), routing_key='default', priority=5),
        Queue('low', Exchange('low'), routing_key='low', priority=3),
        Queue('background', Exchange('background'), routing_key='background', priority=1),
    )
else:
    CELERY_TASK_QUEUES = ()

CELERY_TASK_ROUTES = {
    # Critical priority - User-facing operations
    'backend.tasks.notification_tasks.send_alert_notification': {
        'queue': 'critical',
        'priority': TaskPriority.CRITICAL
    },
    'backend.tasks.portfolio_tasks.execute_order': {
        'queue': 'critical',
        'priority': TaskPriority.CRITICAL
    },

    # High priority - Real-time updates
    'backend.tasks.data_tasks.fetch_stock_data': {
        'queue': 'high',
        'priority': TaskPriority.HIGH
    },
    'backend.tasks.portfolio_tasks.update_portfolio_value': {
        'queue': 'high',
        'priority': TaskPriority.HIGH
    },
    'backend.tasks.analysis_tasks.check_watchlist_price_alerts': {
        'queue': 'high',
        'priority': TaskPriority.HIGH
    },

    # Normal priority - Regular tasks
    'backend.tasks.analysis_tasks.analyze_stock': {
        'queue': 'default',
        'priority': TaskPriority.NORMAL
    },
    'backend.tasks.data_tasks.fetch_fundamental_data': {
        'queue': 'default',
        'priority': TaskPriority.NORMAL
    },
    'backend.tasks.notification_tasks.check_price_alerts': {
        'queue': 'default',
        'priority': TaskPriority.NORMAL
    },

    # Low priority - Batch operations
    'backend.tasks.analysis_tasks.run_daily_analysis': {
        'queue': 'low',
        'priority': TaskPriority.LOW
    },
    'backend.tasks.data_tasks.fetch_all_market_data': {
        'queue': 'low',
        'priority': TaskPriority.LOW
    },
    'backend.tasks.portfolio_tasks.update_all_portfolio_values': {
        'queue': 'low',
        'priority': TaskPriority.LOW
    },

    # Background priority - Maintenance
    'backend.tasks.maintenance_tasks.cleanup_old_data': {
        'queue': 'background',
        'priority': TaskPriority.BACKGROUND
    },
    'backend.tasks.maintenance_tasks.backup_database': {
        'queue': 'background',
        'priority': TaskPriority.BACKGROUND
    },
    'backend.tasks.maintenance_tasks.optimize_database': {
        'queue': 'background',
        'priority': TaskPriority.BACKGROUND
    },
    'backend.tasks.notification_tasks.send_daily_summaries': {
        'queue': 'background',
        'priority': TaskPriority.BACKGROUND
    },
}


# Retry Configuration
RETRY_CONFIGS = {
    # Critical tasks - retry quickly but not many times
    'critical': {
        'max_retries': 3,
        'retry_backoff': True,
        'retry_backoff_max': 300,  # 5 minutes max
        'retry_jitter': True,
        'autoretry_for': (Exception,),
        'retry_kwargs': {'countdown': 10}
    },

    # High priority - moderate retries
    'high': {
        'max_retries': 5,
        'retry_backoff': True,
        'retry_backoff_max': 600,  # 10 minutes max
        'retry_jitter': True,
        'autoretry_for': (Exception,),
        'retry_kwargs': {'countdown': 30}
    },

    # Normal priority - standard retries
    'normal': {
        'max_retries': 5,
        'retry_backoff': True,
        'retry_backoff_max': 1800,  # 30 minutes max
        'retry_jitter': True,
        'autoretry_for': (Exception,),
        'retry_kwargs': {'countdown': 60}
    },

    # Low priority - many retries with longer backoff
    'low': {
        'max_retries': 10,
        'retry_backoff': True,
        'retry_backoff_max': 3600,  # 1 hour max
        'retry_jitter': True,
        'autoretry_for': (Exception,),
        'retry_kwargs': {'countdown': 120}
    },

    # Background - very patient retries
    'background': {
        'max_retries': 15,
        'retry_backoff': True,
        'retry_backoff_max': 7200,  # 2 hours max
        'retry_jitter': True,
        'autoretry_for': (Exception,),
        'retry_kwargs': {'countdown': 300}
    }
}


# Rate Limiting Configuration
RATE_LIMITS = {
    # API data fetching - respect external API limits
    'fetch_stock_data': '60/m',  # 60 per minute
    'fetch_fundamental_data': '10/m',  # 10 per minute
    'fetch_news_data': '30/m',  # 30 per minute

    # Analysis tasks
    'analyze_stock': '100/m',  # 100 per minute
    'calculate_all_indicators': '10/h',  # 10 per hour (heavy operation)

    # Notifications
    'send_email': '100/m',  # 100 per minute
    'send_alert_notification': '200/m',  # 200 per minute

    # Portfolio updates
    'update_portfolio_value': '1000/m',  # 1000 per minute
    'execute_order': '500/m',  # 500 per minute

    # Maintenance
    'cleanup_old_data': '1/d',  # Once per day
    'backup_database': '1/d',  # Once per day
    'optimize_database': '1/w',  # Once per week
}


# Task Result Caching Configuration
CACHE_CONFIGS = {
    # Cache results for 1 hour by default
    'default_ttl': 3600,

    # Task-specific cache TTLs (in seconds)
    'ttls': {
        'analyze_stock': 3600,  # 1 hour
        'fetch_stock_data': 300,  # 5 minutes
        'fetch_fundamental_data': 86400,  # 24 hours
        'calculate_portfolio_performance': 3600,  # 1 hour
        'update_portfolio_value': 300,  # 5 minutes
        'check_system_health': 600,  # 10 minutes
    },

    # Cache key prefix
    'prefix': 'celery_result'
}


# Task Time Limits
TIME_LIMITS = {
    # Soft timeout (raises SoftTimeLimitExceeded)
    'soft': {
        'default': 300,  # 5 minutes
        'analyze_stock': 180,  # 3 minutes
        'run_daily_analysis': 1800,  # 30 minutes
        'fetch_all_market_data': 600,  # 10 minutes
        'backup_database': 3600,  # 1 hour
        'optimize_database': 1800,  # 30 minutes
    },

    # Hard timeout (kills task)
    'hard': {
        'default': 600,  # 10 minutes
        'run_daily_analysis': 3600,  # 1 hour
        'backup_database': 7200,  # 2 hours
        'optimize_database': 3600,  # 1 hour
    }
}


# Serialization Configuration
CELERY_TASK_SERIALIZER = 'json'  # Use JSON instead of pickle for security
CELERY_RESULT_SERIALIZER = 'json'
CELERY_ACCEPT_CONTENT = ['json']  # Only accept JSON
CELERY_TIMEZONE = 'UTC'
CELERY_ENABLE_UTC = True


class CachedTask(Task):
    """
    Base task class with result caching support

    Automatically caches task results in Redis for configured TTL
    """

    def __call__(self, *args, **kwargs):
        """Execute task with caching"""
        # Generate cache key
        cache_key = self._get_cache_key(args, kwargs)

        # Check cache
        redis_client = get_redis_client()
        cached_result = redis_client.get(cache_key)

        if cached_result:
            try:
                logger.info(f"Cache hit for task {self.name}")
                return json.loads(cached_result)
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Invalid cached result for {self.name}, recomputing")

        # Execute task
        result = super().__call__(*args, **kwargs)

        # Cache result
        if result and not isinstance(result, dict) or not result.get('error'):
            ttl = CACHE_CONFIGS['ttls'].get(
                self.name.split('.')[-1],
                CACHE_CONFIGS['default_ttl']
            )
            try:
                redis_client.setex(
                    cache_key,
                    ttl,
                    json.dumps(result, default=str)
                )
                logger.info(f"Cached result for task {self.name} (TTL: {ttl}s)")
            except (TypeError, ValueError) as e:
                logger.warning(f"Could not cache result for {self.name}: {e}")

        return result

    def _get_cache_key(self, args: tuple, kwargs: dict) -> str:
        """Generate cache key from task name and arguments"""
        key_parts = [
            CACHE_CONFIGS['prefix'],
            self.name,
            json.dumps(args, sort_keys=True, default=str),
            json.dumps(kwargs, sort_keys=True, default=str)
        ]
        return ':'.join(key_parts)


class MonitoredTask(Task):
    """
    Base task class with execution monitoring

    Tracks execution time, failure rate, and stores metrics
    """

    def __call__(self, *args, **kwargs):
        """Execute task with monitoring"""
        start_time = datetime.now(timezone.utc)
        task_name = self.name.split('.')[-1]

        try:
            # Execute task
            result = super().__call__(*args, **kwargs)

            # Record success
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self._record_execution(task_name, execution_time, success=True)

            return result

        except Exception as e:
            # Record failure
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self._record_execution(task_name, execution_time, success=False, error=str(e))
            raise

    def _record_execution(
        self,
        task_name: str,
        execution_time: float,
        success: bool,
        error: Optional[str] = None
    ):
        """Record task execution metrics"""
        redis_client = get_redis_client()
        timestamp = datetime.now(timezone.utc).isoformat()

        # Store execution record
        execution_key = f"task_execution:{task_name}:{timestamp}"
        execution_data = {
            'task_name': task_name,
            'execution_time': execution_time,
            'success': success,
            'error': error,
            'timestamp': timestamp
        }

        try:
            redis_client.setex(
                execution_key,
                86400,  # Keep for 24 hours
                json.dumps(execution_data)
            )

            # Update aggregated metrics
            self._update_metrics(redis_client, task_name, execution_time, success)

        except Exception as e:
            logger.error(f"Error recording task execution: {e}")

    def _update_metrics(
        self,
        redis_client: Redis,
        task_name: str,
        execution_time: float,
        success: bool
    ):
        """Update aggregated task metrics"""
        metrics_key = f"task_metrics:{task_name}"

        try:
            # Get current metrics
            metrics_json = redis_client.get(metrics_key)
            if metrics_json:
                metrics = json.loads(metrics_json)
            else:
                metrics = {
                    'total_executions': 0,
                    'successful_executions': 0,
                    'failed_executions': 0,
                    'total_execution_time': 0,
                    'avg_execution_time': 0,
                    'min_execution_time': float('inf'),
                    'max_execution_time': 0,
                    'last_updated': None
                }

            # Update metrics
            metrics['total_executions'] += 1
            if success:
                metrics['successful_executions'] += 1
            else:
                metrics['failed_executions'] += 1

            metrics['total_execution_time'] += execution_time
            metrics['avg_execution_time'] = (
                metrics['total_execution_time'] / metrics['total_executions']
            )
            metrics['min_execution_time'] = min(
                metrics['min_execution_time'],
                execution_time
            )
            metrics['max_execution_time'] = max(
                metrics['max_execution_time'],
                execution_time
            )
            metrics['last_updated'] = datetime.now(timezone.utc).isoformat()

            # Calculate failure rate
            metrics['failure_rate'] = (
                metrics['failed_executions'] / metrics['total_executions']
                if metrics['total_executions'] > 0 else 0
            )

            # Store updated metrics
            redis_client.setex(
                metrics_key,
                86400,  # Keep for 24 hours
                json.dumps(metrics)
            )

        except Exception as e:
            logger.error(f"Error updating task metrics: {e}")


class TaskMonitor:
    """Utility class for monitoring task performance"""

    def __init__(self):
        self.redis_client = get_redis_client()

    def get_task_metrics(self, task_name: str) -> Optional[Dict[str, Any]]:
        """Get metrics for a specific task"""
        metrics_key = f"task_metrics:{task_name}"

        try:
            metrics_json = self.redis_client.get(metrics_key)
            if metrics_json:
                return json.loads(metrics_json)
            return None
        except Exception as e:
            logger.error(f"Error getting task metrics: {e}")
            return None

    def get_all_task_metrics(self) -> List[Dict[str, Any]]:
        """Get metrics for all tasks"""
        try:
            pattern = "task_metrics:*"
            keys = self.redis_client.keys(pattern)

            all_metrics = []
            for key in keys:
                task_name = key.decode('utf-8').replace('task_metrics:', '')
                metrics = self.get_task_metrics(task_name)
                if metrics:
                    metrics['task_name'] = task_name
                    all_metrics.append(metrics)

            return sorted(
                all_metrics,
                key=lambda x: x.get('total_executions', 0),
                reverse=True
            )
        except Exception as e:
            logger.error(f"Error getting all task metrics: {e}")
            return []

    def get_queue_depth(self, queue_name: str = 'default') -> int:
        """Get current depth of a task queue"""
        try:
            from backend.tasks.celery_app import celery_app
            inspect = celery_app.control.inspect()

            # Get reserved tasks
            reserved = inspect.reserved()
            if not reserved:
                return 0

            total = 0
            for worker, tasks in reserved.items():
                for task in tasks:
                    if task.get('delivery_info', {}).get('routing_key') == queue_name:
                        total += 1

            return total
        except Exception as e:
            logger.error(f"Error getting queue depth: {e}")
            return -1

    def get_task_failure_rate(self, task_name: str) -> float:
        """Get failure rate for a specific task"""
        metrics = self.get_task_metrics(task_name)
        if metrics:
            return metrics.get('failure_rate', 0.0)
        return 0.0

    def get_slow_tasks(self, threshold_seconds: float = 60.0) -> List[Dict[str, Any]]:
        """Get tasks with average execution time above threshold"""
        all_metrics = self.get_all_task_metrics()

        slow_tasks = [
            m for m in all_metrics
            if m.get('avg_execution_time', 0) > threshold_seconds
        ]

        return sorted(
            slow_tasks,
            key=lambda x: x.get('avg_execution_time', 0),
            reverse=True
        )

    def get_failing_tasks(self, min_failure_rate: float = 0.1) -> List[Dict[str, Any]]:
        """Get tasks with failure rate above threshold"""
        all_metrics = self.get_all_task_metrics()

        failing_tasks = [
            m for m in all_metrics
            if m.get('failure_rate', 0) > min_failure_rate
            and m.get('total_executions', 0) > 10  # Minimum sample size
        ]

        return sorted(
            failing_tasks,
            key=lambda x: x.get('failure_rate', 0),
            reverse=True
        )


def get_retry_config(priority: str = 'normal') -> Dict[str, Any]:
    """
    Get retry configuration for a given priority level

    Args:
        priority: Priority level (critical, high, normal, low, background)

    Returns:
        Dict with retry configuration
    """
    return RETRY_CONFIGS.get(priority, RETRY_CONFIGS['normal'])


def get_rate_limit(task_name: str) -> Optional[str]:
    """
    Get rate limit for a specific task

    Args:
        task_name: Name of the task

    Returns:
        Rate limit string (e.g., '60/m') or None
    """
    return RATE_LIMITS.get(task_name)


def apply_task_config(task_func, priority: str = 'normal', **extra_config):
    """
    Apply standard configuration to a Celery task

    Args:
        task_func: Task function to configure
        priority: Priority level
        **extra_config: Additional configuration options

    Returns:
        Configured task
    """
    config = get_retry_config(priority)
    config.update(extra_config)

    return task_func(**config)
