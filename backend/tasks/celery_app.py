"""
Celery application configuration and initialization
"""
from celery import Celery
from celery.schedules import crontab
from kombu import Exchange, Queue
import os
from datetime import timedelta

# Get configuration from environment
# Prefer REDIS_URL env var if set (for Docker), otherwise build from components
REDIS_URL_ENV = os.getenv('REDIS_URL')
if REDIS_URL_ENV:
    # Use provided REDIS_URL directly (strip db number if present, we add our own)
    REDIS_URL = REDIS_URL_ENV.rsplit('/', 1)[0] if '/' in REDIS_URL_ENV else REDIS_URL_ENV
else:
    # Build from components for local development
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = os.getenv('REDIS_PORT', '6379')
    REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', '')
    if REDIS_PASSWORD:
        REDIS_URL = f'redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}'
    else:
        REDIS_URL = f'redis://{REDIS_HOST}:{REDIS_PORT}'

# Create Celery app
celery_app = Celery(
    'investment_app',
    broker=f'{REDIS_URL}/0',
    backend=f'{REDIS_URL}/1',
    include=[
        'backend.tasks.data_tasks',
        'backend.tasks.analysis_tasks',
        'backend.tasks.portfolio_tasks',
        'backend.tasks.notification_tasks',
        'backend.tasks.maintenance_tasks'
    ]
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Result backend settings
    result_expires=3600,  # Results expire after 1 hour
    result_persistent=True,
    result_compression='gzip',
    
    # Worker settings - Optimized for 4x throughput with memory safety
    # -------------------------------------------------------------------------
    # These settings work with concurrency=4 workers in docker-compose
    #
    # Memory Budget per Container (1024MB dev / 1280MB prod):
    #   - Base Celery overhead: ~100MB
    #   - 4 workers * 200MB peak each: ~800MB
    #   - Safety margin: ~180-380MB
    # -------------------------------------------------------------------------

    # Prefetch multiplier: 1 task per worker to prevent memory spikes
    # With concurrency=4, this means max 4 tasks queued at once
    worker_prefetch_multiplier=1,

    # Worker recycling: Recycle after 25 tasks to prevent memory leaks
    # This is especially important for ML tasks that load large models
    # CLI override: --max-tasks-per-child=25
    worker_max_tasks_per_child=25,

    # Rate limits: Enabled to respect API quotas
    worker_disable_rate_limits=False,

    # Pool type: prefork provides process isolation for memory safety
    # Each worker runs in its own process, preventing shared memory issues
    # CLI override: --pool=prefork
    worker_pool='prefork',

    # Memory limit per worker child: 512MB (524288 KB)
    # Workers exceeding this are automatically recycled
    # CLI override: --max-memory-per-child=524288
    worker_max_memory_per_child=524288,  # 512MB in KB
    
    # Task execution settings
    task_track_started=True,
    task_time_limit=1800,  # 30 minutes hard limit
    task_soft_time_limit=1500,  # 25 minutes soft limit
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    
    # Queue configuration
    task_default_queue='default',
    task_default_exchange='tasks',
    task_default_exchange_type='topic',
    task_default_routing_key='task.default',
    
    # Define queues
    task_queues=(
        Queue('default', Exchange('tasks'), routing_key='task.#'),
        Queue('high_priority', Exchange('tasks'), routing_key='high.#'),
        Queue('low_priority', Exchange('tasks'), routing_key='low.#'),
        Queue('data_ingestion', Exchange('tasks'), routing_key='data.#'),
        Queue('analysis', Exchange('tasks'), routing_key='analysis.#'),
        Queue('notifications', Exchange('tasks'), routing_key='notify.#'),
    ),
    
    # Route tasks to specific queues
    task_routes={
        'backend.tasks.data_tasks.*': {'queue': 'data_ingestion'},
        'backend.tasks.analysis_tasks.*': {'queue': 'analysis'},
        'backend.tasks.notification_tasks.*': {'queue': 'notifications'},
        'backend.tasks.portfolio_tasks.rebalance_portfolio': {'queue': 'high_priority'},
        'backend.tasks.maintenance_tasks.*': {'queue': 'low_priority'},
    },
    
    # Beat scheduler configuration
    beat_schedule={
        # Data ingestion tasks - respecting API rate limits
        'fetch-market-data-morning': {
            'task': 'backend.tasks.data_tasks.fetch_all_market_data',
            'schedule': crontab(hour=9, minute=30),  # 9:30 AM EST (market open)
            'options': {'queue': 'data_ingestion'}
        },
        'fetch-market-data-midday': {
            'task': 'backend.tasks.data_tasks.fetch_all_market_data',
            'schedule': crontab(hour=12, minute=0),  # Noon
            'options': {'queue': 'data_ingestion'}
        },
        'fetch-market-data-close': {
            'task': 'backend.tasks.data_tasks.fetch_all_market_data',
            'schedule': crontab(hour=16, minute=30),  # 4:30 PM EST (after market close)
            'options': {'queue': 'data_ingestion'}
        },
        
        # Analysis tasks
        'daily-analysis': {
            'task': 'backend.tasks.analysis_tasks.run_daily_analysis',
            'schedule': crontab(hour=5, minute=0),  # 5 AM EST (pre-market)
            'options': {'queue': 'analysis'}
        },
        'update-recommendations': {
            'task': 'backend.tasks.analysis_tasks.update_all_recommendations',
            'schedule': crontab(hour='*/6'),  # Every 6 hours
            'options': {'queue': 'analysis'}
        },
        'calculate-technical-indicators': {
            'task': 'backend.tasks.analysis_tasks.calculate_all_indicators',
            'schedule': crontab(hour=17, minute=0),  # 5 PM EST
            'options': {'queue': 'analysis'}
        },
        
        # Portfolio tasks
        'update-portfolio-values': {
            'task': 'backend.tasks.portfolio_tasks.update_all_portfolio_values',
            'schedule': timedelta(minutes=15),  # Every 15 minutes during market hours
            'options': {'queue': 'default'}
        },
        'check-rebalancing': {
            'task': 'backend.tasks.portfolio_tasks.check_rebalancing_needed',
            'schedule': crontab(hour=6, minute=0, day_of_week=1),  # Monday 6 AM
            'options': {'queue': 'default'}
        },
        
        # Notification tasks
        'send-daily-summary': {
            'task': 'backend.tasks.notification_tasks.send_daily_summaries',
            'schedule': crontab(hour=7, minute=0),  # 7 AM EST
            'options': {'queue': 'notifications'}
        },
        'check-price-alerts': {
            'task': 'backend.tasks.notification_tasks.check_price_alerts',
            'schedule': timedelta(minutes=5),  # Every 5 minutes during market hours
            'options': {'queue': 'notifications'}
        },
        'check-watchlist-price-alerts': {
            'task': 'check_watchlist_price_alerts',
            'schedule': crontab(minute='*/5', hour='9-16', day_of_week='mon-fri'),  # Every 5 min during market hours
            'options': {'queue': 'analysis'}
        },
        
        # Maintenance tasks
        'cleanup-old-data': {
            'task': 'backend.tasks.maintenance_tasks.cleanup_old_data',
            'schedule': crontab(hour=2, minute=0),  # 2 AM EST
            'options': {'queue': 'low_priority'}
        },
        'optimize-database': {
            'task': 'backend.tasks.maintenance_tasks.optimize_database',
            'schedule': crontab(hour=3, minute=0, day_of_week=0),  # Sunday 3 AM
            'options': {'queue': 'low_priority'}
        },
        'backup-database': {
            'task': 'backend.tasks.maintenance_tasks.backup_database',
            'schedule': crontab(hour=1, minute=0),  # Daily at 1 AM
            'options': {'queue': 'low_priority'}
        },
        'generate-reports': {
            'task': 'backend.tasks.maintenance_tasks.generate_system_reports',
            'schedule': crontab(hour=0, minute=0, day_of_month=1),  # First day of month
            'options': {'queue': 'low_priority'}
        },
    },
    
    # Celery beat settings
    beat_scheduler='celery.beat:PersistentScheduler',
    beat_schedule_filename='celerybeat-schedule.db',
    beat_sync_every=10,
    
    # Error handling
    task_annotations={
        '*': {'rate_limit': '100/m'},  # Default rate limit
        'backend.tasks.data_tasks.fetch_stock_data': {'rate_limit': '5/m'},  # Respect API limits
    },
)

# Task priorities
class TaskPriority:
    LOW = 0
    NORMAL = 5
    HIGH = 9
    CRITICAL = 10

# Custom task base class with memory management
from celery import Task
import gc
import logging

logger = logging.getLogger(__name__)


class BaseTask(Task):
    """Base task with additional logging, error handling, and memory management"""

    autoretry_for = (Exception,)
    retry_kwargs = {'max_retries': 3, 'countdown': 60}

    def before_start(self, task_id, args, kwargs):
        """Called before task execution"""
        logger.info(f"Starting task {self.name} with id {task_id}")

    def after_return(self, status, retval, task_id, args, kwargs, einfo):
        """Called after task execution - includes memory cleanup"""
        logger.info(f"Task {self.name} with id {task_id} completed with status {status}")
        # Force garbage collection after each task to prevent memory accumulation
        # This is especially important for ML tasks that load large models
        gc.collect()

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called on task failure"""
        logger.error(f"Task {self.name} with id {task_id} failed: {exc}")
        # Force cleanup on failure too
        gc.collect()

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Called when task is retried"""
        logger.warning(f"Retrying task {self.name} with id {task_id} due to {exc}")
        # Cleanup before retry
        gc.collect()

# Register base task
celery_app.Task = BaseTask


# =============================================================================
# Memory Monitoring and Worker Signals
# =============================================================================
from celery.signals import worker_ready, worker_shutdown, task_postrun, task_failure
import resource


def get_memory_usage_mb():
    """Get current memory usage in MB"""
    try:
        # Get memory usage from /proc/self/status (Linux)
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024  # Convert KB to MB
    except (FileNotFoundError, IOError):
        # Fallback for non-Linux systems
        try:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            return usage.ru_maxrss / 1024  # macOS returns in bytes, Linux in KB
        except Exception:
            return 0
    return 0


@worker_ready.connect
def on_worker_ready(sender, **kwargs):
    """Log when worker is ready"""
    logger.info(f"Celery worker ready. Initial memory: {get_memory_usage_mb():.1f} MB")


@worker_shutdown.connect
def on_worker_shutdown(sender, **kwargs):
    """Log when worker shuts down"""
    logger.info(f"Celery worker shutting down. Final memory: {get_memory_usage_mb():.1f} MB")


@task_postrun.connect
def on_task_postrun(sender=None, task_id=None, task=None, retval=None, state=None, **kwargs):
    """Log memory usage after each task"""
    memory_mb = get_memory_usage_mb()
    if memory_mb > 300:  # Log warning if memory exceeds 300MB
        logger.warning(f"High memory usage after task {sender.name}: {memory_mb:.1f} MB")
    elif memory_mb > 200:  # Log info if memory exceeds 200MB
        logger.info(f"Memory usage after task {sender.name}: {memory_mb:.1f} MB")


@task_failure.connect
def on_task_failure(sender=None, task_id=None, exception=None, traceback=None, **kwargs):
    """Log memory on task failure for debugging OOM issues"""
    memory_mb = get_memory_usage_mb()
    logger.error(f"Task {sender.name} failed. Memory at failure: {memory_mb:.1f} MB")


if __name__ == '__main__':
    celery_app.start()