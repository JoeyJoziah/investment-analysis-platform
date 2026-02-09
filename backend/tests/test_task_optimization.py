"""
Tests for Celery task optimization configuration

Tests priority routing, retry configuration, caching, rate limiting,
and monitoring utilities.
"""
import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone, timedelta

try:
    from celery import Task
    HAS_CELERY = True
except ImportError:
    Task = object
    HAS_CELERY = False

try:
    from redis import Redis
    HAS_REDIS = True
except ImportError:
    Redis = None
    HAS_REDIS = False

requires_celery = pytest.mark.skipif(not HAS_CELERY, reason="celery not installed")

from backend.tasks.task_config import (
    TaskPriority,
    CELERY_TASK_ROUTES,
    CELERY_TASK_QUEUES,
    RETRY_CONFIGS,
    RATE_LIMITS,
    CACHE_CONFIGS,
    TIME_LIMITS,
    CachedTask,
    MonitoredTask,
    TaskMonitor,
    get_retry_config,
    get_rate_limit,
    apply_task_config
)


@requires_celery
class TestPriorityRouting:
    """Test priority queue routing configuration"""

    def test_queue_definitions(self):
        """Test that all priority queues are defined"""
        assert len(CELERY_TASK_QUEUES) == 5

        queue_names = [q.name for q in CELERY_TASK_QUEUES]
        assert 'critical' in queue_names
        assert 'high' in queue_names
        assert 'default' in queue_names
        assert 'low' in queue_names
        assert 'background' in queue_names

    def test_queue_priorities(self):
        """Test that queues have correct priority levels"""
        queue_dict = {q.name: q.priority for q in CELERY_TASK_QUEUES}

        assert queue_dict['critical'] == 10
        assert queue_dict['high'] == 7
        assert queue_dict['default'] == 5
        assert queue_dict['low'] == 3
        assert queue_dict['background'] == 1

    def test_critical_task_routing(self):
        """Test that critical tasks are routed to critical queue"""
        critical_tasks = [
            'backend.tasks.notification_tasks.send_alert_notification',
            'backend.tasks.portfolio_tasks.execute_order'
        ]

        for task_name in critical_tasks:
            route = CELERY_TASK_ROUTES.get(task_name)
            assert route is not None
            assert route['queue'] == 'critical'
            assert route['priority'] == TaskPriority.CRITICAL

    def test_high_priority_task_routing(self):
        """Test that high priority tasks are routed correctly"""
        high_priority_tasks = [
            'backend.tasks.data_tasks.fetch_stock_data',
            'backend.tasks.portfolio_tasks.update_portfolio_value'
        ]

        for task_name in high_priority_tasks:
            route = CELERY_TASK_ROUTES.get(task_name)
            assert route is not None
            assert route['queue'] == 'high'
            assert route['priority'] == TaskPriority.HIGH

    def test_background_task_routing(self):
        """Test that background tasks are routed to background queue"""
        background_tasks = [
            'backend.tasks.maintenance_tasks.cleanup_old_data',
            'backend.tasks.maintenance_tasks.backup_database'
        ]

        for task_name in background_tasks:
            route = CELERY_TASK_ROUTES.get(task_name)
            assert route is not None
            assert route['queue'] == 'background'
            assert route['priority'] == TaskPriority.BACKGROUND


class TestRetryConfiguration:
    """Test retry configuration for different priority levels"""

    def test_retry_configs_exist(self):
        """Test that retry configs exist for all priorities"""
        assert 'critical' in RETRY_CONFIGS
        assert 'high' in RETRY_CONFIGS
        assert 'normal' in RETRY_CONFIGS
        assert 'low' in RETRY_CONFIGS
        assert 'background' in RETRY_CONFIGS

    def test_critical_retry_config(self):
        """Test critical priority retry configuration"""
        config = RETRY_CONFIGS['critical']

        assert config['max_retries'] == 3
        assert config['retry_backoff'] is True
        assert config['retry_backoff_max'] == 300
        assert config['retry_jitter'] is True
        assert config['retry_kwargs']['countdown'] == 10

    def test_normal_retry_config(self):
        """Test normal priority retry configuration"""
        config = RETRY_CONFIGS['normal']

        assert config['max_retries'] == 5
        assert config['retry_backoff'] is True
        assert config['retry_backoff_max'] == 1800
        assert config['retry_jitter'] is True
        assert config['retry_kwargs']['countdown'] == 60

    def test_background_retry_config(self):
        """Test background priority retry configuration"""
        config = RETRY_CONFIGS['background']

        assert config['max_retries'] == 15
        assert config['retry_backoff'] is True
        assert config['retry_backoff_max'] == 7200
        assert config['retry_kwargs']['countdown'] == 300

    def test_get_retry_config_function(self):
        """Test get_retry_config helper function"""
        critical_config = get_retry_config('critical')
        assert critical_config['max_retries'] == 3

        normal_config = get_retry_config('normal')
        assert normal_config['max_retries'] == 5

        # Test default fallback
        default_config = get_retry_config('nonexistent')
        assert default_config == RETRY_CONFIGS['normal']

    def test_retry_backoff_increases(self):
        """Test that retry configs have increasing backoff times by priority"""
        assert RETRY_CONFIGS['critical']['retry_backoff_max'] < \
               RETRY_CONFIGS['normal']['retry_backoff_max']
        assert RETRY_CONFIGS['normal']['retry_backoff_max'] < \
               RETRY_CONFIGS['background']['retry_backoff_max']


class TestRateLimiting:
    """Test rate limiting configuration"""

    def test_rate_limits_exist(self):
        """Test that rate limits are defined for key tasks"""
        assert 'fetch_stock_data' in RATE_LIMITS
        assert 'send_email' in RATE_LIMITS
        assert 'analyze_stock' in RATE_LIMITS
        assert 'backup_database' in RATE_LIMITS

    def test_api_rate_limits(self):
        """Test rate limits for external API calls"""
        assert RATE_LIMITS['fetch_stock_data'] == '60/m'
        assert RATE_LIMITS['fetch_fundamental_data'] == '10/m'
        assert RATE_LIMITS['fetch_news_data'] == '30/m'

    def test_notification_rate_limits(self):
        """Test rate limits for notifications"""
        assert RATE_LIMITS['send_email'] == '100/m'
        assert RATE_LIMITS['send_alert_notification'] == '200/m'

    def test_maintenance_rate_limits(self):
        """Test rate limits for maintenance tasks"""
        assert RATE_LIMITS['cleanup_old_data'] == '1/d'
        assert RATE_LIMITS['backup_database'] == '1/d'
        assert RATE_LIMITS['optimize_database'] == '1/w'

    def test_get_rate_limit_function(self):
        """Test get_rate_limit helper function"""
        rate_limit = get_rate_limit('fetch_stock_data')
        assert rate_limit == '60/m'

        # Test nonexistent task
        no_limit = get_rate_limit('nonexistent_task')
        assert no_limit is None


class TestResultCaching:
    """Test task result caching configuration"""

    def test_cache_config_exists(self):
        """Test that cache configuration is defined"""
        assert 'default_ttl' in CACHE_CONFIGS
        assert 'ttls' in CACHE_CONFIGS
        assert 'prefix' in CACHE_CONFIGS

    def test_default_ttl(self):
        """Test default cache TTL is 1 hour"""
        assert CACHE_CONFIGS['default_ttl'] == 3600

    def test_task_specific_ttls(self):
        """Test task-specific cache TTLs"""
        ttls = CACHE_CONFIGS['ttls']

        assert ttls['analyze_stock'] == 3600  # 1 hour
        assert ttls['fetch_stock_data'] == 300  # 5 minutes
        assert ttls['fetch_fundamental_data'] == 86400  # 24 hours

    def test_cache_prefix(self):
        """Test cache key prefix is set"""
        assert CACHE_CONFIGS['prefix'] == 'celery_result'


@requires_celery
@pytest.mark.asyncio
class TestCachedTask:
    """Test CachedTask base class with result caching"""

    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client"""
        mock = MagicMock(spec=Redis)
        return mock

    @pytest.fixture
    def cached_task(self, mock_redis):
        """Create a CachedTask instance"""
        with patch('backend.tasks.task_config.get_redis_client', return_value=mock_redis):
            task = CachedTask()
            task.name = 'test_task'
            task.run = Mock(return_value={'result': 'success'})
            return task

    def test_cache_miss_executes_task(self, cached_task, mock_redis):
        """Test that cache miss executes the task"""
        mock_redis.get.return_value = None

        with patch('backend.tasks.task_config.get_redis_client', return_value=mock_redis):
            result = cached_task(arg1='test')

        # Task should be executed
        assert result == {'result': 'success'}
        mock_redis.get.assert_called_once()

    def test_cache_hit_returns_cached_result(self, cached_task, mock_redis):
        """Test that cache hit returns cached result without executing task"""
        cached_result = json.dumps({'result': 'cached'})
        mock_redis.get.return_value = cached_result

        with patch('backend.tasks.task_config.get_redis_client', return_value=mock_redis):
            result = cached_task(arg1='test')

        # Should return cached result
        assert result == {'result': 'cached'}
        # Task should not be executed
        cached_task.run.assert_not_called()

    def test_result_is_cached(self, cached_task, mock_redis):
        """Test that task result is cached"""
        mock_redis.get.return_value = None

        with patch('backend.tasks.task_config.get_redis_client', return_value=mock_redis):
            result = cached_task(arg1='test')

        # Result should be cached
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        assert call_args[0][1] > 0  # TTL should be positive

    def test_error_results_not_cached(self, cached_task, mock_redis):
        """Test that error results are not cached"""
        mock_redis.get.return_value = None
        cached_task.run.return_value = {'error': 'Something failed'}

        with patch('backend.tasks.task_config.get_redis_client', return_value=mock_redis):
            result = cached_task(arg1='test')

        # Error result should not be cached
        mock_redis.setex.assert_not_called()


@requires_celery
@pytest.mark.asyncio
class TestMonitoredTask:
    """Test MonitoredTask base class with execution monitoring"""

    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client"""
        mock = MagicMock(spec=Redis)
        return mock

    @pytest.fixture
    def monitored_task(self, mock_redis):
        """Create a MonitoredTask instance"""
        with patch('backend.tasks.task_config.get_redis_client', return_value=mock_redis):
            task = MonitoredTask()
            task.name = 'test_monitored_task'
            task.run = Mock(return_value={'status': 'ok'})
            return task

    def test_successful_execution_recorded(self, monitored_task, mock_redis):
        """Test that successful execution is recorded"""
        with patch('backend.tasks.task_config.get_redis_client', return_value=mock_redis):
            result = monitored_task()

        # Should record execution
        assert mock_redis.setex.called
        assert result == {'status': 'ok'}

    def test_failed_execution_recorded(self, monitored_task, mock_redis):
        """Test that failed execution is recorded"""
        monitored_task.run.side_effect = ValueError("Test error")

        with patch('backend.tasks.task_config.get_redis_client', return_value=mock_redis):
            with pytest.raises(ValueError):
                monitored_task()

        # Should record failure
        assert mock_redis.setex.called

    def test_execution_time_tracked(self, monitored_task, mock_redis):
        """Test that execution time is tracked"""
        with patch('backend.tasks.task_config.get_redis_client', return_value=mock_redis):
            monitored_task()

        # Check that execution time was recorded
        call_args = mock_redis.setex.call_args_list
        assert len(call_args) >= 1

        # Parse stored data
        stored_data = json.loads(call_args[0][0][2])
        assert 'execution_time' in stored_data
        assert stored_data['execution_time'] >= 0


class TestTaskMonitor:
    """Test TaskMonitor utility class"""

    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client"""
        mock = MagicMock(spec=Redis)
        return mock

    @pytest.fixture
    def task_monitor(self, mock_redis):
        """Create TaskMonitor instance"""
        with patch('backend.tasks.task_config.get_redis_client', return_value=mock_redis):
            return TaskMonitor()

    def test_get_task_metrics(self, task_monitor, mock_redis):
        """Test getting metrics for a specific task"""
        metrics = {
            'total_executions': 100,
            'successful_executions': 95,
            'failed_executions': 5,
            'avg_execution_time': 1.5,
            'failure_rate': 0.05
        }
        mock_redis.get.return_value = json.dumps(metrics)

        result = task_monitor.get_task_metrics('test_task')

        assert result is not None
        assert result['total_executions'] == 100
        assert result['failure_rate'] == 0.05

    def test_get_task_metrics_not_found(self, task_monitor, mock_redis):
        """Test getting metrics for nonexistent task"""
        mock_redis.get.return_value = None

        result = task_monitor.get_task_metrics('nonexistent_task')

        assert result is None

    def test_get_task_failure_rate(self, task_monitor, mock_redis):
        """Test getting failure rate for a task"""
        metrics = {
            'failure_rate': 0.15
        }
        mock_redis.get.return_value = json.dumps(metrics)

        failure_rate = task_monitor.get_task_failure_rate('test_task')

        assert failure_rate == 0.15

    def test_get_slow_tasks(self, task_monitor, mock_redis):
        """Test getting slow tasks above threshold"""
        # Mock keys and metrics
        mock_redis.keys.return_value = [
            b'task_metrics:slow_task',
            b'task_metrics:fast_task'
        ]

        def mock_get(key):
            if key == 'task_metrics:slow_task':
                return json.dumps({
                    'avg_execution_time': 120.0,
                    'total_executions': 50
                })
            elif key == 'task_metrics:fast_task':
                return json.dumps({
                    'avg_execution_time': 5.0,
                    'total_executions': 100
                })
            return None

        mock_redis.get.side_effect = mock_get

        slow_tasks = task_monitor.get_slow_tasks(threshold_seconds=60.0)

        assert len(slow_tasks) == 1
        assert slow_tasks[0]['task_name'] == 'slow_task'
        assert slow_tasks[0]['avg_execution_time'] == 120.0

    def test_get_failing_tasks(self, task_monitor, mock_redis):
        """Test getting tasks with high failure rate"""
        # Mock keys and metrics
        mock_redis.keys.return_value = [
            b'task_metrics:failing_task',
            b'task_metrics:healthy_task'
        ]

        def mock_get(key):
            if key == 'task_metrics:failing_task':
                return json.dumps({
                    'failure_rate': 0.25,
                    'total_executions': 50
                })
            elif key == 'task_metrics:healthy_task':
                return json.dumps({
                    'failure_rate': 0.01,
                    'total_executions': 100
                })
            return None

        mock_redis.get.side_effect = mock_get

        failing_tasks = task_monitor.get_failing_tasks(min_failure_rate=0.1)

        assert len(failing_tasks) == 1
        assert failing_tasks[0]['task_name'] == 'failing_task'
        assert failing_tasks[0]['failure_rate'] == 0.25


class TestTimeLimits:
    """Test task time limit configuration"""

    def test_soft_time_limits_exist(self):
        """Test that soft time limits are defined"""
        assert 'default' in TIME_LIMITS['soft']
        assert 'analyze_stock' in TIME_LIMITS['soft']
        assert 'backup_database' in TIME_LIMITS['soft']

    def test_hard_time_limits_exist(self):
        """Test that hard time limits are defined"""
        assert 'default' in TIME_LIMITS['hard']
        assert 'run_daily_analysis' in TIME_LIMITS['hard']
        assert 'backup_database' in TIME_LIMITS['hard']

    def test_hard_limit_greater_than_soft(self):
        """Test that hard limits are greater than soft limits"""
        for task in TIME_LIMITS['soft'].keys():
            if task in TIME_LIMITS['hard']:
                assert TIME_LIMITS['hard'][task] >= TIME_LIMITS['soft'][task]

    def test_default_time_limits(self):
        """Test default time limit values"""
        assert TIME_LIMITS['soft']['default'] == 300  # 5 minutes
        assert TIME_LIMITS['hard']['default'] == 600  # 10 minutes


class TestSerializationConfig:
    """Test serialization configuration for security"""

    def test_json_serialization(self):
        """Test that JSON serialization is enforced"""
        from backend.tasks.task_config import (
            CELERY_TASK_SERIALIZER,
            CELERY_RESULT_SERIALIZER,
            CELERY_ACCEPT_CONTENT
        )

        assert CELERY_TASK_SERIALIZER == 'json'
        assert CELERY_RESULT_SERIALIZER == 'json'
        assert CELERY_ACCEPT_CONTENT == ['json']

    def test_utc_timezone(self):
        """Test that UTC timezone is configured"""
        from backend.tasks.task_config import (
            CELERY_TIMEZONE,
            CELERY_ENABLE_UTC
        )

        assert CELERY_TIMEZONE == 'UTC'
        assert CELERY_ENABLE_UTC is True


class TestApplyTaskConfig:
    """Test apply_task_config helper function"""

    def test_apply_critical_config(self):
        """Test applying critical priority configuration"""
        mock_task = Mock()
        result = apply_task_config(mock_task, priority='critical')

        # Should have been called with retry config
        mock_task.assert_called_once()
        call_kwargs = mock_task.call_args[1]
        assert 'max_retries' in call_kwargs
        assert call_kwargs['max_retries'] == 3

    def test_apply_normal_config(self):
        """Test applying normal priority configuration"""
        mock_task = Mock()
        result = apply_task_config(mock_task, priority='normal')

        mock_task.assert_called_once()
        call_kwargs = mock_task.call_args[1]
        assert call_kwargs['max_retries'] == 5

    def test_apply_with_extra_config(self):
        """Test applying configuration with extra options"""
        mock_task = Mock()
        result = apply_task_config(
            mock_task,
            priority='normal',
            time_limit=120,
            soft_time_limit=60
        )

        mock_task.assert_called_once()
        call_kwargs = mock_task.call_args[1]
        assert call_kwargs['time_limit'] == 120
        assert call_kwargs['soft_time_limit'] == 60
