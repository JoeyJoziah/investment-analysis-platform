"""
Comprehensive Test Suite for Cache Management API Router

Tests all cache management endpoints including:
- Metrics retrieval with historical data
- Cost analysis and budget tracking
- Performance reports with recommendations
- Cache operations (invalidation, warming)
- Health checks and statistics

Coverage target: >=80% for backend/api/routers/cache_management.py
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock
import json

from backend.tests.conftest import assert_success_response, assert_api_error_response


# ============================================================================
# FIXTURES - Mock Objects and Test Data
# ============================================================================

@pytest.fixture
def mock_cache_metrics():
    """Mock cache metrics data"""
    return {
        'timestamp': datetime.utcnow().isoformat(),
        'hit_ratio': 0.85,
        'total_requests': 10000,
        'api_calls_saved': 8500,
        'estimated_cost_savings': 425.50,
        'storage_bytes': 52428800,  # 50 MB
        'l1_hits': 4000,
        'l1_misses': 1000,
        'l2_hits': 3500,
        'l2_misses': 1000,
        'l3_hits': 1000,
        'l3_misses': 500,
        'cache_metrics': {
            'total_requests': 10000,
            'hit_ratio': 0.85,
            'l1_hits': 4000,
            'l1_misses': 1000,
            'l2_hits': 3500,
            'l2_misses': 1000,
            'l3_hits': 1000,
            'l3_misses': 500,
            'api_calls_saved': 8500,
            'estimated_cost_savings': 425.50
        },
        'l1_cache_stats': {
            'size': 10485760,
            'entries': 1500,
            'evictions': 200
        },
        'l2_cache_stats': {
            'size': 20971520,
            'entries': 3000,
            'evictions': 150
        },
        'active_warming_tasks': 5
    }


@pytest.fixture
def mock_historical_metrics():
    """Mock historical metrics (24h)"""
    metrics_list = []
    now = datetime.utcnow()
    for i in range(24):
        timestamp = now - timedelta(hours=i)
        metrics_list.append({
            'timestamp': timestamp.isoformat(),
            'hit_ratio': 0.80 + (i * 0.001),
            'total_requests': 10000 - (i * 100),
            'api_calls_saved': 8500 - (i * 85)
        })
    return metrics_list


@pytest.fixture
def mock_cost_analysis():
    """Mock cost analysis data"""
    return {
        'current_costs': {
            'daily_cost': 8.50,
            'monthly_cost': 255.00,
            'budget_utilization_percent': 42.5
        },
        'savings': {
            'api_calls_saved': 8500,
            'estimated_savings': 425.50
        },
        'budget_status': {
            'remaining_budget': 345.00,
            'on_track': True
        }
    }


@pytest.fixture
def mock_performance_report():
    """Mock performance report data"""
    return {
        'timestamp': datetime.utcnow().isoformat(),
        'overall_hit_ratio': 0.85,
        'total_requests': 10000,
        'api_calls_saved': 8500,
        'cost_savings': 425.50,
        'recommendations': [
            {
                'priority': 'high',
                'category': 'cache_strategy',
                'title': 'Increase L2 cache size',
                'description': 'L2 cache effectiveness is 70%. Increase size to improve hit ratio.',
                'estimated_improvement': '5-8% hit ratio improvement',
                'action': 'Increase L2 cache from 20MB to 30MB'
            },
            {
                'priority': 'medium',
                'category': 'warming',
                'title': 'Enable predictive cache warming',
                'description': 'Based on usage patterns, could pre-warm 15% of frequently accessed data.',
                'estimated_improvement': '3-5% hit ratio improvement',
                'action': 'Enable predictive warming for popular symbols'
            },
            {
                'priority': 'low',
                'category': 'ttl_optimization',
                'title': 'Optimize TTL for real-time quotes',
                'description': 'Current TTL of 60s could be optimized based on change frequency.',
                'estimated_improvement': '1-2% hit ratio improvement',
                'action': 'Analyze quote change frequency and adjust TTL'
            }
        ],
        'trends': {
            'last_7_days': [
                {'date': (datetime.utcnow() - timedelta(days=i)).isoformat(), 'hit_ratio': 0.82 + (i * 0.001)}
                for i in range(7)
            ],
            'direction': 'improving',
            'change_percent': 2.5
        },
        'top_cached_items': [
            {'key': 'AAPL:quote:realtime', 'hits': 5000, 'size_bytes': 1024},
            {'key': 'MSFT:quote:realtime', 'hits': 4500, 'size_bytes': 1024},
            {'key': 'GOOGL:overview:company', 'hits': 2000, 'size_bytes': 8192},
        ]
    }


@pytest.fixture
def mock_cache_health():
    """Mock cache health status"""
    return {
        'timestamp': datetime.utcnow().isoformat(),
        'overall_status': 'healthy',
        'components': {
            'cache_manager': {
                'status': 'healthy',
                'total_requests': 10000,
                'hit_ratio': 0.85
            },
            'redis': {
                'status': 'healthy'
            },
            'query_cache': {
                'status': 'healthy',
                'total_queries': 5000,
                'hit_rate': 0.78
            },
            'monitoring': {
                'status': 'healthy',
                'metrics_history_size': 1440
            }
        }
    }


@pytest.fixture
def mock_cache_statistics():
    """Mock cache statistics"""
    return {
        'timestamp': datetime.utcnow().isoformat(),
        'cache_layer_statistics': {
            'l1': {
                'hits': 4000,
                'misses': 1000,
                'effectiveness': 0.40,
                'stats': {
                    'size': 10485760,
                    'entries': 1500,
                    'evictions': 200
                }
            },
            'l2': {
                'hits': 3500,
                'misses': 1000,
                'effectiveness': 0.35,
                'stats': {
                    'size': 20971520,
                    'entries': 3000,
                    'evictions': 150
                }
            },
            'l3': {
                'hits': 1000,
                'misses': 500,
                'effectiveness': 0.10
            }
        },
        'query_cache_statistics': {
            'total_queries': 5000,
            'cache_hits': 3900,
            'hit_rate': 0.78
        },
        'storage_statistics': {
            'total_bytes': 52428800,
            'total_mb': 50.0,
            'active_warming_tasks': 5
        },
        'performance_metrics': {
            'overall_hit_ratio': 0.85,
            'api_calls_saved': 8500,
            'estimated_cost_savings': 425.50
        }
    }


@pytest.fixture
def mock_api_usage():
    """Mock API usage data"""
    return {
        'timestamp': datetime.utcnow().isoformat(),
        'api_usage': {
            'alpha_vantage': {
                'remaining_calls': 12,
                'allocation_plan': [
                    {'time': '10:00', 'calls': 5},
                    {'time': '14:00', 'calls': 5}
                ]
            },
            'finnhub': {
                'remaining_calls': 45,
                'allocation_plan': []
            },
            'polygon': {
                'remaining_calls': 4,
                'allocation_plan': []
            },
            'newsapi': {
                'remaining_calls': 800,
                'allocation_plan': []
            }
        },
        'total_allocated_calls': 10
    }


# ============================================================================
# TESTS - METRICS (3 tests)
# ============================================================================

class TestCacheMetrics:
    """Test cache metrics endpoints"""

    @pytest.mark.asyncio
    async def test_get_cache_metrics_success(self, async_client: AsyncClient, mock_cache_metrics):
        """Test successful cache metrics retrieval"""
        with patch('backend.api.routers.cache_management.get_cache_monitor') as mock_monitor:
            mock_instance = AsyncMock()
            mock_instance.get_current_metrics = AsyncMock(return_value=mock_cache_metrics)
            mock_monitor.return_value = mock_instance

            response = await async_client.get("/api/cache/metrics")
            data = assert_success_response(response)

            assert data['timestamp'] is not None
            assert data['hit_ratio'] == 0.85
            assert data['total_requests'] == 10000
            assert data['api_calls_saved'] == 8500
            assert data['estimated_cost_savings'] == 425.50
            assert data['storage_bytes'] == 52428800

    @pytest.mark.asyncio
    async def test_get_cache_metrics_with_historical(self, async_client: AsyncClient,
                                                      mock_cache_metrics, mock_historical_metrics):
        """Test cache metrics with 24-hour historical data"""
        combined_data = {**mock_cache_metrics, 'historical_data': mock_historical_metrics}

        with patch('backend.api.routers.cache_management.get_cache_monitor') as mock_monitor:
            mock_instance = AsyncMock()
            mock_instance.get_current_metrics = AsyncMock(return_value=mock_cache_metrics)
            mock_instance.get_historical_metrics = AsyncMock(return_value=mock_historical_metrics)
            mock_monitor.return_value = mock_instance

            response = await async_client.get("/api/cache/metrics", params={"include_historical": "true"})
            data = assert_success_response(response)

            assert 'historical_data' in data
            assert len(data['historical_data']) == 24
            assert data['historical_data'][0]['hit_ratio'] > 0.80

    @pytest.mark.asyncio
    async def test_cache_metrics_structure(self, async_client: AsyncClient, mock_cache_metrics):
        """Test cache metrics response structure and schema"""
        with patch('backend.api.routers.cache_management.get_cache_monitor') as mock_monitor:
            mock_instance = AsyncMock()
            mock_instance.get_current_metrics = AsyncMock(return_value=mock_cache_metrics)
            mock_monitor.return_value = mock_instance

            response = await async_client.get("/api/cache/metrics")
            data = assert_success_response(response)

            # Verify required fields
            required_fields = [
                'timestamp', 'hit_ratio', 'total_requests',
                'api_calls_saved', 'estimated_cost_savings', 'storage_bytes'
            ]
            for field in required_fields:
                assert field in data, f"Missing required field: {field}"

            # Verify data types
            assert isinstance(data['timestamp'], str)
            assert isinstance(data['hit_ratio'], float)
            assert isinstance(data['total_requests'], int)
            assert isinstance(data['api_calls_saved'], int)
            assert isinstance(data['estimated_cost_savings'], (int, float))
            assert isinstance(data['storage_bytes'], int)


# ============================================================================
# TESTS - COST ANALYSIS (3 tests)
# ============================================================================

class TestCostAnalysis:
    """Test cost analysis endpoints"""

    @pytest.mark.asyncio
    async def test_get_cost_analysis_success(self, async_client: AsyncClient, mock_cost_analysis):
        """Test successful cost analysis retrieval"""
        with patch('backend.api.routers.cache_management.get_cache_monitor') as mock_monitor:
            mock_instance = AsyncMock()
            mock_instance.get_cost_analysis = AsyncMock(return_value=mock_cost_analysis)
            mock_monitor.return_value = mock_instance

            response = await async_client.get("/api/cache/cost-analysis")
            data = assert_success_response(response)

            assert data['current_daily_cost'] == 8.50
            assert data['current_monthly_cost'] == 255.00
            assert data['budget_utilization_percent'] == 42.5
            assert data['api_calls_saved'] == 8500
            assert data['estimated_savings'] == 425.50
            assert data['remaining_budget'] == 345.00
            assert data['on_track'] is True

    @pytest.mark.asyncio
    async def test_cost_analysis_calculations(self, async_client: AsyncClient):
        """Test cost analysis calculations and derived metrics"""
        cost_data = {
            'current_costs': {
                'daily_cost': 10.00,
                'monthly_cost': 300.00,
                'budget_utilization_percent': 75.0
            },
            'savings': {
                'api_calls_saved': 10000,
                'estimated_savings': 500.00
            },
            'budget_status': {
                'remaining_budget': 100.00,
                'on_track': True
            }
        }

        with patch('backend.api.routers.cache_management.get_cache_monitor') as mock_monitor:
            mock_instance = AsyncMock()
            mock_instance.get_cost_analysis = AsyncMock(return_value=cost_data)
            mock_monitor.return_value = mock_instance

            response = await async_client.get("/api/cache/cost-analysis")
            data = assert_success_response(response)

            # Verify calculations
            assert data['current_monthly_cost'] == data['current_daily_cost'] * 30
            assert data['budget_utilization_percent'] == 75.0
            assert data['remaining_budget'] == 100.00

    @pytest.mark.asyncio
    async def test_cost_analysis_budget_alerts(self, async_client: AsyncClient):
        """Test budget alert when >80% budget used"""
        cost_data = {
            'current_costs': {
                'daily_cost': 8.00,
                'monthly_cost': 240.00,
                'budget_utilization_percent': 85.0  # Over 80% threshold
            },
            'savings': {
                'api_calls_saved': 8500,
                'estimated_savings': 425.50
            },
            'budget_status': {
                'remaining_budget': 45.00,
                'on_track': False  # Alert triggered
            }
        }

        with patch('backend.api.routers.cache_management.get_cache_monitor') as mock_monitor:
            mock_instance = AsyncMock()
            mock_instance.get_cost_analysis = AsyncMock(return_value=cost_data)
            mock_monitor.return_value = mock_instance

            response = await async_client.get("/api/cache/cost-analysis")
            data = assert_success_response(response)

            # Verify budget alert
            assert data['budget_utilization_percent'] > 80.0
            assert data['on_track'] is False


# ============================================================================
# TESTS - PERFORMANCE REPORT (3 tests)
# ============================================================================

class TestPerformanceReport:
    """Test performance report endpoints"""

    @pytest.mark.asyncio
    async def test_get_performance_report_success(self, async_client: AsyncClient, mock_performance_report):
        """Test successful comprehensive performance report"""
        with patch('backend.api.routers.cache_management.get_cache_monitor') as mock_monitor:
            mock_instance = AsyncMock()
            mock_instance.get_performance_report = AsyncMock(return_value=mock_performance_report)
            mock_monitor.return_value = mock_instance

            response = await async_client.get("/api/cache/performance-report")
            data = assert_success_response(response)

            assert data['overall_hit_ratio'] == 0.85
            assert data['total_requests'] == 10000
            assert data['api_calls_saved'] == 8500
            assert data['cost_savings'] == 425.50
            assert 'recommendations' in data

    @pytest.mark.asyncio
    async def test_performance_report_recommendations(self, async_client: AsyncClient, mock_performance_report):
        """Test performance report includes optimization suggestions"""
        with patch('backend.api.routers.cache_management.get_cache_monitor') as mock_monitor:
            mock_instance = AsyncMock()
            mock_instance.get_performance_report = AsyncMock(return_value=mock_performance_report)
            mock_monitor.return_value = mock_instance

            response = await async_client.get("/api/cache/performance-report")
            data = assert_success_response(response)

            assert len(data['recommendations']) > 0

            # Verify recommendation structure
            for rec in data['recommendations']:
                assert 'priority' in rec
                assert 'category' in rec
                assert 'title' in rec
                assert 'description' in rec
                assert rec['priority'] in ['high', 'medium', 'low']

            # Verify specific recommendations
            priorities = [r['priority'] for r in data['recommendations']]
            assert 'high' in priorities

    @pytest.mark.asyncio
    async def test_performance_report_trends(self, async_client: AsyncClient, mock_performance_report):
        """Test performance report includes time-series trend data"""
        with patch('backend.api.routers.cache_management.get_cache_monitor') as mock_monitor:
            mock_instance = AsyncMock()
            mock_instance.get_performance_report = AsyncMock(return_value=mock_performance_report)
            mock_monitor.return_value = mock_instance

            response = await async_client.get("/api/cache/performance-report")
            data = assert_success_response(response)

            assert 'trends' in data
            assert 'last_7_days' in data['trends']
            assert len(data['trends']['last_7_days']) == 7
            assert 'direction' in data['trends']
            assert 'change_percent' in data['trends']


# ============================================================================
# TESTS - CACHE OPERATIONS (3 tests)
# ============================================================================

class TestCacheOperations:
    """Test cache operation endpoints"""

    @pytest.mark.asyncio
    async def test_invalidate_cache_by_pattern(self, async_client: AsyncClient):
        """Test pattern-based cache invalidation"""
        with patch('backend.api.routers.cache_management.get_cache_manager') as mock_cache, \
             patch('backend.api.routers.cache_management.get_invalidation_manager') as mock_inv:

            mock_cache_instance = AsyncMock()
            mock_cache_instance.invalidate_pattern = AsyncMock()
            mock_cache.return_value = mock_cache_instance

            mock_inv_instance = MagicMock()
            mock_inv.return_value = mock_inv_instance

            response = await async_client.post("/api/cache/invalidate", params={"pattern": "*:quote:*"})
            data = assert_success_response(response)

            assert 'message' in data
            assert 'operations' in data
            assert data['operations'] >= 1
            assert 'timestamp' in data
            mock_cache_instance.invalidate_pattern.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalidate_cache_by_symbol(self, async_client: AsyncClient):
        """Test symbol-specific cache invalidation"""
        with patch('backend.api.routers.cache_management.get_cache_manager') as mock_cache, \
             patch('backend.api.routers.cache_management.get_invalidation_manager') as mock_inv:

            mock_cache_instance = AsyncMock()
            mock_cache.return_value = mock_cache_instance

            mock_inv_instance = AsyncMock()
            mock_inv_instance.invalidate_by_symbol = AsyncMock()
            mock_inv.return_value = mock_inv_instance

            response = await async_client.post("/api/cache/invalidate", params={"symbol": "AAPL"})
            data = assert_success_response(response)

            assert 'message' in data
            assert 'operations' in data
            assert data['operations'] >= 1
            mock_inv_instance.invalidate_by_symbol.assert_called_once_with('AAPL')

    @pytest.mark.asyncio
    async def test_warm_cache_success(self, async_client: AsyncClient):
        """Test successful cache warming"""
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        data_types = ['real_time_quote', 'company_overview']

        with patch('backend.api.routers.cache_management.get_cache_warmer') as mock_warmer, \
             patch('backend.api.routers.cache_management.get_cache_manager') as mock_cache:

            mock_warmer_instance = MagicMock()
            mock_warmer.return_value = mock_warmer_instance

            mock_cache_instance = AsyncMock()
            mock_cache_instance.warm_cache = AsyncMock()
            mock_cache.return_value = mock_cache_instance

            response = await async_client.post(
                "/api/cache/warm",
                params={
                    "symbols": symbols,
                    "data_types": data_types
                }
            )
            data = assert_success_response(response)

            assert 'message' in data
            assert len(data['symbols']) == 3
            assert len(data['data_types']) == 2
            assert data['total_tasks'] > 0
            assert 'timestamp' in data
            mock_cache_instance.warm_cache.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalidate_cache_missing_parameters(self, async_client: AsyncClient):
        """Test cache invalidation fails when no parameters specified"""
        with patch('backend.api.routers.cache_management.get_cache_manager') as mock_cache, \
             patch('backend.api.routers.cache_management.get_invalidation_manager') as mock_inv:

            mock_cache_instance = AsyncMock()
            mock_cache.return_value = mock_cache_instance

            mock_inv_instance = MagicMock()
            mock_inv.return_value = mock_inv_instance

            response = await async_client.post("/api/cache/invalidate", params={})

            assert response.status_code == 400


# ============================================================================
# TESTS - HEALTH CHECKS (3 tests)
# ============================================================================

class TestCacheHealth:
    """Test cache health check endpoints"""

    @pytest.mark.asyncio
    async def test_cache_health_all_components(self, async_client: AsyncClient, mock_cache_health):
        """Test cache health check with all components healthy"""
        with patch('backend.api.routers.cache_management.get_cache_manager') as mock_cache, \
             patch('backend.api.routers.cache_management.get_query_cache_manager') as mock_query, \
             patch('backend.api.routers.cache_management.get_cache_monitor') as mock_monitor:

            mock_cache_instance = AsyncMock()
            mock_cache_instance.get_metrics = AsyncMock(return_value={
                'cache_metrics': mock_cache_health['components']['cache_manager'],
                'storage_bytes': 52428800,
                'l1_cache_stats': {},
                'l2_cache_stats': {}
            })
            mock_cache_instance.redis_client = AsyncMock()
            mock_cache_instance.redis_client.ping = AsyncMock()
            mock_cache.return_value = mock_cache_instance

            mock_query_instance = MagicMock()
            mock_query_instance.get_query_statistics = MagicMock(
                return_value=mock_cache_health['components']['query_cache']
            )
            mock_query.return_value = mock_query_instance

            mock_monitor_instance = AsyncMock()
            mock_monitor_instance.monitoring_active = True
            mock_monitor_instance.metrics_history = list(range(1440))
            mock_monitor.return_value = mock_monitor_instance

            response = await async_client.get("/api/cache/health")
            data = assert_success_response(response)

            assert data['overall_status'] == 'healthy'
            assert 'components' in data
            assert 'cache_manager' in data['components']
            assert 'redis' in data['components']
            assert 'query_cache' in data['components']
            assert 'monitoring' in data['components']

    @pytest.mark.asyncio
    async def test_cache_health_degraded(self, async_client: AsyncClient):
        """Test cache health check with partial failure"""
        with patch('backend.api.routers.cache_management.get_cache_manager') as mock_cache, \
             patch('backend.api.routers.cache_management.get_query_cache_manager') as mock_query, \
             patch('backend.api.routers.cache_management.get_cache_monitor') as mock_monitor:

            mock_cache_instance = AsyncMock()
            mock_cache_instance.get_metrics = AsyncMock(
                side_effect=Exception("Cache manager unavailable")
            )
            mock_cache_instance.redis_client = None
            mock_cache.return_value = mock_cache_instance

            mock_query_instance = MagicMock()
            mock_query_instance.get_query_statistics = MagicMock(
                return_value={'total_queries': 0, 'hit_rate': 0}
            )
            mock_query.return_value = mock_query_instance

            mock_monitor_instance = AsyncMock()
            mock_monitor_instance.monitoring_active = False
            mock_monitor_instance.metrics_history = []
            mock_monitor.return_value = mock_monitor_instance

            response = await async_client.get("/api/cache/health")
            data = assert_success_response(response)

            # Overall status should be degraded due to cache manager failure
            assert data['overall_status'] == 'degraded'

    @pytest.mark.asyncio
    async def test_cache_statistics_detailed(self, async_client: AsyncClient, mock_cache_statistics):
        """Test detailed cache statistics with all layers"""
        with patch('backend.api.routers.cache_management.get_cache_manager') as mock_cache, \
             patch('backend.api.routers.cache_management.get_query_cache_manager') as mock_query:

            mock_cache_instance = AsyncMock()
            mock_cache_instance.get_metrics = AsyncMock(return_value={
                'cache_metrics': mock_cache_statistics['performance_metrics'],
                'storage_bytes': mock_cache_statistics['storage_statistics']['total_bytes'],
                'active_warming_tasks': 5,
                'l1_cache_stats': mock_cache_statistics['cache_layer_statistics']['l1']['stats'],
                'l2_cache_stats': mock_cache_statistics['cache_layer_statistics']['l2']['stats'],
                'l1_hits': 4000,
                'l1_misses': 1000,
                'l2_hits': 3500,
                'l2_misses': 1000,
                'l3_hits': 1000,
                'l3_misses': 500
            })
            mock_cache.return_value = mock_cache_instance

            mock_query_instance = MagicMock()
            mock_query_instance.get_query_statistics = MagicMock(
                return_value=mock_cache_statistics['query_cache_statistics']
            )
            mock_query.return_value = mock_query_instance

            response = await async_client.get("/api/cache/statistics", params={})
            data = assert_success_response(response)

            # Verify cache layer statistics
            assert 'cache_layer_statistics' in data
            assert 'l1' in data['cache_layer_statistics']
            assert 'l2' in data['cache_layer_statistics']
            assert 'l3' in data['cache_layer_statistics']

            # Verify L1 metrics
            assert data['cache_layer_statistics']['l1']['hits'] == 4000
            assert data['cache_layer_statistics']['l1']['effectiveness'] > 0

            # Verify storage statistics
            assert 'storage_statistics' in data
            assert data['storage_statistics']['total_mb'] == 50.0

            # Verify performance metrics
            assert 'performance_metrics' in data
            assert data['performance_metrics']['overall_hit_ratio'] == 0.85


# ============================================================================
# TESTS - API USAGE (Bonus endpoint)
# ============================================================================

class TestApiUsage:
    """Test API usage tracking endpoints"""

    @pytest.mark.asyncio
    async def test_get_api_usage_success(self, async_client: AsyncClient, mock_api_usage):
        """Test successful API usage retrieval"""
        with patch('backend.api.routers.cache_management.get_policy_manager') as mock_policy:
            mock_policy_instance = MagicMock()
            mock_policy_instance.optimize_daily_api_allocation = MagicMock(
                return_value={
                    'alpha_vantage': [
                        {'time': '10:00', 'calls': 5},
                        {'time': '14:00', 'calls': 5}
                    ]
                }
            )
            mock_policy_instance.get_remaining_api_calls = MagicMock(
                side_effect=lambda provider: {
                    'alpha_vantage': 12,
                    'finnhub': 45,
                    'polygon': 4,
                    'newsapi': 800
                }.get(provider, 0)
            )
            mock_policy.return_value = mock_policy_instance

            response = await async_client.get("/api/cache/api-usage")
            data = assert_success_response(response)

            assert 'timestamp' in data
            assert 'api_usage' in data
            assert 'alpha_vantage' in data['api_usage']
            assert 'finnhub' in data['api_usage']
            assert 'polygon' in data['api_usage']
            assert 'newsapi' in data['api_usage']


# ============================================================================
# TESTS - ERROR HANDLING
# ============================================================================

class TestErrorHandling:
    """Test error handling across cache management endpoints"""

    @pytest.mark.asyncio
    async def test_get_metrics_error_handling(self, async_client: AsyncClient):
        """Test error handling when cache monitor fails"""
        with patch('backend.api.routers.cache_management.get_cache_monitor') as mock_monitor:
            mock_monitor.side_effect = Exception("Cache monitor unavailable")

            response = await async_client.get("/api/cache/metrics")
            assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_get_cost_analysis_error_handling(self, async_client: AsyncClient):
        """Test error handling in cost analysis"""
        with patch('backend.api.routers.cache_management.get_cache_monitor') as mock_monitor:
            mock_monitor.side_effect = Exception("Database connection failed")

            response = await async_client.get("/api/cache/cost-analysis")
            assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_warm_cache_symbol_limit(self, async_client: AsyncClient):
        """Test cache warming respects symbol limit (max 20)"""
        symbols = [f'SYM{i}' for i in range(25)]  # 25 symbols, exceeds limit
        data_types = ['real_time_quote']

        with patch('backend.api.routers.cache_management.get_cache_warmer') as mock_warmer, \
             patch('backend.api.routers.cache_management.get_cache_manager') as mock_cache:

            mock_warmer_instance = MagicMock()
            mock_warmer.return_value = mock_warmer_instance

            mock_cache_instance = AsyncMock()
            mock_cache_instance.warm_cache = AsyncMock()
            mock_cache.return_value = mock_cache_instance

            response = await async_client.post(
                "/api/cache/warm",
                params={"symbols": symbols, "data_types": data_types}
            )
            data = assert_success_response(response)

            # Should be limited to 20 symbols
            warming_call = mock_cache_instance.warm_cache.call_args
            if warming_call:
                warming_tasks = warming_call[0][0]
                # Max 20 symbols * 1 data type = 20 tasks
                assert len(warming_tasks) <= 20


# ============================================================================
# TESTS - API RESPONSE WRAPPER VALIDATION
# ============================================================================

class TestApiResponseWrapper:
    """Test ApiResponse wrapper compliance"""

    @pytest.mark.asyncio
    async def test_metrics_response_wrapper(self, async_client: AsyncClient, mock_cache_metrics):
        """Test metrics response uses ApiResponse wrapper correctly"""
        with patch('backend.api.routers.cache_management.get_cache_monitor') as mock_monitor:
            mock_instance = AsyncMock()
            mock_instance.get_current_metrics = AsyncMock(return_value=mock_cache_metrics)
            mock_monitor.return_value = mock_instance

            response = await async_client.get("/api/cache/metrics")

            # Verify wrapper structure
            json_data = response.json()
            assert 'success' in json_data
            assert json_data['success'] is True
            assert 'data' in json_data
            assert json_data['data'] is not None

    @pytest.mark.asyncio
    async def test_cost_analysis_response_wrapper(self, async_client: AsyncClient, mock_cost_analysis):
        """Test cost analysis response uses ApiResponse wrapper correctly"""
        with patch('backend.api.routers.cache_management.get_cache_monitor') as mock_monitor:
            mock_instance = AsyncMock()
            mock_instance.get_cost_analysis = AsyncMock(return_value=mock_cost_analysis)
            mock_monitor.return_value = mock_instance

            response = await async_client.get("/api/cache/cost-analysis")

            json_data = response.json()
            assert json_data['success'] is True
            assert 'data' in json_data

    @pytest.mark.asyncio
    async def test_invalidate_response_wrapper(self, async_client: AsyncClient):
        """Test cache invalidation response uses ApiResponse wrapper correctly"""
        with patch('backend.api.routers.cache_management.get_cache_manager') as mock_cache, \
             patch('backend.api.routers.cache_management.get_invalidation_manager') as mock_inv:

            mock_cache_instance = AsyncMock()
            mock_cache_instance.invalidate_pattern = AsyncMock()
            mock_cache.return_value = mock_cache_instance

            mock_inv_instance = MagicMock()
            mock_inv.return_value = mock_inv_instance

            response = await async_client.post("/api/cache/invalidate", params={"pattern": "*:quote:*"})

            json_data = response.json()
            assert json_data['success'] is True
            assert 'data' in json_data
            assert 'message' in json_data['data']


# ============================================================================
# EDGE CASES & BOUNDARY TESTS
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    @pytest.mark.asyncio
    async def test_zero_hit_ratio(self, async_client: AsyncClient):
        """Test metrics with zero hit ratio"""
        metrics_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'hit_ratio': 0.0,
            'total_requests': 0,
            'api_calls_saved': 0,
            'estimated_cost_savings': 0.0,
            'storage_bytes': 0
        }

        with patch('backend.api.routers.cache_management.get_cache_monitor') as mock_monitor:
            mock_instance = AsyncMock()
            mock_instance.get_current_metrics = AsyncMock(return_value=metrics_data)
            mock_monitor.return_value = mock_instance

            response = await async_client.get("/api/cache/metrics")
            data = assert_success_response(response)

            assert data['hit_ratio'] == 0.0
            assert data['total_requests'] == 0

    @pytest.mark.asyncio
    async def test_high_hit_ratio(self, async_client: AsyncClient):
        """Test metrics with maximum hit ratio"""
        metrics_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'hit_ratio': 1.0,
            'total_requests': 50000,
            'api_calls_saved': 50000,
            'estimated_cost_savings': 2500.0,
            'storage_bytes': 104857600
        }

        with patch('backend.api.routers.cache_management.get_cache_monitor') as mock_monitor:
            mock_instance = AsyncMock()
            mock_instance.get_current_metrics = AsyncMock(return_value=metrics_data)
            mock_monitor.return_value = mock_instance

            response = await async_client.get("/api/cache/metrics")
            data = assert_success_response(response)

            assert data['hit_ratio'] == 1.0
            assert data['api_calls_saved'] == 50000

    @pytest.mark.asyncio
    async def test_empty_recommendations(self, async_client: AsyncClient):
        """Test performance report with no recommendations"""
        report_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'overall_hit_ratio': 0.95,
            'total_requests': 10000,
            'api_calls_saved': 9500,
            'cost_savings': 475.0,
            'recommendations': [],
            'trends': {'last_7_days': [], 'direction': 'stable', 'change_percent': 0}
        }

        with patch('backend.api.routers.cache_management.get_cache_monitor') as mock_monitor:
            mock_instance = AsyncMock()
            mock_instance.get_performance_report = AsyncMock(return_value=report_data)
            mock_monitor.return_value = mock_instance

            response = await async_client.get("/api/cache/performance-report")
            data = assert_success_response(response)

            assert len(data['recommendations']) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
