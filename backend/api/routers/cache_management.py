"""
Cache Management API Router

This module provides endpoints for monitoring and managing the comprehensive caching system,
including performance metrics, cost analysis, and cache operations.
"""

from fastapi import APIRouter, Query, HTTPException, Depends, status, Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel, Field

from backend.utils.cache_monitoring import get_cache_monitor
from backend.utils.comprehensive_cache import get_cache_manager
from backend.utils.intelligent_cache_policies import get_policy_manager, get_cache_warmer
from backend.utils.database_query_cache import get_query_cache_manager
from backend.utils.api_cache_decorators import get_invalidation_manager
from backend.auth.oauth2 import get_current_admin_user
from backend.models.unified_models import User
from backend.models.api_response import ApiResponse, success_response
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


# Allowlist of cache key namespace prefixes that may be targeted by an
# invalidation pattern. These mirror the prefixes produced by the cache
# manager (see ComprehensiveCacheManager.key_prefixes). Any invalidation
# request must be anchored to exactly one of these namespaces so that user
# input can never reach an unbounded Redis SCAN (e.g. "*" flushing the whole
# keyspace). See findings #197 / #199.
ALLOWED_NAMESPACE_PREFIXES = frozenset({
    'api:resp',
    'db:query',
    'comp',
    'market',
    'user',
    'analysis',
    'quote',
    'overview',
    'technical',
    'ml_pred',
    'recommend',
    'stocks',
    'misc',
})

# Data-type aliases accepted on the invalidate endpoint, mapped to their
# canonical namespace prefix. Keeps the public API stable while still
# anchoring every invalidation to an allowlisted namespace.
DATA_TYPE_TO_PREFIX = {
    'api_response': 'api:resp',
    'db_query': 'db:query',
    'computation': 'comp',
    'market_data': 'market',
    'user_data': 'user',
    'analysis': 'analysis',
    'real_time_quote': 'quote',
    'company_overview': 'overview',
    'technical_indicators': 'technical',
    'ml_predictions': 'ml_pred',
    'recommendations': 'recommend',
    'stock_list': 'stocks',
}


def _namespace_of(pattern: str) -> Optional[str]:
    """
    Return the allowlisted namespace prefix that fully contains ``pattern``.

    A pattern is considered safe only when its leading, wildcard-free segment
    matches a known namespace prefix exactly. This rejects unrestricted
    wildcards ("*", "?*", "[a-z]*") and namespace-escaping inputs before they
    reach Redis SCAN.
    """
    if not pattern:
        return None

    # Split off the wildcard-free leading portion. Any Redis glob metacharacter
    # (*, ?, [, ]) terminates the anchor; the anchor must, on its own, equal an
    # allowed prefix so the SCAN can never widen beyond that namespace.
    anchor_chars: List[str] = []
    for ch in pattern:
        if ch in '*?[]':
            break
        anchor_chars.append(ch)
    anchor = ''.join(anchor_chars)

    if not anchor:
        # Leading wildcard -> no namespace anchor at all -> unbounded scan.
        return None

    for prefix in ALLOWED_NAMESPACE_PREFIXES:
        # The anchor must be exactly the prefix, or the prefix followed by a
        # delimiter, so "marketing*" cannot masquerade as the "market" namespace
        # while "market:*" and "market" are both accepted.
        if anchor == prefix or anchor.startswith(f"{prefix}:"):
            return prefix
    return None


def validate_invalidation_pattern(pattern: str) -> str:
    """
    Validate a user-supplied cache invalidation pattern.

    Returns the pattern unchanged when it is anchored to an allowlisted
    namespace prefix. Raises ``HTTPException`` (400) when the pattern contains
    an unrestricted leading wildcard or escapes the known namespaces, ensuring
    no user input reaches an unbounded Redis SCAN (#197).
    """
    if _namespace_of(pattern) is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Invalid cache pattern: must be anchored to one of the allowed "
                f"namespaces ({', '.join(sorted(ALLOWED_NAMESPACE_PREFIXES))}) "
                "and may not begin with an unrestricted wildcard."
            ),
        )
    return pattern


# Response models
class CacheMetricsResponse(BaseModel):
    """Cache metrics response"""
    timestamp: datetime
    hit_ratio: float
    total_requests: int
    api_calls_saved: int
    estimated_cost_savings: float
    storage_bytes: int
    
    class Config:
        from_attributes = True


class CostAnalysisResponse(BaseModel):
    """Cost analysis response"""
    current_daily_cost: float
    current_monthly_cost: float
    budget_utilization_percent: float
    api_calls_saved: int
    estimated_savings: float
    remaining_budget: float
    on_track: bool


class PerformanceReportResponse(BaseModel):
    """Performance report response"""
    overall_hit_ratio: float
    total_requests: int
    api_calls_saved: int
    cost_savings: float
    recommendations: List[Dict[str, Any]]


@router.get("/metrics")
async def get_cache_metrics(
    include_historical: bool = Query(False, description="Include historical metrics")
) -> ApiResponse[Dict[str, Any]]:
    """
    Get comprehensive cache performance metrics
    
    Returns real-time cache performance data including:
    - Hit/miss ratios for all cache layers
    - API usage statistics
    - Cost savings analysis
    - Storage utilization
    """
    try:
        cache_monitor = await get_cache_monitor()
        metrics = await cache_monitor.get_current_metrics()
        
        if include_historical:
            historical = await cache_monitor.get_historical_metrics(hours_back=24)
            metrics['historical_data'] = historical

        return success_response(data=metrics)
        
    except Exception as e:
        logger.error(f"Error retrieving cache metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving cache metrics: {str(e)}"
        )


@router.get("/cost-analysis")
async def get_cost_analysis() -> ApiResponse[CostAnalysisResponse]:
    """
    Get comprehensive cost analysis for the caching system
    
    Provides detailed breakdown of:
    - Current API costs by provider
    - Budget utilization
    - Projected monthly costs
    - Savings from caching
    """
    try:
        cache_monitor = await get_cache_monitor()
        cost_analysis = await cache_monitor.get_cost_analysis()

        return success_response(data=CostAnalysisResponse(
            current_daily_cost=cost_analysis['current_costs']['daily_cost'],
            current_monthly_cost=cost_analysis['current_costs']['monthly_cost'],
            budget_utilization_percent=cost_analysis['current_costs']['budget_utilization_percent'],
            api_calls_saved=cost_analysis['savings']['api_calls_saved'],
            estimated_savings=cost_analysis['savings']['estimated_savings'],
            remaining_budget=cost_analysis['budget_status']['remaining_budget'],
            on_track=cost_analysis['budget_status']['on_track']
        ))
        
    except Exception as e:
        logger.error(f"Error retrieving cost analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving cost analysis: {str(e)}"
        )


@router.get("/performance-report")
async def get_performance_report() -> ApiResponse[Dict[str, Any]]:
    """
    Get comprehensive performance report with optimization recommendations
    
    Includes:
    - Performance trends over time
    - Top performing cache entries
    - Optimization recommendations
    - Alert status
    """
    try:
        cache_monitor = await get_cache_monitor()
        report = await cache_monitor.get_performance_report()

        return success_response(data=report)
        
    except Exception as e:
        logger.error(f"Error generating performance report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating performance report: {str(e)}"
        )


@router.get("/api-usage")
async def get_api_usage() -> ApiResponse[Dict[str, Any]]:
    """
    Get current API usage statistics for all providers
    
    Shows usage against limits for:
    - Alpha Vantage (25 calls/day)
    - Finnhub (60 calls/minute)
    - Polygon.io (5 calls/minute)
    - NewsAPI (1000 calls/day)
    """
    try:
        policy_manager = get_policy_manager()
        
        # Get daily allocation plan
        allocation = policy_manager.optimize_daily_api_allocation()
        
        # Get remaining calls for each provider
        usage_summary = {}
        for provider in ['alpha_vantage', 'finnhub', 'polygon', 'newsapi']:
            remaining = policy_manager.get_remaining_api_calls(provider)
            usage_summary[provider] = {
                'remaining_calls': remaining,
                'allocation_plan': allocation.get(provider, [])
            }
        
        return success_response(data={
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'api_usage': usage_summary,
            'total_allocated_calls': sum(len(plan) for plan in allocation.values())
        })
        
    except Exception as e:
        logger.error(f"Error retrieving API usage: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving API usage: {str(e)}"
        )


@router.post("/invalidate")
async def invalidate_cache(
    pattern: Optional[str] = Query(None, description="Cache key pattern to invalidate (must be anchored to an allowed namespace)"),
    symbol: Optional[str] = Query(None, description="Stock symbol to invalidate"),
    data_type: Optional[str] = Query(None, description="Data type to invalidate"),
    current_user: User = Depends(get_current_admin_user),
) -> ApiResponse[Dict[str, Any]]:
    """
    Invalidate cache entries based on pattern, symbol, or data type

    This is an administrative endpoint that allows manual cache invalidation.
    Use with caution as it can impact performance temporarily.
    """
    try:
        cache_manager = await get_cache_manager()
        invalidation_manager = get_invalidation_manager()
        
        invalidated_count = 0

        if pattern:
            # Reject unbounded / namespace-escaping patterns before they reach
            # the Redis SCAN (#197).
            safe_pattern = validate_invalidation_pattern(pattern)
            await cache_manager.invalidate_pattern(safe_pattern)
            invalidated_count += 1
            logger.info(f"Invalidated cache pattern: {safe_pattern}")

        if symbol:
            await invalidation_manager.invalidate_by_symbol(symbol.upper())
            invalidated_count += 1
            logger.info(f"Invalidated cache for symbol: {symbol}")

        if data_type:
            # Map the requested data type to its canonical namespace prefix so
            # the resulting pattern is anchored (no leading wildcard) (#197).
            prefix = DATA_TYPE_TO_PREFIX.get(data_type)
            if prefix is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        "Invalid data_type: must be one of "
                        f"{', '.join(sorted(DATA_TYPE_TO_PREFIX))}."
                    ),
                )
            await cache_manager.invalidate_pattern(f"{prefix}:*")
            invalidated_count += 1
            logger.info(f"Invalidated cache for data type: {data_type}")
        
        if not any([pattern, symbol, data_type]):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Must specify at least one of: pattern, symbol, or data_type"
            )

        return success_response(data={
            "message": f"Cache invalidation completed",
            "operations": invalidated_count,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error invalidating cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error invalidating cache: {str(e)}"
        )


@router.post("/warm")
async def warm_cache(
    symbols: List[str] = Query(..., description="Stock symbols to warm in cache"),
    data_types: List[str] = Query(["real_time_quote", "company_overview"], description="Data types to warm"),
    current_user: User = Depends(get_current_admin_user),
) -> ApiResponse[Dict[str, Any]]:
    """
    Manually warm cache with specified symbols and data types

    This endpoint allows administrators to preload frequently accessed data
    into the cache to improve performance.
    """
    try:
        cache_warmer = get_cache_warmer()
        
        # Create warming tasks
        warming_tasks = []
        for symbol in symbols[:20]:  # Limit to 20 symbols to avoid overwhelming APIs
            for data_type in data_types:
                warming_tasks.append({
                    'data_type': data_type,
                    'identifier': symbol.upper(),
                    'priority': 1
                })
        
        # Execute warming
        cache_manager = await get_cache_manager()
        await cache_manager.warm_cache(warming_tasks, priority=1)

        return success_response(data={
            "message": f"Cache warming initiated for {len(symbols)} symbols",
            "symbols": [s.upper() for s in symbols],
            "data_types": data_types,
            "total_tasks": len(warming_tasks),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error warming cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error warming cache: {str(e)}"
        )


@router.get("/health")
async def get_cache_health() -> ApiResponse[Dict[str, Any]]:
    """
    Get cache system health status

    Provides a quick health check of all caching components:
    - Cache manager connectivity
    - Redis status
    - Database cache status
    - Monitoring system status
    """
    try:
        health_status = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_status": "healthy",
            "components": {}
        }
        
        # Test cache manager
        try:
            cache_manager = await get_cache_manager()
            metrics = await cache_manager.get_metrics()
            health_status["components"]["cache_manager"] = {
                "status": "healthy",
                "total_requests": metrics['cache_metrics']['total_requests'],
                "hit_ratio": metrics['cache_metrics']['hit_ratio']
            }
        except Exception as e:
            health_status["components"]["cache_manager"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["overall_status"] = "degraded"
        
        # Test Redis connectivity
        try:
            cache_manager = await get_cache_manager()
            if cache_manager.redis_client:
                await cache_manager.redis_client.ping()
                health_status["components"]["redis"] = {"status": "healthy"}
            else:
                health_status["components"]["redis"] = {"status": "not_configured"}
        except Exception as e:
            health_status["components"]["redis"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["overall_status"] = "degraded"
        
        # Test query cache
        try:
            query_cache = get_query_cache_manager()
            stats = query_cache.get_query_statistics()
            health_status["components"]["query_cache"] = {
                "status": "healthy",
                "total_queries": stats['total_queries'],
                "hit_rate": stats['hit_rate']
            }
        except Exception as e:
            health_status["components"]["query_cache"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["overall_status"] = "degraded"
        
        # Test monitoring system
        try:
            cache_monitor = await get_cache_monitor()
            health_status["components"]["monitoring"] = {
                "status": "healthy" if cache_monitor.monitoring_active else "inactive",
                "metrics_history_size": len(cache_monitor.metrics_history)
            }
        except Exception as e:
            health_status["components"]["monitoring"] = {
                "status": "unhealthy",
                "error": str(e)
            }

        return success_response(data=health_status)
        
    except Exception as e:
        logger.error(f"Error checking cache health: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error checking cache health: {str(e)}"
        )


@router.get("/statistics")
async def get_cache_statistics() -> ApiResponse[Dict[str, Any]]:
    """
    Get detailed cache statistics for analysis and debugging

    Provides comprehensive statistics about cache usage patterns,
    performance metrics, and storage utilization.
    """
    try:
        cache_manager = await get_cache_manager()
        query_cache = get_query_cache_manager()
        
        # Get comprehensive statistics
        cache_stats = await cache_manager.get_metrics()
        query_stats = query_cache.get_query_statistics()
        
        # Calculate additional derived metrics
        total_requests = cache_stats['cache_metrics']['total_requests']
        if total_requests > 0:
            l1_effectiveness = cache_stats['cache_metrics']['l1_hits'] / total_requests
            l2_effectiveness = cache_stats['cache_metrics']['l2_hits'] / total_requests
            l3_effectiveness = cache_stats['cache_metrics']['l3_hits'] / total_requests
        else:
            l1_effectiveness = l2_effectiveness = l3_effectiveness = 0

        return success_response(data={
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cache_layer_statistics": {
                "l1": {
                    "hits": cache_stats['cache_metrics']['l1_hits'],
                    "misses": cache_stats['cache_metrics']['l1_misses'],
                    "effectiveness": l1_effectiveness,
                    "stats": cache_stats['l1_cache_stats']
                },
                "l2": {
                    "hits": cache_stats['cache_metrics']['l2_hits'],
                    "misses": cache_stats['cache_metrics']['l2_misses'],
                    "effectiveness": l2_effectiveness,
                    "stats": cache_stats['l2_cache_stats']
                },
                "l3": {
                    "hits": cache_stats['cache_metrics']['l3_hits'],
                    "misses": cache_stats['cache_metrics']['l3_misses'],
                    "effectiveness": l3_effectiveness
                }
            },
            "query_cache_statistics": query_stats,
            "storage_statistics": {
                "total_bytes": cache_stats['storage_bytes'],
                "total_mb": cache_stats['storage_bytes'] / (1024 * 1024),
                "active_warming_tasks": cache_stats['active_warming_tasks']
            },
            "performance_metrics": {
                "overall_hit_ratio": cache_stats['cache_metrics']['hit_ratio'],
                "api_calls_saved": cache_stats['cache_metrics']['api_calls_saved'],
                "estimated_cost_savings": cache_stats['cache_metrics']['estimated_cost_savings']
            }
        })
        
    except Exception as e:
        logger.error(f"Error retrieving cache statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving cache statistics: {str(e)}"
        )