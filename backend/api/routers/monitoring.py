"""
Monitoring and Observability Endpoints
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List
import logging
import os
from datetime import datetime, timezone

from backend.utils.grafana_client import grafana_client
from backend.utils.auth import get_current_user
from backend.utils.cost_monitor import cost_monitor
from backend.models.api_response import ApiResponse, success_response
from backend.models.monitoring_schemas import (
    HealthCheckResponse,
    CostMetrics,
    DashboardLinks,
    AnnotationResponse,
    AlertTestResponse,
    ApiUsageMetrics,
    ProviderUsage
)

router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])

logger = logging.getLogger(__name__)


@router.get("/health")
async def health_check() -> ApiResponse[HealthCheckResponse]:
    """Complete system health check"""
    services: Dict[str, Any] = {"api": "healthy"}
    errors: Dict[str, str] = {}

    # Check Redis connectivity
    try:
        from backend.utils.cache import get_redis
        redis_client = await get_redis()
        await redis_client.ping()
        services["redis"] = "healthy"
    except Exception as exc:
        logger.warning("Redis health check failed: %s", exc)
        services["redis"] = "unhealthy"
        errors["redis"] = str(exc)

    # Check database connectivity
    try:
        from sqlalchemy import text
        from backend.config.database import db_manager
        async with db_manager.get_session() as session:
            await session.execute(text("SELECT 1"))
        services["database"] = "healthy"
    except Exception as exc:
        logger.warning("Database health check failed: %s", exc)
        services["database"] = "unhealthy"
        errors["database"] = str(exc)

    # Check Grafana (existing, does not raise)
    services["grafana"] = grafana_client.test_connection()

    # Determine overall status
    redis_ok = services["redis"] == "healthy"
    db_ok = services["database"] == "healthy"

    if redis_ok and db_ok:
        overall = "healthy"
    elif redis_ok or db_ok:
        overall = "degraded"
    else:
        overall = "unhealthy"

    response_data = HealthCheckResponse(
        status=overall,
        timestamp=datetime.now(timezone.utc).isoformat(),
        services=services,
    )

    return success_response(data=response_data)


@router.get("/metrics/cost")
async def get_cost_metrics() -> ApiResponse[CostMetrics]:
    """Get current cost tracking metrics"""
    daily_costs = await cost_monitor.get_daily_costs()
    monthly_estimate = await cost_monitor.get_monthly_estimate()

    return success_response(data=CostMetrics(
        daily_costs=daily_costs,
        monthly_estimate=monthly_estimate,
        budget_remaining=50.0 - monthly_estimate,
        budget_percentage=(monthly_estimate / 50.0) * 100
    ))


@router.get("/grafana/dashboards")
async def get_dashboard_links() -> ApiResponse[DashboardLinks]:
    """Get Grafana dashboard URLs"""
    base_url = os.getenv('GRAFANA_URL', 'http://localhost:3001')

    return success_response(data=DashboardLinks(
        main=f"{base_url}/d/investment-analysis",
        api_usage=f"{base_url}/d/api-usage",
        ml_performance=f"{base_url}/d/ml-performance",
        cost_tracking=f"{base_url}/d/cost-tracking",
        system_metrics=f"{base_url}/d/system-metrics"
    ))


@router.post("/grafana/annotation")
async def create_annotation(
    text: str,
    tags: List[str] = None,
    current_user: dict = Depends(get_current_user)
) -> ApiResponse[AnnotationResponse]:
    """Create an annotation in Grafana (for important events)"""
    success = grafana_client.create_annotation(
        text=f"[{current_user['email']}] {text}",
        tags=tags or ["user-action"]
    )

    if not success:
        raise HTTPException(status_code=500, detail="Failed to create annotation")

    return success_response(data=AnnotationResponse(message="Annotation created successfully"))


@router.post("/alerts/test")
async def test_alert_system() -> ApiResponse[AlertTestResponse]:
    """Test the alert system"""
    # Create a test alert in Grafana
    alert_created = grafana_client.create_alert(
        name="Test Alert",
        condition={
            "evaluator": {
                "params": [40],
                "type": "gt"
            },
            "operator": {
                "type": "and"
            },
            "query": {
                "params": ["A", "5m", "now"]
            },
            "reducer": {
                "params": [],
                "type": "avg"
            },
            "type": "query"
        },
        message="This is a test alert from the Investment Analysis Platform"
    )

    return success_response(data=AlertTestResponse(
        alert_created=alert_created,
        grafana_connected=grafana_client.test_connection()
    ))


@router.get("/metrics/api-usage")
async def get_api_usage_metrics() -> ApiResponse[ApiUsageMetrics]:
    """Get API usage metrics for all providers"""
    av_used = await cost_monitor.get_provider_usage("alpha_vantage")

    return success_response(data=ApiUsageMetrics(
        alpha_vantage=ProviderUsage(
            daily_limit=25,
            used_today=av_used,
            remaining=25 - av_used
        ),
        finnhub={
            "minute_limit": 60,
            "daily_limit": 86400,
            "used_today": await cost_monitor.get_provider_usage("finnhub")
        },
        polygon={
            "minute_limit": 5,
            "daily_limit": 7200,
            "used_today": await cost_monitor.get_provider_usage("polygon")
        }
    ))