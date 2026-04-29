from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session
from typing import Dict, Optional, Any
import psutil
import logging
from datetime import datetime, timezone
from backend.utils.database import get_db_sync, engine
from backend.utils.cache import get_redis_client
from backend.models.api_response import ApiResponse, success_response
from backend.security.advanced_rate_limiter import get_default_rate_limiting_rules

router = APIRouter(tags=["health"])

def _get_fallback_models_state() -> Dict[str, Any]:
    """Return ML model fallback state for the /health response.

    Per PRD audit 2026-04 F-03-003: a deployment running with
    DummyLSTM/DummyXGBoost/DummyProphet substitutes is producing fabricated
    investment outputs. The health endpoint surfaces ``fallback_models`` and
    ``fallback_models_count`` so readiness probes / SRE alerts can fail fast.
    """
    try:
        from backend.ml.model_manager import get_model_manager
        mgr = get_model_manager()
        fallback = mgr.get_fallback_models()
        return {
            "fallback_models": fallback,
            "fallback_models_count": len(fallback),
        }
    except Exception:  # pragma: no cover - never let health crash on ml import
        return {"fallback_models": [], "fallback_models_count": 0}


@router.get("")
async def health_check() -> ApiResponse[Dict[str, Any]]:
    """Basic health check endpoint.

    Includes ML ``fallback_models`` state per PRD audit 2026-04 §3 D /
    F-03-003. Empty ``fallback_models`` array == healthy production.
    """
    fb = _get_fallback_models_state()
    overall = "healthy" if fb["fallback_models_count"] == 0 else "degraded"
    return success_response(data={
        "status": overall,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
        "service": "investment-analysis-api",
        **fb,
    })

@router.get("/readiness")
async def readiness_check() -> ApiResponse[Dict[str, Any]]:
    """Check if all services are ready"""
    logger = logging.getLogger(__name__)
    checks = {
        "database": False,
        "cache": False,
        "api": True
    }
    errors = {}

    # Check Redis
    try:
        redis_client = get_redis_client()
        redis_client.ping()
        checks["cache"] = True
    except Exception as e:
        errors["cache"] = str(e)
        logger.error(f"Redis health check failed: {e}")

    # Check Database
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()

            # Check if tables exist
            table_check = conn.execute(text("""
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_schema = 'public'
            """))
            table_count = table_check.scalar()

            if table_count > 0:
                checks["database"] = True
            else:
                errors["database"] = "No tables found in database"
    except Exception as e:
        errors["database"] = str(e)
        logger.error(f"Database health check failed: {e}")

    # F-03-003: production-critical ML models in Dummy* fallback fail readiness.
    fb = _get_fallback_models_state()
    checks["ml_models"] = fb["fallback_models_count"] == 0
    if not checks["ml_models"]:
        errors["ml_models"] = (
            f"{fb['fallback_models_count']} model(s) in fallback: "
            f"{fb['fallback_models']}"
        )

    all_ready = all(checks.values())

    data = {
        "status": "ready" if all_ready else "not ready",
        "checks": checks,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **fb,
    }

    if errors:
        data["errors"] = errors

    return success_response(data=data)

@router.get("/metrics")
async def get_metrics() -> ApiResponse[Dict[str, Any]]:
    """Get system metrics"""
    try:
        # Get database connection pool stats
        pool_stats = {
            "size": engine.pool.size(),
            "checked_in": engine.pool.checkedin(),
            "overflow": engine.pool.overflow(),
            "total": engine.pool.total()
        }
    except:
        pool_stats = None

    # Get Redis info
    redis_info = None
    try:
        redis_client = get_redis_client()
        info = redis_client.info()
        redis_info = {
            "used_memory": info.get('used_memory_human'),
            "connected_clients": info.get('connected_clients'),
            "total_commands_processed": info.get('total_commands_processed'),
            "keyspace_hits": info.get('keyspace_hits'),
            "keyspace_misses": info.get('keyspace_misses')
        }
    except:
        pass

    metrics = {
        "system": {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory": {
                "total": psutil.virtual_memory().total,
                "used": psutil.virtual_memory().used,
                "percent": psutil.virtual_memory().percent,
                "available": psutil.virtual_memory().available
            },
            "disk": {
                "total": psutil.disk_usage('/').total,
                "used": psutil.disk_usage('/').used,
                "free": psutil.disk_usage('/').free,
                "percent": psutil.disk_usage('/').percent
            },
            "network": {
                "connections": len(psutil.net_connections())
            }
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    if pool_stats:
        metrics["database_pool"] = pool_stats

    if redis_info:
        metrics["redis"] = redis_info

    return success_response(data=metrics)

@router.get("/liveness")
async def liveness_check() -> ApiResponse[Dict[str, Any]]:
    """Kubernetes liveness probe endpoint"""
    return success_response(data={
        "status": "alive",
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

@router.get("/startup")
async def startup_check() -> ApiResponse[Dict[str, Any]]:
    """Kubernetes startup probe endpoint"""
    # Check if critical services are initialized
    try:
        # Quick database check
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        # Quick Redis check
        redis_client = get_redis_client()
        redis_client.ping()

        return success_response(data={
            "status": "started",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service not ready: {str(e)}")

@router.get("/ping")
async def ping() -> ApiResponse[Dict[str, Any]]:
    """Simple ping endpoint for health checks"""
    return success_response(data={"status": "pong", "timestamp": datetime.now(timezone.utc).isoformat()})

@router.get("/rate-limiter")
async def rate_limiter_health() -> ApiResponse[Dict[str, Any]]:
    """Check rate limiter and Redis health"""
    logger = logging.getLogger(__name__)

    redis_status = {
        "available": False,
        "error": None
    }

    # Check Redis health for rate limiting
    try:
        redis_client = get_redis_client()
        redis_client.ping()
        redis_status["available"] = True
        logger.info("Rate limiter Redis health check: PASS")
    except Exception as e:
        redis_status["error"] = str(e)
        logger.warning(f"Rate limiter Redis health check FAILED: {e}")

    # Check rate limiting rules are configured
    try:
        rules = get_default_rate_limiting_rules()
        rules_configured = len(rules) > 0
    except Exception as e:
        logger.error(f"Could not load rate limiting rules: {e}")
        rules_configured = False

    status = {
        "status": "healthy" if redis_status["available"] and rules_configured else "degraded",
        "redis": redis_status,
        "rules_configured": rules_configured,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "note": "Rate limiter will fall back to in-memory mode if Redis is unavailable"
    }

    return success_response(data=status)