from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session
from typing import Dict, Optional
import psutil
import logging
from datetime import datetime
from backend.utils.database import get_db_sync, engine
from backend.utils.cache import get_redis_client

router = APIRouter(tags=["health"])

@router.get("")
async def health_check() -> Dict:
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "service": "investment-analysis-api"
    }

@router.get("/readiness")
async def readiness_check() -> Dict:
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
    
    all_ready = all(checks.values())
    
    response = {
        "status": "ready" if all_ready else "not ready",
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if errors:
        response["errors"] = errors
    
    return response

@router.get("/metrics")
async def get_metrics() -> Dict:
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
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if pool_stats:
        metrics["database_pool"] = pool_stats
    
    if redis_info:
        metrics["redis"] = redis_info
    
    return metrics

@router.get("/liveness")
async def liveness_check() -> Dict:
    """Kubernetes liveness probe endpoint"""
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/startup")
async def startup_check() -> Dict:
    """Kubernetes startup probe endpoint"""
    # Check if critical services are initialized
    try:
        # Quick database check
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        # Quick Redis check
        redis_client = get_redis_client()
        redis_client.ping()
        
        return {
            "status": "started",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service not ready: {str(e)}")