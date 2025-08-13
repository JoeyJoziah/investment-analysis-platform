"""
Optimized FastAPI Application with Graceful Shutdown
Budget-conscious configuration for $50/month infrastructure
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import logging
import os
from datetime import datetime
from typing import Optional

from backend.api.routers import (
    stocks, analysis, recommendations, portfolio,
    auth, health, websocket, admin
)
from backend.utils.database_optimized import (
    init_database, close_database, check_database_health
)
from backend.utils.cache import init_cache
from backend.utils.graceful_shutdown import shutdown_handler, init_graceful_shutdown
from backend.utils.cache_warming import CacheWarmingStrategy
from backend.config.settings import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager with graceful shutdown support.
    """
    # Startup
    logger.info("Starting Investment Analysis Platform (Optimized)...")
    
    try:
        # Initialize database with optimized pooling
        db = init_database()
        app.state.db = db
        logger.info("Database initialized with optimized connection pooling")
        
        # Check database health
        db_health = check_database_health()
        logger.info(f"Database health: {db_health}")
        
        # Initialize cache
        await init_cache()
        logger.info("Cache initialized")
        
        # Initialize cache warming strategy
        cache_warmer = CacheWarmingStrategy()
        app.state.cache_warmer = cache_warmer
        
        # Register components with shutdown handler
        shutdown_handler.register_component('database', db)
        shutdown_handler.register_component('cache_warming', cache_warmer)
        
        # Initialize graceful shutdown
        init_graceful_shutdown(app)
        
        # Start background scheduler with reduced resources
        from backend.tasks.scheduler import start_scheduler
        scheduler = await start_scheduler()
        app.state.scheduler = scheduler
        shutdown_handler.register_component('scheduler', scheduler)
        logger.info("Background scheduler started")
        
        # Load ML models (only if sufficient memory)
        try:
            import psutil
            available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # GB
            
            if available_memory > 1.5:  # Only load if >1.5GB available
                from backend.ml.model_manager import get_model_manager
                model_manager = get_model_manager()
                app.state.model_manager = model_manager
                logger.info("ML models loaded")
            else:
                logger.warning(f"Insufficient memory for ML models ({available_memory:.1f}GB available)")
                app.state.model_manager = None
        except ImportError:
            logger.warning("ML model manager not available, using fallback")
            app.state.model_manager = None
        
        # Warm critical caches if not recovering from shutdown
        if not shutdown_handler.is_shutting_down:
            logger.info("Starting initial cache warming...")
            try:
                import asyncio
                warm_task = asyncio.create_task(cache_warmer.warm_critical_caches())
                # Don't wait, let it run in background
            except Exception as e:
                logger.error(f"Cache warming failed: {e}")
        
        logger.info("Application startup complete")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Investment Analysis Platform...")
    
    try:
        # Graceful shutdown is handled by shutdown_handler
        if not shutdown_handler.is_shutting_down:
            await shutdown_handler.shutdown()
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
    
    logger.info("Shutdown complete")


# Create FastAPI app with optimized settings
app = FastAPI(
    title="Investment Analysis Platform (Optimized)",
    description="Budget-Optimized AI-Powered Stock Analysis & Recommendations",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/api/docs" if settings.DEBUG else None,
    redoc_url="/api/redoc" if settings.DEBUG else None,
    # Reduce default limits
    max_shutdown_wait=30,  # Max 30 seconds for graceful shutdown
)

# Add CORS middleware with secure configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS.split(',') if hasattr(settings, 'ALLOWED_ORIGINS') else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600  # Cache preflight requests
)


# Custom middleware for connection limit enforcement
@app.middleware("http")
async def limit_connections(request: Request, call_next):
    """Enforce connection limits to prevent database exhaustion."""
    try:
        # Check database pool status
        if hasattr(app.state, 'db'):
            pool_status = app.state.db.get_pool_status()
            utilization = pool_status['checked_out'] / (pool_status['size'] + pool_status['max_overflow'])
            
            # Reject requests if pool is exhausted
            if utilization > 0.95:
                logger.warning(f"Connection pool near exhaustion: {utilization:.1%}")
                return JSONResponse(
                    status_code=503,
                    content={"detail": "Service temporarily unavailable due to high load"}
                )
    except Exception as e:
        logger.error(f"Error checking connection pool: {e}")
    
    response = await call_next(request)
    return response


# Health check with shutdown awareness
@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint that reflects shutdown state."""
    if shutdown_handler.is_shutting_down:
        return JSONResponse(
            status_code=503,
            content={
                "status": "shutting_down",
                "message": "Service is shutting down gracefully"
            }
        )
    
    # Check component health
    db_health = check_database_health() if hasattr(app.state, 'db') else {"status": "unknown"}
    
    return {
        "status": "healthy" if db_health.get("status") == "healthy" else "degraded",
        "timestamp": datetime.now().isoformat(),
        "database": db_health,
        "shutdown_handler": {
            "is_shutting_down": shutdown_handler.is_shutting_down,
            "metrics": shutdown_handler.metrics
        }
    }


# Database pool status endpoint (admin only)
@app.get("/api/admin/pool-status", tags=["admin"])
async def get_pool_status():
    """Get current database connection pool status."""
    if not hasattr(app.state, 'db'):
        raise HTTPException(status_code=503, detail="Database not initialized")
    
    pool_status = app.state.db.get_pool_status()
    recommendations = app.state.db.calculate_optimal_pool_size()
    
    return {
        "current_status": pool_status,
        "recommendations": recommendations,
        "health": check_database_health()
    }


# Include routers
app.include_router(health.router, prefix="/api/health", tags=["health"])
app.include_router(auth.router, prefix="/api/auth", tags=["authentication"])
app.include_router(stocks.router, prefix="/api/stocks", tags=["stocks"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(recommendations.router, prefix="/api/recommendations", tags=["recommendations"])
app.include_router(portfolio.router, prefix="/api/portfolio", tags=["portfolio"])
app.include_router(websocket.router, prefix="/api/ws", tags=["websocket"])
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])


# Metrics endpoint
@app.get("/metrics", tags=["monitoring"])
async def get_metrics():
    """Prometheus-compatible metrics endpoint."""
    metrics = []
    
    # Database metrics
    if hasattr(app.state, 'db'):
        pool_status = app.state.db.get_pool_status()
        metrics.append(f'db_pool_size {pool_status["size"]}')
        metrics.append(f'db_pool_checked_out {pool_status["checked_out"]}')
        metrics.append(f'db_pool_overflow {pool_status["overflow"]}')
    
    # Cache warming metrics
    if hasattr(app.state, 'cache_warmer'):
        warming_metrics = app.state.cache_warmer._metrics
        metrics.append(f'cache_warmed_total {warming_metrics["caches_warmed"]}')
        metrics.append(f'cache_warming_errors_total {warming_metrics["warming_errors"]}')
    
    # Shutdown metrics
    metrics.append(f'shutdown_requested {1 if shutdown_handler.is_shutting_down else 0}')
    metrics.append(f'shutdown_tasks_cancelled {shutdown_handler.metrics["tasks_cancelled"]}')
    metrics.append(f'shutdown_tasks_completed {shutdown_handler.metrics["tasks_completed"]}')
    
    return "\n".join(metrics)


if __name__ == "__main__":
    # Run with optimized settings
    uvicorn.run(
        "backend.api.main_optimized:app",
        host="0.0.0.0",
        port=8000,
        workers=2,  # Reduced from default
        loop="uvloop",  # More efficient event loop
        limit_concurrency=100,  # Limit concurrent connections
        limit_max_requests=10000,  # Restart workers periodically
        timeout_keep_alive=5,  # Reduce keepalive timeout
        access_log=False,  # Disable access logs in production for performance
        log_level="info"
    )