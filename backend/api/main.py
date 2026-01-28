"""
Main FastAPI Application - World-Leading Investment Analysis Platform
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import logging
from datetime import datetime
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from backend.api.routers import (
    stocks, analysis, recommendations, portfolio,
    auth, health, admin, cache_management,
    websocket, agents, gdpr, watchlist, thesis
)
from backend.api.versioning import (
    V1DeprecationMiddleware,
    v1_migration_router,
    v1_migration_metrics
)
from backend.utils.database import init_db, close_db
from backend.config.database import initialize_database, cleanup_database
from backend.utils.cache import init_cache
from backend.utils.comprehensive_cache import get_cache_manager
from backend.utils.intelligent_cache_policies import start_intelligent_caching
from backend.utils.cache_monitoring import initialize_cache_monitoring
from backend.utils.database_query_cache import setup_cache_invalidation_triggers
from backend.utils.api_cache_decorators import CacheControlMiddleware
from backend.utils.monitoring import PrometheusMiddleware, export_metrics
from backend.config.settings import settings
from backend.middleware.error_handler import register_exception_handlers

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    """
    # Startup
    logger.info("Starting Investment Analysis Platform...")
    
    # Initialize async database
    await initialize_database()
    logger.info("Async database initialized")
    
    # Initialize legacy database for backwards compatibility
    await init_db()
    logger.info("Legacy database initialized")
    
    # Initialize cache
    await init_cache()
    logger.info("Basic cache initialized")
    
    # Initialize comprehensive caching system
    try:
        cache_manager = await get_cache_manager()
        logger.info("Comprehensive cache manager initialized")
        
        # Start intelligent caching services
        await start_intelligent_caching()
        logger.info("Intelligent caching services started")
        
        # Initialize cache monitoring
        await initialize_cache_monitoring()
        logger.info("Cache monitoring initialized")
        
        # Setup database cache invalidation
        await setup_cache_invalidation_triggers()
        logger.info("Cache invalidation triggers setup")
        
    except Exception as e:
        logger.warning(f"Failed to initialize advanced caching features: {e}")
        logger.info("Continuing with basic caching only")
    
    # Start background tasks
    from backend.tasks.scheduler import start_scheduler
    scheduler = await start_scheduler()
    logger.info("Background scheduler started")

    # Start WebSocket cleanup task
    from backend.api.routers.websocket import start_cleanup_task
    start_cleanup_task()
    logger.info("WebSocket cleanup task started")

    # Load ML models
    try:
        from backend.ml.model_manager import get_model_manager
        model_manager = get_model_manager()
        logger.info("ML model manager initialized")
    except ImportError:
        logger.warning("ML model manager not available, using fallback")
        model_manager = None
    app.state.model_manager = model_manager
    logger.info("ML models loaded")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Investment Analysis Platform...")
    
    # Clean up async database
    await cleanup_database()
    logger.info("Async database cleaned up")
    
    # Clean up legacy database
    await close_db()
    logger.info("Legacy database closed")
    
    if 'scheduler' in locals():
        await scheduler.shutdown()
    logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Investment Analysis Platform",
    description="World-Leading AI-Powered Stock Analysis & Recommendations",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs" if settings.DEBUG else None,
    redoc_url="/api/redoc" if settings.DEBUG else None
)

# Register standardized error handlers
register_exception_handlers(app)

# Add comprehensive security middleware stack
# This provides CORS, security headers, rate limiting, input validation, and injection prevention
try:
    from backend.security.security_config import add_comprehensive_security_middleware
    add_comprehensive_security_middleware(app)
    logger.info("Comprehensive security middleware stack enabled")
except Exception as e:
    logger.warning(f"Failed to initialize comprehensive security middleware: {e}")
    logger.info("Falling back to basic CORS middleware")
    # Fallback to basic CORS if comprehensive security fails
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:8000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Add Prometheus monitoring
app.add_middleware(PrometheusMiddleware)

# Add comprehensive cache control middleware
app.add_middleware(
    CacheControlMiddleware,
    default_cache_control="public, max-age=300",
    cache_excluded_paths=["/api/auth/", "/api/admin/", "/api/ws/", "/api/metrics"]
)

# Add V1 API deprecation middleware
# This handles V1 requests with deprecation warnings, usage tracking, and optional redirects
# Set enable_redirects=True to automatically redirect V1 requests to V2
# Set strict_mode=True to immediately return 410 for V1 requests (post-sunset)
# IMPORTANT: Disabled during testing to prevent 410 errors in test suite
import os
if os.getenv("TESTING", "False").lower() != "true":
    app.add_middleware(
        V1DeprecationMiddleware,
        enable_redirects=False,  # Set to True for automatic redirects
        grace_period_days=30,    # Days after sunset to still allow V1 (with warnings)
        strict_mode=False        # Set to True to immediately reject V1 requests
    )

# Include routers
app.include_router(health.router, prefix="/api/health", tags=["health"])
app.include_router(auth.router, prefix="/api/auth", tags=["authentication"])
app.include_router(stocks.router, prefix="/api/stocks", tags=["stocks"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(recommendations.router, prefix="/api/recommendations", tags=["recommendations"])
app.include_router(portfolio.router, prefix="/api/portfolio", tags=["portfolio"])
app.include_router(websocket.router, prefix="/api/ws", tags=["websocket"])
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])
app.include_router(agents.router, prefix="/api/agents", tags=["agents"])
app.include_router(cache_management.router, prefix="/api/cache", tags=["cache"])
app.include_router(gdpr.router, prefix="/api/v1", tags=["gdpr"])
app.include_router(watchlist.router, prefix="/api", tags=["watchlists"])
app.include_router(thesis.router, prefix="/api/v1", tags=["investment-thesis"])
app.include_router(v1_migration_router)  # V1 migration monitoring endpoints


@app.get("/")
async def root():
    """
    Root endpoint
    """
    return {
        "message": "Investment Analysis Platform API",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/api/metrics")
async def get_metrics():
    """
    Prometheus metrics endpoint
    """
    return export_metrics()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )