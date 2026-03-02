"""
Main FastAPI Application - World-Leading Investment Analysis Platform
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import logging
from datetime import datetime, timezone
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from backend.api.routers import (
    stocks, analysis, recommendations, portfolio,
    auth, health, admin, cache_management,
    websocket, agents, gdpr, watchlist, thesis,
    news, ml,
)
from backend.api.routers import settings as settings_router
from backend.api.versioning import (
    V1DeprecationMiddleware,
    v1_migration_router,
    v1_migration_metrics
)
from backend.utils.database import init_db, close_db
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
    
    # Initialize database (async engine + table creation + data seeding)
    # Note: init_db() calls initialize_database() internally, so a separate
    # initialize_database() call is not needed. The legacy sync engine
    # (backend.utils.database.engine) is created at import time and requires
    # no explicit init; it is still used by the health router and Celery tasks.
    await init_db()
    logger.info("Database initialized")
    
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
    
    # Clean up database connections (async engine + legacy sync engine)
    # Note: close_db() calls cleanup_database() internally, so a separate
    # cleanup_database() call is not needed.
    await close_db()
    logger.info("Database connections closed")
    
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

# Initialize priority-based middleware stack
from backend.middleware.stack import MiddlewareStack, MiddlewarePriority
from backend.api.security_integration import register_security_middleware
import os

is_testing = os.getenv("TESTING", "False").lower() == "true"

try:
    # Create middleware stack
    stack = MiddlewareStack(app)

    # Import middleware classes
    from fastapi.middleware.cors import CORSMiddleware
    from backend.middleware.security_headers import SecurityHeadersMiddleware, SecurityHeadersConfig
    from backend.security.csrf_protection import CSRFMiddleware, CSRFConfig
    from backend.security.advanced_rate_limiter import RateLimitingMiddleware, get_default_rate_limiting_rules
    from backend.middleware.request_size_limiter import RequestSizeLimiterMiddleware, RequestSizeLimits
    from backend.security.audit_logging import AuditMiddleware
    from fastapi.middleware.gzip import GZipMiddleware
    from backend.middleware.response_optimizer import ResponseTimingMiddleware, ETagMiddleware

    # Configure CORS
    cors_origins = ["http://localhost:3000", "http://localhost:8000"]
    if settings.ENVIRONMENT == "development":
        cors_origins.extend(["http://127.0.0.1:3000", "http://127.0.0.1:8000"])

    # Register middleware in priority order (highest priority = outermost)

    # 1. Response Timing - Outermost for accurate timing
    stack.register(
        "response_timing",
        ResponseTimingMiddleware,
        MiddlewarePriority.HIGHEST,
        {}
    )

    # 2. CORS - Must be early to set headers before other processing
    stack.register(
        "cors",
        CORSMiddleware,
        MiddlewarePriority.CORS,
        {
            "allow_origins": cors_origins,
            "allow_credentials": True,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
            "expose_headers": ["X-RateLimit-Remaining", "X-RateLimit-Reset", "X-Request-ID", "X-CSRF-Token", "X-Response-Time", "ETag"]
        }
    )

    # 3. Security Headers - After CORS
    stack.register(
        "security_headers",
        SecurityHeadersMiddleware,
        MiddlewarePriority.SECURITY_HEADERS,
        {"config": SecurityHeadersConfig()},
        skip_in_testing=False  # Keep security headers in testing
    )

    # 4. CSRF Protection - After security headers
    csrf_secret = os.getenv("CSRF_SECRET_KEY")
    if not csrf_secret and not is_testing:
        logger.warning("CSRF_SECRET_KEY not set, using auto-generated key for development")

    stack.register(
        "csrf",
        CSRFMiddleware,
        MiddlewarePriority.CSRF,
        {"config": CSRFConfig(secret_key=csrf_secret)},
        skip_in_testing=True  # CSRF interferes with test client
    )

    # 5. Rate Limiting - After CSRF
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    rate_limit_rules = get_default_rate_limiting_rules()

    stack.register(
        "rate_limiting",
        RateLimitingMiddleware,
        MiddlewarePriority.RATE_LIMITING,
        {"rules": rate_limit_rules, "redis_url": redis_url},
        skip_in_testing=True  # Rate limiting can interfere with tests
    )

    # 6. Request Size Limits - After rate limiting
    stack.register(
        "request_size",
        RequestSizeLimiterMiddleware,
        MiddlewarePriority.REQUEST_SIZE,
        {"config": RequestSizeLimits()},
        skip_in_testing=False  # Keep size limits in testing
    )

    # 7. Audit Logging - Monitor requests
    stack.register(
        "audit",
        AuditMiddleware,
        MiddlewarePriority.AUDIT,
        {},
        skip_in_testing=True  # Audit can interfere with test client
    )

    # 8. Prometheus Monitoring
    stack.register(
        "prometheus",
        PrometheusMiddleware,
        MiddlewarePriority.MONITORING,
        {}
    )

    # 9. Cache Control
    stack.register(
        "cache_control",
        CacheControlMiddleware,
        MiddlewarePriority.CACHING,
        {
            "default_cache_control": "public, max-age=300",
            "cache_excluded_paths": ["/api/v1/auth/", "/api/v1/admin/", "/api/v1/ws/", "/api/v1/metrics"]
        }
    )

    # 10. ETag Generation - After caching, before compression
    stack.register(
        "etag",
        ETagMiddleware,
        MiddlewarePriority.CACHING - 100,  # Just after cache control
        {
            "excluded_paths": ["/api/v1/auth/", "/api/v1/admin/", "/api/v1/ws/", "/api/health"],
            "weak_etag": False
        }
    )

    # 11. GZip Compression - Innermost (compresses final response)
    stack.register(
        "gzip",
        GZipMiddleware,
        MiddlewarePriority.COMPRESSION,
        {"minimum_size": 1000},
        skip_in_testing=True  # GZip can interfere with test client
    )

    # 12. V1 API Deprecation - After compression
    if not is_testing:
        stack.register(
            "v1_deprecation",
            V1DeprecationMiddleware,
            MiddlewarePriority.LOW,
            {
                "enable_redirects": False,
                "grace_period_days": 30,
                "strict_mode": False
            },
            skip_in_testing=True
        )

    # Apply all middleware in priority order
    stack.apply(is_testing=is_testing)

    # Log middleware stack summary
    logger.info("\n" + stack.get_stack_summary())
    logger.info("Priority-based middleware stack initialized successfully")

except Exception as e:
    logger.error(f"Failed to initialize middleware stack: {e}", exc_info=True)
    logger.warning("Falling back to basic CORS middleware only")

    # Emergency fallback - just CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:8000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Include routers - all API routers use /api/v1/ prefix for consistency
# Health/status endpoints stay outside versioned prefix
app.include_router(health.router, prefix="/api/health", tags=["health"])

# Versioned API routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["authentication"])
app.include_router(stocks.router, prefix="/api/v1/stocks", tags=["stocks"])
app.include_router(analysis.router, prefix="/api/v1/analysis", tags=["analysis"])
app.include_router(recommendations.router, prefix="/api/v1/recommendations", tags=["recommendations"])
app.include_router(portfolio.router, prefix="/api/v1/portfolio", tags=["portfolio"])
app.include_router(websocket.router, prefix="/api/v1/ws", tags=["websocket"])
app.include_router(admin.router, prefix="/api/v1/admin", tags=["admin"])
app.include_router(agents.router, prefix="/api/v1/agents", tags=["agents"])
app.include_router(cache_management.router, prefix="/api/v1/cache", tags=["cache"])
app.include_router(gdpr.router, prefix="/api/v1", tags=["gdpr"])
app.include_router(watchlist.router, prefix="/api/v1/watchlists", tags=["watchlists"])
app.include_router(thesis.router, prefix="/api/v1/thesis", tags=["investment-thesis"])
app.include_router(news.router, prefix="/api/v1/news", tags=["news"])
app.include_router(ml.router, prefix="/api/v1/ml", tags=["ml"])
app.include_router(settings_router.router, prefix="/api/v1/settings", tags=["settings"])
app.include_router(v1_migration_router)  # V1 migration monitoring endpoints (self-prefixed)


@app.get("/")
async def root():
    """
    Root endpoint
    """
    return {
        "message": "Investment Analysis Platform API",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/api/v1/metrics")
async def get_metrics():
    """
    Prometheus metrics endpoint
    """
    return export_metrics()


# ---------------------------------------------------------------------------
# Socket.IO – mount alongside FastAPI so both share the same process.
# The ASGIApp routes /socket.io/ requests to the Socket.IO server and
# forwards all other requests to the FastAPI app unchanged.
# ---------------------------------------------------------------------------
from backend.services.socketio_service import create_socketio_asgi_app

# ``socket_app`` is the top-level ASGI callable that should be passed to the
# ASGI server (e.g. uvicorn).  The FastAPI ``app`` continues to work normally
# for all non-Socket.IO paths.
socket_app = create_socketio_asgi_app(app)


if __name__ == "__main__":
    uvicorn.run(
        "backend.api.main:socket_app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )