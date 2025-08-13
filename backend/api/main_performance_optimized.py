"""
Performance-Optimized FastAPI Application
Enhanced with comprehensive optimizations for high-throughput production use
"""

import asyncio
import time
import logging
import gc
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any

import uvloop
from fastapi import FastAPI, Request, Response, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import ORJSONResponse
import uvicorn
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from prometheus_fastapi_instrumentator import Instrumentator

from backend.config.settings import settings
from backend.utils.memory_manager import get_memory_manager
from backend.utils.dynamic_resource_manager import get_resource_manager
from backend.utils.enhanced_parallel_processor import enhanced_parallel_processor
from backend.analytics.recommendation_engine_optimized import get_optimized_recommendation_engine

# Import routers
from backend.api.routers import (
    health, stocks, recommendations, analysis, 
    portfolio, auth, admin, websocket, agents
)

# Configure uvloop for better async performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

logger = logging.getLogger(__name__)

# Prometheus metrics
request_count = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
request_duration = Histogram('http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])
active_connections = Gauge('http_active_connections', 'Active HTTP connections')
memory_usage = Gauge('app_memory_usage_bytes', 'Application memory usage')
gc_collections = Counter('python_gc_collections_total', 'Total garbage collections', ['generation'])


class PerformanceMiddleware:
    """High-performance middleware with optimizations"""
    
    def __init__(self, app):
        self.app = app
        self.active_requests = 0
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        start_time = time.perf_counter()
        self.active_requests += 1
        active_connections.set(self.active_requests)
        
        method = scope["method"]
        path = scope["path"]
        
        # Optimize path matching for common endpoints
        if path.startswith("/api/health"):
            endpoint = "health"
        elif path.startswith("/api/stocks"):
            endpoint = "stocks"
        elif path.startswith("/api/recommendations"):
            endpoint = "recommendations"
        elif path.startswith("/api/analysis"):
            endpoint = "analysis"
        else:
            endpoint = "other"
        
        try:
            await self.app(scope, receive, send)
            status = "success"
        except Exception as e:
            status = "error"
            logger.error(f"Request error: {e}")
            raise
        finally:
            # Update metrics
            duration = time.perf_counter() - start_time
            request_count.labels(method=method, endpoint=endpoint, status=status).inc()
            request_duration.labels(method=method, endpoint=endpoint).observe(duration)
            
            self.active_requests -= 1
            active_connections.set(self.active_requests)


class ConnectionPoolMiddleware:
    """Middleware for connection pool optimization"""
    
    def __init__(self, app):
        self.app = app
        self.connection_pools = {}
    
    async def __call__(self, scope, receive, send):
        # Add connection pool info to request state
        if scope["type"] == "http":
            scope["state"] = scope.get("state", {})
            scope["state"]["connection_pools"] = self.connection_pools
        
        await self.app(scope, receive, send)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced application lifespan management"""
    logger.info("Starting performance-optimized FastAPI application...")
    
    # Initialize memory manager
    memory_manager = await get_memory_manager()
    
    # Initialize resource manager with aggressive optimization
    resource_manager = await get_resource_manager(
        monitoring_interval_s=15,  # More frequent monitoring
        enable_auto_scaling=True,
        enable_predictive_scaling=True
    )
    
    # Initialize enhanced parallel processor
    await enhanced_parallel_processor.initialize()
    
    # Initialize optimized recommendation engine
    recommendation_engine = await get_optimized_recommendation_engine()
    
    # Pre-warm critical components
    await _prewarm_application()
    
    # Configure garbage collection for performance
    gc.set_threshold(1000, 15, 15)  # More aggressive for high-throughput
    
    # Start background tasks
    background_task = asyncio.create_task(_background_optimization_loop())
    
    logger.info("Performance-optimized application startup complete")
    
    yield
    
    # Shutdown sequence
    logger.info("Shutting down performance-optimized application...")
    
    # Cancel background tasks
    background_task.cancel()
    try:
        await background_task
    except asyncio.CancelledError:
        pass
    
    # Shutdown components in reverse order
    if recommendation_engine:
        await recommendation_engine.shutdown()
    
    await enhanced_parallel_processor.shutdown()
    
    if resource_manager:
        await resource_manager.shutdown()
    
    if memory_manager:
        await memory_manager.shutdown()
    
    # Final cleanup
    gc.collect()
    
    logger.info("Performance-optimized application shutdown complete")


async def _prewarm_application():
    """Pre-warm application components for better performance"""
    logger.info("Pre-warming application components...")
    
    # Pre-warm recommendation engine
    try:
        engine = await get_optimized_recommendation_engine()
        # Trigger initialization of analysis engines
        _ = engine.technical_engine
        _ = engine.fundamental_engine
        _ = engine.sentiment_engine
        logger.info("Pre-warmed recommendation engines")
    except Exception as e:
        logger.warning(f"Failed to pre-warm recommendation engine: {e}")
    
    # Pre-warm parallel processor connection pools
    try:
        await enhanced_parallel_processor.initialize()
        logger.info("Pre-warmed parallel processor")
    except Exception as e:
        logger.warning(f"Failed to pre-warm parallel processor: {e}")
    
    # Pre-compile frequently used regex patterns
    import re
    patterns = [
        r'^[A-Z]{1,5}$',  # Stock ticker validation
        r'^\d{4}-\d{2}-\d{2}$',  # Date validation
        r'^[a-zA-Z0-9_-]+$'  # General identifier validation
    ]
    
    for pattern in patterns:
        re.compile(pattern)
    
    logger.info("Pre-warming complete")


async def _background_optimization_loop():
    """Background optimization loop"""
    while True:
        try:
            # Update memory metrics
            import psutil
            process = psutil.Process()
            memory_usage.set(process.memory_info().rss)
            
            # Update GC metrics
            gc_stats = gc.get_stats()
            for i, stats in enumerate(gc_stats):
                gc_collections.labels(generation=str(i))._value._value = stats['collections']
            
            # Force GC occasionally for better memory management
            collected = gc.collect()
            if collected > 0:
                logger.debug(f"Background GC collected {collected} objects")
            
            await asyncio.sleep(30)  # Run every 30 seconds
            
        except Exception as e:
            logger.error(f"Background optimization error: {e}")
            await asyncio.sleep(60)


# Create optimized FastAPI application
app = FastAPI(
    title="Investment Analysis API - Performance Optimized",
    description="High-performance investment analysis and recommendation API",
    version="2.0.0",
    docs_url="/docs" if settings.ENVIRONMENT == "development" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT == "development" else None,
    default_response_class=ORJSONResponse,  # Faster JSON serialization
    lifespan=lifespan
)

# Add performance middleware
app.add_middleware(PerformanceMiddleware)
app.add_middleware(ConnectionPoolMiddleware)

# Add security middleware
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["*"] if settings.ENVIRONMENT == "development" else settings.ALLOWED_HOSTS
)

# Add compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add CORS middleware with optimization
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    max_age=3600  # Cache preflight requests
)

# Initialize Prometheus instrumentator
instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics"],
    env_var_name="ENABLE_METRICS",
    inprogress_name="http_requests_inprogress",
    inprogress_labels=True
)

instrumentator.instrument(app)


@app.middleware("http")
async def add_performance_headers(request: Request, call_next):
    """Add performance-related headers"""
    start_time = time.perf_counter()
    
    response = await call_next(request)
    
    # Add performance headers
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-API-Version"] = "2.0.0"
    
    # Add cache headers for static content
    if request.url.path.startswith("/static"):
        response.headers["Cache-Control"] = "public, max-age=3600"
    elif request.url.path.startswith("/api/health"):
        response.headers["Cache-Control"] = "no-cache"
    else:
        response.headers["Cache-Control"] = "private, max-age=300"
    
    return response


@app.middleware("http")
async def optimize_response_compression(request: Request, call_next):
    """Optimize response compression"""
    response = await call_next(request)
    
    # Add compression hints
    accept_encoding = request.headers.get("accept-encoding", "")
    if "br" in accept_encoding:
        response.headers["Vary"] = "Accept-Encoding"
    
    return response


# Health check endpoint (optimized)
@app.get("/health", tags=["health"], response_class=ORJSONResponse)
async def health_check():
    """Ultra-fast health check"""
    return {"status": "healthy", "timestamp": time.time()}


# Metrics endpoint
@app.get("/metrics", include_in_schema=False)
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Performance stats endpoint
@app.get("/api/performance", tags=["monitoring"], response_class=ORJSONResponse)
async def get_performance_stats():
    """Get comprehensive performance statistics"""
    try:
        stats = {}
        
        # Memory manager stats
        try:
            memory_manager = await get_memory_manager()
            stats["memory"] = memory_manager.get_memory_stats()
        except Exception as e:
            stats["memory"] = {"error": str(e)}
        
        # Resource manager stats
        try:
            resource_manager = await get_resource_manager()
            stats["resources"] = resource_manager.get_performance_stats()
        except Exception as e:
            stats["resources"] = {"error": str(e)}
        
        # Parallel processor stats
        try:
            stats["parallel_processor"] = enhanced_parallel_processor.get_enhanced_performance_stats()
        except Exception as e:
            stats["parallel_processor"] = {"error": str(e)}
        
        # Recommendation engine stats
        try:
            engine = await get_optimized_recommendation_engine()
            stats["recommendation_engine"] = engine.get_performance_stats()
        except Exception as e:
            stats["recommendation_engine"] = {"error": str(e)}
        
        # System stats
        import psutil
        process = psutil.Process()
        stats["system"] = {
            "cpu_percent": process.cpu_percent(),
            "memory_rss_mb": process.memory_info().rss / (1024 * 1024),
            "memory_vms_mb": process.memory_info().vms / (1024 * 1024),
            "num_threads": process.num_threads(),
            "connections": len(process.connections()) if hasattr(process, 'connections') else 0
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting performance stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Memory optimization endpoint
@app.post("/api/optimize/memory", tags=["optimization"], response_class=ORJSONResponse)
async def optimize_memory():
    """Trigger memory optimization"""
    try:
        memory_manager = await get_memory_manager()
        await memory_manager.aggressive_cleanup()
        
        # Get updated stats
        stats = memory_manager.get_memory_stats()
        
        return {
            "status": "memory_optimized",
            "stats": stats,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Memory optimization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Resource optimization endpoint
@app.post("/api/optimize/resources", tags=["optimization"], response_class=ORJSONResponse)
async def optimize_resources():
    """Trigger resource optimization"""
    try:
        resource_manager = await get_resource_manager()
        await resource_manager.force_optimization()
        
        # Get updated stats
        stats = resource_manager.get_performance_stats()
        
        return {
            "status": "resources_optimized",
            "stats": stats,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Resource optimization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Include routers with optimized prefixes
app.include_router(health.router, prefix="/api/health", tags=["health"])
app.include_router(stocks.router, prefix="/api/stocks", tags=["stocks"])
app.include_router(recommendations.router, prefix="/api/recommendations", tags=["recommendations"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(portfolio.router, prefix="/api/portfolio", tags=["portfolio"])
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])
app.include_router(websocket.router, prefix="/ws", tags=["websocket"])
app.include_router(agents.router, prefix="/api/agents", tags=["agents"])

# Expose the instrumentator
instrumentator.expose(app)


# Optimized application factory
def create_optimized_app() -> FastAPI:
    """Create performance-optimized FastAPI application"""
    return app


if __name__ == "__main__":
    # Production-optimized uvicorn configuration
    uvicorn.run(
        "backend.api.main_performance_optimized:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=1,  # Use single worker with async for better memory sharing
        loop="uvloop",  # Use uvloop for better performance
        http="httptools",  # Use httptools for better HTTP parsing
        log_level=settings.LOG_LEVEL.lower(),
        access_log=settings.ENVIRONMENT == "development",
        reload=settings.ENVIRONMENT == "development",
        # Performance optimizations
        backlog=2048,  # Increased connection backlog
        max_concurrent_connections=1000,  # Increased concurrent connections
        timeout_keep_alive=75,  # Keep-alive timeout
        timeout_graceful_shutdown=30,  # Graceful shutdown timeout
        limit_concurrency=800,  # Limit concurrent requests
        limit_max_requests=10000,  # Max requests per worker before restart
        # SSL optimization (if using HTTPS)
        ssl_version=3 if settings.USE_SSL else None,
        ssl_cert_reqs=0 if settings.USE_SSL else None,
    )