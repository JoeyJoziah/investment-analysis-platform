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

from backend.api.routers import (
    stocks, analysis, recommendations, portfolio,
    auth, health, websocket, admin  # , agents - temporarily disabled
)
from backend.utils.database import init_db, close_db
from backend.config.database import initialize_database, cleanup_database
from backend.utils.cache import init_cache
from backend.utils.monitoring import PrometheusMiddleware, export_metrics
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
    logger.info("Cache initialized")
    
    # Start background tasks
    from backend.tasks.scheduler import start_scheduler
    scheduler = await start_scheduler()
    logger.info("Background scheduler started")
    
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

# Add CORS middleware with secure configuration
from backend.utils.cors import setup_cors
setup_cors(app)

# Add Prometheus monitoring
app.add_middleware(PrometheusMiddleware)

# Include routers
app.include_router(health.router, prefix="/api/health", tags=["health"])
app.include_router(auth.router, prefix="/api/auth", tags=["authentication"])
app.include_router(stocks.router, prefix="/api/stocks", tags=["stocks"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(recommendations.router, prefix="/api/recommendations", tags=["recommendations"])
app.include_router(portfolio.router, prefix="/api/portfolio", tags=["portfolio"])
app.include_router(websocket.router, prefix="/api/ws", tags=["websocket"])
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])
# app.include_router(agents.router, prefix="/api/agents", tags=["agents"]) # Temporarily disabled


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """
    Global HTTP exception handler
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """
    Global exception handler for unhandled errors
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url)
        }
    )


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