"""
Investment Analysis Platform - API Routers
This module exports all API routers for the FastAPI application.
"""

from . import (
    stocks,
    analysis,
    recommendations,
    portfolio,
    auth,
    health,
    admin,
    cache_management,
    websocket,
    agents,
    monitoring
)

__all__ = [
    "stocks",
    "analysis",
    "recommendations",
    "portfolio",
    "auth",
    "health",
    "admin",
    "cache_management",
    "websocket",
    "agents",
    "monitoring"
]
