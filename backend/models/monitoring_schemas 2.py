"""
Pydantic schemas for monitoring and observability endpoints
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime

class ServiceHealth(BaseModel):
    """Individual service health status"""
    status: str
    message: Optional[str] = None

class HealthCheckResponse(BaseModel):
    """System health check response"""
    status: str
    timestamp: str
    services: Dict[str, Any]

class CostMetrics(BaseModel):
    """Cost tracking metrics response"""
    daily_costs: Dict[str, float]
    monthly_estimate: float
    budget_remaining: float
    budget_percentage: float

class DashboardLinks(BaseModel):
    """Grafana dashboard URLs"""
    main: str
    api_usage: str
    ml_performance: str
    cost_tracking: str
    system_metrics: str

class AnnotationResponse(BaseModel):
    """Grafana annotation creation response"""
    message: str

class AlertTestResponse(BaseModel):
    """Alert system test response"""
    alert_created: bool
    grafana_connected: bool

class ProviderUsage(BaseModel):
    """API provider usage metrics"""
    daily_limit: int
    used_today: int
    remaining: int
    minute_limit: Optional[int] = None

class ApiUsageMetrics(BaseModel):
    """API usage metrics for all providers"""
    alpha_vantage: ProviderUsage
    finnhub: Dict[str, int]
    polygon: Dict[str, int]
