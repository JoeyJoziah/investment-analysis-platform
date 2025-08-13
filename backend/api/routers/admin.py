from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime, date, timedelta
from enum import Enum
import random
import uuid
import os
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])

# Enums
class UserRole(str, Enum):
    SUPER_ADMIN = "super_admin"
    ADMIN = "admin"
    MODERATOR = "moderator"
    ANALYST = "analyst"
    USER = "user"

class SystemStatus(str, Enum):
    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    PARTIAL_OUTAGE = "partial_outage"
    MAJOR_OUTAGE = "major_outage"
    MAINTENANCE = "maintenance"

class ServiceStatus(str, Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    STARTING = "starting"
    STOPPING = "stopping"
    ERROR = "error"

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ConfigSection(str, Enum):
    API_KEYS = "api_keys"
    DATABASE = "database"
    CACHE = "cache"
    SECURITY = "security"
    FEATURES = "features"
    LIMITS = "limits"
    MONITORING = "monitoring"

# Pydantic models
class SystemHealth(BaseModel):
    status: SystemStatus
    uptime: int  # seconds
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    active_connections: int
    request_rate: float  # requests per second
    error_rate: float
    response_time_avg: float  # milliseconds
    services: Dict[str, ServiceStatus]
    last_check: datetime

class User(BaseModel):
    id: str
    email: EmailStr
    full_name: str
    role: UserRole
    is_active: bool
    is_verified: bool
    created_at: datetime
    last_login: Optional[datetime]
    subscription_tier: Optional[str]
    api_calls_today: int
    storage_used_mb: float

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None
    is_verified: Optional[bool] = None
    subscription_tier: Optional[str] = None

class ApiUsageStats(BaseModel):
    endpoint: str
    method: str
    total_calls: int
    successful_calls: int
    failed_calls: int
    avg_response_time: float
    p95_response_time: float
    p99_response_time: float
    total_data_transferred: float  # MB
    unique_users: int
    last_called: datetime

class SystemMetrics(BaseModel):
    timestamp: datetime
    cpu: Dict[str, float]
    memory: Dict[str, float]
    disk: Dict[str, float]
    network: Dict[str, float]
    database: Dict[str, Any]
    cache: Dict[str, Any]
    queue: Dict[str, int]

class BackgroundJob(BaseModel):
    id: str
    name: str
    type: str
    status: JobStatus
    progress: float = Field(..., ge=0, le=100)
    started_at: datetime
    completed_at: Optional[datetime]
    error_message: Optional[str]
    result: Optional[Dict[str, Any]]
    retry_count: int

class ConfigUpdate(BaseModel):
    section: ConfigSection
    key: str
    value: Any
    description: Optional[str] = None

class AuditLog(BaseModel):
    id: str
    timestamp: datetime
    user_id: str
    user_email: str
    action: str
    resource_type: str
    resource_id: Optional[str]
    details: Dict[str, Any]
    ip_address: str
    user_agent: str
    success: bool
    error_message: Optional[str] = None

class Announcement(BaseModel):
    id: str
    title: str
    message: str
    type: str  # info, warning, critical
    active: bool
    start_time: datetime
    end_time: Optional[datetime]
    target_users: Optional[List[str]] = None  # None means all users

class DataExport(BaseModel):
    export_type: str  # users, transactions, analytics, logs
    format: str  # csv, json, excel
    date_range: Optional[Dict[str, date]] = None
    filters: Optional[Dict[str, Any]] = None

class SystemCommand(BaseModel):
    command: str
    parameters: Optional[Dict[str, Any]] = None
    execute_at: Optional[datetime] = None

# Helper functions - SECURE ADMIN AUTHENTICATION
from backend.auth.oauth2 import get_current_admin_user

def check_admin_permission(current_user = Depends(get_current_admin_user)):
    """Dependency to check admin permissions using JWT authentication"""
    # The get_current_admin_user dependency already validates:
    # 1. Valid JWT token
    # 2. User exists and is active
    # 3. User has admin privileges
    return current_user

# Endpoints
@router.get("/health", response_model=SystemHealth)
async def get_system_health(current_user = Depends(check_admin_permission)) -> SystemHealth:
    """Get comprehensive system health status"""
    
    return SystemHealth(
        status=SystemStatus.OPERATIONAL,
        uptime=random.randint(86400, 864000),  # 1-10 days in seconds
        cpu_usage=random.uniform(20, 80),
        memory_usage=random.uniform(30, 70),
        disk_usage=random.uniform(40, 60),
        active_connections=random.randint(10, 100),
        request_rate=random.uniform(10, 100),
        error_rate=random.uniform(0, 5),
        response_time_avg=random.uniform(50, 200),
        services={
            "api": ServiceStatus.RUNNING,
            "database": ServiceStatus.RUNNING,
            "cache": ServiceStatus.RUNNING,
            "worker": ServiceStatus.RUNNING,
            "scheduler": ServiceStatus.RUNNING,
            "websocket": ServiceStatus.RUNNING
        },
        last_check=datetime.utcnow()
    )

@router.get("/users", response_model=List[User])
async def list_users(
    current_user = Depends(check_admin_permission),
    limit: int = Query(50, le=500),
    offset: int = 0,
    role: Optional[UserRole] = None,
    is_active: Optional[bool] = None
) -> List[User]:
    """List all users with filtering options"""
    
    users = []
    for i in range(100):
        user = User(
            id=str(uuid.uuid4()),
            email=f"user{i}@example.com",
            full_name=f"User {i}",
            role=random.choice(list(UserRole)),
            is_active=random.choice([True, False]),
            is_verified=random.choice([True, False]),
            created_at=datetime.utcnow() - timedelta(days=random.randint(1, 365)),
            last_login=datetime.utcnow() - timedelta(days=random.randint(0, 30)) if random.random() > 0.3 else None,
            subscription_tier=random.choice([None, "free", "basic", "premium", "enterprise"]),
            api_calls_today=random.randint(0, 1000),
            storage_used_mb=random.uniform(0, 1000)
        )
        
        if role and user.role != role:
            continue
        if is_active is not None and user.is_active != is_active:
            continue
        
        users.append(user)
    
    return users[offset:offset + limit]

@router.get("/users/{user_id}", response_model=User)
async def get_user_details(
    user_id: str,
    current_user = Depends(check_admin_permission)
) -> User:
    """Get detailed information about a specific user"""
    logger.info(f"Admin {current_user.username} accessing user details for {user_id}")
    
    return User(
        id=user_id,
        email="user@example.com",
        full_name="John Doe",
        role=UserRole.USER,
        is_active=True,
        is_verified=True,
        created_at=datetime.utcnow() - timedelta(days=180),
        last_login=datetime.utcnow() - timedelta(hours=2),
        subscription_tier="premium",
        api_calls_today=150,
        storage_used_mb=250.5
    )

@router.patch("/users/{user_id}", response_model=User)
async def update_user(
    user_id: str,
    update: UserUpdate,
    current_user = Depends(check_admin_permission)
) -> User:
    """Update user information"""
    
    # In production, update user in database
    user = await get_user_details(user_id, current_user)
    
    update_data = update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(user, field, value)
    
    return user

@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    current_user = Depends(check_admin_permission)
) -> Dict[str, str]:
    """Delete a user account"""
    
    return {
        "message": f"User {user_id} has been deleted",
        "status": "success"
    }

@router.get("/analytics/api-usage", response_model=List[ApiUsageStats])
async def get_api_usage_stats(
    current_user = Depends(check_admin_permission),
    days_back: int = Query(7, le=90)
) -> List[ApiUsageStats]:
    """Get API usage statistics"""
    
    endpoints = [
        ("/stocks", "GET"),
        ("/stocks/{symbol}", "GET"),
        ("/analysis/analyze", "POST"),
        ("/recommendations", "GET"),
        ("/portfolio", "GET"),
        ("/auth/login", "POST")
    ]
    
    stats = []
    for endpoint, method in endpoints:
        stats.append(ApiUsageStats(
            endpoint=endpoint,
            method=method,
            total_calls=random.randint(1000, 50000),
            successful_calls=random.randint(900, 49000),
            failed_calls=random.randint(10, 1000),
            avg_response_time=random.uniform(50, 500),
            p95_response_time=random.uniform(100, 1000),
            p99_response_time=random.uniform(200, 2000),
            total_data_transferred=random.uniform(100, 10000),
            unique_users=random.randint(10, 500),
            last_called=datetime.utcnow() - timedelta(minutes=random.randint(0, 60))
        ))
    
    return stats

@router.get("/metrics", response_model=SystemMetrics)
async def get_system_metrics(admin: bool = Depends(check_admin_permission)) -> SystemMetrics:
    """Get detailed system metrics"""
    
    return SystemMetrics(
        timestamp=datetime.utcnow(),
        cpu={
            "usage_percent": random.uniform(20, 80),
            "load_average_1m": random.uniform(0.5, 2.0),
            "load_average_5m": random.uniform(0.5, 2.0),
            "load_average_15m": random.uniform(0.5, 2.0),
            "cores": 8
        },
        memory={
            "total_gb": 16,
            "used_gb": random.uniform(4, 12),
            "free_gb": random.uniform(4, 12),
            "cached_gb": random.uniform(1, 4),
            "usage_percent": random.uniform(30, 75)
        },
        disk={
            "total_gb": 500,
            "used_gb": random.uniform(100, 300),
            "free_gb": random.uniform(200, 400),
            "usage_percent": random.uniform(20, 60),
            "read_mb_s": random.uniform(10, 100),
            "write_mb_s": random.uniform(5, 50)
        },
        network={
            "bytes_sent": random.randint(1000000, 10000000),
            "bytes_recv": random.randint(1000000, 10000000),
            "packets_sent": random.randint(10000, 100000),
            "packets_recv": random.randint(10000, 100000),
            "errors": random.randint(0, 10),
            "dropped": random.randint(0, 5)
        },
        database={
            "connections_active": random.randint(5, 50),
            "connections_idle": random.randint(10, 100),
            "queries_per_second": random.uniform(10, 100),
            "slow_queries": random.randint(0, 10),
            "replication_lag_ms": random.uniform(0, 100)
        },
        cache={
            "hits": random.randint(10000, 100000),
            "misses": random.randint(100, 1000),
            "hit_rate": random.uniform(0.85, 0.99),
            "memory_used_mb": random.uniform(100, 500),
            "evictions": random.randint(0, 100)
        },
        queue={
            "pending": random.randint(0, 100),
            "processing": random.randint(0, 20),
            "completed": random.randint(1000, 10000),
            "failed": random.randint(0, 50),
            "retry": random.randint(0, 10)
        }
    )

@router.get("/jobs", response_model=List[BackgroundJob])
async def list_background_jobs(
    current_user = Depends(check_admin_permission),
    status: Optional[JobStatus] = None
) -> List[BackgroundJob]:
    """List background jobs"""
    
    jobs = []
    job_types = ["data_sync", "analysis", "report_generation", "cleanup", "backup"]
    
    for i in range(20):
        job_status = status or random.choice(list(JobStatus))
        started = datetime.utcnow() - timedelta(minutes=random.randint(0, 120))
        
        jobs.append(BackgroundJob(
            id=str(uuid.uuid4()),
            name=f"Job_{i}",
            type=random.choice(job_types),
            status=job_status,
            progress=random.uniform(0, 100) if job_status == JobStatus.RUNNING else 100 if job_status == JobStatus.COMPLETED else 0,
            started_at=started,
            completed_at=started + timedelta(minutes=random.randint(1, 30)) if job_status == JobStatus.COMPLETED else None,
            error_message="Connection timeout" if job_status == JobStatus.FAILED else None,
            result={"records_processed": random.randint(100, 10000)} if job_status == JobStatus.COMPLETED else None,
            retry_count=random.randint(0, 3)
        ))
    
    if status:
        jobs = [j for j in jobs if j.status == status]
    
    return jobs

@router.post("/jobs/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    current_user = Depends(check_admin_permission)
) -> Dict[str, str]:
    """Cancel a running job"""
    
    return {
        "message": f"Job {job_id} has been cancelled",
        "status": "success"
    }

@router.post("/jobs/{job_id}/retry")
async def retry_job(
    job_id: str,
    current_user = Depends(check_admin_permission)
) -> Dict[str, str]:
    """Retry a failed job"""
    
    return {
        "message": f"Job {job_id} has been queued for retry",
        "status": "success",
        "new_job_id": str(uuid.uuid4())
    }

@router.get("/config", response_model=Dict[str, Any])
async def get_configuration(
    current_user = Depends(check_admin_permission),
    section: Optional[ConfigSection] = None
) -> Dict[str, Any]:
    """Get system configuration"""
    
    config = {
        "api_keys": {
            "alpha_vantage": "***REDACTED***",
            "finnhub": "***REDACTED***",
            "polygon": "***REDACTED***",
            "news_api": "***REDACTED***"
        },
        "database": {
            "host": "postgres",
            "port": 5432,
            "name": "investment_db",
            "pool_size": 20,
            "max_overflow": 10
        },
        "cache": {
            "host": "redis",
            "port": 6379,
            "ttl_default": 300,
            "max_memory": "512mb"
        },
        "security": {
            "jwt_expiration_minutes": 1440,
            "password_min_length": 8,
            "require_2fa": False,
            "allowed_origins": ["http://localhost:3000"]
        },
        "features": {
            "real_time_quotes": True,
            "ml_predictions": True,
            "social_sentiment": True,
            "options_trading": False,
            "crypto_trading": False
        },
        "limits": {
            "max_api_calls_per_minute": 60,
            "max_portfolio_size": 100,
            "max_watchlist_size": 50,
            "max_concurrent_connections": 1000
        },
        "monitoring": {
            "prometheus_enabled": True,
            "grafana_enabled": True,
            "sentry_enabled": False,
            "log_level": "INFO"
        }
    }
    
    if section:
        return {section: config.get(section, {})}
    
    return config

@router.patch("/config")
async def update_configuration(
    update: ConfigUpdate,
    current_user = Depends(check_admin_permission),
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> Dict[str, str]:
    """Update system configuration"""
    
    # In production, update configuration in database/config file
    background_tasks.add_task(reload_configuration, update.section)
    
    return {
        "message": f"Configuration updated: {update.section}.{update.key}",
        "status": "success",
        "requires_restart": update.section in [ConfigSection.DATABASE, ConfigSection.CACHE]
    }

@router.get("/audit-logs", response_model=List[AuditLog])
async def get_audit_logs(
    current_user = Depends(check_admin_permission),
    user_id: Optional[str] = None,
    action: Optional[str] = None,
    limit: int = Query(100, le=1000),
    offset: int = 0
) -> List[AuditLog]:
    """Get audit logs"""
    
    logs = []
    actions = ["login", "logout", "create", "update", "delete", "export", "import"]
    resources = ["user", "portfolio", "trade", "configuration", "report"]
    
    for i in range(200):
        log = AuditLog(
            id=str(uuid.uuid4()),
            timestamp=datetime.utcnow() - timedelta(minutes=random.randint(0, 10080)),
            user_id=user_id or str(uuid.uuid4()),
            user_email=f"user{i % 20}@example.com",
            action=action or random.choice(actions),
            resource_type=random.choice(resources),
            resource_id=str(uuid.uuid4()) if random.random() > 0.3 else None,
            details={"ip": f"192.168.1.{random.randint(1, 255)}"},
            ip_address=f"192.168.1.{random.randint(1, 255)}",
            user_agent="Mozilla/5.0...",
            success=random.random() > 0.1,
            error_message="Permission denied" if random.random() < 0.1 else None
        )
        
        if user_id and log.user_id != user_id:
            continue
        if action and log.action != action:
            continue
        
        logs.append(log)
    
    return sorted(logs, key=lambda x: x.timestamp, reverse=True)[offset:offset + limit]

@router.post("/announcements", response_model=Announcement)
async def create_announcement(
    announcement: Announcement,
    current_user = Depends(check_admin_permission)
) -> Announcement:
    """Create a system announcement"""
    
    announcement.id = str(uuid.uuid4())
    return announcement

@router.get("/announcements", response_model=List[Announcement])
async def list_announcements(
    current_user = Depends(check_admin_permission),
    active_only: bool = True
) -> List[Announcement]:
    """List system announcements"""
    
    announcements = [
        Announcement(
            id=str(uuid.uuid4()),
            title="Scheduled Maintenance",
            message="System will be under maintenance on Sunday 2 AM - 4 AM EST",
            type="warning",
            active=True,
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow() + timedelta(days=7)
        ),
        Announcement(
            id=str(uuid.uuid4()),
            title="New Features Released",
            message="Check out our new portfolio analytics dashboard!",
            type="info",
            active=True,
            start_time=datetime.utcnow() - timedelta(days=2),
            end_time=None
        )
    ]
    
    if active_only:
        announcements = [a for a in announcements if a.active]
    
    return announcements

@router.post("/export", response_model=Dict[str, Any])
async def export_data(
    export_request: DataExport,
    current_user = Depends(check_admin_permission),
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> Dict[str, Any]:
    """Export system data"""
    
    job_id = str(uuid.uuid4())
    background_tasks.add_task(process_data_export, job_id, export_request)
    
    return {
        "job_id": job_id,
        "status": "processing",
        "estimated_time_seconds": random.randint(30, 300),
        "download_url": f"/admin/export/{job_id}/download"
    }

@router.post("/command", response_model=Dict[str, Any])
async def execute_system_command(
    command: SystemCommand,
    current_user = Depends(check_admin_permission)
) -> Dict[str, Any]:
    """Execute a system command"""
    
    # List of allowed commands
    allowed_commands = [
        "clear_cache",
        "restart_workers",
        "run_backup",
        "optimize_database",
        "refresh_models",
        "sync_data"
    ]
    
    if command.command not in allowed_commands:
        raise HTTPException(status_code=400, detail=f"Command not allowed: {command.command}")
    
    return {
        "command": command.command,
        "status": "executed",
        "result": {
            "success": True,
            "message": f"Command {command.command} executed successfully",
            "execution_time_ms": random.randint(100, 5000)
        }
    }

@router.post("/maintenance/enable")
async def enable_maintenance_mode(
    current_user = Depends(check_admin_permission),
    message: str = "System is under maintenance"
) -> Dict[str, str]:
    """Enable maintenance mode"""
    
    return {
        "status": "maintenance_enabled",
        "message": message
    }

@router.post("/maintenance/disable")
async def disable_maintenance_mode(admin: bool = Depends(check_admin_permission)) -> Dict[str, str]:
    """Disable maintenance mode"""
    
    return {
        "status": "maintenance_disabled",
        "message": "System is operational"
    }

# Background task functions
async def reload_configuration(section: ConfigSection):
    """Reload configuration after update"""
    print(f"Reloading configuration section: {section}")

async def process_data_export(job_id: str, export_request: DataExport):
    """Process data export in background"""
    print(f"Processing export job {job_id}: {export_request.export_type}")