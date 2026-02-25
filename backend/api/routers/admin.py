from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks, Request
from pydantic import BaseModel, Field, EmailStr, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime, date, timedelta, timezone
from enum import Enum
import random
import uuid
import os
import logging

from backend.models.api_response import ApiResponse, success_response
from backend.utils.security_logger import get_security_logger, sanitize_log_input
from backend.utils.comprehensive_cache import get_cache_manager
import backend.services.admin_service as admin_service

logger = logging.getLogger(__name__)
security_logger = get_security_logger()

router = APIRouter(tags=["admin"])

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

# Protected config sections requiring super admin access (Task 1)
PROTECTED_CONFIG_SECTIONS = [
    ConfigSection.API_KEYS,
    ConfigSection.DATABASE,
    ConfigSection.SECURITY
]

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

# Task 5: Command Parameter Validation
class SystemCommand(BaseModel):
    command: str = Field(..., max_length=100)
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    execute_at: Optional[datetime] = None

    @field_validator('command')
    @classmethod
    def validate_command(cls, v):
        """Validate command against whitelist"""
        allowed_commands = [
            'start', 'stop', 'status', 'restart',
            'clear_cache', 'restart_workers', 'run_backup',
            'optimize_database', 'refresh_models', 'sync_data'
        ]
        if v not in allowed_commands:
            raise ValueError(f"Invalid command: {v}")
        return v

    @field_validator('parameters')
    @classmethod
    def sanitize_parameters(cls, v):
        """Sanitize all string parameters"""
        if not v:
            return {}

        sanitized = {}
        for key, value in v.items():
            if isinstance(value, str):
                # Remove control characters and limit length
                sanitized[key] = value.replace('\n', '').replace('\r', '').replace('\t', ' ')[:200]
            elif isinstance(value, (int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, (list, dict)):
                # Convert to string and sanitize
                sanitized[key] = str(value)[:200]
            else:
                sanitized[key] = str(value)[:200]

        return sanitized

# Helper functions - SECURE ADMIN AUTHENTICATION
from backend.auth.oauth2 import get_current_admin_user

def check_admin_permission(current_user = Depends(get_current_admin_user)):
    """Dependency to check admin permissions using JWT authentication"""
    # The get_current_admin_user dependency already validates:
    # 1. Valid JWT token
    # 2. User exists and is active
    # 3. User has admin privileges
    return current_user

# Task 1: Super Admin Check
def check_super_admin_permission(current_user = Depends(get_current_admin_user)):
    """Dependency to check super admin permissions"""
    # Check if user has super_admin attribute
    if not getattr(current_user, 'is_super_admin', False):
        security_logger.log_authorization_failure(
            user_id=current_user.id,
            action="super_admin_required",
            resource="protected_config",
            reason="User is not a super admin"
        )
        raise HTTPException(
            status_code=403,
            detail="Super admin privileges required"
        )
    return current_user

def get_client_ip(request: Request) -> str:
    """Extract client IP from request"""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

# Endpoints
@router.get("/health")
async def get_system_health(current_user = Depends(check_admin_permission)) -> ApiResponse[SystemHealth]:
    """Get comprehensive system health status"""

    return success_response(data=SystemHealth(**admin_service.get_system_health_data()))

@router.get("/users")
async def list_users(
    current_user = Depends(check_admin_permission),
    limit: int = Query(50, le=500),
    offset: int = 0,
    role: Optional[UserRole] = None,
    is_active: Optional[bool] = None
) -> ApiResponse[List[User]]:
    """List all users with filtering options"""

    raw_users = admin_service.list_users(
        limit=limit,
        offset=offset,
        role=role.value if role else None,
        is_active=is_active,
    )
    return success_response(data=[User(**u) for u in raw_users])

@router.get("/users/{user_id}")
async def get_user_details(
    user_id: str,
    request: Request,
    current_user = Depends(check_admin_permission)
) -> ApiResponse[User]:
    """Get detailed information about a specific user"""
    # Task 2 & 4: Structured security logging with sanitized inputs
    security_logger.log_admin_action(
        action="get_user_details",
        user_id=current_user.id,
        resource=f"user:{sanitize_log_input(user_id)}",
        success=True,
        details={"target_user_id": sanitize_log_input(user_id)},
        ip_address=get_client_ip(request)
    )

    return success_response(data=User(**admin_service.get_user_by_id(user_id)))

@router.patch("/users/{user_id}")
async def update_user(
    user_id: str,
    update: UserUpdate,
    request: Request,
    current_user = Depends(check_admin_permission)
) -> ApiResponse[User]:
    """Update user information"""

    try:
        # In production, update user in database
        user = await get_user_details(user_id, request, current_user)

        update_data = update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(user.data if hasattr(user, 'data') else user, field, value)

        # Task 2: Log user management action
        security_logger.log_user_management(
            admin_id=current_user.id,
            action="update_user",
            target_user_id=int(user_id) if user_id.isdigit() else 0,
            success=True,
            details={"updated_fields": list(update_data.keys())},
            ip_address=get_client_ip(request)
        )

        return success_response(data=user.data if hasattr(user, 'data') else user)
    except Exception as e:
        security_logger.log_user_management(
            admin_id=current_user.id,
            action="update_user",
            target_user_id=int(user_id) if user_id.isdigit() else 0,
            success=False,
            details={"error": sanitize_log_input(str(e))},
            ip_address=get_client_ip(request)
        )
        raise

@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    request: Request,
    current_user = Depends(check_admin_permission)
) -> ApiResponse[Dict[str, Any]]:
    """Delete a user account"""

    try:
        # Task 2: Log user deletion
        security_logger.log_user_management(
            admin_id=current_user.id,
            action="delete_user",
            target_user_id=int(user_id) if user_id.isdigit() else 0,
            success=True,
            ip_address=get_client_ip(request)
        )

        return success_response(data=admin_service.delete_user(user_id))
    except Exception as e:
        security_logger.log_user_management(
            admin_id=current_user.id,
            action="delete_user",
            target_user_id=int(user_id) if user_id.isdigit() else 0,
            success=False,
            details={"error": sanitize_log_input(str(e))},
            ip_address=get_client_ip(request)
        )
        raise

@router.get("/analytics/api-usage")
async def get_api_usage_stats(
    current_user = Depends(check_admin_permission),
    days_back: int = Query(7, le=90)
) -> ApiResponse[List[ApiUsageStats]]:
    """Get API usage statistics"""

    raw_stats = admin_service.get_api_usage_stats(days_back=days_back)
    return success_response(data=[ApiUsageStats(**s) for s in raw_stats])

@router.get("/metrics")
async def get_system_metrics(admin: bool = Depends(check_admin_permission)) -> ApiResponse[SystemMetrics]:
    """Get detailed system metrics"""

    return success_response(data=SystemMetrics(**admin_service.get_system_metrics_data()))

@router.get("/jobs")
async def list_background_jobs(
    current_user = Depends(check_admin_permission),
    status: Optional[JobStatus] = None
) -> ApiResponse[List[BackgroundJob]]:
    """List background jobs"""

    raw_jobs = admin_service.list_background_jobs(
        status=status.value if status else None
    )
    return success_response(data=[BackgroundJob(**j) for j in raw_jobs])

@router.post("/jobs/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    current_user = Depends(check_admin_permission)
) -> ApiResponse[Dict[str, Any]]:
    """Cancel a running job"""

    return success_response(data=admin_service.cancel_job(job_id))

@router.post("/jobs/{job_id}/retry")
async def retry_job(
    job_id: str,
    current_user = Depends(check_admin_permission)
) -> ApiResponse[Dict[str, Any]]:
    """Retry a failed job"""

    return success_response(data=admin_service.retry_job(job_id))

@router.get("/config")
async def get_configuration(
    current_user = Depends(check_admin_permission),
    section: Optional[ConfigSection] = None
) -> ApiResponse[Dict[str, Any]]:
    """Get system configuration"""

    return success_response(
        data=admin_service.get_configuration(
            section=section.value if section else None
        )
    )

@router.patch("/config")
async def update_configuration(
    update: ConfigUpdate,
    request: Request,
    current_user = Depends(check_admin_permission),
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> ApiResponse[Dict[str, Any]]:
    """Update system configuration"""

    try:
        # Task 1: Check if section is protected and requires super admin
        if update.section in PROTECTED_CONFIG_SECTIONS:
            if not getattr(current_user, 'is_super_admin', False):
                security_logger.log_authorization_failure(
                    user_id=current_user.id,
                    action="update_protected_config",
                    resource=f"config:{update.section.value}",
                    reason=f"Super admin privileges required to modify {update.section.value}",
                    ip_address=get_client_ip(request)
                )
                raise HTTPException(
                    status_code=403,
                    detail=f"Super admin privileges required to modify {update.section.value}"
                )

        # Task 2: Log configuration change
        security_logger.log_config_change(
            user_id=current_user.id,
            section=update.section.value,
            key=sanitize_log_input(update.key),
            old_value=None,  # Would fetch from current config in production
            new_value=update.value,
            success=True,
            ip_address=get_client_ip(request)
        )

        # In production, update configuration in database/config file
        background_tasks.add_task(reload_configuration, update.section)

        return success_response(
            data=admin_service.apply_configuration_update(
                section=update.section.value,
                key=update.key,
                value=update.value,
            )
        )
    except HTTPException:
        raise
    except Exception as e:
        security_logger.log_config_change(
            user_id=current_user.id,
            section=update.section.value,
            key=sanitize_log_input(update.key),
            old_value=None,
            new_value=update.value,
            success=False,
            ip_address=get_client_ip(request)
        )
        raise

@router.get("/cache/stats")
async def get_cache_stats(
    current_user = Depends(check_admin_permission)
) -> ApiResponse[Dict[str, Any]]:
    """
    Get comprehensive cache statistics including hit rates per key prefix
    Admin-only endpoint for monitoring cache performance
    """
    try:
        cache_manager = await get_cache_manager()
        metrics = await cache_manager.get_metrics()

        return success_response(data={
            "cache_metrics": metrics.get("cache_metrics", {}),
            "prefix_stats": metrics.get("prefix_stats", {}),
            "l1_cache_stats": metrics.get("l1_cache_stats", {}),
            "l2_cache_stats": metrics.get("l2_cache_stats", {}),
            "storage_bytes": metrics.get("storage_bytes", 0),
            "active_warming_tasks": metrics.get("active_warming_tasks", 0),
            "ttl_policies": metrics.get("ttl_policies", {}),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve cache statistics")

@router.get("/audit-logs")
async def get_audit_logs(
    current_user = Depends(check_admin_permission),
    user_id: Optional[str] = None,
    action: Optional[str] = None,
    limit: int = Query(100, le=1000),
    offset: int = 0
) -> ApiResponse[List[AuditLog]]:
    """Get audit logs"""

    raw_logs = admin_service.get_audit_logs(
        user_id=user_id,
        action=action,
        limit=limit,
        offset=offset,
    )
    return success_response(data=[AuditLog(**log) for log in raw_logs])

@router.post("/announcements")
async def create_announcement(
    announcement: Announcement,
    current_user = Depends(check_admin_permission)
) -> ApiResponse[Announcement]:
    """Create a system announcement"""

    announcement.id = str(uuid.uuid4())
    return success_response(data=announcement)

@router.get("/announcements")
async def list_announcements(
    current_user = Depends(check_admin_permission),
    active_only: bool = True
) -> ApiResponse[List[Announcement]]:
    """List system announcements"""

    raw = admin_service.list_announcements(active_only=active_only)
    return success_response(data=[Announcement(**a) for a in raw])

@router.post("/export")
async def export_data(
    export_request: DataExport,
    request: Request,
    current_user = Depends(check_admin_permission),
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> ApiResponse[Dict[str, Any]]:
    """Export system data"""

    # Task 2: Log data export
    security_logger.log_data_export(
        user_id=current_user.id,
        export_type=sanitize_log_input(export_request.export_type),
        record_count=0,  # Would be actual count in production
        success=True,
        ip_address=get_client_ip(request)
    )

    result = admin_service.initiate_data_export(
        export_type=export_request.export_type
    )
    background_tasks.add_task(process_data_export, result["job_id"], export_request)

    return success_response(data=result)

@router.post("/command")
async def execute_system_command(
    command: SystemCommand,
    request: Request,
    current_user = Depends(check_admin_permission)
) -> ApiResponse[Dict[str, Any]]:
    """Execute a system command with validation and logging"""

    execution_time_ms = random.randint(100, 5000)

    try:
        # Task 2 & 5: Log system command execution with validated parameters
        security_logger.log_system_command(
            user_id=current_user.id,
            command=command.command,
            parameters=command.parameters,
            success=True,
            execution_time_ms=execution_time_ms,
            ip_address=get_client_ip(request)
        )

        return success_response(
            data=admin_service.execute_system_command(
                command=command.command,
                execution_time_ms=execution_time_ms,
            )
        )
    except Exception as e:
        security_logger.log_system_command(
            user_id=current_user.id,
            command=command.command,
            parameters=command.parameters,
            success=False,
            ip_address=get_client_ip(request)
        )
        raise

@router.post("/maintenance/enable")
async def enable_maintenance_mode(
    current_user = Depends(check_admin_permission),
    message: str = "System is under maintenance"
) -> ApiResponse[Dict[str, Any]]:
    """Enable maintenance mode"""

    return success_response(data=admin_service.enable_maintenance_mode(message=message))

@router.post("/maintenance/disable")
async def disable_maintenance_mode(admin: bool = Depends(check_admin_permission)) -> ApiResponse[Dict[str, Any]]:
    """Disable maintenance mode"""

    return success_response(data=admin_service.disable_maintenance_mode())

# Background task functions
async def reload_configuration(section: ConfigSection):
    """Reload configuration after update"""
    print(f"Reloading configuration section: {section}")

async def process_data_export(job_id: str, export_request: DataExport):
    """Process data export in background"""
    print(f"Processing export job {job_id}: {export_request.export_type}")
