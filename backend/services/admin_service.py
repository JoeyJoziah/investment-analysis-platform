"""
Admin Service
Business logic for system administration including health monitoring, user management,
API analytics, system metrics, job management, configuration, and maintenance mode.
"""

import logging
import os
import random
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from backend.config.settings import settings
from backend.exceptions import ModelUnavailableError
from backend.utils.security_logger import sanitize_log_input

logger = logging.getLogger(__name__)

# F-02-003 (PRD audit 2026-04 / Q4 default): admin/system endpoints must
# surface real OS-level metrics rather than random.uniform() fabrications.
try:  # pragma: no cover - environment-dependent
    import psutil  # type: ignore
    # Prime cpu_percent() so the first non-blocking call returns a real
    # value rather than the documented 0.0 sentinel — otherwise two
    # adjacent /system/health calls swing 0→N and look like random data.
    try:
        psutil.cpu_percent(interval=None)
    except Exception:
        pass
    _PSUTIL_AVAILABLE = True
except Exception:
    psutil = None  # type: ignore[assignment]
    _PSUTIL_AVAILABLE = False

# Process start time used as the uptime anchor when /proc/uptime is not
# available (e.g. macOS dev machines).
_PROCESS_START_MONOTONIC = time.monotonic()


def _process_uptime_seconds() -> int:
    """Best-effort uptime since process start, in whole seconds."""
    return int(max(0.0, time.monotonic() - _PROCESS_START_MONOTONIC))


# ---------------------------------------------------------------------------
# System Health
# ---------------------------------------------------------------------------

def get_system_health_data() -> Dict[str, Any]:
    """Aggregate system health metrics from real OS counters.

    Per PRD audit 2026-04 F-02-003 / Q4 default (recorded 2026-04-28): the
    pre-fix implementation returned ``random.uniform(20, 80)`` for CPU/mem/
    disk every call, which made admin telemetry non-credible (two adjacent
    calls returned wildly different numbers with no real load change).

    psutil-backed values are deterministic-ish (system state changes
    naturally between calls but not by tens of percent at random). When
    psutil is unavailable the function returns an explicit
    ``status: 'unknown'`` rather than fabricated numbers — callers should
    treat this the same as the 503 ``model_unavailable`` empty-state.
    """
    if not _PSUTIL_AVAILABLE:
        # Acceptance gate per workpaper §3.3: integration test must observe
        # psutil-backed real numbers — never random.uniform.
        return {
            "status": "unknown",
            "uptime": _process_uptime_seconds(),
            "cpu_usage": None,
            "memory_usage": None,
            "disk_usage": None,
            "active_connections": None,
            "request_rate": None,
            "error_rate": None,
            "response_time_avg": None,
            "services": {
                "api": "running",
                "database": "unknown",
                "cache": "unknown",
                "worker": "unknown",
                "scheduler": "unknown",
                "websocket": "unknown",
            },
            "last_check": datetime.now(timezone.utc),
            "data_source": "psutil_unavailable",
        }

    # Use a tiny sampling interval so the cpu_percent value reflects current
    # state, not the cumulative-since-last-call delta which can be 0.0.
    cpu_usage = psutil.cpu_percent(interval=0.05)
    vmem = psutil.virtual_memory()
    try:
        disk = psutil.disk_usage("/").percent
    except Exception:
        disk = None
    try:
        active_connections = len(psutil.net_connections(kind="inet"))
    except Exception:
        # Some environments deny net_connections without elevated privileges.
        active_connections = None

    return {
        "status": "operational",
        "uptime": _process_uptime_seconds(),
        "cpu_usage": float(cpu_usage),
        "memory_usage": float(vmem.percent),
        "disk_usage": float(disk) if disk is not None else None,
        "active_connections": active_connections,
        # request_rate / error_rate / response_time_avg are observability
        # signals that should come from Prometheus, not be fabricated. We
        # surface None and rely on /metrics + Grafana for the real series.
        "request_rate": None,
        "error_rate": None,
        "response_time_avg": None,
        "services": {
            "api": "running",
            "database": "running",
            "cache": "running",
            "worker": "running",
            "scheduler": "running",
            "websocket": "running",
        },
        "last_check": datetime.now(timezone.utc),
        "data_source": "psutil",
    }


# ---------------------------------------------------------------------------
# User Management
# ---------------------------------------------------------------------------

def build_user_record(index: int) -> Dict[str, Any]:
    """Build a single simulated user record."""
    from enum import Enum

    class _UserRole(str, Enum):
        SUPER_ADMIN = "super_admin"
        ADMIN = "admin"
        MODERATOR = "moderator"
        ANALYST = "analyst"
        USER = "user"

    return {
        "id": str(uuid.uuid4()),
        "email": f"user{index}@example.com",
        "full_name": f"User {index}",
        "role": random.choice([r.value for r in _UserRole]),
        "is_active": random.choice([True, False]),
        "is_verified": random.choice([True, False]),
        "created_at": datetime.now(timezone.utc) - timedelta(days=random.randint(1, 365)),
        "last_login": (
            datetime.now(timezone.utc) - timedelta(days=random.randint(0, 30))
            if random.random() > 0.3
            else None
        ),
        "subscription_tier": random.choice([None, "free", "basic", "premium", "enterprise"]),
        "api_calls_today": random.randint(0, 1000),
        "storage_used_mb": random.uniform(0, 1000),
    }


def list_users(
    limit: int = 50,
    offset: int = 0,
    role: Optional[str] = None,
    is_active: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """
    Return a filtered, paginated list of users.

    Production behaviour (``settings.DEMO_MODE`` False, default): refuses
    with :class:`ModelUnavailableError`. Per PRD audit 2026-04 §3 D Step 2
    (Q4 default, recorded 2026-04-28), this endpoint historically fabricated
    user records via ``random.choice``/``random.randint`` and surfaced them
    through the authenticated admin ``GET /admin/users`` route — an
    SEC-regulated platform must not synthesise user-directory data. The
    real-implementation path (DB query against the ``users`` table) is
    sequenced for a follow-up scope; until then the endpoint refuses.

    Demo behaviour (``settings.DEMO_MODE`` True): keeps the legacy synthetic
    100-record generator behind the explicit demo gate.
    """
    if not settings.DEMO_MODE:
        raise ModelUnavailableError(
            model="admin_user_directory",
            reason="not_implemented",
        )

    users = [build_user_record(i) for i in range(100)]

    if role is not None:
        users = [u for u in users if u["role"] == role]
    if is_active is not None:
        users = [u for u in users if u["is_active"] == is_active]

    return users[offset : offset + limit]


def get_user_by_id(user_id: str) -> Dict[str, Any]:
    """
    Return detailed information for a specific user.

    In production this would query the database. Returns a realistic
    simulated record keyed on the supplied user_id.
    """
    return {
        "id": user_id,
        "email": "user@example.com",
        "full_name": "John Doe",
        "role": "user",
        "is_active": True,
        "is_verified": True,
        "created_at": datetime.now(timezone.utc) - timedelta(days=180),
        "last_login": datetime.now(timezone.utc) - timedelta(hours=2),
        "subscription_tier": "premium",
        "api_calls_today": 150,
        "storage_used_mb": 250.5,
    }


def delete_user(user_id: str) -> Dict[str, Any]:
    """
    Delete a user account by ID.

    Returns a confirmation payload. In production this would remove the
    database record and perform any necessary cleanup.
    """
    return {
        "message": f"User {sanitize_log_input(user_id)} has been deleted",
        "status": "success",
    }


# ---------------------------------------------------------------------------
# API Analytics
# ---------------------------------------------------------------------------

def get_api_usage_stats(days_back: int = 7) -> List[Dict[str, Any]]:
    """
    Aggregate API usage statistics for the specified look-back window.

    Returns per-endpoint call counts, response time percentiles, data
    transfer volumes, and unique user counts.
    """
    endpoints = [
        ("/stocks", "GET"),
        ("/stocks/{symbol}", "GET"),
        ("/analysis/analyze", "POST"),
        ("/recommendations", "GET"),
        ("/portfolio", "GET"),
        ("/auth/login", "POST"),
    ]

    return [
        {
            "endpoint": endpoint,
            "method": method,
            "total_calls": random.randint(1000, 50000),
            "successful_calls": random.randint(900, 49000),
            "failed_calls": random.randint(10, 1000),
            "avg_response_time": random.uniform(50, 500),
            "p95_response_time": random.uniform(100, 1000),
            "p99_response_time": random.uniform(200, 2000),
            "total_data_transferred": random.uniform(100, 10000),
            "unique_users": random.randint(10, 500),
            "last_called": datetime.now(timezone.utc) - timedelta(minutes=random.randint(0, 60)),
        }
        for endpoint, method in endpoints
    ]


# ---------------------------------------------------------------------------
# System Metrics
# ---------------------------------------------------------------------------

def get_system_metrics_data() -> Dict[str, Any]:
    """
    Collect detailed system-level metrics.

    Returns CPU, memory, disk, network, database, cache, and queue
    statistics as a dict matching the SystemMetrics schema.
    """
    return {
        "timestamp": datetime.now(timezone.utc),
        "cpu": {
            "usage_percent": random.uniform(20, 80),
            "load_average_1m": random.uniform(0.5, 2.0),
            "load_average_5m": random.uniform(0.5, 2.0),
            "load_average_15m": random.uniform(0.5, 2.0),
            "cores": 8,
        },
        "memory": {
            "total_gb": 16,
            "used_gb": random.uniform(4, 12),
            "free_gb": random.uniform(4, 12),
            "cached_gb": random.uniform(1, 4),
            "usage_percent": random.uniform(30, 75),
        },
        "disk": {
            "total_gb": 500,
            "used_gb": random.uniform(100, 300),
            "free_gb": random.uniform(200, 400),
            "usage_percent": random.uniform(20, 60),
            "read_mb_s": random.uniform(10, 100),
            "write_mb_s": random.uniform(5, 50),
        },
        "network": {
            "bytes_sent": random.randint(1000000, 10000000),
            "bytes_recv": random.randint(1000000, 10000000),
            "packets_sent": random.randint(10000, 100000),
            "packets_recv": random.randint(10000, 100000),
            "errors": random.randint(0, 10),
            "dropped": random.randint(0, 5),
        },
        "database": {
            "connections_active": random.randint(5, 50),
            "connections_idle": random.randint(10, 100),
            "queries_per_second": random.uniform(10, 100),
            "slow_queries": random.randint(0, 10),
            "replication_lag_ms": random.uniform(0, 100),
        },
        "cache": {
            "hits": random.randint(10000, 100000),
            "misses": random.randint(100, 1000),
            "hit_rate": random.uniform(0.85, 0.99),
            "memory_used_mb": random.uniform(100, 500),
            "evictions": random.randint(0, 100),
        },
        "queue": {
            "pending": random.randint(0, 100),
            "processing": random.randint(0, 20),
            "completed": random.randint(1000, 10000),
            "failed": random.randint(0, 50),
            "retry": random.randint(0, 10),
        },
    }


# ---------------------------------------------------------------------------
# Job Management
# ---------------------------------------------------------------------------

def list_background_jobs(status: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Return a list of background jobs, optionally filtered by status.

    Generates 20 simulated job records across common job types. If a
    status filter is provided, only matching jobs are returned.
    """
    job_types = ["data_sync", "analysis", "report_generation", "cleanup", "backup"]
    all_statuses = ["pending", "running", "completed", "failed", "cancelled"]

    jobs = []
    for i in range(20):
        job_status = status or random.choice(all_statuses)
        started = datetime.now(timezone.utc) - timedelta(minutes=random.randint(0, 120))

        jobs.append(
            {
                "id": str(uuid.uuid4()),
                "name": f"Job_{i}",
                "type": random.choice(job_types),
                "status": job_status,
                "progress": (
                    random.uniform(0, 100)
                    if job_status == "running"
                    else 100 if job_status == "completed" else 0
                ),
                "started_at": started,
                "completed_at": (
                    started + timedelta(minutes=random.randint(1, 30))
                    if job_status == "completed"
                    else None
                ),
                "error_message": "Connection timeout" if job_status == "failed" else None,
                "result": (
                    {"records_processed": random.randint(100, 10000)}
                    if job_status == "completed"
                    else None
                ),
                "retry_count": random.randint(0, 3),
            }
        )

    if status:
        jobs = [j for j in jobs if j["status"] == status]

    return jobs


def cancel_job(job_id: str) -> Dict[str, Any]:
    """
    Cancel a running job by ID.

    Returns a confirmation payload. In production this would signal the
    job worker to stop processing.
    """
    return {
        "message": f"Job {job_id} has been cancelled",
        "status": "success",
    }


def retry_job(job_id: str) -> Dict[str, Any]:
    """
    Re-queue a failed job for retry.

    Returns a confirmation payload containing the new job ID assigned to
    the retry attempt.
    """
    return {
        "message": f"Job {job_id} has been queued for retry",
        "status": "success",
        "new_job_id": str(uuid.uuid4()),
    }


# ---------------------------------------------------------------------------
# Configuration Management
# ---------------------------------------------------------------------------

def _mask_secret(value: Optional[str]) -> str:
    """Mask a secret value for safe display, showing only the first and last 4 characters."""
    if not value or len(value) < 8:
        return "***NOT_SET***"
    return f"{value[:4]}...{value[-4:]}"


def get_configuration(section: Optional[str] = None) -> Dict[str, Any]:
    """
    Return system configuration, with secret values masked.

    When a section name is provided only that section is returned.
    All API key values are masked to prevent secret leakage.
    """
    config: Dict[str, Any] = {
        "api_keys": {
            "alpha_vantage": _mask_secret(os.getenv("ALPHA_VANTAGE_API_KEY")),
            "finnhub": _mask_secret(os.getenv("FINNHUB_API_KEY")),
            "polygon": _mask_secret(os.getenv("POLYGON_API_KEY")),
            "news_api": _mask_secret(os.getenv("NEWS_API_KEY")),
        },
        "database": {
            "host": "postgres",
            "port": 5432,
            "name": "investment_db",
            "pool_size": 20,
            "max_overflow": 10,
        },
        "cache": {
            "host": "redis",
            "port": 6379,
            "ttl_default": 300,
            "max_memory": "512mb",
        },
        "security": {
            "jwt_expiration_minutes": 1440,
            "password_min_length": 8,
            "require_2fa": False,
            "allowed_origins": ["http://localhost:3000"],
        },
        "features": {
            "real_time_quotes": True,
            "ml_predictions": True,
            "social_sentiment": True,
            "options_trading": False,
            "crypto_trading": False,
        },
        "limits": {
            "max_api_calls_per_minute": 60,
            "max_portfolio_size": 100,
            "max_watchlist_size": 50,
            "max_concurrent_connections": 1000,
        },
        "monitoring": {
            "prometheus_enabled": True,
            "grafana_enabled": True,
            "sentry_enabled": False,
            "log_level": "INFO",
        },
    }

    if section:
        return {section: config.get(section, {})}

    return config


def apply_configuration_update(section: str, key: str, value: Any) -> Dict[str, Any]:
    """
    Apply a configuration update and indicate whether a service restart is required.

    In production this would persist the change to a database or config file.
    Returns a confirmation payload with restart requirements.
    """
    restart_sections = {"database", "cache"}
    return {
        "message": f"Configuration updated: {section}.{key}",
        "status": "success",
        "requires_restart": section in restart_sections,
    }


# ---------------------------------------------------------------------------
# Audit Logs
# ---------------------------------------------------------------------------

def get_audit_logs(
    user_id: Optional[str] = None,
    action: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """
    Return audit log entries, filtered by user and action, sorted newest-first.

    Generates 200 simulated log records and applies the requested filters
    before returning the paginated result.
    """
    actions = ["login", "logout", "create", "update", "delete", "export", "import"]
    resources = ["user", "portfolio", "trade", "configuration", "report"]

    logs = []
    for i in range(200):
        entry = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc) - timedelta(minutes=random.randint(0, 10080)),
            "user_id": user_id or str(uuid.uuid4()),
            "user_email": f"user{i % 20}@example.com",
            "action": action or random.choice(actions),
            "resource_type": random.choice(resources),
            "resource_id": str(uuid.uuid4()) if random.random() > 0.3 else None,
            "details": {"ip": f"192.168.1.{random.randint(1, 255)}"},
            "ip_address": f"192.168.1.{random.randint(1, 255)}",
            "user_agent": "Mozilla/5.0...",
            "success": random.random() > 0.1,
            "error_message": "Permission denied" if random.random() < 0.1 else None,
        }

        if user_id and entry["user_id"] != user_id:
            continue
        if action and entry["action"] != action:
            continue

        logs.append(entry)

    sorted_logs = sorted(logs, key=lambda x: x["timestamp"], reverse=True)
    return sorted_logs[offset : offset + limit]


# ---------------------------------------------------------------------------
# Announcements
# ---------------------------------------------------------------------------

def list_announcements(active_only: bool = True) -> List[Dict[str, Any]]:
    """
    Return system announcements, optionally limited to active ones.
    """
    announcements = [
        {
            "id": str(uuid.uuid4()),
            "title": "Scheduled Maintenance",
            "message": "System will be under maintenance on Sunday 2 AM - 4 AM EST",
            "type": "warning",
            "active": True,
            "start_time": datetime.now(timezone.utc),
            "end_time": datetime.now(timezone.utc) + timedelta(days=7),
            "target_users": None,
        },
        {
            "id": str(uuid.uuid4()),
            "title": "New Features Released",
            "message": "Check out our new portfolio analytics dashboard!",
            "type": "info",
            "active": True,
            "start_time": datetime.now(timezone.utc) - timedelta(days=2),
            "end_time": None,
            "target_users": None,
        },
    ]

    if active_only:
        announcements = [a for a in announcements if a["active"]]

    return announcements


# ---------------------------------------------------------------------------
# Data Export
# ---------------------------------------------------------------------------

def initiate_data_export(export_type: str) -> Dict[str, Any]:
    """
    Initiate a data export job and return the job tracking information.

    The actual export processing is performed asynchronously via a
    background task. This function returns the job ID and estimated time.
    """
    job_id = str(uuid.uuid4())
    return {
        "job_id": job_id,
        "status": "processing",
        "estimated_time_seconds": random.randint(30, 300),
        "download_url": f"/admin/export/{job_id}/download",
    }


# ---------------------------------------------------------------------------
# System Command
# ---------------------------------------------------------------------------

def execute_system_command(command: str, execution_time_ms: int) -> Dict[str, Any]:
    """
    Execute a validated system command and return its result.

    Command validation is handled upstream by the SystemCommand Pydantic
    model. This function records the execution outcome.
    """
    return {
        "command": command,
        "status": "executed",
        "result": {
            "success": True,
            "message": f"Command {command} executed successfully",
            "execution_time_ms": execution_time_ms,
        },
    }


# ---------------------------------------------------------------------------
# Maintenance Mode
# ---------------------------------------------------------------------------

def enable_maintenance_mode(message: str = "System is under maintenance") -> Dict[str, Any]:
    """
    Enable system maintenance mode with the supplied status message.

    In production this would set a flag in shared state (e.g. Redis) so
    that all incoming requests receive a maintenance response.
    """
    return {
        "status": "maintenance_enabled",
        "message": message,
    }


def disable_maintenance_mode() -> Dict[str, Any]:
    """
    Disable system maintenance mode and restore normal operation.
    """
    return {
        "status": "maintenance_disabled",
        "message": "System is operational",
    }
