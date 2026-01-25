"""
Comprehensive Audit Logging System
Provides SEC and GDPR compliant audit logging with tamper protection
"""

import os
import json
import hashlib
import hmac
import uuid
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta, timezone
from enum import Enum
from dataclasses import dataclass, asdict, field
from pathlib import Path
import asyncio
import aiofiles
import logging
from concurrent.futures import ThreadPoolExecutor
import gzip
import shutil

# Database imports
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import Column, String, DateTime, Integer, Text, Boolean, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import DeclarativeBase

# FastAPI imports
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# Cryptography for tamper protection
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64

# Redis for caching
import redis.asyncio as aioredis

from .secrets_vault import get_secrets_vault

logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    """Types of audit events"""
    # Authentication & Authorization
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    PASSWORD_CHANGE = "password_change"
    ACCOUNT_LOCKED = "account_locked"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_DENIED = "permission_denied"
    
    # Data Operations
    DATA_ACCESS = "data_access"
    DATA_CREATE = "data_create"
    DATA_UPDATE = "data_update"
    DATA_DELETE = "data_delete"
    DATA_EXPORT = "data_export"
    DATA_IMPORT = "data_import"
    
    # Financial Operations
    PORTFOLIO_CREATE = "portfolio_create"
    PORTFOLIO_UPDATE = "portfolio_update"
    PORTFOLIO_DELETE = "portfolio_delete"
    TRADE_EXECUTE = "trade_execute"
    RECOMMENDATION_GENERATE = "recommendation_generate"
    ANALYSIS_RUN = "analysis_run"
    
    # Security Events
    SECURITY_VIOLATION = "security_violation"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    CSRF_ATTACK = "csrf_attack"
    SQL_INJECTION_ATTEMPT = "sql_injection_attempt"
    XSS_ATTEMPT = "xss_attempt"
    
    # System Events
    SYSTEM_START = "system_start"
    SYSTEM_SHUTDOWN = "system_shutdown"
    CONFIG_CHANGE = "config_change"
    BACKUP_CREATE = "backup_create"
    BACKUP_RESTORE = "backup_restore"
    
    # Compliance Events
    GDPR_DATA_REQUEST = "gdpr_data_request"
    GDPR_DATA_DELETION = "gdpr_data_deletion"
    GDPR_CONSENT_CHANGE = "gdpr_consent_change"
    SEC_REPORT_GENERATION = "sec_report_generation"
    AUDIT_LOG_ACCESS = "audit_log_access"


class AuditSeverity(str, Enum):
    """Severity levels for audit events"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceFramework(str, Enum):
    """Compliance frameworks"""
    SEC = "sec"
    GDPR = "gdpr"
    SOX = "sox"
    HIPAA = "hipaa"
    ISO27001 = "iso27001"


@dataclass
class AuditEvent:
    """Audit event data structure"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: AuditEventType = AuditEventType.DATA_ACCESS
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # User context
    user_id: Optional[str] = None
    username: Optional[str] = None
    session_id: Optional[str] = None
    
    # Request context
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    
    # Event details
    resource: Optional[str] = None
    action: Optional[str] = None
    result: Optional[str] = None  # success, failure, error
    severity: AuditSeverity = AuditSeverity.MEDIUM
    
    # Additional data
    details: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    # Compliance
    compliance_frameworks: List[ComplianceFramework] = field(default_factory=list)
    
    # Tamper protection
    checksum: Optional[str] = None
    signature: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEvent':
        """Create from dictionary"""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['event_type'] = AuditEventType(data['event_type'])
        data['severity'] = AuditSeverity(data['severity'])
        data['compliance_frameworks'] = [ComplianceFramework(f) for f in data.get('compliance_frameworks', [])]
        return cls(**data)


class AuditEventModel:
    """SQLAlchemy model for audit events (for database storage)"""
    __tablename__ = 'audit_events'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_id = Column(String(36), unique=True, nullable=False, index=True)
    event_type = Column(String(50), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    
    # User context
    user_id = Column(String(36), index=True)
    username = Column(String(255), index=True)
    session_id = Column(String(255), index=True)
    
    # Request context
    ip_address = Column(String(45), index=True)
    user_agent = Column(Text)
    endpoint = Column(String(500), index=True)
    method = Column(String(10))
    
    # Event details
    resource = Column(String(500), index=True)
    action = Column(String(100), index=True)
    result = Column(String(50), index=True)
    severity = Column(String(20), index=True)
    
    # Additional data
    details = Column(JSONB)
    tags = Column(JSONB)
    
    # Compliance
    compliance_frameworks = Column(JSONB)
    
    # Tamper protection
    checksum = Column(String(64))
    signature = Column(String(256))
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_audit_timestamp_severity', 'timestamp', 'severity'),
        Index('idx_audit_user_timestamp', 'user_id', 'timestamp'),
        Index('idx_audit_event_type_timestamp', 'event_type', 'timestamp'),
    )


class TamperProtection:
    """Tamper protection for audit logs"""

    def __init__(self):
        # Use a synchronous approach for initial key generation
        # The async version can be used for key rotation
        self.secret_key = self._get_signing_key_sync()
        self._initialized = False

    def _get_signing_key_sync(self) -> bytes:
        """Get or create signing key synchronously for tamper protection"""
        # Use environment variable or generate a key
        # In production, this should be loaded from a secure vault
        key_b64 = os.getenv("AUDIT_SIGNING_KEY")
        if key_b64:
            return base64.b64decode(key_b64)

        # Generate new signing key and cache it
        key = os.urandom(32)
        return key

    async def _get_signing_key(self) -> bytes:
        """Get or create signing key for tamper protection (async version)"""
        try:
            vault = get_secrets_vault()

            key_b64 = await vault.get_secret("audit_signing_key")
            if key_b64:
                return base64.b64decode(key_b64)

            # Generate new signing key
            key = os.urandom(32)
            key_b64 = base64.b64encode(key).decode()

            await vault.store_secret(
                "audit_signing_key",
                key_b64,
                secret_type="encryption_key",
                rotation_policy="quarterly"
            )

            return key
        except Exception:
            # Fallback to synchronous key
            return self._get_signing_key_sync()
    
    def calculate_checksum(self, event: AuditEvent) -> str:
        """Calculate checksum for event"""
        # Create canonical representation
        event_data = event.to_dict()
        # Remove tamper protection fields from checksum calculation
        event_data.pop('checksum', None)
        event_data.pop('signature', None)
        
        canonical = json.dumps(event_data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical.encode()).hexdigest()
    
    def sign_event(self, event: AuditEvent) -> str:
        """Create HMAC signature for event"""
        checksum = self.calculate_checksum(event)
        signature = hmac.new(
            self.secret_key,
            checksum.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def verify_event(self, event: AuditEvent) -> bool:
        """Verify event integrity"""
        if not event.checksum or not event.signature:
            return False
        
        expected_checksum = self.calculate_checksum(event)
        if event.checksum != expected_checksum:
            return False
        
        expected_signature = hmac.new(
            self.secret_key,
            expected_checksum.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(event.signature, expected_signature)


class AuditLogStorage:
    """Storage backend for audit logs"""
    
    def __init__(self, storage_path: str = None, redis_url: str = None):
        # Use a user-writable directory by default
        default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs", "audit")
        self.storage_path = Path(storage_path or os.getenv("AUDIT_LOG_PATH", default_path))
        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            # Fallback to temp directory if default fails
            import tempfile
            self.storage_path = Path(tempfile.gettempdir()) / "investment_platform_audit"
            self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.redis_url = redis_url or os.getenv("REDIS_URL")
        self._redis: Optional[aioredis.Redis] = None
        
        self.tamper_protection = TamperProtection()
        self.executor = ThreadPoolExecutor(max_workers=2)
    
    async def get_redis(self) -> Optional[aioredis.Redis]:
        """Get Redis connection for caching"""
        if not self.redis_url:
            return None
        
        if not self._redis:
            self._redis = aioredis.from_url(self.redis_url)
        return self._redis
    
    async def store_event(self, event: AuditEvent):
        """Store audit event with tamper protection"""
        try:
            # Add tamper protection
            event.checksum = self.tamper_protection.calculate_checksum(event)
            event.signature = self.tamper_protection.sign_event(event)
            
            # Store to multiple backends
            await asyncio.gather(
                self._store_to_file(event),
                self._store_to_redis(event),
                return_exceptions=True
            )
            
        except Exception as e:
            logger.error(f"Failed to store audit event {event.event_id}: {e}")
            # Ensure critical events are always logged somewhere
            await self._emergency_log(event, str(e))
    
    async def _store_to_file(self, event: AuditEvent):
        """Store event to file system"""
        # Create daily log files
        date_str = event.timestamp.strftime("%Y-%m-%d")
        log_file = self.storage_path / f"audit-{date_str}.jsonl"
        
        event_json = json.dumps(event.to_dict(), separators=(',', ':'))
        
        async with aiofiles.open(log_file, "a") as f:
            await f.write(event_json + "\n")
    
    async def _store_to_redis(self, event: AuditEvent):
        """Store event to Redis for fast access"""
        redis = await self.get_redis()
        if not redis:
            return
        
        # Store in sorted set by timestamp for efficient querying
        timestamp_score = event.timestamp.timestamp()
        
        pipe = redis.pipeline()
        
        # Main audit log
        pipe.zadd("audit_events", {event.event_id: timestamp_score})
        pipe.setex(f"audit_event:{event.event_id}", 86400 * 7, json.dumps(event.to_dict()))  # 7 days
        
        # Index by user
        if event.user_id:
            pipe.zadd(f"audit_events:user:{event.user_id}", {event.event_id: timestamp_score})
        
        # Index by event type
        pipe.zadd(f"audit_events:type:{event.event_type.value}", {event.event_id: timestamp_score})
        
        # Index by severity
        pipe.zadd(f"audit_events:severity:{event.severity.value}", {event.event_id: timestamp_score})
        
        # Execute pipeline
        await pipe.execute()
    
    async def _emergency_log(self, event: AuditEvent, error: str):
        """Emergency logging when primary storage fails"""
        emergency_file = self.storage_path / "emergency.log"
        emergency_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": f"Failed to store audit event: {error}",
            "event": event.to_dict()
        }
        
        try:
            async with aiofiles.open(emergency_file, "a") as f:
                await f.write(json.dumps(emergency_entry) + "\n")
        except Exception as e:
            # Last resort: standard logging
            logger.critical(f"Emergency audit log failed: {e}, Event: {event.to_dict()}")
    
    async def query_events(
        self,
        start_time: datetime = None,
        end_time: datetime = None,
        user_id: str = None,
        event_type: AuditEventType = None,
        severity: AuditSeverity = None,
        limit: int = 100
    ) -> List[AuditEvent]:
        """Query audit events with filters"""
        
        # Try Redis first for recent events
        redis_results = await self._query_redis(
            start_time, end_time, user_id, event_type, severity, limit
        )
        
        if redis_results and len(redis_results) >= limit:
            return redis_results[:limit]
        
        # Fall back to file system for historical data
        file_results = await self._query_files(
            start_time, end_time, user_id, event_type, severity, limit - len(redis_results)
        )
        
        # Combine and sort results
        all_results = redis_results + file_results
        all_results.sort(key=lambda x: x.timestamp, reverse=True)
        
        return all_results[:limit]
    
    async def _query_redis(
        self,
        start_time: datetime = None,
        end_time: datetime = None,
        user_id: str = None,
        event_type: AuditEventType = None,
        severity: AuditSeverity = None,
        limit: int = 100
    ) -> List[AuditEvent]:
        """Query events from Redis"""
        redis = await self.get_redis()
        if not redis:
            return []
        
        try:
            # Determine which index to use
            if user_id:
                index_key = f"audit_events:user:{user_id}"
            elif event_type:
                index_key = f"audit_events:type:{event_type.value}"
            elif severity:
                index_key = f"audit_events:severity:{severity.value}"
            else:
                index_key = "audit_events"
            
            # Calculate score range
            min_score = start_time.timestamp() if start_time else "-inf"
            max_score = end_time.timestamp() if end_time else "+inf"
            
            # Get event IDs
            event_ids = await redis.zrevrangebyscore(
                index_key, max_score, min_score, start=0, num=limit
            )
            
            if not event_ids:
                return []
            
            # Get event data
            pipe = redis.pipeline()
            for event_id in event_ids:
                pipe.get(f"audit_event:{event_id.decode()}")
            
            results = await pipe.execute()
            
            events = []
            for result in results:
                if result:
                    try:
                        event_data = json.loads(result)
                        event = AuditEvent.from_dict(event_data)
                        events.append(event)
                    except Exception as e:
                        logger.error(f"Failed to parse event from Redis: {e}")
            
            return events
            
        except Exception as e:
            logger.error(f"Redis query failed: {e}")
            return []
    
    async def _query_files(
        self,
        start_time: datetime = None,
        end_time: datetime = None,
        user_id: str = None,
        event_type: AuditEventType = None,
        severity: AuditSeverity = None,
        limit: int = 100
    ) -> List[AuditEvent]:
        """Query events from file system"""
        events = []
        
        # Determine date range for file scanning
        if not start_time:
            start_time = datetime.now(timezone.utc) - timedelta(days=30)
        if not end_time:
            end_time = datetime.now(timezone.utc)
        
        current_date = start_time.date()
        end_date = end_time.date()
        
        while current_date <= end_date and len(events) < limit:
            date_str = current_date.strftime("%Y-%m-%d")
            log_file = self.storage_path / f"audit-{date_str}.jsonl"
            
            if log_file.exists():
                file_events = await self._scan_log_file(
                    log_file, start_time, end_time, user_id, event_type, severity, limit - len(events)
                )
                events.extend(file_events)
            
            current_date += timedelta(days=1)
        
        return sorted(events, key=lambda x: x.timestamp, reverse=True)
    
    async def _scan_log_file(
        self,
        log_file: Path,
        start_time: datetime,
        end_time: datetime,
        user_id: str,
        event_type: AuditEventType,
        severity: AuditSeverity,
        limit: int
    ) -> List[AuditEvent]:
        """Scan individual log file for matching events"""
        events = []
        
        try:
            async with aiofiles.open(log_file, "r") as f:
                async for line in f:
                    if len(events) >= limit:
                        break
                    
                    try:
                        event_data = json.loads(line.strip())
                        event = AuditEvent.from_dict(event_data)
                        
                        # Apply filters
                        if start_time and event.timestamp < start_time:
                            continue
                        if end_time and event.timestamp > end_time:
                            continue
                        if user_id and event.user_id != user_id:
                            continue
                        if event_type and event.event_type != event_type:
                            continue
                        if severity and event.severity != severity:
                            continue
                        
                        events.append(event)
                        
                    except Exception as e:
                        logger.error(f"Failed to parse log line: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Failed to scan log file {log_file}: {e}")
        
        return events
    
    async def archive_old_logs(self, days_to_keep: int = 2555):  # 7 years for SEC compliance
        """Archive old audit logs"""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
        
        for log_file in self.storage_path.glob("audit-*.jsonl"):
            try:
                # Extract date from filename
                date_part = log_file.stem.replace("audit-", "")
                file_date = datetime.strptime(date_part, "%Y-%m-%d").date()
                
                if datetime.combine(file_date, datetime.min.time()) < cutoff_date.replace(tzinfo=None):
                    # Compress and move to archive
                    archive_dir = self.storage_path / "archive"
                    archive_dir.mkdir(exist_ok=True)
                    
                    compressed_file = archive_dir / f"{log_file.stem}.gz"
                    
                    with open(log_file, "rb") as f_in:
                        with gzip.open(compressed_file, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    log_file.unlink()
                    logger.info(f"Archived audit log: {log_file} -> {compressed_file}")
                    
            except Exception as e:
                logger.error(f"Failed to archive log file {log_file}: {e}")


class AuditLogger:
    """Main audit logging system"""
    
    def __init__(self, storage: AuditLogStorage = None):
        self.storage = storage or AuditLogStorage()
        
        # Rate limiting to prevent log flooding
        self._rate_limits = {}
        self._rate_limit_window = 60  # seconds
        self._rate_limit_max = 100    # events per window
    
    async def log_event(
        self,
        event_type: AuditEventType,
        user_id: str = None,
        username: str = None,
        session_id: str = None,
        ip_address: str = None,
        user_agent: str = None,
        endpoint: str = None,
        method: str = None,
        resource: str = None,
        action: str = None,
        result: str = "success",
        severity: AuditSeverity = AuditSeverity.MEDIUM,
        details: Dict[str, Any] = None,
        tags: List[str] = None,
        compliance_frameworks: List[ComplianceFramework] = None
    ):
        """Log an audit event"""
        
        # Rate limiting check
        if not self._check_rate_limit(user_id or ip_address or "anonymous"):
            logger.warning(f"Rate limit exceeded for audit logging: {user_id or ip_address}")
            return
        
        event = AuditEvent(
            event_type=event_type,
            user_id=user_id,
            username=username,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            endpoint=endpoint,
            method=method,
            resource=resource,
            action=action,
            result=result,
            severity=severity,
            details=details or {},
            tags=tags or [],
            compliance_frameworks=compliance_frameworks or []
        )
        
        await self.storage.store_event(event)
    
    def _check_rate_limit(self, identifier: str) -> bool:
        """Check rate limiting for audit events"""
        now = datetime.now()
        
        if identifier not in self._rate_limits:
            self._rate_limits[identifier] = []
        
        # Clean old entries
        cutoff = now - timedelta(seconds=self._rate_limit_window)
        self._rate_limits[identifier] = [
            timestamp for timestamp in self._rate_limits[identifier] 
            if timestamp > cutoff
        ]
        
        # Check limit
        if len(self._rate_limits[identifier]) >= self._rate_limit_max:
            return False
        
        # Add current event
        self._rate_limits[identifier].append(now)
        return True
    
    async def log_login_success(self, user_id: str, username: str, ip_address: str, user_agent: str):
        """Log successful login"""
        await self.log_event(
            AuditEventType.LOGIN_SUCCESS,
            user_id=user_id,
            username=username,
            ip_address=ip_address,
            user_agent=user_agent,
            action="login",
            result="success",
            severity=AuditSeverity.MEDIUM,
            compliance_frameworks=[ComplianceFramework.SEC, ComplianceFramework.GDPR]
        )
    
    async def log_login_failure(self, username: str, ip_address: str, user_agent: str, reason: str):
        """Log failed login attempt"""
        await self.log_event(
            AuditEventType.LOGIN_FAILURE,
            username=username,
            ip_address=ip_address,
            user_agent=user_agent,
            action="login",
            result="failure",
            severity=AuditSeverity.HIGH,
            details={"failure_reason": reason},
            compliance_frameworks=[ComplianceFramework.SEC]
        )
    
    async def log_data_access(self, user_id: str, resource: str, details: Dict[str, Any] = None):
        """Log data access event"""
        await self.log_event(
            AuditEventType.DATA_ACCESS,
            user_id=user_id,
            resource=resource,
            action="read",
            result="success",
            severity=AuditSeverity.LOW,
            details=details,
            compliance_frameworks=[ComplianceFramework.SEC, ComplianceFramework.GDPR]
        )
    
    async def log_security_violation(
        self, 
        violation_type: str, 
        ip_address: str, 
        user_agent: str,
        details: Dict[str, Any] = None
    ):
        """Log security violation"""
        await self.log_event(
            AuditEventType.SECURITY_VIOLATION,
            ip_address=ip_address,
            user_agent=user_agent,
            action=violation_type,
            result="blocked",
            severity=AuditSeverity.CRITICAL,
            details=details,
            tags=["security", "threat"],
            compliance_frameworks=[ComplianceFramework.SEC, ComplianceFramework.ISO27001]
        )
    
    async def log_portfolio_operation(
        self,
        user_id: str,
        operation: str,
        portfolio_id: str,
        details: Dict[str, Any] = None
    ):
        """Log portfolio operations for SEC compliance"""
        await self.log_event(
            AuditEventType.PORTFOLIO_UPDATE,
            user_id=user_id,
            resource=f"portfolio:{portfolio_id}",
            action=operation,
            result="success",
            severity=AuditSeverity.MEDIUM,
            details=details,
            compliance_frameworks=[ComplianceFramework.SEC]
        )
    
    async def log_gdpr_request(
        self,
        request_type: str,
        user_id: str,
        details: Dict[str, Any] = None
    ):
        """Log GDPR data requests"""
        event_type_map = {
            "data_request": AuditEventType.GDPR_DATA_REQUEST,
            "data_deletion": AuditEventType.GDPR_DATA_DELETION,
            "consent_change": AuditEventType.GDPR_CONSENT_CHANGE
        }
        
        await self.log_event(
            event_type_map.get(request_type, AuditEventType.GDPR_DATA_REQUEST),
            user_id=user_id,
            action=request_type,
            result="success",
            severity=AuditSeverity.HIGH,
            details=details,
            compliance_frameworks=[ComplianceFramework.GDPR],
            tags=["gdpr", "privacy"]
        )
    
    async def query_events(self, **kwargs) -> List[AuditEvent]:
        """Query audit events"""
        return await self.storage.query_events(**kwargs)


class AuditMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for automatic audit logging"""
    
    def __init__(self, app, audit_logger: AuditLogger = None):
        super().__init__(app)
        self.audit_logger = audit_logger or AuditLogger()
        
        # Paths to exclude from audit logging
        self.excluded_paths = [
            "/api/health",
            "/api/metrics", 
            "/static",
            "/favicon.ico"
        ]
    
    async def dispatch(self, request: Request, call_next):
        """Log requests automatically"""
        
        # Skip excluded paths
        if any(request.url.path.startswith(path) for path in self.excluded_paths):
            return await call_next(request)
        
        start_time = datetime.now(timezone.utc)
        
        try:
            response = await call_next(request)
            
            # Log successful requests
            await self._log_request(request, response, start_time)
            
            return response
            
        except Exception as e:
            # Log failed requests
            await self._log_request_error(request, e, start_time)
            raise
    
    async def _log_request(self, request: Request, response: Response, start_time: datetime):
        """Log successful request"""
        
        # Extract user context from JWT or session
        user_id = getattr(request.state, "user_id", None)
        username = getattr(request.state, "username", None)
        session_id = getattr(request.state, "session_id", None)
        
        # Determine severity based on endpoint and method
        severity = AuditSeverity.LOW
        if request.method in ["POST", "PUT", "DELETE", "PATCH"]:
            severity = AuditSeverity.MEDIUM
        
        if request.url.path.startswith("/api/admin/"):
            severity = AuditSeverity.HIGH
        
        # Determine event type based on endpoint
        event_type = AuditEventType.DATA_ACCESS
        if request.method == "POST":
            event_type = AuditEventType.DATA_CREATE
        elif request.method in ["PUT", "PATCH"]:
            event_type = AuditEventType.DATA_UPDATE
        elif request.method == "DELETE":
            event_type = AuditEventType.DATA_DELETE
        
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        await self.audit_logger.log_event(
            event_type=event_type,
            user_id=user_id,
            username=username,
            session_id=session_id,
            ip_address=request.client.host,
            user_agent=request.headers.get("User-Agent"),
            endpoint=request.url.path,
            method=request.method,
            result="success",
            severity=severity,
            details={
                "status_code": response.status_code,
                "processing_time_seconds": processing_time
            }
        )
    
    async def _log_request_error(self, request: Request, error: Exception, start_time: datetime):
        """Log request that resulted in error"""
        
        user_id = getattr(request.state, "user_id", None)
        username = getattr(request.state, "username", None)
        session_id = getattr(request.state, "session_id", None)
        
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        await self.audit_logger.log_event(
            event_type=AuditEventType.DATA_ACCESS,
            user_id=user_id,
            username=username,
            session_id=session_id,
            ip_address=request.client.host,
            user_agent=request.headers.get("User-Agent"),
            endpoint=request.url.path,
            method=request.method,
            result="error",
            severity=AuditSeverity.HIGH,
            details={
                "error_type": type(error).__name__,
                "error_message": str(error),
                "processing_time_seconds": processing_time
            }
        )


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get global audit logger instance"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


# Utility functions for common audit events
async def audit_login_success(user_id: str, username: str, ip_address: str, user_agent: str):
    """Utility function to audit successful login"""
    logger = get_audit_logger()
    await logger.log_login_success(user_id, username, ip_address, user_agent)


async def audit_login_failure(username: str, ip_address: str, user_agent: str, reason: str):
    """Utility function to audit failed login"""
    logger = get_audit_logger()
    await logger.log_login_failure(username, ip_address, user_agent, reason)


async def audit_data_access(user_id: str, resource: str, details: Dict[str, Any] = None):
    """Utility function to audit data access"""
    logger = get_audit_logger()
    await logger.log_data_access(user_id, resource, details)


async def audit_security_violation(violation_type: str, ip_address: str, user_agent: str, details: Dict[str, Any] = None):
    """Utility function to audit security violations"""
    logger = get_audit_logger()
    await logger.log_security_violation(violation_type, ip_address, user_agent, details)