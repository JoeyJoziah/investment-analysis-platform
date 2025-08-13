"""Audit logging for SEC compliance"""

import asyncio
import json
import logging
import traceback
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from sqlalchemy.orm import Session

from backend.models.unified_models import AuditLog
from backend.streaming.kafka_client import kafka_processor
from backend.utils.data_anonymization import data_anonymizer
from backend.utils.monitoring import metrics

logger = logging.getLogger(__name__)


class AuditAction(str, Enum):
    """Audit action types for SEC compliance"""
    # Authentication
    LOGIN = "user.login"
    LOGOUT = "user.logout"
    LOGIN_FAILED = "user.login_failed"
    PASSWORD_CHANGE = "user.password_change"
    
    # Data Access
    DATA_VIEW = "data.view"
    DATA_EXPORT = "data.export"
    DATA_MODIFY = "data.modify"
    DATA_DELETE = "data.delete"
    
    # Trading/Recommendations
    RECOMMENDATION_GENERATED = "recommendation.generated"
    RECOMMENDATION_VIEWED = "recommendation.viewed"
    RECOMMENDATION_ACTED = "recommendation.acted"
    TRADE_EXECUTED = "trade.executed"
    
    # Analysis
    ANALYSIS_STARTED = "analysis.started"
    ANALYSIS_COMPLETED = "analysis.completed"
    ANALYSIS_FAILED = "analysis.failed"
    MODEL_PREDICTION = "model.prediction"
    
    # System
    CONFIG_CHANGE = "system.config_change"
    API_KEY_CREATED = "system.api_key_created"
    API_KEY_REVOKED = "system.api_key_revoked"
    PERMISSION_CHANGE = "system.permission_change"
    
    # Compliance
    CONSENT_GIVEN = "compliance.consent_given"
    CONSENT_WITHDRAWN = "compliance.consent_withdrawn"
    DATA_REQUEST = "compliance.data_request"
    DATA_DELETION = "compliance.data_deletion"
    
    # Security
    SECURITY_ALERT = "security.alert"
    SUSPICIOUS_ACTIVITY = "security.suspicious_activity"
    ACCESS_DENIED = "security.access_denied"
    RATE_LIMIT_EXCEEDED = "security.rate_limit_exceeded"


class AuditLogger:
    """Comprehensive audit logging for SEC compliance"""
    
    def __init__(self):
        self.enabled = True
        self._local_buffer = []
        self._buffer_size = 100
        
    async def log(
        self,
        action: Union[AuditAction, str],
        user_id: Optional[int] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        session_id: Optional[str] = None,
        severity: str = "info",
        db: Optional[Session] = None
    ) -> Optional[str]:
        """
        Create audit log entry
        
        Args:
            action: Action performed
            user_id: User who performed action
            resource_type: Type of resource accessed
            resource_id: ID of resource accessed
            details: Additional details
            ip_address: Client IP address
            user_agent: Client user agent
            session_id: Session identifier
            severity: Log severity (info, warning, error, critical)
            db: Database session
            
        Returns:
            Audit log ID
        """
        if not self.enabled:
            return None
            
        try:
            # Create audit entry
            audit_entry = {
                "id": self._generate_audit_id(),
                "timestamp": datetime.utcnow(),
                "action": action.value if isinstance(action, AuditAction) else action,
                "user_id": user_id,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "details": details or {},
                "ip_address": data_anonymizer.anonymize_ip(ip_address) if ip_address else None,
                "user_agent": user_agent,
                "session_id": session_id,
                "severity": severity
            }
            
            # Add system context
            audit_entry["details"]["system_context"] = {
                "environment": "production",  # Should come from settings
                "version": "1.0.0",  # Should come from settings
                "server_time": datetime.utcnow().isoformat()
            }
            
            # Log to database if session provided
            if db:
                await self._log_to_database(audit_entry, db)
                
            # Send to Kafka for real-time processing
            await self._send_to_kafka(audit_entry)
            
            # Buffer locally for batch processing
            self._buffer_audit_entry(audit_entry)
            
            # Track metrics
            metrics.audit_logs_created.labels(
                action=audit_entry["action"],
                user_type="user" if user_id else "system"
            ).inc()
            
            # Log critical actions
            if severity in ["error", "critical"]:
                logger.error(f"Audit log: {action} - {details}")
                
            return audit_entry["id"]
            
        except Exception as e:
            logger.error(f"Failed to create audit log: {e}")
            logger.error(traceback.format_exc())
            return None
            
    async def _log_to_database(
        self,
        audit_entry: Dict[str, Any],
        db: Session
    ):
        """Save audit log to database"""
        try:
            audit_log = AuditLog(
                action=audit_entry["action"],
                user_id=audit_entry["user_id"],
                resource_type=audit_entry["resource_type"],
                resource_id=audit_entry["resource_id"],
                details=json.dumps(audit_entry["details"]),
                ip_address=audit_entry["ip_address"],
                user_agent=audit_entry["user_agent"],
                session_id=audit_entry["session_id"],
                created_at=audit_entry["timestamp"]
            )
            
            db.add(audit_log)
            db.commit()
            
        except Exception as e:
            logger.error(f"Failed to save audit log to database: {e}")
            db.rollback()
            
    async def _send_to_kafka(self, audit_entry: Dict[str, Any]):
        """Send audit log to Kafka for real-time processing"""
        try:
            await kafka_processor.audit_log(
                action=audit_entry["action"],
                user_id=audit_entry["user_id"],
                details=audit_entry
            )
        except Exception as e:
            logger.error(f"Failed to send audit log to Kafka: {e}")
            
    def _buffer_audit_entry(self, audit_entry: Dict[str, Any]):
        """Buffer audit entry for batch processing"""
        self._local_buffer.append(audit_entry)
        
        # Flush buffer if full
        if len(self._local_buffer) >= self._buffer_size:
            self._flush_buffer()
            
    def _flush_buffer(self):
        """Flush audit buffer to persistent storage"""
        if not self._local_buffer:
            return
            
        try:
            # In production, this would write to a secure audit trail
            # For now, just clear the buffer
            logger.info(f"Flushing {len(self._local_buffer)} audit entries")
            self._local_buffer.clear()
            
        except Exception as e:
            logger.error(f"Failed to flush audit buffer: {e}")
            
    def _generate_audit_id(self) -> str:
        """Generate unique audit log ID"""
        import uuid
        return str(uuid.uuid4())
        
    async def log_login(
        self,
        user_id: int,
        success: bool,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log login attempt"""
        action = AuditAction.LOGIN if success else AuditAction.LOGIN_FAILED
        await self.log(
            action=action,
            user_id=user_id if success else None,
            details={
                **(details or {}),
                "success": success,
                "login_method": details.get("login_method", "password") if details else "password"
            },
            ip_address=ip_address,
            user_agent=user_agent,
            severity="info" if success else "warning"
        )
        
    async def log_data_access(
        self,
        user_id: int,
        resource_type: str,
        resource_id: str,
        action: str = "view",
        details: Optional[Dict[str, Any]] = None
    ):
        """Log data access for compliance"""
        action_map = {
            "view": AuditAction.DATA_VIEW,
            "export": AuditAction.DATA_EXPORT,
            "modify": AuditAction.DATA_MODIFY,
            "delete": AuditAction.DATA_DELETE
        }
        
        await self.log(
            action=action_map.get(action, AuditAction.DATA_VIEW),
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details
        )
        
    async def log_recommendation(
        self,
        action: str,
        ticker: str,
        recommendation_id: str,
        user_id: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log recommendation-related actions"""
        action_map = {
            "generated": AuditAction.RECOMMENDATION_GENERATED,
            "viewed": AuditAction.RECOMMENDATION_VIEWED,
            "acted": AuditAction.RECOMMENDATION_ACTED
        }
        
        await self.log(
            action=action_map.get(action, AuditAction.RECOMMENDATION_VIEWED),
            user_id=user_id,
            resource_type="recommendation",
            resource_id=recommendation_id,
            details={
                **(details or {}),
                "ticker": ticker
            }
        )
        
    async def log_analysis(
        self,
        analysis_type: str,
        status: str,
        ticker: Optional[str] = None,
        duration: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log analysis operations"""
        action_map = {
            "started": AuditAction.ANALYSIS_STARTED,
            "completed": AuditAction.ANALYSIS_COMPLETED,
            "failed": AuditAction.ANALYSIS_FAILED
        }
        
        await self.log(
            action=action_map.get(status, AuditAction.ANALYSIS_STARTED),
            resource_type="analysis",
            resource_id=analysis_type,
            details={
                **(details or {}),
                "analysis_type": analysis_type,
                "ticker": ticker,
                "duration_seconds": duration
            },
            severity="error" if status == "failed" else "info"
        )
        
    async def log_security_event(
        self,
        event_type: str,
        user_id: Optional[int] = None,
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log security-related events"""
        action_map = {
            "alert": AuditAction.SECURITY_ALERT,
            "suspicious": AuditAction.SUSPICIOUS_ACTIVITY,
            "denied": AuditAction.ACCESS_DENIED,
            "rate_limit": AuditAction.RATE_LIMIT_EXCEEDED
        }
        
        await self.log(
            action=action_map.get(event_type, AuditAction.SECURITY_ALERT),
            user_id=user_id,
            details={
                **(details or {}),
                "event_type": event_type
            },
            ip_address=ip_address,
            severity="warning" if event_type in ["suspicious", "denied"] else "info"
        )
        
    async def log_compliance_action(
        self,
        action_type: str,
        user_id: int,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log compliance-related actions"""
        action_map = {
            "consent_given": AuditAction.CONSENT_GIVEN,
            "consent_withdrawn": AuditAction.CONSENT_WITHDRAWN,
            "data_request": AuditAction.DATA_REQUEST,
            "data_deletion": AuditAction.DATA_DELETION
        }
        
        await self.log(
            action=action_map.get(action_type, AuditAction.DATA_REQUEST),
            user_id=user_id,
            resource_type="compliance",
            details={
                **(details or {}),
                "compliance_action": action_type
            }
        )
        
    def search_audit_logs(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        user_id: Optional[int] = None,
        action: Optional[str] = None,
        resource_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search audit logs (for compliance reporting)
        
        Args:
            start_date: Start date filter
            end_date: End date filter
            user_id: User ID filter
            action: Action filter
            resource_type: Resource type filter
            limit: Maximum results
            
        Returns:
            List of audit log entries
        """
        # In production, this would query the audit database
        # For now, return empty list
        return []
        
    def generate_compliance_report(
        self,
        report_type: str,
        start_date: datetime,
        end_date: datetime,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate compliance report
        
        Args:
            report_type: Type of report (sec, gdpr, etc.)
            start_date: Report start date
            end_date: Report end date
            filters: Additional filters
            
        Returns:
            Compliance report data
        """
        report = {
            "report_type": report_type,
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "generated_at": datetime.utcnow().isoformat(),
            "summary": {},
            "details": []
        }
        
        # Add report-specific data
        if report_type == "sec":
            report["summary"] = {
                "total_trades": 0,
                "total_recommendations": 0,
                "system_changes": 0,
                "security_incidents": 0
            }
        elif report_type == "gdpr":
            report["summary"] = {
                "data_requests": 0,
                "deletions": 0,
                "consent_changes": 0,
                "breaches": 0
            }
            
        return report


# Global audit logger instance
audit_logger = AuditLogger()


# Decorator for automatic audit logging
def audit_action(
    action: AuditAction,
    resource_type: Optional[str] = None,
    extract_resource_id: Optional[Callable] = None
):
    """Decorator to automatically audit function calls"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            # Extract context
            user_id = kwargs.get("current_user").id if "current_user" in kwargs else None
            resource_id = extract_resource_id(*args, **kwargs) if extract_resource_id else None
            
            # Log action start
            await audit_logger.log(
                action=action,
                user_id=user_id,
                resource_type=resource_type,
                resource_id=resource_id,
                details={"function": func.__name__}
            )
            
            # Execute function
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                # Log failure
                await audit_logger.log(
                    action=AuditAction.ANALYSIS_FAILED,
                    user_id=user_id,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    details={
                        "function": func.__name__,
                        "error": str(e)
                    },
                    severity="error"
                )
                raise
                
        def sync_wrapper(*args, **kwargs):
            # For sync functions, use asyncio
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(async_wrapper(*args, **kwargs))
            
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        
    return decorator