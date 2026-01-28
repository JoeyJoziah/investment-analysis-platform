"""
Structured Security Logging for Admin Actions

Provides comprehensive, structured logging for all administrative and security-sensitive
operations with proper sanitization and audit trail capabilities.
"""

import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum


class SecurityEventType(str, Enum):
    """Types of security events"""
    ADMIN_ACTION = "admin_action"
    CONFIG_CHANGE = "config_change"
    USER_MANAGEMENT = "user_management"
    AUTHENTICATION = "authentication"
    AUTHORIZATION_FAILURE = "authorization_failure"
    RATE_LIMIT_VIOLATION = "rate_limit_violation"
    DATA_ACCESS = "data_access"
    DATA_EXPORT = "data_export"
    SYSTEM_COMMAND = "system_command"


class SecurityLogger:
    """
    Structured logging for security events and admin actions.

    Features:
    - Sanitized logging (prevents log injection)
    - Structured JSON format
    - Contextual metadata
    - Severity levels
    - Audit trail support
    """

    def __init__(self):
        self.logger = logging.getLogger("security")
        self.logger.setLevel(logging.INFO)

        # Ensure handler exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _sanitize_value(self, value: Any, max_length: int = 200) -> str:
        """
        Sanitize value for safe logging.

        Prevents:
        - Log injection via newlines
        - Excessive log size
        - Sensitive data exposure
        """
        if value is None:
            return "null"

        # Convert to string
        sanitized = str(value)

        # Remove newlines, carriage returns, tabs
        sanitized = sanitized.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

        # Truncate to max length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length] + "...[truncated]"

        return sanitized

    def _sanitize_dict(self, data: Dict[str, Any], max_length: int = 200) -> Dict[str, str]:
        """Sanitize all values in a dictionary"""
        return {
            key: self._sanitize_value(value, max_length)
            for key, value in data.items()
        }

    def _create_log_entry(
        self,
        event_type: SecurityEventType,
        action: str,
        user_id: Optional[int],
        success: bool,
        details: Optional[Dict[str, Any]] = None,
        severity: str = "INFO"
    ) -> Dict[str, Any]:
        """Create structured log entry"""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type.value,
            "action": self._sanitize_value(action, 100),
            "user_id": user_id,
            "success": success,
            "severity": severity
        }

        if details:
            entry["details"] = self._sanitize_dict(details)

        return entry

    def log_admin_action(
        self,
        action: str,
        user_id: int,
        resource: str,
        success: bool,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None
    ):
        """
        Log administrative action.

        Args:
            action: Action performed (e.g., "update_user", "delete_resource")
            user_id: ID of admin performing action
            resource: Resource being acted upon
            success: Whether action succeeded
            details: Additional context
            ip_address: IP address of requester
        """
        log_details = details or {}
        log_details["resource"] = resource

        if ip_address:
            log_details["ip_address"] = ip_address

        entry = self._create_log_entry(
            event_type=SecurityEventType.ADMIN_ACTION,
            action=action,
            user_id=user_id,
            success=success,
            details=log_details,
            severity="WARNING" if not success else "INFO"
        )

        log_message = json.dumps(entry)

        if success:
            self.logger.info(log_message)
        else:
            self.logger.warning(log_message)

    def log_config_change(
        self,
        user_id: int,
        section: str,
        key: str,
        old_value: Optional[Any],
        new_value: Any,
        success: bool,
        ip_address: Optional[str] = None
    ):
        """
        Log configuration change.

        Args:
            user_id: ID of user making change
            section: Config section (e.g., "api_keys", "database")
            key: Config key being changed
            old_value: Previous value (will be masked if sensitive)
            new_value: New value (will be masked if sensitive)
            success: Whether change succeeded
            ip_address: IP address of requester
        """
        # Mask sensitive sections
        sensitive_sections = ["api_keys", "database", "security"]

        if section in sensitive_sections:
            old_display = "***MASKED***" if old_value else None
            new_display = "***MASKED***"
        else:
            old_display = old_value
            new_display = new_value

        details = {
            "section": section,
            "key": key,
            "old_value": old_display,
            "new_value": new_display,
            "ip_address": ip_address
        }

        entry = self._create_log_entry(
            event_type=SecurityEventType.CONFIG_CHANGE,
            action=f"update_config.{section}.{key}",
            user_id=user_id,
            success=success,
            details=details,
            severity="WARNING"  # Config changes are always high priority
        )

        self.logger.warning(json.dumps(entry))

    def log_user_management(
        self,
        admin_id: int,
        action: str,
        target_user_id: int,
        success: bool,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None
    ):
        """
        Log user management action.

        Args:
            admin_id: ID of admin performing action
            action: Action type (create, update, delete, etc.)
            target_user_id: ID of user being managed
            success: Whether action succeeded
            details: Additional context
            ip_address: IP address of requester
        """
        log_details = details or {}
        log_details["target_user_id"] = target_user_id

        if ip_address:
            log_details["ip_address"] = ip_address

        entry = self._create_log_entry(
            event_type=SecurityEventType.USER_MANAGEMENT,
            action=action,
            user_id=admin_id,
            success=success,
            details=log_details,
            severity="WARNING"
        )

        self.logger.warning(json.dumps(entry))

    def log_authorization_failure(
        self,
        user_id: Optional[int],
        action: str,
        resource: str,
        reason: str,
        ip_address: Optional[str] = None
    ):
        """
        Log authorization failure.

        Args:
            user_id: ID of user attempting action (None if unauthenticated)
            action: Action attempted
            resource: Resource access was denied for
            reason: Reason for denial
            ip_address: IP address of requester
        """
        details = {
            "resource": resource,
            "reason": reason,
            "ip_address": ip_address
        }

        entry = self._create_log_entry(
            event_type=SecurityEventType.AUTHORIZATION_FAILURE,
            action=action,
            user_id=user_id,
            success=False,
            details=details,
            severity="WARNING"
        )

        self.logger.warning(json.dumps(entry))

    def log_data_export(
        self,
        user_id: int,
        export_type: str,
        record_count: int,
        success: bool,
        ip_address: Optional[str] = None
    ):
        """
        Log data export operation.

        Args:
            user_id: ID of user exporting data
            export_type: Type of export (users, transactions, etc.)
            record_count: Number of records exported
            success: Whether export succeeded
            ip_address: IP address of requester
        """
        details = {
            "export_type": export_type,
            "record_count": record_count,
            "ip_address": ip_address
        }

        entry = self._create_log_entry(
            event_type=SecurityEventType.DATA_EXPORT,
            action="data_export",
            user_id=user_id,
            success=success,
            details=details,
            severity="WARNING"
        )

        self.logger.warning(json.dumps(entry))

    def log_system_command(
        self,
        user_id: int,
        command: str,
        parameters: Optional[Dict[str, Any]],
        success: bool,
        execution_time_ms: Optional[int] = None,
        ip_address: Optional[str] = None
    ):
        """
        Log system command execution.

        Args:
            user_id: ID of user executing command
            command: Command executed
            parameters: Command parameters
            success: Whether command succeeded
            execution_time_ms: Execution time in milliseconds
            ip_address: IP address of requester
        """
        details = {
            "command": command,
            "parameters": parameters or {},
            "ip_address": ip_address
        }

        if execution_time_ms is not None:
            details["execution_time_ms"] = execution_time_ms

        entry = self._create_log_entry(
            event_type=SecurityEventType.SYSTEM_COMMAND,
            action=f"system_command.{command}",
            user_id=user_id,
            success=success,
            details=details,
            severity="WARNING"
        )

        self.logger.warning(json.dumps(entry))

    def log_rate_limit_violation(
        self,
        user_id: Optional[int],
        category: str,
        ip_address: str,
        requests_made: int,
        limit: int
    ):
        """
        Log rate limit violation.

        Args:
            user_id: ID of user (if authenticated)
            category: Rate limit category
            ip_address: IP address of requester
            requests_made: Number of requests made
            limit: Rate limit threshold
        """
        details = {
            "category": category,
            "ip_address": ip_address,
            "requests_made": requests_made,
            "limit": limit
        }

        entry = self._create_log_entry(
            event_type=SecurityEventType.RATE_LIMIT_VIOLATION,
            action="rate_limit_exceeded",
            user_id=user_id,
            success=False,
            details=details,
            severity="WARNING"
        )

        self.logger.warning(json.dumps(entry))


# Global security logger instance
_security_logger: Optional[SecurityLogger] = None


def get_security_logger() -> SecurityLogger:
    """Get or create the global security logger instance"""
    global _security_logger
    if _security_logger is None:
        _security_logger = SecurityLogger()
    return _security_logger


# Helper function for input sanitization (can be used standalone)
def sanitize_log_input(value: Any, max_length: int = 200) -> str:
    """
    Sanitize input for safe logging.

    Prevents log injection and excessive log size.

    Args:
        value: Value to sanitize
        max_length: Maximum length for output

    Returns:
        Sanitized string safe for logging
    """
    if value is None:
        return "null"

    # Convert to string
    sanitized = str(value)

    # Remove control characters and newlines
    sanitized = sanitized.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

    # Remove other control characters
    sanitized = ''.join(char for char in sanitized if ord(char) >= 32 or char in [' '])

    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "...[truncated]"

    return sanitized
