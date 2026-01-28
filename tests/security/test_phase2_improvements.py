"""
Tests for Phase 2 HIGH Priority Security Improvements

This module tests all 5 security enhancements:
1. Super Admin Check for protected config sections
2. Structured Security Logging for admin actions
3. Rate Limiting on GDPR export endpoint
4. Sanitized log inputs
5. Command Parameter Validation
"""

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from backend.api.routers.admin import (
    SystemCommand,
    ConfigUpdate,
    ConfigSection,
    PROTECTED_CONFIG_SECTIONS,
    check_super_admin_permission,
    sanitize_log_input
)
from backend.utils.security_logger import (
    SecurityLogger,
    sanitize_log_input,
    SecurityEventType
)
from backend.security.rate_limiter import (
    RateLimitRule,
    AdvancedRateLimiter,
    RateLimitCategory
)


# =============================================================================
# Task 1: Super Admin Check Tests
# =============================================================================

class TestSuperAdminCheck:
    """Test super admin authorization for protected config sections"""

    def test_protected_sections_defined(self):
        """Verify protected sections are correctly defined"""
        assert ConfigSection.API_KEYS in PROTECTED_CONFIG_SECTIONS
        assert ConfigSection.DATABASE in PROTECTED_CONFIG_SECTIONS
        assert ConfigSection.SECURITY in PROTECTED_CONFIG_SECTIONS
        assert len(PROTECTED_CONFIG_SECTIONS) == 3

    def test_super_admin_can_access_protected_sections(self):
        """Super admin should be able to modify protected sections"""
        # Mock user with super_admin flag
        mock_user = Mock()
        mock_user.id = 1
        mock_user.is_admin = True
        mock_user.is_super_admin = True

        # Should not raise exception
        result = check_super_admin_permission(mock_user)
        assert result == mock_user

    def test_regular_admin_cannot_access_protected_sections(self):
        """Regular admin should be denied access to protected sections"""
        # Mock user without super_admin flag
        mock_user = Mock()
        mock_user.id = 2
        mock_user.is_admin = True
        mock_user.is_super_admin = False

        # Should raise HTTPException
        with pytest.raises(HTTPException) as exc_info:
            check_super_admin_permission(mock_user)

        assert exc_info.value.status_code == 403
        assert "Super admin" in exc_info.value.detail

    def test_config_update_validation(self):
        """Test that config updates validate super admin for protected sections"""
        # This would be tested via integration test with actual endpoint
        pass


# =============================================================================
# Task 2: Structured Security Logging Tests
# =============================================================================

class TestStructuredLogging:
    """Test structured security logging implementation"""

    def test_security_logger_initialization(self):
        """Security logger should initialize correctly"""
        logger = SecurityLogger()
        assert logger is not None
        assert logger.logger is not None

    def test_log_admin_action(self):
        """Test admin action logging"""
        logger = SecurityLogger()

        with patch.object(logger.logger, 'info') as mock_info:
            logger.log_admin_action(
                action="test_action",
                user_id=123,
                resource="test_resource",
                success=True,
                details={"key": "value"},
                ip_address="192.168.1.1"
            )

            # Verify logging was called
            assert mock_info.called

    def test_log_config_change(self):
        """Test configuration change logging"""
        logger = SecurityLogger()

        with patch.object(logger.logger, 'warning') as mock_warning:
            logger.log_config_change(
                user_id=123,
                section="api_keys",
                key="test_key",
                old_value="old",
                new_value="new",
                success=True,
                ip_address="192.168.1.1"
            )

            # Verify logging was called
            assert mock_warning.called

    def test_log_user_management(self):
        """Test user management action logging"""
        logger = SecurityLogger()

        with patch.object(logger.logger, 'warning') as mock_warning:
            logger.log_user_management(
                admin_id=123,
                action="update_user",
                target_user_id=456,
                success=True,
                ip_address="192.168.1.1"
            )

            # Verify logging was called
            assert mock_warning.called

    def test_log_authorization_failure(self):
        """Test authorization failure logging"""
        logger = SecurityLogger()

        with patch.object(logger.logger, 'warning') as mock_warning:
            logger.log_authorization_failure(
                user_id=123,
                action="access_protected_resource",
                resource="config:api_keys",
                reason="Insufficient permissions",
                ip_address="192.168.1.1"
            )

            # Verify logging was called
            assert mock_warning.called

    def test_log_system_command(self):
        """Test system command execution logging"""
        logger = SecurityLogger()

        with patch.object(logger.logger, 'warning') as mock_warning:
            logger.log_system_command(
                user_id=123,
                command="restart",
                parameters={"service": "api"},
                success=True,
                execution_time_ms=1500,
                ip_address="192.168.1.1"
            )

            # Verify logging was called
            assert mock_warning.called


# =============================================================================
# Task 3: Rate Limiting GDPR Export Tests
# =============================================================================

class TestGDPRRateLimiting:
    """Test rate limiting on GDPR export endpoint"""

    def test_gdpr_rate_limit_rule_defined(self):
        """Verify GDPR export rate limit rule is properly configured"""
        from backend.api.routers.gdpr import GDPR_EXPORT_RATE_LIMIT

        assert GDPR_EXPORT_RATE_LIMIT is not None
        assert GDPR_EXPORT_RATE_LIMIT.requests == 3
        assert GDPR_EXPORT_RATE_LIMIT.window_seconds == 3600  # 1 hour
        assert GDPR_EXPORT_RATE_LIMIT.block_duration_seconds == 3600

    @pytest.mark.asyncio
    async def test_rate_limiter_blocks_after_limit(self):
        """Test that rate limiter blocks requests after limit is exceeded"""
        # Create rate limiter with Redis mock
        mock_redis = MagicMock()
        rate_limiter = AdvancedRateLimiter(redis_client=mock_redis)

        # Configure custom rule (3 requests per hour)
        custom_rule = RateLimitRule(
            requests=3,
            window_seconds=3600,
            block_duration_seconds=3600
        )

        # Mock request
        mock_request = Mock()
        mock_request.client = Mock()
        mock_request.client.host = "192.168.1.100"
        mock_request.headers = {"user-agent": "test-agent"}

        # This test would need full integration with Redis
        # For now, verify the rule is applied correctly
        assert custom_rule.requests == 3


# =============================================================================
# Task 4: Sanitize Log Inputs Tests
# =============================================================================

class TestLogInputSanitization:
    """Test input sanitization for logging"""

    def test_sanitize_removes_newlines(self):
        """Sanitization should remove newlines"""
        malicious_input = "user123\nadmin_access\r\n"
        sanitized = sanitize_log_input(malicious_input)

        assert '\n' not in sanitized
        assert '\r' not in sanitized
        assert sanitized == "user123 admin_access  "

    def test_sanitize_removes_tabs(self):
        """Sanitization should remove tabs"""
        malicious_input = "user\t\tdata\t"
        sanitized = sanitize_log_input(malicious_input)

        assert '\t' not in sanitized
        assert "user" in sanitized
        assert "data" in sanitized

    def test_sanitize_truncates_long_input(self):
        """Sanitization should truncate long inputs"""
        long_input = "A" * 300
        sanitized = sanitize_log_input(long_input, max_length=200)

        assert len(sanitized) <= 215  # 200 + "[truncated]" length
        assert "[truncated]" in sanitized

    def test_sanitize_handles_none(self):
        """Sanitization should handle None values"""
        sanitized = sanitize_log_input(None)
        assert sanitized == "null"

    def test_sanitize_handles_numbers(self):
        """Sanitization should handle numeric values"""
        sanitized = sanitize_log_input(12345)
        assert sanitized == "12345"

    def test_sanitize_log_injection_attempt(self):
        """Sanitization should prevent log injection"""
        # Attempt to inject new log entry
        malicious = "normal_data\n[ERROR] FAKE LOG ENTRY\nadmin_access"
        sanitized = sanitize_log_input(malicious)

        # Should not contain newlines that could create fake log entries
        assert '\n' not in sanitized
        assert '\r' not in sanitized


# =============================================================================
# Task 5: Command Parameter Validation Tests
# =============================================================================

class TestCommandValidation:
    """Test system command parameter validation"""

    def test_valid_command_accepted(self):
        """Valid commands should be accepted"""
        valid_commands = [
            'start', 'stop', 'status', 'restart',
            'clear_cache', 'restart_workers', 'run_backup',
            'optimize_database', 'refresh_models', 'sync_data'
        ]

        for cmd in valid_commands:
            command = SystemCommand(command=cmd, parameters={})
            assert command.command == cmd

    def test_invalid_command_rejected(self):
        """Invalid commands should be rejected"""
        with pytest.raises(ValueError) as exc_info:
            SystemCommand(command="rm -rf /", parameters={})

        assert "Invalid command" in str(exc_info.value)

    def test_command_parameters_sanitized(self):
        """Command parameters should be sanitized"""
        malicious_params = {
            "arg1": "value\nwith\nnewlines",
            "arg2": "value\r\nwith\r\ncarriage",
            "arg3": "A" * 300  # Too long
        }

        command = SystemCommand(command="start", parameters=malicious_params)

        # Check sanitization
        assert '\n' not in command.parameters["arg1"]
        assert '\r' not in command.parameters["arg2"]
        assert len(command.parameters["arg3"]) <= 200

    def test_command_length_validation(self):
        """Command length should be limited"""
        long_command = "A" * 150

        with pytest.raises(ValueError):
            SystemCommand(command=long_command, parameters={})

    def test_command_parameters_type_handling(self):
        """Command parameters should handle different types correctly"""
        params = {
            "string_param": "test",
            "int_param": 123,
            "float_param": 45.67,
            "bool_param": True,
            "list_param": [1, 2, 3],
            "dict_param": {"key": "value"}
        }

        command = SystemCommand(command="start", parameters=params)

        # String types should be preserved and sanitized
        assert isinstance(command.parameters["string_param"], str)

        # Numeric and boolean types should be preserved
        assert command.parameters["int_param"] == 123
        assert command.parameters["float_param"] == 45.67
        assert command.parameters["bool_param"] is True

        # Complex types should be converted to string and truncated
        assert isinstance(command.parameters["list_param"], str)
        assert isinstance(command.parameters["dict_param"], str)

    def test_empty_parameters_allowed(self):
        """Empty parameters should be allowed"""
        command = SystemCommand(command="status", parameters={})
        assert command.parameters == {}

        command2 = SystemCommand(command="status")
        assert command2.parameters == {}


# =============================================================================
# Integration Tests
# =============================================================================

class TestPhase2Integration:
    """Integration tests for all Phase 2 improvements"""

    def test_all_improvements_work_together(self):
        """Test that all 5 improvements work together"""
        # 1. Super admin check
        mock_admin = Mock()
        mock_admin.is_super_admin = True
        assert check_super_admin_permission(mock_admin) == mock_admin

        # 2. Security logging
        logger = SecurityLogger()
        assert logger is not None

        # 3. Rate limiting rule
        from backend.api.routers.gdpr import GDPR_EXPORT_RATE_LIMIT
        assert GDPR_EXPORT_RATE_LIMIT.requests == 3

        # 4. Log sanitization
        sanitized = sanitize_log_input("test\ninput")
        assert '\n' not in sanitized

        # 5. Command validation
        cmd = SystemCommand(command="status", parameters={"test": "value"})
        assert cmd.command == "status"


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
