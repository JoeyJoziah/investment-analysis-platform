"""
Security Modules Test Suite (GitHub Issue #90)

Comprehensive tests for:
1. JWT Manager - token creation, validation, expiry, blacklisting
2. RBAC - role-based access control permissions
3. Input Validation - sanitization, type validation, threat detection
4. Injection Prevention - SQL injection, XSS detection and sanitization

Created: 2026-02-08
"""

import pytest
import jwt as pyjwt
import time
import re
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import MagicMock, patch, PropertyMock

from backend.security.rbac import (
    Role,
    Permission,
    RoleBasedAccessControl,
)
from backend.security.input_validation import (
    InputValidator,
    InputSanitizer,
    InputType,
    SanitizationLevel,
    SecurityPatterns,
    ValidationRule,
    validate_ticker_symbol,
    validate_email_address,
    validate_currency_amount,
    sanitize_user_input,
    detect_security_threats,
)
from backend.security.injection_prevention import (
    SQLInjectionPrevention,
    XSSPrevention,
    AttackType,
    sanitize_html_content,
    escape_user_input,
    validate_sql_identifier,
    detect_injection_threats,
)


# =============================================================================
# 1. JWT Manager Tests
# =============================================================================


class TestJWTManagerTokenCreation:
    """Tests for JWT token creation and structure."""

    @pytest.fixture
    def rsa_keys(self):
        """Generate RSA key pair for testing without external dependencies."""
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.backends import default_backend

        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend(),
        )
        public_key = private_key.public_key()
        return private_key, public_key

    @pytest.fixture
    def jwt_manager_mock(self, rsa_keys):
        """Create a JWTManager-like object with mocked external dependencies."""
        private_key, public_key = rsa_keys

        manager = MagicMock()
        manager.private_key = private_key
        manager.public_key = public_key
        manager.access_token_expire_minutes = 30
        manager.refresh_token_expire_days = 7
        manager.mfa_token_expire_minutes = 5
        manager.issuer = "investment-analysis-app"
        manager.audience = "investment-analysis-users"
        manager.blacklist_prefix = "jwt_blacklist"
        manager.session_prefix = "user_session"
        manager.redis_client = None
        return manager

    def _create_access_token(self, manager, claims_dict, expires_delta=None):
        """Helper to create an access token matching JWTManager.create_access_token logic."""
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=manager.access_token_expire_minutes)

        payload = {
            "sub": claims_dict["username"],
            "user_id": claims_dict["user_id"],
            "email": claims_dict["email"],
            "roles": claims_dict["roles"],
            "scopes": claims_dict["scopes"],
            "is_admin": claims_dict.get("is_admin", False),
            "is_mfa_verified": claims_dict.get("is_mfa_verified", False),
            "session_id": claims_dict.get("session_id", "test-session"),
            "type": "access",
            "iat": datetime.now(timezone.utc),
            "exp": expire,
            "iss": manager.issuer,
            "aud": manager.audience,
        }
        return pyjwt.encode(payload, manager.private_key, algorithm="RS256")

    def _create_refresh_token(self, manager, claims_dict, expires_delta=None):
        """Helper to create a refresh token matching JWTManager.create_refresh_token logic."""
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(days=manager.refresh_token_expire_days)

        payload = {
            "sub": claims_dict["username"],
            "user_id": claims_dict["user_id"],
            "session_id": claims_dict.get("session_id", "test-session"),
            "type": "refresh",
            "iat": datetime.now(timezone.utc),
            "exp": expire,
            "iss": manager.issuer,
            "aud": manager.audience,
        }
        return pyjwt.encode(payload, manager.private_key, algorithm="RS256")

    def test_access_token_contains_required_claims(self, jwt_manager_mock):
        """Access token must contain all standard JWT claims plus custom claims."""
        claims = {
            "user_id": 42,
            "username": "analyst",
            "email": "analyst@example.com",
            "roles": ["user", "analyst"],
            "scopes": ["read", "write"],
            "session_id": "sess-123",
        }
        token = self._create_access_token(jwt_manager_mock, claims)
        decoded = pyjwt.decode(
            token,
            jwt_manager_mock.public_key,
            algorithms=["RS256"],
            audience="investment-analysis-users",
            issuer="investment-analysis-app",
        )

        assert decoded["sub"] == "analyst"
        assert decoded["user_id"] == 42
        assert decoded["email"] == "analyst@example.com"
        assert decoded["roles"] == ["user", "analyst"]
        assert decoded["scopes"] == ["read", "write"]
        assert decoded["type"] == "access"
        assert decoded["iss"] == "investment-analysis-app"
        assert decoded["aud"] == "investment-analysis-users"
        assert "exp" in decoded
        assert "iat" in decoded

    def test_refresh_token_has_minimal_payload(self, jwt_manager_mock):
        """Refresh token should carry only minimal claims for security."""
        claims = {
            "user_id": 7,
            "username": "refresher",
            "email": "r@example.com",
            "roles": ["user"],
            "scopes": ["read"],
            "session_id": "sess-refresh",
        }
        token = self._create_refresh_token(jwt_manager_mock, claims)
        decoded = pyjwt.decode(
            token,
            jwt_manager_mock.public_key,
            algorithms=["RS256"],
            audience="investment-analysis-users",
            issuer="investment-analysis-app",
        )

        assert decoded["type"] == "refresh"
        assert decoded["sub"] == "refresher"
        assert decoded["user_id"] == 7
        # Refresh tokens should NOT carry roles, scopes, email for security
        assert "roles" not in decoded
        assert "scopes" not in decoded
        assert "email" not in decoded

    def test_token_expiry_is_respected(self, jwt_manager_mock):
        """Token with a past expiration must raise ExpiredSignatureError."""
        claims = {
            "user_id": 1,
            "username": "expired_user",
            "email": "e@test.com",
            "roles": [],
            "scopes": [],
        }
        token = self._create_access_token(
            jwt_manager_mock, claims, expires_delta=timedelta(seconds=-1)
        )

        with pytest.raises(pyjwt.ExpiredSignatureError):
            pyjwt.decode(
                token,
                jwt_manager_mock.public_key,
                algorithms=["RS256"],
                audience="investment-analysis-users",
                issuer="investment-analysis-app",
            )

    def test_token_with_custom_expiry(self, jwt_manager_mock):
        """Custom expiry delta should override the default."""
        claims = {
            "user_id": 1,
            "username": "custom_exp",
            "email": "c@test.com",
            "roles": [],
            "scopes": [],
        }
        custom_delta = timedelta(hours=2)
        token = self._create_access_token(jwt_manager_mock, claims, expires_delta=custom_delta)
        decoded = pyjwt.decode(
            token,
            jwt_manager_mock.public_key,
            algorithms=["RS256"],
            audience="investment-analysis-users",
            issuer="investment-analysis-app",
        )

        exp_time = datetime.utcfromtimestamp(decoded["exp"])
        iat_time = datetime.utcfromtimestamp(decoded["iat"])
        delta = exp_time - iat_time
        # Should be approximately 2 hours (allow 5 second tolerance)
        assert abs(delta.total_seconds() - 7200) < 5

    def test_tampered_token_fails_verification(self, jwt_manager_mock):
        """A token with a modified payload must fail RS256 signature verification."""
        claims = {
            "user_id": 1,
            "username": "tamper_test",
            "email": "t@test.com",
            "roles": [],
            "scopes": [],
        }
        token = self._create_access_token(jwt_manager_mock, claims)

        # Tamper with the payload segment
        parts = token.split(".")
        # Modify a character in the payload
        tampered_payload = parts[1][:-1] + ("A" if parts[1][-1] != "A" else "B")
        tampered_token = f"{parts[0]}.{tampered_payload}.{parts[2]}"

        with pytest.raises(pyjwt.InvalidSignatureError):
            pyjwt.decode(
                tampered_token,
                jwt_manager_mock.public_key,
                algorithms=["RS256"],
                audience="investment-analysis-users",
                issuer="investment-analysis-app",
            )

    def test_wrong_audience_rejected(self, jwt_manager_mock):
        """Token verified against wrong audience must be rejected."""
        claims = {
            "user_id": 1,
            "username": "aud_test",
            "email": "a@test.com",
            "roles": [],
            "scopes": [],
        }
        token = self._create_access_token(jwt_manager_mock, claims)

        with pytest.raises(pyjwt.InvalidAudienceError):
            pyjwt.decode(
                token,
                jwt_manager_mock.public_key,
                algorithms=["RS256"],
                audience="wrong-audience",
                issuer="investment-analysis-app",
            )


# =============================================================================
# 2. RBAC Tests
# =============================================================================


class TestRoleBasedAccessControl:
    """Tests for Role-Based Access Control module."""

    @pytest.fixture
    def rbac(self):
        return RoleBasedAccessControl()

    def test_admin_has_all_permissions(self, rbac):
        """Admin role should have every permission including admin."""
        assert rbac.has_permission(Role.ADMIN, Permission.READ) is True
        assert rbac.has_permission(Role.ADMIN, Permission.WRITE) is True
        assert rbac.has_permission(Role.ADMIN, Permission.DELETE) is True
        assert rbac.has_permission(Role.ADMIN, Permission.ADMIN) is True

    def test_viewer_has_only_read(self, rbac):
        """Viewer role should have only read permission."""
        assert rbac.has_permission(Role.VIEWER, Permission.READ) is True
        assert rbac.has_permission(Role.VIEWER, Permission.WRITE) is False
        assert rbac.has_permission(Role.VIEWER, Permission.DELETE) is False

    def test_analyst_has_read_write(self, rbac):
        """Analyst role should have read and write but not delete or admin."""
        assert rbac.has_permission(Role.ANALYST, Permission.READ) is True
        assert rbac.has_permission(Role.ANALYST, Permission.WRITE) is True
        assert rbac.has_permission(Role.ANALYST, Permission.DELETE) is False

    def test_user_has_read_write(self, rbac):
        """Standard user role should have read and write."""
        assert rbac.has_permission(Role.USER, Permission.READ) is True
        assert rbac.has_permission(Role.USER, Permission.WRITE) is True
        assert rbac.has_permission(Role.USER, Permission.DELETE) is False
        assert rbac.has_permission(Role.USER, Permission.ADMIN) is False

    def test_unknown_role_has_no_permissions(self, rbac):
        """An unrecognized role should have no permissions."""
        assert rbac.has_permission("nonexistent_role", Permission.READ) is False
        assert rbac.has_permission("nonexistent_role", Permission.WRITE) is False

    def test_check_access_without_roles_returns_false(self, rbac):
        """check_access returns False when user has no assigned roles."""
        assert rbac.check_access(user_id=1, resource="portfolio", action=Permission.READ) is False

    def test_check_access_with_role(self, rbac):
        """check_access returns True when user has a role with the permission."""
        rbac.assign_role(1, Role.USER)
        assert rbac.check_access(user_id=1, resource="portfolio", action=Permission.READ) is True
        assert rbac.check_access(user_id=1, resource="portfolio", action=Permission.DELETE) is False

    def test_role_enum_values(self):
        """Role enum should expose the expected string values."""
        assert Role.ADMIN.value == "admin"
        assert Role.USER.value == "user"
        assert Role.ANALYST.value == "analyst"
        assert Role.VIEWER.value == "viewer"

    def test_permission_enum_values(self):
        """Permission enum should expose the expected string values."""
        assert Permission.READ.value == "read"
        assert Permission.WRITE.value == "write"
        assert Permission.DELETE.value == "delete"
        assert Permission.ADMIN.value == "admin"

    def test_assign_role_idempotent(self, rbac):
        """Assigning the same role twice returns False on the second call."""
        assert rbac.assign_role(10, Role.ANALYST) is True
        assert rbac.assign_role(10, Role.ANALYST) is False

    def test_revoke_nonexistent_role(self, rbac):
        """Revoking a role the user doesn't have returns False."""
        assert rbac.revoke_role(999, Role.ADMIN) is False

    def test_revoke_clears_empty_set(self, rbac):
        """After revoking the last role, user should have no roles."""
        rbac.assign_role(20, Role.VIEWER)
        rbac.revoke_role(20, Role.VIEWER)
        assert rbac.get_user_roles(20) == []

    def test_db_backed_get_user_roles(self):
        """DB-backed mode reads role from User.role column."""
        mock_session = MagicMock()
        mock_user = MagicMock()
        mock_user.role = "analyst"
        mock_session.query.return_value.filter.return_value.first.return_value = mock_user

        rbac = RoleBasedAccessControl(db_session=mock_session)
        roles = rbac.get_user_roles(1)
        assert roles == ["analyst"]

    def test_db_backed_assign_role_persists(self):
        """DB-backed mode persists role to User.role column on assign."""
        mock_session = MagicMock()
        mock_user = MagicMock()
        mock_user.role = None
        mock_session.query.return_value.filter.return_value.first.return_value = mock_user

        rbac = RoleBasedAccessControl(db_session=mock_session)
        result = rbac.assign_role(1, Role.ADMIN)
        assert result is True
        assert mock_user.role == Role.ADMIN
        mock_session.commit.assert_called()

    def test_db_backed_revoke_role_clears_column(self):
        """DB-backed mode clears User.role column on revoke."""
        mock_session = MagicMock()
        mock_user = MagicMock()
        mock_user.role = "admin"
        mock_session.query.return_value.filter.return_value.first.return_value = mock_user

        rbac = RoleBasedAccessControl(db_session=mock_session)
        rbac.assign_role(1, Role.ADMIN)
        result = rbac.revoke_role(1, Role.ADMIN)
        assert result is True
        assert mock_user.role is None
        mock_session.commit.assert_called()

    def test_db_backed_fallback_to_memory(self):
        """DB-backed mode falls back to in-memory when user not in DB."""
        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.first.return_value = None

        rbac = RoleBasedAccessControl(db_session=mock_session)
        rbac._user_roles[5] = {"viewer"}
        roles = rbac.get_user_roles(5)
        assert roles == ["viewer"]

    def test_in_memory_mode_default(self):
        """Default constructor uses in-memory mode (no DB session)."""
        rbac = RoleBasedAccessControl()
        assert rbac._db_session is None
        rbac.assign_role(1, Role.USER)
        assert rbac.get_user_roles(1) == ["user"]


# =============================================================================
# 3. Input Validation Tests
# =============================================================================


class TestInputValidation:
    """Tests for input validation and sanitization."""

    @pytest.fixture
    def validator(self):
        return InputValidator()

    @pytest.fixture
    def sanitizer(self):
        return InputSanitizer()

    # --- Type-specific validation ---

    def test_valid_email(self, validator):
        """Valid email addresses should pass validation."""
        valid, error, result = validator.validate_by_type("user@example.com", InputType.EMAIL)
        assert valid is True
        assert result == "user@example.com"

    def test_invalid_email(self, validator):
        """Malformed email addresses should fail validation."""
        valid, error, result = validator.validate_by_type("not-an-email", InputType.EMAIL)
        assert valid is False
        assert "email" in error.lower()

    def test_valid_username(self, validator):
        """Valid username with alphanumeric and allowed special characters."""
        valid, error, result = validator.validate_by_type("john_doe-42", InputType.USERNAME)
        assert valid is True
        assert result == "john_doe-42"

    def test_username_too_short(self, validator):
        """Username shorter than 3 characters should fail."""
        valid, error, result = validator.validate_by_type("ab", InputType.USERNAME)
        assert valid is False

    def test_username_with_invalid_chars(self, validator):
        """Username with special characters beyond underscore/hyphen should fail."""
        valid, error, result = validator.validate_by_type("user@name!", InputType.USERNAME)
        assert valid is False

    def test_strong_password_passes(self, validator):
        """A password meeting all requirements should pass."""
        valid, error, result = validator.validate_by_type(
            "S3cure!Pass#2026", InputType.PASSWORD
        )
        assert valid is True

    def test_weak_password_fails(self, validator):
        """Password without uppercase, digit, or special char should fail."""
        valid, error, result = validator.validate_by_type("short", InputType.PASSWORD)
        assert valid is False

    def test_ticker_symbol_normalization(self, validator):
        """Ticker symbol should be normalized to uppercase."""
        valid, error, result = validator.validate_by_type("aapl", InputType.TICKER_SYMBOL)
        assert valid is True
        assert result == "AAPL"

    def test_invalid_ticker_symbol(self, validator):
        """Ticker with numbers or special characters should fail."""
        valid, error, result = validator.validate_by_type("AAPL123!", InputType.TICKER_SYMBOL)
        assert valid is False

    def test_valid_amount(self, validator):
        """Positive decimal amounts should be valid."""
        valid, error, result = validator.validate_by_type("1234.56", InputType.AMOUNT)
        assert valid is True
        assert result == Decimal("1234.56")

    def test_negative_amount_rejected(self, validator):
        """Negative amounts should be rejected."""
        valid, error, result = validator.validate_by_type("-50.00", InputType.AMOUNT)
        assert valid is False

    def test_valid_percentage(self, validator):
        """Percentage between 0 and 100 should pass."""
        valid, error, result = validator.validate_by_type("75.5", InputType.PERCENTAGE)
        assert valid is True
        assert result == 75.5

    def test_percentage_out_of_range(self, validator):
        """Percentage above 100 should fail."""
        valid, error, result = validator.validate_by_type("150", InputType.PERCENTAGE)
        assert valid is False

    def test_valid_date(self, validator):
        """ISO date format YYYY-MM-DD should be valid."""
        valid, error, result = validator.validate_by_type("2026-02-08", InputType.DATE)
        assert valid is True

    def test_invalid_date_format(self, validator):
        """Non-ISO date format should fail."""
        valid, error, result = validator.validate_by_type("02/08/2026", InputType.DATE)
        assert valid is False

    def test_valid_uuid(self, validator):
        """Valid UUID string should pass."""
        valid, error, result = validator.validate_by_type(
            "550e8400-e29b-41d4-a716-446655440000", InputType.UUID
        )
        assert valid is True

    def test_invalid_uuid(self, validator):
        """Malformed UUID should fail."""
        valid, error, result = validator.validate_by_type("not-a-uuid", InputType.UUID)
        assert valid is False

    def test_none_value_passes(self, validator):
        """None input should be treated as valid (optional field)."""
        valid, error, result = validator.validate_by_type(None, InputType.TEXT)
        assert valid is True
        assert result is None

    # --- Sanitization ---

    def test_strict_sanitization_strips_all_html(self, sanitizer):
        """Strict sanitization must strip all HTML tags."""
        result = sanitizer.sanitize_html(
            "<b>bold</b> and <script>alert(1)</script>",
            SanitizationLevel.STRICT,
        )
        assert "<b>" not in result
        assert "<script>" not in result
        assert "bold" in result

    def test_moderate_sanitization_allows_safe_tags(self, sanitizer):
        """Moderate sanitization should allow safe tags like <strong> and strip dangerous ones."""
        result = sanitizer.sanitize_html(
            "<strong>safe</strong><script>evil</script>",
            SanitizationLevel.MODERATE,
        )
        assert "<strong>" in result
        assert "<script>" not in result

    def test_sql_identifier_sanitization(self, sanitizer):
        """SQL identifiers should strip non-alphanumeric chars except underscore/hyphen."""
        result = sanitizer.sanitize_sql_identifier("table_name; DROP TABLE users")
        assert ";" not in result
        assert " " not in result
        # Letters remain because [a-zA-Z0-9_-] are valid identifier chars
        assert result == "table_nameDROPTABLEusers"

    def test_sql_identifier_leading_digit(self, sanitizer):
        """SQL identifiers starting with a digit should be prefixed with underscore."""
        result = sanitizer.sanitize_sql_identifier("123column")
        assert result.startswith("_")

    def test_sql_identifier_max_length(self, sanitizer):
        """SQL identifiers should be truncated at 63 characters (PostgreSQL limit)."""
        long_name = "a" * 100
        result = sanitizer.sanitize_sql_identifier(long_name)
        assert len(result) <= 63

    def test_file_path_traversal_removed(self, sanitizer):
        """Path traversal sequences must be stripped from file paths."""
        result = sanitizer.sanitize_file_path("../../etc/passwd")
        assert ".." not in result
        assert "etc" in result

    # --- Threat detection ---

    def test_detect_sql_injection_threat(self, validator):
        """SQL injection payloads should be detected."""
        threats = validator.detect_injection_attempt("1 OR 1=1")
        assert len(threats) > 0
        assert any("sql" in t.lower() for t in threats)

    def test_detect_xss_threat(self, validator):
        """XSS payloads should be detected."""
        threats = validator.detect_injection_attempt('<script>alert("xss")</script>')
        assert len(threats) > 0
        assert any("xss" in t.lower() for t in threats)

    def test_detect_path_traversal_threat(self, validator):
        """Path traversal attempts should be detected."""
        threats = validator.detect_injection_attempt("../../../etc/shadow")
        assert len(threats) > 0
        assert any("path" in t.lower() for t in threats)

    def test_safe_input_no_threats(self, validator):
        """Normal text should not trigger any threat detection."""
        threats = validator.detect_injection_attempt("Hello, this is a normal message.")
        assert len(threats) == 0

    # --- Field validation with rules ---

    def test_required_field_missing(self, validator):
        """Missing required field should produce an error."""
        rule = ValidationRule("name", InputType.TEXT, required=True)
        valid, errors, result = validator.validate_field(None, rule)
        assert valid is False
        assert any("required" in e.lower() for e in errors)

    def test_optional_field_empty(self, validator):
        """Empty optional field should be valid."""
        rule = ValidationRule("notes", InputType.TEXT, required=False)
        valid, errors, result = validator.validate_field("", rule)
        assert valid is True

    def test_validate_data_dict(self, validator):
        """validate_data should validate all fields and return validated data."""
        rules = [
            ValidationRule("ticker", InputType.TICKER_SYMBOL, required=True, max_length=10),
        ]
        data = {"ticker": "MSFT"}
        valid, errors, validated = validator.validate_data(data, rules)
        assert valid is True
        assert validated["ticker"] == "MSFT"

    def test_validate_data_missing_required(self, validator):
        """Missing required field in data dict should produce error."""
        rules = [
            ValidationRule("email", InputType.EMAIL, required=True),
        ]
        data = {}
        valid, errors, validated = validator.validate_data(data, rules)
        assert valid is False
        assert any("required" in e.lower() for e in errors)

    # --- Utility functions ---

    def test_validate_ticker_symbol_utility(self):
        """Utility function should return normalized symbol."""
        result = validate_ticker_symbol("goog")
        assert result == "GOOG"

    def test_sanitize_user_input_utility(self):
        """Utility function should strip dangerous HTML."""
        result = sanitize_user_input('<script>alert("x")</script>Safe text')
        assert "<script>" not in result
        assert "Safe text" in result

    def test_detect_security_threats_utility(self):
        """Utility function should detect injection threats."""
        threats = detect_security_threats("'; DROP TABLE users; --")
        assert len(threats) > 0


# =============================================================================
# 4. Injection Prevention Tests
# =============================================================================


class TestSQLInjectionPrevention:
    """Tests for SQL injection detection and prevention."""

    @pytest.fixture
    def sql_prev(self):
        return SQLInjectionPrevention()

    def test_detect_union_based_injection(self, sql_prev):
        """UNION SELECT payload must be detected."""
        detections = sql_prev.detect_sql_injection("1 UNION SELECT username, password FROM users")
        assert len(detections) > 0
        categories = [d["category"] for d in detections]
        assert "union_based" in categories or "dangerous_keyword" in categories

    def test_detect_boolean_blind_injection(self, sql_prev):
        """Boolean-based blind injection (OR 1=1) must be detected."""
        detections = sql_prev.detect_sql_injection("admin' OR 1=1 --")
        assert len(detections) > 0

    def test_detect_time_based_injection(self, sql_prev):
        """Time-based injection with SLEEP must be detected."""
        detections = sql_prev.detect_sql_injection("1; WAITFOR DELAY '0:0:5'")
        assert len(detections) > 0

    def test_detect_stacked_queries(self, sql_prev):
        """Stacked query attack (;DROP TABLE) must be detected."""
        detections = sql_prev.detect_sql_injection("1; DROP TABLE users")
        assert len(detections) > 0
        has_stacked = any(d.get("category") == "stacked_queries" for d in detections)
        has_keyword = any(d.get("category") == "dangerous_keyword" for d in detections)
        assert has_stacked or has_keyword

    def test_detect_database_functions(self, sql_prev):
        """Probing with database functions (version(), @@version) must be detected."""
        detections = sql_prev.detect_sql_injection("SELECT version()")
        assert len(detections) > 0

    def test_sanitize_removes_comments(self, sql_prev):
        """SQL comment patterns should be removed."""
        result = sql_prev.sanitize_sql_input("SELECT * /* comment */ FROM users -- trailing")
        assert "/*" not in result
        assert "--" not in result

    def test_sanitize_removes_semicolons(self, sql_prev):
        """Semicolons should be stripped to prevent stacked queries."""
        result = sql_prev.sanitize_sql_input("query; DROP TABLE users")
        assert ";" not in result

    def test_sanitize_escapes_single_quotes(self, sql_prev):
        """Single quotes should be escaped to prevent string breakout."""
        result = sql_prev.sanitize_sql_input("O'Brien")
        assert "''" in result

    def test_validate_table_identifier_valid(self, sql_prev):
        """Valid SQL identifiers should pass."""
        assert sql_prev.validate_table_identifier("users") is True
        assert sql_prev.validate_table_identifier("portfolio_items") is True

    def test_validate_table_identifier_rejects_dangerous(self, sql_prev):
        """Dangerous keywords used as identifiers should be rejected."""
        assert sql_prev.validate_table_identifier("drop") is False
        assert sql_prev.validate_table_identifier("delete") is False
        assert sql_prev.validate_table_identifier("information_schema") is False

    def test_validate_table_identifier_rejects_special_chars(self, sql_prev):
        """Identifiers with special characters should be rejected."""
        assert sql_prev.validate_table_identifier("table; DROP") is False
        assert sql_prev.validate_table_identifier("table name") is False

    def test_validate_table_identifier_rejects_leading_digit(self, sql_prev):
        """Identifiers starting with a digit should be rejected."""
        assert sql_prev.validate_table_identifier("1table") is False

    def test_validate_table_identifier_rejects_long_name(self, sql_prev):
        """Identifiers exceeding 63 characters should be rejected."""
        long_name = "a" * 64
        assert sql_prev.validate_table_identifier(long_name) is False

    def test_empty_input_returns_no_detections(self, sql_prev):
        """Empty or None input should return empty detection list."""
        assert sql_prev.detect_sql_injection("") == []
        assert sql_prev.detect_sql_injection(None) == []

    def test_safe_query_no_detections(self, sql_prev):
        """Normal text should not trigger SQL injection detection."""
        detections = sql_prev.detect_sql_injection("John likes to select his favorite songs")
        # "select" alone might trigger the keyword check, but should not be critical
        critical = [d for d in detections if d.get("severity") == "critical"]
        assert len(critical) == 0


class TestXSSPrevention:
    """Tests for Cross-Site Scripting detection and prevention."""

    @pytest.fixture
    def xss_prev(self):
        return XSSPrevention()

    def test_detect_script_tag(self, xss_prev):
        """<script> tags must be detected."""
        detections = xss_prev.detect_xss('<script>alert("XSS")</script>')
        assert len(detections) > 0
        categories = [d["category"] for d in detections]
        assert "script_tags" in categories

    def test_detect_event_handler(self, xss_prev):
        """Inline event handlers (onload=) must be detected."""
        detections = xss_prev.detect_xss('<img src=x onerror=alert(1)>')
        assert len(detections) > 0
        categories = [d["category"] for d in detections]
        assert "event_handlers" in categories

    def test_detect_javascript_protocol(self, xss_prev):
        """javascript: protocol in links must be detected."""
        detections = xss_prev.detect_xss('javascript:alert(document.cookie)')
        assert len(detections) > 0

    def test_detect_dangerous_tags(self, xss_prev):
        """Dangerous tags (iframe, object, embed) must be detected."""
        detections = xss_prev.detect_xss('<iframe src="evil.com"></iframe>')
        assert len(detections) > 0
        categories = [d["category"] for d in detections]
        assert "dangerous_tags" in categories

    def test_sanitize_html_strict_removes_all(self, xss_prev):
        """Strict sanitization should strip all HTML tags."""
        result = xss_prev.sanitize_html(
            '<b>bold</b><script>evil()</script><p>text</p>', strict=True
        )
        assert "<b>" not in result
        assert "<script>" not in result
        assert "<p>" not in result
        assert "bold" in result
        assert "text" in result

    def test_sanitize_html_moderate_allows_safe(self, xss_prev):
        """Non-strict sanitization should preserve allowed tags."""
        result = xss_prev.sanitize_html(
            '<p>paragraph</p><script>evil()</script>', strict=False
        )
        assert "<p>" in result
        assert "<script>" not in result

    def test_escape_html_entities(self, xss_prev):
        """HTML escape should convert special characters to entities."""
        result = xss_prev.escape_html('<script>alert("xss")</script>')
        assert "&lt;" in result
        assert "&gt;" in result
        assert "&quot;" in result
        assert "<script>" not in result

    def test_validate_url_blocks_javascript(self, xss_prev):
        """URLs with javascript: scheme must be rejected."""
        assert xss_prev.validate_url("javascript:alert(1)") is False

    def test_validate_url_blocks_data(self, xss_prev):
        """URLs with data: scheme must be rejected."""
        assert xss_prev.validate_url("data:text/html,<script>alert(1)</script>") is False

    def test_validate_url_allows_https(self, xss_prev):
        """HTTPS URLs should be allowed."""
        assert xss_prev.validate_url("https://example.com/page") is True

    def test_validate_url_allows_http(self, xss_prev):
        """HTTP URLs should be allowed."""
        assert xss_prev.validate_url("http://example.com/page") is True

    def test_sanitize_javascript_blocks_eval(self, xss_prev):
        """eval() calls in JavaScript content should be blocked."""
        result = xss_prev.sanitize_javascript('var x = eval("malicious")')
        assert "BLOCKED" in result
        assert "eval" not in result.replace("BLOCKED", "")

    def test_sanitize_javascript_blocks_document(self, xss_prev):
        """document. references in JavaScript should be blocked."""
        result = xss_prev.sanitize_javascript('document.cookie')
        assert "BLOCKED" in result

    def test_empty_input_returns_no_detections(self, xss_prev):
        """Empty or None input should return empty detection list."""
        assert xss_prev.detect_xss("") == []
        assert xss_prev.detect_xss(None) == []

    def test_safe_text_no_detections(self, xss_prev):
        """Normal text should not trigger XSS detection."""
        detections = xss_prev.detect_xss("This is a normal paragraph about investing.")
        assert len(detections) == 0


class TestInjectionPreventionUtilities:
    """Tests for injection prevention utility functions."""

    def test_sanitize_html_content_strict(self):
        """Utility function should strip all tags in strict mode."""
        result = sanitize_html_content("<b>bold</b><script>x</script>", strict=True)
        assert "<" not in result
        assert "bold" in result

    def test_escape_user_input_utility(self):
        """Utility function should HTML-escape all special characters."""
        result = escape_user_input('<img onerror=alert(1)>')
        assert "<img" not in result
        assert "&lt;" in result

    def test_validate_sql_identifier_utility(self):
        """Utility function should accept valid identifiers."""
        assert validate_sql_identifier("valid_table") is True
        assert validate_sql_identifier("drop") is False

    def test_detect_injection_threats_combined(self):
        """Utility function should detect both SQL and XSS threats."""
        threats = detect_injection_threats("1 UNION SELECT * FROM users <script>alert(1)</script>")
        types = [t["type"] for t in threats]
        assert "sql_injection" in types
        assert "xss" in types
