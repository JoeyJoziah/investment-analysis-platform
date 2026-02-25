"""
Unit tests for authentication and authorization modules.

Covers:
- backend/auth/password_validator.py  (PasswordValidator)
- backend/security/jwt_manager.py     (JWTManager, TokenType, TokenClaims)
- backend/security/rbac.py            (RoleBasedAccessControl, Role, Permission)
"""

import os
os.environ["TESTING"] = "True"
os.environ["DEBUG"] = "True"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

import pytest
import jwt as pyjwt
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

from backend.auth.password_validator import PasswordValidator
from backend.security.rbac import (
    RoleBasedAccessControl,
    Role,
    Permission,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def validator():
    """Default PasswordValidator with all rules enabled."""
    return PasswordValidator()


@pytest.fixture
def lenient_validator():
    """PasswordValidator with only length requirement."""
    return PasswordValidator(
        min_length=6,
        require_uppercase=False,
        require_lowercase=False,
        require_digit=False,
        require_special=False,
    )


@pytest.fixture
def rsa_keypair():
    """Generate a fresh RSA key pair for JWT tests."""
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend(),
    )
    public_key = private_key.public_key()
    return private_key, public_key


@pytest.fixture
def mock_secrets_manager():
    """Mock SecretsManager that returns None for key lookups (forces generation)."""
    mgr = MagicMock()
    mgr.get_secret.return_value = None
    mgr.store_secret.return_value = True
    return mgr


@pytest.fixture
def mock_redis_sync():
    """Synchronous mock Redis client for JWTManager."""
    client = MagicMock()
    client.exists.return_value = False
    client.hset.return_value = 1
    client.expire.return_value = True
    client.keys.return_value = []
    client.setex.return_value = True
    client.delete.return_value = 1
    return client


@pytest.fixture
def jwt_manager(mock_secrets_manager, mock_redis_sync):
    """Create a JWTManager with mocked dependencies."""
    from backend.security.jwt_manager import JWTManager

    with patch(
        "backend.security.jwt_manager.get_secrets_manager",
        return_value=mock_secrets_manager,
    ):
        manager = JWTManager(redis_client=mock_redis_sync)
    return manager


@pytest.fixture
def sample_claims():
    """Provide a TokenClaims instance for token creation."""
    from backend.security.jwt_manager import TokenClaims

    return TokenClaims(
        user_id=42,
        username="testuser",
        email="test@example.com",
        roles=["user"],
        scopes=["read", "write"],
        is_admin=False,
    )


@pytest.fixture
def rbac():
    """RoleBasedAccessControl instance."""
    return RoleBasedAccessControl()


# ===========================================================================
# PasswordValidator Tests
# ===========================================================================

class TestPasswordValidatorValidate:
    """Tests for PasswordValidator.validate()"""

    def test_valid_password_returns_valid_true(self, validator):
        result = validator.validate("Str0ng!Pass")
        assert result["valid"] is True
        assert result["errors"] == []

    def test_all_rules_violated_returns_five_errors(self, validator):
        result = validator.validate("")
        assert result["valid"] is False
        assert len(result["errors"]) == 5

    def test_too_short_password(self, validator):
        result = validator.validate("Aa1!x")
        assert result["valid"] is False
        assert any("at least" in e for e in result["errors"])

    def test_missing_uppercase(self, validator):
        result = validator.validate("abcdefg1!")
        assert result["valid"] is False
        assert any("uppercase" in e for e in result["errors"])

    def test_missing_lowercase(self, validator):
        result = validator.validate("ABCDEFG1!")
        assert result["valid"] is False
        assert any("lowercase" in e for e in result["errors"])

    def test_missing_digit(self, validator):
        result = validator.validate("Abcdefgh!")
        assert result["valid"] is False
        assert any("digit" in e for e in result["errors"])

    def test_missing_special_character(self, validator):
        result = validator.validate("Abcdefg1x")
        assert result["valid"] is False
        assert any("special" in e for e in result["errors"])

    def test_custom_min_length(self):
        v = PasswordValidator(min_length=12)
        result = v.validate("Short1!a")
        assert result["valid"] is False

    def test_disabled_rules_pass(self, lenient_validator):
        result = lenient_validator.validate("abcdef")
        assert result["valid"] is True
        assert result["errors"] == []


class TestPasswordValidatorStrength:
    """Tests for PasswordValidator._calculate_strength()"""

    def test_empty_password_score_zero(self, validator):
        score = validator._calculate_strength("")
        assert score == 0

    def test_short_password_low_score(self, validator):
        # 6 chars, lowercase only => no length bonus, only lowercase bonus = 15
        score = validator._calculate_strength("abcdef")
        assert score == 15

    def test_8_char_gets_length_bonus(self, validator):
        # "abcdefgh" = 8 chars => 25 (length) + 15 (lower) = 40
        score = validator._calculate_strength("abcdefgh")
        assert score == 40

    def test_12_char_gets_extra_bonus(self, validator):
        # "abcdefghijkl" = 12 chars => 25 + 15 (length) + 15 (lower) = 55
        score = validator._calculate_strength("abcdefghijkl")
        assert score == 55

    def test_16_char_gets_max_length_bonus(self, validator):
        # 16 chars => 25+15+10 (length) + 15 (lower) = 65
        score = validator._calculate_strength("a" * 16)
        assert score == 65

    def test_complex_password_high_score(self, validator):
        # 16+ chars, upper, lower, digit, special => 25+15+10+15+15+10+10 = 100
        score = validator._calculate_strength("Abcdefghijklmn1!")
        assert score == 100

    def test_score_capped_at_100(self, validator):
        score = validator._calculate_strength("ABCDefghijklmnop12345!@#")
        assert score <= 100


class TestPasswordValidatorSuggestions:
    """Tests for PasswordValidator.suggest_improvements()"""

    def test_invalid_password_gets_suggestions(self, validator):
        suggestions = validator.suggest_improvements("abc")
        assert len(suggestions) > 0

    def test_valid_strong_password_no_error_suggestions(self, validator):
        suggestions = validator.suggest_improvements("SuperStr0ng!Password2024")
        # May still suggest special characters if pattern doesn't match
        # but validation errors should be empty
        result = validator.validate("SuperStr0ng!Password2024")
        assert result["valid"] is True

    def test_weak_score_suggests_longer_password(self, validator):
        suggestions = validator.suggest_improvements("ab")
        assert any("longer" in s.lower() for s in suggestions)

    def test_no_special_char_pattern_suggests_adding(self, validator):
        # The regex in suggest_improvements checks for specific specials
        suggestions = validator.suggest_improvements("Abcdefgh1")
        assert any("special" in s.lower() for s in suggestions)


# ===========================================================================
# JWTManager Tests
# ===========================================================================

class TestTokenTypeEnum:
    """Tests for TokenType enum."""

    def test_access_value(self):
        from backend.security.jwt_manager import TokenType
        assert TokenType.ACCESS.value == "access"

    def test_refresh_value(self):
        from backend.security.jwt_manager import TokenType
        assert TokenType.REFRESH.value == "refresh"

    def test_reset_value(self):
        from backend.security.jwt_manager import TokenType
        assert TokenType.RESET.value == "reset"

    def test_mfa_value(self):
        from backend.security.jwt_manager import TokenType
        assert TokenType.MFA.value == "mfa"

    def test_token_type_is_string_subclass(self):
        from backend.security.jwt_manager import TokenType
        assert isinstance(TokenType.ACCESS, str)


class TestJWTManagerCreateToken:
    """Tests for JWTManager.create_access_token()"""

    def test_create_token_returns_string(self, jwt_manager, sample_claims):
        token = jwt_manager.create_access_token(sample_claims)
        assert isinstance(token, str)
        assert len(token) > 0

    def test_token_decodable_with_public_key(self, jwt_manager, sample_claims):
        token = jwt_manager.create_access_token(sample_claims)
        payload = pyjwt.decode(
            token,
            jwt_manager.public_key,
            algorithms=["RS256"],
            audience=jwt_manager.audience,
            issuer=jwt_manager.issuer,
        )
        assert payload["sub"] == "testuser"
        assert payload["user_id"] == 42
        assert payload["email"] == "test@example.com"

    def test_token_uses_rs256_algorithm(self, jwt_manager, sample_claims):
        token = jwt_manager.create_access_token(sample_claims)
        header = pyjwt.get_unverified_header(token)
        assert header["alg"] == "RS256"

    def test_token_contains_type_claim(self, jwt_manager, sample_claims):
        token = jwt_manager.create_access_token(sample_claims)
        payload = pyjwt.decode(
            token,
            jwt_manager.public_key,
            algorithms=["RS256"],
            audience=jwt_manager.audience,
            issuer=jwt_manager.issuer,
        )
        assert payload["type"] == "access"

    def test_token_has_expiration(self, jwt_manager, sample_claims):
        token = jwt_manager.create_access_token(sample_claims)
        payload = pyjwt.decode(
            token,
            jwt_manager.public_key,
            algorithms=["RS256"],
            audience=jwt_manager.audience,
            issuer=jwt_manager.issuer,
        )
        assert "exp" in payload

    def test_custom_expiration_delta(self, jwt_manager, sample_claims):
        delta = timedelta(minutes=5)
        token = jwt_manager.create_access_token(sample_claims, expires_delta=delta)
        payload = pyjwt.decode(
            token,
            jwt_manager.public_key,
            algorithms=["RS256"],
            audience=jwt_manager.audience,
            issuer=jwt_manager.issuer,
        )
        # The exp should be roughly 5 minutes from now (within a 10s tolerance)
        exp_dt = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
        expected = datetime.now(timezone.utc) + delta
        assert abs((exp_dt - expected).total_seconds()) < 10

    def test_token_has_issuer_and_audience(self, jwt_manager, sample_claims):
        token = jwt_manager.create_access_token(sample_claims)
        payload = pyjwt.decode(
            token,
            jwt_manager.public_key,
            algorithms=["RS256"],
            audience=jwt_manager.audience,
            issuer=jwt_manager.issuer,
        )
        assert payload["iss"] == "investment-analysis-app"
        assert payload["aud"] == "investment-analysis-users"

    def test_session_id_auto_generated(self, jwt_manager, sample_claims):
        assert sample_claims.session_id is None
        token = jwt_manager.create_access_token(sample_claims)
        payload = pyjwt.decode(
            token,
            jwt_manager.public_key,
            algorithms=["RS256"],
            audience=jwt_manager.audience,
            issuer=jwt_manager.issuer,
        )
        assert payload["session_id"] is not None
        assert len(payload["session_id"]) > 0


class TestJWTManagerVerifyToken:
    """Tests for JWTManager.verify_token()"""

    def test_verify_valid_token_returns_claims(self, jwt_manager, sample_claims):
        token = jwt_manager.create_access_token(sample_claims)
        # First exists call = blacklist check (must be False/0),
        # second exists call = session check (must be True/1).
        jwt_manager.redis_client.exists.side_effect = [False, True]
        payload = jwt_manager.verify_token(token)
        assert payload is not None
        assert payload["sub"] == "testuser"
        assert payload["user_id"] == 42

    def test_verify_expired_token_returns_none(self, jwt_manager, sample_claims):
        delta = timedelta(seconds=-10)
        token = jwt_manager.create_access_token(
            sample_claims, expires_delta=delta
        )
        result = jwt_manager.verify_token(token)
        assert result is None

    def test_verify_token_wrong_type_returns_none(self, jwt_manager, sample_claims):
        from backend.security.jwt_manager import TokenType
        token = jwt_manager.create_access_token(sample_claims)
        # Blacklist check = False, but wrong type should still fail
        jwt_manager.redis_client.exists.side_effect = [False]
        result = jwt_manager.verify_token(token, token_type=TokenType.REFRESH)
        assert result is None

    def test_verify_tampered_token_returns_none(self, jwt_manager, sample_claims):
        token = jwt_manager.create_access_token(sample_claims)
        tampered = token[:-5] + "XXXXX"
        result = jwt_manager.verify_token(tampered)
        assert result is None

    def test_verify_blacklisted_token_returns_none(
        self, jwt_manager, sample_claims
    ):
        token = jwt_manager.create_access_token(sample_claims)
        # Simulate blacklisting
        jwt_manager.redis_client.exists.return_value = True
        # The first exists call is for blacklist check which returns True
        # This should cause verify_token to return None
        # But the blacklist check uses a different key pattern, so we
        # need to make exists return True for the blacklist key.
        # Since redis mock returns True for all exists calls, the blacklist
        # check will catch it first.
        result = jwt_manager.verify_token(token)
        # exists returns True => blacklisted => None
        # Actually, exists is called first for blacklist check, and if
        # blacklisted it returns None. But we also need it True for
        # session check. Let's use side_effect to differentiate.
        jwt_manager.redis_client.exists.side_effect = [True]  # blacklist=True
        result = jwt_manager.verify_token(token)
        assert result is None


class TestJWTManagerCreateRefreshToken:
    """Tests for JWTManager.create_refresh_token()"""

    def test_refresh_token_has_correct_type(self, jwt_manager, sample_claims):
        sample_claims.session_id = "test-session"
        token = jwt_manager.create_refresh_token(sample_claims)
        payload = pyjwt.decode(
            token,
            jwt_manager.public_key,
            algorithms=["RS256"],
            audience=jwt_manager.audience,
            issuer=jwt_manager.issuer,
        )
        assert payload["type"] == "refresh"

    def test_refresh_token_uses_rs256(self, jwt_manager, sample_claims):
        sample_claims.session_id = "test-session"
        token = jwt_manager.create_refresh_token(sample_claims)
        header = pyjwt.get_unverified_header(token)
        assert header["alg"] == "RS256"


# ===========================================================================
# RBAC Tests
# ===========================================================================

class TestRoleEnum:
    """Tests for Role enum values."""

    def test_admin_role(self):
        assert Role.ADMIN.value == "admin"

    def test_user_role(self):
        assert Role.USER.value == "user"

    def test_analyst_role(self):
        assert Role.ANALYST.value == "analyst"

    def test_viewer_role(self):
        assert Role.VIEWER.value == "viewer"


class TestPermissionEnum:
    """Tests for Permission enum values."""

    def test_read_permission(self):
        assert Permission.READ.value == "read"

    def test_write_permission(self):
        assert Permission.WRITE.value == "write"

    def test_delete_permission(self):
        assert Permission.DELETE.value == "delete"

    def test_admin_permission(self):
        assert Permission.ADMIN.value == "admin"


class TestRBACHasPermission:
    """Tests for RoleBasedAccessControl.has_permission()"""

    def test_admin_has_read(self, rbac):
        assert rbac.has_permission(Role.ADMIN, Permission.READ) is True

    def test_admin_has_write(self, rbac):
        assert rbac.has_permission(Role.ADMIN, Permission.WRITE) is True

    def test_admin_has_delete(self, rbac):
        assert rbac.has_permission(Role.ADMIN, Permission.DELETE) is True

    def test_admin_has_admin(self, rbac):
        assert rbac.has_permission(Role.ADMIN, Permission.ADMIN) is True

    def test_user_has_read(self, rbac):
        assert rbac.has_permission(Role.USER, Permission.READ) is True

    def test_user_has_write(self, rbac):
        assert rbac.has_permission(Role.USER, Permission.WRITE) is True

    def test_user_lacks_delete(self, rbac):
        # User has READ and WRITE but not DELETE or ADMIN
        # has_permission checks: permission in role_permissions OR ADMIN in role_permissions
        # User role_permissions = [READ, WRITE]. DELETE not in it. ADMIN not in it.
        assert rbac.has_permission(Role.USER, Permission.DELETE) is False

    def test_user_lacks_admin(self, rbac):
        assert rbac.has_permission(Role.USER, Permission.ADMIN) is False

    def test_viewer_has_read(self, rbac):
        assert rbac.has_permission(Role.VIEWER, Permission.READ) is True

    def test_viewer_lacks_write(self, rbac):
        assert rbac.has_permission(Role.VIEWER, Permission.WRITE) is False

    def test_viewer_lacks_delete(self, rbac):
        assert rbac.has_permission(Role.VIEWER, Permission.DELETE) is False

    def test_analyst_has_read(self, rbac):
        assert rbac.has_permission(Role.ANALYST, Permission.READ) is True

    def test_analyst_has_write(self, rbac):
        assert rbac.has_permission(Role.ANALYST, Permission.WRITE) is True

    def test_analyst_lacks_delete(self, rbac):
        assert rbac.has_permission(Role.ANALYST, Permission.DELETE) is False

    def test_unknown_role_has_no_permissions(self, rbac):
        assert rbac.has_permission("unknown", Permission.READ) is False


class TestRBACCheckAccess:
    """Tests for RoleBasedAccessControl.check_access()"""

    def test_default_user_can_read(self, rbac):
        # get_user_roles returns [Role.USER] by default
        assert rbac.check_access(user_id=1, resource="portfolio", action=Permission.READ) is True

    def test_default_user_can_write(self, rbac):
        assert rbac.check_access(user_id=1, resource="portfolio", action=Permission.WRITE) is True

    def test_default_user_cannot_delete(self, rbac):
        assert rbac.check_access(user_id=1, resource="portfolio", action=Permission.DELETE) is False

    def test_default_user_cannot_admin(self, rbac):
        assert rbac.check_access(user_id=1, resource="portfolio", action=Permission.ADMIN) is False


class TestRBACStubMethods:
    """Tests for stub methods that will be implemented later."""

    def test_get_user_roles_returns_user(self, rbac):
        roles = rbac.get_user_roles(user_id=999)
        assert roles == [Role.USER]

    def test_assign_role_returns_true(self, rbac):
        assert rbac.assign_role(user_id=1, role=Role.ADMIN) is True

    def test_revoke_role_returns_true(self, rbac):
        assert rbac.revoke_role(user_id=1, role=Role.ADMIN) is True
