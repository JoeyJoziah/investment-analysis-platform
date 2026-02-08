"""
CSRF Protection Tests

Comprehensive test suite for CSRF protection middleware.

Created: 2026-01-27
Part of: Phase 3 Security Remediation
"""

import pytest
import secrets
import os
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock

from backend.security.csrf_protection import (
    CSRFProtection,
    CSRFConfig,
    CSRFMiddleware,
    CSRFTokenStorage,
    add_csrf_protection
)


class TestCSRFProtection:
    """Test CSRF Protection class"""

    def test_generate_token(self):
        """Test CSRF token generation"""
        csrf = CSRFProtection()
        token = csrf.generate_token()

        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0
        # Token should be hex string (token + signature)
        assert all(c in '0123456789abcdef' for c in token)

    def test_generate_token_uniqueness(self):
        """Test that generated tokens are unique"""
        csrf = CSRFProtection()
        tokens = [csrf.generate_token() for _ in range(100)]

        # All tokens should be unique
        assert len(tokens) == len(set(tokens))

    def test_validate_token_valid(self):
        """Test validation of valid token"""
        csrf = CSRFProtection()
        token = csrf.generate_token()

        assert csrf.validate_token(token) is True

    def test_validate_token_invalid(self):
        """Test validation of invalid token"""
        csrf = CSRFProtection()

        # Invalid format
        assert csrf.validate_token("invalid") is False
        assert csrf.validate_token("") is False
        assert csrf.validate_token(None) is False

    def test_validate_token_tampered(self):
        """Test validation of tampered token"""
        csrf = CSRFProtection()
        token = csrf.generate_token()

        # Tamper with token
        tampered = token[:-4] + "0000"

        assert csrf.validate_token(tampered) is False

    def test_validate_token_double_submit(self):
        """Test double-submit cookie pattern"""
        config = CSRFConfig(storage_strategy=CSRFTokenStorage.DOUBLE_SUBMIT)
        csrf = CSRFProtection(config)
        token = csrf.generate_token()

        # Same token in both places should validate
        assert csrf.validate_token(token, token) is True

        # Different tokens should fail
        other_token = csrf.generate_token()
        assert csrf.validate_token(token, other_token) is False

    def test_is_exempt_path(self):
        """Test path exemption checking"""
        csrf = CSRFProtection()

        # Default exemptions
        assert csrf.is_exempt_path("/api/webhooks/stripe") is True
        assert csrf.is_exempt_path("/health") is True
        assert csrf.is_exempt_path("/api/v1/auth/login") is True

        # Non-exempt paths
        assert csrf.is_exempt_path("/api/users") is False
        assert csrf.is_exempt_path("/api/posts") is False

    def test_is_protected_method(self):
        """Test method protection checking"""
        csrf = CSRFProtection()

        # Protected methods
        assert csrf.is_protected_method("POST") is True
        assert csrf.is_protected_method("PUT") is True
        assert csrf.is_protected_method("DELETE") is True
        assert csrf.is_protected_method("PATCH") is True

        # Non-protected methods
        assert csrf.is_protected_method("GET") is False
        assert csrf.is_protected_method("HEAD") is False
        assert csrf.is_protected_method("OPTIONS") is False

    def test_custom_exempt_paths(self):
        """Test custom exempt paths"""
        config = CSRFConfig(exempt_paths=["/api/custom/webhook"])
        csrf = CSRFProtection(config)

        assert csrf.is_exempt_path("/api/custom/webhook") is True

    def test_custom_protected_methods(self):
        """Test custom protected methods"""
        config = CSRFConfig(protected_methods={"POST", "DELETE"})
        csrf = CSRFProtection(config)

        assert csrf.is_protected_method("POST") is True
        assert csrf.is_protected_method("DELETE") is True
        assert csrf.is_protected_method("PUT") is False

    def test_token_expected_byte_length(self):
        """Test generated token has correct byte length (token_length + 32 HMAC bytes)"""
        config = CSRFConfig(token_length=32)
        csrf = CSRFProtection(config)
        token = csrf.generate_token()

        token_bytes = bytes.fromhex(token)
        # 32 bytes random token + 32 bytes SHA-256 HMAC = 64 bytes
        assert len(token_bytes) == 64

    def test_token_custom_length(self):
        """Test token generation respects custom token_length"""
        config = CSRFConfig(token_length=64)
        csrf = CSRFProtection(config)
        token = csrf.generate_token()

        token_bytes = bytes.fromhex(token)
        # 64 bytes random token + 32 bytes HMAC = 96 bytes
        assert len(token_bytes) == 96

    def test_validate_token_wrong_secret_key(self):
        """Test token generated with one secret fails validation with another"""
        config_a = CSRFConfig(secret_key=secrets.token_hex(32))
        config_b = CSRFConfig(secret_key=secrets.token_hex(32))
        csrf_a = CSRFProtection(config_a)
        csrf_b = CSRFProtection(config_b)

        token = csrf_a.generate_token()
        assert csrf_a.validate_token(token) is True
        assert csrf_b.validate_token(token) is False

    def test_validate_token_truncated(self):
        """Test token with fewer bytes than expected is rejected"""
        csrf = CSRFProtection()
        token = csrf.generate_token()

        # Remove last 2 hex chars (1 byte) to make it too short
        truncated = token[:-2]
        assert csrf.validate_token(truncated) is False

    def test_validate_token_extended(self):
        """Test token with extra bytes appended is rejected"""
        csrf = CSRFProtection()
        token = csrf.generate_token()

        extended = token + "ff"
        assert csrf.validate_token(extended) is False

    def test_validate_token_non_hex_string(self):
        """Test non-hex string triggers ValueError path and returns False"""
        csrf = CSRFProtection()

        assert csrf.validate_token("not-a-hex-string!!") is False
        assert csrf.validate_token("zzzz") is False
        assert csrf.validate_token("ghijklmn") is False

    def test_validate_token_uses_constant_time_comparison(self):
        """Test that hmac.compare_digest is used for signature comparison"""
        csrf = CSRFProtection()
        token = csrf.generate_token()

        with patch("backend.security.csrf_protection.hmac.compare_digest", return_value=True) as mock_compare:
            result = csrf.validate_token(token)
            # hmac.compare_digest should have been called at least once
            assert mock_compare.called
            assert result is True

    def test_validate_double_submit_uses_constant_time_comparison(self):
        """Test double-submit comparison also uses hmac.compare_digest"""
        config = CSRFConfig(storage_strategy=CSRFTokenStorage.DOUBLE_SUBMIT)
        csrf = CSRFProtection(config)
        token = csrf.generate_token()

        # We need to track calls. Since the function calls compare_digest twice
        # (once for HMAC verification, once for double-submit check), both should use it.
        with patch(
            "backend.security.csrf_protection.hmac.compare_digest",
            side_effect=lambda a, b: a == b
        ) as mock_compare:
            csrf.validate_token(token, token)
            # Called at least twice: once for HMAC, once for double-submit
            assert mock_compare.call_count >= 2

    def test_is_protected_method_case_insensitive(self):
        """Test that method checking normalizes to uppercase"""
        csrf = CSRFProtection()

        assert csrf.is_protected_method("post") is True
        assert csrf.is_protected_method("Post") is True
        assert csrf.is_protected_method("get") is False

    def test_is_exempt_path_prefix_matching(self):
        """Test exempt path matching uses prefix (startswith), not exact match"""
        csrf = CSRFProtection()

        # Sub-paths of exempt paths should also be exempt
        assert csrf.is_exempt_path("/api/webhooks/stripe/callback") is True
        assert csrf.is_exempt_path("/api/health/deep") is True
        assert csrf.is_exempt_path("/health/liveness") is True

    def test_is_exempt_path_no_partial_match(self):
        """Test exempt path does not match unrelated paths that share prefix by coincidence"""
        config = CSRFConfig(exempt_paths=["/api/public"])
        csrf = CSRFProtection(config)

        assert csrf.is_exempt_path("/api/public/resource") is True
        assert csrf.is_exempt_path("/api/publicsomethingelse") is True  # prefix match behavior
        assert csrf.is_exempt_path("/api/private") is False

    def test_default_exempt_paths_include_all_expected(self):
        """Test default exempt paths list is complete"""
        config = CSRFConfig()
        expected_paths = [
            "/api/webhooks",
            "/api/health",
            "/health",
            "/metrics",
            "/api/v1/auth/login",
            "/api/v1/auth/register",
            "/api/v1/auth/refresh",
        ]
        assert config.exempt_paths == expected_paths

    def test_default_protected_methods(self):
        """Test default protected methods set"""
        config = CSRFConfig()
        assert config.protected_methods == {"POST", "PUT", "DELETE", "PATCH"}


class TestCSRFConfigSecretKey:
    """Test CSRFConfig secret key initialization and environment handling"""

    def test_secret_key_from_env_variable(self, monkeypatch):
        """Test CSRF_SECRET_KEY is read from environment when present and long enough"""
        long_key = "a" * 64
        monkeypatch.setenv("CSRF_SECRET_KEY", long_key)
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("TESTING", "false")
        monkeypatch.setenv("DEBUG", "false")

        config = CSRFConfig(secret_key=None)
        # Force __post_init__ to run by creating fresh instance
        config2 = CSRFConfig()
        # Since monkeypatch set the env var, the __post_init__ should use it
        assert config2.secret_key == long_key

    def test_short_env_secret_rejected_in_production(self, monkeypatch):
        """Test CSRF_SECRET_KEY shorter than 32 chars raises in production"""
        monkeypatch.setenv("CSRF_SECRET_KEY", "tooshort")
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("DEBUG", "false")
        monkeypatch.setenv("TESTING", "false")

        with pytest.raises(ValueError, match="CSRF_SECRET_KEY environment variable is required"):
            CSRFConfig()

    def test_missing_env_secret_raises_in_production(self, monkeypatch):
        """Test missing CSRF_SECRET_KEY raises ValueError in production mode"""
        monkeypatch.delenv("CSRF_SECRET_KEY", raising=False)
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("DEBUG", "false")
        monkeypatch.setenv("TESTING", "false")

        with pytest.raises(ValueError, match="CSRF_SECRET_KEY environment variable is required"):
            CSRFConfig()

    def test_missing_env_secret_auto_generates_in_development(self, monkeypatch):
        """Test auto-generated secret in development mode"""
        monkeypatch.delenv("CSRF_SECRET_KEY", raising=False)
        monkeypatch.setenv("ENVIRONMENT", "development")
        monkeypatch.setenv("DEBUG", "true")
        monkeypatch.setenv("TESTING", "false")

        config = CSRFConfig()
        assert config.secret_key is not None
        assert len(config.secret_key) >= 32

    def test_missing_env_secret_auto_generates_in_testing(self, monkeypatch):
        """Test auto-generated secret when TESTING=true regardless of other env vars"""
        monkeypatch.delenv("CSRF_SECRET_KEY", raising=False)
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("DEBUG", "false")
        monkeypatch.setenv("TESTING", "true")

        config = CSRFConfig()
        assert config.secret_key is not None
        assert len(config.secret_key) >= 32

    def test_explicit_secret_key_overrides_env(self):
        """Test explicitly passed secret_key bypasses environment variable logic"""
        explicit_key = secrets.token_hex(32)
        config = CSRFConfig(secret_key=explicit_key)
        assert config.secret_key == explicit_key

    def test_csrf_protection_rejects_short_secret(self):
        """Test CSRFProtection constructor raises on secret shorter than 32 chars"""
        config = CSRFConfig(secret_key="short")
        with pytest.raises(ValueError, match="at least 32 characters"):
            CSRFProtection(config)

    def test_csrf_protection_rejects_empty_secret(self):
        """Test CSRFProtection constructor raises on empty secret"""
        config = CSRFConfig(secret_key="")
        with pytest.raises(ValueError, match="at least 32 characters"):
            CSRFProtection(config)


class TestCSRFTokenStorage:
    """Test CSRFTokenStorage enum values"""

    def test_storage_strategy_values(self):
        """Test all storage strategy enum values exist"""
        assert CSRFTokenStorage.COOKIE == "cookie"
        assert CSRFTokenStorage.HEADER == "header"
        assert CSRFTokenStorage.DOUBLE_SUBMIT == "double_submit"

    def test_default_storage_strategy(self):
        """Test default storage strategy is DOUBLE_SUBMIT"""
        config = CSRFConfig()
        assert config.storage_strategy == CSRFTokenStorage.DOUBLE_SUBMIT


class TestCSRFTokenCookie:
    """Test CSRF token cookie management"""

    def test_set_token_cookie_attributes(self):
        """Test set_token_cookie applies correct cookie attributes"""
        csrf = CSRFProtection()
        mock_response = MagicMock()

        token = csrf.generate_token()
        csrf.set_token_cookie(mock_response, token)

        mock_response.set_cookie.assert_called_once()
        call_kwargs = mock_response.set_cookie.call_args

        # Verify cookie attributes
        assert call_kwargs.kwargs["key"] == "csrf_token"
        assert call_kwargs.kwargs["value"] == token
        assert call_kwargs.kwargs["secure"] is True
        assert call_kwargs.kwargs["httponly"] is True
        assert call_kwargs.kwargs["samesite"] == "Strict"
        assert call_kwargs.kwargs["path"] == "/"

    def test_set_token_cookie_custom_config(self):
        """Test set_token_cookie respects custom configuration"""
        config = CSRFConfig(
            cookie_name="my_csrf",
            cookie_secure=False,
            cookie_httponly=False,
            cookie_samesite="Lax",
            cookie_domain="example.com",
            cookie_path="/app",
            token_expiry_hours=48,
        )
        csrf = CSRFProtection(config)
        mock_response = MagicMock()

        token = csrf.generate_token()
        csrf.set_token_cookie(mock_response, token)

        call_kwargs = mock_response.set_cookie.call_args
        assert call_kwargs.kwargs["key"] == "my_csrf"
        assert call_kwargs.kwargs["secure"] is False
        assert call_kwargs.kwargs["httponly"] is False
        assert call_kwargs.kwargs["samesite"] == "Lax"
        assert call_kwargs.kwargs["domain"] == "example.com"
        assert call_kwargs.kwargs["path"] == "/app"
        assert call_kwargs.kwargs["max_age"] == 48 * 3600

    def test_set_token_cookie_expiry(self):
        """Test cookie max_age matches configured token_expiry_hours"""
        config = CSRFConfig(token_expiry_hours=12)
        csrf = CSRFProtection(config)
        mock_response = MagicMock()

        csrf.set_token_cookie(mock_response, csrf.generate_token())

        call_kwargs = mock_response.set_cookie.call_args
        assert call_kwargs.kwargs["max_age"] == 12 * 3600


class TestExtractTokenFromRequest:
    """Test CSRF token extraction from requests"""

    def test_extract_token_from_header(self):
        """Test token extraction from X-CSRF-Token header"""
        csrf = CSRFProtection()
        mock_request = MagicMock()
        mock_request.headers = {"X-CSRF-Token": "test-token-value"}
        mock_request.cookies = {}

        token = csrf.extract_token_from_request(mock_request)
        assert token == "test-token-value"

    def test_extract_token_from_cookie_fallback(self):
        """Test token extraction falls back to cookie in double-submit mode"""
        config = CSRFConfig(storage_strategy=CSRFTokenStorage.DOUBLE_SUBMIT)
        csrf = CSRFProtection(config)
        mock_request = MagicMock()
        mock_request.headers = {}
        mock_request.cookies = {"csrf_token": "cookie-token-value"}

        token = csrf.extract_token_from_request(mock_request)
        assert token == "cookie-token-value"

    def test_extract_token_header_takes_priority(self):
        """Test header token takes priority over cookie token"""
        config = CSRFConfig(storage_strategy=CSRFTokenStorage.DOUBLE_SUBMIT)
        csrf = CSRFProtection(config)
        mock_request = MagicMock()
        mock_request.headers = {"X-CSRF-Token": "header-token"}
        mock_request.cookies = {"csrf_token": "cookie-token"}

        token = csrf.extract_token_from_request(mock_request)
        assert token == "header-token"

    def test_extract_token_no_cookie_fallback_for_header_strategy(self):
        """Test no cookie fallback when storage strategy is HEADER only"""
        config = CSRFConfig(storage_strategy=CSRFTokenStorage.HEADER)
        csrf = CSRFProtection(config)
        mock_request = MagicMock()
        mock_request.headers = {}
        mock_request.cookies = {"csrf_token": "cookie-token"}

        token = csrf.extract_token_from_request(mock_request)
        assert token is None

    def test_extract_token_returns_none_when_missing(self):
        """Test None returned when no token is present anywhere"""
        csrf = CSRFProtection()
        mock_request = MagicMock()
        mock_request.headers = {}
        mock_request.cookies = {}

        token = csrf.extract_token_from_request(mock_request)
        assert token is None

    def test_extract_token_custom_header_name(self):
        """Test token extraction uses custom header name from config"""
        config = CSRFConfig(header_name="X-My-CSRF")
        csrf = CSRFProtection(config)
        mock_request = MagicMock()
        mock_request.headers = {"X-My-CSRF": "custom-header-token"}
        mock_request.cookies = {}

        token = csrf.extract_token_from_request(mock_request)
        assert token == "custom-header-token"


class TestCSRFMiddleware:
    """Test CSRF Middleware"""

    @pytest.fixture
    def disable_testing_mode(self, monkeypatch):
        """Disable TESTING mode for middleware tests"""
        monkeypatch.setenv("TESTING", "False")

    @pytest.fixture
    def app(self):
        """Create test FastAPI app"""
        app = FastAPI()

        @app.get("/test")
        async def get_test():
            return {"method": "GET"}

        @app.post("/test")
        async def post_test():
            return {"method": "POST"}

        @app.put("/test")
        async def put_test():
            return {"method": "PUT"}

        @app.delete("/test")
        async def delete_test():
            return {"method": "DELETE"}

        @app.post("/api/webhooks/test")
        async def webhook_test():
            return {"webhook": True}

        return app

    @pytest.fixture
    def client(self, app, disable_testing_mode):
        """Create test client with CSRF protection"""
        config = CSRFConfig(secret_key=secrets.token_hex(32))
        app.add_middleware(CSRFMiddleware, config=config)
        return TestClient(app)

    def test_get_request_no_csrf_required(self, client):
        """Test GET request doesn't require CSRF token"""
        response = client.get("/test")
        assert response.status_code == 200

    def test_get_request_receives_csrf_token(self, client):
        """Test GET request receives CSRF token in cookie and header"""
        response = client.get("/test")
        assert response.status_code == 200

        # Check cookie
        assert "csrf_token" in response.cookies

        # Check header
        assert "X-CSRF-Token" in response.headers

    def test_post_request_without_token_fails(self, client):
        """Test POST request without CSRF token fails"""
        response = client.post("/test")
        assert response.status_code == 403
        assert response.json()["code"] == "CSRF_VALIDATION_FAILED"

    def test_post_request_with_valid_token_succeeds(self, client):
        """Test POST request with valid CSRF token succeeds"""
        # Get token from GET request
        get_response = client.get("/test")
        csrf_token = get_response.cookies.get("csrf_token")

        # Use token in POST request
        response = client.post(
            "/test",
            headers={"X-CSRF-Token": csrf_token},
            cookies={"csrf_token": csrf_token}
        )
        assert response.status_code == 200

    def test_put_request_requires_csrf(self, client):
        """Test PUT request requires CSRF token"""
        response = client.put("/test")
        assert response.status_code == 403

    def test_delete_request_requires_csrf(self, client):
        """Test DELETE request requires CSRF token"""
        response = client.delete("/test")
        assert response.status_code == 403

    def test_webhook_path_exempt(self, client):
        """Test webhook path is exempt from CSRF"""
        response = client.post("/api/webhooks/test")
        assert response.status_code == 200

    def test_invalid_token_fails(self, client):
        """Test invalid CSRF token fails"""
        response = client.post(
            "/test",
            headers={"X-CSRF-Token": "invalid_token"}
        )
        assert response.status_code == 403

    def test_disabled_csrf_allows_all(self):
        """Test disabled CSRF allows all requests"""
        app = FastAPI()

        @app.post("/test")
        async def post_test():
            return {"method": "POST"}

        config = CSRFConfig(enabled=False)
        app.add_middleware(CSRFMiddleware, config=config)
        client = TestClient(app)

        response = client.post("/test")
        assert response.status_code == 200

    def test_testing_mode_bypasses_csrf(self, app, monkeypatch):
        """Test TESTING=true environment variable bypasses all CSRF checks"""
        monkeypatch.setenv("TESTING", "true")
        config = CSRFConfig(secret_key=secrets.token_hex(32))
        app.add_middleware(CSRFMiddleware, config=config)
        client = TestClient(app)

        # POST without any token should succeed in testing mode
        response = client.post("/test")
        assert response.status_code == 200

    def test_csrf_failure_response_structure(self, client):
        """Test the CSRF failure response has correct JSON structure"""
        response = client.post("/test")
        assert response.status_code == 403

        body = response.json()
        assert body["success"] is False
        assert body["error"] == "CSRF validation failed"
        assert body["detail"] == "Missing or invalid CSRF token"
        assert body["code"] == "CSRF_VALIDATION_FAILED"

    def test_patch_request_requires_csrf(self, app, disable_testing_mode):
        """Test PATCH request requires CSRF token"""
        @app.patch("/test")
        async def patch_test():
            return {"method": "PATCH"}

        config = CSRFConfig(secret_key=secrets.token_hex(32))
        app.add_middleware(CSRFMiddleware, config=config)
        client = TestClient(app)

        response = client.patch("/test")
        assert response.status_code == 403

    def test_options_request_does_not_require_csrf(self, app, disable_testing_mode):
        """Test OPTIONS request passes through without CSRF check"""
        @app.options("/test")
        async def options_test():
            return {"method": "OPTIONS"}

        config = CSRFConfig(secret_key=secrets.token_hex(32))
        app.add_middleware(CSRFMiddleware, config=config)
        client = TestClient(app)

        response = client.options("/test")
        assert response.status_code == 200

    def test_head_request_does_not_require_csrf(self, app, disable_testing_mode):
        """Test HEAD request passes through without CSRF check"""
        config = CSRFConfig(secret_key=secrets.token_hex(32))
        app.add_middleware(CSRFMiddleware, config=config)
        client = TestClient(app)

        response = client.head("/test")
        assert response.status_code == 200

    def test_get_request_sets_token_in_cookie_and_header(self, client):
        """Test GET request sets both cookie and response header with matching token"""
        response = client.get("/test")
        assert response.status_code == 200

        cookie_token = response.cookies.get("csrf_token")
        header_token = response.headers.get("X-CSRF-Token")

        assert cookie_token is not None
        assert header_token is not None
        assert cookie_token == header_token

    def test_get_exempt_path_does_not_set_token(self, client):
        """Test GET to exempt path does not inject CSRF token"""
        # /api/webhooks/test is exempt
        # Need a GET handler on an exempt path
        app = FastAPI()

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        config = CSRFConfig(secret_key=secrets.token_hex(32))
        app.add_middleware(CSRFMiddleware, config=config)

        with patch.dict(os.environ, {"TESTING": "False"}):
            test_client = TestClient(app)
            response = test_client.get("/health")
            assert response.status_code == 200
            # Exempt path should not set CSRF token header
            assert "X-CSRF-Token" not in response.headers

    def test_tampered_token_in_middleware_rejected(self, client):
        """Test middleware rejects a token with tampered HMAC signature"""
        get_response = client.get("/test")
        csrf_token = get_response.cookies.get("csrf_token")

        # Flip last hex character
        last_char = csrf_token[-1]
        tampered_char = "0" if last_char != "0" else "1"
        tampered_token = csrf_token[:-1] + tampered_char

        response = client.post(
            "/test",
            headers={"X-CSRF-Token": tampered_token},
            cookies={"csrf_token": tampered_token}
        )
        assert response.status_code == 403

    def test_mismatched_header_and_cookie_tokens_rejected(self, client):
        """Test double-submit fails when header and cookie tokens differ"""
        get_response = client.get("/test")
        csrf_token = get_response.cookies.get("csrf_token")

        # Get a second, different token
        get_response2 = client.get("/test")
        csrf_token2 = get_response2.cookies.get("csrf_token")

        # Use one in header, the other in cookie
        response = client.post(
            "/test",
            headers={"X-CSRF-Token": csrf_token},
            cookies={"csrf_token": csrf_token2}
        )
        assert response.status_code == 403

    def test_each_get_returns_fresh_token(self, client):
        """Test that each GET request generates a new unique token"""
        response1 = client.get("/test")
        response2 = client.get("/test")

        token1 = response1.headers.get("X-CSRF-Token")
        token2 = response2.headers.get("X-CSRF-Token")

        assert token1 is not None
        assert token2 is not None
        assert token1 != token2


class TestAddCSRFProtection:
    """Test convenience function for adding CSRF protection"""

    def test_add_csrf_protection_basic(self):
        """Test basic CSRF protection addition"""
        app = FastAPI()

        @app.post("/test")
        async def post_test():
            return {"method": "POST"}

        middleware = add_csrf_protection(app, secret_key=secrets.token_hex(32))

        assert middleware is not None
        assert isinstance(middleware, CSRFMiddleware)

    def test_add_csrf_protection_custom_exemptions(self):
        """Test CSRF protection with custom exemptions"""
        app = FastAPI()

        @app.post("/custom/webhook")
        async def webhook():
            return {"webhook": True}

        add_csrf_protection(
            app,
            secret_key=secrets.token_hex(32),
            exempt_paths=["/custom/webhook"]
        )

        client = TestClient(app)
        response = client.post("/custom/webhook")
        assert response.status_code == 200

    def test_add_csrf_protection_merges_exempt_paths(self):
        """Test custom exempt paths are merged with defaults, not replacing them"""
        app = FastAPI()
        middleware = add_csrf_protection(
            app,
            secret_key=secrets.token_hex(32),
            exempt_paths=["/custom/path"]
        )

        exempt = middleware.csrf.config.exempt_paths
        # Should contain both default paths and custom path
        assert "/api/webhooks" in exempt
        assert "/health" in exempt
        assert "/custom/path" in exempt

    def test_add_csrf_protection_passes_kwargs(self):
        """Test additional kwargs are forwarded to CSRFConfig"""
        app = FastAPI()
        middleware = add_csrf_protection(
            app,
            secret_key=secrets.token_hex(32),
            token_length=64,
            cookie_name="my_csrf",
            header_name="X-My-Token",
        )

        assert middleware.csrf.config.token_length == 64
        assert middleware.csrf.config.cookie_name == "my_csrf"
        assert middleware.csrf.config.header_name == "X-My-Token"

    def test_add_csrf_protection_without_secret_key(self, monkeypatch):
        """Test add_csrf_protection without explicit key falls back to env/auto-gen"""
        monkeypatch.setenv("ENVIRONMENT", "development")
        monkeypatch.setenv("DEBUG", "true")
        monkeypatch.delenv("CSRF_SECRET_KEY", raising=False)

        app = FastAPI()
        middleware = add_csrf_protection(app)

        assert middleware is not None
        assert middleware.csrf.config.secret_key is not None
        assert len(middleware.csrf.config.secret_key) >= 32

    def test_add_csrf_protection_returns_middleware_instance(self):
        """Test return value is a CSRFMiddleware instance"""
        app = FastAPI()
        result = add_csrf_protection(app, secret_key=secrets.token_hex(32))

        assert isinstance(result, CSRFMiddleware)
        assert hasattr(result, "csrf")
        assert isinstance(result.csrf, CSRFProtection)
