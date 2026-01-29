"""
CSRF Protection Tests

Comprehensive test suite for CSRF protection middleware.

Created: 2026-01-27
Part of: Phase 3 Security Remediation
"""

import pytest
import secrets
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

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
        assert csrf.is_exempt_path("/api/auth/login") is True

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


class TestCSRFMiddleware:
    """Test CSRF Middleware"""

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
    def client(self, app):
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
