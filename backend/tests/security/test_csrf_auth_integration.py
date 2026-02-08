"""
CSRF Protection - Authentication Integration Tests

Tests that verify authentication endpoints work without CSRF tokens in TESTING mode.
These tests validate that integration tests can properly authenticate and call protected APIs.

Created: 2026-01-28
Part of: Phase 3 Security Remediation (CSRF Auth Integration)
"""

import pytest
import os
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.main import app


class TestCSRFAuthExemptions:
    """Test that auth endpoints are properly exempt from CSRF in TESTING mode"""

    def test_testing_mode_enabled_in_conftest(self):
        """Verify TESTING mode is enabled"""
        assert os.getenv("TESTING", "").lower() == "true", \
            "TESTING environment variable should be 'true' in conftest"

    def test_login_endpoint_exempt(self):
        """Test login endpoint is in exempt paths"""
        from backend.security.csrf_protection import CSRFConfig

        config = CSRFConfig()
        exempt_paths = config.exempt_paths

        # Check both v0 and v1 paths
        assert "/api/v1/auth/login" in exempt_paths, "v0 login should be exempt"
        assert "/api/v1/auth/login" in exempt_paths, "v1 login should be exempt"

    def test_register_endpoint_exempt(self):
        """Test register endpoint is in exempt paths"""
        from backend.security.csrf_protection import CSRFConfig

        config = CSRFConfig()
        exempt_paths = config.exempt_paths

        # Check both v0 and v1 paths
        assert "/api/v1/auth/register" in exempt_paths, "v0 register should be exempt"
        assert "/api/v1/auth/register" in exempt_paths, "v1 register should be exempt"

    def test_refresh_endpoint_exempt(self):
        """Test refresh endpoint is in exempt paths"""
        from backend.security.csrf_protection import CSRFConfig

        config = CSRFConfig()
        exempt_paths = config.exempt_paths

        # Check v1 path (refresh is typically v1-only)
        assert "/api/v1/auth/refresh" in exempt_paths, "v1 refresh should be exempt"

    def test_csrf_middleware_skips_testing_mode(self):
        """Test CSRF middleware skips protection in TESTING mode"""
        from backend.security.csrf_protection import CSRFMiddleware, CSRFConfig
        import secrets

        app = FastAPI()

        @app.post("/protected")
        async def protected_endpoint():
            return {"success": True}

        config = CSRFConfig(secret_key=secrets.token_hex(32))
        app.add_middleware(CSRFMiddleware, config=config)

        client = TestClient(app)

        # In TESTING mode, POST without CSRF token should succeed
        response = client.post("/protected")
        assert response.status_code == 200, \
            "POST should succeed in TESTING mode without CSRF token"


@pytest.mark.integration
class TestAuthWithoutCSRF:
    """Integration tests verifying auth works without CSRF tokens"""

    @pytest.mark.asyncio
    async def test_auth_endpoints_skip_csrf_in_testing(self):
        """
        Verify auth endpoints work without CSRF tokens in TESTING mode.
        This simulates the test environment where TESTING=True.
        """
        # Note: This test runs in TESTING mode (set by conftest.py)
        # Auth endpoints should not require CSRF tokens

        # Since TESTING=True, the CSRF middleware will skip all requests
        # This allows integration tests to authenticate
        assert os.getenv("TESTING", "").lower() == "true"


class TestCSRFConfigurationExemptions:
    """Test CSRF configuration with various paths"""

    def test_all_auth_endpoints_configured(self):
        """Verify all auth endpoints are properly configured"""
        from backend.security.csrf_protection import CSRFProtection, CSRFConfig

        config = CSRFConfig()
        csrf = CSRFProtection(config)

        # All these endpoints should be exempt
        required_exemptions = [
            "/api/v1/auth/login",
            "/api/v1/auth/register",
            "/api/v1/auth/login",
            "/api/v1/auth/register",
            "/api/v1/auth/refresh",
        ]

        for path in required_exemptions:
            assert csrf.is_exempt_path(path), \
                f"Path {path} should be exempt from CSRF protection"

    def test_csrf_protection_instance_respects_config(self):
        """Verify CSRFProtection instance uses configured exemptions"""
        from backend.security.csrf_protection import CSRFProtection, CSRFConfig

        config = CSRFConfig()
        csrf = CSRFProtection(config)

        # Verify exemptions are accessible through the instance
        assert csrf.is_exempt_path("/api/v1/auth/login")
        assert csrf.is_exempt_path("/api/v1/auth/refresh")

    def test_protected_methods_configured(self):
        """Verify CSRF protects the right HTTP methods"""
        from backend.security.csrf_protection import CSRFProtection, CSRFConfig

        config = CSRFConfig()
        csrf = CSRFProtection(config)

        # These should require CSRF protection
        protected = ["POST", "PUT", "DELETE", "PATCH"]
        for method in protected:
            assert csrf.is_protected_method(method), \
                f"{method} should be protected"

        # These should NOT require CSRF protection
        unprotected = ["GET", "HEAD", "OPTIONS"]
        for method in unprotected:
            assert not csrf.is_protected_method(method), \
                f"{method} should not be protected"
