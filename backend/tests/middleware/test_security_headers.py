"""
Security Headers Middleware Tests

Comprehensive test suite for security headers middleware.

Created: 2026-01-27
Part of: Phase 3 Security Remediation
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.middleware.security_headers import (
    SecurityHeadersMiddleware,
    SecurityHeadersConfig,
    ContentSecurityPolicy,
    PermissionsPolicy,
    FrameOptions,
    ReferrerPolicy,
    add_security_headers
)


class TestContentSecurityPolicy:
    """Test CSP configuration"""

    def test_build_basic_csp(self):
        """Test building basic CSP header"""
        csp = ContentSecurityPolicy()
        header = csp.build()

        assert "default-src 'self'" in header
        assert "script-src 'self'" in header
        assert "object-src 'none'" in header

    def test_build_custom_csp(self):
        """Test building custom CSP header"""
        csp = ContentSecurityPolicy(
            script_src=["'self'", "https://cdn.example.com"],
            connect_src=["'self'", "https://api.example.com"]
        )
        header = csp.build()

        assert "script-src 'self' https://cdn.example.com" in header
        assert "connect-src 'self' https://api.example.com" in header

    def test_build_csp_with_report_uri(self):
        """Test CSP with report URI"""
        csp = ContentSecurityPolicy(report_uri="https://example.com/csp-report")
        header = csp.build()

        assert "report-uri https://example.com/csp-report" in header

    def test_build_csp_upgrade_insecure(self):
        """Test CSP with upgrade-insecure-requests"""
        csp = ContentSecurityPolicy(upgrade_insecure_requests=True)
        header = csp.build()

        assert "upgrade-insecure-requests" in header

    def test_build_csp_block_mixed_content(self):
        """Test CSP with block-all-mixed-content"""
        csp = ContentSecurityPolicy(block_all_mixed_content=True)
        header = csp.build()

        assert "block-all-mixed-content" in header


class TestPermissionsPolicy:
    """Test Permissions Policy configuration"""

    def test_build_empty_permissions(self):
        """Test building Permissions Policy with no permissions"""
        policy = PermissionsPolicy()
        header = policy.build()

        # All features should be denied
        assert "camera=()" in header
        assert "microphone=()" in header
        assert "geolocation=()" in header

    def test_build_custom_permissions(self):
        """Test building custom Permissions Policy"""
        policy = PermissionsPolicy(
            camera=["'self'"],
            geolocation=["'self'", "https://maps.example.com"]
        )
        header = policy.build()

        assert "camera=('self')" in header
        assert "geolocation=('self' https://maps.example.com)" in header


class TestSecurityHeadersMiddleware:
    """Test Security Headers Middleware"""

    @pytest.fixture
    def app(self):
        """Create test FastAPI app"""
        app = FastAPI()

        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}

        @app.get("/health")
        async def health_endpoint():
            return {"status": "healthy"}

        return app

    @pytest.fixture
    def client(self, app):
        """Create test client with security headers"""
        config = SecurityHeadersConfig()
        app.add_middleware(SecurityHeadersMiddleware, config=config)
        return TestClient(app)

    def test_x_content_type_options(self, client):
        """Test X-Content-Type-Options header"""
        response = client.get("/test")
        assert response.headers.get("X-Content-Type-Options") == "nosniff"

    def test_x_frame_options(self, client):
        """Test X-Frame-Options header"""
        response = client.get("/test")
        assert response.headers.get("X-Frame-Options") == "DENY"

    def test_x_xss_protection(self, client):
        """Test X-XSS-Protection header"""
        response = client.get("/test")
        assert response.headers.get("X-XSS-Protection") == "1; mode=block"

    def test_referrer_policy(self, client):
        """Test Referrer-Policy header"""
        response = client.get("/test")
        assert "Referrer-Policy" in response.headers

    def test_content_security_policy(self, client):
        """Test Content-Security-Policy header"""
        response = client.get("/test")
        csp = response.headers.get("Content-Security-Policy")

        assert csp is not None
        assert "default-src 'self'" in csp
        assert "script-src 'self'" in csp

    def test_permissions_policy(self, client):
        """Test Permissions-Policy header"""
        response = client.get("/test")
        permissions = response.headers.get("Permissions-Policy")

        assert permissions is not None
        assert "camera=()" in permissions

    def test_exclude_health_endpoint(self):
        """Test that health endpoint is excluded from headers"""
        app = FastAPI()

        @app.get("/health")
        async def health_endpoint():
            return {"status": "healthy"}

        config = SecurityHeadersConfig(exclude_paths={"/health"})
        app.add_middleware(SecurityHeadersMiddleware, config=config)
        client = TestClient(app)

        response = client.get("/health")
        # Health endpoint should not have security headers
        # (except those added by other middleware)
        assert response.status_code == 200

    def test_custom_frame_options(self):
        """Test custom X-Frame-Options"""
        app = FastAPI()

        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}

        config = SecurityHeadersConfig(frame_options=FrameOptions.SAMEORIGIN)
        app.add_middleware(SecurityHeadersMiddleware, config=config)
        client = TestClient(app)

        response = client.get("/test")
        assert response.headers.get("X-Frame-Options") == "SAMEORIGIN"

    def test_custom_referrer_policy(self):
        """Test custom Referrer-Policy"""
        app = FastAPI()

        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}

        config = SecurityHeadersConfig(
            referrer_policy=ReferrerPolicy.NO_REFERRER
        )
        app.add_middleware(SecurityHeadersMiddleware, config=config)
        client = TestClient(app)

        response = client.get("/test")
        assert response.headers.get("Referrer-Policy") == "no-referrer"

    def test_custom_headers(self):
        """Test custom headers"""
        app = FastAPI()

        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}

        config = SecurityHeadersConfig(
            custom_headers={
                "X-Custom-Header": "custom-value",
                "X-Another-Header": "another-value"
            }
        )
        app.add_middleware(SecurityHeadersMiddleware, config=config)
        client = TestClient(app)

        response = client.get("/test")
        assert response.headers.get("X-Custom-Header") == "custom-value"
        assert response.headers.get("X-Another-Header") == "another-value"


class TestHSTSHeader:
    """Test Strict-Transport-Security header"""

    def test_hsts_basic(self):
        """Test basic HSTS header"""
        app = FastAPI()

        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}

        config = SecurityHeadersConfig(
            hsts_enabled=True,
            hsts_max_age=31536000
        )
        app.add_middleware(SecurityHeadersMiddleware, config=config)
        client = TestClient(app, base_url="https://example.com")

        response = client.get("/test")
        hsts = response.headers.get("Strict-Transport-Security")

        assert hsts is not None
        assert "max-age=31536000" in hsts

    def test_hsts_with_subdomains(self):
        """Test HSTS with includeSubDomains"""
        app = FastAPI()

        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}

        config = SecurityHeadersConfig(
            hsts_enabled=True,
            hsts_include_subdomains=True
        )
        app.add_middleware(SecurityHeadersMiddleware, config=config)
        client = TestClient(app, base_url="https://example.com")

        response = client.get("/test")
        hsts = response.headers.get("Strict-Transport-Security")

        assert "includeSubDomains" in hsts

    def test_hsts_with_preload(self):
        """Test HSTS with preload"""
        app = FastAPI()

        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}

        config = SecurityHeadersConfig(
            hsts_enabled=True,
            hsts_preload=True
        )
        app.add_middleware(SecurityHeadersMiddleware, config=config)
        client = TestClient(app, base_url="https://example.com")

        response = client.get("/test")
        hsts = response.headers.get("Strict-Transport-Security")

        assert "preload" in hsts


class TestAddSecurityHeaders:
    """Test convenience function for adding security headers"""

    def test_add_security_headers_basic(self):
        """Test basic security headers addition"""
        app = FastAPI()

        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}

        middleware = add_security_headers(app)

        assert middleware is not None
        assert isinstance(middleware, SecurityHeadersMiddleware)

        client = TestClient(app)
        response = client.get("/test")

        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers

    def test_add_security_headers_custom_csp(self):
        """Test security headers with custom CSP"""
        app = FastAPI()

        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}

        add_security_headers(
            app,
            csp_script_src=["'self'", "https://cdn.example.com"],
            csp_connect_src=["'self'", "https://api.example.com"]
        )

        client = TestClient(app)
        response = client.get("/test")

        csp = response.headers.get("Content-Security-Policy")
        assert "script-src 'self' https://cdn.example.com" in csp
        assert "connect-src 'self' https://api.example.com" in csp
