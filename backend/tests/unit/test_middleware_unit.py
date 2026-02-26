"""
Unit tests for backend/middleware/ module.

Pure unit tests using unittest.mock for ASGI apps, requests, and responses.
Tests internal logic, edge cases, data classes, enums, and error paths
that are NOT covered by the integration-level tests in backend/tests/middleware/.

Covers:
- backend/middleware/error_handler.py
- backend/middleware/request_size_limiter.py
- backend/middleware/security_headers.py
- backend/middleware/stack.py
- backend/middleware/response_optimizer.py
"""

import os
os.environ["TESTING"] = "True"
os.environ["DEBUG"] = "True"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

import hashlib
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from fastapi import FastAPI
from fastapi.exceptions import HTTPException, RequestValidationError
from pydantic import ValidationError, BaseModel
from starlette.datastructures import Headers

# ---- imports under test ----
from backend.middleware.error_handler import (
    http_exception_handler,
    validation_exception_handler,
    general_exception_handler,
    register_exception_handlers,
)
from backend.middleware.request_size_limiter import (
    ContentType,
    RequestSizeLimits,
    RequestSizeLimiterMiddleware,
)
from backend.middleware.security_headers import (
    FrameOptions,
    ReferrerPolicy,
    ContentSecurityPolicy,
    PermissionsPolicy,
    SecurityHeadersConfig,
    SecurityHeadersMiddleware,
)
from backend.middleware.stack import (
    MiddlewarePriority,
    MiddlewareRegistration,
    MiddlewareStack,
)
from backend.middleware.response_optimizer import (
    ResponseTimingMiddleware,
    ETagMiddleware,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_request(path="/test", method="GET", headers=None, scheme="http", client_host="127.0.0.1"):
    """Build a mock Starlette-style Request."""
    req = MagicMock()
    req.url.path = path
    req.url.scheme = scheme
    req.method = method
    req.headers = headers or {}
    req.state = MagicMock()
    if client_host:
        req.client.host = client_host
    else:
        req.client = None
    return req


def _make_response(status_code=200, headers=None):
    """Build a mock Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = dict(headers or {})
    return resp


# ===========================================================================
# error_handler.py
# ===========================================================================

class TestHttpExceptionHandler:
    """Unit tests for http_exception_handler."""

    @pytest.mark.asyncio
    async def test_returns_json_with_correct_status(self):
        request = _make_request("/api/items", "GET")
        exc = HTTPException(status_code=404, detail="Item not found")

        resp = await http_exception_handler(request, exc)

        assert resp.status_code == 404
        body = resp.body.decode()
        assert "Item not found" in body
        assert "HTTP_404" in body

    @pytest.mark.asyncio
    async def test_returns_json_for_403(self):
        request = _make_request("/api/secret", "POST")
        exc = HTTPException(status_code=403, detail="Forbidden")

        resp = await http_exception_handler(request, exc)

        assert resp.status_code == 403
        body = resp.body.decode()
        assert "Forbidden" in body
        assert "HTTP_403" in body

    @pytest.mark.asyncio
    async def test_returns_json_for_500(self):
        request = _make_request()
        exc = HTTPException(status_code=500, detail="Server error")

        resp = await http_exception_handler(request, exc)

        assert resp.status_code == 500
        assert b'"success": false' in resp.body or b'"success":false' in resp.body

    @pytest.mark.asyncio
    async def test_body_contains_success_false(self):
        request = _make_request()
        exc = HTTPException(status_code=401, detail="Unauthorized")

        resp = await http_exception_handler(request, exc)
        import json
        data = json.loads(resp.body)

        assert data["success"] is False
        assert data["error"] == "Unauthorized"
        assert data["code"] == "HTTP_401"


class TestValidationExceptionHandler:
    """Unit tests for validation_exception_handler."""

    @pytest.mark.asyncio
    async def test_handles_request_validation_error(self):
        request = _make_request("/api/users", "POST")

        # Build a RequestValidationError with realistic error structure
        errors = [
            {"loc": ("body", "email"), "msg": "field required", "type": "value_error.missing"},
            {"loc": ("body", "age"), "msg": "ensure this value is greater than 0", "type": "value_error.number.not_gt"},
        ]
        exc = RequestValidationError(errors=errors)

        resp = await validation_exception_handler(request, exc)

        assert resp.status_code == 422
        import json
        data = json.loads(resp.body)
        assert data["success"] is False
        assert data["code"] == "VALIDATION_ERROR"
        assert "body.email" in data["detail"]
        assert "body.age" in data["detail"]

    @pytest.mark.asyncio
    async def test_handles_single_validation_error(self):
        request = _make_request()
        errors = [
            {"loc": ("query", "page"), "msg": "value is not a valid integer", "type": "type_error.integer"},
        ]
        exc = RequestValidationError(errors=errors)

        resp = await validation_exception_handler(request, exc)

        import json
        data = json.loads(resp.body)
        assert "query.page" in data["detail"]

    @pytest.mark.asyncio
    async def test_handles_pydantic_validation_error(self):
        """Test handler with a real Pydantic ValidationError."""
        request = _make_request()

        class TestModel(BaseModel):
            name: str
            count: int

        try:
            TestModel(name=123, count="not_a_number")
        except ValidationError as exc:
            resp = await validation_exception_handler(request, exc)

            assert resp.status_code == 422
            import json
            data = json.loads(resp.body)
            assert data["code"] == "VALIDATION_ERROR"
            assert "count" in data["detail"]


class TestGeneralExceptionHandler:
    """Unit tests for general_exception_handler."""

    @pytest.mark.asyncio
    async def test_returns_500_with_generic_message(self):
        request = _make_request("/crash", "GET")
        exc = RuntimeError("something broke internally")

        resp = await general_exception_handler(request, exc)

        assert resp.status_code == 500
        import json
        data = json.loads(resp.body)
        assert data["success"] is False
        assert data["code"] == "INTERNAL_ERROR"
        # Should NOT leak the internal error message
        assert "something broke internally" not in data["error"]
        assert "Internal server error" in data["error"]

    @pytest.mark.asyncio
    async def test_does_not_expose_traceback(self):
        request = _make_request()
        exc = ValueError("secret DB password mismatch")

        resp = await general_exception_handler(request, exc)

        body_str = resp.body.decode()
        assert "secret DB password" not in body_str


class TestRegisterExceptionHandlers:
    """Unit tests for register_exception_handlers."""

    def test_registers_all_handlers(self):
        app = MagicMock()
        register_exception_handlers(app)

        assert app.add_exception_handler.call_count == 4

        registered_types = [call.args[0] for call in app.add_exception_handler.call_args_list]
        assert HTTPException in registered_types
        assert RequestValidationError in registered_types
        assert ValidationError in registered_types
        assert Exception in registered_types


# ===========================================================================
# request_size_limiter.py
# ===========================================================================

class TestContentTypeEnum:
    """Unit tests for ContentType enum."""

    def test_json_value(self):
        assert ContentType.JSON == "application/json"

    def test_form_value(self):
        assert ContentType.FORM == "application/x-www-form-urlencoded"

    def test_multipart_value(self):
        assert ContentType.MULTIPART == "multipart/form-data"

    def test_text_value(self):
        assert ContentType.TEXT == "text/plain"

    def test_octet_stream_value(self):
        assert ContentType.OCTET_STREAM == "application/octet-stream"


class TestRequestSizeLimitsDataclass:
    """Unit tests for RequestSizeLimits defaults and configuration."""

    def test_default_exempt_paths(self):
        limits = RequestSizeLimits()
        assert "/health" in limits.exempt_paths
        assert "/metrics" in limits.exempt_paths

    def test_default_error_message(self):
        limits = RequestSizeLimits()
        assert limits.error_message == "Request payload too large"

    def test_log_violations_default_true(self):
        limits = RequestSizeLimits()
        assert limits.log_violations is True

    def test_text_limit_default(self):
        limits = RequestSizeLimits()
        assert limits.text_limit == 524_288  # 512 KB


class TestRequestSizeLimiterInternalMethods:
    """Unit tests for RequestSizeLimiterMiddleware internal methods."""

    def _make_middleware(self, config=None):
        mock_app = MagicMock()
        return RequestSizeLimiterMiddleware(mock_app, config=config)

    def test_get_content_type_strips_charset(self):
        mw = self._make_middleware()
        req = _make_request(headers={"content-type": "application/json; charset=utf-8"})
        assert mw._get_content_type(req) == "application/json"

    def test_get_content_type_lowercases(self):
        mw = self._make_middleware()
        req = _make_request(headers={"content-type": "Application/JSON"})
        assert mw._get_content_type(req) == "application/json"

    def test_get_content_type_empty_header(self):
        mw = self._make_middleware()
        req = _make_request(headers={})
        assert mw._get_content_type(req) == ""

    def test_get_size_limit_json(self):
        config = RequestSizeLimits(json_limit=5000)
        mw = self._make_middleware(config)
        req = _make_request(headers={"content-type": "application/json"})
        assert mw._get_size_limit(req) == 5000

    def test_get_size_limit_form(self):
        config = RequestSizeLimits(form_limit=7000)
        mw = self._make_middleware(config)
        req = _make_request(headers={"content-type": "application/x-www-form-urlencoded"})
        assert mw._get_size_limit(req) == 7000

    def test_get_size_limit_multipart(self):
        config = RequestSizeLimits(file_upload_limit=20_000)
        mw = self._make_middleware(config)
        req = _make_request(headers={"content-type": "multipart/form-data; boundary=something"})
        assert mw._get_size_limit(req) == 20_000

    def test_get_size_limit_text(self):
        config = RequestSizeLimits(text_limit=3000)
        mw = self._make_middleware(config)
        req = _make_request(headers={"content-type": "text/plain"})
        assert mw._get_size_limit(req) == 3000

    def test_get_size_limit_path_override(self):
        config = RequestSizeLimits(
            json_limit=1000,
            path_limits={"/api/bulk": 99_999}
        )
        mw = self._make_middleware(config)
        req = _make_request(path="/api/bulk/upload", headers={"content-type": "application/json"})
        assert mw._get_size_limit(req) == 99_999

    def test_get_size_limit_falls_back_to_default(self):
        config = RequestSizeLimits(default_limit=42_000)
        mw = self._make_middleware(config)
        req = _make_request(headers={"content-type": "application/xml"})
        assert mw._get_size_limit(req) == 42_000

    def test_is_exempt_path_true(self):
        config = RequestSizeLimits(exempt_paths={"/health", "/readiness"})
        mw = self._make_middleware(config)
        assert mw._is_exempt_path("/health") is True

    def test_is_exempt_path_false(self):
        mw = self._make_middleware()
        assert mw._is_exempt_path("/api/data") is False

    def test_format_size_bytes(self):
        mw = self._make_middleware()
        assert mw._format_size(500) == "500 B"

    def test_format_size_kilobytes(self):
        mw = self._make_middleware()
        assert mw._format_size(2048) == "2.0 KB"

    def test_format_size_megabytes(self):
        mw = self._make_middleware()
        assert mw._format_size(5_242_880) == "5.0 MB"

    @pytest.mark.asyncio
    async def test_dispatch_skips_get_requests(self):
        mw = self._make_middleware()
        req = _make_request(method="GET")
        mock_resp = _make_response()
        call_next = AsyncMock(return_value=mock_resp)

        resp = await mw.dispatch(req, call_next)

        call_next.assert_awaited_once_with(req)
        assert resp is mock_resp

    @pytest.mark.asyncio
    async def test_dispatch_skips_exempt_path(self):
        config = RequestSizeLimits(exempt_paths={"/health"})
        mw = self._make_middleware(config)
        req = _make_request(path="/health", method="POST")
        mock_resp = _make_response()
        call_next = AsyncMock(return_value=mock_resp)

        resp = await mw.dispatch(req, call_next)

        call_next.assert_awaited_once_with(req)
        assert resp is mock_resp

    @pytest.mark.asyncio
    async def test_dispatch_rejects_oversized_post(self):
        config = RequestSizeLimits(json_limit=100)
        mw = self._make_middleware(config)
        req = _make_request(
            path="/api/data",
            method="POST",
            headers={"content-type": "application/json", "content-length": "5000"},
        )
        call_next = AsyncMock()

        resp = await mw.dispatch(req, call_next)

        assert resp.status_code == 413
        call_next.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dispatch_allows_within_limit(self):
        config = RequestSizeLimits(json_limit=10_000)
        mw = self._make_middleware(config)
        req = _make_request(
            path="/api/data",
            method="POST",
            headers={"content-type": "application/json", "content-length": "500"},
        )
        mock_resp = _make_response()
        call_next = AsyncMock(return_value=mock_resp)

        resp = await mw.dispatch(req, call_next)

        call_next.assert_awaited_once()
        assert resp is mock_resp

    @pytest.mark.asyncio
    async def test_dispatch_handles_invalid_content_length(self):
        """Invalid Content-Length should not crash; request proceeds."""
        mw = self._make_middleware()
        req = _make_request(
            method="POST",
            headers={"content-type": "application/json", "content-length": "not_a_number"},
        )
        mock_resp = _make_response()
        call_next = AsyncMock(return_value=mock_resp)

        resp = await mw.dispatch(req, call_next)

        call_next.assert_awaited_once()
        assert resp is mock_resp

    @pytest.mark.asyncio
    async def test_dispatch_catches_too_large_exception(self):
        config = RequestSizeLimits(json_limit=100)
        mw = self._make_middleware(config)
        req = _make_request(
            method="PUT",
            headers={"content-type": "application/json"},
        )
        call_next = AsyncMock(side_effect=Exception("request body too large"))

        resp = await mw.dispatch(req, call_next)

        assert resp.status_code == 413

    @pytest.mark.asyncio
    async def test_dispatch_reraises_non_size_exception(self):
        mw = self._make_middleware()
        req = _make_request(
            method="PATCH",
            headers={"content-type": "application/json"},
        )
        call_next = AsyncMock(side_effect=RuntimeError("unrelated error"))

        with pytest.raises(RuntimeError, match="unrelated error"):
            await mw.dispatch(req, call_next)

    @pytest.mark.asyncio
    async def test_dispatch_no_client_host(self):
        """When request.client is None, logging should still work."""
        config = RequestSizeLimits(json_limit=10)
        mw = self._make_middleware(config)
        req = _make_request(
            method="POST",
            headers={"content-type": "application/json", "content-length": "5000"},
            client_host=None,
        )
        call_next = AsyncMock()

        resp = await mw.dispatch(req, call_next)

        assert resp.status_code == 413

    @pytest.mark.asyncio
    async def test_dispatch_logging_disabled(self):
        """When log_violations is False, should still reject."""
        config = RequestSizeLimits(json_limit=10, log_violations=False)
        mw = self._make_middleware(config)
        req = _make_request(
            method="POST",
            headers={"content-type": "application/json", "content-length": "5000"},
        )
        call_next = AsyncMock()

        resp = await mw.dispatch(req, call_next)

        assert resp.status_code == 413


# ===========================================================================
# security_headers.py
# ===========================================================================

class TestFrameOptionsEnum:
    """Unit tests for FrameOptions enum."""

    def test_deny_value(self):
        assert FrameOptions.DENY.value == "DENY"

    def test_sameorigin_value(self):
        assert FrameOptions.SAMEORIGIN.value == "SAMEORIGIN"


class TestReferrerPolicyEnum:
    """Unit tests for ReferrerPolicy enum values."""

    def test_no_referrer(self):
        assert ReferrerPolicy.NO_REFERRER.value == "no-referrer"

    def test_strict_origin_when_cross_origin(self):
        assert ReferrerPolicy.STRICT_ORIGIN_WHEN_CROSS_ORIGIN.value == "strict-origin-when-cross-origin"

    def test_same_origin(self):
        assert ReferrerPolicy.SAME_ORIGIN.value == "same-origin"


class TestContentSecurityPolicyBuild:
    """Unit tests for ContentSecurityPolicy.build()."""

    def test_empty_lists_omit_directives(self):
        csp = ContentSecurityPolicy(
            default_src=[],
            script_src=[],
            style_src=[],
            img_src=[],
            font_src=[],
            connect_src=[],
            frame_src=[],
            object_src=[],
            base_uri=[],
            form_action=[],
            frame_ancestors=[],
            upgrade_insecure_requests=False,
            block_all_mixed_content=False,
        )
        header = csp.build()
        assert header == ""

    def test_only_boolean_directives(self):
        csp = ContentSecurityPolicy(
            default_src=[],
            script_src=[],
            style_src=[],
            img_src=[],
            font_src=[],
            connect_src=[],
            frame_src=[],
            object_src=[],
            base_uri=[],
            form_action=[],
            frame_ancestors=[],
            upgrade_insecure_requests=True,
            block_all_mixed_content=True,
        )
        header = csp.build()
        assert "upgrade-insecure-requests" in header
        assert "block-all-mixed-content" in header

    def test_report_uri_included(self):
        csp = ContentSecurityPolicy(
            report_uri="https://report.example.com/csp"
        )
        header = csp.build()
        assert "report-uri https://report.example.com/csp" in header

    def test_multiple_sources_space_separated(self):
        csp = ContentSecurityPolicy(
            script_src=["'self'", "'unsafe-inline'", "https://cdn.test.com"]
        )
        header = csp.build()
        assert "script-src 'self' 'unsafe-inline' https://cdn.test.com" in header


class TestPermissionsPolicyBuild:
    """Unit tests for PermissionsPolicy.build()."""

    def test_all_denied_by_default(self):
        policy = PermissionsPolicy()
        header = policy.build()
        # Every feature should be denied: feature=()
        for feature in ["accelerometer", "camera", "microphone", "geolocation", "payment", "usb"]:
            assert f"{feature}=()" in header

    def test_mixed_allowed_and_denied(self):
        policy = PermissionsPolicy(
            camera=["'self'"],
            microphone=[],
        )
        header = policy.build()
        assert "camera=('self')" in header
        assert "microphone=()" in header


class TestSecurityHeadersConfigPostInit:
    """Unit tests for SecurityHeadersConfig __post_init__."""

    def test_creates_default_csp_when_none(self):
        config = SecurityHeadersConfig(csp=None)
        assert config.csp is not None
        assert isinstance(config.csp, ContentSecurityPolicy)

    def test_creates_default_permissions_when_none(self):
        config = SecurityHeadersConfig(permissions_policy=None)
        assert config.permissions_policy is not None
        assert isinstance(config.permissions_policy, PermissionsPolicy)

    def test_preserves_provided_csp(self):
        custom_csp = ContentSecurityPolicy(script_src=["https://custom.com"])
        config = SecurityHeadersConfig(csp=custom_csp)
        assert config.csp is custom_csp

    def test_default_exclude_paths(self):
        config = SecurityHeadersConfig()
        assert "/metrics" in config.exclude_paths
        assert "/health" in config.exclude_paths


class TestSecurityHeadersMiddlewareInternal:
    """Unit tests for SecurityHeadersMiddleware internal methods."""

    def _make_middleware(self, config=None):
        mock_app = MagicMock()
        return SecurityHeadersMiddleware(mock_app, config=config)

    def test_should_apply_headers_excludes_configured_paths(self):
        config = SecurityHeadersConfig(exclude_paths={"/internal", "/health"})
        mw = self._make_middleware(config)
        assert mw._should_apply_headers("/internal") is False
        assert mw._should_apply_headers("/health") is False
        assert mw._should_apply_headers("/api/users") is True

    def test_build_hsts_basic(self):
        config = SecurityHeadersConfig(
            hsts_max_age=300,
            hsts_include_subdomains=False,
            hsts_preload=False,
        )
        mw = self._make_middleware(config)
        assert mw._build_hsts_header() == "max-age=300"

    def test_build_hsts_with_subdomains_and_preload(self):
        config = SecurityHeadersConfig(
            hsts_max_age=86400,
            hsts_include_subdomains=True,
            hsts_preload=True,
        )
        mw = self._make_middleware(config)
        hsts = mw._build_hsts_header()
        assert "max-age=86400" in hsts
        assert "includeSubDomains" in hsts
        assert "preload" in hsts

    @pytest.mark.asyncio
    async def test_dispatch_skips_excluded_path(self):
        config = SecurityHeadersConfig(exclude_paths={"/metrics"})
        mw = self._make_middleware(config)
        req = _make_request(path="/metrics")
        raw_resp = _make_response()
        call_next = AsyncMock(return_value=raw_resp)

        resp = await mw.dispatch(req, call_next)

        assert resp is raw_resp
        # Headers dict should NOT have security headers added
        assert "X-Content-Type-Options" not in resp.headers

    @pytest.mark.asyncio
    async def test_dispatch_adds_all_default_headers(self):
        config = SecurityHeadersConfig()
        mw = self._make_middleware(config)
        req = _make_request(path="/api/data", scheme="https")
        raw_resp = _make_response()
        call_next = AsyncMock(return_value=raw_resp)

        resp = await mw.dispatch(req, call_next)

        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        assert resp.headers["X-Frame-Options"] == "DENY"
        assert resp.headers["X-XSS-Protection"] == "1; mode=block"
        assert "Strict-Transport-Security" in resp.headers
        assert "Content-Security-Policy" in resp.headers
        assert "Referrer-Policy" in resp.headers
        assert "Permissions-Policy" in resp.headers

    @pytest.mark.asyncio
    async def test_dispatch_no_hsts_on_http(self):
        config = SecurityHeadersConfig(hsts_enabled=True)
        mw = self._make_middleware(config)
        req = _make_request(path="/api/data", scheme="http")
        raw_resp = _make_response()
        call_next = AsyncMock(return_value=raw_resp)

        resp = await mw.dispatch(req, call_next)

        assert "Strict-Transport-Security" not in resp.headers

    @pytest.mark.asyncio
    async def test_dispatch_disables_csp(self):
        config = SecurityHeadersConfig(csp_enabled=False)
        mw = self._make_middleware(config)
        req = _make_request(path="/api/data")
        raw_resp = _make_response()
        call_next = AsyncMock(return_value=raw_resp)

        resp = await mw.dispatch(req, call_next)

        assert "Content-Security-Policy" not in resp.headers

    @pytest.mark.asyncio
    async def test_dispatch_disables_permissions_policy(self):
        config = SecurityHeadersConfig(permissions_policy_enabled=False)
        mw = self._make_middleware(config)
        req = _make_request(path="/api/data")
        raw_resp = _make_response()
        call_next = AsyncMock(return_value=raw_resp)

        resp = await mw.dispatch(req, call_next)

        assert "Permissions-Policy" not in resp.headers

    @pytest.mark.asyncio
    async def test_dispatch_custom_headers(self):
        config = SecurityHeadersConfig(
            custom_headers={"X-My-Header": "hello", "X-Another": "world"}
        )
        mw = self._make_middleware(config)
        req = _make_request(path="/api/data")
        raw_resp = _make_response()
        call_next = AsyncMock(return_value=raw_resp)

        resp = await mw.dispatch(req, call_next)

        assert resp.headers["X-My-Header"] == "hello"
        assert resp.headers["X-Another"] == "world"

    @pytest.mark.asyncio
    async def test_dispatch_nosniff_disabled(self):
        config = SecurityHeadersConfig(content_type_nosniff=False)
        mw = self._make_middleware(config)
        req = _make_request(path="/api/data")
        raw_resp = _make_response()
        call_next = AsyncMock(return_value=raw_resp)

        resp = await mw.dispatch(req, call_next)

        assert "X-Content-Type-Options" not in resp.headers

    def test_warns_on_short_hsts_max_age(self):
        with patch("backend.middleware.security_headers.logger") as mock_logger:
            config = SecurityHeadersConfig(hsts_enabled=True, hsts_max_age=10)
            SecurityHeadersMiddleware(MagicMock(), config=config)
            mock_logger.warning.assert_called_once()


# ===========================================================================
# stack.py
# ===========================================================================

class TestMiddlewarePriorityValues:
    """Unit tests for MiddlewarePriority enum additional values."""

    def test_highest_above_error_handler(self):
        assert MiddlewarePriority.HIGHEST > MiddlewarePriority.ERROR_HANDLER

    def test_lowest_below_compression(self):
        assert MiddlewarePriority.LOWEST < MiddlewarePriority.COMPRESSION

    def test_audit_between_monitoring_and_caching(self):
        assert MiddlewarePriority.MONITORING > MiddlewarePriority.AUDIT > MiddlewarePriority.CACHING


class TestMiddlewareRegistrationDefaults:
    """Unit tests for MiddlewareRegistration dataclass defaults."""

    def test_enabled_default_true(self):
        reg = MiddlewareRegistration(
            name="x", middleware_class=object, priority=MiddlewarePriority.NORMAL, config={}
        )
        assert reg.enabled is True

    def test_skip_in_testing_default_false(self):
        reg = MiddlewareRegistration(
            name="x", middleware_class=object, priority=MiddlewarePriority.NORMAL, config={}
        )
        assert reg.skip_in_testing is False


class TestMiddlewareStackUnit:
    """Additional unit tests for MiddlewareStack not covered by integration tests."""

    def test_register_returns_self_for_chaining(self):
        app = FastAPI()
        stack = MiddlewareStack(app)
        from starlette.middleware.base import BaseHTTPMiddleware

        class Dummy(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                return await call_next(request)

        result = stack.register("a", Dummy, MiddlewarePriority.NORMAL)
        assert result is stack

    def test_register_stores_config(self):
        app = FastAPI()
        stack = MiddlewareStack(app)
        from starlette.middleware.base import BaseHTTPMiddleware

        class Dummy(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                return await call_next(request)

        stack.register("a", Dummy, MiddlewarePriority.HIGH, config={"key": "val"})
        assert stack.middlewares[0].config == {"key": "val"}

    def test_register_default_config_is_empty_dict(self):
        app = FastAPI()
        stack = MiddlewareStack(app)
        from starlette.middleware.base import BaseHTTPMiddleware

        class Dummy(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                return await call_next(request)

        stack.register("a", Dummy, MiddlewarePriority.NORMAL)
        assert stack.middlewares[0].config == {}

    def test_apply_sets_applied_flag(self):
        app = FastAPI()
        stack = MiddlewareStack(app)
        assert stack._applied is False
        stack.apply()
        assert stack._applied is True

    def test_apply_twice_does_not_raise(self):
        """Second apply() is a no-op."""
        app = FastAPI()
        stack = MiddlewareStack(app)
        stack.apply()
        stack.apply()  # should not raise
        assert stack._applied is True

    def test_get_stack_summary_disabled_marker(self):
        app = FastAPI()
        stack = MiddlewareStack(app)
        from starlette.middleware.base import BaseHTTPMiddleware

        class Dummy(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                return await call_next(request)

        stack.register("my_disabled", Dummy, MiddlewarePriority.NORMAL, enabled=False)
        summary = stack.get_stack_summary()
        assert "my_disabled" in summary

    def test_apply_raises_on_middleware_error(self):
        """If add_middleware raises, apply should propagate the error."""
        app = MagicMock()
        app.add_middleware.side_effect = TypeError("bad config")
        stack = MiddlewareStack(app)
        from starlette.middleware.base import BaseHTTPMiddleware

        class Dummy(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                return await call_next(request)

        stack.register("bad", Dummy, MiddlewarePriority.NORMAL)

        with pytest.raises(TypeError, match="bad config"):
            stack.apply()


# ===========================================================================
# response_optimizer.py
# ===========================================================================

class TestETagMiddlewareInternalMethods:
    """Unit tests for ETagMiddleware internal helper methods."""

    def _make_etag_mw(self, **kwargs):
        return ETagMiddleware(app=MagicMock(), **kwargs)

    def test_should_generate_etag_get_200_true(self):
        mw = self._make_etag_mw(excluded_paths=[])
        assert mw._should_generate_etag("/api/data", "GET", 200, {}) is True

    def test_should_generate_etag_head_200_true(self):
        mw = self._make_etag_mw(excluded_paths=[])
        assert mw._should_generate_etag("/api/data", "HEAD", 200, {}) is True

    def test_should_generate_etag_post_false(self):
        mw = self._make_etag_mw()
        assert mw._should_generate_etag("/api/data", "POST", 200, {}) is False

    def test_should_generate_etag_non_200_false(self):
        mw = self._make_etag_mw(excluded_paths=[])
        assert mw._should_generate_etag("/api/data", "GET", 404, {}) is False

    def test_should_generate_etag_excluded_path_false(self):
        mw = self._make_etag_mw(excluded_paths=["/api/v1/auth/"])
        assert mw._should_generate_etag("/api/v1/auth/login", "GET", 200, {}) is False

    def test_should_generate_etag_existing_etag_false(self):
        mw = self._make_etag_mw(excluded_paths=[])
        assert mw._should_generate_etag("/api/data", "GET", 200, {b"etag": b'"existing"'}) is False

    def test_generate_etag_strong(self):
        mw = self._make_etag_mw(weak_etag=False)
        content = b'{"message":"hello"}'
        etag = mw._generate_etag(content)
        expected_hash = hashlib.md5(content).hexdigest()[:16]
        assert etag == f'"{expected_hash}"'

    def test_generate_etag_weak(self):
        mw = self._make_etag_mw(weak_etag=True)
        content = b'{"data":"test"}'
        etag = mw._generate_etag(content)
        expected_hash = hashlib.md5(content).hexdigest()[:16]
        assert etag == f'W/"{expected_hash}"'

    def test_generate_etag_deterministic(self):
        mw = self._make_etag_mw()
        content = b"fixed content"
        assert mw._generate_etag(content) == mw._generate_etag(content)

    def test_generate_etag_different_content(self):
        mw = self._make_etag_mw()
        assert mw._generate_etag(b"aaa") != mw._generate_etag(b"bbb")

    def test_check_if_none_match_empty_string_false(self):
        mw = self._make_etag_mw()
        assert mw._check_if_none_match("", '"abc"') is False

    def test_check_if_none_match_exact_match_true(self):
        mw = self._make_etag_mw()
        assert mw._check_if_none_match('"abc123"', '"abc123"') is True

    def test_check_if_none_match_no_match_false(self):
        mw = self._make_etag_mw()
        assert mw._check_if_none_match('"xyz"', '"abc"') is False

    def test_check_if_none_match_wildcard_true(self):
        mw = self._make_etag_mw()
        assert mw._check_if_none_match("*", '"anything"') is True

    def test_check_if_none_match_multiple_etags(self):
        mw = self._make_etag_mw()
        assert mw._check_if_none_match('"aaa", "bbb", "ccc"', '"bbb"') is True

    def test_check_if_none_match_multiple_no_match(self):
        mw = self._make_etag_mw()
        assert mw._check_if_none_match('"aaa", "bbb"', '"zzz"') is False

    @pytest.mark.xfail(
        reason="Known bug: strip('\"') before replace('W/', '') leaves inner quote "
               "when client sends W/\"...\" but server etag is strong. "
               "See _check_if_none_match in response_optimizer.py."
    )
    def test_check_if_none_match_weak_etag_comparison(self):
        mw = self._make_etag_mw()
        # Per HTTP spec, weak comparison should match W/"abc" to "abc"
        assert mw._check_if_none_match('W/"abc"', '"abc"') is True

    def test_check_if_none_match_weak_client_strong_server_actual_behavior(self):
        """Document actual behavior: W/\"abc\" does NOT match \"abc\" due to strip order."""
        mw = self._make_etag_mw()
        # This currently returns False because the implementation strips quotes
        # before removing the W/ prefix, leaving a stray inner quote.
        assert mw._check_if_none_match('W/"abc"', '"abc"') is False

    def test_default_excluded_paths(self):
        mw = self._make_etag_mw()
        assert "/api/v1/auth/" in mw.excluded_paths
        assert "/api/v1/admin/" in mw.excluded_paths
        assert "/api/v1/ws/" in mw.excluded_paths
        assert "/api/health" in mw.excluded_paths

    @pytest.mark.asyncio
    async def test_call_non_http_scope_passthrough(self):
        """Non-HTTP scopes (e.g. websocket) should pass straight through."""
        mock_app = AsyncMock()
        mw = ETagMiddleware(app=mock_app)

        scope = {"type": "websocket"}
        receive = AsyncMock()
        send = AsyncMock()

        await mw(scope, receive, send)

        mock_app.assert_awaited_once_with(scope, receive, send)


class TestResponseTimingMiddlewareUnit:
    """Unit tests for ResponseTimingMiddleware dispatch logic."""

    @pytest.mark.asyncio
    async def test_dispatch_adds_header(self):
        mw = ResponseTimingMiddleware(app=MagicMock())
        req = _make_request()
        resp = _make_response()
        call_next = AsyncMock(return_value=resp)

        result = await mw.dispatch(req, call_next)

        assert "X-Response-Time" in result.headers
        assert result.headers["X-Response-Time"].endswith("ms")

    @pytest.mark.asyncio
    async def test_dispatch_propagates_exception(self):
        mw = ResponseTimingMiddleware(app=MagicMock())
        req = _make_request()
        call_next = AsyncMock(side_effect=RuntimeError("boom"))

        with pytest.raises(RuntimeError, match="boom"):
            await mw.dispatch(req, call_next)
