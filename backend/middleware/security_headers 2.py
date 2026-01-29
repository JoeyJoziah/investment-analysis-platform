"""
Security Headers Middleware

Implements comprehensive security headers for defense-in-depth:
- X-Content-Type-Options (nosniff)
- X-Frame-Options (clickjacking protection)
- X-XSS-Protection (legacy XSS filter)
- Strict-Transport-Security (HSTS for HTTPS)
- Content-Security-Policy (CSP for XSS prevention)
- Referrer-Policy (control referrer information)
- Permissions-Policy (control browser features)

Created: 2026-01-27
Part of: Phase 3 Security Remediation (HIGH-4)
"""

import logging
from typing import Optional, Dict, List, Set
from dataclasses import dataclass, field
from enum import Enum

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class FrameOptions(str, Enum):
    """X-Frame-Options values"""
    DENY = "DENY"
    SAMEORIGIN = "SAMEORIGIN"


class ReferrerPolicy(str, Enum):
    """Referrer-Policy values"""
    NO_REFERRER = "no-referrer"
    NO_REFERRER_WHEN_DOWNGRADE = "no-referrer-when-downgrade"
    ORIGIN = "origin"
    ORIGIN_WHEN_CROSS_ORIGIN = "origin-when-cross-origin"
    SAME_ORIGIN = "same-origin"
    STRICT_ORIGIN = "strict-origin"
    STRICT_ORIGIN_WHEN_CROSS_ORIGIN = "strict-origin-when-cross-origin"


@dataclass
class ContentSecurityPolicy:
    """Content Security Policy configuration"""
    default_src: List[str] = field(default_factory=lambda: ["'self'"])
    script_src: List[str] = field(default_factory=lambda: ["'self'"])
    style_src: List[str] = field(default_factory=lambda: ["'self'", "'unsafe-inline'"])
    img_src: List[str] = field(default_factory=lambda: ["'self'", "data:", "https:"])
    font_src: List[str] = field(default_factory=lambda: ["'self'", "data:"])
    connect_src: List[str] = field(default_factory=lambda: ["'self'"])
    frame_src: List[str] = field(default_factory=lambda: ["'none'"])
    object_src: List[str] = field(default_factory=lambda: ["'none'"])
    base_uri: List[str] = field(default_factory=lambda: ["'self'"])
    form_action: List[str] = field(default_factory=lambda: ["'self'"])
    frame_ancestors: List[str] = field(default_factory=lambda: ["'none'"])
    upgrade_insecure_requests: bool = True
    block_all_mixed_content: bool = True
    report_uri: Optional[str] = None

    def build(self) -> str:
        """
        Build CSP header value

        Returns:
            Formatted CSP header string
        """
        directives = []

        # Standard directives
        if self.default_src:
            directives.append(f"default-src {' '.join(self.default_src)}")
        if self.script_src:
            directives.append(f"script-src {' '.join(self.script_src)}")
        if self.style_src:
            directives.append(f"style-src {' '.join(self.style_src)}")
        if self.img_src:
            directives.append(f"img-src {' '.join(self.img_src)}")
        if self.font_src:
            directives.append(f"font-src {' '.join(self.font_src)}")
        if self.connect_src:
            directives.append(f"connect-src {' '.join(self.connect_src)}")
        if self.frame_src:
            directives.append(f"frame-src {' '.join(self.frame_src)}")
        if self.object_src:
            directives.append(f"object-src {' '.join(self.object_src)}")
        if self.base_uri:
            directives.append(f"base-uri {' '.join(self.base_uri)}")
        if self.form_action:
            directives.append(f"form-action {' '.join(self.form_action)}")
        if self.frame_ancestors:
            directives.append(f"frame-ancestors {' '.join(self.frame_ancestors)}")

        # Boolean directives
        if self.upgrade_insecure_requests:
            directives.append("upgrade-insecure-requests")
        if self.block_all_mixed_content:
            directives.append("block-all-mixed-content")

        # Reporting
        if self.report_uri:
            directives.append(f"report-uri {self.report_uri}")

        return "; ".join(directives)


@dataclass
class PermissionsPolicy:
    """Permissions-Policy (formerly Feature-Policy) configuration"""
    accelerometer: List[str] = field(default_factory=lambda: [])
    ambient_light_sensor: List[str] = field(default_factory=lambda: [])
    autoplay: List[str] = field(default_factory=lambda: [])
    battery: List[str] = field(default_factory=lambda: [])
    camera: List[str] = field(default_factory=lambda: [])
    display_capture: List[str] = field(default_factory=lambda: [])
    geolocation: List[str] = field(default_factory=lambda: [])
    gyroscope: List[str] = field(default_factory=lambda: [])
    magnetometer: List[str] = field(default_factory=lambda: [])
    microphone: List[str] = field(default_factory=lambda: [])
    midi: List[str] = field(default_factory=lambda: [])
    payment: List[str] = field(default_factory=lambda: [])
    usb: List[str] = field(default_factory=lambda: [])

    def build(self) -> str:
        """
        Build Permissions-Policy header value

        Returns:
            Formatted Permissions-Policy header string
        """
        policies = []

        features = {
            "accelerometer": self.accelerometer,
            "ambient-light-sensor": self.ambient_light_sensor,
            "autoplay": self.autoplay,
            "battery": self.battery,
            "camera": self.camera,
            "display-capture": self.display_capture,
            "geolocation": self.geolocation,
            "gyroscope": self.gyroscope,
            "magnetometer": self.magnetometer,
            "microphone": self.microphone,
            "midi": self.midi,
            "payment": self.payment,
            "usb": self.usb,
        }

        for feature, origins in features.items():
            if origins:
                # Format: feature=(origin1 origin2)
                origins_str = " ".join(origins)
                policies.append(f"{feature}=({origins_str})")
            else:
                # Empty list means deny all
                policies.append(f"{feature}=()")

        return ", ".join(policies)


@dataclass
class SecurityHeadersConfig:
    """Security headers configuration"""
    # X-Content-Type-Options
    content_type_nosniff: bool = True

    # X-Frame-Options
    frame_options: FrameOptions = FrameOptions.DENY

    # X-XSS-Protection (legacy, but still useful)
    xss_protection: str = "1; mode=block"

    # Strict-Transport-Security (HSTS)
    hsts_enabled: bool = True
    hsts_max_age: int = 31536000  # 1 year in seconds
    hsts_include_subdomains: bool = True
    hsts_preload: bool = False

    # Content-Security-Policy
    csp_enabled: bool = True
    csp: Optional[ContentSecurityPolicy] = None

    # Referrer-Policy
    referrer_policy: ReferrerPolicy = ReferrerPolicy.STRICT_ORIGIN_WHEN_CROSS_ORIGIN

    # Permissions-Policy
    permissions_policy_enabled: bool = True
    permissions_policy: Optional[PermissionsPolicy] = None

    # Custom headers
    custom_headers: Dict[str, str] = field(default_factory=dict)

    # Paths to exclude from security headers
    exclude_paths: Set[str] = field(default_factory=lambda: {"/metrics", "/health"})

    def __post_init__(self):
        """Initialize default CSP and Permissions Policy if not provided"""
        if self.csp is None:
            self.csp = ContentSecurityPolicy()

        if self.permissions_policy is None:
            self.permissions_policy = PermissionsPolicy()


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Security Headers Middleware

    Automatically adds comprehensive security headers to all responses
    for defense-in-depth protection against common web vulnerabilities.
    """

    def __init__(self, app: ASGIApp, config: Optional[SecurityHeadersConfig] = None):
        """
        Initialize security headers middleware

        Args:
            app: ASGI application
            config: Security headers configuration
        """
        super().__init__(app)
        self.config = config or SecurityHeadersConfig()

        # Validate configuration
        if self.config.hsts_enabled and self.config.hsts_max_age < 300:
            logger.warning("HSTS max-age should be at least 300 seconds (5 minutes)")

    def _should_apply_headers(self, path: str) -> bool:
        """
        Check if security headers should be applied to this path

        Args:
            path: Request path

        Returns:
            True if headers should be applied
        """
        return path not in self.config.exclude_paths

    def _build_hsts_header(self) -> str:
        """
        Build Strict-Transport-Security header value

        Returns:
            HSTS header string
        """
        value = f"max-age={self.config.hsts_max_age}"

        if self.config.hsts_include_subdomains:
            value += "; includeSubDomains"

        if self.config.hsts_preload:
            value += "; preload"

        return value

    async def dispatch(self, request: Request, call_next):
        """
        Process request and add security headers to response

        Args:
            request: Incoming request
            call_next: Next middleware/handler

        Returns:
            Response with security headers added
        """
        # Process request
        response = await call_next(request)

        # Check if we should apply headers
        if not self._should_apply_headers(request.url.path):
            return response

        # X-Content-Type-Options
        if self.config.content_type_nosniff:
            response.headers["X-Content-Type-Options"] = "nosniff"

        # X-Frame-Options
        response.headers["X-Frame-Options"] = self.config.frame_options.value

        # X-XSS-Protection (legacy but still useful)
        if self.config.xss_protection:
            response.headers["X-XSS-Protection"] = self.config.xss_protection

        # Strict-Transport-Security (only over HTTPS)
        if self.config.hsts_enabled and request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = self._build_hsts_header()

        # Content-Security-Policy
        if self.config.csp_enabled and self.config.csp:
            csp_value = self.config.csp.build()
            if csp_value:
                response.headers["Content-Security-Policy"] = csp_value

        # Referrer-Policy
        response.headers["Referrer-Policy"] = self.config.referrer_policy.value

        # Permissions-Policy
        if self.config.permissions_policy_enabled and self.config.permissions_policy:
            permissions_value = self.config.permissions_policy.build()
            if permissions_value:
                response.headers["Permissions-Policy"] = permissions_value

        # Custom headers
        for header_name, header_value in self.config.custom_headers.items():
            response.headers[header_name] = header_value

        return response


def add_security_headers(
    app,
    csp_script_src: Optional[List[str]] = None,
    csp_connect_src: Optional[List[str]] = None,
    exclude_paths: Optional[Set[str]] = None,
    **kwargs
) -> SecurityHeadersMiddleware:
    """
    Add security headers to FastAPI application

    Args:
        app: FastAPI application
        csp_script_src: Additional script-src origins for CSP
        csp_connect_src: Additional connect-src origins for CSP
        exclude_paths: Paths to exclude from security headers
        **kwargs: Additional SecurityHeadersConfig parameters

    Returns:
        SecurityHeadersMiddleware instance

    Example:
        ```python
        from fastapi import FastAPI
        from backend.middleware.security_headers import add_security_headers

        app = FastAPI()
        add_security_headers(
            app,
            csp_script_src=["'self'", "https://cdn.example.com"],
            csp_connect_src=["'self'", "https://api.example.com"]
        )
        ```
    """
    config_params = kwargs.copy()

    # Build CSP if custom sources provided
    if csp_script_src or csp_connect_src:
        csp = ContentSecurityPolicy()
        if csp_script_src:
            csp.script_src = csp_script_src
        if csp_connect_src:
            csp.connect_src = csp_connect_src
        config_params["csp"] = csp

    if exclude_paths:
        config_params["exclude_paths"] = exclude_paths

    config = SecurityHeadersConfig(**config_params)
    middleware = SecurityHeadersMiddleware(app, config)
    app.add_middleware(SecurityHeadersMiddleware, config=config)

    logger.info("Security headers middleware enabled")
    return middleware
