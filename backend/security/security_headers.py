"""
Comprehensive Security Headers and CORS Configuration System
Implements security headers, CORS policies, and HTTP security best practices
"""

import os
import json
from typing import Dict, List, Optional, Union, Set, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
import logging
from urllib.parse import urlparse

# FastAPI imports
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class CSPDirective(str, Enum):
    """Content Security Policy directives"""
    DEFAULT_SRC = "default-src"
    SCRIPT_SRC = "script-src"
    STYLE_SRC = "style-src"
    IMG_SRC = "img-src"
    FONT_SRC = "font-src"
    CONNECT_SRC = "connect-src"
    MEDIA_SRC = "media-src"
    OBJECT_SRC = "object-src"
    CHILD_SRC = "child-src"
    FRAME_SRC = "frame-src"
    WORKER_SRC = "worker-src"
    FRAME_ANCESTORS = "frame-ancestors"
    FORM_ACTION = "form-action"
    BASE_URI = "base-uri"
    PLUGIN_TYPES = "plugin-types"
    SANDBOX = "sandbox"
    UPGRADE_INSECURE_REQUESTS = "upgrade-insecure-requests"
    BLOCK_ALL_MIXED_CONTENT = "block-all-mixed-content"


class ReferrerPolicy(str, Enum):
    """Referrer Policy options"""
    NO_REFERRER = "no-referrer"
    NO_REFERRER_WHEN_DOWNGRADE = "no-referrer-when-downgrade"
    ORIGIN = "origin"
    ORIGIN_WHEN_CROSS_ORIGIN = "origin-when-cross-origin"
    SAME_ORIGIN = "same-origin"
    STRICT_ORIGIN = "strict-origin"
    STRICT_ORIGIN_WHEN_CROSS_ORIGIN = "strict-origin-when-cross-origin"
    UNSAFE_URL = "unsafe-url"


class SameSite(str, Enum):
    """SameSite cookie attribute options"""
    STRICT = "Strict"
    LAX = "Lax"
    NONE = "None"


@dataclass
class SecurityHeadersConfig:
    """Configuration for security headers"""
    # HSTS Configuration
    hsts_max_age: int = 31536000  # 1 year
    hsts_include_subdomains: bool = True
    hsts_preload: bool = True
    
    # CSP Configuration
    csp_directives: Dict[str, List[str]] = None
    csp_report_uri: Optional[str] = None
    csp_report_only: bool = False
    
    # Frame Options
    frame_options: str = "DENY"  # DENY, SAMEORIGIN, or ALLOW-FROM url
    
    # Content Type Options
    content_type_options: bool = True
    
    # XSS Protection
    xss_protection: str = "1; mode=block"
    
    # Referrer Policy
    referrer_policy: ReferrerPolicy = ReferrerPolicy.STRICT_ORIGIN_WHEN_CROSS_ORIGIN
    
    # Permissions Policy
    permissions_policy: Dict[str, List[str]] = None
    
    # Feature Policy (deprecated, use Permissions Policy)
    feature_policy: Dict[str, List[str]] = None
    
    # Custom headers
    custom_headers: Dict[str, str] = None
    
    def __post_init__(self):
        if self.csp_directives is None:
            self.csp_directives = self._get_default_csp()
        
        if self.permissions_policy is None:
            self.permissions_policy = self._get_default_permissions_policy()
        
        if self.custom_headers is None:
            self.custom_headers = {}
    
    def _get_default_csp(self) -> Dict[str, List[str]]:
        """Get default Content Security Policy"""
        return {
            CSPDirective.DEFAULT_SRC: ["'self'"],
            CSPDirective.SCRIPT_SRC: ["'self'", "'unsafe-inline'", "'unsafe-eval'"],
            CSPDirective.STYLE_SRC: ["'self'", "'unsafe-inline'"],
            CSPDirective.IMG_SRC: ["'self'", "data:", "https:"],
            CSPDirective.FONT_SRC: ["'self'", "data:", "https:"],
            CSPDirective.CONNECT_SRC: ["'self'", "wss:", "ws:"],
            CSPDirective.MEDIA_SRC: ["'self'"],
            CSPDirective.OBJECT_SRC: ["'none'"],
            CSPDirective.FRAME_SRC: ["'none'"],
            CSPDirective.FRAME_ANCESTORS: ["'none'"],
            CSPDirective.FORM_ACTION: ["'self'"],
            CSPDirective.BASE_URI: ["'self'"],
            CSPDirective.UPGRADE_INSECURE_REQUESTS: []
        }
    
    def _get_default_permissions_policy(self) -> Dict[str, List[str]]:
        """Get default Permissions Policy"""
        return {
            "geolocation": [],
            "microphone": [],
            "camera": [],
            "payment": [],
            "usb": [],
            "magnetometer": [],
            "accelerometer": [],
            "gyroscope": [],
            "fullscreen": ["'self'"],
            "autoplay": []
        }


@dataclass
class CORSConfig:
    """Configuration for CORS (Cross-Origin Resource Sharing)"""
    allowed_origins: List[str] = None
    allowed_origin_regex: Optional[str] = None
    allowed_methods: List[str] = None
    allowed_headers: List[str] = None
    exposed_headers: List[str] = None
    allow_credentials: bool = False
    max_age: int = 86400  # 24 hours
    
    def __post_init__(self):
        if self.allowed_origins is None:
            self.allowed_origins = ["http://localhost:3000", "http://localhost:8000"]
        
        if self.allowed_methods is None:
            self.allowed_methods = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]
        
        if self.allowed_headers is None:
            self.allowed_headers = [
                "Accept",
                "Accept-Language",
                "Content-Type",
                "Authorization",
                "X-Requested-With",
                "X-CSRF-Token",
                "X-Session-ID"
            ]
        
        if self.exposed_headers is None:
            self.exposed_headers = [
                "X-RateLimit-Limit",
                "X-RateLimit-Remaining",
                "X-RateLimit-Reset"
            ]


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for adding comprehensive security headers"""
    
    def __init__(self, app: ASGIApp, config: SecurityHeadersConfig = None):
        super().__init__(app)
        self.config = config or SecurityHeadersConfig()
    
    async def dispatch(self, request: Request, call_next) -> StarletteResponse:
        """Add security headers to response"""
        response = await call_next(request)
        
        # Add security headers
        self._add_hsts_header(response, request)
        self._add_csp_header(response)
        self._add_frame_options_header(response)
        self._add_content_type_options_header(response)
        self._add_xss_protection_header(response)
        self._add_referrer_policy_header(response)
        self._add_permissions_policy_header(response)
        self._add_feature_policy_header(response)
        self._add_custom_headers(response)
        
        return response
    
    def _add_hsts_header(self, response: StarletteResponse, request: Request):
        """Add HTTP Strict Transport Security header"""
        # Only add HSTS for HTTPS requests
        if request.url.scheme == "https":
            hsts_value = f"max-age={self.config.hsts_max_age}"
            
            if self.config.hsts_include_subdomains:
                hsts_value += "; includeSubDomains"
            
            if self.config.hsts_preload:
                hsts_value += "; preload"
            
            response.headers["Strict-Transport-Security"] = hsts_value
    
    def _add_csp_header(self, response: StarletteResponse):
        """Add Content Security Policy header"""
        csp_parts = []
        
        for directive, sources in self.config.csp_directives.items():
            if sources:
                csp_parts.append(f"{directive} {' '.join(sources)}")
            else:
                # Directives like upgrade-insecure-requests don't need sources
                csp_parts.append(directive)
        
        if self.config.csp_report_uri:
            csp_parts.append(f"report-uri {self.config.csp_report_uri}")
        
        csp_header = "; ".join(csp_parts)
        
        if self.config.csp_report_only:
            response.headers["Content-Security-Policy-Report-Only"] = csp_header
        else:
            response.headers["Content-Security-Policy"] = csp_header
    
    def _add_frame_options_header(self, response: StarletteResponse):
        """Add X-Frame-Options header"""
        if self.config.frame_options:
            response.headers["X-Frame-Options"] = self.config.frame_options
    
    def _add_content_type_options_header(self, response: StarletteResponse):
        """Add X-Content-Type-Options header"""
        if self.config.content_type_options:
            response.headers["X-Content-Type-Options"] = "nosniff"
    
    def _add_xss_protection_header(self, response: StarletteResponse):
        """Add X-XSS-Protection header"""
        if self.config.xss_protection:
            response.headers["X-XSS-Protection"] = self.config.xss_protection
    
    def _add_referrer_policy_header(self, response: StarletteResponse):
        """Add Referrer-Policy header"""
        response.headers["Referrer-Policy"] = self.config.referrer_policy.value
    
    def _add_permissions_policy_header(self, response: StarletteResponse):
        """Add Permissions-Policy header"""
        if self.config.permissions_policy:
            policy_parts = []
            for directive, allowlist in self.config.permissions_policy.items():
                if allowlist:
                    allowlist_str = " ".join(f'"{origin}"' if origin.startswith("'") else f'"{origin}"' for origin in allowlist)
                    policy_parts.append(f"{directive}=({allowlist_str})")
                else:
                    policy_parts.append(f"{directive}=()")
            
            if policy_parts:
                response.headers["Permissions-Policy"] = ", ".join(policy_parts)
    
    def _add_feature_policy_header(self, response: StarletteResponse):
        """Add Feature-Policy header (deprecated, but for legacy support)"""
        if self.config.feature_policy:
            policy_parts = []
            for directive, allowlist in self.config.feature_policy.items():
                if allowlist:
                    allowlist_str = " ".join(allowlist)
                    policy_parts.append(f"{directive} {allowlist_str}")
                else:
                    policy_parts.append(f"{directive} 'none'")
            
            if policy_parts:
                response.headers["Feature-Policy"] = "; ".join(policy_parts)
    
    def _add_custom_headers(self, response: StarletteResponse):
        """Add custom security headers"""
        for header, value in self.config.custom_headers.items():
            response.headers[header] = value


class CORSValidator:
    """CORS request validator"""
    
    def __init__(self, config: CORSConfig):
        self.config = config
    
    def is_origin_allowed(self, origin: str) -> bool:
        """Check if origin is allowed"""
        if not origin:
            return True  # Same-origin requests don't have Origin header
        
        # Check exact matches
        if origin in self.config.allowed_origins:
            return True
        
        # Check wildcard
        if "*" in self.config.allowed_origins:
            return True
        
        # Check regex pattern if configured
        if self.config.allowed_origin_regex:
            import re
            pattern = re.compile(self.config.allowed_origin_regex)
            if pattern.match(origin):
                return True
        
        return False
    
    def get_allowed_methods(self, request_method: str = None) -> List[str]:
        """Get allowed methods for the request"""
        return self.config.allowed_methods
    
    def get_allowed_headers(self, requested_headers: List[str] = None) -> List[str]:
        """Get allowed headers for the request"""
        if not requested_headers:
            return self.config.allowed_headers
        
        # Filter requested headers to only allowed ones
        allowed = []
        for header in requested_headers:
            if header.lower() in [h.lower() for h in self.config.allowed_headers]:
                allowed.append(header)
        
        return allowed


class EnhancedCORSMiddleware(BaseHTTPMiddleware):
    """Enhanced CORS middleware with additional security features"""
    
    def __init__(self, app: ASGIApp, config: CORSConfig = None):
        super().__init__(app)
        self.config = config or CORSConfig()
        self.validator = CORSValidator(self.config)
    
    async def dispatch(self, request: Request, call_next) -> StarletteResponse:
        """Handle CORS with enhanced security"""
        origin = request.headers.get("Origin")
        
        # Handle preflight requests
        if request.method == "OPTIONS" and origin:
            return self._handle_preflight_request(request, origin)
        
        # Process actual request
        response = await call_next(request)
        
        # Add CORS headers to response
        self._add_cors_headers(response, request, origin)
        
        return response
    
    def _handle_preflight_request(self, request: Request, origin: str) -> StarletteResponse:
        """Handle CORS preflight request"""
        # Check if origin is allowed
        if not self.validator.is_origin_allowed(origin):
            return StarletteResponse(status_code=403, content="CORS: Origin not allowed")
        
        # Get requested method and headers
        requested_method = request.headers.get("Access-Control-Request-Method")
        requested_headers = self._parse_header_list(
            request.headers.get("Access-Control-Request-Headers", "")
        )
        
        # Validate requested method
        if requested_method and requested_method not in self.config.allowed_methods:
            return StarletteResponse(status_code=403, content="CORS: Method not allowed")
        
        # Create preflight response
        response = StarletteResponse(status_code=204)
        
        # Add CORS headers
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Methods"] = ", ".join(self.config.allowed_methods)
        
        allowed_headers = self.validator.get_allowed_headers(requested_headers)
        if allowed_headers:
            response.headers["Access-Control-Allow-Headers"] = ", ".join(allowed_headers)
        
        if self.config.allow_credentials:
            response.headers["Access-Control-Allow-Credentials"] = "true"
        
        response.headers["Access-Control-Max-Age"] = str(self.config.max_age)
        
        return response
    
    def _add_cors_headers(self, response: StarletteResponse, request: Request, origin: str):
        """Add CORS headers to actual response"""
        if not origin:
            return
        
        # Check if origin is allowed
        if not self.validator.is_origin_allowed(origin):
            return
        
        # Add CORS headers
        response.headers["Access-Control-Allow-Origin"] = origin
        
        if self.config.exposed_headers:
            response.headers["Access-Control-Expose-Headers"] = ", ".join(self.config.exposed_headers)
        
        if self.config.allow_credentials:
            response.headers["Access-Control-Allow-Credentials"] = "true"
        
        # Add Vary header for caching
        vary_headers = []
        if "Vary" in response.headers:
            vary_headers = self._parse_header_list(response.headers["Vary"])
        
        if "Origin" not in vary_headers:
            vary_headers.append("Origin")
            response.headers["Vary"] = ", ".join(vary_headers)
    
    def _parse_header_list(self, header_value: str) -> List[str]:
        """Parse comma-separated header list"""
        if not header_value:
            return []
        return [h.strip() for h in header_value.split(",") if h.strip()]


class SecureCookieManager:
    """Secure cookie management utilities"""
    
    def __init__(self, secure: bool = True, http_only: bool = True, same_site: SameSite = SameSite.LAX):
        self.secure = secure
        self.http_only = http_only
        self.same_site = same_site
    
    def set_secure_cookie(
        self,
        response: StarletteResponse,
        key: str,
        value: str,
        max_age: int = None,
        expires: datetime = None,
        path: str = "/",
        domain: str = None,
        secure: bool = None,
        http_only: bool = None,
        same_site: SameSite = None
    ):
        """Set a secure cookie with appropriate security attributes"""
        
        # Use instance defaults if not specified
        if secure is None:
            secure = self.secure
        if http_only is None:
            http_only = self.http_only
        if same_site is None:
            same_site = self.same_site
        
        # Set the cookie
        response.set_cookie(
            key=key,
            value=value,
            max_age=max_age,
            expires=expires,
            path=path,
            domain=domain,
            secure=secure,
            httponly=http_only,
            samesite=same_site.value
        )


def setup_security_headers(app: FastAPI, config: SecurityHeadersConfig = None):
    """Setup security headers middleware"""
    config = config or get_production_security_config()
    app.add_middleware(SecurityHeadersMiddleware, config=config)


def setup_cors(app: FastAPI, config: CORSConfig = None):
    """Setup CORS middleware"""
    config = config or get_production_cors_config()
    app.add_middleware(EnhancedCORSMiddleware, config=config)


def get_development_security_config() -> SecurityHeadersConfig:
    """Get security configuration for development environment"""
    return SecurityHeadersConfig(
        hsts_max_age=0,  # Disable HSTS in development
        hsts_include_subdomains=False,
        hsts_preload=False,
        csp_directives={
            CSPDirective.DEFAULT_SRC: ["'self'"],
            CSPDirective.SCRIPT_SRC: ["'self'", "'unsafe-inline'", "'unsafe-eval'", "localhost:*"],
            CSPDirective.STYLE_SRC: ["'self'", "'unsafe-inline'", "localhost:*"],
            CSPDirective.IMG_SRC: ["'self'", "data:", "localhost:*"],
            CSPDirective.FONT_SRC: ["'self'", "data:", "localhost:*"],
            CSPDirective.CONNECT_SRC: ["'self'", "ws://localhost:*", "wss://localhost:*", "localhost:*"],
            CSPDirective.FRAME_ANCESTORS: ["'self'"]
        },
        frame_options="SAMEORIGIN",
        csp_report_only=True  # Use report-only mode in development
    )


def get_production_security_config() -> SecurityHeadersConfig:
    """Get security configuration for production environment"""
    return SecurityHeadersConfig(
        hsts_max_age=31536000,  # 1 year
        hsts_include_subdomains=True,
        hsts_preload=True,
        csp_directives={
            CSPDirective.DEFAULT_SRC: ["'self'"],
            CSPDirective.SCRIPT_SRC: ["'self'", "'unsafe-inline'"],  # Remove unsafe-eval in production
            CSPDirective.STYLE_SRC: ["'self'", "'unsafe-inline'"],
            CSPDirective.IMG_SRC: ["'self'", "data:", "https:"],
            CSPDirective.FONT_SRC: ["'self'", "data:"],
            CSPDirective.CONNECT_SRC: ["'self'", "wss:", "https:"],
            CSPDirective.MEDIA_SRC: ["'self'"],
            CSPDirective.OBJECT_SRC: ["'none'"],
            CSPDirective.FRAME_SRC: ["'none'"],
            CSPDirective.FRAME_ANCESTORS: ["'none'"],
            CSPDirective.FORM_ACTION: ["'self'"],
            CSPDirective.BASE_URI: ["'self'"],
            CSPDirective.UPGRADE_INSECURE_REQUESTS: []
        },
        frame_options="DENY",
        permissions_policy={
            "geolocation": [],
            "microphone": [],
            "camera": [],
            "payment": [],
            "usb": [],
            "magnetometer": [],
            "accelerometer": [],
            "gyroscope": [],
            "fullscreen": ["'self'"],
            "autoplay": []
        }
    )


def get_development_cors_config() -> CORSConfig:
    """Get CORS configuration for development environment"""
    return CORSConfig(
        allowed_origins=[
            "http://localhost:3000",
            "http://localhost:3001",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:3001"
        ],
        allowed_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allowed_headers=[
            "Accept",
            "Accept-Language",
            "Content-Type",
            "Authorization",
            "X-Requested-With",
            "X-CSRF-Token",
            "X-Session-ID"
        ],
        allow_credentials=True,
        max_age=3600  # 1 hour
    )


def get_production_cors_config() -> CORSConfig:
    """Get CORS configuration for production environment"""
    # In production, get allowed origins from environment
    allowed_origins = os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")
    allowed_origins = [origin.strip() for origin in allowed_origins if origin.strip()]
    
    if not allowed_origins:
        # Fallback to secure defaults
        allowed_origins = ["https://yourdomain.com", "https://api.yourdomain.com"]
    
    return CORSConfig(
        allowed_origins=allowed_origins,
        allowed_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allowed_headers=[
            "Accept",
            "Accept-Language",
            "Content-Type",
            "Authorization",
            "X-Requested-With",
            "X-CSRF-Token"
        ],
        exposed_headers=[
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset"
        ],
        allow_credentials=True,
        max_age=86400  # 24 hours
    )


class SecurityAuditLogger:
    """Log security-related events for auditing"""
    
    def __init__(self):
        self.logger = logging.getLogger("security_audit")
    
    def log_cors_violation(self, origin: str, path: str, ip: str):
        """Log CORS violation"""
        self.logger.warning(
            f"CORS violation: Origin '{origin}' not allowed for path '{path}' from IP {ip}"
        )
    
    def log_csp_violation(self, report: Dict):
        """Log CSP violation report"""
        self.logger.warning(f"CSP violation report: {json.dumps(report, indent=2)}")
    
    def log_security_header_missing(self, header: str, path: str):
        """Log missing security header"""
        self.logger.info(f"Security header '{header}' not set for path '{path}'")


# Global instances
_security_audit_logger = SecurityAuditLogger()


def get_security_audit_logger() -> SecurityAuditLogger:
    """Get global security audit logger"""
    return _security_audit_logger


def get_security_config() -> SecurityHeadersConfig:
    """Get security configuration based on environment.

    Alias function for backwards compatibility with security_config.py imports.
    Returns production config by default, development config in dev environments.
    """
    import os
    environment = os.getenv("ENVIRONMENT", "production").lower()
    if environment in ["development", "dev", "local"]:
        return get_development_security_config()
    return get_production_security_config()