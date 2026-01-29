"""
CSRF Protection Middleware

Implements comprehensive CSRF (Cross-Site Request Forgery) protection with:
- Token generation and validation
- Configurable exemptions for webhooks and API keys
- Multiple token storage strategies (cookies, headers)
- Token rotation and expiration
- Double-submit cookie pattern

Created: 2026-01-27
Part of: Phase 3 Security Remediation (HIGH-3)
"""

import secrets
import hashlib
import hmac
import logging
from typing import Optional, List, Set, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class CSRFTokenStorage(str, Enum):
    """CSRF token storage strategies"""
    COOKIE = "cookie"
    HEADER = "header"
    DOUBLE_SUBMIT = "double_submit"  # Cookie + Header validation


@dataclass
class CSRFConfig:
    """CSRF protection configuration"""
    enabled: bool = True
    token_length: int = 32
    cookie_name: str = "csrf_token"
    header_name: str = "X-CSRF-Token"
    token_expiry_hours: int = 24
    storage_strategy: CSRFTokenStorage = CSRFTokenStorage.DOUBLE_SUBMIT

    # Methods requiring CSRF protection
    protected_methods: Set[str] = None

    # Paths exempt from CSRF (webhooks, public APIs)
    exempt_paths: List[str] = None

    # Cookie settings
    cookie_secure: bool = True  # HTTPS only
    cookie_httponly: bool = True  # Not accessible via JavaScript
    cookie_samesite: str = "Strict"  # Strict, Lax, or None
    cookie_domain: Optional[str] = None
    cookie_path: str = "/"

    # Secret key for HMAC validation
    secret_key: Optional[str] = None

    def __post_init__(self):
        """Initialize default values"""
        if self.protected_methods is None:
            self.protected_methods = {"POST", "PUT", "DELETE", "PATCH"}

        if self.exempt_paths is None:
            self.exempt_paths = [
                "/api/webhooks",
                "/api/health",
                "/health",
                "/metrics",
                "/api/auth/login",
                "/api/auth/register",
            ]

        if self.secret_key is None:
            # Generate a random secret key (should be set from environment)
            self.secret_key = secrets.token_hex(32)
            logger.warning("Using auto-generated CSRF secret key. Set CSRF_SECRET_KEY in production!")


class CSRFProtection:
    """
    CSRF Protection implementation

    Provides CSRF token generation, validation, and management
    using industry-standard patterns.
    """

    def __init__(self, config: Optional[CSRFConfig] = None):
        """
        Initialize CSRF protection

        Args:
            config: CSRF configuration (uses defaults if None)
        """
        self.config = config or CSRFConfig()

        if not self.config.secret_key or len(self.config.secret_key) < 32:
            raise ValueError("CSRF secret key must be at least 32 characters")

    def generate_token(self) -> str:
        """
        Generate a cryptographically secure CSRF token

        Returns:
            CSRF token as hex string
        """
        # Generate random token
        random_token = secrets.token_bytes(self.config.token_length)

        # Create HMAC signature for additional security
        signature = hmac.new(
            self.config.secret_key.encode(),
            random_token,
            hashlib.sha256
        ).digest()

        # Combine token and signature
        token = random_token + signature
        return token.hex()

    def validate_token(self, token: str, stored_token: Optional[str] = None) -> bool:
        """
        Validate a CSRF token

        Args:
            token: Token to validate
            stored_token: Expected token (for double-submit pattern)

        Returns:
            True if token is valid
        """
        try:
            if not token:
                return False

            # Decode token
            token_bytes = bytes.fromhex(token)

            if len(token_bytes) != self.config.token_length + 32:  # token + HMAC
                return False

            # Split token and signature
            random_token = token_bytes[:self.config.token_length]
            provided_signature = token_bytes[self.config.token_length:]

            # Verify HMAC signature
            expected_signature = hmac.new(
                self.config.secret_key.encode(),
                random_token,
                hashlib.sha256
            ).digest()

            if not hmac.compare_digest(provided_signature, expected_signature):
                return False

            # For double-submit pattern, compare with stored token
            if stored_token and self.config.storage_strategy == CSRFTokenStorage.DOUBLE_SUBMIT:
                return hmac.compare_digest(token, stored_token)

            return True

        except (ValueError, TypeError) as e:
            logger.debug(f"CSRF token validation error: {e}")
            return False

    def is_exempt_path(self, path: str) -> bool:
        """
        Check if a path is exempt from CSRF protection

        Args:
            path: Request path

        Returns:
            True if path is exempt
        """
        for exempt_path in self.config.exempt_paths:
            if path.startswith(exempt_path):
                return True
        return False

    def is_protected_method(self, method: str) -> bool:
        """
        Check if HTTP method requires CSRF protection

        Args:
            method: HTTP method

        Returns:
            True if method requires protection
        """
        return method.upper() in self.config.protected_methods

    def extract_token_from_request(self, request: Request) -> Optional[str]:
        """
        Extract CSRF token from request

        Args:
            request: FastAPI request object

        Returns:
            CSRF token or None
        """
        # Try header first
        token = request.headers.get(self.config.header_name)

        if not token and self.config.storage_strategy == CSRFTokenStorage.DOUBLE_SUBMIT:
            # Try cookie for double-submit pattern
            token = request.cookies.get(self.config.cookie_name)

        return token

    def set_token_cookie(self, response: Response, token: str):
        """
        Set CSRF token as cookie

        Args:
            response: FastAPI response object
            token: CSRF token to set
        """
        expires = datetime.utcnow() + timedelta(hours=self.config.token_expiry_hours)

        response.set_cookie(
            key=self.config.cookie_name,
            value=token,
            max_age=self.config.token_expiry_hours * 3600,
            expires=expires.strftime("%a, %d %b %Y %H:%M:%S GMT"),
            path=self.config.cookie_path,
            domain=self.config.cookie_domain,
            secure=self.config.cookie_secure,
            httponly=self.config.cookie_httponly,
            samesite=self.config.cookie_samesite
        )


class CSRFMiddleware(BaseHTTPMiddleware):
    """
    CSRF Protection Middleware

    Automatically validates CSRF tokens for protected methods
    and injects tokens into responses.
    """

    def __init__(self, app: ASGIApp, config: Optional[CSRFConfig] = None):
        """
        Initialize CSRF middleware

        Args:
            app: ASGI application
            config: CSRF configuration
        """
        super().__init__(app)
        self.csrf = CSRFProtection(config)

    async def dispatch(self, request: Request, call_next: Callable):
        """
        Process request with CSRF protection

        Args:
            request: Incoming request
            call_next: Next middleware/handler

        Returns:
            Response with CSRF protection applied
        """
        # Skip if CSRF is disabled
        if not self.csrf.config.enabled:
            return await call_next(request)

        # Check if path is exempt
        if self.csrf.is_exempt_path(request.url.path):
            return await call_next(request)

        # Check if method requires protection
        if self.csrf.is_protected_method(request.method):
            # Extract token from request
            token = self.csrf.extract_token_from_request(request)

            # Validate token
            stored_token = None
            if self.csrf.config.storage_strategy == CSRFTokenStorage.DOUBLE_SUBMIT:
                stored_token = request.cookies.get(self.csrf.config.cookie_name)

            if not self.csrf.validate_token(token or "", stored_token):
                logger.warning(
                    f"CSRF validation failed for {request.method} {request.url.path}",
                    extra={
                        "ip": request.client.host if request.client else "unknown",
                        "user_agent": request.headers.get("user-agent", "unknown")
                    }
                )

                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={
                        "success": False,
                        "error": "CSRF validation failed",
                        "detail": "Missing or invalid CSRF token",
                        "code": "CSRF_VALIDATION_FAILED"
                    }
                )

        # Process request
        response = await call_next(request)

        # Generate and set new token for GET requests
        if request.method == "GET" and not self.csrf.is_exempt_path(request.url.path):
            new_token = self.csrf.generate_token()
            self.csrf.set_token_cookie(response, new_token)

            # Also set in response header for JavaScript access
            response.headers[self.csrf.config.header_name] = new_token

        return response


# Convenience function to add CSRF protection to FastAPI app
def add_csrf_protection(
    app,
    secret_key: Optional[str] = None,
    exempt_paths: Optional[List[str]] = None,
    **kwargs
) -> CSRFMiddleware:
    """
    Add CSRF protection to FastAPI application

    Args:
        app: FastAPI application
        secret_key: Secret key for HMAC (from environment)
        exempt_paths: Additional paths to exempt from CSRF
        **kwargs: Additional CSRFConfig parameters

    Returns:
        CSRFMiddleware instance

    Example:
        ```python
        from fastapi import FastAPI
        from backend.security.csrf_protection import add_csrf_protection

        app = FastAPI()
        add_csrf_protection(
            app,
            secret_key=os.getenv("CSRF_SECRET_KEY"),
            exempt_paths=["/api/webhooks/stripe"]
        )
        ```
    """
    config_params = kwargs.copy()

    if secret_key:
        config_params["secret_key"] = secret_key

    if exempt_paths:
        default_exemptions = CSRFConfig().exempt_paths
        config_params["exempt_paths"] = default_exemptions + exempt_paths

    config = CSRFConfig(**config_params)
    middleware = CSRFMiddleware(app, config)
    app.add_middleware(CSRFMiddleware, config=config)

    logger.info(f"CSRF protection enabled with {config.storage_strategy} strategy")
    return middleware
