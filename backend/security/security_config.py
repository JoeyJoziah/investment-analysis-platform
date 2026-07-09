"""
Enhanced Security Configuration and Hardening System
Provides comprehensive security configurations for different environments
"""

import os
import secrets
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.sessions import SessionMiddleware
import redis

from backend.config.settings import settings
from .security_headers import SecurityHeadersMiddleware, get_security_config as get_headers_config
from .input_validation import ValidationMiddleware
from .advanced_rate_limiter import RateLimitingMiddleware, get_default_rate_limiting_rules
from .injection_prevention import InjectionPreventionMiddleware
from .audit_logging import AuditMiddleware, get_audit_logger

logger = logging.getLogger(__name__)


class Environment(str, Enum):
    """Application environments"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class InsecureSecretError(RuntimeError):
    """Raised when a required secret is missing in a production environment.

    Finding #201: Production must never fall back to an ephemeral, in-process
    secret (e.g. ``secrets.token_urlsafe(32)``). Ephemeral secrets silently
    invalidate tokens/sessions across restarts and across worker processes, and
    are not covered by the production assertions in settings.py.
    """


def _is_production_environment() -> bool:
    """Return True when the application is running in production.

    Reads the live ``ENVIRONMENT`` value rather than a snapshot so the check is
    correct under monkeypatched environments in tests.
    """
    return os.getenv("ENVIRONMENT", settings.ENVIRONMENT).strip().lower() == "production"


def _require_secret(env_var: str, dev_default: str) -> str:
    """Resolve a required secret with fail-fast behaviour in production.

    In production a missing/empty environment variable raises
    ``InsecureSecretError`` at import/startup time instead of silently
    generating an ephemeral value. Outside production a stable, clearly-marked
    development default is returned so local/test runs do not require secrets.

    Args:
        env_var: Name of the environment variable holding the secret.
        dev_default: Deterministic placeholder used ONLY outside production.

    Returns:
        The configured secret string.

    Raises:
        InsecureSecretError: If running in production and the secret is unset.
    """
    value = os.getenv(env_var)
    if value:
        return value

    if _is_production_environment():
        raise InsecureSecretError(
            f"{env_var} must be set in production. Refusing to start with an "
            f"ephemeral auto-generated secret, which would invalidate tokens and "
            f"sessions across restarts and worker processes."
        )

    logger.warning(
        "%s is not set; using an insecure development default. "
        "This is acceptable only outside production.",
        env_var,
    )
    return dev_default


class SecurityConfig:
    """Security configuration and hardening settings"""

    # HTTPS Settings
    FORCE_HTTPS = os.getenv("FORCE_HTTPS", "false").lower() == "true"

    # CORS Settings
    ALLOWED_ORIGINS = [
        "http://localhost:3000",
        "https://investment-analysis.com",
        "https://api.investment-analysis.com"
    ]

    if settings.ENVIRONMENT == "development":
        ALLOWED_ORIGINS.extend([
            "http://localhost:8000",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8000"
        ])

    ALLOWED_METHODS = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    ALLOWED_HEADERS = [
        "Authorization",
        "Content-Type",
        "X-Requested-With",
        "X-API-Key"
    ]

    # Session Settings
    # Finding #201: fail fast in production instead of generating an ephemeral
    # secret that would invalidate sessions across restarts/workers.
    SESSION_SECRET_KEY = _require_secret(
        "SESSION_SECRET_KEY",
        "dev-only-insecure-session-secret-do-not-use-in-production",
    )
    SESSION_MAX_AGE = 3600  # 1 hour

    # Rate Limiting
    RATE_LIMIT_ENABLED = True
    DEFAULT_RATE_LIMIT = "100/hour"
    STRICT_RATE_LIMIT = "10/minute"

    # Security Headers
    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' data:; "
            "connect-src 'self' ws: wss:"
        ),
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": (
            "geolocation=(), "
            "microphone=(), "
            "camera=(), "
            "payment=(), "
            "usb=(), "
            "magnetometer=(), "
            "gyroscope=(), "
            "accelerometer=()"
        )
    }

    # Password Policy
    PASSWORD_MIN_LENGTH = 12
    PASSWORD_REQUIRE_UPPERCASE = True
    PASSWORD_REQUIRE_LOWERCASE = True
    PASSWORD_REQUIRE_DIGITS = True
    PASSWORD_REQUIRE_SPECIAL = True
    PASSWORD_MAX_AGE_DAYS = 90

    # ==========================================================================
    # JWT Settings - SINGLE SOURCE OF TRUTH
    # ==========================================================================
    # All JWT configuration should be read from this class.
    # Do NOT define JWT settings elsewhere in the codebase.
    #
    # Supported algorithms:
    #   - HS256: HMAC with SHA-256 (symmetric, uses secret key)
    #   - RS256: RSA with SHA-256 (asymmetric, uses private/public key pair)
    #
    # The jwt_manager.py uses RS256 with RSA key pairs for enhanced security.
    # Fallback/legacy code may use HS256 with JWT_SECRET_KEY.
    # ==========================================================================

    # Primary algorithm for new tokens (RS256 recommended for production)
    JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "RS256")

    # Fallback algorithm for legacy compatibility
    JWT_ALGORITHM_FALLBACK = "HS256"

    # Access token expiration (short-lived for security)
    # Default: 30 minutes - balances security with user experience
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

    # Refresh token expiration (longer-lived for convenience)
    # Default: 7 days - allows users to stay logged in
    JWT_REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "7"))

    # MFA token expiration (very short-lived for security)
    # Default: 5 minutes - MFA verification should be quick
    JWT_MFA_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_MFA_TOKEN_EXPIRE_MINUTES", "5"))

    # Reset token expiration (for password reset flows)
    # Default: 15 minutes - balances security with user experience
    JWT_RESET_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_RESET_TOKEN_EXPIRE_MINUTES", "15"))

    # Secret key for HS256 algorithm (fallback/legacy)
    # In production, this MUST be set via environment variable.
    # Finding #201: fail fast in production instead of generating an ephemeral
    # secret that would invalidate tokens across restarts/workers.
    JWT_SECRET_KEY = _require_secret(
        "JWT_SECRET_KEY",
        "dev-only-insecure-jwt-secret-do-not-use-in-production",
    )

    # Token issuer and audience for validation
    JWT_ISSUER = "investment-analysis-app"
    JWT_AUDIENCE = "investment-analysis-users"

    # API Key Settings
    API_KEY_LENGTH = 32
    API_KEY_PREFIX = "sk_"

    # File Upload Security
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_FILE_TYPES = [".csv", ".json", ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".txt", ".xls", ".xlsx"]
    UPLOAD_SCAN_ENABLED = True

    # MIME Type Allowlist - maps extensions to allowed MIME types
    ALLOWED_MIME_TYPES: Dict[str, List[str]] = {
        ".pdf": ["application/pdf"],
        ".jpg": ["image/jpeg"],
        ".jpeg": ["image/jpeg"],
        ".png": ["image/png"],
        ".gif": ["image/gif"],
        ".csv": ["text/csv", "text/plain", "application/csv"],
        ".txt": ["text/plain"],
        ".json": ["application/json", "text/json", "text/plain"],
        ".xls": ["application/vnd.ms-excel", "application/x-msexcel"],
        ".xlsx": ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"],
    }

    # Magic bytes (file signatures) for content-based detection
    FILE_SIGNATURES: Dict[str, List[bytes]] = {
        "application/pdf": [b"%PDF"],
        "image/jpeg": [b"\xff\xd8\xff"],
        "image/png": [b"\x89PNG\r\n\x1a\n"],
        "image/gif": [b"GIF87a", b"GIF89a"],
        "application/vnd.ms-excel": [b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"],  # OLE Compound Document
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [b"PK\x03\x04"],  # ZIP-based
    }

    # Database Security
    DB_CONNECTION_TIMEOUT = 30
    DB_MAX_CONNECTIONS = 20
    DB_SSL_REQUIRE = os.getenv("DB_SSL_REQUIRE", "false").lower() == "true"

    # Audit Settings
    AUDIT_LOG_RETENTION_DAYS = 2555  # 7 years for SEC compliance
    AUDIT_LOG_ENCRYPTION = True

    # IP Filtering
    BLOCKED_IPS: List[str] = []
    ALLOWED_IPS: Optional[List[str]] = None  # None = allow all

    # Trusted Hosts
    TRUSTED_HOSTS = [
        "localhost",
        "127.0.0.1",
        "testserver",  # For testing with FastAPI TestClient
        "investment-analysis.com",
        "api.investment-analysis.com"
    ]

    # Redis Health Check Settings
    REDIS_HEALTH_CHECK_MAX_RETRIES = 3
    REDIS_HEALTH_CHECK_BASE_DELAY = 1.0  # seconds
    REDIS_HEALTH_CHECK_TIMEOUT = 5  # seconds


class RedisHealthCheckError(Exception):
    """Raised when Redis health check fails after all retries."""
    pass


class RedisHealthChecker:
    """
    Redis connectivity validator with exponential backoff retry.

    Validates Redis is available at startup to ensure rate limiting
    and other critical security features will function correctly.
    """

    def __init__(
        self,
        redis_url: str,
        max_retries: int = SecurityConfig.REDIS_HEALTH_CHECK_MAX_RETRIES,
        base_delay: float = SecurityConfig.REDIS_HEALTH_CHECK_BASE_DELAY,
        timeout: int = SecurityConfig.REDIS_HEALTH_CHECK_TIMEOUT
    ):
        """
        Initialize Redis health checker.

        Args:
            redis_url: Redis connection URL
            max_retries: Maximum number of connection attempts
            base_delay: Base delay in seconds for exponential backoff
            timeout: Connection timeout in seconds
        """
        self.redis_url = redis_url
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.timeout = timeout

    def check_health(self) -> Dict[str, Any]:
        """
        Perform Redis health check with exponential backoff retry.

        Returns:
            Dict with health check results:
                - healthy: bool - whether Redis is reachable
                - latency_ms: float - ping latency in milliseconds
                - attempts: int - number of attempts made
                - error: Optional[str] - error message if unhealthy
        """
        result = {
            "healthy": False,
            "latency_ms": None,
            "attempts": 0,
            "error": None,
            "redis_url": self._mask_redis_url(self.redis_url)
        }

        last_error = None

        for attempt in range(1, self.max_retries + 1):
            result["attempts"] = attempt
            delay = self.base_delay * (2 ** (attempt - 1))  # 1s, 2s, 4s

            try:
                logger.info(
                    f"Redis health check attempt {attempt}/{self.max_retries} "
                    f"(timeout={self.timeout}s)"
                )

                start_time = time.time()
                client = redis.from_url(
                    self.redis_url,
                    socket_timeout=self.timeout,
                    socket_connect_timeout=self.timeout
                )

                # Execute PING command to verify connectivity
                response = client.ping()
                latency = (time.time() - start_time) * 1000  # Convert to ms

                if response:
                    result["healthy"] = True
                    result["latency_ms"] = round(latency, 2)

                    # Get additional Redis info for logging
                    try:
                        info = client.info("server")
                        redis_version = info.get("redis_version", "unknown")
                        logger.info(
                            f"Redis health check PASSED: version={redis_version}, "
                            f"latency={result['latency_ms']}ms, attempts={attempt}"
                        )
                    except redis.RedisError:
                        logger.info(
                            f"Redis health check PASSED: latency={result['latency_ms']}ms, "
                            f"attempts={attempt}"
                        )

                    client.close()
                    return result

            except redis.ConnectionError as e:
                last_error = f"Connection failed: {str(e)}"
                logger.warning(
                    f"Redis health check attempt {attempt} failed: {last_error}"
                )

            except redis.TimeoutError as e:
                last_error = f"Connection timed out: {str(e)}"
                logger.warning(
                    f"Redis health check attempt {attempt} timed out after {self.timeout}s"
                )

            except redis.AuthenticationError as e:
                last_error = f"Authentication failed: {str(e)}"
                logger.error(f"Redis authentication failed: {last_error}")
                # Don't retry on auth errors - it won't help
                break

            except redis.RedisError as e:
                last_error = f"Redis error: {str(e)}"
                logger.warning(
                    f"Redis health check attempt {attempt} error: {last_error}"
                )

            # Wait before next attempt (except on last attempt)
            if attempt < self.max_retries:
                logger.info(f"Retrying Redis connection in {delay}s...")
                time.sleep(delay)

        result["error"] = last_error
        logger.error(
            f"Redis health check FAILED after {result['attempts']} attempts: {last_error}"
        )

        return result

    def _mask_redis_url(self, url: str) -> str:
        """Mask password in Redis URL for safe logging."""
        if "@" in url:
            # URL format: redis://[:password]@host:port/db
            parts = url.split("@")
            if len(parts) == 2:
                auth_part = parts[0]
                host_part = parts[1]
                # Mask the password portion
                if ":" in auth_part:
                    protocol_and_user = auth_part.rsplit(":", 1)[0]
                    return f"{protocol_and_user}:****@{host_part}"
        return url


def validate_redis_connectivity(
    redis_url: str,
    environment: Environment,
    fail_on_error: bool = True
) -> Tuple[bool, Optional[str]]:
    """
    Validate Redis connectivity at startup.

    This function should be called during application initialization to ensure
    Redis is available for rate limiting and other security features.

    Args:
        redis_url: Redis connection URL
        environment: Current environment (production, development, etc.)
        fail_on_error: If True, raise exception on failure in production

    Returns:
        Tuple of (is_healthy, error_message)

    Raises:
        RedisHealthCheckError: If Redis is unavailable in production with fail_on_error=True
    """
    checker = RedisHealthChecker(redis_url)
    result = checker.check_health()

    if result["healthy"]:
        return True, None

    error_msg = (
        f"Redis health check failed: {result['error']}. "
        f"Rate limiting and session storage may not function correctly."
    )

    if environment == Environment.PRODUCTION:
        if fail_on_error:
            logger.critical(
                f"CRITICAL: Redis is unavailable in production. "
                f"Rate limiting is DISABLED. Error: {result['error']}"
            )
            raise RedisHealthCheckError(
                f"Redis is required in production but is unavailable: {result['error']}"
            )
        else:
            logger.error(error_msg)
    else:
        # In development/testing, log warning but allow fallback
        logger.warning(
            f"Redis unavailable in {environment.value} environment. "
            f"Rate limiting will fall back to in-memory storage. "
            f"Error: {result['error']}"
        )

    return False, result["error"]


def add_comprehensive_security_middleware(app: FastAPI) -> None:
    """Add comprehensive security middleware stack to FastAPI app"""

    environment = Environment(settings.ENVIRONMENT.lower())
    is_testing = os.getenv("TESTING", "False").lower() == "true"

    # 1. Audit logging middleware (first to capture everything)
    # Skip in testing mode to prevent AsyncClient compatibility issues
    if not is_testing:
        app.add_middleware(AuditMiddleware)

    # 2. Security headers middleware
    headers_config = get_headers_config()
    app.add_middleware(SecurityHeadersMiddleware, config=headers_config)

    # 3. Rate limiting and DDoS protection
    # First, validate Redis connectivity (critical for rate limiting)
    rate_limit_rules = get_default_rate_limiting_rules()
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # Perform Redis health check with environment-specific behavior:
    # - Production: Fail fast if Redis unavailable (rate limiting is critical)
    # - Development/Testing: Warn but allow fallback to in-memory cache
    redis_healthy, redis_error = validate_redis_connectivity(
        redis_url=redis_url,
        environment=environment,
        fail_on_error=(environment == Environment.PRODUCTION)
    )

    if redis_healthy:
        logger.info("Redis connectivity validated for rate limiting middleware")
    else:
        if environment == Environment.PRODUCTION:
            # In production, this would have raised an exception above
            # This branch handles fail_on_error=False case
            logger.error(
                f"Redis unavailable in production - rate limiting may fail: {redis_error}"
            )
        else:
            logger.warning(
                f"Redis unavailable ({redis_error}). "
                f"Rate limiting middleware will use in-memory fallback cache. "
                f"This is acceptable for {environment.value} but NOT for production."
            )

    # Skip rate limiting in testing mode to avoid Redis dependency
    if not is_testing:
        app.add_middleware(RateLimitingMiddleware, rules=rate_limit_rules, redis_url=redis_url)

    # 4. Input validation and sanitization (skip in testing mode to avoid stream issues)
    if not is_testing:
        app.add_middleware(ValidationMiddleware)

    # 5. Injection prevention (SQL, XSS, etc.)
    # Skip in testing mode to avoid AsyncClient stream compatibility issues
    if not is_testing:
        app.add_middleware(InjectionPreventionMiddleware,
                          enable_sql_protection=True,
                          enable_xss_protection=True,
                          enable_csrf_protection=True,
                          strict_mode=(environment == Environment.PRODUCTION))

    # 6. HTTPS redirect (production only)
    if SecurityConfig.FORCE_HTTPS and environment == Environment.PRODUCTION:
        app.add_middleware(HTTPSRedirectMiddleware)

    # 7. Trusted hosts
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=SecurityConfig.TRUSTED_HOSTS
    )

    # 8. GZIP compression (skip in testing to avoid AsyncClient compatibility issues)
    if not is_testing:
        app.add_middleware(GZipMiddleware, minimum_size=1000)

    # 9. CORS with environment-specific settings
    cors_origins = SecurityConfig.ALLOWED_ORIGINS
    if environment == Environment.DEVELOPMENT:
        cors_origins = cors_origins + [
            "http://localhost:3001",
            "http://127.0.0.1:3001"
        ]
    elif environment == Environment.PRODUCTION:
        # Use only secure origins in production
        cors_origins = [origin for origin in cors_origins if origin.startswith("https://")]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=SecurityConfig.ALLOWED_METHODS,
        allow_headers=SecurityConfig.ALLOWED_HEADERS,
        expose_headers=["X-RateLimit-Remaining", "X-RateLimit-Reset", "X-Request-ID"]
    )

    # 10. Session middleware (skip in testing mode for AsyncClient compatibility)
    if not is_testing:
        app.add_middleware(
            SessionMiddleware,
            secret_key=SecurityConfig.SESSION_SECRET_KEY,
            max_age=SecurityConfig.SESSION_MAX_AGE,
            same_site="strict" if environment == Environment.PRODUCTION else "lax",
            https_only=SecurityConfig.FORCE_HTTPS and environment == Environment.PRODUCTION
        )

    # 11. Enhanced IP filtering middleware
    @app.middleware("http")
    async def enhanced_ip_filter_middleware(request: Request, call_next):
        client_ip = _get_real_client_ip(request)

        # Check blocked IPs
        if client_ip in SecurityConfig.BLOCKED_IPS:
            audit_logger = get_audit_logger()
            await audit_logger.log_security_violation(
                "blocked_ip_access", client_ip, request.headers.get("User-Agent", ""),
                {"blocked_ip": client_ip, "endpoint": str(request.url)}
            )

            from fastapi import HTTPException, status
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied",
                headers={"X-Error-Code": "BLOCKED_IP"}
            )

        # Check allowed IPs (if configured)
        if (SecurityConfig.ALLOWED_IPS is not None and
            client_ip not in SecurityConfig.ALLOWED_IPS):
            audit_logger = get_audit_logger()
            await audit_logger.log_security_violation(
                "unauthorized_ip_access", client_ip, request.headers.get("User-Agent", ""),
                {"unauthorized_ip": client_ip, "endpoint": str(request.url)}
            )

            from fastapi import HTTPException, status
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied",
                headers={"X-Error-Code": "UNAUTHORIZED_IP"}
            )

        # Add security headers to request context
        request.state.client_ip = client_ip
        request.state.request_id = secrets.token_hex(8)

        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id

        return response


def add_security_middleware(app: FastAPI) -> None:
    """Legacy function - use add_comprehensive_security_middleware instead"""
    add_comprehensive_security_middleware(app)


def _get_real_client_ip(request: Request) -> str:
    """Get real client IP considering proxy headers"""
    # Check common proxy headers in order of preference
    headers_to_check = [
        "CF-Connecting-IP",      # Cloudflare
        "X-Real-IP",            # Nginx
        "X-Forwarded-For",      # Standard proxy header
        "X-Client-IP",          # Alternative
        "X-Cluster-Client-IP",  # Cluster environments
    ]

    for header in headers_to_check:
        ip = request.headers.get(header)
        if ip:
            # X-Forwarded-For can contain multiple IPs
            if "," in ip:
                ip = ip.split(",")[0].strip()

            # Basic IP validation
            try:
                import ipaddress
                ipaddress.ip_address(ip)
                return ip
            except ValueError:
                continue

    # Fall back to direct connection IP
    return request.client.host if request.client else "unknown"

# ---------------------------------------------------------------------------
# Re-exports (Wave 12 / #99): validators live in security_validators.py
# ---------------------------------------------------------------------------
from backend.security.security_validators import (  # noqa: E402
    APIKeyManager,
    FileUploadValidator,
    PasswordValidator,
    SecurityScanner,
)

__all__ = [
    'Environment',
    'InsecureSecretError',
    'SecurityConfig',
    'RedisHealthCheckError',
    'RedisHealthChecker',
    'PasswordValidator',
    'APIKeyManager',
    'FileUploadValidator',
    'SecurityScanner',
    'add_comprehensive_security_middleware',
    'add_security_middleware',
    'validate_redis_connectivity',
]
