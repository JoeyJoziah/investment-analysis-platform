"""
Enhanced Security Configuration and Hardening System
Provides comprehensive security configurations for different environments
"""

import os
import secrets
import mimetypes
import logging
import time
from datetime import timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
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
    SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY", secrets.token_urlsafe(32))
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
    # Default: 1 hour - gives users time to complete reset
    JWT_RESET_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_RESET_TOKEN_EXPIRE_MINUTES", "60"))

    # Secret key for HS256 algorithm (fallback/legacy)
    # In production, this MUST be set via environment variable
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))

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

    # 1. Audit logging middleware (first to capture everything)
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
        logger.info(f"Redis connectivity validated for rate limiting middleware")
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

    app.add_middleware(RateLimitingMiddleware, rules=rate_limit_rules, redis_url=redis_url)
    
    # 4. Input validation and sanitization
    app.add_middleware(ValidationMiddleware)
    
    # 5. Injection prevention (SQL, XSS, etc.)
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
    
    # 8. GZIP compression
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
    
    # 10. Session middleware
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


class PasswordValidator:
    """Password validation and strength checking"""
    
    @staticmethod
    def validate_password(password: str) -> Dict[str, bool]:
        """Validate password against security policy"""
        results = {
            "length": len(password) >= SecurityConfig.PASSWORD_MIN_LENGTH,
            "uppercase": any(c.isupper() for c in password) if SecurityConfig.PASSWORD_REQUIRE_UPPERCASE else True,
            "lowercase": any(c.islower() for c in password) if SecurityConfig.PASSWORD_REQUIRE_LOWERCASE else True,
            "digits": any(c.isdigit() for c in password) if SecurityConfig.PASSWORD_REQUIRE_DIGITS else True,
            "special": any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password) if SecurityConfig.PASSWORD_REQUIRE_SPECIAL else True
        }
        
        results["valid"] = all(results.values())
        return results
    
    @staticmethod
    def calculate_strength(password: str) -> int:
        """Calculate password strength score (0-100)"""
        score = 0
        
        # Length scoring
        if len(password) >= 12:
            score += 25
        elif len(password) >= 8:
            score += 15
        
        # Character variety
        if any(c.isupper() for c in password):
            score += 15
        if any(c.islower() for c in password):
            score += 15
        if any(c.isdigit() for c in password):
            score += 15
        if any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            score += 20
        
        # Bonus for length
        score += min(10, (len(password) - 12) * 2)
        
        return min(100, score)
    
    @staticmethod
    def generate_secure_password(length: int = 16) -> str:
        """Generate a cryptographically secure password"""
        import string

        # Ensure we have all required character types
        chars = ""
        password = ""

        if SecurityConfig.PASSWORD_REQUIRE_UPPERCASE:
            chars += string.ascii_uppercase
            password += secrets.choice(string.ascii_uppercase)

        if SecurityConfig.PASSWORD_REQUIRE_LOWERCASE:
            chars += string.ascii_lowercase
            password += secrets.choice(string.ascii_lowercase)

        if SecurityConfig.PASSWORD_REQUIRE_DIGITS:
            chars += string.digits
            password += secrets.choice(string.digits)

        if SecurityConfig.PASSWORD_REQUIRE_SPECIAL:
            special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
            chars += special_chars
            password += secrets.choice(special_chars)

        # Fill remaining length with cryptographically secure random choices
        for _ in range(length - len(password)):
            password += secrets.choice(chars)

        # Shuffle the password using Fisher-Yates with secure randomness
        password_list = list(password)
        for i in range(len(password_list) - 1, 0, -1):
            j = secrets.randbelow(i + 1)
            password_list[i], password_list[j] = password_list[j], password_list[i]

        return ''.join(password_list)


class APIKeyManager:
    """API key management and validation"""
    
    @staticmethod
    def generate_api_key() -> str:
        """Generate a new API key"""
        key = secrets.token_urlsafe(SecurityConfig.API_KEY_LENGTH)
        return f"{SecurityConfig.API_KEY_PREFIX}{key}"
    
    @staticmethod
    def validate_api_key_format(api_key: str) -> bool:
        """Validate API key format"""
        if not api_key.startswith(SecurityConfig.API_KEY_PREFIX):
            return False
        
        # Remove prefix and check length
        key_part = api_key[len(SecurityConfig.API_KEY_PREFIX):]
        return len(key_part) == SecurityConfig.API_KEY_LENGTH
    
    @staticmethod
    def hash_api_key(api_key: str) -> str:
        """Hash API key for storage"""
        import hashlib
        return hashlib.sha256(api_key.encode()).hexdigest()


class FileUploadValidator:
    """
    Comprehensive file upload validation with MIME type detection.
    Validates that actual file content matches claimed extension to prevent
    attackers from bypassing security by renaming malicious files.
    """

    # Logger for security monitoring
    _logger = logging.getLogger("security.file_upload")

    @classmethod
    def detect_mime_type_from_content(cls, file_content: bytes) -> Optional[str]:
        """
        Detect MIME type from file content using magic bytes (file signatures).
        Returns None if the content doesn't match any known signature.
        """
        for mime_type, signatures in SecurityConfig.FILE_SIGNATURES.items():
            for signature in signatures:
                if file_content.startswith(signature):
                    return mime_type
        return None

    @classmethod
    def detect_mime_type_from_extension(cls, filename: str) -> Optional[str]:
        """
        Detect MIME type from file extension using mimetypes library.
        Returns None if extension is unknown.
        """
        mime_type, _ = mimetypes.guess_type(filename)
        return mime_type

    @classmethod
    def is_text_based_file(cls, extension: str) -> bool:
        """Check if file type is text-based (CSV, JSON, TXT) which may not have magic bytes."""
        return extension.lower() in [".csv", ".json", ".txt"]

    @classmethod
    def validate_text_content(cls, file_content: bytes, extension: str) -> Tuple[bool, Optional[str]]:
        """
        Validate text-based files by checking content structure.
        Returns (is_valid, error_message).
        """
        try:
            # Attempt to decode as UTF-8
            text_content = file_content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                # Try latin-1 as fallback
                text_content = file_content.decode("latin-1")
            except Exception:
                return False, "File content is not valid text"

        ext_lower = extension.lower()

        if ext_lower == ".json":
            import json
            try:
                json.loads(text_content)
                return True, None
            except json.JSONDecodeError as e:
                return False, f"Invalid JSON structure: {str(e)[:100]}"

        if ext_lower == ".csv":
            # Basic CSV validation - check for reasonable structure
            lines = text_content.strip().split("\n")
            if len(lines) == 0:
                return False, "Empty CSV file"
            # Check that it has some comma or tab delimiters (common CSV patterns)
            first_line = lines[0]
            if "," not in first_line and "\t" not in first_line and ";" not in first_line:
                # Single column CSV is technically valid
                pass
            return True, None

        if ext_lower == ".txt":
            # Plain text - just ensure it's decodable (already done above)
            return True, None

        return True, None

    @classmethod
    def validate_mime_type(
        cls,
        file_content: bytes,
        filename: str,
        claimed_content_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate that file content matches its claimed type.

        Args:
            file_content: Raw bytes of the uploaded file
            filename: Original filename with extension
            claimed_content_type: Content-Type header from upload (optional)

        Returns:
            Dict with validation results:
                - valid: bool - whether the file passed validation
                - detected_mime: str - MIME type detected from content
                - expected_mime: str - MIME type expected from extension
                - issues: List[str] - list of validation issues
                - extension: str - file extension
        """
        result = {
            "valid": True,
            "detected_mime": None,
            "expected_mime": None,
            "claimed_mime": claimed_content_type,
            "issues": [],
            "extension": None,
            "filename": filename
        }

        # Extract and validate extension
        _, ext = os.path.splitext(filename.lower())
        result["extension"] = ext

        if not ext:
            result["valid"] = False
            result["issues"].append("File has no extension")
            cls._log_rejected_upload(filename, result["issues"], "no_extension")
            return result

        # Check if extension is in allowlist
        if ext not in SecurityConfig.ALLOWED_FILE_TYPES:
            result["valid"] = False
            result["issues"].append(f"File extension '{ext}' is not in allowed list")
            cls._log_rejected_upload(filename, result["issues"], "disallowed_extension")
            return result

        # Get expected MIME types for this extension
        expected_mimes = SecurityConfig.ALLOWED_MIME_TYPES.get(ext, [])
        result["expected_mime"] = expected_mimes[0] if expected_mimes else None

        # Detect MIME type from content
        detected_mime = cls.detect_mime_type_from_content(file_content)
        result["detected_mime"] = detected_mime

        # Handle text-based files specially (they don't have magic bytes)
        if cls.is_text_based_file(ext):
            is_valid_text, text_error = cls.validate_text_content(file_content, ext)
            if not is_valid_text:
                result["valid"] = False
                result["issues"].append(text_error)
                cls._log_rejected_upload(filename, result["issues"], "invalid_text_content")
                return result

            # For text files, also check they don't contain binary/executable signatures
            if detected_mime and detected_mime not in ["text/plain", "text/csv", "application/json"]:
                result["valid"] = False
                result["issues"].append(
                    f"File claims to be {ext} but contains binary content signature for '{detected_mime}'"
                )
                cls._log_rejected_upload(filename, result["issues"], "mime_mismatch_binary_in_text")
                return result

            # Text file validation passed
            result["detected_mime"] = expected_mimes[0] if expected_mimes else "text/plain"
            return result

        # For binary files, detected MIME must be present
        if detected_mime is None:
            # Check if it might be a ZIP-based format (xlsx, docx, etc.)
            if file_content.startswith(b"PK\x03\x04"):
                if ext == ".xlsx":
                    result["detected_mime"] = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                else:
                    result["valid"] = False
                    result["issues"].append(
                        f"File appears to be a ZIP archive but extension is '{ext}'"
                    )
                    cls._log_rejected_upload(filename, result["issues"], "zip_archive_wrong_extension")
                    return result
            else:
                result["valid"] = False
                result["issues"].append(
                    f"Could not detect file type from content - file may be corrupted or disguised"
                )
                cls._log_rejected_upload(filename, result["issues"], "undetectable_mime")
                return result

        # Verify detected MIME type matches expected MIME types for this extension
        if detected_mime not in expected_mimes:
            result["valid"] = False
            result["issues"].append(
                f"MIME type mismatch: file extension is '{ext}' (expects {expected_mimes}) "
                f"but content is '{detected_mime}'"
            )
            cls._log_rejected_upload(filename, result["issues"], "mime_mismatch")
            return result

        # If claimed content type was provided, verify it matches
        if claimed_content_type:
            # Normalize content type (remove charset and other parameters)
            claimed_base = claimed_content_type.split(";")[0].strip().lower()
            if claimed_base not in expected_mimes and claimed_base != detected_mime:
                result["issues"].append(
                    f"Warning: Claimed Content-Type '{claimed_base}' differs from detected '{detected_mime}'"
                )
                # This is a warning, not a rejection - detected content is authoritative

        return result

    @classmethod
    def _log_rejected_upload(
        cls,
        filename: str,
        issues: List[str],
        rejection_type: str
    ) -> None:
        """Log rejected file uploads for security monitoring."""
        cls._logger.warning(
            "File upload rejected: %s - %s (%s)",
            filename,
            rejection_type,
            "; ".join(issues),
            extra={
                "event_type": "file_upload_rejected",
                "upload_filename": filename,  # Renamed to avoid LogRecord conflict
                "rejection_type": rejection_type,
                "issues": issues,
                "security_event": True
            }
        )


class SecurityScanner:
    """Security scanning and vulnerability detection"""

    @staticmethod
    def scan_file_upload(file_content: bytes, filename: str, content_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Comprehensive file upload security scan including MIME type validation.

        Args:
            file_content: Raw bytes of the uploaded file
            filename: Original filename
            content_type: Content-Type header from the upload request (optional)

        Returns:
            Dict with scan results including safety status, detected types, and issues
        """
        results = {
            "safe": True,
            "issues": [],
            "file_type": "unknown",
            "detected_mime": None,
            "extension": None,
            "size": len(file_content),
            "mime_validation": None
        }

        # Check file size first
        if len(file_content) > SecurityConfig.MAX_FILE_SIZE:
            results["safe"] = False
            results["issues"].append(
                f"File size ({len(file_content)} bytes) exceeds limit ({SecurityConfig.MAX_FILE_SIZE} bytes)"
            )

        # Extract extension
        _, ext = os.path.splitext(filename.lower())
        results["extension"] = ext
        results["file_type"] = ext.lstrip(".") if ext else "unknown"

        # Perform MIME type validation
        mime_validation = FileUploadValidator.validate_mime_type(
            file_content, filename, content_type
        )
        results["mime_validation"] = mime_validation
        results["detected_mime"] = mime_validation.get("detected_mime")

        if not mime_validation["valid"]:
            results["safe"] = False
            results["issues"].extend(mime_validation["issues"])

        # Check for suspicious patterns (malware signatures)
        suspicious_patterns = [
            (b"<script", "JavaScript script tag"),
            (b"javascript:", "JavaScript protocol"),
            (b"vbscript:", "VBScript protocol"),
            (b"onload=", "Event handler injection"),
            (b"onerror=", "Event handler injection"),
            (b"eval(", "Eval function call"),
            (b"exec(", "Exec function call"),
            (b"MZ", "Windows executable signature"),  # PE/EXE files
            (b"\x7fELF", "Linux executable signature"),  # ELF files
            (b"#!/", "Shell script shebang"),
            (b"<?php", "PHP code"),
            (b"<%", "ASP/JSP code"),
        ]

        # Use lowercase for pattern matching in text content
        content_lower = file_content.lower()

        for pattern, description in suspicious_patterns:
            pattern_lower = pattern.lower() if isinstance(pattern, bytes) else pattern
            # Check both original and lowercase
            if pattern in file_content or pattern_lower in content_lower:
                # Special handling: Don't flag shebang in text files that legitimately could have it
                if pattern == b"#!/" and ext in [".sh", ".py", ".rb"]:
                    continue
                results["safe"] = False
                results["issues"].append(f"Suspicious pattern detected: {description}")

        # Additional check: Double extension attack (e.g., file.pdf.exe)
        base_name = os.path.basename(filename)
        if base_name.count(".") > 1:
            # Extract all extensions
            parts = base_name.split(".")
            if len(parts) > 2:
                suspicious_exts = [".exe", ".dll", ".bat", ".cmd", ".ps1", ".vbs", ".js", ".hta"]
                for part in parts[1:-1]:  # Check intermediate "extensions"
                    if f".{part.lower()}" in suspicious_exts:
                        results["safe"] = False
                        results["issues"].append(
                            f"Potential double extension attack detected: filename contains '{part}'"
                        )

        # Log if file was rejected
        if not results["safe"]:
            scan_logger = logging.getLogger("security.file_upload")
            scan_logger.warning(
                "File upload scan failed: %s (%s)",
                filename,
                "; ".join(results["issues"]),
                extra={
                    "event_type": "file_scan_failed",
                    "upload_filename": filename,  # Renamed to avoid LogRecord conflict
                    "size": len(file_content),
                    "detected_mime": results["detected_mime"],
                    "issues": results["issues"],
                    "security_event": True
                }
            )

        return results
    
    @staticmethod
    def check_sql_injection(query: str) -> bool:
        """Check for potential SQL injection patterns"""
        suspicious_patterns = [
            "union select",
            "drop table",
            "delete from",
            "insert into",
            "update set",
            "exec(",
            "execute(",
            "--",
            ";--",
            "/*",
            "*/"
        ]
        
        query_lower = query.lower()
        return any(pattern in query_lower for pattern in suspicious_patterns)
    
    @staticmethod
    def validate_input(data: str, max_length: int = 1000) -> Dict[str, Any]:
        """Validate user input for security issues"""
        results = {
            "safe": True,
            "issues": []
        }
        
        # Check length
        if len(data) > max_length:
            results["safe"] = False
            results["issues"].append("Input too long")
        
        # Check for XSS patterns
        xss_patterns = [
            "<script",
            "javascript:",
            "vbscript:",
            "onload=",
            "onerror=",
            "onclick=",
            "onmouseover="
        ]
        
        data_lower = data.lower()
        for pattern in xss_patterns:
            if pattern in data_lower:
                results["safe"] = False
                results["issues"].append(f"XSS pattern detected: {pattern}")
        
        # Check for SQL injection
        if SecurityScanner.check_sql_injection(data):
            results["safe"] = False
            results["issues"].append("Potential SQL injection detected")
        
        return results


# Global security instances
password_validator = PasswordValidator()
api_key_manager = APIKeyManager()
security_scanner = SecurityScanner()
file_upload_validator = FileUploadValidator()


def validate_file_upload(
    file_content: bytes,
    filename: str,
    content_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function for validating file uploads.
    Combines MIME type validation with security scanning.

    Args:
        file_content: Raw bytes of the uploaded file
        filename: Original filename with extension
        content_type: Content-Type header from upload (optional)

    Returns:
        Dict with comprehensive validation results

    Example:
        >>> result = validate_file_upload(file_bytes, "report.pdf", "application/pdf")
        >>> if not result["safe"]:
        ...     raise HTTPException(400, detail=result["issues"])
    """
    return security_scanner.scan_file_upload(file_content, filename, content_type)