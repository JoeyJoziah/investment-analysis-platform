"""
Security Middleware Integration

This module provides a unified function to register all Phase 3 security
middleware with the FastAPI application.

Created: 2026-01-27
Part of: Phase 3 Security Remediation
"""

import os
import logging
from typing import Optional, List, Dict

from fastapi import FastAPI

from backend.security.csrf_protection import add_csrf_protection
from backend.middleware.security_headers import add_security_headers
from backend.middleware.request_size_limiter import add_request_size_limits

logger = logging.getLogger(__name__)


def register_security_middleware(
    app: FastAPI,
    csrf_enabled: bool = True,
    security_headers_enabled: bool = True,
    request_size_limits_enabled: bool = True,
    csrf_secret_key: Optional[str] = None,
    csrf_exempt_paths: Optional[List[str]] = None,
    csp_script_src: Optional[List[str]] = None,
    csp_connect_src: Optional[List[str]] = None,
    json_limit_mb: float = 1.0,
    file_upload_limit_mb: float = 10.0,
    path_size_limits: Optional[Dict[str, int]] = None,
) -> None:
    """
    Register all Phase 3 security middleware with FastAPI application

    This function adds CSRF protection, security headers, and request size
    limits in the correct order for optimal security.

    Args:
        app: FastAPI application instance
        csrf_enabled: Enable CSRF protection
        security_headers_enabled: Enable security headers
        request_size_limits_enabled: Enable request size limits
        csrf_secret_key: Secret key for CSRF tokens (from env if None)
        csrf_exempt_paths: Additional paths to exempt from CSRF
        csp_script_src: Additional script-src origins for CSP
        csp_connect_src: Additional connect-src origins for CSP
        json_limit_mb: JSON payload size limit in MB
        file_upload_limit_mb: File upload size limit in MB
        path_size_limits: Custom size limits for specific paths

    Example:
        ```python
        from fastapi import FastAPI
        from backend.api.security_integration import register_security_middleware

        app = FastAPI()

        register_security_middleware(
            app,
            csrf_exempt_paths=["/api/webhooks/stripe"],
            csp_script_src=["'self'", "https://cdn.example.com"],
            json_limit_mb=1.0,
            file_upload_limit_mb=10.0
        )
        ```
    """
    logger.info("Registering Phase 3 security middleware...")

    # 1. Security Headers (applied to all responses)
    if security_headers_enabled:
        try:
            add_security_headers(
                app,
                csp_script_src=csp_script_src or ["'self'"],
                csp_connect_src=csp_connect_src or ["'self'"],
                exclude_paths={"/metrics", "/health"}
            )
            logger.info("✓ Security headers middleware registered")
        except Exception as e:
            logger.error(f"Failed to register security headers: {e}")
            raise

    # 2. Request Size Limits (checks early, before body parsing)
    if request_size_limits_enabled:
        try:
            add_request_size_limits(
                app,
                json_limit_mb=json_limit_mb,
                file_upload_limit_mb=file_upload_limit_mb,
                path_limits=path_size_limits or {}
            )
            logger.info(
                f"✓ Request size limits registered: "
                f"JSON={json_limit_mb}MB, Files={file_upload_limit_mb}MB"
            )
        except Exception as e:
            logger.error(f"Failed to register request size limits: {e}")
            raise

    # 3. CSRF Protection (validates state-changing requests)
    if csrf_enabled:
        try:
            # Get CSRF secret key from environment or parameter
            secret_key = csrf_secret_key or os.getenv("CSRF_SECRET_KEY")

            if not secret_key or len(secret_key) < 32:
                logger.warning(
                    "CSRF_SECRET_KEY not set or too short. "
                    "Using auto-generated key (not suitable for production with multiple workers)"
                )

            add_csrf_protection(
                app,
                secret_key=secret_key,
                exempt_paths=csrf_exempt_paths or []
            )
            logger.info("✓ CSRF protection middleware registered")
        except Exception as e:
            logger.error(f"Failed to register CSRF protection: {e}")
            raise

    logger.info("✓ Phase 3 security middleware registration complete")


def get_security_middleware_config() -> Dict:
    """
    Get current security middleware configuration from environment

    Returns:
        Dictionary with security configuration

    Example:
        ```python
        config = get_security_middleware_config()
        print(f"CSRF enabled: {config['csrf_enabled']}")
        ```
    """
    return {
        "csrf_enabled": os.getenv("CSRF_ENABLED", "true").lower() == "true",
        "csrf_secret_key": os.getenv("CSRF_SECRET_KEY"),
        "security_headers_enabled": os.getenv("SECURITY_HEADERS_ENABLED", "true").lower() == "true",
        "request_size_limits_enabled": os.getenv("REQUEST_SIZE_LIMITS_ENABLED", "true").lower() == "true",
        "json_limit_mb": float(os.getenv("JSON_SIZE_LIMIT_MB", "1.0")),
        "file_upload_limit_mb": float(os.getenv("FILE_UPLOAD_LIMIT_MB", "10.0")),
        "hsts_max_age": int(os.getenv("HSTS_MAX_AGE", "31536000")),
        "hsts_include_subdomains": os.getenv("HSTS_INCLUDE_SUBDOMAINS", "true").lower() == "true",
    }


def validate_security_configuration() -> bool:
    """
    Validate security configuration before application startup

    Returns:
        True if configuration is valid

    Raises:
        ValueError: If critical security configuration is invalid

    Example:
        ```python
        if not validate_security_configuration():
            raise RuntimeError("Invalid security configuration")
        ```
    """
    config = get_security_middleware_config()
    errors = []

    # Check CSRF secret key in production
    if config["csrf_enabled"]:
        secret_key = config["csrf_secret_key"]
        if not secret_key:
            errors.append("CSRF_SECRET_KEY environment variable not set")
        elif len(secret_key) < 32:
            errors.append("CSRF_SECRET_KEY must be at least 32 characters")

    # Check size limits are reasonable
    if config["json_limit_mb"] < 0.1:
        errors.append("JSON_SIZE_LIMIT_MB too small (minimum 0.1 MB)")
    if config["json_limit_mb"] > 100:
        errors.append("JSON_SIZE_LIMIT_MB too large (maximum 100 MB)")

    if config["file_upload_limit_mb"] < 1:
        errors.append("FILE_UPLOAD_LIMIT_MB too small (minimum 1 MB)")
    if config["file_upload_limit_mb"] > 1000:
        errors.append("FILE_UPLOAD_LIMIT_MB too large (maximum 1000 MB)")

    if errors:
        error_message = "Security configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        logger.error(error_message)
        raise ValueError(error_message)

    logger.info("✓ Security configuration validated")
    return True
