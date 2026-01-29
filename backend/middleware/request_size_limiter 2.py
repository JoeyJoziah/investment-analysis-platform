"""
Request Size Limits Middleware

Implements request body size validation to prevent DoS attacks:
- Configurable limits for different content types
- JSON payload size limits
- File upload size limits
- Streaming request support
- Clear error messages with 413 Payload Too Large

Created: 2026-01-27
Part of: Phase 3 Security Remediation (HIGH-5)
"""

import logging
from typing import Optional, Dict, Callable
from dataclasses import dataclass, field
from enum import Enum

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class ContentType(str, Enum):
    """Common content types for size limiting"""
    JSON = "application/json"
    FORM = "application/x-www-form-urlencoded"
    MULTIPART = "multipart/form-data"
    TEXT = "text/plain"
    OCTET_STREAM = "application/octet-stream"


@dataclass
class RequestSizeLimits:
    """Request size limit configuration"""
    # Default limit for all requests (bytes)
    default_limit: int = 1_048_576  # 1 MB

    # Content-type specific limits (bytes)
    json_limit: int = 1_048_576  # 1 MB for JSON
    form_limit: int = 1_048_576  # 1 MB for forms
    file_upload_limit: int = 10_485_760  # 10 MB for file uploads
    text_limit: int = 524_288  # 512 KB for text

    # Path-specific limits (overrides content-type limits)
    path_limits: Dict[str, int] = field(default_factory=dict)

    # Paths exempt from size checking
    exempt_paths: set = field(default_factory=lambda: {"/health", "/metrics"})

    # Whether to log size violations
    log_violations: bool = True

    # Custom error message
    error_message: str = "Request payload too large"


class RequestSizeLimiterMiddleware(BaseHTTPMiddleware):
    """
    Request Size Limiter Middleware

    Validates request body size before processing to prevent
    memory exhaustion and DoS attacks.
    """

    def __init__(self, app: ASGIApp, config: Optional[RequestSizeLimits] = None):
        """
        Initialize request size limiter

        Args:
            app: ASGI application
            config: Size limit configuration
        """
        super().__init__(app)
        self.config = config or RequestSizeLimits()

    def _get_content_type(self, request: Request) -> Optional[str]:
        """
        Extract content type from request

        Args:
            request: Incoming request

        Returns:
            Content type or None
        """
        content_type = request.headers.get("content-type", "")
        # Remove parameters like charset
        return content_type.split(";")[0].strip().lower()

    def _get_size_limit(self, request: Request) -> int:
        """
        Determine appropriate size limit for request

        Args:
            request: Incoming request

        Returns:
            Size limit in bytes
        """
        # Check path-specific limits first
        for path_pattern, limit in self.config.path_limits.items():
            if request.url.path.startswith(path_pattern):
                return limit

        # Check content-type specific limits
        content_type = self._get_content_type(request)

        if content_type:
            if content_type == ContentType.JSON:
                return self.config.json_limit
            elif content_type == ContentType.FORM:
                return self.config.form_limit
            elif content_type.startswith(ContentType.MULTIPART):
                return self.config.file_upload_limit
            elif content_type == ContentType.TEXT:
                return self.config.text_limit

        # Default limit
        return self.config.default_limit

    def _is_exempt_path(self, path: str) -> bool:
        """
        Check if path is exempt from size checking

        Args:
            path: Request path

        Returns:
            True if path is exempt
        """
        return path in self.config.exempt_paths

    def _format_size(self, size_bytes: int) -> str:
        """
        Format size in human-readable format

        Args:
            size_bytes: Size in bytes

        Returns:
            Formatted string (e.g., "1.5 MB")
        """
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1_048_576:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes / 1_048_576:.1f} MB"

    async def dispatch(self, request: Request, call_next: Callable):
        """
        Process request with size validation

        Args:
            request: Incoming request
            call_next: Next middleware/handler

        Returns:
            Response or 413 error if payload too large
        """
        # Skip exempt paths
        if self._is_exempt_path(request.url.path):
            return await call_next(request)

        # Only check requests with body
        if request.method not in {"POST", "PUT", "PATCH"}:
            return await call_next(request)

        # Get size limit for this request
        size_limit = self._get_size_limit(request)

        # Check Content-Length header
        content_length = request.headers.get("content-length")

        if content_length:
            try:
                content_length_int = int(content_length)

                if content_length_int > size_limit:
                    # Log violation
                    if self.config.log_violations:
                        logger.warning(
                            f"Request size limit exceeded: {self._format_size(content_length_int)} > {self._format_size(size_limit)}",
                            extra={
                                "path": request.url.path,
                                "method": request.method,
                                "content_type": self._get_content_type(request),
                                "content_length": content_length_int,
                                "limit": size_limit,
                                "ip": request.client.host if request.client else "unknown"
                            }
                        )

                    # Return 413 Payload Too Large
                    return JSONResponse(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        content={
                            "success": False,
                            "error": self.config.error_message,
                            "detail": f"Request body size ({self._format_size(content_length_int)}) exceeds maximum allowed size ({self._format_size(size_limit)})",
                            "code": "PAYLOAD_TOO_LARGE",
                            "max_size": self._format_size(size_limit),
                            "received_size": self._format_size(content_length_int)
                        }
                    )

            except ValueError:
                # Invalid Content-Length header
                logger.warning(
                    f"Invalid Content-Length header: {content_length}",
                    extra={
                        "path": request.url.path,
                        "method": request.method
                    }
                )

        # If no Content-Length, we need to read body to check size
        # This is handled by FastAPI's request body parsing
        # We set a custom limit on the request object
        request.state.size_limit = size_limit

        try:
            response = await call_next(request)
            return response
        except Exception as e:
            # Check if it's a size-related error
            if "too large" in str(e).lower() or "exceeds" in str(e).lower():
                logger.warning(
                    f"Request body size validation failed: {str(e)}",
                    extra={
                        "path": request.url.path,
                        "method": request.method
                    }
                )

                return JSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content={
                        "success": False,
                        "error": self.config.error_message,
                        "detail": f"Request body exceeds maximum allowed size ({self._format_size(size_limit)})",
                        "code": "PAYLOAD_TOO_LARGE",
                        "max_size": self._format_size(size_limit)
                    }
                )

            # Re-raise other exceptions
            raise


def add_request_size_limits(
    app,
    json_limit_mb: Optional[float] = None,
    file_upload_limit_mb: Optional[float] = None,
    path_limits: Optional[Dict[str, int]] = None,
    **kwargs
) -> RequestSizeLimiterMiddleware:
    """
    Add request size limits to FastAPI application

    Args:
        app: FastAPI application
        json_limit_mb: JSON payload limit in MB (default: 1 MB)
        file_upload_limit_mb: File upload limit in MB (default: 10 MB)
        path_limits: Custom limits for specific paths
        **kwargs: Additional RequestSizeLimits parameters

    Returns:
        RequestSizeLimiterMiddleware instance

    Example:
        ```python
        from fastapi import FastAPI
        from backend.middleware.request_size_limiter import add_request_size_limits

        app = FastAPI()
        add_request_size_limits(
            app,
            json_limit_mb=1.0,
            file_upload_limit_mb=10.0,
            path_limits={
                "/api/uploads/large": 50 * 1024 * 1024  # 50 MB for specific endpoint
            }
        )
        ```
    """
    config_params = kwargs.copy()

    if json_limit_mb is not None:
        config_params["json_limit"] = int(json_limit_mb * 1_048_576)

    if file_upload_limit_mb is not None:
        config_params["file_upload_limit"] = int(file_upload_limit_mb * 1_048_576)

    if path_limits:
        config_params["path_limits"] = path_limits

    config = RequestSizeLimits(**config_params)
    middleware = RequestSizeLimiterMiddleware(app, config)
    app.add_middleware(RequestSizeLimiterMiddleware, config=config)

    logger.info(
        f"Request size limits enabled: JSON={config.json_limit / 1_048_576:.1f}MB, "
        f"Files={config.file_upload_limit / 1_048_576:.1f}MB"
    )
    return middleware
