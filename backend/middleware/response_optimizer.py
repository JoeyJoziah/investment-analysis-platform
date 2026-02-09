"""
Response Optimizer Middleware

Optimizes API response performance through:
- Response timing tracking (X-Response-Time header)
- ETag generation for conditional requests (304 Not Modified)
- Response headers optimization

Created: 2026-02-08
Part of: Issue #12 - API Response Time Optimization
"""

import hashlib
import logging
import time
from typing import Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from starlette.datastructures import Headers, MutableHeaders

logger = logging.getLogger(__name__)


class ResponseTimingMiddleware(BaseHTTPMiddleware):
    """
    Tracks response time and adds X-Response-Time header.
    Should be outermost middleware for accurate timing.
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()

        try:
            response = await call_next(request)

            # Calculate response time in milliseconds
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Add response time header
            response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

            # Log slow requests (>1000ms)
            if duration_ms > 1000:
                logger.warning(
                    f"Slow request: {request.method} {request.url.path} "
                    f"took {duration_ms:.2f}ms"
                )

            return response

        except Exception as e:
            # Still track time even on errors
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"Request failed after {duration_ms:.2f}ms: "
                f"{request.method} {request.url.path} - {e}"
            )
            raise


class ETagMiddleware:
    """
    Generates ETags for cacheable responses and handles conditional requests.
    Returns 304 Not Modified when appropriate.

    Note: Uses raw ASGI interface instead of BaseHTTPMiddleware for proper
    body access in streaming responses.
    """

    def __init__(
        self,
        app: ASGIApp,
        excluded_paths: Optional[list] = None,
        weak_etag: bool = False
    ):
        """
        Args:
            app: ASGI application
            excluded_paths: List of path prefixes to exclude from ETag generation
            weak_etag: Use weak ETags (W/) for dynamic content
        """
        self.app = app
        self.excluded_paths = excluded_paths or [
            "/api/v1/auth/",
            "/api/v1/admin/",
            "/api/v1/ws/",
            "/api/health"
        ]
        self.weak_etag = weak_etag

    def _should_generate_etag(self, path: str, method: str, status_code: int, headers: dict) -> bool:
        """Determine if ETag should be generated for this response"""
        # Only for successful GET/HEAD requests
        if method not in ["GET", "HEAD"]:
            return False

        if status_code != 200:
            return False

        # Skip excluded paths
        if any(path.startswith(excluded) for excluded in self.excluded_paths):
            return False

        # Skip if ETag already present
        if b"etag" in headers:
            return False

        return True

    def _generate_etag(self, content: bytes) -> str:
        """
        Generate ETag from response content

        Args:
            content: Response body bytes

        Returns:
            ETag value (quoted, optionally weak)
        """
        # Use MD5 for fast hashing (not for security)
        content_hash = hashlib.md5(content).hexdigest()[:16]

        if self.weak_etag:
            return f'W/"{content_hash}"'
        else:
            return f'"{content_hash}"'

    def _check_if_none_match(self, if_none_match: str, etag: str) -> bool:
        """
        Check if If-None-Match header matches the ETag

        Returns:
            True if match (should return 304), False otherwise
        """
        if not if_none_match:
            return False

        # Parse comma-separated list of ETags
        client_etags = [tag.strip() for tag in if_none_match.split(",")]

        # Check for match (strip quotes for comparison)
        server_etag_unquoted = etag.strip('"').replace('W/', '')

        for client_etag in client_etags:
            client_etag_unquoted = client_etag.strip('"').replace('W/', '')

            if client_etag_unquoted == server_etag_unquoted or client_etag == "*":
                return True

        return False

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Extract request info
        method = scope["method"]
        path = scope["path"]
        headers = Headers(scope=scope)
        if_none_match = headers.get("if-none-match")

        # Collect response
        response_started = False
        start_sent = False  # Track if we already sent the start message
        status_code = 200
        response_headers = []
        body_parts = []
        should_process = False  # Track if we should generate ETag for this response

        async def send_wrapper(message):
            nonlocal response_started, start_sent, status_code, response_headers, should_process

            if message["type"] == "http.response.start":
                response_started = True
                status_code = message["status"]
                response_headers = list(message.get("headers", []))

                # Check if we should process this response
                headers_dict = dict(response_headers)
                should_process = self._should_generate_etag(path, method, status_code, headers_dict)

                if not should_process:
                    # Don't process - send immediately
                    start_sent = True
                    await send(message)
                    return

                # Hold the start message - we need the body first
                return

            elif message["type"] == "http.response.body":
                # If start already sent (not processing), just pass through
                if start_sent:
                    await send(message)
                    return

                body = message.get("body", b"")
                if body:
                    body_parts.append(body)

                # If this is the last chunk, generate ETag
                if not message.get("more_body", False):
                    full_body = b"".join(body_parts)

                    if full_body and should_process:
                        # Generate ETag
                        etag = self._generate_etag(full_body)

                        # Check for conditional request
                        if if_none_match and self._check_if_none_match(if_none_match, etag):
                            # Return 304 Not Modified
                            await send({
                                "type": "http.response.start",
                                "status": 304,
                                "headers": [
                                    (b"etag", etag.encode()),
                                ],
                            })
                            await send({
                                "type": "http.response.body",
                                "body": b"",
                            })
                            start_sent = True
                            return

                        # Add ETag header to response
                        response_headers.append((b"etag", etag.encode()))

                    # Send the start message (with ETag if we added it)
                    await send({
                        "type": "http.response.start",
                        "status": status_code,
                        "headers": response_headers,
                    })
                    start_sent = True

                    # Send the body
                    await send(message)
                else:
                    # More body chunks coming - hold them
                    return
            else:
                await send(message)

        await self.app(scope, receive, send_wrapper)
