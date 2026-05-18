"""
Middleware Stack Manager

Provides priority-based middleware registration to ensure correct ordering
and prevent conflicts between security, monitoring, and functional middleware.

Created: 2026-02-08
Part of: Issue #7 - Middleware Optimization
"""

import logging
from typing import List, Callable, Optional, Dict, Any
from enum import IntEnum
from dataclasses import dataclass
from fastapi import FastAPI

logger = logging.getLogger(__name__)


class MiddlewarePriority(IntEnum):
    """
    Middleware execution priority levels.

    CRITICAL: Remember FastAPI middleware is like an onion - first added = outermost = last executed

    For request flow:
    1. Request enters at OUTERMOST (highest number)
    2. Passes through layers toward INNERMOST (lowest number)
    3. Reaches endpoint
    4. Response passes back through layers from INNERMOST to OUTERMOST

    Priority ordering (execution order for requests):
    - ERROR_HANDLER (10000) - Outermost, catches all exceptions
    - CORS (9000) - Early, sets CORS headers before other processing
    - SECURITY_HEADERS (8000) - After CORS, adds security headers
    - CSRF (7000) - After security headers, validates CSRF tokens
    - RATE_LIMITING (6000) - After CSRF, rate limit requests
    - REQUEST_SIZE (5000) - After rate limiting, check request sizes
    - AUTHENTICATION (4000) - After size check, validate auth
    - MONITORING (3000) - After auth, track metrics
    - CACHING (2000) - After monitoring, handle cache
    - COMPRESSION (1000) - Innermost, compress responses
    """

    # Outermost - Error handling catches everything
    ERROR_HANDLER = 10000

    # Very early - CORS must be early to set headers before other middleware
    CORS = 9000

    # Security layers (outer to inner)
    SECURITY_HEADERS = 8000
    CSRF = 7000
    RATE_LIMITING = 6000
    REQUEST_SIZE = 5000
    AUTHENTICATION = 4000

    # Monitoring and auditing
    MONITORING = 3000
    AUDIT = 2900

    # Performance optimizations
    CACHING = 2000

    # Innermost - Compression last (compresses final response)
    COMPRESSION = 1000

    # Custom middleware can use these
    HIGHEST = 15000
    HIGH = 12000
    NORMAL = 5000
    LOW = 3000
    LOWEST = 500


@dataclass
class MiddlewareRegistration:
    """Represents a middleware to be registered"""
    name: str
    middleware_class: type
    priority: MiddlewarePriority
    config: Dict[str, Any]
    enabled: bool = True
    skip_in_testing: bool = False


class MiddlewareStack:
    """
    Manages middleware registration with priority-based ordering.

    Usage:
        stack = MiddlewareStack(app)
        stack.register(
            "csrf",
            CSRFMiddleware,
            MiddlewarePriority.CSRF,
            {"config": csrf_config}
        )
        stack.apply()
    """

    def __init__(self, app: FastAPI):
        """
        Initialize middleware stack

        Args:
            app: FastAPI application instance
        """
        self.app = app
        self.middlewares: List[MiddlewareRegistration] = []
        self._applied = False

    def register(
        self,
        name: str,
        middleware_class: type,
        priority: MiddlewarePriority,
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
        skip_in_testing: bool = False
    ) -> 'MiddlewareStack':
        """
        Register a middleware for later application

        Args:
            name: Middleware name (for logging)
            middleware_class: Middleware class to instantiate
            priority: Execution priority
            config: Configuration dict passed to middleware
            enabled: Whether middleware is enabled
            skip_in_testing: Skip this middleware in testing mode

        Returns:
            Self for chaining
        """
        if self._applied:
            raise RuntimeError(
                "Cannot register middleware after stack has been applied. "
                "Call register() before apply()."
            )

        registration = MiddlewareRegistration(
            name=name,
            middleware_class=middleware_class,
            priority=priority,
            config=config or {},
            enabled=enabled,
            skip_in_testing=skip_in_testing
        )

        self.middlewares.append(registration)
        logger.debug(
            f"Registered middleware: {name} (priority={getattr(priority, 'name', int(priority))}:{getattr(priority, 'value', int(priority))})"
        )

        return self

    def apply(self, is_testing: bool = False) -> None:
        """
        Apply all registered middleware in priority order

        Args:
            is_testing: Whether running in testing mode
        """
        if self._applied:
            logger.warning("Middleware stack already applied, skipping")
            return

        # Sort by priority (highest to lowest)
        # This ensures outermost middleware (highest priority) is added first
        sorted_middlewares = sorted(
            self.middlewares,
            key=lambda m: int(m.priority),
            reverse=True  # Highest priority first
        )

        logger.info("Applying middleware stack in priority order:")

        applied_count = 0
        skipped_count = 0

        for middleware in sorted_middlewares:
            # Skip disabled middleware
            if not middleware.enabled:
                logger.debug(f"  ❌ {middleware.name} (disabled)")
                skipped_count += 1
                continue

            # Skip middleware that should be skipped in testing
            if is_testing and middleware.skip_in_testing:
                logger.debug(f"  ⏭️  {middleware.name} (skipped in testing)")
                skipped_count += 1
                continue

            try:
                self.app.add_middleware(
                    middleware.middleware_class,
                    **middleware.config
                )
                logger.info(
                    f"  ✓ {middleware.name} "
                    f"(priority={getattr(middleware.priority, 'name', str(int(middleware.priority)))}:{int(middleware.priority)})"
                )
                applied_count += 1

            except Exception as e:
                logger.error(
                    f"  ✗ Failed to apply {middleware.name}: {e}",
                    exc_info=True
                )
                raise

        self._applied = True
        logger.info(
            f"Middleware stack applied: {applied_count} active, {skipped_count} skipped"
        )

    def get_stack_summary(self) -> str:
        """
        Get a human-readable summary of the middleware stack

        Returns:
            Formatted string showing middleware order
        """
        if not self.middlewares:
            return "No middleware registered"

        sorted_middlewares = sorted(
            self.middlewares,
            key=lambda m: int(m.priority),
            reverse=True
        )

        lines = ["Middleware Stack (outermost → innermost):"]
        lines.append("=" * 60)

        for i, middleware in enumerate(sorted_middlewares, 1):
            status = "✓" if middleware.enabled else "✗"
            testing_flag = " [skip in testing]" if middleware.skip_in_testing else ""
            lines.append(
                f"{i:2d}. {status} {middleware.name:20s} "
                f"(priority={int(middleware.priority):5d}){testing_flag}"
            )

        lines.append("=" * 60)
        return "\n".join(lines)
