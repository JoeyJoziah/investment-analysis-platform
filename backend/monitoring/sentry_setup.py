"""
Optional Sentry error monitoring bootstrap (#102).

Initializes sentry_sdk when SENTRY_DSN is set. Safe no-op when the DSN is
missing or sentry_sdk is not installed — never raises into the app lifecycle.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

_initialized = False


def init_sentry(
    *,
    dsn: Optional[str] = None,
    environment: Optional[str] = None,
    traces_sample_rate: Optional[float] = None,
) -> bool:
    """
    Initialize Sentry for the backend process.

    Returns:
        True if Sentry was initialized, False if skipped or failed.
    """
    global _initialized
    if _initialized:
        return True

    dsn = (dsn if dsn is not None else os.getenv("SENTRY_DSN", "")).strip()
    if not dsn or dsn.startswith("optional_"):
        logger.info("Sentry DSN not configured — error monitoring disabled")
        return False

    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.logging import LoggingIntegration
    except ImportError:
        logger.warning(
            "sentry_sdk not installed — set SENTRY_DSN and install sentry-sdk to enable"
        )
        return False

    env = environment or os.getenv("SENTRY_ENVIRONMENT") or os.getenv(
        "ENVIRONMENT", "development"
    )
    try:
        rate = (
            traces_sample_rate
            if traces_sample_rate is not None
            else float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.1"))
        )
    except ValueError:
        rate = 0.1

    try:
        sentry_sdk.init(
            dsn=dsn,
            environment=env,
            traces_sample_rate=max(0.0, min(1.0, rate)),
            integrations=[
                FastApiIntegration(),
                LoggingIntegration(level=logging.INFO, event_level=logging.ERROR),
            ],
            send_default_pii=False,
        )
        _initialized = True
        logger.info("Sentry error monitoring initialized (env=%s)", env)
        return True
    except Exception as exc:  # pragma: no cover - never break startup
        logger.warning("Sentry init failed (continuing without it): %s", exc)
        return False


def capture_exception(exc: BaseException, **context: Any) -> None:
    """Best-effort exception capture when Sentry is active."""
    if not _initialized:
        return
    try:
        import sentry_sdk

        with sentry_sdk.push_scope() as scope:
            for key, value in context.items():
                scope.set_extra(key, value)
            sentry_sdk.capture_exception(exc)
    except Exception as err:  # pragma: no cover
        logger.debug("sentry capture_exception failed: %s", err)


def is_sentry_enabled() -> bool:
    """Return whether Sentry was successfully initialized in this process."""
    return _initialized
