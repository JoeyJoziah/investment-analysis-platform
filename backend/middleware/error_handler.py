"""
Global Error Handler Middleware

This middleware provides standardized error handling across all API endpoints.
All exceptions are converted to the standard ErrorResponse format.

Created: 2026-01-27
Part of: P1 API Standardization Initiative
"""

from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError, HTTPException
from pydantic import ValidationError
from typing import Union
import logging

from backend.models.api_response import ErrorResponse, error_response

logger = logging.getLogger(__name__)


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    Handle FastAPI HTTPException instances

    Converts HTTPException to standardized ErrorResponse format
    """
    logger.warning(
        f"HTTP exception: {exc.status_code} - {exc.detail}",
        extra={"path": request.url.path, "method": request.method}
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=error_response(
            error=exc.detail,
            code=f"HTTP_{exc.status_code}"
        ).model_dump()
    )


async def validation_exception_handler(
    request: Request,
    exc: Union[RequestValidationError, ValidationError]
) -> JSONResponse:
    """
    Handle Pydantic validation errors

    Converts validation errors to standardized ErrorResponse with details
    """
    # Extract validation errors
    errors = []
    for error in exc.errors():
        field = ".".join(str(loc) for loc in error["loc"])
        message = error["msg"]
        errors.append(f"{field}: {message}")

    error_detail = "; ".join(errors)

    logger.warning(
        f"Validation error: {error_detail}",
        extra={"path": request.url.path, "method": request.method}
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response(
            error="Validation error",
            detail=error_detail,
            code="VALIDATION_ERROR"
        ).model_dump()
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle all other unhandled exceptions

    Catches any exception that wasn't handled by specific handlers
    """
    logger.error(
        f"Unhandled exception: {str(exc)}",
        extra={"path": request.url.path, "method": request.method},
        exc_info=True
    )

    # Don't expose internal error details in production
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response(
            error="Internal server error",
            detail="An unexpected error occurred. Please try again later.",
            code="INTERNAL_ERROR"
        ).model_dump()
    )


async def model_unavailable_handler(request: Request, exc) -> JSONResponse:
    """Convert ``ModelUnavailableError`` / ``InsufficientDataError`` to 503.

    Per PRD audit 2026-04 Workstream D / Q4 default: refusing-to-serve when
    ML models are in DummyLSTM/DummyXGBoost/DummyProphet fallback or when
    feature data is insufficient is the SEC-conservative posture. Returns
    the canonical structured payload so the frontend (G3 phase 4) has a
    stable contract: ``{error, model, reason, request_id}``.
    """
    import uuid as _uuid
    from backend.exceptions import InsufficientDataError, ModelUnavailableError

    if isinstance(exc, ModelUnavailableError):
        model = exc.model
        reason = exc.reason
        rid = exc.request_id or _uuid.uuid4().hex
    elif isinstance(exc, InsufficientDataError):
        model = (exc.details or {}).get("metric") or (exc.details or {}).get("feature") or "unknown"
        reason = exc.reason or "insufficient_data"
        rid = _uuid.uuid4().hex
    else:  # pragma: no cover - defensive
        model = "unknown"
        reason = "fallback_active"
        rid = _uuid.uuid4().hex

    logger.warning(
        "model_unavailable model=%s reason=%s path=%s request_id=%s",
        model, reason, request.url.path, rid,
    )

    try:  # Sentry breadcrumb tagged ``model_unavailable``
        import sentry_sdk
        sentry_sdk.add_breadcrumb(
            category="model_unavailable",
            message=f"503 model_unavailable: {model} ({reason})",
            level="warning",
            data={
                "model": model,
                "reason": reason,
                "request_id": rid,
                "endpoint": request.url.path,
            },
        )
    except Exception:  # pragma: no cover - never let telemetry crash a 503
        pass

    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": "model_unavailable",
            "model": model,
            "reason": reason,
            "request_id": rid,
        },
    )


def register_exception_handlers(app):
    """
    Register all exception handlers with the FastAPI app

    Usage in main.py:
        from backend.middleware.error_handler import register_exception_handlers

        app = FastAPI()
        register_exception_handlers(app)
    """
    from backend.exceptions import InsufficientDataError, ModelUnavailableError

    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(ValidationError, validation_exception_handler)
    # F-02-003 / F-03-003 / F-03-005: structured 503 ``model_unavailable``
    app.add_exception_handler(ModelUnavailableError, model_unavailable_handler)
    app.add_exception_handler(InsufficientDataError, model_unavailable_handler)
    app.add_exception_handler(Exception, general_exception_handler)
