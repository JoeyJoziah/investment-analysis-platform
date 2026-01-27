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


def register_exception_handlers(app):
    """
    Register all exception handlers with the FastAPI app

    Usage in main.py:
        from backend.middleware.error_handler import register_exception_handlers

        app = FastAPI()
        register_exception_handlers(app)
    """
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(ValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)
