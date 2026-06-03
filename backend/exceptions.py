"""
Custom Exceptions for Investment Analysis Platform
Provides specialized exceptions for error handling across the application.
"""
from typing import Optional


class AppException(Exception):
    """Base application exception"""
    pass


class ValidationError(AppException):
    """Raised when data validation fails"""
    pass


class AuthenticationError(AppException):
    """Raised when authentication fails"""
    pass


class AuthorizationError(AppException):
    """Raised when user lacks permission"""
    pass


class NotFoundError(AppException):
    """Raised when a requested resource is not found"""
    pass


class ConflictError(AppException):
    """Raised when a resource conflict occurs"""
    pass


class StaleDataError(ConflictError):
    """
    Raised when optimistic locking detects concurrent modification.

    This exception indicates that the data being updated has been modified
    by another transaction since it was read. The client should:
    1. Re-fetch the latest data
    2. Re-apply their changes
    3. Retry the update operation

    Attributes:
        entity_type: Type of entity (Portfolio, Position, InvestmentThesis)
        entity_id: ID of the entity
        expected_version: Version the client had
        current_version: Current version in database
    """
    def __init__(
        self,
        entity_type: str,
        entity_id: int,
        expected_version: int,
        current_version: int
    ):
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.expected_version = expected_version
        self.current_version = current_version

        message = (
            f"Concurrent modification detected for {entity_type} ID {entity_id}. "
            f"Expected version {expected_version}, but current version is {current_version}. "
            f"Please refresh and retry."
        )
        super().__init__(message)


class DatabaseError(AppException):
    """Raised when database operation fails"""
    pass


class ExternalServiceError(AppException):
    """Raised when external service call fails"""
    pass


class RateLimitError(AppException):
    """Raised when rate limit is exceeded"""
    pass


class InsufficientBalanceError(ConflictError):
    """Raised when portfolio has insufficient cash balance"""
    pass


class InvalidPositionError(ConflictError):
    """Raised when attempting to sell more shares than owned"""
    pass


class InsufficientDataError(AppException):
    """
    Raised when insufficient data is available to compute a result.

    Used in analysis pipelines (risk metrics, drift detection, feature stats)
    when the upstream data is missing, too short, or otherwise inadequate to
    produce a faithful answer. Callers should surface this as HTTP 503
    `model_unavailable` rather than fabricating placeholder values.

    Per PRD audit 2026-04 Q4 default: returning fake values for SEC-regulated
    investment outputs is a compliance exposure; refuse instead.
    """
    def __init__(self, reason: str = "insufficient_data", details: Optional[dict] = None):
        self.reason = reason
        self.details = details or {}
        super().__init__(f"Insufficient data: {reason}")


class ModelUnavailableError(AppException):
    """
    Raised when an ML model required to serve a request is not available.

    Triggered when `model_manager` is in `Dummy*` fallback (model binaries
    missing) or when a feature pipeline cannot supply real inputs. API
    handlers convert this to HTTP 503 with a structured
    `{"error": "model_unavailable", "model": ..., "reason": ...}` payload.

    Per PRD audit 2026-04 §3 D and Q4 default (recorded 2026-04-28).
    """
    def __init__(
        self,
        model: str = "unknown",
        reason: str = "fallback_active",
        request_id: Optional[str] = None,
    ):
        self.model = model
        self.reason = reason
        self.request_id = request_id
        super().__init__(
            f"Model '{model}' unavailable ({reason}); "
            "platform refuses to fabricate values."
        )
