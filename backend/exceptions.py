"""
Custom Exceptions for Investment Analysis Platform
Provides specialized exceptions for error handling across the application.
"""

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
