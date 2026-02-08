"""
Base classes and types for domain contracts.

This module provides the foundational abstractions for all domain contracts,
ensuring consistent error handling, result wrapping, and contract verification.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union
import logging

logger = logging.getLogger(__name__)


class ContractErrorCode(Enum):
    """Standard error codes for contract violations."""
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    UNAUTHORIZED = "UNAUTHORIZED"
    RATE_LIMITED = "RATE_LIMITED"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    CONTRACT_VIOLATION = "CONTRACT_VIOLATION"
    TIMEOUT = "TIMEOUT"
    DATA_INTEGRITY_ERROR = "DATA_INTEGRITY_ERROR"


@dataclass(frozen=True)
class ContractError:
    """
    Represents an error from a contract operation.

    Attributes:
        code: Error classification code
        message: Human-readable error message
        details: Additional error context
        timestamp: When the error occurred
        source_domain: The domain that generated the error
    """
    code: ContractErrorCode
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source_domain: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            "code": self.code.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "source_domain": self.source_domain,
        }


T = TypeVar("T")


@dataclass
class ContractResult(Generic[T]):
    """
    Result wrapper for contract operations.

    Provides a consistent way to return success or failure from any
    contract method, enabling proper error handling across domain boundaries.

    Attributes:
        success: Whether the operation succeeded
        data: The result data (if successful)
        error: Error details (if failed)
        metadata: Additional operation metadata
    """
    success: bool
    data: Optional[T] = None
    error: Optional[ContractError] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, data: T, **metadata) -> "ContractResult[T]":
        """Create a successful result."""
        return cls(success=True, data=data, metadata=metadata)

    @classmethod
    def fail(
        cls,
        code: ContractErrorCode,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        source_domain: Optional[str] = None
    ) -> "ContractResult[T]":
        """Create a failed result."""
        error = ContractError(
            code=code,
            message=message,
            details=details,
            source_domain=source_domain
        )
        return cls(success=False, error=error)

    def map(self, func) -> "ContractResult":
        """Transform the result data if successful."""
        if self.success and self.data is not None:
            try:
                return ContractResult.ok(func(self.data), **self.metadata)
            except Exception as e:
                return ContractResult.fail(
                    ContractErrorCode.INTERNAL_ERROR,
                    f"Error mapping result: {str(e)}"
                )
        return self

    def unwrap(self) -> T:
        """Get the data or raise an exception if failed."""
        if not self.success:
            raise ValueError(f"Contract failed: {self.error}")
        return self.data

    def unwrap_or(self, default: T) -> T:
        """Get the data or return a default value."""
        if self.success and self.data is not None:
            return self.data
        return default


class DomainContract(ABC):
    """
    Base class for all domain contracts.

    Defines the interface that all domain contracts must implement,
    including health checks, capability declarations, and versioning.
    """

    @property
    @abstractmethod
    def domain_name(self) -> str:
        """Return the name of this domain."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Return the contract version."""
        pass

    @property
    @abstractmethod
    def capabilities(self) -> List[str]:
        """Return the list of capabilities this contract provides."""
        pass

    @abstractmethod
    async def health_check(self) -> ContractResult[Dict[str, Any]]:
        """
        Check if the domain service is healthy.

        Returns:
            ContractResult containing health status and diagnostics
        """
        pass

    def validate_contract(self) -> ContractResult[bool]:
        """
        Validate that this contract implementation is complete.

        Returns:
            ContractResult indicating if the contract is valid
        """
        errors = []

        # Check required properties
        if not self.domain_name:
            errors.append("domain_name is required")
        if not self.version:
            errors.append("version is required")
        if not self.capabilities:
            errors.append("capabilities list is required")

        if errors:
            return ContractResult.fail(
                ContractErrorCode.CONTRACT_VIOLATION,
                "Contract validation failed",
                details={"errors": errors},
                source_domain=self.domain_name
            )

        return ContractResult.ok(True)

    def log_operation(self, operation: str, details: Optional[Dict] = None):
        """Log a contract operation for traceability."""
        logger.info(
            f"[{self.domain_name}] {operation}",
            extra={"domain": self.domain_name, "details": details or {}}
        )


@dataclass
class ContractMetrics:
    """
    Metrics for contract operations.

    Tracks performance and reliability metrics for cross-domain calls.
    """
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_latency_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate the success rate."""
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls

    @property
    def average_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.successful_calls == 0:
            return 0.0
        return self.total_latency_ms / self.successful_calls

    def record_success(self, latency_ms: float):
        """Record a successful call."""
        self.total_calls += 1
        self.successful_calls += 1
        self.total_latency_ms += latency_ms

    def record_failure(self):
        """Record a failed call."""
        self.total_calls += 1
        self.failed_calls += 1
