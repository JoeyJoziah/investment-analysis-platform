"""Custom exceptions for the investment analysis platform"""

from typing import Any, Optional


class InvestmentAnalysisException(Exception):
    """Base exception for all custom exceptions"""
    
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class APIException(InvestmentAnalysisException):
    """Base exception for API-related errors"""
    
    def __init__(
        self, 
        message: str, 
        status_code: int = 500,
        details: Optional[dict[str, Any]] = None
    ):
        super().__init__(message, details)
        self.status_code = status_code


class DataIngestionException(InvestmentAnalysisException):
    """Exception for data ingestion errors"""
    pass


class AnalysisException(InvestmentAnalysisException):
    """Exception for analysis-related errors"""
    pass


class MLModelException(InvestmentAnalysisException):
    """Exception for ML model errors"""
    pass


class ValidationException(APIException):
    """Exception for validation errors"""
    
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message, status_code=422, details=details)


class AuthenticationException(APIException):
    """Exception for authentication errors"""
    
    def __init__(self, message: str = "Authentication failed", details: Optional[dict[str, Any]] = None):
        super().__init__(message, status_code=401, details=details)


class AuthorizationException(APIException):
    """Exception for authorization errors"""
    
    def __init__(self, message: str = "Insufficient permissions", details: Optional[dict[str, Any]] = None):
        super().__init__(message, status_code=403, details=details)


class NotFoundException(APIException):
    """Exception for not found errors"""
    
    def __init__(self, resource: str, identifier: Any):
        message = f"{resource} not found: {identifier}"
        super().__init__(message, status_code=404, details={"resource": resource, "identifier": str(identifier)})


class RateLimitException(APIException):
    """Exception for rate limit errors"""
    
    def __init__(
        self, 
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
        details: Optional[dict[str, Any]] = None
    ):
        details = details or {}
        if retry_after:
            details["retry_after"] = retry_after
        super().__init__(message, status_code=429, details=details)


class ExternalAPIException(InvestmentAnalysisException):
    """Exception for external API errors"""
    
    def __init__(
        self,
        api_name: str,
        message: str,
        status_code: Optional[int] = None,
        details: Optional[dict[str, Any]] = None
    ):
        details = details or {}
        details.update({
            "api_name": api_name,
            "status_code": status_code
        })
        super().__init__(message, details)


class CircuitBreakerError(InvestmentAnalysisException):
    """Exception raised when circuit breaker is open"""
    
    def __init__(self, message: str, circuit_name: str):
        super().__init__(message, {"circuit_name": circuit_name})
        self.circuit_name = circuit_name


class DatabaseException(InvestmentAnalysisException):
    """Exception for database-related errors"""
    pass


class CacheException(InvestmentAnalysisException):
    """Exception for cache-related errors"""
    pass


class ConfigurationException(InvestmentAnalysisException):
    """Exception for configuration errors"""
    pass


class ComplianceException(InvestmentAnalysisException):
    """Exception for compliance-related errors"""
    
    def __init__(
        self,
        message: str,
        regulation: str,
        details: Optional[dict[str, Any]] = None
    ):
        details = details or {}
        details["regulation"] = regulation
        super().__init__(message, details)


class DataQualityException(InvestmentAnalysisException):
    """Exception for data quality issues"""
    
    def __init__(
        self,
        message: str,
        data_source: str,
        quality_score: Optional[float] = None,
        details: Optional[dict[str, Any]] = None
    ):
        details = details or {}
        details.update({
            "data_source": data_source,
            "quality_score": quality_score
        })
        super().__init__(message, details)


class CostLimitException(InvestmentAnalysisException):
    """Exception when approaching or exceeding cost limits"""
    
    def __init__(
        self,
        message: str,
        current_cost: float,
        limit: float,
        service: Optional[str] = None,
        details: Optional[dict[str, Any]] = None
    ):
        details = details or {}
        details.update({
            "current_cost": current_cost,
            "limit": limit,
            "service": service,
            "percentage_used": (current_cost / limit * 100) if limit > 0 else 100
        })
        super().__init__(message, details)