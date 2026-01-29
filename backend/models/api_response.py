"""
Standard API Response Models

This module defines the standard response format for all API endpoints.
All routers should return responses wrapped in ApiResponse for consistency.

Created: 2026-01-27
Part of: P1 API Standardization Initiative
"""

from typing import Generic, TypeVar, Optional, Any, Dict
from pydantic import BaseModel, Field


# Generic type variable for response data
T = TypeVar('T')


class PaginationMeta(BaseModel):
    """Pagination metadata for list responses"""

    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number (1-indexed)")
    limit: int = Field(..., description="Items per page")
    pages: Optional[int] = Field(None, description="Total number of pages")

    class Config:
        json_schema_extra = {
            "example": {
                "total": 100,
                "page": 1,
                "limit": 20,
                "pages": 5
            }
        }


class ApiResponse(BaseModel, Generic[T]):
    """
    Standard API response wrapper

    All API endpoints should return responses in this format for consistency.

    Examples:
        Success response with data:
        ```python
        return ApiResponse(
            success=True,
            data=stock_list,
            meta=PaginationMeta(total=100, page=1, limit=20)
        )
        ```

        Success response without data:
        ```python
        return ApiResponse(success=True)
        ```

        Error response:
        ```python
        return ApiResponse(
            success=False,
            error="Stock not found"
        )
        ```
    """

    success: bool = Field(..., description="Whether the request was successful")
    data: Optional[T] = Field(None, description="Response data (null on error)")
    error: Optional[str] = Field(None, description="Error message (null on success)")
    meta: Optional[PaginationMeta] = Field(None, description="Metadata (pagination, etc.)")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "data": {"id": 1, "symbol": "AAPL", "name": "Apple Inc."},
                "error": None,
                "meta": None
            }
        }


class ErrorResponse(BaseModel):
    """
    Standard error response

    Used for HTTP error responses (400, 401, 403, 404, 500, etc.)
    """

    success: bool = Field(False, description="Always False for errors")
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    code: Optional[str] = Field(None, description="Error code for programmatic handling")

    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": "Resource not found",
                "detail": "Stock with symbol 'INVALID' does not exist",
                "code": "RESOURCE_NOT_FOUND"
            }
        }


class SuccessResponse(BaseModel):
    """
    Standard success response without data

    Used for operations that don't return data (delete, update with no return, etc.)
    """

    success: bool = Field(True, description="Always True for success")
    message: Optional[str] = Field(None, description="Success message")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Resource deleted successfully"
            }
        }


# Convenience type aliases for common response types
def success_response(data: Optional[T] = None, meta: Optional[PaginationMeta] = None) -> ApiResponse[T]:
    """
    Helper function to create success responses

    Args:
        data: Response data
        meta: Pagination or other metadata

    Returns:
        ApiResponse with success=True
    """
    return ApiResponse(success=True, data=data, meta=meta)


def error_response(error: str, detail: Optional[str] = None, code: Optional[str] = None) -> ErrorResponse:
    """
    Helper function to create error responses

    Args:
        error: Error message
        detail: Detailed error information
        code: Error code for programmatic handling

    Returns:
        ErrorResponse with success=False
    """
    return ErrorResponse(success=False, error=error, detail=detail, code=code)


def paginated_response(
    data: list[T],
    total: int,
    page: int,
    limit: int
) -> ApiResponse[list[T]]:
    """
    Helper function to create paginated responses

    Args:
        data: List of items for current page
        total: Total number of items across all pages
        page: Current page number (1-indexed)
        limit: Items per page

    Returns:
        ApiResponse with pagination metadata
    """
    pages = (total + limit - 1) // limit if limit > 0 else 0
    meta = PaginationMeta(total=total, page=page, limit=limit, pages=pages)
    return ApiResponse(success=True, data=data, meta=meta)
