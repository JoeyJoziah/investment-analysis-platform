"""
Response Utilities - Field Filtering for API Responses

Provides utilities for filtering API response data to only include requested fields,
reducing payload size and improving response times.

Created: 2026-02-08
Part of: Issue #12 - API Response Time Optimization
"""

from typing import Any, Dict, List, Optional, Set, Union
import logging

logger = logging.getLogger(__name__)


def filter_response_fields(
    data: Union[Dict[str, Any], List[Dict[str, Any]]],
    fields: Optional[str]
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Filter response data to only include specified fields.

    Args:
        data: Response data (dict or list of dicts)
        fields: Comma-separated field names (e.g., "ticker,price,change")
                Supports nested fields with dot notation (e.g., "user.name,user.email")

    Returns:
        Filtered data with only requested fields

    Examples:
        >>> data = {"ticker": "AAPL", "price": 150.0, "volume": 1000000}
        >>> filter_response_fields(data, "ticker,price")
        {"ticker": "AAPL", "price": 150.0}

        >>> data = [{"ticker": "AAPL", "price": 150.0}, {"ticker": "GOOGL", "price": 2800.0}]
        >>> filter_response_fields(data, "ticker")
        [{"ticker": "AAPL"}, {"ticker": "GOOGL"}]

        >>> data = {"user": {"name": "John", "email": "john@example.com"}, "price": 150.0}
        >>> filter_response_fields(data, "user.name,price")
        {"user": {"name": "John"}, "price": 150.0}
    """
    # If no fields specified, return full response
    if not fields or not fields.strip():
        return data

    # Parse field list
    field_list = [f.strip() for f in fields.split(",") if f.strip()]
    if not field_list:
        return data

    # Handle list of objects
    if isinstance(data, list):
        return [_filter_dict_fields(item, field_list) for item in data]

    # Handle single object
    if isinstance(data, dict):
        return _filter_dict_fields(data, field_list)

    # For non-dict/non-list data, return as-is
    return data


def _filter_dict_fields(
    data: Dict[str, Any],
    field_list: List[str]
) -> Dict[str, Any]:
    """
    Filter a single dictionary to only include specified fields.

    Supports nested field notation (e.g., "user.name" to get data["user"]["name"])

    Args:
        data: Dictionary to filter
        field_list: List of field names to include

    Returns:
        Filtered dictionary
    """
    result = {}

    # Group nested fields by parent
    nested_fields = {}  # {parent: [child1, child2, ...]}
    direct_fields = []  # [field1, field2, ...]

    for field in field_list:
        if "." in field:
            # Nested field (e.g., "user.name")
            parts = field.split(".", 1)
            parent = parts[0]
            child = parts[1]

            if parent not in nested_fields:
                nested_fields[parent] = []
            nested_fields[parent].append(child)
        else:
            # Direct field
            direct_fields.append(field)

    # Add direct fields
    for field in direct_fields:
        if field in data:
            result[field] = data[field]
        else:
            # Field not found - ignore silently (clients may request fields that don't exist)
            logger.debug(f"Field '{field}' not found in data")

    # Add nested fields
    for parent, children in nested_fields.items():
        if parent in data and isinstance(data[parent], dict):
            # Recursively filter nested object
            filtered_nested = _filter_dict_fields(data[parent], children)
            if filtered_nested:  # Only add if not empty
                result[parent] = filtered_nested
        elif parent in data and isinstance(data[parent], list):
            # Handle list of nested objects
            filtered_list = [
                _filter_dict_fields(item, children)
                for item in data[parent]
                if isinstance(item, dict)
            ]
            if filtered_list:
                result[parent] = filtered_list
        else:
            logger.debug(f"Nested field parent '{parent}' not found or not a dict/list")

    return result


def parse_field_list(fields: Optional[str]) -> List[str]:
    """
    Parse a comma-separated field list into a list of field names.

    Args:
        fields: Comma-separated field names

    Returns:
        List of field names (empty list if fields is None or empty)

    Examples:
        >>> parse_field_list("ticker,price,change")
        ["ticker", "price", "change"]

        >>> parse_field_list("  ticker , price  ")
        ["ticker", "price"]

        >>> parse_field_list(None)
        []

        >>> parse_field_list("")
        []
    """
    if not fields or not fields.strip():
        return []

    return [f.strip() for f in fields.split(",") if f.strip()]


def validate_fields(
    fields: Optional[str],
    allowed_fields: Set[str]
) -> tuple[bool, Optional[str]]:
    """
    Validate that requested fields are allowed.

    Args:
        fields: Comma-separated field names
        allowed_fields: Set of allowed field names

    Returns:
        Tuple of (is_valid, error_message)

    Examples:
        >>> validate_fields("ticker,price", {"ticker", "price", "volume"})
        (True, None)

        >>> validate_fields("ticker,invalid", {"ticker", "price"})
        (False, "Invalid fields: invalid")
    """
    if not fields or not fields.strip():
        return True, None

    field_list = parse_field_list(fields)
    invalid_fields = [f for f in field_list if f not in allowed_fields]

    if invalid_fields:
        return False, f"Invalid fields: {', '.join(invalid_fields)}"

    return True, None
