"""
Numpy Serialization Utilities

Converts numpy arrays and scalar types to native Python types for JSON serialization.
This prevents Pydantic serialization errors when numpy objects are included in API responses.
"""

from typing import Any, Dict, List, Union
import numpy as np


def sanitize_numpy(obj: Any) -> Any:
    """
    Recursively convert numpy types to native Python types for JSON serialization.

    Args:
        obj: Any object that may contain numpy types

    Returns:
        The same object structure with numpy types converted to Python types

    Examples:
        >>> sanitize_numpy(np.array([1, 2, 3]))
        [1, 2, 3]

        >>> sanitize_numpy(np.float64(3.14))
        3.14

        >>> sanitize_numpy({"prices": np.array([1.5, 2.5]), "count": np.int64(10)})
        {"prices": [1.5, 2.5], "count": 10}
    """
    # Handle numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # Handle numpy floating point types
    # Note: np.float128 doesn't exist on all platforms (e.g., macOS)
    float_types = (np.float16, np.float32, np.float64)
    if hasattr(np, 'float128'):
        float_types = float_types + (np.float128,)
    if isinstance(obj, float_types):
        return float(obj)

    # Handle numpy integer types
    if isinstance(obj, (np.int8, np.int16, np.int32, np.int64,
                        np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)

    # Handle numpy boolean
    if isinstance(obj, np.bool_):
        return bool(obj)

    # Handle numpy complex types (convert to string for JSON compatibility)
    if isinstance(obj, (np.complex64, np.complex128)):
        return str(obj)

    # Recursively handle dictionaries
    if isinstance(obj, dict):
        return {k: sanitize_numpy(v) for k, v in obj.items()}

    # Recursively handle lists and tuples
    if isinstance(obj, (list, tuple)):
        return [sanitize_numpy(item) for item in obj]

    # Return unchanged if not a numpy type
    return obj


def ensure_json_serializable(data: Union[Dict, List, Any]) -> Union[Dict, List, Any]:
    """
    Ensure data structure is JSON-serializable by converting numpy types.

    This is an alias for sanitize_numpy with a more explicit name.

    Args:
        data: Data structure to sanitize

    Returns:
        JSON-serializable version of the data
    """
    return sanitize_numpy(data)
