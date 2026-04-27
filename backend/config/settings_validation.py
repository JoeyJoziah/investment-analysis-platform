"""
Security Validation Utilities for Application Settings

This module provides secure validation functions for:
- Secret key validation (preventing weak/default keys)
- Production environment validation
- Security configuration checks
"""

import secrets
import logging
from typing import Callable

logger = logging.getLogger(__name__)


def generate_secure_key(length: int = 64) -> str:
    """
    Generate a cryptographically secure random key.

    Args:
        length: Number of random bytes to use (default 64)

    Returns:
        URL-safe base64-encoded random string
    """
    return secrets.token_urlsafe(length)


def validate_secret_key(value: str, key_name: str) -> str:
    """
    Validate that a secret key is secure and not a default/weak value.

    Security requirements:
    - Must not be empty
    - Must not contain common weak patterns
    - Must be at least 32 characters long

    Args:
        value: The secret key value to validate
        key_name: Name of the key for error messages

    Returns:
        The validated value

    Raises:
        ValueError: If the key fails validation
    """
    if not value:
        raise ValueError(f"{key_name} must be set - use a cryptographically secure random value")

    # List of weak patterns that indicate a default/placeholder value
    weak_patterns = [
        "your-", "change-", "secret", "password", "default",
        "example", "test", "dev", "123", "abc", "xxx", "changeme",
        "placeholder", "replace", "todo", "fixme"
    ]

    value_lower = value.lower()
    for pattern in weak_patterns:
        if pattern in value_lower:
            raise ValueError(
                f"{key_name} contains weak pattern '{pattern}'. "
                f"Use a cryptographically secure random value instead. "
                f"Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(64))\""
            )

    if len(value) < 32:
        raise ValueError(
            f"{key_name} must be at least 32 characters long for security. "
            f"Current length: {len(value)}. "
            f"Generate a secure key with: python -c \"import secrets; print(secrets.token_urlsafe(64))\""
        )

    return value


def validate_production_debug(debug: bool, environment: str) -> bool:
    """
    Validate that DEBUG mode is not enabled in production.

    Args:
        debug: The DEBUG setting value
        environment: The ENVIRONMENT setting value

    Returns:
        The validated debug value

    Raises:
        ValueError: If DEBUG is True in production
    """
    if environment == 'production' and debug:
        raise ValueError(
            "DEBUG mode cannot be enabled in production environment. "
            "Set ENVIRONMENT to 'development' or 'staging' to enable DEBUG mode, "
            "or set DEBUG=false for production."
        )
    return debug


def warn_debug_logging_in_production(log_level: str, environment: str) -> None:
    """
    Warn if DEBUG log level is used in production.

    Args:
        log_level: The LOG_LEVEL setting value
        environment: The ENVIRONMENT setting value
    """
    if environment == 'production' and log_level == 'DEBUG':
        logger.warning(
            "DEBUG log level in production may expose sensitive information. "
            "Consider using INFO or WARNING level instead."
        )


# Validator factory functions for pydantic field_validator
def create_secret_key_validator(key_name: str) -> Callable:
    """
    Create a pydantic field_validator for secret keys.

    Args:
        key_name: Name of the key being validated

    Returns:
        Validator function for use with @field_validator
    """
    def validator(cls, v):
        return validate_secret_key(v, key_name)
    return validator
