"""
Authentication utilities wrapper for backward compatibility.

This module provides authentication utilities by wrapping the core
authentication modules in backend.auth and backend.security.

The get_current_user function here is a thin wrapper around the canonical
backend.auth.oauth2.get_current_user that converts the returned User ORM
object into a plain dict. Routers in agents.py and monitoring.py depend on
the dict return type (e.g. current_user.get("username")), while most other
routers depend directly on the ORM-returning version in backend.auth.oauth2.

By using Depends(_get_current_user_orm) in the function signature, FastAPI's
dependency injection system ensures that overriding backend.auth.oauth2.get_current_user
in tests automatically cascades here too - eliminating the double-mock requirement.
"""

from typing import Optional, Any, Dict
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

# Import the canonical ORM-returning function as the single source of truth.
# This wrapper depends on it via FastAPI DI so test overrides cascade automatically.
from backend.auth.oauth2 import (
    get_current_user as _get_current_user_orm,
    get_current_user_from_token,
)
from backend.security.enhanced_auth import (
    UserRole,
    require_role,
    get_auth_manager,
    UserSession,
)

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token", auto_error=False)


def _user_to_dict(user: Any) -> Dict[str, Any]:
    """
    Convert a User ORM object (or an already-resolved dict) to a plain dict.

    Handles both the real User ORM and the test fixture case where the
    dependency override already returns a dict (so we do not double-convert).
    """
    if isinstance(user, dict):
        return user

    return {
        "id": getattr(user, "id", None),
        "user_id": getattr(user, "user_id", None),
        "username": getattr(user, "username", None),
        "email": getattr(user, "email", None),
        "full_name": getattr(user, "full_name", None),
        "role": getattr(user, "role", "user"),
        "is_active": getattr(user, "is_active", True),
        "is_admin": getattr(user, "is_admin", False),
        "is_premium": getattr(user, "is_premium", False),
        "is_verified": getattr(user, "is_verified", False),
    }


async def get_current_user(
    user: Any = Depends(_get_current_user_orm),
) -> Dict[str, Any]:
    """
    Get the current authenticated user as a plain dict.

    This is a thin wrapper over backend.auth.oauth2.get_current_user that
    converts the User ORM object to a dict for routers that expect dict-style
    access (e.g. current_user.get("username"), current_user["email"]).

    Because this function declares Depends(_get_current_user_orm) in its
    signature, any test override of backend.auth.oauth2.get_current_user
    automatically propagates here - no separate override is needed.

    Args:
        user: Resolved User ORM object (or dict if already overridden in tests)

    Returns:
        Dict containing user fields: id, user_id, username, email, full_name,
        role, is_active, is_admin, is_premium, is_verified
    """
    return _user_to_dict(user)


async def get_optional_user(
    token: Optional[str] = Depends(oauth2_scheme),
) -> Optional[Dict[str, Any]]:
    """
    Get the current user as a dict if authenticated, otherwise return None.

    Useful for endpoints that work differently for authenticated vs anonymous
    users.

    Args:
        token: The OAuth2 bearer token from the request (optional)

    Returns:
        Dict of user fields or None if not authenticated
    """
    if token is None:
        return None

    try:
        user = await _get_current_user_orm(token)
        return _user_to_dict(user)
    except HTTPException:
        return None


def require_admin(current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Dependency that requires the current user to be an admin.

    Args:
        current_user: The current authenticated user dict

    Returns:
        The current user dict if they are an admin

    Raises:
        HTTPException: If the user is not an admin
    """
    # Support both dict and ORM access patterns
    if isinstance(current_user, dict):
        user_role = current_user.get("role")
        user_is_admin = current_user.get("is_admin", False)
    else:
        user_role = getattr(current_user, "role", None)
        user_is_admin = getattr(current_user, "is_admin", False)

    # Accept multiple admin role representations
    admin_roles = [UserRole.ADMIN, UserRole.SUPER_ADMIN, "admin", "super_admin", "ADMIN", "SUPER_ADMIN"]

    if user_role not in admin_roles and not user_is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )

    return current_user


def require_role_check(*allowed_roles: UserRole):
    """
    Factory function that creates a dependency for role-based access control.

    Args:
        allowed_roles: List of roles that are allowed to access the endpoint

    Returns:
        A dependency function that checks user roles
    """
    async def role_checker(current_user: Dict[str, Any] = Depends(get_current_user)):
        if isinstance(current_user, dict):
            user_role = current_user.get("role")
        else:
            user_role = getattr(current_user, "role", None)

        if user_role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required roles: {[r.value for r in allowed_roles]}",
            )

        return current_user

    return role_checker


# Export commonly used items
__all__ = [
    "get_current_user",
    "get_optional_user",
    "require_admin",
    "require_role_check",
    "UserRole",
    "oauth2_scheme",
]
