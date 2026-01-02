"""
Authentication utilities wrapper for backward compatibility.

This module provides authentication utilities by wrapping the core
authentication modules in backend.auth and backend.security.
"""

from functools import wraps
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

# Import from core authentication modules
from backend.auth.oauth2 import (
    get_current_user as _get_current_user,
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


async def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    Get the current authenticated user from the request token.
    
    This is a wrapper that uses the core authentication module.
    
    Args:
        token: The OAuth2 bearer token from the request
        
    Returns:
        The authenticated user or raises HTTPException
    """
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return await _get_current_user(token)


async def get_optional_user(token: Optional[str] = Depends(oauth2_scheme)):
    """
    Get the current user if authenticated, otherwise return None.
    
    Useful for endpoints that work differently for authenticated vs anonymous users.
    
    Args:
        token: The OAuth2 bearer token from the request (optional)
        
    Returns:
        The authenticated user or None
    """
    if token is None:
        return None
    
    try:
        return await _get_current_user(token)
    except HTTPException:
        return None


def require_admin(current_user = Depends(get_current_user)):
    """
    Dependency that requires the current user to be an admin.
    
    Args:
        current_user: The current authenticated user
        
    Returns:
        The current user if they are an admin
        
    Raises:
        HTTPException: If the user is not an admin
    """
    # Check if user has admin role
    user_role = getattr(current_user, 'role', None)
    
    # Accept multiple admin role representations
    admin_roles = [UserRole.ADMIN, UserRole.SUPER_ADMIN, 'admin', 'super_admin', 'ADMIN', 'SUPER_ADMIN']
    
    if user_role not in admin_roles:
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
    async def role_checker(current_user = Depends(get_current_user)):
        user_role = getattr(current_user, 'role', None)
        
        if user_role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required roles: {[r.value for r in allowed_roles]}",
            )
        
        return current_user
    
    return role_checker


# Export commonly used items
__all__ = [
    'get_current_user',
    'get_optional_user',
    'require_admin',
    'require_role_check',
    'UserRole',
    'oauth2_scheme',
]
