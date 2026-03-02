"""
Role-Based Access Control (RBAC) Module

WARNING: This is a STUB module. The in-memory role-permission mapping and
has_permission() work correctly for static checks. However, get_user_roles(),
assign_role(), revoke_role(), and check_access() are NOT implemented and will
raise NotImplementedError. See backend/auth/oauth2.py for production auth.

TODO: Implement full RBAC functionality with database integration in future phase.
"""

from typing import List, Optional, Dict, Any
from enum import Enum


class Role(str, Enum):
    """User roles"""
    ADMIN = "admin"
    USER = "user"
    ANALYST = "analyst"
    VIEWER = "viewer"


class Permission(str, Enum):
    """System permissions"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"


class RoleBasedAccessControl:
    """Role-Based Access Control manager (stub implementation)"""

    def __init__(self):
        self._roles: Dict[str, List[str]] = {
            Role.ADMIN: [Permission.READ, Permission.WRITE, Permission.DELETE, Permission.ADMIN],
            Role.ANALYST: [Permission.READ, Permission.WRITE],
            Role.USER: [Permission.READ, Permission.WRITE],
            Role.VIEWER: [Permission.READ],
        }

    def has_permission(self, role: str, permission: str) -> bool:
        """Check if role has permission"""
        role_permissions = self._roles.get(role, [])
        return permission in role_permissions or Permission.ADMIN in role_permissions

    def get_user_roles(self, user_id: int) -> List[str]:
        """Get roles for user.

        Raises NotImplementedError because this stub has no database backing.
        See backend/auth/oauth2.py for production auth.
        """
        raise NotImplementedError(
            "Stub: requires database integration. "
            "See backend/auth/oauth2.py for production auth."
        )

    def assign_role(self, user_id: int, role: str) -> bool:
        """Assign role to user.

        Raises NotImplementedError because this stub has no database backing.
        See backend/auth/oauth2.py for production auth.
        """
        raise NotImplementedError(
            "Stub: requires database integration. "
            "See backend/auth/oauth2.py for production auth."
        )

    def revoke_role(self, user_id: int, role: str) -> bool:
        """Revoke role from user.

        Raises NotImplementedError because this stub has no database backing.
        See backend/auth/oauth2.py for production auth.
        """
        raise NotImplementedError(
            "Stub: requires database integration. "
            "See backend/auth/oauth2.py for production auth."
        )

    def check_access(self, user_id: int, resource: str, action: str) -> bool:
        """Check if user has access to resource/action.

        Raises NotImplementedError because this depends on get_user_roles()
        which requires database integration.
        See backend/auth/oauth2.py for production auth.
        """
        raise NotImplementedError(
            "Stub: requires database integration. "
            "See backend/auth/oauth2.py for production auth."
        )
