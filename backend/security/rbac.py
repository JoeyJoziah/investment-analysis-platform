"""
Role-Based Access Control (RBAC) Module

Provides in-memory role-permission management with support for multi-role
assignment per user. The has_permission() check uses a static role→permission
map. get_user_roles(), assign_role(), revoke_role(), and check_access() use
an in-memory user→roles store that can be populated at startup from the DB.

For JWT-based authentication see backend/auth/oauth2.py.
"""

from typing import List, Optional, Dict, Set
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


# Default role→permission mapping
ROLE_PERMISSIONS: Dict[str, List[str]] = {
    Role.ADMIN: [Permission.READ, Permission.WRITE, Permission.DELETE, Permission.ADMIN],
    Role.ANALYST: [Permission.READ, Permission.WRITE],
    Role.USER: [Permission.READ, Permission.WRITE],
    Role.VIEWER: [Permission.READ],
}


class RoleBasedAccessControl:
    """Role-Based Access Control manager with in-memory user-role store."""

    def __init__(self):
        self._roles: Dict[str, List[str]] = dict(ROLE_PERMISSIONS)
        self._user_roles: Dict[int, Set[str]] = {}

    def has_permission(self, role: str, permission: str) -> bool:
        """Check if role has permission"""
        role_permissions = self._roles.get(role, [])
        return permission in role_permissions or Permission.ADMIN in role_permissions

    def get_user_roles(self, user_id: int) -> List[str]:
        """Get roles assigned to a user.

        Returns an empty list if the user has no assigned roles.
        """
        return sorted(self._user_roles.get(user_id, set()))

    def assign_role(self, user_id: int, role: str) -> bool:
        """Assign a role to a user.

        Returns True if the role was newly assigned, False if already present.
        """
        if user_id not in self._user_roles:
            self._user_roles[user_id] = set()
        if role in self._user_roles[user_id]:
            return False
        self._user_roles[user_id].add(role)
        return True

    def revoke_role(self, user_id: int, role: str) -> bool:
        """Revoke a role from a user.

        Returns True if the role was removed, False if user didn't have it.
        """
        user_roles = self._user_roles.get(user_id)
        if user_roles is None or role not in user_roles:
            return False
        user_roles.discard(role)
        if not user_roles:
            del self._user_roles[user_id]
        return True

    def check_access(self, user_id: int, resource: str, action: str) -> bool:
        """Check if user has access to perform action on resource.

        Resolves user roles via get_user_roles(), then checks each role
        for the requested permission.
        """
        user_roles = self.get_user_roles(user_id)
        if not user_roles:
            return False
        return any(self.has_permission(role, action) for role in user_roles)
