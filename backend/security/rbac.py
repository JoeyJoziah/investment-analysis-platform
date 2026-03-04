"""
Role-Based Access Control (RBAC) Module

Provides role-permission management with support for multi-role assignment
per user.  Operates in two modes:

1. **In-memory** (default) — roles are stored in a dict; suitable for tests
   and single-process deployments.
2. **DB-backed** — when an optional SQLAlchemy ``Session`` is provided, roles
   are read from / written to the ``User.role`` column in the database.

The ``has_permission()`` check always uses the static ``ROLE_PERMISSIONS``
map regardless of storage backend.

For JWT-based authentication see ``backend/auth/oauth2.py``.
"""

from typing import List, Optional, Dict, Set
from enum import Enum

import logging

logger = logging.getLogger(__name__)


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
    """Role-Based Access Control manager.

    Args:
        db_session: Optional SQLAlchemy session.  When provided, user roles
            are persisted via the ``User.role`` column.  When ``None``
            (default), roles live only in memory (backward-compatible).
    """

    def __init__(self, db_session=None):
        self._roles: Dict[str, List[str]] = dict(ROLE_PERMISSIONS)
        self._user_roles: Dict[int, Set[str]] = {}
        self._db_session = db_session

    # ------------------------------------------------------------------
    # Internal DB helpers
    # ------------------------------------------------------------------

    def _get_user_from_db(self, user_id: int):
        """Fetch a User ORM object by id.  Returns None on miss."""
        if self._db_session is None:
            return None
        try:
            from backend.models.unified_models import User
            return self._db_session.query(User).filter(User.id == user_id).first()
        except Exception:
            logger.debug("DB lookup failed for user_id=%s", user_id)
            return None

    def _persist_role_to_db(self, user_id: int, role: str) -> bool:
        """Write a role to the User.role column.  Returns True on success."""
        user = self._get_user_from_db(user_id)
        if user is None:
            return False
        user.role = role
        self._db_session.commit()
        return True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def has_permission(self, role: str, permission: str) -> bool:
        """Check if role has permission"""
        role_permissions = self._roles.get(role, [])
        return permission in role_permissions or Permission.ADMIN in role_permissions

    def get_user_roles(self, user_id: int) -> List[str]:
        """Get roles assigned to a user.

        When DB-backed, reads from the ``User.role`` column and falls back
        to the in-memory store.  Returns an empty list if the user has no
        assigned roles.
        """
        if self._db_session is not None:
            user = self._get_user_from_db(user_id)
            if user is not None and user.role:
                return [user.role]
        return sorted(self._user_roles.get(user_id, set()))

    def assign_role(self, user_id: int, role: str) -> bool:
        """Assign a role to a user.

        Returns True if the role was newly assigned, False if already present.
        When DB-backed, also persists to the ``User.role`` column.
        """
        if user_id not in self._user_roles:
            self._user_roles[user_id] = set()
        if role in self._user_roles[user_id]:
            return False
        self._user_roles[user_id].add(role)
        if self._db_session is not None:
            self._persist_role_to_db(user_id, role)
        return True

    def revoke_role(self, user_id: int, role: str) -> bool:
        """Revoke a role from a user.

        Returns True if the role was removed, False if user didn't have it.
        When DB-backed, clears the ``User.role`` column.
        """
        user_roles = self._user_roles.get(user_id)
        if user_roles is None or role not in user_roles:
            return False
        user_roles.discard(role)
        if not user_roles:
            del self._user_roles[user_id]
        if self._db_session is not None:
            user = self._get_user_from_db(user_id)
            if user is not None:
                user.role = None
                self._db_session.commit()
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
