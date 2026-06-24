"""
Regression tests for audit finding #198 — Unauthenticated GDPR destructive/PII
endpoints (OWASP A01, Critical).

Before the fix, three GDPR endpoints had no auth dependency:
- POST /users/me/delete-request/{request_id}/process (process_deletion_request)
- POST /admin/retention/enforce               (enforce_retention_policies)
- GET  /users/me/delete-request/{request_id}/audit (get_deletion_audit)

The fix requires:
- process_deletion_request / enforce_retention_policies -> admin only.
- get_deletion_audit -> authenticated user + ownership enforcement.
- A router-level get_current_user dependency as defense-in-depth.

These tests assert at the route-signature level that each endpoint declares the
correct auth dependency, plus a unit test of the ownership helper. This avoids
wiring the full app/TestClient while still proving the security contract.
"""

# T2.7: required env must exist before importing backend modules — settings
# instantiates at import and security_config fails fast in production. Setting a
# non-production ENVIRONMENT keeps this module collectable under `--noconftest`.
import os

os.environ.setdefault("TESTING", "True")
os.environ.setdefault("ENVIRONMENT", "testing")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-testing-only")
os.environ.setdefault("JWT_SECRET_KEY", "test-jwt-secret-key-for-testing-only")
os.environ.setdefault("SESSION_SECRET_KEY", "test-session-secret-for-testing-only")
os.environ.setdefault("MASTER_SECRET_KEY", "m" * 130)

import hashlib
from types import SimpleNamespace

import pytest

from backend.api.routers import gdpr
from backend.api.routers.gdpr import (
    router,
    get_deletion_audit,
    process_deletion_request,
    enforce_retention_policies,
)
from backend.auth.oauth2 import get_current_user, get_current_admin_user

# Imported defensively: against the pre-fix module this helper does not exist.
# Keeping the import optional ensures the auth-dependency assertions below still
# run (and fail meaningfully) rather than erroring at collection time.
_owns_deletion_audit = getattr(gdpr, "_owns_deletion_audit", None)


def _auth_dependency_names(endpoint) -> set:
    """Collect every auth-related dependency callable name for an endpoint.

    Walks both the route's own dependant tree and the router-level dependencies,
    so router-wide defense-in-depth guards are included.
    """
    names: set = set()

    def _call_name(call) -> str:
        # Most dependencies are functions (have __name__); some, like
        # OAuth2PasswordBearer, are class instances — fall back to the class name.
        return getattr(call, "__name__", type(call).__name__)

    def collect(dep) -> None:
        if dep.call is not None:
            names.add(_call_name(dep.call))
        for sub in dep.dependencies:
            collect(sub)

    matched = False
    for route in router.routes:
        if getattr(route, "endpoint", None) is endpoint:
            matched = True
            for dep in route.dependant.dependencies:
                collect(dep)
    assert matched, f"endpoint {endpoint.__name__} not registered on router"

    # Router-level dependencies apply to every route (defense-in-depth).
    # These are raw ``Depends`` markers exposing ``.dependency`` (the callable),
    # unlike the solved ``Dependant`` nodes above which expose ``.call``.
    for dep in router.dependencies:
        call = getattr(dep, "dependency", None)
        if call is not None:
            names.add(_call_name(call))
    return names


def test_router_requires_authentication_defense_in_depth():
    """Router-level guard: no GDPR route may be reachable anonymously."""
    router_dep_names = {
        getattr(d.dependency, "__name__", type(d.dependency).__name__)
        for d in router.dependencies
        if getattr(d, "dependency", None) is not None
    }
    assert get_current_user.__name__ in router_dep_names, (
        "Finding #198: GDPR router must declare a router-level "
        "get_current_user dependency (rejects 401 for anonymous callers)."
    )


def test_process_deletion_request_requires_admin():
    """Destructive erasure processing must be admin-only (rejects non-admin 403)."""
    names = _auth_dependency_names(process_deletion_request)
    assert get_current_admin_user.__name__ in names, (
        "Finding #198: process_deletion_request must require "
        "get_current_admin_user; anonymous/non-admin callers must be rejected."
    )


def test_enforce_retention_policies_requires_admin():
    """Destructive retention scheduling must be admin-only (rejects non-admin 403)."""
    names = _auth_dependency_names(enforce_retention_policies)
    assert get_current_admin_user.__name__ in names, (
        "Finding #198: enforce_retention_policies must require "
        "get_current_admin_user; anonymous/non-admin callers must be rejected."
    )


def test_get_deletion_audit_requires_authentication():
    """Audit read must require an authenticated user (rejects anonymous 401)."""
    names = _auth_dependency_names(get_deletion_audit)
    assert get_current_user.__name__ in names, (
        "Finding #198: get_deletion_audit must require get_current_user; "
        "anonymous IDOR enumeration must be rejected with 401."
    )


def test_get_deletion_audit_enforces_ownership_helper_exists():
    """Ownership enforcement must be wired (helper present and used)."""
    assert callable(_owns_deletion_audit), (
        "Finding #198: get_deletion_audit must enforce ownership so a "
        "non-owner cannot read another user's deletion audit."
    )


def test_owns_deletion_audit_matches_only_the_subject():
    """The ownership helper grants access only to the audit's subject user."""
    assert callable(_owns_deletion_audit), (
        "Finding #198: ownership helper _owns_deletion_audit must exist."
    )
    owner = SimpleNamespace(id=42)
    other = SimpleNamespace(id=999)
    reference = hashlib.sha256(str(owner.id).encode()).hexdigest()[:16]
    audit = {"anonymized_user_reference": reference}

    assert _owns_deletion_audit(audit, owner) is True
    assert _owns_deletion_audit(audit, other) is False
