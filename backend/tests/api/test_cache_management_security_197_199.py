"""
Security regression tests for cache_management router (findings #197 / #199).

#197 — Redis SCAN pattern injection: a user-supplied ``pattern`` query param
       reached ``cache_manager.invalidate_pattern`` (-> ``redis.keys(pattern)``)
       unbounded, so ``*`` flushed the entire keyspace. Patterns must now be
       anchored to an allowlisted namespace prefix.

#199 — ``/cache/invalidate`` and ``/cache/warm`` shipped with authentication
       commented out while the router was registered. Both destructive
       endpoints must now declare a real admin auth dependency.

These tests are unit/introspection level by design: they exercise the pure
pattern-validation function directly and assert route dependencies via FastAPI
router introspection, avoiding heavy full-app wiring.
"""

# Required settings must exist before importing the router module, since
# backend.config.settings instantiates Settings() at import time.
import os

os.environ.setdefault("TESTING", "True")
# T2.7: ENVIRONMENT must be non-production so security_config does not raise
# InsecureSecretError at import time (it fails fast in production only). Keeps
# the module collectable under `--noconftest` without weakening the #201
# import-time guard (which has its own dedicated test).
os.environ.setdefault("ENVIRONMENT", "testing")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/1")
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-testing-only")
os.environ.setdefault("JWT_SECRET_KEY", "test-jwt-secret-key-for-testing-only")
os.environ.setdefault("SESSION_SECRET_KEY", "test-session-secret-for-testing-only")
os.environ.setdefault("MASTER_SECRET_KEY", "m" * 130)

import inspect

import pytest
from fastapi import HTTPException

from backend.api.routers import cache_management as cm
from backend.auth.oauth2 import get_current_admin_user


# ---------------------------------------------------------------------------
# #197 — pattern validation rejects unbounded / namespace-escaping patterns
# ---------------------------------------------------------------------------

UNSAFE_PATTERNS = [
    "*",            # flush entire keyspace
    "*:quote:*",    # leading wildcard, no namespace anchor
    "",             # empty -> no anchor
    "?*",           # leading single-char wildcard
    "[a-z]*",       # leading character-class wildcard
    "evil:*",       # anchored to a namespace that is not allowlisted
    "marketing:*",  # must not masquerade as the "market" namespace
]

SAFE_PATTERNS = [
    "quote:*",
    "market:*",
    "market:AAPL:*",
    "api:resp:*",
    "db:query:*",
    "stocks",
]


@pytest.mark.parametrize("pattern", UNSAFE_PATTERNS)
def test_unsafe_patterns_are_rejected(pattern):
    """Unbounded or namespace-escaping patterns must raise HTTP 400 (#197)."""
    with pytest.raises(HTTPException) as exc_info:
        cm.validate_invalidation_pattern(pattern)
    assert exc_info.value.status_code == 400


@pytest.mark.parametrize("pattern", SAFE_PATTERNS)
def test_safe_patterns_are_accepted(pattern):
    """Patterns anchored to an allowlisted namespace pass through unchanged."""
    assert cm.validate_invalidation_pattern(pattern) == pattern


def test_data_type_path_never_uses_leading_wildcard():
    """
    The data_type mapping must only produce namespace-anchored patterns so the
    SCAN can never widen beyond a single namespace (#197).
    """
    for data_type, prefix in cm.DATA_TYPE_TO_PREFIX.items():
        produced = f"{prefix}:*"
        assert not produced.startswith("*")
        # And the produced pattern itself passes validation.
        assert cm.validate_invalidation_pattern(produced) == produced


# ---------------------------------------------------------------------------
# #199 — destructive endpoints declare real admin auth and no placeholder
# ---------------------------------------------------------------------------

DESTRUCTIVE_PATHS = {"/invalidate", "/warm"}


def _routes_by_path():
    return {route.path: route for route in cm.router.routes}


@pytest.mark.parametrize("path", sorted(DESTRUCTIVE_PATHS))
def test_destructive_endpoint_requires_admin_auth(path):
    """
    /invalidate and /warm must declare get_current_admin_user as a dependency
    (#199). Asserted via router introspection rather than full-app wiring.
    """
    routes = _routes_by_path()
    assert path in routes, f"route {path} not registered"
    route = routes[path]

    dependant = route.dependant
    dependency_calls = {dep.call for dep in dependant.dependencies}
    assert get_current_admin_user in dependency_calls, (
        f"{path} does not require admin authentication"
    )


def test_no_commented_out_auth_placeholder_remains():
    """No 'uncomment for authentication' placeholder may remain (#199)."""
    source = inspect.getsource(cm)
    lowered = source.lower()
    assert "uncomment for authentication" not in lowered
    assert "# current_user" not in lowered
