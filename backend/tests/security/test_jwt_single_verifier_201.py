"""Regression tests for audit finding #201.

Finding #201 — JWT: duplicate weak verifier + RS256/string-key confusion +
auto-generated secret fallbacks (Critical).

These tests assert the FIXED behaviour:

(a) ``backend/api/routers/auth.py`` no longer defines its own JWT decode/encode
    path. It must NOT call ``jwt.decode``/``jwt.encode`` directly and must NOT
    define a local ``get_current_user`` or ``create_access_token`` — instead it
    delegates to the canonical ``backend.auth.oauth2`` dependencies, which route
    through ``backend.security.jwt_manager`` (RS256 + RSA + blacklist + session +
    issuer/audience checks).

(b) ``backend/security/security_config.py`` FAILS FAST when ``JWT_SECRET_KEY`` /
    ``SESSION_SECRET_KEY`` are unset in production (raising on startup) instead of
    silently generating an ephemeral ``secrets.token_urlsafe`` value, while still
    permitting a safe development default outside production.

Implementation note: assertions are performed at the source/AST level so the
tests do not require importing the full FastAPI app (Redis/DB/middleware), which
would be heavy and environment-dependent. The production fail-fast behaviour is
exercised by importing the real ``security_config`` module under a monkeypatched
environment via ``importlib``.
"""

import ast
import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
AUTH_ROUTER_PATH = REPO_ROOT / "backend" / "api" / "routers" / "auth.py"
SECURITY_CONFIG_PATH = REPO_ROOT / "backend" / "security" / "security_config.py"


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


# ---------------------------------------------------------------------------
# (a) auth.py delegates to the single verification entry point
# ---------------------------------------------------------------------------


def test_auth_router_does_not_define_local_get_current_user():
    """auth.py must NOT define its own get_current_user (it had a weak one)."""
    tree = _parse(AUTH_ROUTER_PATH)
    local_defs = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "get_current_user" not in local_defs, (
        "auth.py must not define a local get_current_user; it must re-export the "
        "canonical dependency from backend.auth.oauth2."
    )


def test_auth_router_does_not_define_local_create_access_token():
    """auth.py must NOT define its own token minting (string-key/RS256 confusion)."""
    tree = _parse(AUTH_ROUTER_PATH)
    local_defs = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "create_access_token" not in local_defs, (
        "auth.py must not define a local create_access_token; token issuance must "
        "flow through backend.auth.oauth2 / jwt_manager."
    )


def test_auth_router_has_no_direct_jwt_decode_or_encode():
    """auth.py must not call jwt.decode / jwt.encode directly anywhere."""
    tree = _parse(AUTH_ROUTER_PATH)
    offending = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"decode", "encode"} and isinstance(
                node.func.value, ast.Name
            ) and node.func.value.id == "jwt":
                offending.append(node.func.attr)
    assert not offending, (
        f"auth.py must not call jwt.{{decode,encode}} directly; found: {offending}. "
        "All verification/issuance must route through jwt_manager."
    )


def test_auth_router_does_not_import_jose_or_low_level_jwt():
    """The weak path imported `from jose import JWTError, jwt`; it must be gone."""
    source = AUTH_ROUTER_PATH.read_text(encoding="utf-8")
    assert "from jose import" not in source, (
        "auth.py must not import the low-level jose jwt primitives used by the "
        "removed weak verifier."
    )


def test_auth_router_reexports_canonical_dependencies():
    """auth.py must import the canonical deps so dependents keep working."""
    tree = _parse(AUTH_ROUTER_PATH)
    imported_from_oauth2 = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "backend.auth.oauth2":
            imported_from_oauth2.update(alias.name for alias in node.names)

    for required in ("get_current_user", "get_current_admin_user", "create_tokens"):
        assert required in imported_from_oauth2, (
            f"auth.py must re-export `{required}` from backend.auth.oauth2 "
            f"(single verification entry point). Imported: {imported_from_oauth2}"
        )


# ---------------------------------------------------------------------------
# (b) security_config.py fails fast in production for missing secrets
# ---------------------------------------------------------------------------


def _reimport_security_config():
    """Force a fresh import of security_config so module-level secret resolution
    re-runs under the current (monkeypatched) environment."""
    for mod_name in list(sys.modules):
        if mod_name == "backend.security.security_config":
            del sys.modules[mod_name]
    return importlib.import_module("backend.security.security_config")


@pytest.mark.parametrize("missing_var", ["JWT_SECRET_KEY", "SESSION_SECRET_KEY"])
def test_security_config_fails_fast_in_production_when_secret_unset(
    monkeypatch, missing_var
):
    """Production boot must FAIL when a required secret env var is unset."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    # Ensure BOTH are unset so resolution reaches the offending one regardless of
    # class-body evaluation order; the first unset one encountered raises.
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    monkeypatch.delenv("SESSION_SECRET_KEY", raising=False)

    with pytest.raises(Exception) as exc_info:
        _reimport_security_config()

    # The raised error should be the dedicated fail-fast type for a secret.
    assert "production" in str(exc_info.value).lower()
    assert exc_info.type.__name__ in {"InsecureSecretError", "RuntimeError"}


def test_security_config_allows_dev_default_when_not_production(monkeypatch):
    """Outside production a safe, stable dev default is permitted (no raise)."""
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    monkeypatch.delenv("SESSION_SECRET_KEY", raising=False)

    module = _reimport_security_config()
    cfg = module.SecurityConfig

    # Defaults must be present, non-empty, and clearly marked as dev-only.
    assert cfg.JWT_SECRET_KEY
    assert cfg.SESSION_SECRET_KEY
    assert "production" in cfg.JWT_SECRET_KEY.lower()
    assert "production" in cfg.SESSION_SECRET_KEY.lower()


def test_security_config_uses_provided_secret_in_production(monkeypatch):
    """When secrets ARE provided in production, import must succeed and use them."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("JWT_SECRET_KEY", "real-prod-jwt-secret-value")
    monkeypatch.setenv("SESSION_SECRET_KEY", "real-prod-session-secret-value")

    module = _reimport_security_config()
    cfg = module.SecurityConfig

    assert cfg.JWT_SECRET_KEY == "real-prod-jwt-secret-value"
    assert cfg.SESSION_SECRET_KEY == "real-prod-session-secret-value"


@pytest.fixture(autouse=True)
def _restore_security_config_module():
    """Re-import security_config with a clean dev env after each test so we leave
    the module registry in a consistent, importable state for other tests."""
    yield
    import os

    os.environ["ENVIRONMENT"] = "development"
    os.environ.setdefault("JWT_SECRET_KEY", "dev-test-jwt-secret")
    os.environ.setdefault("SESSION_SECRET_KEY", "dev-test-session-secret")
    try:
        _reimport_security_config()
    except Exception:
        # Never let teardown mask a test failure.
        pass
