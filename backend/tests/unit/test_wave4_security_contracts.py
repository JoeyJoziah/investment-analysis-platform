"""Wave 4: lock security contracts for #80–#84 (already implemented on main)."""
from __future__ import annotations

import inspect
import os

os.environ.setdefault("SECRET_KEY", "test-secret-wave4")
os.environ.setdefault("JWT_SECRET_KEY", "test-jwt-wave4")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("TESTING", "True")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("MASTER_SECRET_KEY", "test-master-wave4")


def test_csrf_validate_uses_compare_digest():
    """#84: constant-time CSRF validation."""
    from backend.security import csrf_protection as csrf

    source = inspect.getsource(csrf.CSRFProtection.validate_token)
    assert "hmac.compare_digest" in source


def test_csrf_production_requires_secret_key():
    """#80: production must not silently invent CSRF secrets."""
    from backend.security import csrf_protection as csrf

    source = inspect.getsource(csrf.CSRFConfig.__post_init__)
    from pathlib import Path

    main_src = Path("backend/api/main.py").read_text(encoding="utf-8")
    assert "CSRF_SECRET_KEY" in source
    assert "required in production" in source or "required in production" in main_src
    assert "is_production" in source or "production" in main_src.lower()


def test_refresh_endpoint_has_auth_rate_limit():
    """#81: /refresh must share auth rate limiting with login/token."""
    from backend.api.routers import auth as auth_mod

    source = inspect.getsource(auth_mod.refresh_token)
    assert "auth_rate_limit" in source


def test_portfolio_mutations_require_current_user():
    """#82: portfolio write routes depend on get_current_user."""
    from backend.api.routers import portfolio as port

    for name in ("add_position", "remove_position", "rebalance_portfolio"):
        fn = getattr(port, name)
        source = inspect.getsource(fn)
        assert "get_current_user" in source
    module_src = inspect.getsource(port)
    assert module_src.count("Depends(get_current_user)") >= 8
