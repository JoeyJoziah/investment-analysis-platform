"""Wave 5: password-reset TTL, DB SSL production, utcnow elimination."""
from __future__ import annotations

import inspect
import os
from pathlib import Path

os.environ.setdefault("SECRET_KEY", "test-secret-wave5")
os.environ.setdefault("JWT_SECRET_KEY", "test-jwt-wave5")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("TESTING", "True")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("MASTER_SECRET_KEY", "test-master-wave5")


def test_password_reset_token_expires_in_15_minutes():
    """#83: financial apps should use short-lived password-reset tokens."""
    from backend.security.security_config import SecurityConfig

    assert SecurityConfig.JWT_RESET_TOKEN_EXPIRE_MINUTES == 15


def test_production_db_connect_args_require_ssl():
    """#85: production asyncpg path forces ssl=require and rejects disable."""
    source = Path("backend/config/database.py").read_text(encoding="utf-8")
    assert 'connect_args["ssl"] = "require"' in source
    assert "sslmode=disable" in source
    assert "must not disable SSL in production" in source


def test_backend_has_no_datetime_utcnow():
    """#96: datetime.utcnow is deprecated; production code should be clean."""
    root = Path("backend")
    offenders = []
    for path in root.rglob("*.py"):
        if "venv" in path.parts or "__pycache__" in path.parts:
            continue
        # Skip this test file (it mentions utcnow by design) and pure fixtures
        if path.name == "test_wave5_hardening_contracts.py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        # Detect call sites only (not string mentions in docs/comments alone is hard;
        # require open-paren form used in real code).
        if "datetime.utcnow(" in text:
            offenders.append(str(path))
    assert offenders == [], f"utcnow still present in: {offenders[:10]}"


def test_wave4_security_contracts_still_importable():
    """Regression glue: Wave 4 contracts remain loadable after Wave 5."""
    from backend.security.csrf_protection import CSRFProtection

    source = inspect.getsource(CSRFProtection.validate_token)
    assert "compare_digest" in source
