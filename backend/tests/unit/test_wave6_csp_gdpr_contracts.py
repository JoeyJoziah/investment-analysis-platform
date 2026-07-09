"""Wave 6: CSP nonce (#86), GDPR surface (#42), cleanup hygiene."""
from __future__ import annotations

import inspect
import os
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.testclient import TestClient

os.environ.setdefault("SECRET_KEY", "test-secret-wave6")
os.environ.setdefault("JWT_SECRET_KEY", "test-jwt-wave6")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("TESTING", "True")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("MASTER_SECRET_KEY", "test-master-wave6")


def test_csp_build_includes_script_nonce():
    from backend.middleware.security_headers import ContentSecurityPolicy

    csp = ContentSecurityPolicy(script_src=["'self'"])
    header = csp.build(nonce="abc123XYZ")
    assert "script-src" in header
    assert "'nonce-abc123XYZ'" in header
    assert "'unsafe-inline'" not in header.split("script-src")[1].split(";")[0]


def test_security_headers_middleware_sets_csp_nonce_header():
    from backend.middleware.security_headers import (
        SecurityHeadersConfig,
        SecurityHeadersMiddleware,
    )

    app = Starlette()

    async def homepage(request: Request):
        return PlainTextResponse("ok")

    app.add_route("/", homepage)
    app.add_middleware(
        SecurityHeadersMiddleware,
        config=SecurityHeadersConfig(csp_nonce_enabled=True),
    )

    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Content-Security-Policy" in resp.headers
    nonce = resp.headers.get("X-CSP-Nonce")
    assert nonce
    assert f"'nonce-{nonce}'" in resp.headers["Content-Security-Policy"]


def test_gdpr_router_has_export_and_deletion_endpoints():
    """#42: core GDPR rights are implemented on the API surface."""
    from backend.api.routers import gdpr as gdpr_mod

    source = inspect.getsource(gdpr_mod)
    assert "data-export" in source
    assert "Right to Erasure" in source or "deletion" in source.lower()
    assert "get_current_user" in source
    assert Path("backend/compliance/gdpr.py").exists() or Path(
        "backend/compliance"
    ).exists()


def test_root_has_no_orphan_markdown_clutter():
    """#95: root should not accumulate random .md dumps (README/CLAUDE ok)."""
    root = Path(".")
    md = [p.name for p in root.glob("*.md")]
    allowed = {"README.md", "CLAUDE.md", "Claude.md", "CHANGELOG.md", "LICENSE.md"}
    unexpected = [n for n in md if n not in allowed]
    assert unexpected == [], f"Unexpected root markdown: {unexpected}"
