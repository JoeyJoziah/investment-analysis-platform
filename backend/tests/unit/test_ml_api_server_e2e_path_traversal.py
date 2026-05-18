"""
HTTP-level end-to-end test for F-03-006 (model_name path traversal).

The companion unit test ``test_ml_api_server_path_traversal.py`` covers the
validator helpers (``_validate_model_name``, ``_safe_model_path``) directly
with FastAPI stubbed out. This file complements it with a true E2E test
that drives the real ``app`` via ``fastapi.testclient.TestClient``,
exercising routing → validator → exception handler → status-code
mapping end-to-end. The acceptance criterion from the workpaper is:

    curl -X POST .../models/../../../etc/passwd/load → 400 (not 500)

Source-level grep tests catch the validator's existence; this test
proves the HTTP boundary actually returns 400 with a useful body
instead of leaking a 500 stack trace.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Project root for backend.* package resolution under --noconftest.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


@pytest.fixture(scope="module")
def client():
    """A real TestClient over the production FastAPI app.

    Defensively evicts ``sys.modules`` entries for fastapi /
    pydantic / backend.* / joblib / uvicorn before importing, so a
    sibling test that stubbed those (like
    ``test_ml_api_server_path_traversal.py`` which inserts
    ``MagicMock`` instances to test the validator helpers in
    isolation) doesn't leak its stubs into our real-FastAPI import.
    """
    for prefix in (
        "fastapi", "pydantic", "backend", "joblib", "uvicorn",
        "ml_api_server_under_test",
    ):
        for name in list(sys.modules):
            if name == prefix or name.startswith(prefix + "."):
                del sys.modules[name]

    from fastapi.testclient import TestClient
    from backend.ml.ml_api_server import app

    return TestClient(app)


# Inputs WITHOUT path separators that DO route to the load endpoint —
# these are rejected by the application-level validator with HTTP 400.
# (The route pattern ``/models/{model_name}/load`` only matches a single
# path segment, so slash-containing inputs never reach the handler;
# those are tested separately below as routing-layer rejections.)
VALIDATOR_REJECTED_INPUTS = [
    "model;rmrf",          # shell semicolon
    "model$(whoami)",      # command substitution
    "model`id`",           # backtick command sub
    "model with spaces",   # spaces
    "model.pkl",           # dot (rule is [A-Za-z0-9_-]+)
    "model@bad",           # @ symbol
    "model#hash",          # hash
]

# Inputs WITH path separators (raw or URL-encoded). These never reach
# the handler because Starlette path routing rejects them — the URL
# doesn't match the single-segment ``{model_name}`` pattern. Either way
# the security outcome is correct: no 500, no filesystem touch.
ROUTING_REJECTED_INPUTS = [
    "..",                           # bare dotdot — URL-normalized to /load (parent dir)
    "../../../etc/passwd",          # raw traversal
    "%2e%2e/%2e%2e/etc/passwd",     # URL-encoded traversal
    "model/with/slashes",           # embedded slashes
    "..%2F..%2Fetc%2Fpasswd",       # mixed encoding
]


@pytest.mark.parametrize("bad_name", VALIDATOR_REJECTED_INPUTS)
def test_validator_rejects_single_segment_inputs_with_http_400(
    client, bad_name: str
) -> None:
    """F-03-006: malformed single-segment ``model_name`` → HTTP 400 (not 500).

    These inputs all reach ``load_model_endpoint`` because they don't
    contain path separators. The application-level
    ``_validate_model_name`` regex (``[A-Za-z0-9_-]+``) rejects each,
    raising ValueError → mapped to HTTP 400 by the ``except ValueError``
    arm.
    """
    from urllib.parse import quote

    encoded = quote(bad_name, safe="")
    resp = client.post(f"/models/{encoded}/load")

    assert resp.status_code == 400, (
        f"input {bad_name!r} → HTTP {resp.status_code} "
        f"(body: {resp.text[:200]!r})"
    )

    body = resp.json()
    assert "detail" in body
    detail = body["detail"].lower()
    assert "model_name" in detail or "invalid" in detail, (
        f"response detail must mention the validation failure: {body!r}"
    )


@pytest.mark.parametrize("bad_name", ROUTING_REJECTED_INPUTS)
def test_routing_rejects_path_traversal_at_starlette_layer(
    client, bad_name: str
) -> None:
    """F-03-006: traversal inputs with separators never reach the handler.

    Starlette's single-segment ``{model_name}`` route pattern can't
    match a path with embedded slashes (raw or URL-encoded). The
    request is rejected with HTTP 404 BEFORE any handler runs — which
    is the strongest possible security outcome: no application code is
    invoked, no filesystem touch happens, no stack trace leaks.

    Critical assertion: response code is NOT 500 and the body does NOT
    leak a stack trace.
    """
    from urllib.parse import quote

    encoded = quote(bad_name, safe="")
    resp = client.post(f"/models/{encoded}/load")

    assert resp.status_code in (404, 400, 422), (
        f"traversal input {bad_name!r} produced unexpected HTTP "
        f"{resp.status_code} (body: {resp.text[:200]!r})"
    )
    assert resp.status_code != 500, (
        f"server returned 500 on traversal input {bad_name!r} — "
        f"FileNotFoundError or stack trace leaking?"
    )
    # Stack trace check: body should be plain JSON, not a traceback.
    body_text = resp.text
    for tb_marker in ("Traceback", "File \"", "line "):
        assert tb_marker not in body_text, (
            f"response body contains traceback marker {tb_marker!r}: "
            f"{body_text[:300]!r}"
        )


def test_load_model_endpoint_500_not_leaked_on_traversal(client) -> None:
    """F-03-006 anti-regression: traversal must NOT produce HTTP 500.

    Critical security property: the previous version leaked
    ``FileNotFoundError`` (HTTP 500) for any input, which both confirmed
    the traversal attempt was attempted on the filesystem AND included
    a stack trace in the response. Both signals are gone.
    """
    resp = client.post("/models/..%2F..%2F..%2Fetc%2Fpasswd/load")
    assert resp.status_code != 500, (
        f"traversal input produced HTTP 500 — server is still leaking "
        f"stack traces for malformed model names (body: {resp.text[:300]!r})"
    )


def test_load_model_endpoint_accepts_well_formed_name(client) -> None:
    """F-03-006 sanity: legitimate name passes validation, fails at filesystem.

    A name like ``test_model`` should clear ``_validate_model_name`` +
    ``_safe_model_path`` and reach ``FileNotFoundError`` → HTTP 404.
    This distinguishes "bad input" (400) from "no such model" (404).
    """
    resp = client.post("/models/test_model_does_not_exist/load")
    # Valid name → no traversal → FileNotFoundError → 404.
    assert resp.status_code == 404, (
        f"well-formed name should yield 404 (not found), got "
        f"HTTP {resp.status_code}: {resp.text[:200]!r}"
    )
