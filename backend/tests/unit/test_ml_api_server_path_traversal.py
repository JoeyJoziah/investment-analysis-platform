"""
Regression tests for the model path-traversal guard.

F-03-006 (audit 2026-04, G2a sub-theme A step 28):
``backend/ml/ml_api_server.py`` accepted an arbitrary ``model_name``
from the ``POST /models/{model_name}/load`` route and concatenated it
directly into a filesystem path. A caller could supply
``../../../etc/passwd`` (or any other traversal) and read host files
through the model-load surface. The fix enforces a strict character
class AND verifies the resolved path stays under the models root.

The module imports FastAPI / uvicorn which the audit env may not have.
We exercise the validator helpers directly via ``importlib.util``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


_API_PATH = (
    Path(__file__).resolve().parents[2]
    / "ml"
    / "ml_api_server.py"
)


def _load_module(monkeypatch: pytest.MonkeyPatch):
    """Load ml_api_server with FastAPI / uvicorn / joblib stubbed out."""

    for name in (
        "fastapi", "fastapi.middleware", "fastapi.middleware.cors",
        "fastapi.responses", "uvicorn", "joblib", "pydantic",
    ):
        if name not in sys.modules:
            stub = MagicMock()
            stub.__path__ = []
            sys.modules[name] = stub

    # FastAPI() returns an app; patch its decorators to be no-ops.
    fastapi_stub = sys.modules["fastapi"]
    fake_app = MagicMock()
    for method in ("get", "post", "put", "delete", "on_event"):
        getattr(fake_app, method).return_value = lambda f: f
    fastapi_stub.FastAPI = MagicMock(return_value=fake_app)
    fastapi_stub.HTTPException = type("HTTPException", (Exception,), {
        "__init__": lambda self, status_code, detail: setattr(self, "status_code", status_code) or setattr(self, "detail", detail)
    })
    fastapi_stub.BackgroundTasks = MagicMock()

    pydantic_stub = sys.modules["pydantic"]
    pydantic_stub.BaseModel = type("BaseModel", (), {})
    pydantic_stub.Field = lambda *a, **kw: None

    spec = importlib.util.spec_from_file_location(
        "ml_api_server_under_test", _API_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "bad_name",
    [
        "../../../etc/passwd",
        "model/../../../secret",
        "..",
        "/etc/passwd",
        "model with spaces",
        "model;rm -rf /",
        "model$(whoami)",
        "",
        "model.pkl",  # would build path with .pkl.pkl, but the dot is also invalid
    ],
)
def test_validate_rejects_path_traversal_attempts(
    monkeypatch: pytest.MonkeyPatch, bad_name: str
) -> None:
    """F-03-006: any non-[A-Za-z0-9_-]+ name must raise ValueError."""

    mod = _load_module(monkeypatch)
    with pytest.raises(ValueError, match="invalid model_name"):
        mod._validate_model_name(bad_name)


def test_validate_accepts_well_formed_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """F-03-006: legitimate names pass through unchanged."""

    mod = _load_module(monkeypatch)
    for ok in ("xgboost_v1", "lstm-30d", "abc", "Model_42"):
        assert mod._validate_model_name(ok) == ok


def test_safe_model_path_returns_path_under_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """F-03-006: ``_safe_model_path`` resolves under the configured root."""

    mod = _load_module(monkeypatch)
    p = mod._safe_model_path(mod._MODELS_ROOT, "good_name", ".pkl")
    # Must be relative to the configured root.
    p.relative_to(mod._MODELS_ROOT)


def test_load_model_rejects_traversal_before_filesystem_touch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F-03-006: load_model() raises before any joblib.load() call."""

    mod = _load_module(monkeypatch)
    # Patch joblib.load to a sentinel that would mark the test as
    # "validator did not run".
    called = {"value": False}

    def fail_joblib_load(_path):
        called["value"] = True
        raise AssertionError("joblib.load was reached on bad input")

    monkeypatch.setattr(mod.joblib, "load", fail_joblib_load)

    with pytest.raises(ValueError):
        mod.load_model("../../../etc/passwd")
    assert called["value"] is False
