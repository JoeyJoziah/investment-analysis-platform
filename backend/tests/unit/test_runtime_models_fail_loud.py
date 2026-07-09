"""T1.1 fail-first regression tests for backend/ml/runtime_models.ModelManager.

Per PRD-LOKI-2026-06-STATE-REMEDIATION task T1.1 / decision D1: when no
pre-trained weight artifacts exist, the runtime ``ModelManager`` MUST NOT serve
a random-initialised LSTM/Transformer. It must refuse-to-serve by raising
``ModelUnavailableError`` (mapped to HTTP 503 ``model_unavailable``) rather than
returning a fabricated prediction.

These exercise the public surface only (no training pipeline run) and use
``asyncio.run`` so they don't depend on a pytest-asyncio event-loop fixture.
"""
from __future__ import annotations

import asyncio

import pandas as pd
import pytest


def _ohlcv(rows: int = 60) -> pd.DataFrame:
    return pd.DataFrame({c: [1.0] * rows for c in ("open", "high", "low", "close", "volume")})


def test_new_manager_starts_unloaded():
    from backend.ml.runtime_models import ModelManager

    mgr = ModelManager()
    # A freshly constructed manager has no real weights and must not be servable.
    assert mgr.weights_loaded is False


def test_load_models_without_weights_stays_unloaded():
    from backend.ml.runtime_models import ModelManager

    mgr = ModelManager()
    # No weight files on disk -> load must NOT flip weights_loaded to True
    # (the former bare `except:` silently kept random-init models here).
    asyncio.run(mgr.load_models())
    assert mgr.weights_loaded is False


def test_predict_refuses_to_serve_when_no_weights():
    from backend.exceptions import ModelUnavailableError
    from backend.ml.runtime_models import ModelManager

    mgr = ModelManager()
    assert mgr.weights_loaded is False

    with pytest.raises(ModelUnavailableError) as excinfo:
        asyncio.run(mgr.predict("AAPL", _ohlcv()))

    err = excinfo.value
    assert err.model == "runtime_ensemble"
    assert err.reason == "binary_missing"


def test_predict_guard_runs_before_feature_prep():
    """The refusal must short-circuit before any prediction work happens, so an
    empty/degenerate frame still raises ModelUnavailableError (never a value)."""
    from backend.exceptions import ModelUnavailableError
    from backend.ml.runtime_models import ModelManager

    mgr = ModelManager()
    with pytest.raises(ModelUnavailableError):
        asyncio.run(mgr.predict("AAPL", pd.DataFrame()))
