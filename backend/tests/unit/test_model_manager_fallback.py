"""F-03-003 fail-first regression tests for backend/ml/model_manager.py.

Per PRD audit 2026-04 Workstream D §3 / Q4 default (recorded 2026-04-28),
when no trained model binaries are present in ``ml_models/`` the
``ModelManager`` MUST expose its fallback state so that:

1. ``GET /health`` surfaces ``fallback_models`` and ``fallback_models_count``.
2. The readiness probe fails when any production-critical model is in
   ``Dummy*`` fallback.
3. Endpoints producing investment outputs can call ``assert_real_model``
   to refuse-to-serve with HTTP 503 ``model_unavailable`` instead of
   shipping random.uniform / np.random.* fabrications to users.

These tests cover the model_manager API surface used by F-02-003 + F-03-003
fixes; they don't require a real training pipeline run — they exercise the
in-process ``DummyLSTM/DummyXGBoost/DummyProphet`` fallback path that is
already what production hits today.
"""
from __future__ import annotations

import os

import pytest


def _fresh_manager(tmp_path):
    # Force a brand-new manager pointing at an empty directory so every
    # registered model lands in fallback state.
    from backend.ml.model_manager import ModelManager
    return ModelManager(models_path=str(tmp_path), enable_hf_fallback=False)


class TestModelManagerFallbackDetectionF03003:
    def test_get_fallback_models_lists_dummy_models(self, tmp_path):
        mgr = _fresh_manager(tmp_path)
        fallback = mgr.get_fallback_models()
        # With no binaries in tmp_path, every registered model must be a
        # Dummy* fallback.
        assert "lstm_price_predictor" in fallback
        assert "xgboost_classifier" in fallback
        assert "prophet_forecaster" in fallback

    def test_is_using_fallback_true_when_binaries_missing(self, tmp_path):
        mgr = _fresh_manager(tmp_path)
        assert mgr.is_using_fallback("lstm_price_predictor") is True
        assert mgr.is_using_fallback("xgboost_classifier") is True

    def test_assert_real_model_raises_model_unavailable(self, tmp_path):
        from backend.exceptions import ModelUnavailableError

        mgr = _fresh_manager(tmp_path)
        with pytest.raises(ModelUnavailableError) as excinfo:
            mgr.assert_real_model("lstm_price_predictor")

        # The 503 payload depends on model + reason being populated correctly
        # (frontend G3-phase-4 contract).
        err = excinfo.value
        assert err.model == "lstm_price_predictor"
        assert err.reason in {"binary_missing", "fallback_active"}

    def test_assert_real_model_unknown_model_raises(self, tmp_path):
        """Unknown model names must also refuse-to-serve, not silently 200."""
        from backend.exceptions import ModelUnavailableError

        mgr = _fresh_manager(tmp_path)
        with pytest.raises(ModelUnavailableError):
            mgr.assert_real_model("not_a_registered_model_at_all")
