"""F-03-005 fail-first regression tests for backend/ml/feature_store.py.

Per PRD audit 2026-04 Workstream D §4 / Q4 default (recorded 2026-04-28),
``FeatureStore.monitor_feature_drift`` and
``FeatureStore.get_feature_statistics`` MUST NOT fabricate values via
``np.random.normal`` / ``np.random.randint`` / ``np.random.uniform``.

When real feature data is unavailable they must raise
``InsufficientDataError`` so the API surfaces HTTP 503 ``model_unavailable``
instead of shipping random drift alerts and random feature counts to
SRE / monitoring dashboards.
"""
from __future__ import annotations

import os
import tempfile

import pytest


def _make_store():
    from backend.ml.feature_store import FeatureStore
    from backend.ml.feature_types import (
        ComputeMode,
        FeatureType,
    )

    tmp = tempfile.mkdtemp(prefix="featurestore-f03005-")
    store = FeatureStore(
        storage_path=tmp,
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
        db_url=None,
        enable_caching=False,
    )
    # Register a feature so the early-exit "feature not registered" branch
    # is not the reason any test fails.
    store.register_feature(
        name="rsi_14d",
        description="RSI 14-day window",
        feature_type=FeatureType.NUMERICAL,
        compute_mode=ComputeMode.STREAMING,
        computation_logic="ta.rsi(close, 14)",
    )
    return store


class TestMonitorFeatureDriftDeterminismF03005:
    """Drift score must be reproducible across calls or refuse to serve.

    Pre-fix, ``monitor_feature_drift`` calls ``np.random.normal(0,1,1000)``
    each invocation, so two consecutive calls produce different
    ``drift_score`` values. That is a silent random-alert bug.
    """

    def test_drift_score_is_reproducible_or_raises(self):
        from backend.exceptions import InsufficientDataError

        store = _make_store()
        try:
            m1 = store.monitor_feature_drift("rsi_14d")
            m2 = store.monitor_feature_drift("rsi_14d")
        except InsufficientDataError:
            return  # post-fix: refusing to fabricate is acceptable

        # If the function returned anything, the score must be reproducible
        # across calls with no underlying data change.
        assert m1 is not None and m2 is not None, (
            "F-03-005: monitor_feature_drift returned None silently — must "
            "raise InsufficientDataError instead"
        )
        assert m1.drift_score == m2.drift_score, (
            f"F-03-005: drift_score not reproducible "
            f"({m1.drift_score} != {m2.drift_score}) — feature_store is "
            "fabricating values via np.random.* instead of querying real DB"
        )


class TestGetFeatureStatisticsDeterminismF03005:
    """Feature statistics must not be sampled from random distributions."""

    def test_count_is_reproducible_or_raises(self):
        from backend.exceptions import InsufficientDataError

        store = _make_store()
        try:
            s1 = store.get_feature_statistics(["rsi_14d"])
            s2 = store.get_feature_statistics(["rsi_14d"])
        except InsufficientDataError:
            return  # post-fix: refusal is acceptable

        if not s1 or not s2 or "rsi_14d" not in s1 or "rsi_14d" not in s2:
            return  # nothing to compare

        assert s1["rsi_14d"]["count"] == s2["rsi_14d"]["count"], (
            "F-03-005: get_feature_statistics 'count' not reproducible — "
            "uses np.random.randint(1000, 10000) instead of real DB query"
        )
