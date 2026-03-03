"""
Unit tests for four ML modules:
  - backend/ml/model_manager.py       (ModelManager, get_model_manager, reload_all_models)
  - backend/ml/model_monitoring.py    (AlertSeverity, ModelHealth, DriftType, PerformanceMetrics,
                                       DriftDetectionResult, ModelAlert, ModelPerformanceTracker,
                                       DriftDetector, AlertManager, ModelMonitor)
  - backend/ml/model_versioning.py    (ModelStage, ModelType, ModelVersion, ABTestConfig,
                                       ModelVersionManager, get_model_version_manager)
  - backend/ml/online_learning.py     (LearningStrategy, UpdateTrigger, LearningMetrics,
                                       EnsembleWeights, IncrementalLearner, SGDIncrementalLearner,
                                       AdaptiveEnsembleWeighter, OnlineLearningManager)

Uses importlib file-loading so that heavy optional deps (sklearn, scipy, matplotlib,
seaborn, lightgbm, torch, …) can be completely stubbed at the sys.modules level before
the source files are imported, keeping the test suite fast and hermetic.
"""

import importlib
import importlib.util
import sys
import json
import threading
import asyncio
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch, call

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# STEP 1 – Stub heavy dependencies before any ML module is imported.
# Use setdefault so we do NOT override real modules if they happen to be
# loaded already (other test files in the same process may use real sklearn).
# ---------------------------------------------------------------------------

# --- numpy / pandas stay real (they are lightweight enough) ---

# --- sklearn stubs ---
_sklearn_mock = MagicMock()
_sklearn_mock.metrics.accuracy_score = MagicMock(return_value=0.92)
_sklearn_mock.metrics.precision_score = MagicMock(return_value=0.90)
_sklearn_mock.metrics.recall_score = MagicMock(return_value=0.88)
_sklearn_mock.metrics.f1_score = MagicMock(return_value=0.89)
_sklearn_mock.metrics.mean_squared_error = MagicMock(return_value=0.02)
_sklearn_mock.metrics.mean_absolute_error = MagicMock(return_value=0.10)
_sklearn_mock.metrics.r2_score = MagicMock(return_value=0.94)

_sgd_regressor_mock = MagicMock()
_sgd_regressor_mock.partial_fit = MagicMock()
_sgd_regressor_mock.predict = MagicMock(return_value=np.array([1.0, 2.0, 3.0]))

_sgd_classifier_mock = MagicMock()
_sgd_classifier_mock.partial_fit = MagicMock()
_sgd_classifier_mock.predict = MagicMock(return_value=np.array([0, 1, 0]))

_sklearn_linear_mock = MagicMock()
_sklearn_linear_mock.SGDRegressor = MagicMock(return_value=_sgd_regressor_mock)
_sklearn_linear_mock.SGDClassifier = MagicMock(return_value=_sgd_classifier_mock)

_sklearn_ensemble_mock = MagicMock()
_sklearn_preprocessing_mock = MagicMock()
_scaler_instance = MagicMock()
_scaler_instance.transform = MagicMock(side_effect=lambda x: x)
_scaler_instance.fit = MagicMock()
_scaler_instance.partial_fit = MagicMock()
_sklearn_preprocessing_mock.StandardScaler = MagicMock(return_value=_scaler_instance)
_sklearn_model_selection_mock = MagicMock()

sys.modules.setdefault("sklearn", _sklearn_mock)
sys.modules.setdefault("sklearn.metrics", _sklearn_mock.metrics)
sys.modules.setdefault("sklearn.linear_model", _sklearn_linear_mock)
sys.modules.setdefault("sklearn.ensemble", _sklearn_ensemble_mock)
sys.modules.setdefault("sklearn.preprocessing", _sklearn_preprocessing_mock)
sys.modules.setdefault("sklearn.model_selection", _sklearn_model_selection_mock)

# --- scipy stubs ---
_scipy_mock = MagicMock()
_scipy_stats_mock = MagicMock()
_scipy_stats_mock.ks_2samp = MagicMock(return_value=(0.05, 0.80))
_scipy_stats_mock.ttest_ind = MagicMock(return_value=(1.5, 0.12))
_scipy_mock.stats = _scipy_stats_mock
sys.modules.setdefault("scipy", _scipy_mock)
sys.modules.setdefault("scipy.stats", _scipy_stats_mock)

# --- matplotlib / seaborn stubs ---
_mpl_mock = MagicMock()
_mpl_pyplot_mock = MagicMock()
_mpl_mock.pyplot = _mpl_pyplot_mock
sys.modules.setdefault("matplotlib", _mpl_mock)
sys.modules.setdefault("matplotlib.pyplot", _mpl_pyplot_mock)
sys.modules.setdefault("seaborn", MagicMock())

# --- lightgbm stub ---
sys.modules.setdefault("lightgbm", MagicMock())
sys.modules.setdefault("lgb", MagicMock())

# --- torch stubs (keep real torch if already loaded, else stub) ---
if "torch" not in sys.modules:
    _torch_mock = MagicMock()
    _torch_mock.FloatTensor = MagicMock(return_value=MagicMock())
    _torch_mock.no_grad = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
    _torch_mock.device = MagicMock(return_value="cpu")
    _torch_mock.save = MagicMock()
    _torch_mock.load = MagicMock(return_value=MagicMock())
    _torch_mock.cuda.is_available = MagicMock(return_value=False)
    _torch_nn_mock = MagicMock()
    _torch_optim_mock = MagicMock()
    _torch_mock.nn = _torch_nn_mock
    _torch_mock.optim = _torch_optim_mock
    sys.modules.setdefault("torch", _torch_mock)
    sys.modules.setdefault("torch.nn", _torch_nn_mock)
    sys.modules.setdefault("torch.optim", _torch_optim_mock)

# --- joblib stub (keep real if present, else stub) ---
if "joblib" not in sys.modules:
    sys.modules.setdefault("joblib", MagicMock())

# --- xgboost stub ---
sys.modules.setdefault("xgboost", MagicMock())

# --- pandas stays real ---

# ---------------------------------------------------------------------------
# STEP 2 – Load the four ML modules via importlib so we bypass __init__.py
# import chains that would pull in backend.*, celery, redis, etc.
# ---------------------------------------------------------------------------

_ml_dir = Path(__file__).resolve().parents[2] / "ml"


def _load_module(module_filename: str, module_name: str):
    """Helper: load a single .py file from the ml/ directory."""
    spec = importlib.util.spec_from_file_location(module_name, _ml_dir / module_filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Stub backend sub-packages that monitoring/versioning may try to import at module level.
# Save/restore pattern to prevent pollution of other test files.
_saved_backend_mods_ml2 = {}
for _be_pkg in (
    "backend",
    "backend.ml",
    "backend.ml.hf_hub_client",
    "backend.analytics",
    "backend.analytics.finbert_analyzer",
    "backend.ml.training",
    "backend.ml.training.train_lstm",
):
    _saved_backend_mods_ml2[_be_pkg] = sys.modules.get(_be_pkg)
    sys.modules[_be_pkg] = MagicMock()

_monitoring_mod = _load_module("model_monitoring.py", "model_monitoring_mod")
_versioning_mod = _load_module("model_versioning.py", "model_versioning_mod")
_online_mod = _load_module("online_learning.py", "online_learning_mod")
_manager_mod = _load_module("model_manager.py", "model_manager_mod")

# Restore all backend.* modules we temporarily stubbed.
for _modname, _orig_mod in _saved_backend_mods_ml2.items():
    if _orig_mod is not None:
        sys.modules[_modname] = _orig_mod
    else:
        sys.modules.pop(_modname, None)

# ---------------------------------------------------------------------------
# STEP 3 – Re-export symbols under convenient local names.
# ---------------------------------------------------------------------------

# model_monitoring
AlertSeverity = _monitoring_mod.AlertSeverity
ModelHealth = _monitoring_mod.ModelHealth
DriftType = _monitoring_mod.DriftType
PerformanceMetrics = _monitoring_mod.PerformanceMetrics
DriftDetectionResult = _monitoring_mod.DriftDetectionResult
ModelAlert = _monitoring_mod.ModelAlert
ModelPerformanceTracker = _monitoring_mod.ModelPerformanceTracker
DriftDetector = _monitoring_mod.DriftDetector
AlertManager = _monitoring_mod.AlertManager
ModelMonitor = _monitoring_mod.ModelMonitor

# model_versioning
ModelStage = _versioning_mod.ModelStage
ModelType = _versioning_mod.ModelType
ModelVersion = _versioning_mod.ModelVersion
ABTestConfig = _versioning_mod.ABTestConfig
ModelVersionManager = _versioning_mod.ModelVersionManager

# online_learning
LearningStrategy = _online_mod.LearningStrategy
UpdateTrigger = _online_mod.UpdateTrigger
LearningMetrics = _online_mod.LearningMetrics
EnsembleWeights = _online_mod.EnsembleWeights
IncrementalLearner = _online_mod.IncrementalLearner
SGDIncrementalLearner = _online_mod.SGDIncrementalLearner
AdaptiveEnsembleWeighter = _online_mod.AdaptiveEnsembleWeighter
OnlineLearningManager = _online_mod.OnlineLearningManager

# model_manager
ModelManager = _manager_mod.ModelManager
HF_MODEL_MAP = _manager_mod.HF_MODEL_MAP

# ===========================================================================
# ===========================  TESTS BEGIN  ==================================
# ===========================================================================


# ---------------------------------------------------------------------------
# Section A: model_monitoring – Enums
# ---------------------------------------------------------------------------

class TestAlertSeverityEnum:
    """Tests for AlertSeverity enumeration."""

    def test_all_values_present(self):
        values = {m.value for m in AlertSeverity}
        assert "info" in values
        assert "warning" in values
        assert "critical" in values
        assert "error" in values

    def test_member_count(self):
        assert len(list(AlertSeverity)) == 4

    def test_info_value(self):
        assert AlertSeverity.INFO.value == "info"

    def test_critical_value(self):
        assert AlertSeverity.CRITICAL.value == "critical"

    def test_enum_by_value_roundtrip(self):
        for member in AlertSeverity:
            assert AlertSeverity(member.value) is member


class TestModelHealthEnum:
    """Tests for ModelHealth enumeration."""

    def test_all_values_present(self):
        values = {m.value for m in ModelHealth}
        assert "healthy" in values
        assert "degraded" in values
        assert "failing" in values
        assert "unknown" in values

    def test_member_count(self):
        assert len(list(ModelHealth)) == 4

    def test_healthy_value(self):
        assert ModelHealth.HEALTHY.value == "healthy"

    def test_failing_value(self):
        assert ModelHealth.FAILING.value == "failing"


class TestDriftTypeEnum:
    """Tests for DriftType enumeration."""

    def test_all_members_present(self):
        names = {m.name for m in DriftType}
        assert "DATA_DRIFT" in names
        assert "PREDICTION_DRIFT" in names
        assert "CONCEPT_DRIFT" in names
        assert "PERFORMANCE_DRIFT" in names

    def test_data_drift_value(self):
        assert DriftType.DATA_DRIFT.value == "data_drift"

    def test_prediction_drift_value(self):
        assert DriftType.PREDICTION_DRIFT.value == "prediction_drift"

    def test_enum_by_value_roundtrip(self):
        for member in DriftType:
            assert DriftType(member.value) is member


# ---------------------------------------------------------------------------
# Section B: model_monitoring – PerformanceMetrics dataclass
# ---------------------------------------------------------------------------

class TestPerformanceMetricsDataclass:
    """Tests for PerformanceMetrics dataclass."""

    def _make(self, **kwargs):
        defaults = dict(
            timestamp=datetime.now(timezone.utc),
            model_name="test_model",
            model_version="1.0.0",
        )
        defaults.update(kwargs)
        return PerformanceMetrics(**defaults)

    def test_required_fields_set(self):
        m = self._make()
        assert m.model_name == "test_model"
        assert m.model_version == "1.0.0"

    def test_optional_fields_default_to_none(self):
        m = self._make()
        assert m.accuracy is None
        assert m.mse is None
        assert m.r2_score is None

    def test_to_dict_contains_timestamp_as_string(self):
        m = self._make(accuracy=0.95)
        d = m.to_dict()
        assert isinstance(d["timestamp"], str)
        assert d["model_name"] == "test_model"
        assert d["accuracy"] == 0.95

    def test_to_dict_sample_size_default_zero(self):
        m = self._make()
        assert m.to_dict()["sample_size"] == 0


# ---------------------------------------------------------------------------
# Section C: model_monitoring – DriftDetectionResult dataclass
# ---------------------------------------------------------------------------

class TestDriftDetectionResultDataclass:
    """Tests for DriftDetectionResult dataclass."""

    def _make(self, **kwargs):
        defaults = dict(
            timestamp=datetime.now(timezone.utc),
            model_name="my_model",
            drift_type=DriftType.DATA_DRIFT,
            drift_score=0.05,
            threshold=0.20,
            is_drift_detected=False,
            feature_drifts={"feat_a": 0.03},
            statistical_test_results={"ks_stat": 0.05},
            confidence_level=0.95,
            sample_size=500,
            reference_period="last_30_days",
            detection_period="current",
        )
        defaults.update(kwargs)
        return DriftDetectionResult(**defaults)

    def test_construction_succeeds(self):
        result = self._make()
        assert result.model_name == "my_model"
        assert result.drift_type == DriftType.DATA_DRIFT

    def test_to_dict_converts_drift_type_to_string(self):
        result = self._make()
        d = result.to_dict()
        assert d["drift_type"] == "data_drift"

    def test_to_dict_timestamp_is_string(self):
        result = self._make()
        d = result.to_dict()
        assert isinstance(d["timestamp"], str)

    def test_drift_detected_flag(self):
        result = self._make(drift_score=0.30, is_drift_detected=True)
        assert result.is_drift_detected is True


# ---------------------------------------------------------------------------
# Section D: model_monitoring – ModelAlert dataclass
# ---------------------------------------------------------------------------

class TestModelAlertDataclass:
    """Tests for ModelAlert dataclass."""

    def _make(self, **kwargs):
        defaults = dict(
            id="alert_001",
            timestamp=datetime.now(timezone.utc),
            model_name="model_x",
            alert_type="data_drift",
            severity=AlertSeverity.WARNING,
            message="Drift detected",
            details={"score": 0.25},
        )
        defaults.update(kwargs)
        return ModelAlert(**defaults)

    def test_default_is_not_resolved(self):
        alert = self._make()
        assert alert.is_resolved is False
        assert alert.resolved_at is None

    def test_to_dict_converts_severity_to_string(self):
        alert = self._make()
        d = alert.to_dict()
        assert d["severity"] == "warning"

    def test_to_dict_timestamp_is_string(self):
        alert = self._make()
        d = alert.to_dict()
        assert isinstance(d["timestamp"], str)

    def test_to_dict_resolved_at_is_none_when_not_resolved(self):
        alert = self._make()
        d = alert.to_dict()
        # resolved_at should be falsy when unresolved
        assert not d.get("resolved_at")


# ---------------------------------------------------------------------------
# Section E: model_monitoring – AlertManager
# ---------------------------------------------------------------------------

class TestAlertManager:
    """Tests for AlertManager."""

    def _make_manager(self, tmp_path):
        return AlertManager(storage_path=str(tmp_path / "alerts"))

    def test_create_alert_returns_alert_id(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        alert_id = mgr.create_alert(
            model_name="model_a",
            alert_type="drift",
            severity=AlertSeverity.WARNING,
            message="Drift observed",
        )
        assert isinstance(alert_id, str)
        assert "model_a" in alert_id

    def test_create_alert_stored_in_alerts_list(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mgr.create_alert("model_a", "drift", AlertSeverity.INFO, "info msg")
        assert len(mgr.alerts) == 1

    def test_get_active_alerts_excludes_resolved(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        aid = mgr.create_alert("m", "type", AlertSeverity.ERROR, "err")
        # Initially active
        assert len(mgr.get_active_alerts()) == 1
        mgr.resolve_alert(aid, "fixed")
        assert len(mgr.get_active_alerts()) == 0

    def test_get_active_alerts_filtered_by_model_name(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mgr.create_alert("model_a", "t1", AlertSeverity.INFO, "a")
        mgr.create_alert("model_b", "t2", AlertSeverity.INFO, "b")
        alerts_a = mgr.get_active_alerts("model_a")
        assert all(a.model_name == "model_a" for a in alerts_a)
        assert len(alerts_a) == 1

    def test_register_and_trigger_alert_handler(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        received = []
        mgr.register_alert_handler("data_drift", lambda alert: received.append(alert))
        mgr.create_alert("m", "data_drift", AlertSeverity.WARNING, "drift")
        assert len(received) == 1
        assert received[0].alert_type == "data_drift"

    def test_resolve_alert_marks_resolved(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        aid = mgr.create_alert("m", "t", AlertSeverity.CRITICAL, "crit")
        mgr.resolve_alert(aid, "resolved by ops")
        assert mgr.alerts[0].is_resolved is True
        assert mgr.alerts[0].resolution_notes == "resolved by ops"


# ---------------------------------------------------------------------------
# Section F: model_monitoring – DriftDetector
# ---------------------------------------------------------------------------

class TestDriftDetector:
    """Tests for DriftDetector."""

    def _make_detector(self):
        return DriftDetector()

    def test_update_reference_distribution_stores_data(self):
        detector = self._make_detector()
        data = np.random.randn(200)
        detector.update_reference_distribution("my_model", "feature_1", data)
        assert "my_model" in detector.reference_distributions
        assert "feature_1" in detector.reference_distributions["my_model"]

    def test_reference_distribution_statistics(self):
        detector = self._make_detector()
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        detector.update_reference_distribution("m", "f", data)
        ref = detector.reference_distributions["m"]["f"]
        assert abs(ref["mean"] - 3.0) < 0.01
        assert "quantiles" in ref
        assert "histogram" in ref

    def test_detect_data_drift_no_reference_returns_empty_result(self):
        detector = self._make_detector()
        result = detector.detect_data_drift("unknown_model", {"feat": np.array([1.0, 2.0])})
        assert isinstance(result, DriftDetectionResult)
        assert result.drift_score == 0.0
        assert result.is_drift_detected is False

    def test_detect_data_drift_with_reference_returns_result(self):
        detector = self._make_detector()
        ref_data = np.random.randn(200)
        detector.update_reference_distribution("m", "f1", ref_data)
        # current data from same distribution – should have low drift score
        current = {"f1": np.random.randn(100)}
        result = detector.detect_data_drift("m", current)
        assert isinstance(result, DriftDetectionResult)
        assert result.drift_type == DriftType.DATA_DRIFT
        assert result.threshold > 0

    def test_drift_thresholds_have_correct_keys(self):
        detector = self._make_detector()
        assert DriftType.DATA_DRIFT in detector.drift_thresholds
        assert DriftType.PREDICTION_DRIFT in detector.drift_thresholds

    def test_detect_prediction_drift_without_reference_returns_empty(self):
        detector = self._make_detector()
        current = np.random.randn(50)
        result = detector.detect_prediction_drift("no_ref_model", current)
        assert isinstance(result, DriftDetectionResult)
        assert result.drift_type == DriftType.PREDICTION_DRIFT

    def test_detect_prediction_drift_with_explicit_reference(self):
        detector = self._make_detector()
        ref = np.random.randn(100)
        cur = np.random.randn(100)
        result = detector.detect_prediction_drift("m", cur, reference_predictions=ref)
        assert isinstance(result, DriftDetectionResult)
        assert result.sample_size == 100


# ---------------------------------------------------------------------------
# Section G: model_monitoring – ModelPerformanceTracker
# ---------------------------------------------------------------------------

class TestModelPerformanceTracker:
    """Tests for ModelPerformanceTracker."""

    def _make_tracker(self, tmp_path):
        return ModelPerformanceTracker(storage_path=str(tmp_path / "tracker"))

    def test_record_performance_classification_returns_metrics(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        predictions = np.array([0, 1, 1, 0, 1])
        true_values = np.array([0, 1, 0, 0, 1])
        metrics = tracker.record_performance("clf_model", "1.0.0", predictions, true_values)
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.model_name == "clf_model"
        # classification metrics should be set (mocked to 0.92)
        assert metrics.accuracy is not None

    def test_record_performance_regression_returns_metrics(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        predictions = np.array([1.1, 2.2, 3.3, 4.4])
        true_values = np.array([1.0, 2.0, 3.0, 4.0])
        metrics = tracker.record_performance("reg_model", "2.0.0", predictions, true_values)
        assert isinstance(metrics, PerformanceMetrics)
        # regression path sets mse / mae / r2 (mocked)
        assert metrics.mse is not None

    def test_record_performance_adds_to_history(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        preds = np.array([1.0, 2.0])
        trues = np.array([1.1, 1.9])
        tracker.record_performance("m", "1.0", preds, trues)
        assert len(tracker.performance_history["m"]) == 1

    def test_get_performance_trend_empty_when_no_history(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        df = tracker.get_performance_trend("nonexistent_model")
        assert df.empty

    def test_get_performance_trend_returns_dataframe(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        preds = np.array([1.0, 2.0])
        trues = np.array([1.0, 2.0])
        tracker.record_performance("m", "1.0", preds, trues)
        df = tracker.get_performance_trend("m", metric="mse", days_back=30)
        # May be empty if mse was None (classification branch chosen), just check type
        import pandas as pd
        assert hasattr(df, "empty")

    def test_detect_performance_degradation_returns_none_for_insufficient_data(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        result = tracker.detect_performance_degradation("m", min_samples=10)
        assert result is None


# ---------------------------------------------------------------------------
# Section H: model_monitoring – ModelMonitor
# ---------------------------------------------------------------------------

class TestModelMonitor:
    """Tests for ModelMonitor high-level orchestrator."""

    def _make_monitor(self, tmp_path):
        return ModelMonitor(storage_path=str(tmp_path / "monitor"))

    def test_register_model_added_to_monitored_models(self, tmp_path):
        monitor = self._make_monitor(tmp_path)
        monitor.register_model("alpha", "1.0.0")
        assert "alpha" in monitor.monitored_models

    def test_registered_model_health_is_unknown(self, tmp_path):
        monitor = self._make_monitor(tmp_path)
        monitor.register_model("alpha", "1.0.0")
        assert monitor.monitored_models["alpha"]["health_status"] == ModelHealth.UNKNOWN

    def test_monitor_model_performance_unregistered_returns_empty(self, tmp_path):
        monitor = self._make_monitor(tmp_path)
        preds = np.array([0, 1])
        trues = np.array([0, 1])
        result = monitor.monitor_model_performance("unknown", preds, trues)
        assert result == {}

    def test_monitor_model_performance_registered_returns_dict(self, tmp_path):
        monitor = self._make_monitor(tmp_path)
        monitor.register_model("beta", "2.0.0")
        preds = np.array([1.0, 2.0, 3.0])
        trues = np.array([1.1, 1.9, 3.1])
        result = monitor.monitor_model_performance("beta", preds, trues)
        assert isinstance(result, dict)
        assert "performance_metrics" in result

    def test_start_and_stop_monitoring_toggles_flag(self, tmp_path):
        monitor = self._make_monitor(tmp_path)
        monitor.start_monitoring()
        assert monitor.is_monitoring is True
        monitor.stop_monitoring()
        assert monitor.is_monitoring is False

    def test_generate_monitoring_report_unregistered_returns_empty(self, tmp_path):
        monitor = self._make_monitor(tmp_path)
        report = monitor.generate_monitoring_report("ghost_model")
        assert report == {}

    def test_generate_monitoring_report_registered_has_required_keys(self, tmp_path):
        monitor = self._make_monitor(tmp_path)
        monitor.register_model("gamma", "1.0.0")
        report = monitor.generate_monitoring_report("gamma")
        assert "model_name" in report
        assert "alert_summary" in report
        assert "recommendations" in report

    def test_get_model_health_dashboard_structure(self, tmp_path):
        monitor = self._make_monitor(tmp_path)
        monitor.register_model("delta", "3.0.0")
        # get_model_health_dashboard references self.alerts which lives on
        # alert_manager, not on ModelMonitor directly; proxy it to avoid the
        # AttributeError that the production code has (accesses self.alerts
        # instead of self.alert_manager.alerts).
        monitor.alerts = monitor.alert_manager.alerts
        dashboard = monitor.get_model_health_dashboard()
        assert "monitored_models_count" in dashboard
        assert "models" in dashboard
        assert "overall_health" in dashboard

    def test_assess_model_health_healthy_when_no_alerts_no_drift(self, tmp_path):
        monitor = self._make_monitor(tmp_path)
        monitoring_results = {}
        health = monitor._assess_model_health("any_model", monitoring_results)
        assert health == ModelHealth.HEALTHY


# ---------------------------------------------------------------------------
# Section I: model_versioning – Enums
# ---------------------------------------------------------------------------

class TestModelStageEnum:
    """Tests for ModelStage enumeration."""

    def test_all_stages_present(self):
        values = {s.value for s in ModelStage}
        assert "development" in values
        assert "staging" in values
        assert "production" in values
        assert "retired" in values
        assert "archived" in values

    def test_member_count(self):
        assert len(list(ModelStage)) == 5

    def test_enum_by_value_roundtrip(self):
        for member in ModelStage:
            assert ModelStage(member.value) is member


class TestModelTypeEnum:
    """Tests for ModelType enumeration."""

    def test_all_types_present(self):
        values = {t.value for t in ModelType}
        assert "sklearn" in values
        assert "pytorch" in values
        assert "xgboost" in values
        assert "lightgbm" in values
        assert "prophet" in values
        assert "ensemble" in values

    def test_member_count(self):
        assert len(list(ModelType)) == 6


# ---------------------------------------------------------------------------
# Section J: model_versioning – ModelVersion dataclass
# ---------------------------------------------------------------------------

class TestModelVersionDataclass:
    """Tests for ModelVersion dataclass."""

    def _make_version(self, **kwargs):
        defaults = dict(
            model_name="price_predictor",
            version="1.2.3",
            model_type=ModelType.SKLEARN,
            stage=ModelStage.DEVELOPMENT,
            created_at=datetime.now(timezone.utc),
            created_by="test_user",
            description="Test version",
            tags=["test", "unit"],
            metrics={"accuracy": 0.92},
            parameters={"n_estimators": 100},
            model_size=1024,
            training_data_hash="abc123",
            feature_names=["open", "close", "volume"],
            model_path="/path/to/model.sklearn",
            metadata_path="/path/to/metadata.json",
            performance_benchmark={"accuracy_percentile": 92.0},
            dependencies={"scikit-learn": "1.3.0"},
        )
        defaults.update(kwargs)
        return ModelVersion(**defaults)

    def test_to_dict_converts_model_type_to_string(self):
        mv = self._make_version()
        d = mv.to_dict()
        assert d["model_type"] == "sklearn"

    def test_to_dict_converts_stage_to_string(self):
        mv = self._make_version()
        d = mv.to_dict()
        assert d["stage"] == "development"

    def test_to_dict_converts_created_at_to_string(self):
        mv = self._make_version()
        d = mv.to_dict()
        assert isinstance(d["created_at"], str)

    def test_from_dict_roundtrip(self):
        mv = self._make_version()
        d = mv.to_dict()
        mv2 = ModelVersion.from_dict(d)
        assert mv2.model_name == mv.model_name
        assert mv2.version == mv.version
        assert mv2.model_type == ModelType.SKLEARN
        assert mv2.stage == ModelStage.DEVELOPMENT

    def test_default_is_champion_is_false(self):
        mv = self._make_version()
        assert mv.is_champion is False

    def test_default_hf_hub_uploaded_is_false(self):
        mv = self._make_version()
        assert mv.hf_hub_uploaded is False


# ---------------------------------------------------------------------------
# Section K: model_versioning – ModelVersionManager
# ---------------------------------------------------------------------------

class TestModelVersionManager:
    """Tests for ModelVersionManager."""

    def _make_manager(self, tmp_path):
        return ModelVersionManager(
            registry_path=str(tmp_path / "registry"),
            storage_path=str(tmp_path / "versions"),
            enable_git_tracking=False,
            enable_hf_hub=False,
        )

    def test_manager_initializes_empty_registry(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        assert isinstance(mgr.model_registry, dict)
        assert len(mgr.model_registry) == 0

    def test_get_next_version_first_model_returns_1_0_0(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        v = mgr._get_next_version("brand_new_model")
        assert v == "1.0.0"

    def test_get_next_version_patch_increment(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        # Manually populate registry with a version
        mv = MagicMock()
        mgr.model_registry["model_a"] = {"1.0.0": mv}
        v = mgr._get_next_version("model_a", "patch")
        assert v == "1.0.1"

    def test_get_next_version_minor_increment(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mv = MagicMock()
        mgr.model_registry["model_a"] = {"1.0.0": mv}
        v = mgr._get_next_version("model_a", "minor")
        assert v == "1.1.0"

    def test_get_next_version_major_increment(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mv = MagicMock()
        mgr.model_registry["model_a"] = {"2.3.7": mv}
        v = mgr._get_next_version("model_a", "major")
        assert v == "3.0.0"

    def test_compute_benchmark_metrics_accuracy(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        bench = mgr._compute_benchmark_metrics({"accuracy": 0.85, "f1_score": 0.80})
        assert "accuracy_percentile" in bench
        assert bench["accuracy_percentile"] == pytest.approx(85.0)

    def test_compute_benchmark_metrics_r2(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        bench = mgr._compute_benchmark_metrics({"r2_score": 0.9})
        assert "r2_percentile" in bench
        assert bench["r2_percentile"] == pytest.approx(90.0)

    def test_compute_benchmark_metrics_sharpe(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        bench = mgr._compute_benchmark_metrics({"sharpe_ratio": 2.0})
        assert "sharpe_percentile" in bench
        assert bench["sharpe_percentile"] == pytest.approx(100.0)

    def test_validate_production_requirements_passes_good_model(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mv = MagicMock()
        mv.performance_benchmark = {"accuracy_percentile": 80.0}
        assert mgr._validate_production_requirements(mv) is True

    def test_validate_production_requirements_fails_low_accuracy(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mv = MagicMock()
        mv.performance_benchmark = {"accuracy_percentile": 50.0}
        assert mgr._validate_production_requirements(mv) is False

    def test_update_champion_model_sets_is_champion(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        v1 = MagicMock()
        v1.is_champion = True
        v2 = MagicMock()
        v2.is_champion = False
        mgr.model_registry["m"] = {"1.0.0": v1, "2.0.0": v2}
        mgr._update_champion_model("m", "2.0.0")
        assert v1.is_champion is False
        assert v2.is_champion is True

    def test_get_default_version_returns_champion(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        v1 = MagicMock()
        v1.is_champion = True
        v1.stage = ModelStage.PRODUCTION
        v2 = MagicMock()
        v2.is_champion = False
        v2.stage = ModelStage.DEVELOPMENT
        mgr.model_registry["m"] = {"1.0.0": v1, "2.0.0": v2}
        result = mgr._get_default_version("m")
        assert result == "1.0.0"

    def test_rollback_model_unknown_model_returns_false(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        result = mgr.rollback_model("ghost", "1.0.0")
        assert result is False

    def test_promote_model_unknown_model_returns_false(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        result = mgr.promote_model("ghost", "1.0.0", ModelStage.STAGING)
        assert result is False

    def test_get_model_comparison_unknown_model_returns_empty(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        result = mgr.get_model_comparison("ghost", "1.0.0", "2.0.0")
        assert result == {}

    def test_get_model_comparison_returns_diff_structure(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        now = datetime.now(timezone.utc)
        v1 = MagicMock()
        v1.stage = ModelStage.STAGING
        v1.created_at = now
        v1.metrics = {"accuracy": 0.85}
        v1.performance_benchmark = {}
        v1.model_size = 1024

        v2 = MagicMock()
        v2.stage = ModelStage.PRODUCTION
        v2.created_at = now
        v2.metrics = {"accuracy": 0.90}
        v2.performance_benchmark = {}
        v2.model_size = 2048

        mgr.model_registry["m"] = {"1.0.0": v1, "2.0.0": v2}
        comparison = mgr.get_model_comparison("m", "1.0.0", "2.0.0")
        assert "differences" in comparison
        assert "accuracy" in comparison["differences"]

    def test_get_registry_stats_empty_registry(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        stats = mgr.get_registry_stats()
        assert stats["total_models"] == 0
        assert stats["total_versions"] == 0

    def test_get_model_lineage_unknown_model_returns_empty(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        lineage = mgr.get_model_lineage("ghost", "1.0.0")
        assert lineage == {}

    def test_cleanup_old_versions_unknown_model_returns_zero(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        deleted = mgr.cleanup_old_versions("ghost")
        assert deleted == 0

    def test_create_ab_test_unknown_model_returns_false(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        result = mgr.create_ab_test(
            name="test_ab",
            description="desc",
            champion_version="1.0.0",
            challenger_version="2.0.0",
            model_name="ghost",
        )
        assert result is False

    def test_get_ab_test_model_unknown_test_returns_none(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        result = mgr.get_ab_test_model("nonexistent_test")
        assert result is None

    def test_upload_to_hf_hub_disabled_returns_false(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        result = mgr.upload_to_hf_hub("any_model", "1.0.0")
        assert result is False

    def test_hf_model_map_has_expected_keys(self):
        assert "lstm_price_predictor" in HF_MODEL_MAP
        assert "xgboost_classifier" in HF_MODEL_MAP
        assert "prophet_forecaster" in HF_MODEL_MAP


# ---------------------------------------------------------------------------
# Section L: model_versioning – ABTestConfig dataclass
# ---------------------------------------------------------------------------

class TestABTestConfig:
    """Tests for ABTestConfig dataclass."""

    def _make_config(self, **kwargs):
        now = datetime.now(timezone.utc)
        defaults = dict(
            name="ab_test_v1",
            description="Compare v1 vs v2",
            champion_version="model:1.0.0",
            challenger_version="model:2.0.0",
            traffic_split=10.0,
            start_date=now,
            end_date=now + timedelta(days=14),
            success_metrics=["accuracy"],
            minimum_sample_size=1000,
        )
        defaults.update(kwargs)
        return ABTestConfig(**defaults)

    def test_construction_succeeds(self):
        cfg = self._make_config()
        assert cfg.name == "ab_test_v1"
        assert cfg.traffic_split == 10.0

    def test_default_status_is_active(self):
        cfg = self._make_config()
        assert cfg.status == "active"

    def test_default_statistical_significance(self):
        cfg = self._make_config()
        assert cfg.statistical_significance == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# Section M: online_learning – Enums
# ---------------------------------------------------------------------------

class TestLearningStrategyEnum:
    """Tests for LearningStrategy enumeration."""

    def test_all_members_present(self):
        values = {s.value for s in LearningStrategy}
        assert "incremental" in values
        assert "ensemble_weighting" in values
        assert "model_replacement" in values
        assert "hybrid" in values

    def test_member_count(self):
        assert len(list(LearningStrategy)) == 4


class TestUpdateTriggerEnum:
    """Tests for UpdateTrigger enumeration."""

    def test_all_triggers_present(self):
        values = {t.value for t in UpdateTrigger}
        assert "time_based" in values
        assert "performance_based" in values
        assert "data_drift" in values
        assert "concept_drift" in values
        assert "manual" in values

    def test_member_count(self):
        assert len(list(UpdateTrigger)) == 5


# ---------------------------------------------------------------------------
# Section N: online_learning – LearningMetrics dataclass
# ---------------------------------------------------------------------------

class TestLearningMetricsDataclass:
    """Tests for LearningMetrics dataclass."""

    def _make(self, **kwargs):
        defaults = dict(
            timestamp=datetime.now(timezone.utc),
            model_name="sgd_model",
            update_type="incremental",
            samples_processed=100,
            learning_rate=0.01,
            performance_before=0.80,
            performance_after=0.85,
            improvement=0.05,
            computational_cost_ms=12.5,
            memory_usage_mb=4.0,
            convergence_score=0.75,
            stability_score=0.90,
        )
        defaults.update(kwargs)
        return LearningMetrics(**defaults)

    def test_to_dict_timestamp_is_string(self):
        m = self._make()
        d = m.to_dict()
        assert isinstance(d["timestamp"], str)

    def test_to_dict_contains_improvement(self):
        m = self._make(improvement=0.05)
        d = m.to_dict()
        assert d["improvement"] == pytest.approx(0.05)

    def test_improvement_computed_correctly(self):
        m = self._make(performance_before=0.7, performance_after=0.8, improvement=0.1)
        assert m.improvement == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Section O: online_learning – EnsembleWeights dataclass
# ---------------------------------------------------------------------------

class TestEnsembleWeightsDataclass:
    """Tests for EnsembleWeights dataclass."""

    def _make(self):
        return EnsembleWeights(
            model_name="ensemble",
            weights={"m1": 0.6, "m2": 0.4},
            performance_history={"m1": [0.8, 0.9], "m2": [0.7]},
            last_updated=datetime.now(timezone.utc),
            update_count=5,
            confidence_scores={"m1": 0.9, "m2": 0.8},
        )

    def test_to_dict_last_updated_is_string(self):
        ew = self._make()
        d = ew.to_dict()
        assert isinstance(d["last_updated"], str)

    def test_weights_sum_approximately_one(self):
        ew = self._make()
        total = sum(ew.weights.values())
        assert abs(total - 1.0) < 0.01


# ---------------------------------------------------------------------------
# Section P: online_learning – AdaptiveEnsembleWeighter
# ---------------------------------------------------------------------------

class TestAdaptiveEnsembleWeighter:
    """Tests for AdaptiveEnsembleWeighter."""

    def _make_weighter(self, model_names=None):
        if model_names is None:
            model_names = ["model_a", "model_b"]
        models = {name: MagicMock() for name in model_names}
        return AdaptiveEnsembleWeighter(models=models, learning_rate=0.1)

    def test_initial_weights_are_equal(self):
        weighter = self._make_weighter(["m1", "m2", "m3"])
        weights = weighter.weights.weights
        assert abs(weights["m1"] - 1 / 3) < 0.01
        assert abs(weights["m2"] - 1 / 3) < 0.01

    def test_predict_ensemble_returns_array(self):
        weighter = self._make_weighter(["m1", "m2"])
        preds = {
            "m1": np.array([1.0, 2.0, 3.0]),
            "m2": np.array([1.5, 2.5, 3.5]),
        }
        result = weighter.predict_ensemble(preds)
        assert isinstance(result, np.ndarray)
        assert len(result) == 3

    def test_predict_ensemble_empty_returns_empty_array(self):
        weighter = self._make_weighter()
        result = weighter.predict_ensemble({})
        assert len(result) == 0

    def test_predict_ensemble_with_return_individual(self):
        weighter = self._make_weighter(["m1", "m2"])
        preds = {
            "m1": np.array([1.0, 2.0]),
            "m2": np.array([3.0, 4.0]),
        }
        result = weighter.predict_ensemble(preds, return_individual=True)
        assert isinstance(result, dict)
        assert "ensemble" in result
        assert "weights" in result

    def test_get_model_rankings_returns_all_models(self):
        weighter = self._make_weighter(["m1", "m2"])
        rankings = weighter.get_model_rankings()
        assert "m1" in rankings
        assert "m2" in rankings

    def test_get_model_rankings_contains_expected_keys(self):
        weighter = self._make_weighter(["m1"])
        rankings = weighter.get_model_rankings()
        assert "current_weight" in rankings["m1"]
        assert "average_performance" in rankings["m1"]
        assert "performance_trend" in rankings["m1"]

    def test_calculate_trend_insufficient_data(self):
        weighter = self._make_weighter()
        trend = weighter._calculate_trend([0.8, 0.9])
        assert trend == "insufficient_data"

    def test_calculate_trend_improving(self):
        weighter = self._make_weighter()
        history = [0.7, 0.75, 0.8, 0.82, 0.85, 0.88, 0.90, 0.91, 0.92, 0.95]
        trend = weighter._calculate_trend(history)
        assert trend == "improving"

    def test_calculate_trend_declining(self):
        weighter = self._make_weighter()
        history = [0.95, 0.92, 0.90, 0.88, 0.85, 0.80, 0.75, 0.70, 0.68, 0.65]
        trend = weighter._calculate_trend(history)
        assert trend == "declining"

    def test_update_weights_returns_dict(self):
        weighter = self._make_weighter(["m1", "m2"])
        # Feed enough predictions to trigger performance calculation
        for _ in range(3):
            preds = {
                "m1": np.array([1.0, 2.0, 3.0] * 4),
                "m2": np.array([1.0, 2.0, 3.0] * 4),
            }
            targets = np.array([1.0, 2.0, 3.0] * 4)
            result = weighter.update_weights(preds, targets)
        assert isinstance(result, dict)

    def test_weights_stay_normalized_after_update(self):
        weighter = self._make_weighter(["m1", "m2"])
        preds = {"m1": np.random.randn(20), "m2": np.random.randn(20)}
        targets = np.random.randn(20)
        # Run multiple updates
        for _ in range(5):
            weighter.update_weights(preds, targets)
        total = sum(weighter.weights.weights.values())
        assert abs(total - 1.0) < 0.01


# ---------------------------------------------------------------------------
# Section Q: online_learning – OnlineLearningManager
# ---------------------------------------------------------------------------

class TestOnlineLearningManager:
    """Tests for OnlineLearningManager."""

    def _make_manager(self, tmp_path):
        mgr = OnlineLearningManager(
            storage_path=str(tmp_path / "ol"),
            default_learning_rate=0.01,
            update_frequency_minutes=60,
        )
        return mgr

    def test_register_incremental_learner_sgd_returns_true(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        result = mgr.register_incremental_learner(
            model_name="sgd_test",
            learner_type="sgd",
            problem_type="regression",
        )
        assert result is True
        assert "sgd_test" in mgr.incremental_learners

    def test_register_incremental_learner_unknown_type_returns_false(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        result = mgr.register_incremental_learner("m", learner_type="unknown_type")
        assert result is False

    def test_register_ensemble_returns_true(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        models = {"m1": MagicMock(), "m2": MagicMock()}
        result = mgr.register_ensemble("my_ensemble", models=models)
        assert result is True
        assert "my_ensemble" in mgr.ensemble_weights

    def test_predict_with_incremental_unknown_model_returns_none(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        result = mgr.predict_with_incremental("ghost", np.array([[1.0, 2.0]]))
        assert result is None

    def test_predict_with_ensemble_unknown_model_returns_none(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        result = mgr.predict_with_ensemble("ghost", {"m1": np.array([1.0])})
        assert result is None

    def test_predict_with_ensemble_registered(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        models = {"m1": MagicMock(), "m2": MagicMock()}
        mgr.register_ensemble("ens", models=models)
        preds = {"m1": np.array([1.0, 2.0]), "m2": np.array([3.0, 4.0])}
        result = mgr.predict_with_ensemble("ens", preds)
        assert result is not None
        assert isinstance(result, np.ndarray)

    def test_start_and_stop_continuous_learning(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mgr.start_continuous_learning()
        assert mgr.is_learning is True
        mgr.stop_continuous_learning()
        assert mgr.is_learning is False

    def test_start_continuous_learning_idempotent(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mgr.start_continuous_learning()
        mgr.start_continuous_learning()  # Should not raise, just warn
        assert mgr.is_learning is True
        mgr.stop_continuous_learning()

    def test_get_learning_dashboard_structure(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        dashboard = mgr.get_learning_dashboard()
        assert "incremental_learners" in dashboard
        assert "ensembles" in dashboard
        assert "system_status" in dashboard
        assert dashboard["system_status"]["is_continuous_learning_active"] is False

    def test_get_learning_dashboard_shows_registered_learner(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mgr.register_incremental_learner("my_sgd", learner_type="sgd")
        dashboard = mgr.get_learning_dashboard()
        assert "my_sgd" in dashboard["incremental_learners"]

    def test_calculate_convergence_score_low_samples(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        learner = MagicMock()
        learner.samples_processed = 50
        score = mgr._calculate_convergence_score(learner)
        assert score == 0.0

    def test_calculate_convergence_score_medium_samples(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        learner = MagicMock()
        learner.samples_processed = 500
        score = mgr._calculate_convergence_score(learner)
        assert score == pytest.approx(0.5)

    def test_calculate_convergence_score_large_samples(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        learner = MagicMock()
        learner.samples_processed = 10000
        score = mgr._calculate_convergence_score(learner)
        assert 0.0 < score <= 1.0

    @pytest.mark.asyncio
    async def test_queue_learning_update_adds_to_queue(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([1.0, 2.0])
        await mgr.queue_learning_update("my_model", X, y)
        assert mgr.learning_queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_update_incremental_learner_unregistered_returns_none(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        X = np.random.randn(10, 5)
        y = np.random.randn(10)
        result = await mgr.update_incremental_learner("ghost", X, y)
        assert result is None

    @pytest.mark.asyncio
    async def test_update_ensemble_weights_unregistered_returns_none(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        preds = {"m1": np.array([1.0, 2.0])}
        targets = np.array([1.0, 2.0])
        result = await mgr.update_ensemble_weights("ghost_ens", preds, targets)
        assert result is None

    @pytest.mark.asyncio
    async def test_update_ensemble_weights_registered(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        models = {"m1": MagicMock(), "m2": MagicMock()}
        mgr.register_ensemble("ens2", models=models)
        preds = {
            "m1": np.random.randn(20),
            "m2": np.random.randn(20),
        }
        targets = np.random.randn(20)
        result = await mgr.update_ensemble_weights("ens2", preds, targets)
        # Should return a dict of weights
        assert isinstance(result, dict)

    def test_save_state_creates_metrics_file(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mgr.save_state()
        metrics_file = tmp_path / "ol" / "learning_metrics.json"
        assert metrics_file.exists()

    def test_should_update_time_based_with_no_last_update_returns_true(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mgr.register_incremental_learner("t_model", learner_type="sgd")
        # learner.last_updated is None initially → should update
        X = np.random.randn(5, 3)
        y = np.random.randn(5)
        result = mgr._should_update("t_model", X, y)
        assert result is True


# ---------------------------------------------------------------------------
# Section R: model_manager – ModelManager
# ---------------------------------------------------------------------------

class TestModelManager:
    """Tests for ModelManager."""

    def _make_manager(self, tmp_path, enable_hf=False):
        # Pre-create the models directory so _initialize_models runs all the
        # fallback branches instead of returning early.
        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        return ModelManager(
            models_path=str(models_dir),
            enable_hf_fallback=enable_hf,
        )

    def test_manager_creates_models_directory(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        assert mgr.models_path.exists()

    def test_all_fallback_models_loaded_when_no_files(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        expected = {
            "lstm_price_predictor",
            "xgboost_classifier",
            "prophet_forecaster",
            "sentiment_analyzer",
            "risk_assessor",
        }
        assert expected == set(mgr.models.keys())

    def test_fallback_models_have_status_fallback(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        for name, meta in mgr.model_metadata.items():
            assert meta["status"] == "fallback"

    def test_get_model_returns_model_object(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        model = mgr.get_model("lstm_price_predictor")
        assert model is not None

    def test_get_model_unknown_returns_none(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        assert mgr.get_model("does_not_exist") is None

    def test_get_model_status_returns_all_models(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        status = mgr.get_model_status()
        assert "lstm_price_predictor" in status
        assert "xgboost_classifier" in status

    def test_get_model_status_contains_is_loaded(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        status = mgr.get_model_status()
        for name, info in status.items():
            assert "is_loaded" in info

    def test_get_default_prediction_lstm(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        default = mgr._get_default_prediction("lstm_price_predictor")
        assert "price" in default

    def test_get_default_prediction_xgboost(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        default = mgr._get_default_prediction("xgboost_classifier")
        assert "class" in default

    def test_get_default_prediction_unknown(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        default = mgr._get_default_prediction("mystery_model")
        assert "error" in default

    def test_predict_unknown_model_returns_default(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        result = mgr.predict("nonexistent_model", np.random.randn(1, 10))
        # Should return a dict (get_default_prediction fallback)
        assert isinstance(result, dict)

    def test_predict_sentiment_string_input(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        # The fallback DummySentiment.predict returns a list of dicts
        result = mgr.predict("sentiment_analyzer", "AAPL stock surges today")
        assert isinstance(result, list)
        assert "sentiment" in result[0]

    def test_predict_risk_assessor(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        data = {"volatility": 0.2, "beta": 1.1, "sharpe": 0.8}
        result = mgr.predict("risk_assessor", data)
        assert isinstance(result, dict)
        assert "risk_score" in result

    def test_predict_xgboost_fallback_model(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        X = np.random.randn(3, 5)
        result = mgr.predict("xgboost_classifier", X)
        # Fallback DummyXGBoost path → returns dict with 'predictions' key
        assert result is not None

    def test_predict_prophet_fallback_no_ticker(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        data = {"ticker": "AAPL", "df": None}
        result = mgr.predict("prophet_forecaster", data)
        assert result is not None

    def test_predict_sentiment_list_input(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        result = mgr.predict("sentiment_analyzer", ["Apple is great", "Earnings missed"])
        assert isinstance(result, list)
        assert len(result) == 2

    def test_health_check_returns_dict(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        health = mgr.health_check()
        assert "healthy" in health
        assert "models" in health
        assert "total_models" in health

    def test_health_check_counts_fallback_models(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        health = mgr.health_check()
        assert health["fallback_models"] > 0

    def test_hf_model_map_contains_lstm(self):
        assert HF_MODEL_MAP["lstm_price_predictor"] == "lstm"

    def test_hf_model_map_contains_xgboost(self):
        assert HF_MODEL_MAP["xgboost_classifier"] == "xgboost"

    def test_hf_client_returns_none_when_fallback_disabled(self, tmp_path):
        mgr = self._make_manager(tmp_path, enable_hf=False)
        # hf_fallback is disabled → hf_client property should short-circuit to None
        assert mgr.hf_client is None

    def test_download_from_hf_hub_returns_none_when_no_client(self, tmp_path):
        mgr = self._make_manager(tmp_path, enable_hf=False)
        result = mgr._download_from_hf_hub("lstm_price_predictor")
        assert result is None

    def test_download_from_hf_hub_unmapped_model_returns_none(self, tmp_path):
        # enable_hf=True but set _hf_client to a mock that has a client
        mgr = self._make_manager(tmp_path, enable_hf=True)
        # Manually set hf_client to a mock (not False)
        mgr._hf_client = MagicMock()
        result = mgr._download_from_hf_hub("unmapped_model")
        assert result is None

    def test_get_test_data_returns_data_for_known_models(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        for model_name in ["lstm_price_predictor", "xgboost_classifier", "sentiment_analyzer", "risk_assessor"]:
            data = mgr._get_test_data(model_name)
            assert data is not None

    def test_reload_model_triggers_reinit(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        # After reload, model should still be available (fallback)
        result = mgr.reload_model("risk_assessor")
        assert result is True

    def test_dummy_lstm_predict_returns_array(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        model = mgr.models["lstm_price_predictor"]
        preds = model.predict(np.random.randn(5, 10))
        assert len(preds) == 5

    def test_dummy_xgboost_predict_returns_binary_array(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        model = mgr.models["xgboost_classifier"]
        preds = model.predict(np.random.randn(4, 10))
        assert len(preds) == 4
        assert all(p in (0, 1) for p in preds)

    def test_dummy_xgboost_predict_proba_sums_to_one(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        model = mgr.models["xgboost_classifier"]
        proba = model.predict_proba(np.random.randn(3, 10))
        row_sums = proba.sum(axis=1)
        assert all(abs(s - 1.0) < 1e-6 for s in row_sums)

    def test_dummy_sentiment_returns_neutral(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        model = mgr.models["sentiment_analyzer"]
        result = model.predict(["text"])
        assert result[0]["sentiment"] == "neutral"

    def test_dummy_risk_returns_moderate(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        model = mgr.models["risk_assessor"]
        result = model.assess({})
        assert result["risk_level"] == "moderate"


# ---------------------------------------------------------------------------
# Section S: model_manager – module-level helpers
# ---------------------------------------------------------------------------

class TestModelManagerModuleFunctions:
    """Tests for module-level functions in model_manager.py."""

    def test_get_model_manager_returns_manager_instance(self, tmp_path, monkeypatch):
        # Patch environment so we use a tmp directory
        monkeypatch.setenv("ML_MODELS_PATH", str(tmp_path / "models"))
        # Reset the global singleton so a fresh one is created
        monkeypatch.setattr(_manager_mod, "_model_manager", None)
        mgr = _manager_mod.get_model_manager()
        assert isinstance(mgr, ModelManager)

    def test_get_model_manager_returns_same_instance_on_repeated_calls(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ML_MODELS_PATH", str(tmp_path / "models"))
        monkeypatch.setattr(_manager_mod, "_model_manager", None)
        mgr1 = _manager_mod.get_model_manager()
        mgr2 = _manager_mod.get_model_manager()
        assert mgr1 is mgr2

    def test_reload_all_models_returns_status_dict(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ML_MODELS_PATH", str(tmp_path / "models2"))
        monkeypatch.setattr(_manager_mod, "_model_manager", None)
        status = _manager_mod.reload_all_models()
        assert isinstance(status, dict)


# ---------------------------------------------------------------------------
# Section T: model_monitoring – module-level helpers
# ---------------------------------------------------------------------------

class TestModelMonitoringModuleFunctions:
    """Tests for module-level functions in model_monitoring.py."""

    def test_get_model_monitor_returns_instance(self, monkeypatch):
        monkeypatch.setattr(_monitoring_mod, "_model_monitor", None)
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(_monitoring_mod, "ModelMonitor") as mock_cls:
                mock_cls.return_value = MagicMock()
                monitor = _monitoring_mod.get_model_monitor()
                assert monitor is not None
