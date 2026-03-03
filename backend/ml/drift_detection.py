"""
Drift detection for model inputs and outputs.

Extracted from model_monitoring.py.  Contains the DriftDetector class only.
Import via the original path (backend.ml.model_monitoring) or directly from here.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import numpy as np

try:
    from scipy import stats as _scipy_stats
except ImportError:  # pragma: no cover
    _scipy_stats = None  # type: ignore[assignment]

try:
    from backend.ml.monitoring_types import DriftType, DriftDetectionResult
except ImportError:  # pragma: no cover
    from monitoring_types import DriftType, DriftDetectionResult  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


class DriftDetector:
    """Advanced drift detection for model inputs and outputs"""

    def __init__(self) -> None:
        self.reference_distributions: Dict[str, Dict[str, Any]] = {}
        self.drift_thresholds = {
            DriftType.DATA_DRIFT: 0.2,
            DriftType.PREDICTION_DRIFT: 0.15,
            DriftType.CONCEPT_DRIFT: 0.25,
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update_reference_distribution(
        self,
        model_name: str,
        feature_name: str,
        reference_data: np.ndarray,
    ) -> None:
        """Update reference distribution for drift detection"""

        if model_name not in self.reference_distributions:
            self.reference_distributions[model_name] = {}

        self.reference_distributions[model_name][feature_name] = {
            'mean': np.mean(reference_data),
            'std': np.std(reference_data),
            'quantiles': np.percentile(reference_data, [5, 25, 50, 75, 95]),
            'histogram': np.histogram(reference_data, bins=50),
            'updated_at': datetime.now(timezone.utc),
        }

    def detect_data_drift(
        self,
        model_name: str,
        current_features: Dict[str, np.ndarray],
        confidence_level: float = 0.95,
    ) -> DriftDetectionResult:
        """Detect drift in input features"""

        feature_drifts: Dict[str, float] = {}
        statistical_tests: Dict[str, Any] = {}
        overall_drift_score = 0.0

        if model_name not in self.reference_distributions:
            logger.warning(f"No reference distribution for model {model_name}")
            return self._create_empty_drift_result(model_name, DriftType.DATA_DRIFT)

        reference_dist = self.reference_distributions[model_name]

        for feature_name, current_data in current_features.items():
            if feature_name not in reference_dist:
                continue

            ref_data = reference_dist[feature_name]
            drift_scores: Dict[str, Any] = {}

            # Kolmogorov-Smirnov test
            try:
                ref_sample = np.random.normal(
                    ref_data['mean'],
                    ref_data['std'],
                    size=len(current_data),
                )
                if _scipy_stats is not None:
                    ks_stat, ks_p_value = _scipy_stats.ks_2samp(ref_sample, current_data)
                else:
                    ks_stat, ks_p_value = 0.0, 1.0
                drift_scores['ks_statistic'] = ks_stat
                drift_scores['ks_p_value'] = ks_p_value
            except Exception as exc:
                logger.error(f"Error in KS test for {feature_name}: {exc}")
                drift_scores['ks_statistic'] = 0.0
                drift_scores['ks_p_value'] = 1.0

            # Population Stability Index
            try:
                psi_score = self._calculate_psi(ref_data, current_data)
                drift_scores['psi'] = psi_score
            except Exception as exc:
                logger.error(f"Error calculating PSI for {feature_name}: {exc}")
                drift_scores['psi'] = 0.0

            # Jensen-Shannon Distance
            try:
                js_distance = self._calculate_js_distance(ref_data, current_data)
                drift_scores['js_distance'] = js_distance
            except Exception as exc:
                logger.error(f"Error calculating JS distance for {feature_name}: {exc}")
                drift_scores['js_distance'] = 0.0

            feature_drift_score = (
                0.4 * drift_scores['psi']
                + 0.3 * drift_scores['ks_statistic']
                + 0.3 * drift_scores['js_distance']
            )

            feature_drifts[feature_name] = feature_drift_score
            statistical_tests[feature_name] = drift_scores
            overall_drift_score += feature_drift_score

        if feature_drifts:
            overall_drift_score /= len(feature_drifts)

        threshold = self.drift_thresholds[DriftType.DATA_DRIFT]
        is_drift_detected = overall_drift_score > threshold

        return DriftDetectionResult(
            timestamp=datetime.now(timezone.utc),
            model_name=model_name,
            drift_type=DriftType.DATA_DRIFT,
            drift_score=overall_drift_score,
            threshold=threshold,
            is_drift_detected=is_drift_detected,
            feature_drifts=feature_drifts,
            statistical_test_results=statistical_tests,
            confidence_level=confidence_level,
            sample_size=len(next(iter(current_features.values()))),
            reference_period="last_30_days",
            detection_period="current",
        )

    def detect_prediction_drift(
        self,
        model_name: str,
        current_predictions: np.ndarray,
        reference_predictions: Optional[np.ndarray] = None,
    ) -> DriftDetectionResult:
        """Detect drift in model predictions"""

        if reference_predictions is None:
            if (
                model_name in self.reference_distributions
                and 'predictions' in self.reference_distributions[model_name]
            ):
                ref_data = self.reference_distributions[model_name]['predictions']
                reference_predictions = np.random.normal(
                    ref_data['mean'],
                    ref_data['std'],
                    size=len(current_predictions),
                )
            else:
                return self._create_empty_drift_result(model_name, DriftType.PREDICTION_DRIFT)

        # Statistical tests
        if _scipy_stats is not None:
            ks_stat, ks_p_value = _scipy_stats.ks_2samp(reference_predictions, current_predictions)
        else:
            ks_stat, ks_p_value = 0.0, 1.0

        psi_score = self._calculate_psi_arrays(reference_predictions, current_predictions)
        js_distance = self._calculate_js_distance_arrays(reference_predictions, current_predictions)

        drift_score = 0.4 * psi_score + 0.3 * ks_stat + 0.3 * js_distance

        threshold = self.drift_thresholds[DriftType.PREDICTION_DRIFT]
        is_drift_detected = drift_score > threshold

        return DriftDetectionResult(
            timestamp=datetime.now(timezone.utc),
            model_name=model_name,
            drift_type=DriftType.PREDICTION_DRIFT,
            drift_score=drift_score,
            threshold=threshold,
            is_drift_detected=is_drift_detected,
            feature_drifts={'predictions': drift_score},
            statistical_test_results={
                'ks_statistic': ks_stat,
                'ks_p_value': ks_p_value,
                'psi': psi_score,
                'js_distance': js_distance,
            },
            confidence_level=0.95,
            sample_size=len(current_predictions),
            reference_period="reference",
            detection_period="current",
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _calculate_psi(self, ref_data: Dict[str, Any], current_data: np.ndarray) -> float:
        """Calculate Population Stability Index using stored reference data"""
        try:
            quantiles = ref_data['quantiles']
            bins = [-np.inf] + list(quantiles) + [np.inf]

            ref_hist, _ = ref_data['histogram']
            ref_freq = ref_hist / ref_hist.sum()

            cur_hist, _ = np.histogram(current_data, bins=bins[1:-1])
            cur_freq = cur_hist / cur_hist.sum() if cur_hist.sum() > 0 else cur_hist

            min_len = min(len(ref_freq), len(cur_freq))
            ref_freq = ref_freq[:min_len]
            cur_freq = cur_freq[:min_len]

            ref_freq = np.where(ref_freq == 0, 0.0001, ref_freq)
            cur_freq = np.where(cur_freq == 0, 0.0001, cur_freq)

            psi = np.sum((cur_freq - ref_freq) * np.log(cur_freq / ref_freq))
            return min(abs(psi), 1.0)
        except Exception as exc:
            logger.error(f"Error calculating PSI: {exc}")
            return 0.0

    def _calculate_psi_arrays(
        self, reference: np.ndarray, current: np.ndarray, bins: int = 10
    ) -> float:
        """Calculate PSI between two arrays"""
        try:
            combined_data = np.concatenate([reference, current])
            _, bin_edges = np.histogram(combined_data, bins=bins)

            ref_hist, _ = np.histogram(reference, bins=bin_edges)
            cur_hist, _ = np.histogram(current, bins=bin_edges)

            ref_freq = ref_hist / ref_hist.sum() if ref_hist.sum() > 0 else ref_hist
            cur_freq = cur_hist / cur_hist.sum() if cur_hist.sum() > 0 else cur_hist

            ref_freq = np.where(ref_freq == 0, 0.0001, ref_freq)
            cur_freq = np.where(cur_freq == 0, 0.0001, cur_freq)

            psi = np.sum((cur_freq - ref_freq) * np.log(cur_freq / ref_freq))
            return min(abs(psi), 1.0)
        except Exception as exc:
            logger.error(f"Error calculating PSI: {exc}")
            return 0.0

    def _calculate_js_distance(
        self, ref_data: Dict[str, Any], current_data: np.ndarray
    ) -> float:
        """Calculate Jensen-Shannon distance using stored reference data"""
        try:
            ref_hist, bin_edges = ref_data['histogram']
            cur_hist, _ = np.histogram(current_data, bins=bin_edges)

            ref_prob = ref_hist / ref_hist.sum() if ref_hist.sum() > 0 else ref_hist
            cur_prob = cur_hist / cur_hist.sum() if cur_hist.sum() > 0 else cur_hist

            m = 0.5 * (ref_prob + cur_prob)

            def kl_div(p: np.ndarray, q: np.ndarray) -> float:
                return float(np.sum(p * np.log(p / (q + 1e-10) + 1e-10)))

            js_distance = 0.5 * kl_div(ref_prob, m) + 0.5 * kl_div(cur_prob, m)
            return min(js_distance, 1.0)
        except Exception as exc:
            logger.error(f"Error calculating JS distance: {exc}")
            return 0.0

    def _calculate_js_distance_arrays(
        self, reference: np.ndarray, current: np.ndarray, bins: int = 50
    ) -> float:
        """Calculate Jensen-Shannon distance between two arrays"""
        try:
            min_val = min(reference.min(), current.min())
            max_val = max(reference.max(), reference.max())

            ref_hist, _ = np.histogram(reference, bins=bins, range=(min_val, max_val))
            cur_hist, _ = np.histogram(current, bins=bins, range=(min_val, max_val))

            ref_prob = ref_hist / ref_hist.sum() if ref_hist.sum() > 0 else ref_hist
            cur_prob = cur_hist / cur_hist.sum() if cur_hist.sum() > 0 else cur_hist

            m = 0.5 * (ref_prob + cur_prob)

            def kl_div(p: np.ndarray, q: np.ndarray) -> float:
                return float(np.sum(p * np.log(p / (q + 1e-10) + 1e-10)))

            js_distance = 0.5 * kl_div(ref_prob, m) + 0.5 * kl_div(cur_prob, m)
            return min(js_distance, 1.0)
        except Exception as exc:
            logger.error(f"Error calculating JS distance: {exc}")
            return 0.0

    def _create_empty_drift_result(
        self, model_name: str, drift_type: DriftType
    ) -> DriftDetectionResult:
        """Create empty drift result when no reference data available"""
        return DriftDetectionResult(
            timestamp=datetime.now(timezone.utc),
            model_name=model_name,
            drift_type=drift_type,
            drift_score=0.0,
            threshold=self.drift_thresholds[drift_type],
            is_drift_detected=False,
            feature_drifts={},
            statistical_test_results={},
            confidence_level=0.95,
            sample_size=0,
            reference_period="unavailable",
            detection_period="current",
        )


__all__ = ["DriftDetector"]
