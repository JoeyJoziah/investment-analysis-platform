"""
Feature Drift Detection
FeatureDriftDetector class for monitoring feature distribution shifts.
"""

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    from backend.ml.feature_types import FeatureDriftMetrics
except ImportError:
    from feature_types import FeatureDriftMetrics

logger = logging.getLogger(__name__)


class FeatureDriftDetector:
    """Feature drift detection and monitoring"""

    def __init__(
        self,
        reference_window_days: int = 30,
        detection_window_days: int = 7,
    ):
        self.reference_window_days = reference_window_days
        self.detection_window_days = detection_window_days

    def detect_drift(
        self,
        feature_name: str,
        reference_data: pd.Series,
        current_data: pd.Series,
    ) -> FeatureDriftMetrics:
        """Detect drift between reference and current data"""

        psi = self._calculate_psi(reference_data, current_data)
        ks_stat = self._calculate_ks_statistic(reference_data, current_data)
        js_distance = self._calculate_js_distance(reference_data, current_data)

        mean_shift = (
            abs(current_data.mean() - reference_data.mean()) / reference_data.std()
            if reference_data.std() > 0
            else 0
        )
        std_shift = (
            abs(current_data.std() - reference_data.std()) / reference_data.std()
            if reference_data.std() > 0
            else 0
        )

        drift_score = (
            0.4 * psi
            + 0.3 * ks_stat
            + 0.2 * js_distance
            + 0.1 * (mean_shift + std_shift)
        )

        drift_threshold = 0.25
        distribution_shift_detected = drift_score > drift_threshold

        return FeatureDriftMetrics(
            feature_name=feature_name,
            timestamp=datetime.now(timezone.utc),
            population_stability_index=psi,
            kolmogorov_smirnov_statistic=ks_stat,
            jensen_shannon_distance=js_distance,
            mean_shift=mean_shift,
            std_shift=std_shift,
            distribution_shift_detected=distribution_shift_detected,
            drift_score=drift_score,
        )

    def _calculate_psi(
        self, reference: pd.Series, current: pd.Series, bins: int = 10
    ) -> float:
        """Calculate Population Stability Index"""
        try:
            if len(reference) == 0 or len(current) == 0:
                return 1.0

            _, bin_edges = np.histogram(reference.dropna(), bins=bins)
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf

            ref_freq = pd.cut(reference, bins=bin_edges).value_counts().values
            cur_freq = pd.cut(current, bins=bin_edges).value_counts().values

            ref_pct = ref_freq / ref_freq.sum()
            cur_pct = cur_freq / cur_freq.sum()

            ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
            cur_pct = np.where(cur_pct == 0, 0.0001, cur_pct)

            psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
            return min(psi, 1.0)

        except Exception as e:
            logger.error(f"Error calculating PSI: {e}")
            return 0.0

    def _calculate_ks_statistic(
        self, reference: pd.Series, current: pd.Series
    ) -> float:
        """Calculate Kolmogorov-Smirnov statistic"""
        try:
            from scipy import stats

            ref_clean = reference.dropna()
            cur_clean = current.dropna()

            if len(ref_clean) == 0 or len(cur_clean) == 0:
                return 0.0

            ks_stat, _ = stats.ks_2samp(ref_clean, cur_clean)
            return ks_stat

        except Exception as e:
            logger.error(f"Error calculating KS statistic: {e}")
            return 0.0

    def _calculate_js_distance(
        self, reference: pd.Series, current: pd.Series, bins: int = 50
    ) -> float:
        """Calculate Jensen-Shannon distance"""
        try:
            min_val = min(reference.min(), current.min())
            max_val = max(reference.max(), current.max())

            ref_hist, _ = np.histogram(
                reference.dropna(), bins=bins, range=(min_val, max_val), density=True
            )
            cur_hist, _ = np.histogram(
                current.dropna(), bins=bins, range=(min_val, max_val), density=True
            )

            ref_prob = ref_hist / ref_hist.sum() if ref_hist.sum() > 0 else ref_hist
            cur_prob = cur_hist / cur_hist.sum() if cur_hist.sum() > 0 else cur_hist

            m = 0.5 * (ref_prob + cur_prob)

            def kl_divergence(p, q):
                return np.sum(p * np.log(p / q + 1e-10))

            js_distance = (
                0.5 * kl_divergence(ref_prob, m)
                + 0.5 * kl_divergence(cur_prob, m)
            )
            return min(js_distance, 1.0)

        except Exception as e:
            logger.error(f"Error calculating JS distance: {e}")
            return 0.0
