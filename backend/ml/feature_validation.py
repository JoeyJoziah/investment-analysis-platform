"""
Feature Validation
FeatureValidator class for feature quality scoring and validation.
"""

import logging
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

try:
    from backend.ml.feature_types import FeatureDefinition, FeatureType
except ImportError:
    from feature_types import FeatureDefinition, FeatureType

logger = logging.getLogger(__name__)


class FeatureValidator:
    """Feature validation and quality scoring"""

    def __init__(self):
        self.validation_rules = {
            'numerical': self._validate_numerical,
            'categorical': self._validate_categorical,
            'boolean': self._validate_boolean,
            'datetime': self._validate_datetime,
        }

    def validate_feature(
        self, feature_def: FeatureDefinition, values: pd.Series
    ) -> Tuple[pd.Series, List[str]]:
        """Validate feature values and return quality scores"""
        errors = []
        quality_scores = pd.Series(1.0, index=values.index)

        # Type validation
        if feature_def.feature_type in self.validation_rules:
            type_valid, type_errors = self.validation_rules[feature_def.feature_type](values)
            errors.extend(type_errors)
            quality_scores = quality_scores * type_valid.astype(float)

        # Custom validation rules
        for rule_name, rule_config in feature_def.validation_rules.items():
            rule_valid, rule_errors = self._apply_validation_rule(
                rule_name, rule_config, values
            )
            errors.extend(rule_errors)
            quality_scores = quality_scores * rule_valid.astype(float)

        return quality_scores, errors

    def _validate_numerical(self, values: pd.Series) -> Tuple[pd.Series, List[str]]:
        """Validate numerical features"""
        errors = []
        valid = pd.Series(True, index=values.index)

        # Check for non-numeric values
        non_numeric = ~pd.to_numeric(values, errors='coerce').notna()
        if non_numeric.any():
            errors.append(f"Found {non_numeric.sum()} non-numeric values")
            valid = valid & ~non_numeric

        # Check for infinite values
        numeric_values = pd.to_numeric(values, errors='coerce')
        infinite_mask = np.isinf(numeric_values)
        if infinite_mask.any():
            errors.append(f"Found {infinite_mask.sum()} infinite values")
            valid = valid & ~infinite_mask

        return valid, errors

    def _validate_categorical(self, values: pd.Series) -> Tuple[pd.Series, List[str]]:
        """Validate categorical features"""
        errors = []
        valid = pd.Series(True, index=values.index)

        # Check for null values
        null_mask = values.isna()
        if null_mask.any():
            errors.append(f"Found {null_mask.sum()} null values")
            valid = valid & ~null_mask

        return valid, errors

    def _validate_boolean(self, values: pd.Series) -> Tuple[pd.Series, List[str]]:
        """Validate boolean features"""
        errors = []
        valid = pd.Series(True, index=values.index)

        boolean_values = values.isin([True, False, 0, 1, 'true', 'false', 'True', 'False'])
        if not boolean_values.all():
            errors.append(f"Found {(~boolean_values).sum()} invalid boolean values")
            valid = valid & boolean_values

        return valid, errors

    def _validate_datetime(self, values: pd.Series) -> Tuple[pd.Series, List[str]]:
        """Validate datetime features"""
        errors = []
        valid = pd.Series(True, index=values.index)

        try:
            pd.to_datetime(values)
        except Exception:
            errors.append("Invalid datetime format detected")
            valid = pd.Series(False, index=values.index)

        return valid, errors

    def _apply_validation_rule(
        self,
        rule_name: str,
        rule_config: Dict[str, Any],
        values: pd.Series,
    ) -> Tuple[pd.Series, List[str]]:
        """Apply custom validation rule"""
        errors = []
        valid = pd.Series(True, index=values.index)

        if rule_name == "range":
            min_val = rule_config.get('min')
            max_val = rule_config.get('max')

            if min_val is not None:
                below_min = values < min_val
                if below_min.any():
                    errors.append(f"Found {below_min.sum()} values below minimum {min_val}")
                    valid = valid & ~below_min

            if max_val is not None:
                above_max = values > max_val
                if above_max.any():
                    errors.append(f"Found {above_max.sum()} values above maximum {max_val}")
                    valid = valid & ~above_max

        elif rule_name == "allowed_values":
            allowed = rule_config.get('values', [])
            not_allowed = ~values.isin(allowed)
            if not_allowed.any():
                errors.append(f"Found {not_allowed.sum()} values not in allowed set")
                valid = valid & ~not_allowed

        elif rule_name == "not_null":
            if rule_config.get('required', True):
                null_mask = values.isna()
                if null_mask.any():
                    errors.append(f"Found {null_mask.sum()} null values")
                    valid = valid & ~null_mask

        return valid, errors
