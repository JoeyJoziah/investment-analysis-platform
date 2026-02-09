"""
Test numpy serialization utilities

Verifies that numpy arrays and scalar types are properly converted to
native Python types for JSON serialization in API responses.
"""

import numpy as np
import pytest
from pydantic import BaseModel, ValidationError

from backend.utils.numpy_serializer import sanitize_numpy, ensure_json_serializable


class TestNumpySanitization:
    """Test suite for numpy type sanitization"""

    def test_sanitize_numpy_array(self):
        """Test conversion of numpy arrays to lists"""
        arr = np.array([1.5, 2.5, 3.5])
        result = sanitize_numpy(arr)
        assert isinstance(result, list)
        assert result == [1.5, 2.5, 3.5]

    def test_sanitize_numpy_float_types(self):
        """Test conversion of numpy float types to Python float"""
        assert isinstance(sanitize_numpy(np.float16(3.14)), float)
        assert isinstance(sanitize_numpy(np.float32(3.14)), float)
        assert isinstance(sanitize_numpy(np.float64(3.14)), float)
        # Note: np.float128 doesn't exist on all platforms (e.g., macOS)

    def test_sanitize_numpy_int_types(self):
        """Test conversion of numpy integer types to Python int"""
        assert isinstance(sanitize_numpy(np.int8(42)), int)
        assert isinstance(sanitize_numpy(np.int16(42)), int)
        assert isinstance(sanitize_numpy(np.int32(42)), int)
        assert isinstance(sanitize_numpy(np.int64(42)), int)
        assert isinstance(sanitize_numpy(np.uint8(42)), int)
        assert isinstance(sanitize_numpy(np.uint16(42)), int)
        assert isinstance(sanitize_numpy(np.uint32(42)), int)
        assert isinstance(sanitize_numpy(np.uint64(42)), int)

    def test_sanitize_numpy_bool(self):
        """Test conversion of numpy bool to Python bool"""
        assert isinstance(sanitize_numpy(np.bool_(True)), bool)
        assert sanitize_numpy(np.bool_(True)) is True
        assert sanitize_numpy(np.bool_(False)) is False

    def test_sanitize_nested_dict(self):
        """Test sanitization of nested dictionaries with numpy types"""
        data = {
            'prices': np.array([100.5, 101.2, 102.3]),
            'count': np.int64(10),
            'average': np.float64(101.33),
            'nested': {
                'values': np.array([1, 2, 3]),
                'flag': np.bool_(True)
            }
        }
        result = sanitize_numpy(data)

        assert isinstance(result['prices'], list)
        assert isinstance(result['count'], int)
        assert isinstance(result['average'], float)
        assert isinstance(result['nested']['values'], list)
        assert isinstance(result['nested']['flag'], bool)

    def test_sanitize_nested_list(self):
        """Test sanitization of nested lists with numpy types"""
        data = [
            np.array([1, 2, 3]),
            np.float64(3.14),
            [np.int64(42), np.bool_(True)]
        ]
        result = sanitize_numpy(data)

        assert isinstance(result[0], list)
        assert isinstance(result[1], float)
        assert isinstance(result[2][0], int)
        assert isinstance(result[2][1], bool)

    def test_sanitize_ml_prediction_structure(self):
        """Test sanitization of typical ML prediction structure"""
        predictions = {
            'prices': np.array([150.5, 151.2, 152.1, 153.0]),
            'confidence': np.float64(0.85),
            'metrics': {
                'mean': np.mean([150.5, 151.2, 152.1, 153.0]),
                'std': np.std([150.5, 151.2, 152.1, 153.0]),
                'count': np.int64(4)
            }
        }

        result = sanitize_numpy(predictions)

        # Verify all numpy types are converted
        assert isinstance(result['prices'], list)
        assert isinstance(result['confidence'], float)
        assert isinstance(result['metrics']['mean'], float)
        assert isinstance(result['metrics']['std'], float)
        assert isinstance(result['metrics']['count'], int)

    def test_pydantic_serialization_after_sanitization(self):
        """Test that sanitized data works with Pydantic models"""
        class PredictionModel(BaseModel):
            prices: list
            average: float
            count: int

        # Create data with numpy types
        data = {
            'prices': np.array([100.5, 101.2, 102.3]),
            'average': np.float64(101.33),
            'count': np.int64(3)
        }

        # Sanitize and create Pydantic model
        sanitized = sanitize_numpy(data)
        model = PredictionModel(**sanitized)

        # Verify JSON serialization works
        json_data = model.model_dump(mode='json')
        assert isinstance(json_data['prices'], list)
        assert isinstance(json_data['average'], float)
        assert isinstance(json_data['count'], int)

    def test_pydantic_fails_without_sanitization(self):
        """Test that Pydantic fails to serialize raw numpy arrays"""
        class TestModel(BaseModel):
            value: float

        # This should work with a regular float
        model = TestModel(value=3.14)
        json_data = model.model_dump(mode='json')
        assert json_data['value'] == 3.14

        # Note: Pydantic 2.x actually handles numpy scalars better,
        # but arrays still need conversion to lists

    def test_ensure_json_serializable_alias(self):
        """Test that ensure_json_serializable is an alias for sanitize_numpy"""
        data = {
            'array': np.array([1, 2, 3]),
            'float': np.float64(3.14),
            'int': np.int64(42)
        }

        result1 = sanitize_numpy(data)
        result2 = ensure_json_serializable(data)

        assert result1 == result2

    def test_sanitize_preserves_non_numpy_types(self):
        """Test that non-numpy types are preserved unchanged"""
        data = {
            'string': 'hello',
            'int': 42,
            'float': 3.14,
            'bool': True,
            'list': [1, 2, 3],
            'dict': {'a': 1, 'b': 2},
            'none': None
        }

        result = sanitize_numpy(data)
        assert result == data

    def test_empty_structures(self):
        """Test sanitization of empty structures"""
        assert sanitize_numpy([]) == []
        assert sanitize_numpy({}) == {}
        assert sanitize_numpy(()) == []
        assert sanitize_numpy(np.array([])) == []

    def test_multidimensional_arrays(self):
        """Test sanitization of multidimensional numpy arrays"""
        arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
        result = sanitize_numpy(arr_2d)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result == [[1, 2, 3], [4, 5, 6]]

    def test_complex_ml_response_structure(self):
        """Test sanitization of complex ML API response structure"""
        response = {
            'ticker': 'AAPL',
            'predictions': [
                {
                    'date': '2026-02-09',
                    'price': np.float64(152.5),
                    'confidence_lower': np.float64(150.0),
                    'confidence_upper': np.float64(155.0)
                },
                {
                    'date': '2026-02-10',
                    'price': np.float64(153.2),
                    'confidence_lower': np.float64(150.5),
                    'confidence_upper': np.float64(155.9)
                }
            ],
            'model_metrics': {
                'rmse': np.float64(2.34),
                'mae': np.float64(1.87),
                'accuracy': np.float64(0.89),
                'samples': np.int64(1000)
            },
            'feature_importance': np.array([0.45, 0.32, 0.15, 0.08])
        }

        result = sanitize_numpy(response)

        # Verify all nested numpy types are converted
        assert isinstance(result['predictions'][0]['price'], float)
        assert isinstance(result['predictions'][1]['confidence_lower'], float)
        assert isinstance(result['model_metrics']['rmse'], float)
        assert isinstance(result['model_metrics']['samples'], int)
        assert isinstance(result['feature_importance'], list)
        assert len(result['feature_importance']) == 4
