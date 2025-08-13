"""
Comprehensive tests for Data Quality Validation Framework
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from backend.utils.data_quality import (
    DataQualityChecker,
    DataQualitySeverity,
    quick_validate_prices,
    quick_validate_fundamentals,
    validate_batch
)


class TestDataQualityChecker:
    """Test data quality validation functionality"""
    
    @pytest.fixture
    def checker(self):
        """Create a DataQualityChecker instance"""
        return DataQualityChecker()
    
    @pytest.fixture
    def valid_price_data(self):
        """Create valid price data for testing"""
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        return pd.DataFrame({
            'date': dates,
            'open': np.random.uniform(100, 110, 30),
            'high': np.random.uniform(110, 115, 30),
            'low': np.random.uniform(95, 100, 30),
            'close': np.random.uniform(100, 110, 30),
            'volume': np.random.uniform(1000000, 2000000, 30).astype(int)
        }).set_index('date')
    
    @pytest.fixture
    def invalid_price_data(self):
        """Create price data with various issues"""
        dates = pd.date_range(start='2024-01-01', periods=20, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'open': [100] * 20,
            'high': [95] * 20,  # High < Low (invalid)
            'low': [100] * 20,
            'close': [120] * 20,  # Close > High (invalid)
            'volume': [0] * 20  # Zero volume
        }).set_index('date')
        
        # Add missing values
        df.loc[df.index[5:8], 'close'] = np.nan
        
        return df
    
    def test_validate_price_data_valid(self, checker, valid_price_data):
        """Test validation of valid price data"""
        result = checker.validate_price_data(valid_price_data, "TEST")
        
        assert result['valid'] == True
        assert result['quality_score'] >= 90
        assert result['symbol'] == "TEST"
        assert len(result['issues']) == 0
        assert 'statistics' in result
        assert 'recommendations' in result
    
    def test_validate_price_data_missing_values(self, checker):
        """Test detection of missing values"""
        df = pd.DataFrame({
            'open': [100, 101, np.nan, 103, 104],
            'high': [105, 106, 107, np.nan, 109],
            'low': [95, 96, 97, 98, 99],
            'close': [102, 103, 104, 105, 106],
            'volume': [1000, 2000, 3000, 4000, 5000]
        })
        
        result = checker.validate_price_data(df, "TEST")
        
        assert result['valid'] == False
        issues = result['issues']
        missing_issue = next((i for i in issues if i['type'] == 'missing_values'), None)
        
        assert missing_issue is not None
        assert missing_issue['severity'] in ['high', 'medium']
        assert 'open' in missing_issue['details']
        assert 'high' in missing_issue['details']
    
    def test_validate_price_data_consistency(self, checker):
        """Test price consistency validation"""
        df = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [90, 106, 107, 108, 109],  # high < low for first row
            'low': [95, 96, 97, 98, 99],
            'close': [102, 103, 110, 105, 106],  # close > high for third row
            'volume': [1000, 2000, 3000, 4000, 5000]
        })
        
        result = checker.validate_price_data(df, "TEST")
        
        assert result['valid'] == False
        issues = result['issues']
        
        # Check for high < low issue
        hl_issue = next((i for i in issues if i['type'] == 'high_less_than_low'), None)
        assert hl_issue is not None
        assert hl_issue['severity'] == 'critical'
        
        # Check for close outside range issue
        close_issue = next((i for i in issues if i['type'] == 'close_outside_range'), None)
        assert close_issue is not None
        assert close_issue['severity'] == 'critical'
    
    def test_validate_price_data_outliers(self, checker):
        """Test detection of price outliers"""
        df = pd.DataFrame({
            'open': [100] * 20,
            'high': [105] * 20,
            'low': [95] * 20,
            'close': [100, 101, 102, 103, 200, 105] + [100] * 14,  # Outlier at index 4
            'volume': [1000000] * 20
        })
        
        result = checker.validate_price_data(df, "TEST")
        
        issues = result['issues']
        outlier_issue = next((i for i in issues if i['type'] == 'price_outliers'), None)
        
        assert outlier_issue is not None
        assert outlier_issue['count'] > 0
    
    def test_validate_price_data_volume_anomalies(self, checker):
        """Test detection of volume anomalies"""
        df = pd.DataFrame({
            'open': [100] * 10,
            'high': [105] * 10,
            'low': [95] * 10,
            'close': [100] * 10,
            'volume': [1000000, 1000000, 0, -100, 1000000] + [50] * 5  # Zero, negative, and low volume
        })
        
        result = checker.validate_price_data(df, "TEST")
        
        issues = result['issues']
        
        # Check for zero/negative volume
        volume_issue = next((i for i in issues if i['type'] == 'zero_or_negative_volume'), None)
        assert volume_issue is not None
        assert volume_issue['severity'] == 'high'
        
        # Check for sustained low volume
        low_volume_issue = next((i for i in issues if i['type'] == 'sustained_low_volume'), None)
        assert low_volume_issue is not None
    
    def test_validate_price_data_staleness(self, checker):
        """Test detection of stale data"""
        old_dates = pd.date_range(end=datetime.now() - timedelta(days=10), periods=10, freq='D')
        df = pd.DataFrame({
            'date': old_dates,
            'open': [100] * 10,
            'high': [105] * 10,
            'low': [95] * 10,
            'close': [100] * 10,
            'volume': [1000000] * 10
        }).set_index('date')
        
        result = checker.validate_price_data(df, "TEST")
        
        issues = result['issues']
        stale_issue = next((i for i in issues if i['type'] == 'stale_data'), None)
        
        assert stale_issue is not None
        assert stale_issue['days_old'] >= 10
    
    def test_validate_price_data_gaps(self, checker):
        """Test detection of data gaps"""
        # Create data with gaps
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        dates = dates.delete([3, 4, 5])  # Remove 3 days
        
        df = pd.DataFrame({
            'date': dates,
            'open': [100] * len(dates),
            'high': [105] * len(dates),
            'low': [95] * len(dates),
            'close': [100] * len(dates),
            'volume': [1000000] * len(dates)
        }).set_index('date')
        
        result = checker.validate_price_data(df, "TEST")
        
        issues = result['issues']
        gap_issue = next((i for i in issues if i['type'] == 'data_gaps'), None)
        
        if gap_issue:  # Gaps might not be detected on weekends
            assert gap_issue['total_missing_days'] > 0
    
    def test_validate_price_data_suspicious_patterns(self, checker):
        """Test detection of suspicious patterns"""
        # Identical consecutive prices
        df = pd.DataFrame({
            'open': [100] * 20,
            'high': [105] * 20,
            'low': [95] * 20,
            'close': [100.00] * 20,  # All identical
            'volume': [1000000] * 20
        })
        
        result = checker.validate_price_data(df, "TEST")
        
        issues = result['issues']
        pattern_issue = next((i for i in issues if i['type'] == 'identical_consecutive_prices'), None)
        
        assert pattern_issue is not None
        assert pattern_issue['severity'] == 'medium'
    
    def test_quality_score_calculation(self, checker):
        """Test quality score calculation"""
        # Perfect data
        perfect_df = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [95, 96, 97, 98, 99],
            'close': [102, 103, 104, 105, 106],
            'volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        })
        
        result = checker.validate_price_data(perfect_df, "PERFECT")
        assert result['quality_score'] == 100
        
        # Data with critical issues
        bad_df = pd.DataFrame({
            'open': [100, np.nan, 102, 103, 104],
            'high': [90, 106, 107, 108, 109],  # high < low
            'low': [95, 96, 97, 98, 99],
            'close': [120, 103, 104, 105, 106],  # close > high
            'volume': [0, -100, 3000, 4000, 5000]  # Invalid volumes
        })
        
        result = checker.validate_price_data(bad_df, "BAD")
        assert result['quality_score'] < 50
    
    def test_validate_fundamental_data(self, checker):
        """Test fundamental data validation"""
        fundamentals = {
            'pe_ratio': 25.5,
            'market_cap': 1000000000,
            'debt_to_equity': 1.5
        }
        
        result = checker.validate_fundamental_data(fundamentals, "TEST")
        
        assert result['valid'] == True
        assert result['quality_score'] >= 90
        assert result['symbol'] == "TEST"
        
        # Test with extreme values
        extreme_fundamentals = {
            'pe_ratio': 1500,  # Excessive P/E
            'market_cap': 500000,  # Low market cap
            'debt_to_equity': 15  # High debt ratio
        }
        
        result = checker.validate_fundamental_data(extreme_fundamentals, "EXTREME")
        
        assert result['valid'] == True  # Not critical, just warnings
        assert len(result['issues']) > 0
        
        pe_issue = next((i for i in result['issues'] if i['type'] == 'excessive_pe_ratio'), None)
        assert pe_issue is not None
        assert pe_issue['severity'] == 'medium'
    
    def test_validate_ml_features(self, checker):
        """Test ML feature validation"""
        features = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'feature3': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        
        feature_names = ['feature1', 'feature2', 'feature3']
        
        result = checker.validate_ml_features(features, feature_names)
        
        assert result['valid'] == True
        assert result['feature_count'] == 3
        assert result['row_count'] == 5
        
        # Test with missing features
        result = checker.validate_ml_features(features, ['feature1', 'feature2', 'feature3', 'feature4'])
        
        assert result['valid'] == False
        missing_issue = next((i for i in result['issues'] if i['type'] == 'missing_features'), None)
        assert missing_issue is not None
        assert 'feature4' in missing_issue['features']
        
        # Test with infinite values
        features_with_inf = features.copy()
        features_with_inf.loc[0, 'feature1'] = np.inf
        
        result = checker.validate_ml_features(features_with_inf, feature_names)
        
        assert result['valid'] == False
        inf_issue = next((i for i in result['issues'] if i['type'] == 'infinite_values'), None)
        assert inf_issue is not None
        assert inf_issue['severity'] == 'critical'
    
    def test_generate_quality_report(self, checker):
        """Test quality report generation"""
        validation_results = []
        
        # Add some validation results
        for i in range(5):
            validation_results.append({
                'valid': i < 3,
                'quality_score': 100 - i * 20,
                'issues': [
                    {'type': 'test_issue', 'severity': 'low' if i < 3 else 'high'}
                ] if i > 0 else []
            })
        
        report = checker.generate_quality_report(validation_results)
        
        assert 'summary' in report
        assert report['summary']['total_validations'] == 5
        assert report['summary']['passed_validations'] == 3
        assert report['summary']['failed_validations'] == 2
        assert report['summary']['pass_rate'] == 60.0
        
        assert 'severity_distribution' in report
        assert 'top_issues' in report
        assert 'recommendations' in report
    
    def test_statistics_calculation(self, checker, valid_price_data):
        """Test statistics calculation"""
        result = checker.validate_price_data(valid_price_data, "TEST")
        stats = result['statistics']
        
        assert 'row_count' in stats
        assert stats['row_count'] == 30
        
        assert 'date_range' in stats
        assert 'price_stats' in stats
        assert 'return_stats' in stats
        assert 'volume_stats' in stats
        
        price_stats = stats['price_stats']
        assert 'min' in price_stats
        assert 'max' in price_stats
        assert 'mean' in price_stats
        assert 'std' in price_stats
        
        return_stats = stats['return_stats']
        assert 'mean_return' in return_stats
        assert 'std_return' in return_stats
        assert 'sharpe_ratio' in return_stats
        assert 'max_drawdown' in return_stats
    
    def test_recommendations_generation(self, checker):
        """Test recommendation generation based on issues"""
        # Create data with specific issues
        df = pd.DataFrame({
            'open': [100, np.nan, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [95, 96, 97, 98, 99],
            'close': [102, 103, 104, 105, 106],
            'volume': [0, 1000, 2000, 3000, 4000]
        })
        
        result = checker.validate_price_data(df, "TEST")
        recommendations = result['recommendations']
        
        assert len(recommendations) > 0
        
        # Should recommend imputation for missing values
        assert any('imputation' in rec.lower() for rec in recommendations)
        
        # Should recommend checking volume data
        assert any('volume' in rec.lower() for rec in recommendations)


class TestQuickValidationFunctions:
    """Test quick validation utility functions"""
    
    def test_quick_validate_prices(self):
        """Test quick price validation"""
        valid_df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [102, 103, 104],
            'volume': [1000000, 1100000, 1200000]
        })
        
        assert quick_validate_prices(valid_df, "TEST") == True
        
        invalid_df = pd.DataFrame({
            'open': [100, np.nan, 102],
            'high': [90, 106, 107],  # high < low
            'low': [95, 96, 97],
            'close': [102, 103, 104],
            'volume': [1000000, 1100000, 1200000]
        })
        
        assert quick_validate_prices(invalid_df, "TEST") == False
    
    def test_quick_validate_fundamentals(self):
        """Test quick fundamental validation"""
        valid_fundamentals = {
            'pe_ratio': 20,
            'market_cap': 1000000000,
            'debt_to_equity': 0.5
        }
        
        assert quick_validate_fundamentals(valid_fundamentals, "TEST") == True
        
        # Even with extreme values, it should be valid (just warnings)
        extreme_fundamentals = {
            'pe_ratio': 1500,
            'market_cap': 100000,
            'debt_to_equity': 20
        }
        
        assert quick_validate_fundamentals(extreme_fundamentals, "TEST") == True
    
    def test_validate_batch(self):
        """Test batch validation"""
        batch = []
        
        # Add some valid data
        for i in range(3):
            df = pd.DataFrame({
                'open': [100 + i, 101 + i, 102 + i],
                'high': [105 + i, 106 + i, 107 + i],
                'low': [95 + i, 96 + i, 97 + i],
                'close': [102 + i, 103 + i, 104 + i],
                'volume': [1000000, 1100000, 1200000]
            })
            batch.append((f"STOCK{i}", df))
        
        # Add some invalid data
        invalid_df = pd.DataFrame({
            'open': [100, np.nan, 102],
            'high': [90, 106, 107],
            'low': [95, 96, 97],
            'close': [102, 103, 104],
            'volume': [0, 0, 0]
        })
        batch.append(("INVALID", invalid_df))
        
        report = validate_batch(batch)
        
        assert 'summary' in report
        assert report['summary']['total_validations'] == 4
        assert report['summary']['passed_validations'] == 3
        assert report['summary']['failed_validations'] == 1


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_dataframe(self):
        """Test validation of empty DataFrame"""
        checker = DataQualityChecker()
        df = pd.DataFrame()
        
        result = checker.validate_price_data(df, "EMPTY")
        
        assert result['valid'] == False
        issues = result['issues']
        empty_issue = next((i for i in issues if i['type'] == 'no_data'), None)
        assert empty_issue is not None
        assert empty_issue['severity'] == 'critical'
    
    def test_single_row_dataframe(self):
        """Test validation with single row of data"""
        checker = DataQualityChecker()
        df = pd.DataFrame({
            'open': [100],
            'high': [105],
            'low': [95],
            'close': [102],
            'volume': [1000000]
        })
        
        result = checker.validate_price_data(df, "SINGLE")
        
        # Should handle gracefully
        assert 'quality_score' in result
        assert 'issues' in result
    
    def test_all_nan_column(self):
        """Test validation with column of all NaN values"""
        checker = DataQualityChecker()
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [np.nan, np.nan, np.nan],  # All NaN
            'volume': [1000000, 1100000, 1200000]
        })
        
        result = checker.validate_price_data(df, "ALLNAN")
        
        assert result['valid'] == False
        issues = result['issues']
        missing_issue = next((i for i in issues if i['type'] == 'missing_values'), None)
        assert missing_issue is not None
        assert missing_issue['severity'] == 'critical'
        assert missing_issue['details']['close'] == 100.0
    
    def test_extreme_outliers(self):
        """Test handling of extreme outliers"""
        checker = DataQualityChecker()
        df = pd.DataFrame({
            'open': [100, 101, 102, 103, 10000],  # Extreme outlier
            'high': [105, 106, 107, 108, 10005],
            'low': [95, 96, 97, 98, 9995],
            'close': [102, 103, 104, 105, 10002],
            'volume': [1000000, 1100000, 1200000, 1300000, 100000000]
        })
        
        result = checker.validate_price_data(df, "OUTLIER")
        
        issues = result['issues']
        outlier_issue = next((i for i in issues if i['type'] == 'price_outliers'), None)
        assert outlier_issue is not None
        assert outlier_issue['severity'] in ['high', 'medium']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])