"""
Data quality validation framework for ensuring data integrity and reliability.
Implements comprehensive checks for price data, fundamentals, and ML features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from scipy import stats
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class DataQualitySeverity(Enum):
    """Severity levels for data quality issues."""
    CRITICAL = "critical"  # Data unusable, must fix
    HIGH = "high"         # Major issues, should fix
    MEDIUM = "medium"     # Notable issues, investigate
    LOW = "low"          # Minor issues, monitor
    INFO = "info"        # Informational only


class DataQualityChecker:
    """
    Comprehensive data quality validation system.
    Performs checks on price data, fundamentals, and ML features.
    """
    
    def __init__(self):
        self.validation_rules = self._initialize_rules()
        self.anomaly_detector = IsolationForest(
            contamination=0.01,
            random_state=42
        )
        self.quality_scores = {}
        self.issue_history = []
        
    def _initialize_rules(self) -> Dict:
        """Initialize validation rules for different data types."""
        return {
            'price_data': {
                'max_daily_change': 0.5,  # 50% max daily move
                'min_volume': 100,         # Minimum volume threshold
                'max_spread_ratio': 0.1,   # Max bid-ask spread ratio
                'stale_data_days': 1,      # Days before data considered stale
            },
            'fundamental_data': {
                'max_pe_ratio': 1000,       # Maximum acceptable P/E
                'min_market_cap': 1000000,  # $1M minimum market cap
                'max_debt_ratio': 10,       # Maximum debt/equity ratio
            },
            'technical_indicators': {
                'rsi_range': (0, 100),      # RSI must be 0-100
                'volume_outlier_std': 3,    # Volume outlier threshold
            }
        }
    
    def validate_price_data(
        self, 
        df: pd.DataFrame,
        symbol: str = "UNKNOWN"
    ) -> Dict[str, Any]:
        """
        Validate price data quality with comprehensive checks.
        
        Args:
            df: DataFrame with columns [date, open, high, low, close, volume]
            symbol: Stock symbol for logging
            
        Returns:
            Dictionary with validation results and quality score
        """
        issues = []
        
        # 1. Check for missing values
        missing_check = self._check_missing_values(df)
        if missing_check:
            issues.append(missing_check)
        
        # 2. Check price consistency (high >= low, close within high-low range)
        consistency_issues = self._check_price_consistency(df)
        if consistency_issues:
            issues.extend(consistency_issues)
        
        # 3. Check for price outliers
        outlier_issues = self._check_price_outliers(df)
        if outlier_issues:
            issues.extend(outlier_issues)
        
        # 4. Check for volume anomalies
        volume_issues = self._check_volume_anomalies(df)
        if volume_issues:
            issues.extend(volume_issues)
        
        # 5. Check for data staleness
        staleness_issue = self._check_data_staleness(df)
        if staleness_issue:
            issues.append(staleness_issue)
        
        # 6. Check for data gaps
        gap_issues = self._check_data_gaps(df)
        if gap_issues:
            issues.extend(gap_issues)
        
        # 7. Check for suspicious patterns
        pattern_issues = self._check_suspicious_patterns(df)
        if pattern_issues:
            issues.extend(pattern_issues)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(issues)
        
        # Store results
        result = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'valid': len([i for i in issues if i['severity'] in ['critical', 'high']]) == 0,
            'quality_score': quality_score,
            'issues': issues,
            'statistics': self._calculate_statistics(df),
            'recommendations': self._generate_recommendations(issues)
        }
        
        # Log critical issues
        for issue in issues:
            if issue['severity'] == DataQualitySeverity.CRITICAL.value:
                logger.error(f"Critical data quality issue for {symbol}: {issue['type']}")
        
        return result
    
    def _check_missing_values(self, df: pd.DataFrame) -> Optional[Dict]:
        """Check for missing values in the data."""
        missing_cols = df.columns[df.isnull().any()].tolist()
        
        if missing_cols:
            missing_pct = (df[missing_cols].isnull().sum() / len(df) * 100).to_dict()
            
            # Determine severity based on percentage
            max_pct = max(missing_pct.values())
            if max_pct > 20:
                severity = DataQualitySeverity.CRITICAL
            elif max_pct > 10:
                severity = DataQualitySeverity.HIGH
            elif max_pct > 5:
                severity = DataQualitySeverity.MEDIUM
            else:
                severity = DataQualitySeverity.LOW
            
            return {
                'type': 'missing_values',
                'severity': severity.value,
                'details': missing_pct,
                'affected_rows': df[missing_cols].isnull().any(axis=1).sum()
            }
        return None
    
    def _check_price_consistency(self, df: pd.DataFrame) -> List[Dict]:
        """Check for price consistency violations."""
        issues = []
        
        # Check high >= low
        invalid_hl = df['high'] < df['low']
        if invalid_hl.any():
            issues.append({
                'type': 'high_less_than_low',
                'severity': DataQualitySeverity.CRITICAL.value,
                'count': invalid_hl.sum(),
                'dates': df.index[invalid_hl].tolist()[:10]  # First 10 dates
            })
        
        # Check close within high-low range
        invalid_close = (df['close'] > df['high']) | (df['close'] < df['low'])
        if invalid_close.any():
            issues.append({
                'type': 'close_outside_range',
                'severity': DataQualitySeverity.CRITICAL.value,
                'count': invalid_close.sum(),
                'dates': df.index[invalid_close].tolist()[:10]
            })
        
        # Check open within reasonable range of previous close
        if len(df) > 1:
            df_sorted = df.sort_index()
            gap_pct = abs((df_sorted['open'].iloc[1:].values / 
                          df_sorted['close'].iloc[:-1].values - 1))
            large_gaps = gap_pct > 0.3  # 30% gap
            
            if large_gaps.any():
                issues.append({
                    'type': 'large_price_gaps',
                    'severity': DataQualitySeverity.MEDIUM.value,
                    'count': large_gaps.sum(),
                    'max_gap_pct': gap_pct.max() * 100
                })
        
        return issues
    
    def _check_price_outliers(self, df: pd.DataFrame) -> List[Dict]:
        """Check for price outliers using statistical methods."""
        issues = []
        
        # Calculate returns
        returns = df['close'].pct_change().dropna()
        
        if len(returns) > 20:
            # Z-score method
            z_scores = np.abs(stats.zscore(returns))
            outliers_z = z_scores > 3
            
            # IQR method
            Q1 = returns.quantile(0.25)
            Q3 = returns.quantile(0.75)
            IQR = Q3 - Q1
            outliers_iqr = (returns < (Q1 - 1.5 * IQR)) | (returns > (Q3 + 1.5 * IQR))
            
            # Combine methods
            outliers = outliers_z | outliers_iqr
            
            if outliers.any():
                outlier_returns = returns[outliers]
                
                # Check if outliers are extreme
                extreme_outliers = abs(outlier_returns) > self.validation_rules['price_data']['max_daily_change']
                
                severity = DataQualitySeverity.HIGH if extreme_outliers.any() else DataQualitySeverity.MEDIUM
                
                issues.append({
                    'type': 'price_outliers',
                    'severity': severity.value,
                    'count': outliers.sum(),
                    'extreme_count': extreme_outliers.sum() if extreme_outliers.any() else 0,
                    'max_return': outlier_returns.max() * 100,
                    'min_return': outlier_returns.min() * 100,
                    'dates': df.index[outliers].tolist()[:10]
                })
        
        return issues
    
    def _check_volume_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """Check for volume anomalies and suspicious patterns."""
        issues = []
        
        if 'volume' not in df.columns:
            return issues
        
        # Check for zero or negative volume
        zero_volume = df['volume'] <= 0
        if zero_volume.any():
            issues.append({
                'type': 'zero_or_negative_volume',
                'severity': DataQualitySeverity.HIGH.value,
                'count': zero_volume.sum(),
                'dates': df.index[zero_volume].tolist()[:10]
            })
        
        # Check for volume outliers
        if len(df) > 20:
            volume_z = np.abs(stats.zscore(df['volume'].fillna(0)))
            volume_outliers = volume_z > self.validation_rules['technical_indicators']['volume_outlier_std']
            
            if volume_outliers.any():
                issues.append({
                    'type': 'volume_outliers',
                    'severity': DataQualitySeverity.LOW.value,
                    'count': volume_outliers.sum(),
                    'max_volume': df.loc[volume_outliers, 'volume'].max(),
                    'avg_volume': df['volume'].mean()
                })
        
        # Check for sustained low volume
        if len(df) > 5:
            low_volume = df['volume'] < self.validation_rules['price_data']['min_volume']
            if low_volume.tail(5).all():
                issues.append({
                    'type': 'sustained_low_volume',
                    'severity': DataQualitySeverity.MEDIUM.value,
                    'days': low_volume.sum(),
                    'avg_volume': df['volume'].tail(5).mean()
                })
        
        return issues
    
    def _check_data_staleness(self, df: pd.DataFrame) -> Optional[Dict]:
        """Check if data is stale."""
        if df.empty:
            return {
                'type': 'no_data',
                'severity': DataQualitySeverity.CRITICAL.value,
                'details': 'DataFrame is empty'
            }
        
        # Assume df has datetime index or 'date' column
        if isinstance(df.index, pd.DatetimeIndex):
            last_date = df.index.max()
        elif 'date' in df.columns:
            last_date = pd.to_datetime(df['date']).max()
        else:
            return None
        
        days_old = (datetime.now().date() - last_date.date()).days
        
        if days_old > self.validation_rules['price_data']['stale_data_days']:
            severity = DataQualitySeverity.HIGH if days_old > 7 else DataQualitySeverity.MEDIUM
            
            return {
                'type': 'stale_data',
                'severity': severity.value,
                'last_update': last_date.isoformat(),
                'days_old': days_old
            }
        
        return None
    
    def _check_data_gaps(self, df: pd.DataFrame) -> List[Dict]:
        """Check for gaps in time series data."""
        issues = []
        
        if len(df) < 2:
            return issues
        
        # Check for missing trading days
        if isinstance(df.index, pd.DatetimeIndex):
            dates = df.index
        elif 'date' in df.columns:
            dates = pd.to_datetime(df['date'])
        else:
            return issues
        
        # Create expected business day range
        date_range = pd.bdate_range(start=dates.min(), end=dates.max())
        missing_dates = date_range.difference(dates)
        
        if len(missing_dates) > 0:
            # Group consecutive missing dates
            gaps = []
            if len(missing_dates) > 0:
                gap_start = missing_dates[0]
                gap_end = missing_dates[0]
                
                for i in range(1, len(missing_dates)):
                    if (missing_dates[i] - missing_dates[i-1]).days == 1:
                        gap_end = missing_dates[i]
                    else:
                        gaps.append((gap_start, gap_end))
                        gap_start = missing_dates[i]
                        gap_end = missing_dates[i]
                
                gaps.append((gap_start, gap_end))
            
            # Find significant gaps (> 3 days)
            significant_gaps = [(s, e) for s, e in gaps if (e - s).days > 3]
            
            if significant_gaps:
                issues.append({
                    'type': 'data_gaps',
                    'severity': DataQualitySeverity.MEDIUM.value,
                    'total_missing_days': len(missing_dates),
                    'gap_count': len(significant_gaps),
                    'largest_gap_days': max([(e - s).days for s, e in significant_gaps]),
                    'gaps': [(s.isoformat(), e.isoformat()) for s, e in significant_gaps[:5]]
                })
        
        return issues
    
    def _check_suspicious_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Check for suspicious or manipulated data patterns."""
        issues = []
        
        if len(df) < 10:
            return issues
        
        # Check for identical consecutive prices (potential data feed issue)
        identical_prices = (df['close'].diff() == 0).rolling(5).sum() == 5
        if identical_prices.any():
            issues.append({
                'type': 'identical_consecutive_prices',
                'severity': DataQualitySeverity.MEDIUM.value,
                'periods': identical_prices.sum(),
                'details': 'Five or more consecutive identical closing prices detected'
            })
        
        # Check for round number clustering (potential manipulation)
        round_prices = df['close'].apply(lambda x: x == round(x, 0))
        round_pct = round_prices.sum() / len(df)
        
        if round_pct > 0.3:  # More than 30% round numbers
            issues.append({
                'type': 'round_number_clustering',
                'severity': DataQualitySeverity.LOW.value,
                'percentage': round_pct * 100,
                'details': 'High percentage of round number prices'
            })
        
        # Check for artificial patterns (e.g., linear price movement)
        if len(df) > 20:
            # Calculate autocorrelation
            returns = df['close'].pct_change().dropna()
            if len(returns) > 1:
                autocorr = returns.autocorr()
                
                if abs(autocorr) > 0.95:  # Very high autocorrelation
                    issues.append({
                        'type': 'artificial_pattern',
                        'severity': DataQualitySeverity.HIGH.value,
                        'autocorrelation': autocorr,
                        'details': 'Potentially artificial price pattern detected'
                    })
        
        return issues
    
    def _calculate_quality_score(self, issues: List[Dict]) -> float:
        """
        Calculate overall data quality score (0-100).
        
        Score calculation:
        - Start with 100
        - Deduct points based on issue severity
        - Critical: -25 points
        - High: -15 points
        - Medium: -8 points
        - Low: -3 points
        - Info: -0 points
        """
        score = 100.0
        
        severity_penalties = {
            DataQualitySeverity.CRITICAL.value: 25,
            DataQualitySeverity.HIGH.value: 15,
            DataQualitySeverity.MEDIUM.value: 8,
            DataQualitySeverity.LOW.value: 3,
            DataQualitySeverity.INFO.value: 0
        }
        
        for issue in issues:
            penalty = severity_penalties.get(issue['severity'], 0)
            score -= penalty
        
        return max(0, min(100, score))
    
    def _calculate_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate summary statistics for the data."""
        stats = {}
        
        if not df.empty and 'close' in df.columns:
            returns = df['close'].pct_change().dropna()
            
            stats = {
                'row_count': len(df),
                'date_range': {
                    'start': df.index.min().isoformat() if isinstance(df.index, pd.DatetimeIndex) else str(df.index.min()),
                    'end': df.index.max().isoformat() if isinstance(df.index, pd.DatetimeIndex) else str(df.index.max())
                },
                'price_stats': {
                    'min': df['close'].min(),
                    'max': df['close'].max(),
                    'mean': df['close'].mean(),
                    'std': df['close'].std(),
                    'current': df['close'].iloc[-1] if len(df) > 0 else None
                },
                'return_stats': {
                    'mean_return': returns.mean() * 100,
                    'std_return': returns.std() * 100,
                    'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
                    'max_drawdown': self._calculate_max_drawdown(df['close'])
                } if len(returns) > 0 else {},
                'volume_stats': {
                    'mean': df['volume'].mean(),
                    'median': df['volume'].median(),
                    'total': df['volume'].sum()
                } if 'volume' in df.columns else {}
            }
        
        return stats
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown percentage."""
        cummax = prices.expanding().max()
        drawdown = (prices - cummax) / cummax
        return drawdown.min() * 100
    
    def _generate_recommendations(self, issues: List[Dict]) -> List[str]:
        """Generate actionable recommendations based on issues found."""
        recommendations = []
        
        issue_types = {issue['type'] for issue in issues}
        
        if 'missing_values' in issue_types:
            recommendations.append("Implement data imputation or fetch missing data from alternative sources")
        
        if 'stale_data' in issue_types:
            recommendations.append("Update data fetching schedule or check data source availability")
        
        if 'price_outliers' in issue_types:
            recommendations.append("Review outlier data points for potential errors or corporate actions")
        
        if 'data_gaps' in issue_types:
            recommendations.append("Backfill missing historical data or adjust analysis window")
        
        if 'zero_or_negative_volume' in issue_types:
            recommendations.append("Verify volume data source or filter out invalid volume records")
        
        if 'artificial_pattern' in issue_types:
            recommendations.append("Investigate data source for potential feed issues or manipulation")
        
        if not recommendations:
            recommendations.append("Data quality is acceptable, continue with standard processing")
        
        return recommendations
    
    def validate_fundamental_data(
        self,
        fundamentals: Dict[str, Any],
        symbol: str = "UNKNOWN"
    ) -> Dict[str, Any]:
        """
        Validate fundamental data quality.
        
        Args:
            fundamentals: Dictionary of fundamental metrics
            symbol: Stock symbol
            
        Returns:
            Validation results
        """
        issues = []
        rules = self.validation_rules['fundamental_data']
        
        # Check P/E ratio
        if 'pe_ratio' in fundamentals:
            pe = fundamentals['pe_ratio']
            if pe and pe > rules['max_pe_ratio']:
                issues.append({
                    'type': 'excessive_pe_ratio',
                    'severity': DataQualitySeverity.MEDIUM.value,
                    'value': pe,
                    'threshold': rules['max_pe_ratio']
                })
        
        # Check market cap
        if 'market_cap' in fundamentals:
            mcap = fundamentals['market_cap']
            if mcap and mcap < rules['min_market_cap']:
                issues.append({
                    'type': 'low_market_cap',
                    'severity': DataQualitySeverity.LOW.value,
                    'value': mcap,
                    'threshold': rules['min_market_cap']
                })
        
        # Check debt ratio
        if 'debt_to_equity' in fundamentals:
            debt_ratio = fundamentals['debt_to_equity']
            if debt_ratio and debt_ratio > rules['max_debt_ratio']:
                issues.append({
                    'type': 'high_debt_ratio',
                    'severity': DataQualitySeverity.MEDIUM.value,
                    'value': debt_ratio,
                    'threshold': rules['max_debt_ratio']
                })
        
        quality_score = self._calculate_quality_score(issues)
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'valid': len([i for i in issues if i['severity'] in ['critical', 'high']]) == 0,
            'quality_score': quality_score,
            'issues': issues
        }
    
    def validate_ml_features(
        self,
        features: pd.DataFrame,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """
        Validate ML feature quality.
        
        Args:
            features: DataFrame of features
            feature_names: List of expected feature names
            
        Returns:
            Validation results
        """
        issues = []
        
        # Check for missing features
        missing_features = set(feature_names) - set(features.columns)
        if missing_features:
            issues.append({
                'type': 'missing_features',
                'severity': DataQualitySeverity.HIGH.value,
                'features': list(missing_features)
            })
        
        # Check for infinite or NaN values
        inf_features = features.columns[np.isinf(features).any()].tolist()
        if inf_features:
            issues.append({
                'type': 'infinite_values',
                'severity': DataQualitySeverity.CRITICAL.value,
                'features': inf_features
            })
        
        nan_features = features.columns[features.isnull().any()].tolist()
        if nan_features:
            issues.append({
                'type': 'nan_values',
                'severity': DataQualitySeverity.HIGH.value,
                'features': nan_features
            })
        
        # Check feature distributions
        for col in features.select_dtypes(include=[np.number]).columns:
            # Check for zero variance
            if features[col].std() == 0:
                issues.append({
                    'type': 'zero_variance_feature',
                    'severity': DataQualitySeverity.MEDIUM.value,
                    'feature': col
                })
            
            # Check for extreme skewness
            skewness = features[col].skew()
            if abs(skewness) > 10:
                issues.append({
                    'type': 'extreme_skewness',
                    'severity': DataQualitySeverity.LOW.value,
                    'feature': col,
                    'skewness': skewness
                })
        
        quality_score = self._calculate_quality_score(issues)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'valid': len([i for i in issues if i['severity'] in ['critical', 'high']]) == 0,
            'quality_score': quality_score,
            'issues': issues,
            'feature_count': len(features.columns),
            'row_count': len(features)
        }
    
    def generate_quality_report(
        self,
        validation_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive data quality report.
        
        Args:
            validation_results: List of validation results
            
        Returns:
            Quality report summary
        """
        if not validation_results:
            return {'error': 'No validation results provided'}
        
        # Aggregate results
        total_checks = len(validation_results)
        valid_checks = sum(1 for r in validation_results if r.get('valid', False))
        avg_quality_score = np.mean([r.get('quality_score', 0) for r in validation_results])
        
        # Count issues by severity
        severity_counts = {
            DataQualitySeverity.CRITICAL.value: 0,
            DataQualitySeverity.HIGH.value: 0,
            DataQualitySeverity.MEDIUM.value: 0,
            DataQualitySeverity.LOW.value: 0,
            DataQualitySeverity.INFO.value: 0
        }
        
        issue_type_counts = {}
        
        for result in validation_results:
            for issue in result.get('issues', []):
                severity = issue.get('severity')
                issue_type = issue.get('type')
                
                if severity in severity_counts:
                    severity_counts[severity] += 1
                
                if issue_type:
                    issue_type_counts[issue_type] = issue_type_counts.get(issue_type, 0) + 1
        
        # Generate summary
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_validations': total_checks,
                'passed_validations': valid_checks,
                'failed_validations': total_checks - valid_checks,
                'pass_rate': (valid_checks / total_checks * 100) if total_checks > 0 else 0,
                'average_quality_score': avg_quality_score
            },
            'severity_distribution': severity_counts,
            'top_issues': sorted(
                issue_type_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10],
            'recommendations': self._generate_report_recommendations(severity_counts, issue_type_counts),
            'details': validation_results
        }
        
        return report
    
    def _generate_report_recommendations(
        self,
        severity_counts: Dict,
        issue_type_counts: Dict
    ) -> List[str]:
        """Generate recommendations based on report summary."""
        recommendations = []
        
        if severity_counts[DataQualitySeverity.CRITICAL.value] > 0:
            recommendations.append("URGENT: Address critical data quality issues immediately")
        
        if severity_counts[DataQualitySeverity.HIGH.value] > 5:
            recommendations.append("Multiple high-severity issues detected - review data pipeline")
        
        if 'stale_data' in issue_type_counts and issue_type_counts['stale_data'] > 10:
            recommendations.append("Widespread stale data - check data source connectivity")
        
        if 'missing_values' in issue_type_counts and issue_type_counts['missing_values'] > 20:
            recommendations.append("Excessive missing values - implement data imputation strategy")
        
        if not recommendations:
            recommendations.append("Data quality within acceptable parameters")
        
        return recommendations


# Utility functions for quick validation
def quick_validate_prices(df: pd.DataFrame, symbol: str = "UNKNOWN") -> bool:
    """Quick validation check for price data."""
    checker = DataQualityChecker()
    result = checker.validate_price_data(df, symbol)
    return result['valid']


def quick_validate_fundamentals(fundamentals: Dict, symbol: str = "UNKNOWN") -> bool:
    """Quick validation check for fundamental data."""
    checker = DataQualityChecker()
    result = checker.validate_fundamental_data(fundamentals, symbol)
    return result['valid']


def validate_batch(
    data_batch: List[Tuple[str, pd.DataFrame]]
) -> Dict[str, Any]:
    """
    Validate a batch of stock data.
    
    Args:
        data_batch: List of (symbol, dataframe) tuples
        
    Returns:
        Batch validation report
    """
    checker = DataQualityChecker()
    results = []
    
    for symbol, df in data_batch:
        result = checker.validate_price_data(df, symbol)
        results.append(result)
    
    return checker.generate_quality_report(results)