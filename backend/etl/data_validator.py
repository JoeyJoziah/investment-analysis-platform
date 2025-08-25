"""
Data Validation and Quality Assurance System
Validates financial data quality and consistency across multiple sources
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from enum import Enum
import re
import math

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    BASIC = "basic"
    STANDARD = "standard" 
    STRICT = "strict"
    COMPREHENSIVE = "comprehensive"


class ValidationSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    field_name: str
    severity: ValidationSeverity
    message: str
    expected_value: Any = None
    actual_value: Any = None
    suggestion: str = None


@dataclass
class DataQualityScore:
    overall_score: float  # 0-100
    completeness_score: float
    accuracy_score: float
    consistency_score: float
    timeliness_score: float
    validation_results: List[ValidationResult]
    
    def is_acceptable(self, min_score: float = 70.0) -> bool:
        return self.overall_score >= min_score


class FinancialDataValidator:
    """Comprehensive validator for financial data"""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        
        # Known exchange mappings
        self.exchange_mappings = {
            'NYSE': ['NYSE', 'New York Stock Exchange'],
            'NASDAQ': ['NASDAQ', 'NASDAQ Global Select', 'NASDAQ Global Market', 'NASDAQ Capital Market'],
            'AMEX': ['AMEX', 'NYSE American', 'American Stock Exchange']
        }
        
        # Valid sectors (GICS)
        self.valid_sectors = {
            'Energy', 'Materials', 'Industrials', 'Consumer Discretionary',
            'Consumer Staples', 'Health Care', 'Financials', 'Information Technology',
            'Communication Services', 'Utilities', 'Real Estate'
        }
        
        # Reasonable value ranges for financial metrics
        self.metric_ranges = {
            'price': (0.01, 10000),  # $0.01 to $10,000
            'volume': (0, 1e10),  # 0 to 10 billion shares
            'market_cap': (1e6, 1e13),  # $1M to $10T
            'pe_ratio': (-100, 200),  # Can be negative, but reasonable upper bound
            'price_change_pct': (-50, 50),  # ±50% daily change is extreme but possible
            'dividend_yield': (0, 20),  # 0 to 20%
            'beta': (-3, 3),  # Most stocks between -3 and 3
        }
    
    def validate_stock_data(self, data: Dict, ticker: str) -> DataQualityScore:
        """Validate a complete stock data record"""
        validation_results = []
        
        # Basic validation
        validation_results.extend(self._validate_basic_structure(data, ticker))
        validation_results.extend(self._validate_ticker_format(ticker))
        
        # Price data validation
        if 'current_price' in data or 'price_data' in data:
            validation_results.extend(self._validate_price_data(data, ticker))
        
        # Company information validation
        if 'company_name' in data or 'company_info' in data:
            validation_results.extend(self._validate_company_info(data, ticker))
        
        # Market data validation
        validation_results.extend(self._validate_market_metrics(data, ticker))
        
        # Historical data validation
        if 'historical_data' in data:
            validation_results.extend(self._validate_historical_data(data['historical_data'], ticker))
        
        # Cross-source consistency validation
        if 'sources' in data and len(data.get('sources', {})) > 1:
            validation_results.extend(self._validate_cross_source_consistency(data, ticker))
        
        # Advanced validations for higher levels
        if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.COMPREHENSIVE]:
            validation_results.extend(self._validate_advanced_metrics(data, ticker))
        
        # Calculate quality scores
        scores = self._calculate_quality_scores(data, validation_results)
        
        return DataQualityScore(
            overall_score=scores['overall'],
            completeness_score=scores['completeness'],
            accuracy_score=scores['accuracy'],
            consistency_score=scores['consistency'],
            timeliness_score=scores['timeliness'],
            validation_results=validation_results
        )
    
    def _validate_basic_structure(self, data: Dict, ticker: str) -> List[ValidationResult]:
        """Validate basic data structure"""
        results = []
        
        # Check for required fields
        required_fields = ['ticker', 'timestamp', 'source']
        for field in required_fields:
            if field not in data:
                results.append(ValidationResult(
                    field_name=field,
                    severity=ValidationSeverity.ERROR,
                    message=f"Required field '{field}' is missing",
                    suggestion=f"Ensure {field} is included in data extraction"
                ))
        
        # Validate timestamp
        if 'timestamp' in data:
            try:
                if isinstance(data['timestamp'], str):
                    datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
                elif not isinstance(data['timestamp'], datetime):
                    results.append(ValidationResult(
                        field_name='timestamp',
                        severity=ValidationSeverity.WARNING,
                        message="Timestamp is not in expected format",
                        actual_value=type(data['timestamp']).__name__
                    ))
            except ValueError:
                results.append(ValidationResult(
                    field_name='timestamp',
                    severity=ValidationSeverity.ERROR,
                    message="Invalid timestamp format",
                    actual_value=data.get('timestamp')
                ))
        
        return results
    
    def _validate_ticker_format(self, ticker: str) -> List[ValidationResult]:
        """Validate ticker symbol format"""
        results = []
        
        if not ticker:
            results.append(ValidationResult(
                field_name='ticker',
                severity=ValidationSeverity.CRITICAL,
                message="Ticker symbol is empty or None"
            ))
            return results
        
        # Basic ticker format validation
        if not re.match(r'^[A-Z]{1,5}(-[A-Z])?$', ticker.upper()):
            # Allow some common variations
            if not re.match(r'^[A-Z]{1,5}(\.[A-Z])?$', ticker.upper()):  # BRK.A format
                results.append(ValidationResult(
                    field_name='ticker',
                    severity=ValidationSeverity.WARNING,
                    message="Ticker format may be non-standard",
                    actual_value=ticker,
                    suggestion="Expected format: 1-5 letters, optionally followed by -A, -B, etc."
                ))
        
        # Check for common invalid characters
        if any(char in ticker for char in [' ', '.', '_']):
            results.append(ValidationResult(
                field_name='ticker',
                severity=ValidationSeverity.WARNING,
                message="Ticker contains potentially invalid characters",
                actual_value=ticker
            ))
        
        return results
    
    def _validate_price_data(self, data: Dict, ticker: str) -> List[ValidationResult]:
        """Validate price-related data"""
        results = []
        
        # Get current price from various possible locations
        current_price = self._extract_price(data)
        
        if current_price is not None:
            # Validate price range
            min_price, max_price = self.metric_ranges['price']
            if current_price < min_price or current_price > max_price:
                results.append(ValidationResult(
                    field_name='current_price',
                    severity=ValidationSeverity.ERROR,
                    message=f"Price {current_price} outside reasonable range",
                    actual_value=current_price,
                    expected_value=f"${min_price} - ${max_price}",
                    suggestion="Verify price data source and currency"
                ))
            
            # Check for common price errors
            if current_price == 0:
                results.append(ValidationResult(
                    field_name='current_price',
                    severity=ValidationSeverity.WARNING,
                    message="Price is zero - may indicate delisted stock or data error",
                    actual_value=current_price
                ))
            
            if current_price < 0.01:
                results.append(ValidationResult(
                    field_name='current_price',
                    severity=ValidationSeverity.WARNING,
                    message="Very low price - may be penny stock or data error",
                    actual_value=current_price
                ))
        
        # Validate OHLC data if present
        price_data = data.get('price_data', {})
        if price_data and isinstance(price_data, dict):
            results.extend(self._validate_ohlc_data(price_data, ticker))
        
        # Validate volume
        volume = self._extract_volume(data)
        if volume is not None:
            min_vol, max_vol = self.metric_ranges['volume']
            if volume < min_vol or volume > max_vol:
                results.append(ValidationResult(
                    field_name='volume',
                    severity=ValidationSeverity.WARNING,
                    message=f"Volume {volume:,} outside typical range",
                    actual_value=volume
                ))
        
        return results
    
    def _validate_ohlc_data(self, price_data: Dict, ticker: str) -> List[ValidationResult]:
        """Validate Open, High, Low, Close data"""
        results = []
        
        required_fields = ['open', 'high', 'low', 'close']
        values = {}
        
        # Extract and validate individual values
        for field in required_fields:
            if field in price_data:
                value = price_data[field]
                if isinstance(value, (int, float)) and not math.isnan(value):
                    values[field] = float(value)
                else:
                    results.append(ValidationResult(
                        field_name=f'price_data.{field}',
                        severity=ValidationSeverity.ERROR,
                        message=f"Invalid {field} price value",
                        actual_value=value
                    ))
        
        # Validate OHLC relationships
        if len(values) == 4:  # All OHLC values present
            o, h, l, c = values['open'], values['high'], values['low'], values['close']
            
            # High should be >= all others
            if not (h >= o and h >= l and h >= c):
                results.append(ValidationResult(
                    field_name='price_data.high',
                    severity=ValidationSeverity.ERROR,
                    message="High price is not the highest value in OHLC",
                    actual_value=h,
                    suggestion="Check data source for price inconsistencies"
                ))
            
            # Low should be <= all others
            if not (l <= o and l <= h and l <= c):
                results.append(ValidationResult(
                    field_name='price_data.low',
                    severity=ValidationSeverity.ERROR,
                    message="Low price is not the lowest value in OHLC",
                    actual_value=l,
                    suggestion="Check data source for price inconsistencies"
                ))
            
            # Check for suspicious patterns
            if h == l and h > 0:  # Single price point
                results.append(ValidationResult(
                    field_name='price_data',
                    severity=ValidationSeverity.WARNING,
                    message="High and Low prices are identical - may indicate low liquidity or data issue",
                    actual_value=f"H={h}, L={l}"
                ))
        
        return results
    
    def _validate_company_info(self, data: Dict, ticker: str) -> List[ValidationResult]:
        """Validate company information"""
        results = []
        
        # Validate company name
        company_name = data.get('company_name') or data.get('company_info', {}).get('name')
        if company_name:
            if len(company_name) < 2:
                results.append(ValidationResult(
                    field_name='company_name',
                    severity=ValidationSeverity.WARNING,
                    message="Company name appears too short",
                    actual_value=company_name
                ))
            
            # Check for placeholder names
            if company_name.upper() in [ticker.upper(), 'N/A', 'UNKNOWN', 'NULL']:
                results.append(ValidationResult(
                    field_name='company_name',
                    severity=ValidationSeverity.WARNING,
                    message="Company name appears to be a placeholder",
                    actual_value=company_name
                ))
        
        # Validate sector
        sector = data.get('sector') or data.get('company_info', {}).get('sector')
        if sector and sector not in self.valid_sectors:
            results.append(ValidationResult(
                field_name='sector',
                severity=ValidationSeverity.WARNING,
                message="Sector not in standard GICS sectors",
                actual_value=sector,
                suggestion=f"Expected one of: {', '.join(sorted(self.valid_sectors))}"
            ))
        
        return results
    
    def _validate_market_metrics(self, data: Dict, ticker: str) -> List[ValidationResult]:
        """Validate market-related metrics"""
        results = []
        
        # Market cap validation
        market_cap = self._extract_market_cap(data)
        if market_cap is not None:
            min_cap, max_cap = self.metric_ranges['market_cap']
            if market_cap < min_cap or market_cap > max_cap:
                results.append(ValidationResult(
                    field_name='market_cap',
                    severity=ValidationSeverity.WARNING,
                    message=f"Market cap ${market_cap:,.0f} outside typical range",
                    actual_value=market_cap,
                    expected_value=f"${min_cap:,.0f} - ${max_cap:,.0f}"
                ))
        
        # P/E ratio validation
        pe_ratio = self._extract_pe_ratio(data)
        if pe_ratio is not None:
            min_pe, max_pe = self.metric_ranges['pe_ratio']
            if pe_ratio < min_pe or pe_ratio > max_pe:
                results.append(ValidationResult(
                    field_name='pe_ratio',
                    severity=ValidationSeverity.INFO,
                    message=f"P/E ratio {pe_ratio:.2f} outside typical range",
                    actual_value=pe_ratio,
                    suggestion="Very high/low P/E ratios may indicate growth/distressed stocks"
                ))
        
        # Price change validation
        price_change_pct = data.get('price_change_pct')
        if price_change_pct is not None:
            min_change, max_change = self.metric_ranges['price_change_pct']
            if price_change_pct < min_change or price_change_pct > max_change:
                results.append(ValidationResult(
                    field_name='price_change_pct',
                    severity=ValidationSeverity.WARNING,
                    message=f"Price change {price_change_pct:.2f}% is extreme",
                    actual_value=price_change_pct,
                    suggestion="Verify if this is a stock split, earnings announcement, or data error"
                ))
        
        return results
    
    def _validate_historical_data(self, historical_data: List[Dict], ticker: str) -> List[ValidationResult]:
        """Validate historical price data"""
        results = []
        
        if not historical_data or not isinstance(historical_data, list):
            results.append(ValidationResult(
                field_name='historical_data',
                severity=ValidationSeverity.WARNING,
                message="No historical data available",
                suggestion="Historical data improves analysis quality"
            ))
            return results
        
        # Check data completeness
        if len(historical_data) < 5:
            results.append(ValidationResult(
                field_name='historical_data',
                severity=ValidationSeverity.WARNING,
                message=f"Limited historical data: only {len(historical_data)} data points",
                suggestion="More historical data improves trend analysis"
            ))
        
        # Validate each historical record
        prev_date = None
        for i, record in enumerate(historical_data):
            if isinstance(record, dict):
                # Check for required fields in historical data
                if 'date' not in record:
                    results.append(ValidationResult(
                        field_name=f'historical_data[{i}].date',
                        severity=ValidationSeverity.ERROR,
                        message="Historical record missing date"
                    ))
                
                # Validate date sequence
                if 'date' in record and prev_date:
                    try:
                        current_date = pd.to_datetime(record['date'])
                        if current_date <= prev_date:
                            results.append(ValidationResult(
                                field_name=f'historical_data[{i}].date',
                                severity=ValidationSeverity.WARNING,
                                message="Historical data not in chronological order",
                                actual_value=record['date']
                            ))
                        prev_date = current_date
                    except:
                        results.append(ValidationResult(
                            field_name=f'historical_data[{i}].date',
                            severity=ValidationSeverity.ERROR,
                            message="Invalid date format in historical data",
                            actual_value=record.get('date')
                        ))
                
                # Validate OHLC in historical record
                if all(field in record for field in ['open', 'high', 'low', 'close']):
                    hist_results = self._validate_ohlc_data(record, ticker)
                    # Prefix field names for historical context
                    for result in hist_results:
                        result.field_name = f'historical_data[{i}].{result.field_name}'
                    results.extend(hist_results)
        
        return results
    
    def _validate_cross_source_consistency(self, data: Dict, ticker: str) -> List[ValidationResult]:
        """Validate consistency across multiple data sources"""
        results = []
        
        sources = data.get('sources', {})
        if len(sources) < 2:
            return results
        
        # Extract prices from different sources
        source_prices = {}
        for source_name, source_data in sources.items():
            price = self._extract_price(source_data)
            if price is not None:
                source_prices[source_name] = price
        
        # Check price consistency
        if len(source_prices) >= 2:
            prices = list(source_prices.values())
            price_range = max(prices) - min(prices)
            avg_price = sum(prices) / len(prices)
            
            if avg_price > 0:
                price_variance_pct = (price_range / avg_price) * 100
                
                if price_variance_pct > 5:  # More than 5% difference
                    results.append(ValidationResult(
                        field_name='cross_source_price_consistency',
                        severity=ValidationSeverity.WARNING,
                        message=f"Price varies {price_variance_pct:.1f}% across sources",
                        actual_value=source_prices,
                        suggestion="Check if sources use different currencies or timestamps"
                    ))
        
        return results
    
    def _validate_advanced_metrics(self, data: Dict, ticker: str) -> List[ValidationResult]:
        """Advanced validation for strict/comprehensive levels"""
        results = []
        
        # Validate data freshness
        timestamp = data.get('timestamp')
        if timestamp:
            try:
                if isinstance(timestamp, str):
                    data_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                else:
                    data_time = timestamp
                
                age_hours = (datetime.now() - data_time.replace(tzinfo=None)).total_seconds() / 3600
                
                if age_hours > 24:
                    results.append(ValidationResult(
                        field_name='data_freshness',
                        severity=ValidationSeverity.WARNING,
                        message=f"Data is {age_hours:.1f} hours old",
                        actual_value=f"{age_hours:.1f} hours",
                        suggestion="Consider refreshing data for real-time analysis"
                    ))
            except:
                pass
        
        # Validate completeness score
        completeness = self._calculate_completeness(data)
        if completeness < 0.7:
            results.append(ValidationResult(
                field_name='data_completeness',
                severity=ValidationSeverity.WARNING,
                message=f"Data completeness is {completeness*100:.1f}%",
                actual_value=f"{completeness*100:.1f}%",
                suggestion="Missing fields may impact analysis quality"
            ))
        
        return results
    
    def _calculate_quality_scores(self, data: Dict, validation_results: List[ValidationResult]) -> Dict[str, float]:
        """Calculate various quality scores"""
        
        # Count issues by severity
        error_count = sum(1 for r in validation_results if r.severity == ValidationSeverity.ERROR)
        warning_count = sum(1 for r in validation_results if r.severity == ValidationSeverity.WARNING)
        critical_count = sum(1 for r in validation_results if r.severity == ValidationSeverity.CRITICAL)
        
        # Accuracy score (penalize errors more heavily)
        accuracy_score = max(0, 100 - (critical_count * 40) - (error_count * 15) - (warning_count * 5))
        
        # Completeness score
        completeness_score = self._calculate_completeness(data) * 100
        
        # Consistency score (based on cross-source validation)
        consistency_issues = sum(1 for r in validation_results if 'consistency' in r.field_name.lower())
        consistency_score = max(0, 100 - (consistency_issues * 20))
        
        # Timeliness score
        timeliness_score = self._calculate_timeliness_score(data)
        
        # Overall score (weighted average)
        overall_score = (
            accuracy_score * 0.3 +
            completeness_score * 0.3 +
            consistency_score * 0.2 +
            timeliness_score * 0.2
        )
        
        return {
            'overall': overall_score,
            'accuracy': accuracy_score,
            'completeness': completeness_score,
            'consistency': consistency_score,
            'timeliness': timeliness_score
        }
    
    def _calculate_completeness(self, data: Dict) -> float:
        """Calculate data completeness score (0-1)"""
        essential_fields = [
            'ticker', 'current_price', 'timestamp', 'source'
        ]
        
        important_fields = [
            'volume', 'market_cap', 'company_name', 'price_change', 'price_change_pct'
        ]
        
        optional_fields = [
            'sector', 'industry', 'pe_ratio', 'historical_data'
        ]
        
        # Check field presence across nested structures
        available_fields = self._get_available_fields(data)
        
        essential_present = sum(1 for field in essential_fields if field in available_fields)
        important_present = sum(1 for field in important_fields if field in available_fields)
        optional_present = sum(1 for field in optional_fields if field in available_fields)
        
        # Weighted score
        essential_weight = 0.6
        important_weight = 0.3
        optional_weight = 0.1
        
        completeness = (
            (essential_present / len(essential_fields)) * essential_weight +
            (important_present / len(important_fields)) * important_weight +
            (optional_present / len(optional_fields)) * optional_weight
        )
        
        return min(1.0, completeness)
    
    def _calculate_timeliness_score(self, data: Dict) -> float:
        """Calculate timeliness score based on data age"""
        timestamp = data.get('timestamp')
        if not timestamp:
            return 50.0  # No timestamp, medium score
        
        try:
            if isinstance(timestamp, str):
                data_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                data_time = timestamp
            
            age_hours = (datetime.now() - data_time.replace(tzinfo=None)).total_seconds() / 3600
            
            if age_hours <= 1:
                return 100.0
            elif age_hours <= 6:
                return 90.0
            elif age_hours <= 24:
                return 75.0
            elif age_hours <= 168:  # 1 week
                return 50.0
            else:
                return 25.0
                
        except:
            return 50.0
    
    def _get_available_fields(self, data: Dict, prefix: str = '') -> Set[str]:
        """Recursively get all available fields in nested data structure"""
        fields = set()
        
        for key, value in data.items():
            field_name = f"{prefix}.{key}" if prefix else key
            fields.add(field_name)
            
            if isinstance(value, dict):
                fields.update(self._get_available_fields(value, field_name))
        
        return fields
    
    def _extract_price(self, data: Dict) -> Optional[float]:
        """Extract current price from various possible locations in data"""
        price_fields = [
            'current_price', 'price', 'close', 
            'price_data.close', 'quote.price', 'c'
        ]
        
        for field in price_fields:
            value = self._get_nested_value(data, field)
            if value is not None and isinstance(value, (int, float)) and not math.isnan(value):
                return float(value)
        
        return None
    
    def _extract_volume(self, data: Dict) -> Optional[int]:
        """Extract volume from various possible locations in data"""
        volume_fields = ['volume', 'vol', 'price_data.volume', 'quote.volume', 'v']
        
        for field in volume_fields:
            value = self._get_nested_value(data, field)
            if value is not None and isinstance(value, (int, float)) and not math.isnan(value):
                return int(value)
        
        return None
    
    def _extract_market_cap(self, data: Dict) -> Optional[float]:
        """Extract market cap from various possible locations in data"""
        cap_fields = ['market_cap', 'marketCap', 'company_info.market_cap']
        
        for field in cap_fields:
            value = self._get_nested_value(data, field)
            if value is not None and isinstance(value, (int, float)) and not math.isnan(value):
                return float(value)
        
        return None
    
    def _extract_pe_ratio(self, data: Dict) -> Optional[float]:
        """Extract P/E ratio from various possible locations in data"""
        pe_fields = ['pe_ratio', 'trailingPE', 'company_info.pe_ratio', 'peRatio']
        
        for field in pe_fields:
            value = self._get_nested_value(data, field)
            if value is not None and isinstance(value, (int, float)) and not math.isnan(value):
                return float(value)
        
        return None
    
    def _get_nested_value(self, data: Dict, field_path: str) -> Any:
        """Get value from nested dictionary using dot notation"""
        keys = field_path.split('.')
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current


def validate_extraction_results(results: List[Dict], 
                              validation_level: ValidationLevel = ValidationLevel.STANDARD,
                              min_quality_score: float = 70.0) -> Dict[str, Any]:
    """
    Validate a batch of extraction results
    
    Args:
        results: List of extracted stock data dictionaries
        validation_level: Level of validation to perform
        min_quality_score: Minimum acceptable quality score
    
    Returns:
        Dictionary with validation summary and filtered results
    """
    validator = FinancialDataValidator(validation_level)
    
    validation_summary = {
        'total_records': len(results),
        'valid_records': 0,
        'invalid_records': 0,
        'quality_scores': [],
        'common_issues': {},
        'filtered_results': []
    }
    
    for data in results:
        ticker = data.get('ticker', 'UNKNOWN')
        quality_score = validator.validate_stock_data(data, ticker)
        
        validation_summary['quality_scores'].append(quality_score.overall_score)
        
        # Track common issues
        for result in quality_score.validation_results:
            issue_key = f"{result.severity.value}:{result.message}"
            validation_summary['common_issues'][issue_key] = validation_summary['common_issues'].get(issue_key, 0) + 1
        
        # Filter based on quality score
        if quality_score.overall_score >= min_quality_score:
            validation_summary['valid_records'] += 1
            validation_summary['filtered_results'].append({
                'data': data,
                'quality_score': quality_score.overall_score,
                'validation_results': quality_score.validation_results
            })
        else:
            validation_summary['invalid_records'] += 1
    
    # Calculate summary statistics
    if validation_summary['quality_scores']:
        validation_summary['avg_quality_score'] = np.mean(validation_summary['quality_scores'])
        validation_summary['min_quality_score'] = np.min(validation_summary['quality_scores'])
        validation_summary['max_quality_score'] = np.max(validation_summary['quality_scores'])
    
    return validation_summary


if __name__ == "__main__":
    # Test the validator
    test_data = {
        'ticker': 'AAPL',
        'current_price': 150.25,
        'volume': 75000000,
        'market_cap': 2500000000000,
        'company_name': 'Apple Inc.',
        'sector': 'Information Technology',
        'pe_ratio': 25.5,
        'price_change_pct': 2.1,
        'timestamp': datetime.now(),
        'source': 'yahoo_finance',
        'historical_data': [
            {'date': '2023-01-01', 'open': 148.0, 'high': 152.0, 'low': 147.0, 'close': 150.0},
            {'date': '2023-01-02', 'open': 150.0, 'high': 153.0, 'low': 149.0, 'close': 151.5}
        ]
    }
    
    validator = FinancialDataValidator(ValidationLevel.COMPREHENSIVE)
    quality_score = validator.validate_stock_data(test_data, 'AAPL')
    
    print(f"Quality Score: {quality_score.overall_score:.2f}")
    print(f"Completeness: {quality_score.completeness_score:.2f}")
    print(f"Accuracy: {quality_score.accuracy_score:.2f}")
    print(f"Consistency: {quality_score.consistency_score:.2f}")
    print(f"Timeliness: {quality_score.timeliness_score:.2f}")
    
    print("\\nValidation Issues:")
    for result in quality_score.validation_results:
        print(f"  {result.severity.value.upper()}: {result.field_name} - {result.message}")