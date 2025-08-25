"""
Data Validation and Cleaning Pipeline for Unlimited Stock Data Extraction
Implements comprehensive validation, cleaning, and quality scoring for financial data
"""

import asyncio
import re
import math
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set, Union, Callable
import logging
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
import json
from collections import defaultdict, Counter
import statistics
import warnings

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Validation strictness levels"""
    BASIC = "basic"           # Essential fields only
    STANDARD = "standard"     # Common fields with reasonable bounds
    STRICT = "strict"         # Strict bounds and consistency checks
    COMPREHENSIVE = "comprehensive"  # Full validation with cross-field checks

class DataQualityIssue(Enum):
    """Types of data quality issues"""
    MISSING_REQUIRED = "missing_required_field"
    INVALID_FORMAT = "invalid_format"
    OUT_OF_RANGE = "out_of_range"
    INCONSISTENT = "inconsistent_data"
    SUSPICIOUS = "suspicious_value"
    STALE_DATA = "stale_data"
    INCOMPLETE = "incomplete_data"
    DUPLICATE = "duplicate_data"

@dataclass
class ValidationRule:
    """Represents a validation rule"""
    field_name: str
    rule_type: str  # 'required', 'range', 'format', 'custom'
    parameters: Dict[str, Any] = field(default_factory=dict)
    severity: str = 'error'  # 'error', 'warning', 'info'
    description: str = ""
    validation_function: Optional[Callable] = None

@dataclass
class ValidationResult:
    """Result of a validation check"""
    field_name: str
    issue_type: DataQualityIssue
    severity: str
    message: str
    suggested_fix: Optional[str] = None
    original_value: Any = None
    corrected_value: Any = None

@dataclass
class DataQualityScore:
    """Data quality assessment"""
    overall_score: int  # 0-100
    completeness_score: int
    accuracy_score: int
    consistency_score: int
    timeliness_score: int
    issues: List[ValidationResult] = field(default_factory=list)
    total_fields_checked: int = 0
    valid_fields: int = 0
    
    @property
    def pass_rate(self) -> float:
        return (self.valid_fields / self.total_fields_checked) if self.total_fields_checked > 0 else 0.0

class FinancialDataValidator:
    """Comprehensive validator for financial stock data"""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.validation_rules = self._initialize_validation_rules()
        self.market_data_ranges = self._initialize_market_ranges()
        self.cleaning_enabled = True
        
        # Statistics for adaptive validation
        self.field_statistics = defaultdict(list)
        self.common_issues = Counter()
        
        logger.info(f"Initialized FinancialDataValidator with {validation_level.value} validation level")
    
    def _initialize_validation_rules(self) -> Dict[str, List[ValidationRule]]:
        """Initialize validation rules based on validation level"""
        rules = defaultdict(list)
        
        # Basic rules (required for all levels)
        rules['ticker'].extend([
            ValidationRule(
                'ticker', 'required', {},
                description="Ticker symbol is required"
            ),
            ValidationRule(
                'ticker', 'format', {'pattern': r'^[A-Z]{1,5}$'},
                description="Ticker must be 1-5 uppercase letters"
            )
        ])
        
        # Price validation
        if self.validation_level in [ValidationLevel.BASIC, ValidationLevel.STANDARD, ValidationLevel.STRICT, ValidationLevel.COMPREHENSIVE]:
            rules['current_price'].extend([
                ValidationRule(
                    'current_price', 'range', {'min': 0.01, 'max': 10000.0},
                    description="Stock price must be between $0.01 and $10,000"
                )
            ])
            
            rules['volume'].extend([
                ValidationRule(
                    'volume', 'range', {'min': 0, 'max': 10_000_000_000},
                    description="Volume must be non-negative and reasonable"
                )
            ])
        
        # Standard level rules
        if self.validation_level in [ValidationLevel.STANDARD, ValidationLevel.STRICT, ValidationLevel.COMPREHENSIVE]:
            rules['market_cap'].extend([
                ValidationRule(
                    'market_cap', 'range', {'min': 1_000_000, 'max': 10_000_000_000_000},
                    description="Market cap should be between $1M and $10T"
                )
            ])
            
            rules['pe_ratio'].extend([
                ValidationRule(
                    'pe_ratio', 'range', {'min': 0.1, 'max': 1000.0},
                    severity='warning',
                    description="P/E ratio should be reasonable"
                )
            ])
        
        # Strict level rules
        if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.COMPREHENSIVE]:
            rules['dividend_yield'].extend([
                ValidationRule(
                    'dividend_yield', 'range', {'min': 0.0, 'max': 50.0},
                    description="Dividend yield should be between 0% and 50%"
                )
            ])
            
            rules['beta'].extend([
                ValidationRule(
                    'beta', 'range', {'min': -3.0, 'max': 5.0},
                    severity='warning',
                    description="Beta should typically be between -3 and 5"
                )
            ])
        
        # Comprehensive level rules
        if self.validation_level == ValidationLevel.COMPREHENSIVE:
            # Add cross-field validation rules
            rules['consistency_checks'] = [
                ValidationRule(
                    'price_consistency', 'custom', {},
                    description="Price fields should be consistent",
                    validation_function=self._validate_price_consistency
                ),
                ValidationRule(
                    'financial_ratios', 'custom', {},
                    description="Financial ratios should be reasonable",
                    validation_function=self._validate_financial_ratios
                )
            ]
        
        return rules
    
    def _initialize_market_ranges(self) -> Dict[str, Dict[str, float]]:
        """Initialize reasonable ranges for market data"""
        return {
            'price_ranges': {
                'penny_stock_max': 5.0,
                'typical_max': 1000.0,
                'extreme_max': 10000.0
            },
            'volume_ranges': {
                'illiquid_min': 1000,
                'typical_min': 10000,
                'high_volume': 1000000
            },
            'market_cap_ranges': {
                'micro_cap': 300_000_000,
                'small_cap': 2_000_000_000,
                'mid_cap': 10_000_000_000,
                'large_cap': 200_000_000_000
            }
        }
    
    async def validate_stock_data(self, data: Dict[str, Any], ticker: str = None) -> DataQualityScore:
        """Validate and score stock data quality"""
        if ticker:
            data['ticker'] = ticker
        
        issues = []
        total_fields = 0
        valid_fields = 0
        
        # Track all fields for statistics
        self._update_field_statistics(data)
        
        # Run field-level validations
        for field_name, field_rules in self.validation_rules.items():
            if field_name == 'consistency_checks':
                continue  # Handle separately
            
            total_fields += 1
            field_value = data.get(field_name)
            field_valid = True
            
            for rule in field_rules:
                validation_result = await self._apply_validation_rule(rule, field_value, data)
                if validation_result:
                    issues.append(validation_result)
                    if validation_result.severity == 'error':
                        field_valid = False
            
            if field_valid:
                valid_fields += 1
        
        # Run cross-field validations for comprehensive level
        if self.validation_level == ValidationLevel.COMPREHENSIVE:
            consistency_issues = await self._validate_consistency(data)
            issues.extend(consistency_issues)
        
        # Calculate quality scores
        quality_score = self._calculate_quality_score(data, issues, total_fields, valid_fields)
        
        # Update common issues statistics
        for issue in issues:
            self.common_issues[issue.issue_type.value] += 1
        
        return quality_score
    
    async def _apply_validation_rule(self, rule: ValidationRule, field_value: Any, full_data: Dict) -> Optional[ValidationResult]:
        """Apply a single validation rule"""
        try:
            if rule.rule_type == 'required':
                if field_value is None or field_value == '':
                    return ValidationResult(
                        field_name=rule.field_name,
                        issue_type=DataQualityIssue.MISSING_REQUIRED,
                        severity=rule.severity,
                        message=f"Required field '{rule.field_name}' is missing",
                        original_value=field_value
                    )
            
            elif rule.rule_type == 'range':
                if field_value is not None:
                    try:
                        numeric_value = float(field_value)
                        min_val = rule.parameters.get('min', float('-inf'))
                        max_val = rule.parameters.get('max', float('inf'))
                        
                        if not (min_val <= numeric_value <= max_val):
                            return ValidationResult(
                                field_name=rule.field_name,
                                issue_type=DataQualityIssue.OUT_OF_RANGE,
                                severity=rule.severity,
                                message=f"{rule.field_name} value {numeric_value} is outside expected range [{min_val}, {max_val}]",
                                original_value=field_value,
                                suggested_fix=f"Value should be between {min_val} and {max_val}"
                            )
                    except (ValueError, TypeError):
                        return ValidationResult(
                            field_name=rule.field_name,
                            issue_type=DataQualityIssue.INVALID_FORMAT,
                            severity='error',
                            message=f"{rule.field_name} is not a valid number: {field_value}",
                            original_value=field_value
                        )
            
            elif rule.rule_type == 'format':
                if field_value is not None:
                    pattern = rule.parameters.get('pattern')
                    if pattern and not re.match(pattern, str(field_value)):
                        return ValidationResult(
                            field_name=rule.field_name,
                            issue_type=DataQualityIssue.INVALID_FORMAT,
                            severity=rule.severity,
                            message=f"{rule.field_name} format is invalid: {field_value}",
                            original_value=field_value
                        )
            
            elif rule.rule_type == 'custom' and rule.validation_function:
                return await rule.validation_function(full_data)
            
        except Exception as e:
            logger.error(f"Error applying validation rule {rule.field_name}: {e}")
        
        return None
    
    async def _validate_consistency(self, data: Dict[str, Any]) -> List[ValidationResult]:
        """Run cross-field consistency validations"""
        issues = []
        
        # Price consistency checks
        price_issue = await self._validate_price_consistency(data)
        if price_issue:
            issues.append(price_issue)
        
        # Financial ratios consistency
        ratio_issue = await self._validate_financial_ratios(data)
        if ratio_issue:
            issues.append(ratio_issue)
        
        # Volume vs market cap consistency
        volume_issue = await self._validate_volume_market_cap_consistency(data)
        if volume_issue:
            issues.append(volume_issue)
        
        return issues
    
    async def _validate_price_consistency(self, data: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate price field consistency"""
        try:
            current_price = data.get('current_price')
            previous_close = data.get('previous_close')
            day_high = data.get('day_high')
            day_low = data.get('day_low')
            
            if current_price is None:
                return None
            
            current_price = float(current_price)
            
            # Check if current price is within day's range
            if day_high is not None and day_low is not None:
                day_high = float(day_high)
                day_low = float(day_low)
                
                if not (day_low <= current_price <= day_high):
                    return ValidationResult(
                        field_name='price_consistency',
                        issue_type=DataQualityIssue.INCONSISTENT,
                        severity='warning',
                        message=f"Current price {current_price} is outside day range [{day_low}, {day_high}]",
                        original_value={'current': current_price, 'high': day_high, 'low': day_low}
                    )
                
                # Check if day range is reasonable
                if day_high <= day_low:
                    return ValidationResult(
                        field_name='day_range',
                        issue_type=DataQualityIssue.INCONSISTENT,
                        severity='error',
                        message=f"Day high ({day_high}) should be greater than day low ({day_low})",
                        original_value={'high': day_high, 'low': day_low}
                    )
            
            # Check for extreme price movements
            if previous_close is not None:
                previous_close = float(previous_close)
                price_change_percent = abs((current_price - previous_close) / previous_close) * 100
                
                if price_change_percent > 50:  # More than 50% change
                    return ValidationResult(
                        field_name='price_movement',
                        issue_type=DataQualityIssue.SUSPICIOUS,
                        severity='warning',
                        message=f"Large price movement: {price_change_percent:.1f}% from previous close",
                        original_value={'current': current_price, 'previous': previous_close}
                    )
        
        except (ValueError, TypeError, ZeroDivisionError) as e:
            return ValidationResult(
                field_name='price_consistency',
                issue_type=DataQualityIssue.INVALID_FORMAT,
                severity='error',
                message=f"Error validating price consistency: {e}",
                original_value=data
            )
        
        return None
    
    async def _validate_financial_ratios(self, data: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate financial ratio consistency"""
        try:
            pe_ratio = data.get('pe_ratio')
            eps = data.get('eps')
            current_price = data.get('current_price')
            
            # Check P/E ratio consistency with EPS and price
            if all(v is not None for v in [pe_ratio, eps, current_price]):
                pe_ratio = float(pe_ratio)
                eps = float(eps)
                current_price = float(current_price)
                
                if eps != 0:
                    calculated_pe = current_price / eps
                    pe_difference = abs(calculated_pe - pe_ratio) / pe_ratio if pe_ratio != 0 else float('inf')
                    
                    if pe_difference > 0.1:  # More than 10% difference
                        return ValidationResult(
                            field_name='pe_ratio_consistency',
                            issue_type=DataQualityIssue.INCONSISTENT,
                            severity='warning',
                            message=f"P/E ratio ({pe_ratio}) doesn't match price/EPS calculation ({calculated_pe:.2f})",
                            original_value={'pe_ratio': pe_ratio, 'calculated_pe': calculated_pe},
                            suggested_fix=f"P/E ratio should be approximately {calculated_pe:.2f}"
                        )
        
        except (ValueError, TypeError, ZeroDivisionError):
            # Not a validation error, just missing or invalid data
            pass
        
        return None
    
    async def _validate_volume_market_cap_consistency(self, data: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate volume vs market cap consistency"""
        try:
            volume = data.get('volume')
            market_cap = data.get('market_cap')
            current_price = data.get('current_price')
            
            if all(v is not None for v in [volume, market_cap, current_price]):
                volume = float(volume)
                market_cap = float(market_cap)
                current_price = float(current_price)
                
                # Calculate implied shares outstanding
                shares_outstanding = market_cap / current_price
                
                # Check if volume is reasonable relative to shares outstanding
                volume_turnover = volume / shares_outstanding if shares_outstanding > 0 else 0
                
                if volume_turnover > 1.0:  # More than 100% turnover in one day
                    return ValidationResult(
                        field_name='volume_consistency',
                        issue_type=DataQualityIssue.SUSPICIOUS,
                        severity='warning',
                        message=f"Very high volume turnover: {volume_turnover:.1%} of shares outstanding",
                        original_value={'volume': volume, 'turnover': volume_turnover}
                    )
        
        except (ValueError, TypeError, ZeroDivisionError):
            pass
        
        return None
    
    def _calculate_quality_score(self, data: Dict, issues: List[ValidationResult], total_fields: int, valid_fields: int) -> DataQualityScore:
        """Calculate comprehensive data quality score"""
        
        # Base completeness score
        completeness_score = int((valid_fields / total_fields) * 100) if total_fields > 0 else 0
        
        # Count issues by severity
        error_count = len([i for i in issues if i.severity == 'error'])
        warning_count = len([i for i in issues if i.severity == 'warning'])
        
        # Accuracy score (based on error count)
        accuracy_score = max(0, 100 - (error_count * 20) - (warning_count * 10))
        
        # Consistency score (based on consistency issues)
        consistency_issues = len([i for i in issues if i.issue_type == DataQualityIssue.INCONSISTENT])
        consistency_score = max(0, 100 - (consistency_issues * 15))
        
        # Timeliness score (check if data is fresh)
        timeliness_score = self._calculate_timeliness_score(data)
        
        # Overall score (weighted average)
        overall_score = int(
            (completeness_score * 0.3) +
            (accuracy_score * 0.3) +
            (consistency_score * 0.2) +
            (timeliness_score * 0.2)
        )
        
        return DataQualityScore(
            overall_score=overall_score,
            completeness_score=completeness_score,
            accuracy_score=accuracy_score,
            consistency_score=consistency_score,
            timeliness_score=timeliness_score,
            issues=issues,
            total_fields_checked=total_fields,
            valid_fields=valid_fields
        )
    
    def _calculate_timeliness_score(self, data: Dict) -> int:
        """Calculate timeliness score based on data freshness"""
        timestamp_fields = ['timestamp', 'last_updated', 'extraction_timestamp']
        
        for field in timestamp_fields:
            if field in data:
                try:
                    if isinstance(data[field], str):
                        timestamp = datetime.fromisoformat(data[field].replace('Z', '+00:00'))
                    else:
                        timestamp = data[field]
                    
                    age_hours = (datetime.now() - timestamp).total_seconds() / 3600
                    
                    if age_hours <= 1:
                        return 100
                    elif age_hours <= 24:
                        return max(50, 100 - int(age_hours * 2))
                    elif age_hours <= 168:  # 1 week
                        return max(20, 50 - int((age_hours - 24) / 24 * 5))
                    else:
                        return 20
                    
                except (ValueError, TypeError, AttributeError):
                    continue
        
        # No timestamp found
        return 50
    
    async def clean_and_correct_data(self, data: Dict[str, Any], validation_result: DataQualityScore) -> Dict[str, Any]:
        """Clean and correct data based on validation results"""
        if not self.cleaning_enabled:
            return data
        
        cleaned_data = data.copy()
        corrections_made = []
        
        for issue in validation_result.issues:
            correction = await self._apply_correction(cleaned_data, issue)
            if correction:
                corrections_made.append(correction)
        
        # Log corrections
        if corrections_made:
            logger.info(f"Applied {len(corrections_made)} data corrections for {data.get('ticker', 'unknown')}")
            for correction in corrections_made:
                logger.debug(f"Correction: {correction}")
        
        return cleaned_data
    
    async def _apply_correction(self, data: Dict[str, Any], issue: ValidationResult) -> Optional[str]:
        """Apply automatic correction for a validation issue"""
        try:
            if issue.issue_type == DataQualityIssue.INVALID_FORMAT:
                return await self._fix_format_issue(data, issue)
            
            elif issue.issue_type == DataQualityIssue.OUT_OF_RANGE:
                return await self._fix_range_issue(data, issue)
            
            elif issue.issue_type == DataQualityIssue.MISSING_REQUIRED:
                return await self._fix_missing_data(data, issue)
            
            elif issue.issue_type == DataQualityIssue.SUSPICIOUS:
                return await self._handle_suspicious_data(data, issue)
        
        except Exception as e:
            logger.warning(f"Error applying correction for {issue.field_name}: {e}")
        
        return None
    
    async def _fix_format_issue(self, data: Dict[str, Any], issue: ValidationResult) -> Optional[str]:
        """Fix format-related issues"""
        field_name = issue.field_name
        original_value = issue.original_value
        
        if field_name == 'ticker':
            # Clean up ticker symbol
            if isinstance(original_value, str):
                cleaned_ticker = re.sub(r'[^A-Z]', '', original_value.upper())
                if cleaned_ticker and len(cleaned_ticker) <= 5:
                    data[field_name] = cleaned_ticker
                    return f"Cleaned ticker from '{original_value}' to '{cleaned_ticker}'"
        
        elif field_name in ['current_price', 'volume', 'market_cap']:
            # Try to extract numeric value from string
            if isinstance(original_value, str):
                # Remove common formatting characters
                cleaned_str = re.sub(r'[,$%\s]', '', original_value)
                try:
                    numeric_value = float(cleaned_str)
                    data[field_name] = numeric_value
                    return f"Converted {field_name} from '{original_value}' to {numeric_value}"
                except ValueError:
                    pass
        
        return None
    
    async def _fix_range_issue(self, data: Dict[str, Any], issue: ValidationResult) -> Optional[str]:
        """Fix out-of-range issues"""
        field_name = issue.field_name
        original_value = issue.original_value
        
        # For price fields, check if it might be in cents instead of dollars
        if field_name == 'current_price' and isinstance(original_value, (int, float)):
            if original_value > 1000:  # Likely in cents
                corrected_price = original_value / 100
                if 0.01 <= corrected_price <= 1000:
                    data[field_name] = corrected_price
                    return f"Converted price from cents: {original_value} -> ${corrected_price}"
        
        # For volume, check for unit confusion (millions, thousands)
        elif field_name == 'volume' and isinstance(original_value, (int, float)):
            if original_value < 1000 and original_value > 0:  # Might be in millions
                corrected_volume = int(original_value * 1_000_000)
                data[field_name] = corrected_volume
                return f"Converted volume from millions: {original_value} -> {corrected_volume}"
        
        return None
    
    async def _fix_missing_data(self, data: Dict[str, Any], issue: ValidationResult) -> Optional[str]:
        """Attempt to derive missing required data"""
        field_name = issue.field_name
        
        # Try to derive missing data from other fields
        if field_name == 'ticker' and 'symbol' in data:
            data['ticker'] = data['symbol']
            return "Derived ticker from symbol field"
        
        elif field_name == 'current_price':
            # Try to get from other price fields
            for price_field in ['close', 'last_price', 'price']:
                if price_field in data and data[price_field] is not None:
                    data['current_price'] = data[price_field]
                    return f"Used {price_field} as current_price"
        
        return None
    
    async def _handle_suspicious_data(self, data: Dict[str, Any], issue: ValidationResult) -> Optional[str]:
        """Handle suspicious but potentially valid data"""
        # For now, just flag suspicious data but don't automatically correct
        # This could be enhanced to use historical data for validation
        logger.warning(f"Suspicious data flagged for {data.get('ticker', 'unknown')}: {issue.message}")
        return f"Flagged suspicious data: {issue.message}"
    
    def _update_field_statistics(self, data: Dict[str, Any]):
        """Update field statistics for adaptive validation"""
        for field_name, value in data.items():
            if isinstance(value, (int, float)) and not math.isnan(value):
                self.field_statistics[field_name].append(value)
                
                # Keep only recent statistics (last 1000 values)
                if len(self.field_statistics[field_name]) > 1000:
                    self.field_statistics[field_name] = self.field_statistics[field_name][-1000:]
    
    def get_validation_statistics(self) -> Dict:
        """Get validation statistics and insights"""
        field_stats = {}
        for field_name, values in self.field_statistics.items():
            if values:
                field_stats[field_name] = {
                    'count': len(values),
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'min': min(values),
                    'max': max(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0
                }
        
        return {
            'validation_level': self.validation_level.value,
            'total_validations': sum(self.common_issues.values()),
            'common_issues': dict(self.common_issues),
            'field_statistics': field_stats,
            'rules_count': sum(len(rules) for rules in self.validation_rules.values())
        }

# Batch validation functions
async def validate_batch_data(data_list: List[Dict[str, Any]], 
                            validation_level: ValidationLevel = ValidationLevel.STANDARD,
                            max_concurrent: int = 50) -> List[DataQualityScore]:
    """Validate a batch of stock data concurrently"""
    validator = FinancialDataValidator(validation_level)
    
    # Create semaphore to limit concurrent validations
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def validate_single(data: Dict[str, Any]) -> DataQualityScore:
        async with semaphore:
            return await validator.validate_stock_data(data)
    
    # Run validations concurrently
    tasks = [validate_single(data) for data in data_list]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle exceptions
    valid_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Validation failed for item {i}: {result}")
            # Create a failing validation result
            valid_results.append(DataQualityScore(
                overall_score=0,
                completeness_score=0,
                accuracy_score=0,
                consistency_score=0,
                timeliness_score=0,
                issues=[ValidationResult(
                    field_name='validation_error',
                    issue_type=DataQualityIssue.INVALID_FORMAT,
                    severity='error',
                    message=f"Validation failed: {result}"
                )]
            ))
        else:
            valid_results.append(result)
    
    return valid_results

def validate_extraction_results(results: List[Dict], 
                               validation_level: ValidationLevel = ValidationLevel.STANDARD,
                               min_quality_score: float = 70.0) -> Dict:
    """Validate and filter extraction results synchronously"""
    
    async def async_validate():
        # Validate all results
        validation_results = await validate_batch_data(results, validation_level)
        
        # Filter by quality score
        filtered_results = []
        quality_scores = []
        
        for data, validation in zip(results, validation_results):
            quality_scores.append(validation.overall_score)
            
            if validation.overall_score >= min_quality_score:
                filtered_results.append({
                    'data': data,
                    'quality_score': validation.overall_score,
                    'validation': validation
                })
        
        # Calculate summary statistics
        total_records = len(results)
        valid_records = len(filtered_results)
        avg_quality_score = statistics.mean(quality_scores) if quality_scores else 0
        
        # Collect common issues
        all_issues = []
        for validation in validation_results:
            all_issues.extend(validation.issues)
        
        issue_counts = Counter()
        for issue in all_issues:
            issue_counts[issue.issue_type.value] += 1
        
        return {
            'total_records': total_records,
            'valid_records': valid_records,
            'filtered_records': total_records - valid_records,
            'pass_rate': valid_records / total_records if total_records > 0 else 0,
            'avg_quality_score': avg_quality_score,
            'min_quality_score': min_quality_score,
            'quality_scores': quality_scores,
            'filtered_results': filtered_results,
            'common_issues': dict(issue_counts),
            'validation_level': validation_level.value
        }
    
    # Run async validation in sync context
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(async_validate())

# Testing and example usage
async def test_data_validation():
    """Test the data validation pipeline"""
    
    # Sample test data
    test_data = [
        {
            'ticker': 'AAPL',
            'current_price': 150.25,
            'previous_close': 148.50,
            'day_high': 152.00,
            'day_low': 147.80,
            'volume': 50000000,
            'market_cap': 2500000000000,
            'pe_ratio': 25.5,
            'eps': 5.89,
            'timestamp': datetime.now().isoformat()
        },
        {
            'ticker': 'INVALID_TICKER_SYMBOL',  # Invalid ticker
            'current_price': -10.50,  # Invalid price
            'volume': 'not_a_number',  # Invalid format
            'market_cap': 50000000000000,  # Too high
            'day_high': 100.0,
            'day_low': 120.0,  # Inconsistent (high < low)
        },
        {
            'ticker': 'MSFT',
            'current_price': 325000,  # Likely in cents
            'volume': 25,  # Likely in millions
            'pe_ratio': 28.5,
            'eps': 11.05,
        },
        {
            # Missing required fields
            'some_field': 'some_value',
            'volume': 1000000
        }
    ]
    
    logger.info("Testing data validation pipeline...")
    
    # Test different validation levels
    for level in [ValidationLevel.BASIC, ValidationLevel.STANDARD, ValidationLevel.COMPREHENSIVE]:
        logger.info(f"\nTesting {level.value} validation level:")
        
        validator = FinancialDataValidator(level)
        
        for i, data in enumerate(test_data):
            logger.info(f"  Validating test data {i+1}...")
            
            # Validate
            quality_score = await validator.validate_stock_data(data)
            
            # Clean if needed
            cleaned_data = await validator.clean_and_correct_data(data, quality_score)
            
            logger.info(f"    Original score: {quality_score.overall_score}/100")
            logger.info(f"    Issues found: {len(quality_score.issues)}")
            
            for issue in quality_score.issues[:3]:  # Show first 3 issues
                logger.info(f"      - {issue.severity}: {issue.message}")
            
            if quality_score.issues:
                logger.info(f"      (... and {max(0, len(quality_score.issues)-3)} more issues)")
    
    # Test batch validation
    logger.info("\nTesting batch validation...")
    
    batch_results = validate_extraction_results(
        test_data, 
        ValidationLevel.STANDARD, 
        min_quality_score=50.0
    )
    
    logger.info(f"Batch validation results:")
    logger.info(f"  Total records: {batch_results['total_records']}")
    logger.info(f"  Valid records: {batch_results['valid_records']}")
    logger.info(f"  Pass rate: {batch_results['pass_rate']:.1%}")
    logger.info(f"  Average quality score: {batch_results['avg_quality_score']:.1f}")
    logger.info(f"  Common issues: {batch_results['common_issues']}")
    
    logger.info("Data validation pipeline test completed!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    asyncio.run(test_data_validation())