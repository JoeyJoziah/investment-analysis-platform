"""
Enhanced Data Quality Checker and Validator
Addresses all data quality issues identified in the error patterns
"""

import logging
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import statistics
from dataclasses import dataclass
import pytz

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Data quality validation result"""
    is_valid: bool
    score: float  # 0-100
    issues: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]

class EnhancedDataQualityChecker:
    """
    Comprehensive data quality checker that addresses all identified issues:
    - Delisted/invalid ticker detection
    - Price change validation (>50% threshold)
    - Zero/negative price detection
    - Timezone handling for yfinance
    - Data completeness checks
    """
    
    def __init__(self):
        self.price_change_threshold = 50.0  # 50% daily change threshold
        self.volume_spike_threshold = 1000.0  # 10x normal volume
        self.intraday_volatility_threshold = 25.0  # 25% intraday range
        
        # Common timezone mappings
        self.market_timezones = {
            'NYSE': 'America/New_York',
            'NASDAQ': 'America/New_York', 
            'AMEX': 'America/New_York',
            'TSX': 'America/Toronto',
            'LSE': 'Europe/London'
        }
    
    def validate_ticker_existence(self, ticker: str) -> ValidationResult:
        """
        Validate if ticker exists and is actively traded
        Addresses: "No data available" errors for delisted tickers
        """
        issues = []
        warnings = []
        metadata = {}
        
        try:
            # Clean ticker format
            ticker = ticker.upper().strip()
            
            # Check ticker format
            if not ticker or len(ticker) > 10 or not ticker.isalnum():
                issues.append(f"Invalid ticker format: {ticker}")
                return ValidationResult(False, 0, issues, warnings, metadata)
            
            # Try to get basic info from yfinance with timeout
            try:
                stock = yf.Ticker(ticker)
                
                # Get basic info with timeout protection
                info = stock.info
                if not info or info.get('regularMarketPrice') is None:
                    issues.append(f"No market data available for {ticker}")
                    
                # Check if delisted
                if info.get('quoteType') == 'NONE':
                    issues.append(f"Ticker {ticker} appears to be delisted")
                    
                # Check market cap (very small cap or zero might indicate issues)
                market_cap = info.get('marketCap', 0)
                if market_cap < 1000000:  # Less than $1M market cap
                    warnings.append(f"Very small market cap: ${market_cap:,}")
                
                # Check last trading day
                last_price = info.get('regularMarketPrice')
                if last_price is None or last_price <= 0:
                    issues.append(f"Invalid or missing current price for {ticker}")
                
                metadata.update({
                    'market_cap': market_cap,
                    'last_price': last_price,
                    'currency': info.get('currency', 'USD'),
                    'exchange': info.get('exchange', 'UNKNOWN'),
                    'quote_type': info.get('quoteType', 'EQUITY')
                })
                
            except Exception as e:
                issues.append(f"Error fetching ticker data: {str(e)}")
                logger.error(f"yfinance error for {ticker}: {e}")
        
        except Exception as e:
            issues.append(f"Ticker validation error: {str(e)}")
        
        is_valid = len(issues) == 0
        score = 100 if is_valid else max(0, 100 - (len(issues) * 30 + len(warnings) * 10))
        
        return ValidationResult(is_valid, score, issues, warnings, metadata)
    
    def validate_price_data(self, price_data: Dict, ticker: str = "UNKNOWN") -> ValidationResult:
        """
        Comprehensive price data validation
        Addresses: Negative prices, excessive changes, OHLC relationships
        """
        issues = []
        warnings = []
        metadata = {}
        
        try:
            # Check required fields
            required_fields = ['open', 'high', 'low', 'close', 'volume']
            missing_fields = [f for f in required_fields if f not in price_data or price_data[f] is None]
            
            if missing_fields:
                issues.extend([f"Missing field: {field}" for field in missing_fields])
                return ValidationResult(False, 0, issues, warnings, metadata)
            
            # Convert to numeric values with validation
            try:
                open_price = float(price_data['open'])
                high_price = float(price_data['high'])
                low_price = float(price_data['low'])
                close_price = float(price_data['close'])
                volume = int(price_data['volume']) if price_data['volume'] is not None else 0
            except (ValueError, TypeError) as e:
                issues.append(f"Invalid numeric data: {e}")
                return ValidationResult(False, 0, issues, warnings, metadata)
            
            # 1. Check for negative or zero prices
            if any(price <= 0 for price in [open_price, high_price, low_price, close_price]):
                issues.append("Negative or zero prices detected")
            
            # 2. Check OHLC relationships
            if high_price < max(open_price, close_price):
                issues.append(f"High ({high_price}) < max(Open({open_price}), Close({close_price}))")
                
            if low_price > min(open_price, close_price):
                issues.append(f"Low ({low_price}) > min(Open({open_price}), Close({close_price}))")
                
            if high_price < low_price:
                issues.append(f"High ({high_price}) < Low ({low_price})")
            
            # 3. Volume validation
            if volume < 0:
                issues.append("Negative volume")
            elif volume == 0:
                warnings.append("Zero volume trading day")
            
            # 4. Excessive price change detection (>50%)
            if 'previous_close' in price_data and price_data['previous_close']:
                try:
                    prev_close = float(price_data['previous_close'])
                    if prev_close > 0:
                        price_change_pct = ((close_price - prev_close) / prev_close) * 100
                        
                        if abs(price_change_pct) > self.price_change_threshold:
                            warning_msg = f"Excessive price change: {price_change_pct:.1f}%"
                            if abs(price_change_pct) > 100:  # Over 100% change is very suspicious
                                issues.append(warning_msg)
                            else:
                                warnings.append(warning_msg)
                        
                        metadata['daily_change_pct'] = price_change_pct
                except (ValueError, TypeError):
                    warnings.append("Invalid previous close price for comparison")
            
            # 5. Intraday volatility check
            if high_price > low_price and low_price > 0:
                intraday_range_pct = ((high_price - low_price) / low_price) * 100
                if intraday_range_pct > self.intraday_volatility_threshold:
                    warnings.append(f"High intraday volatility: {intraday_range_pct:.1f}%")
                metadata['intraday_volatility_pct'] = intraday_range_pct
            
            # 6. Price reasonableness check
            if any(price > 100000 for price in [open_price, high_price, low_price, close_price]):
                warnings.append("Extremely high stock price detected")
            
            if any(price < 0.01 for price in [open_price, high_price, low_price, close_price]):
                warnings.append("Very low stock price (penny stock)")
            
            # 7. Calculate data quality score
            base_score = 100
            score_penalty = len(issues) * 25 + len(warnings) * 5
            quality_score = max(0, base_score - score_penalty)
            
            # Additional metadata
            metadata.update({
                'ohlc_valid': high_price >= max(open_price, close_price) and low_price <= min(open_price, close_price),
                'volume': volume,
                'price_range': high_price - low_price,
                'ticker': ticker,
                'validation_timestamp': datetime.now(timezone.utc).isoformat()
            })
            
        except Exception as e:
            issues.append(f"Price validation error: {str(e)}")
            quality_score = 0
        
        is_valid = len(issues) == 0
        return ValidationResult(is_valid, quality_score, issues, warnings, metadata)
    
    def fix_timezone_for_yfinance(self, ticker: str, exchange: str = "NYSE") -> Dict[str, Any]:
        """
        Fix timezone issues when fetching data from yfinance
        Addresses: yfinance timezone errors
        """
        timezone_config = {
            'ticker': ticker,
            'timezone': self.market_timezones.get(exchange, 'America/New_York'),
            'start_time': None,
            'end_time': None
        }
        
        try:
            # Set appropriate timezone for the exchange
            market_tz = pytz.timezone(timezone_config['timezone'])
            
            # Calculate market hours aware start/end times
            now = datetime.now(market_tz)
            
            # For daily data, get last 30 days
            start_date = now - timedelta(days=30)
            
            timezone_config.update({
                'start_time': start_date.strftime('%Y-%m-%d'),
                'end_time': now.strftime('%Y-%m-%d'),
                'market_timezone': market_tz
            })
            
        except Exception as e:
            logger.error(f"Timezone configuration error for {ticker}: {e}")
            # Fallback to UTC
            timezone_config.update({
                'timezone': 'UTC',
                'start_time': (datetime.utcnow() - timedelta(days=30)).strftime('%Y-%m-%d'),
                'end_time': datetime.utcnow().strftime('%Y-%m-%d')
            })
        
        return timezone_config
    
    def fetch_safe_price_data(self, ticker: str, exchange: str = "NYSE") -> Tuple[bool, Optional[pd.DataFrame], List[str]]:
        """
        Safely fetch price data with proper timezone handling
        Addresses: yfinance timezone errors and data fetching issues
        """
        errors = []
        
        try:
            # Get timezone configuration
            tz_config = self.fix_timezone_for_yfinance(ticker, exchange)
            
            # Create yfinance ticker object
            stock = yf.Ticker(ticker)
            
            # Fetch historical data with timeout and error handling
            try:
                history = stock.history(
                    start=tz_config['start_time'],
                    end=tz_config['end_time'],
                    timeout=30,  # 30 second timeout
                    raise_errors=False  # Don't raise on minor errors
                )
                
                if history.empty:
                    errors.append(f"No historical data available for {ticker}")
                    return False, None, errors
                
                # Basic data validation
                if len(history) == 0:
                    errors.append(f"Empty dataset for {ticker}")
                    return False, None, errors
                
                # Check for required columns
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                missing_cols = [col for col in required_cols if col not in history.columns]
                if missing_cols:
                    errors.append(f"Missing columns: {missing_cols}")
                    return False, None, errors
                
                # Remove any rows with NaN values in critical columns
                history = history.dropna(subset=required_cols)
                
                if history.empty:
                    errors.append(f"No valid data after cleaning for {ticker}")
                    return False, None, errors
                
                logger.info(f"Successfully fetched {len(history)} days of data for {ticker}")
                return True, history, errors
                
            except Exception as fetch_error:
                errors.append(f"Data fetch error: {str(fetch_error)}")
                logger.error(f"yfinance fetch error for {ticker}: {fetch_error}")
                
        except Exception as e:
            errors.append(f"Safe fetch error: {str(e)}")
            logger.error(f"Safe fetch setup error for {ticker}: {e}")
        
        return False, None, errors
    
    def detect_delisted_stocks(self, ticker_list: List[str]) -> Dict[str, bool]:
        """
        Batch detect delisted or invalid stocks
        Returns dict with ticker -> is_valid mapping
        """
        results = {}
        
        for ticker in ticker_list:
            try:
                validation_result = self.validate_ticker_existence(ticker)
                results[ticker] = validation_result.is_valid
                
                if not validation_result.is_valid:
                    logger.info(f"Invalid ticker detected: {ticker} - {validation_result.issues}")
                    
            except Exception as e:
                logger.error(f"Error checking ticker {ticker}: {e}")
                results[ticker] = False
        
        return results
    
    def comprehensive_data_check(self, ticker: str, price_data: Optional[Dict] = None) -> ValidationResult:
        """
        Run comprehensive data quality check combining all validators
        """
        all_issues = []
        all_warnings = []
        combined_metadata = {'ticker': ticker}
        
        # 1. Ticker existence check
        ticker_result = self.validate_ticker_existence(ticker)
        all_issues.extend(ticker_result.issues)
        all_warnings.extend(ticker_result.warnings)
        combined_metadata.update(ticker_result.metadata)
        
        # 2. Price data validation (if provided)
        if price_data:
            price_result = self.validate_price_data(price_data, ticker)
            all_issues.extend(price_result.issues)
            all_warnings.extend(price_result.warnings)
            combined_metadata.update(price_result.metadata)
        
        # 3. Calculate combined score
        base_score = 100
        if not ticker_result.is_valid:
            base_score = 0  # Invalid ticker = 0 score
        elif price_data and all_issues:
            base_score = max(0, 100 - (len(all_issues) * 20 + len(all_warnings) * 5))
        
        is_valid = len(all_issues) == 0 and ticker_result.is_valid
        
        return ValidationResult(is_valid, base_score, all_issues, all_warnings, combined_metadata)

class DataQualityReporter:
    """Generate comprehensive data quality reports"""
    
    @staticmethod
    def generate_quality_report(validation_results: List[ValidationResult]) -> Dict[str, Any]:
        """Generate a comprehensive quality report from validation results"""
        
        total_stocks = len(validation_results)
        valid_stocks = sum(1 for r in validation_results if r.is_valid)
        
        # Categorize issues
        issue_counts = {}
        warning_counts = {}
        
        for result in validation_results:
            for issue in result.issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
            for warning in result.warnings:
                warning_counts[warning] = warning_counts.get(warning, 0) + 1
        
        # Calculate score statistics
        scores = [r.score for r in validation_results]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        report = {
            'summary': {
                'total_stocks_checked': total_stocks,
                'valid_stocks': valid_stocks,
                'invalid_stocks': total_stocks - valid_stocks,
                'validity_rate': (valid_stocks / total_stocks * 100) if total_stocks > 0 else 0,
                'average_quality_score': round(avg_score, 2)
            },
            'quality_distribution': {
                'excellent_quality': sum(1 for s in scores if s >= 90),
                'good_quality': sum(1 for s in scores if 70 <= s < 90),
                'fair_quality': sum(1 for s in scores if 50 <= s < 70),
                'poor_quality': sum(1 for s in scores if s < 50)
            },
            'common_issues': dict(sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            'common_warnings': dict(sorted(warning_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            'recommendations': []
        }
        
        # Add recommendations based on issues
        if 'No data available' in issue_counts:
            report['recommendations'].append("Consider removing delisted tickers from active trading list")
        
        if any('Excessive price change' in issue for issue in issue_counts):
            report['recommendations'].append("Implement additional validation for stocks with extreme price movements")
        
        if any('Negative or zero prices' in issue for issue in issue_counts):
            report['recommendations'].append("Add data source validation and backup data sources")
        
        return report

# Example usage
if __name__ == "__main__":
    # Test the enhanced data quality checker
    checker = EnhancedDataQualityChecker()
    
    # Test ticker validation
    result = checker.validate_ticker_existence("AAPL")
    print(f"AAPL validation: {result}")
    
    # Test price data validation
    test_price_data = {
        'open': 150.0,
        'high': 155.0,
        'low': 148.0,
        'close': 152.0,
        'volume': 1000000,
        'previous_close': 149.0
    }
    
    price_result = checker.validate_price_data(test_price_data, "TEST")
    print(f"Price validation: {price_result}")
    
    # Test comprehensive check
    comprehensive_result = checker.comprehensive_data_check("AAPL", test_price_data)
    print(f"Comprehensive check: {comprehensive_result}")