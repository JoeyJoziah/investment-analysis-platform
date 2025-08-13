#!/usr/bin/env python3
"""
Data Validation Script for Investment Analysis Platform

This script validates the loaded historical data and provides data quality reports.

Usage:
    python scripts/validate_data.py                    # Basic validation
    python scripts/validate_data.py --detailed        # Detailed validation
    python scripts/validate_data.py --fix-issues      # Attempt to fix data issues
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import func, and_, or_
from backend.models.database import Stock, PriceHistory, Exchange, Sector, APIUsage
from scripts.load_historical_data import DatabaseManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataValidator:
    """Comprehensive data validation for the investment platform"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.validation_results = {}
        self.issues_found = []
    
    def validate_database_structure(self, session) -> Dict:
        """Validate database structure and basic data integrity"""
        logger.info("Validating database structure...")
        
        results = {}
        
        # Check if tables exist and have data
        tables_to_check = [
            ('stocks', Stock),
            ('exchanges', Exchange),
            ('sectors', Sector),
            ('price_history', PriceHistory),
            ('api_usage', APIUsage)
        ]
        
        for table_name, model_class in tables_to_check:
            try:
                count = session.query(func.count(model_class.id)).scalar()
                results[f"{table_name}_count"] = count
                
                if count == 0:
                    self.issues_found.append(f"Table {table_name} is empty")
                
            except Exception as e:
                results[f"{table_name}_error"] = str(e)
                self.issues_found.append(f"Error querying {table_name}: {e}")
        
        return results
    
    def validate_price_data_quality(self, session) -> Dict:
        """Validate price data for quality issues"""
        logger.info("Validating price data quality...")
        
        results = {}
        
        # Check for invalid OHLC data
        invalid_ohlc = session.query(func.count(PriceHistory.id)).filter(
            or_(
                PriceHistory.high < PriceHistory.low,
                PriceHistory.high < PriceHistory.open,
                PriceHistory.high < PriceHistory.close,
                PriceHistory.low > PriceHistory.open,
                PriceHistory.low > PriceHistory.close,
                PriceHistory.open <= 0,
                PriceHistory.high <= 0,
                PriceHistory.low <= 0,
                PriceHistory.close <= 0
            )
        ).scalar()
        
        results['invalid_ohlc_records'] = invalid_ohlc
        if invalid_ohlc > 0:
            self.issues_found.append(f"Found {invalid_ohlc} records with invalid OHLC data")
        
        # Check for negative volumes
        negative_volume = session.query(func.count(PriceHistory.id)).filter(
            PriceHistory.volume < 0
        ).scalar()
        
        results['negative_volume_records'] = negative_volume
        if negative_volume > 0:
            self.issues_found.append(f"Found {negative_volume} records with negative volume")
        
        # Check for zero volumes (might be valid for some stocks)
        zero_volume = session.query(func.count(PriceHistory.id)).filter(
            PriceHistory.volume == 0
        ).scalar()
        
        results['zero_volume_records'] = zero_volume
        
        # Check for extreme price changes (more than 50% in a day)
        extreme_changes = session.query(func.count(PriceHistory.id)).filter(
            or_(
                (PriceHistory.high / PriceHistory.low) > 1.5,
                (PriceHistory.low / PriceHistory.high) > 1.5
            )
        ).scalar()
        
        results['extreme_price_changes'] = extreme_changes
        if extreme_changes > 50:  # More than 50 might indicate data issues
            self.issues_found.append(f"Found {extreme_changes} records with extreme price changes (>50%)")
        
        # Check for missing adjusted close prices
        missing_adj_close = session.query(func.count(PriceHistory.id)).filter(
            PriceHistory.adjusted_close.is_(None)
        ).scalar()
        
        results['missing_adjusted_close'] = missing_adj_close
        
        return results
    
    def validate_data_completeness(self, session) -> Dict:
        """Validate data completeness and coverage"""
        logger.info("Validating data completeness...")
        
        results = {}
        
        # Check stocks with price data
        stocks_with_data = session.query(func.count(Stock.id.distinct())).join(PriceHistory).scalar()
        total_stocks = session.query(func.count(Stock.id)).scalar()
        
        results['stocks_with_price_data'] = stocks_with_data
        results['total_stocks'] = total_stocks
        results['data_coverage_pct'] = (stocks_with_data / total_stocks * 100) if total_stocks > 0 else 0
        
        if results['data_coverage_pct'] < 80:
            self.issues_found.append(f"Low data coverage: only {results['data_coverage_pct']:.1f}% of stocks have price data")
        
        # Check date coverage
        date_range = session.query(
            func.min(PriceHistory.date),
            func.max(PriceHistory.date)
        ).first()
        
        if date_range[0] and date_range[1]:
            results['earliest_date'] = date_range[0].isoformat()
            results['latest_date'] = date_range[1].isoformat()
            
            days_covered = (date_range[1] - date_range[0]).days
            results['days_covered'] = days_covered
            
            # Check if we have recent data (within last 7 days)
            recent_cutoff = datetime.now() - timedelta(days=7)
            if date_range[1] < recent_cutoff:
                self.issues_found.append(f"Latest data is more than 7 days old: {date_range[1]}")
        
        # Check for gaps in data
        stocks_with_sparse_data = session.query(func.count(Stock.id.distinct())).join(PriceHistory).group_by(Stock.id).having(func.count(PriceHistory.id) < 200).scalar()
        
        results['stocks_with_sparse_data'] = stocks_with_sparse_data or 0
        
        return results
    
    def validate_stock_metadata(self, session) -> Dict:
        """Validate stock metadata completeness"""
        logger.info("Validating stock metadata...")
        
        results = {}
        
        # Check for stocks without names
        unnamed_stocks = session.query(func.count(Stock.id)).filter(
            or_(Stock.name.is_(None), Stock.name == '', Stock.name == Stock.ticker)
        ).scalar()
        
        results['unnamed_stocks'] = unnamed_stocks
        if unnamed_stocks > 0:
            self.issues_found.append(f"Found {unnamed_stocks} stocks without proper names")
        
        # Check for stocks without sector assignment
        unsectored_stocks = session.query(func.count(Stock.id)).filter(
            Stock.sector_id.is_(None)
        ).scalar()
        
        results['unsectored_stocks'] = unsectored_stocks
        
        # Check for inactive stocks
        inactive_stocks = session.query(func.count(Stock.id)).filter(
            Stock.is_active == False
        ).scalar()
        
        results['inactive_stocks'] = inactive_stocks
        
        return results
    
    def validate_api_usage(self, session) -> Dict:
        """Validate API usage tracking"""
        logger.info("Validating API usage tracking...")
        
        results = {}
        
        # Check API usage records
        total_api_calls = session.query(func.count(APIUsage.id)).scalar()
        results['total_api_calls'] = total_api_calls
        
        # Check success rate
        successful_calls = session.query(func.count(APIUsage.id)).filter(
            APIUsage.success == True
        ).scalar()
        
        results['successful_api_calls'] = successful_calls
        results['api_success_rate'] = (successful_calls / total_api_calls * 100) if total_api_calls > 0 else 0
        
        # Check providers
        providers = session.query(APIUsage.provider, func.count(APIUsage.id)).group_by(APIUsage.provider).all()
        results['api_providers'] = {provider: count for provider, count in providers}
        
        return results
    
    def get_data_quality_score(self) -> int:
        """Calculate overall data quality score (0-100)"""
        score = 100
        
        # Deduct points for issues
        score -= len(self.issues_found) * 5  # 5 points per issue
        
        # Check validation results
        if self.validation_results.get('data_coverage_pct', 0) < 50:
            score -= 20
        elif self.validation_results.get('data_coverage_pct', 0) < 80:
            score -= 10
        
        if self.validation_results.get('invalid_ohlc_records', 0) > 0:
            score -= 15
        
        if self.validation_results.get('api_success_rate', 100) < 80:
            score -= 10
        
        return max(0, score)
    
    def run_full_validation(self, detailed: bool = False) -> Dict:
        """Run complete data validation"""
        logger.info("Starting comprehensive data validation...")
        
        with self.db_manager.get_session() as session:
            # Basic structure validation
            structure_results = self.validate_database_structure(session)
            self.validation_results.update(structure_results)
            
            # Price data quality
            price_quality_results = self.validate_price_data_quality(session)
            self.validation_results.update(price_quality_results)
            
            # Data completeness
            completeness_results = self.validate_data_completeness(session)
            self.validation_results.update(completeness_results)
            
            # Stock metadata
            metadata_results = self.validate_stock_metadata(session)
            self.validation_results.update(metadata_results)
            
            # API usage
            api_results = self.validate_api_usage(session)
            self.validation_results.update(api_results)
            
            if detailed:
                # Additional detailed validations
                self._detailed_stock_analysis(session)
                self._detailed_price_analysis(session)
        
        # Calculate quality score
        self.validation_results['quality_score'] = self.get_data_quality_score()
        
        return self.validation_results
    
    def _detailed_stock_analysis(self, session):
        """Perform detailed stock-by-stock analysis"""
        logger.info("Performing detailed stock analysis...")
        
        # Get stocks with data issues
        problematic_stocks = session.query(
            Stock.ticker,
            Stock.name,
            func.count(PriceHistory.id).label('record_count'),
            func.min(PriceHistory.date).label('earliest_date'),
            func.max(PriceHistory.date).label('latest_date')
        ).join(PriceHistory).group_by(Stock.id, Stock.ticker, Stock.name).having(
            func.count(PriceHistory.id) < 100  # Less than 100 records
        ).limit(10).all()
        
        self.validation_results['problematic_stocks'] = [
            {
                'ticker': stock.ticker,
                'name': stock.name,
                'record_count': stock.record_count,
                'date_range': f"{stock.earliest_date} to {stock.latest_date}"
            }
            for stock in problematic_stocks
        ]
    
    def _detailed_price_analysis(self, session):
        """Perform detailed price data analysis"""
        logger.info("Performing detailed price analysis...")
        
        # Find stocks with suspicious price patterns
        suspicious_patterns = session.query(
            Stock.ticker,
            func.max(PriceHistory.high / PriceHistory.low).label('max_daily_range'),
            func.avg(PriceHistory.volume).label('avg_volume')
        ).join(PriceHistory).group_by(Stock.id, Stock.ticker).having(
            func.max(PriceHistory.high / PriceHistory.low) > 2.0  # More than 100% daily range
        ).limit(5).all()
        
        self.validation_results['suspicious_price_patterns'] = [
            {
                'ticker': pattern.ticker,
                'max_daily_range': f"{pattern.max_daily_range:.2f}x",
                'avg_volume': int(pattern.avg_volume or 0)
            }
            for pattern in suspicious_patterns
        ]
    
    def print_validation_report(self):
        """Print comprehensive validation report"""
        print("\n" + "="*80)
        print("INVESTMENT ANALYSIS PLATFORM - DATA VALIDATION REPORT")
        print("="*80)
        
        # Overall quality score
        quality_score = self.validation_results.get('quality_score', 0)
        print(f"\nOVERALL DATA QUALITY SCORE: {quality_score}/100")
        
        if quality_score >= 90:
            print("✅ EXCELLENT - Data quality is excellent")
        elif quality_score >= 75:
            print("✅ GOOD - Data quality is good")
        elif quality_score >= 60:
            print("⚠️  FAIR - Data quality needs attention")
        else:
            print("❌ POOR - Data quality issues need immediate attention")
        
        # Database structure
        print(f"\n📊 DATABASE STRUCTURE:")
        structure_keys = [k for k in self.validation_results.keys() if k.endswith('_count')]
        for key in structure_keys:
            table_name = key.replace('_count', '')
            count = self.validation_results[key]
            print(f"  {table_name.upper()}: {count:,} records")
        
        # Data coverage
        print(f"\n📈 DATA COVERAGE:")
        print(f"  Stocks with price data: {self.validation_results.get('stocks_with_price_data', 0)}")
        print(f"  Total stocks: {self.validation_results.get('total_stocks', 0)}")
        print(f"  Coverage percentage: {self.validation_results.get('data_coverage_pct', 0):.1f}%")
        
        if 'earliest_date' in self.validation_results:
            print(f"  Date range: {self.validation_results['earliest_date']} to {self.validation_results['latest_date']}")
            print(f"  Days covered: {self.validation_results.get('days_covered', 0):,}")
        
        # Data quality issues
        print(f"\n🔍 DATA QUALITY ISSUES:")
        quality_issues = [
            ('invalid_ohlc_records', 'Invalid OHLC records'),
            ('negative_volume_records', 'Negative volume records'),
            ('zero_volume_records', 'Zero volume records'),
            ('extreme_price_changes', 'Extreme price changes'),
            ('missing_adjusted_close', 'Missing adjusted close prices')
        ]
        
        for key, description in quality_issues:
            value = self.validation_results.get(key, 0)
            if value > 0:
                print(f"  {description}: {value:,}")
        
        # API usage
        print(f"\n🌐 API USAGE:")
        print(f"  Total API calls: {self.validation_results.get('total_api_calls', 0):,}")
        print(f"  Success rate: {self.validation_results.get('api_success_rate', 0):.1f}%")
        
        providers = self.validation_results.get('api_providers', {})
        if providers:
            print(f"  Providers used:")
            for provider, count in providers.items():
                print(f"    {provider}: {count:,} calls")
        
        # Issues found
        if self.issues_found:
            print(f"\n⚠️  ISSUES REQUIRING ATTENTION:")
            for i, issue in enumerate(self.issues_found, 1):
                print(f"  {i}. {issue}")
        
        # Detailed analysis
        if 'problematic_stocks' in self.validation_results:
            print(f"\n🔍 STOCKS WITH DATA ISSUES:")
            for stock in self.validation_results['problematic_stocks']:
                print(f"  {stock['ticker']} ({stock['name']}): {stock['record_count']} records, {stock['date_range']}")
        
        if 'suspicious_price_patterns' in self.validation_results:
            print(f"\n🚨 SUSPICIOUS PRICE PATTERNS:")
            for pattern in self.validation_results['suspicious_price_patterns']:
                print(f"  {pattern['ticker']}: Max daily range {pattern['max_daily_range']}, Avg volume {pattern['avg_volume']:,}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if quality_score < 60:
            print("  1. Review and fix critical data quality issues")
            print("  2. Reload data for problematic stocks")
        if self.validation_results.get('data_coverage_pct', 0) < 80:
            print("  3. Increase stock coverage by loading more symbols")
        if self.validation_results.get('api_success_rate', 100) < 90:
            print("  4. Investigate API failures and improve error handling")
        
        print("="*80)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Validate Investment Analysis Platform Data')
    parser.add_argument('--detailed', action='store_true', help='Perform detailed validation')
    parser.add_argument('--fix-issues', action='store_true', help='Attempt to fix data issues (not implemented)')
    
    args = parser.parse_args()
    
    try:
        # Create validator
        validator = DataValidator()
        
        # Run validation
        results = validator.run_full_validation(detailed=args.detailed)
        
        # Print report
        validator.print_validation_report()
        
        # Return appropriate exit code
        quality_score = results.get('quality_score', 0)
        if quality_score < 60:
            sys.exit(1)  # Poor quality
        elif quality_score < 80:
            sys.exit(2)  # Needs attention
        else:
            sys.exit(0)  # Good quality
            
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(3)


if __name__ == "__main__":
    main()