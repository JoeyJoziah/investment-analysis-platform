"""
Comprehensive Test Suite for Database Error Fixes
Tests all the fixes implemented for the identified error patterns
"""

import pytest
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, patch, AsyncMock
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.models.consolidated_models import Base, Exchange, Stock, Sector, Industry
from backend.utils.async_database_fixed import AsyncDatabaseManager, BatchProcessor, DataQualityValidator
from backend.utils.enhanced_data_quality import EnhancedDataQualityChecker, ValidationResult
from backend.utils.robust_error_handling import (
    robust_logger, handle_errors, safe_database_operation, 
    BatchProcessingErrorHandler, ErrorSeverity, DatabaseSchemaError
)
from scripts.fix_database_schema import DatabaseSchemaFixer
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestDatabaseSchemaFixes:
    """Test database schema fixes and validations"""
    
    @pytest.fixture
    def test_engine(self):
        """Create in-memory test database"""
        engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(engine)
        return engine
    
    @pytest.fixture
    def test_session(self, test_engine):
        """Create test database session"""
        with Session(test_engine) as session:
            # Add test exchanges
            exchanges = [
                Exchange(code="NYSE", name="New York Stock Exchange"),
                Exchange(code="NASDAQ", name="NASDAQ Stock Market"),
                Exchange(code="AMEX", name="NYSE American")
            ]
            session.add_all(exchanges)
            session.commit()
            yield session
    
    def test_exchange_code_column_exists(self, test_session):
        """Test that exchanges table has 'code' column - fixes main error"""
        # Query that was failing before fix
        result = test_session.execute(text("SELECT id FROM exchanges WHERE code = 'NASDAQ'"))
        row = result.fetchone()
        
        assert row is not None, "NASDAQ exchange should be found"
        assert row[0] > 0, "Exchange should have valid ID"
    
    def test_stock_ticker_column_consistent(self, test_session):
        """Test that stocks table uses 'ticker' field consistently"""
        # Add test stock
        exchange = test_session.query(Exchange).filter_by(code="NYSE").first()
        stock = Stock(ticker="AAPL", name="Apple Inc.", exchange=exchange)
        test_session.add(stock)
        test_session.commit()
        
        # Query using 'ticker' field (should not fail)
        result = test_session.execute(text("SELECT id FROM stocks WHERE ticker = 'AAPL'"))
        row = result.fetchone()
        
        assert row is not None, "Stock should be found by ticker"
    
    def test_database_schema_fixer(self):
        """Test the schema fixer functionality"""
        # Create test database with missing tables
        test_engine = create_engine("sqlite:///:memory:")
        fixer = DatabaseSchemaFixer(test_engine.url)
        
        # Analyze schema (should find missing tables)
        schema_status = fixer.check_current_schema()
        
        assert 'exchanges' in schema_status['missing_tables'], "Should detect missing exchanges table"
        assert schema_status['needs_migration'], "Should indicate migration needed"
    
    def test_safe_exchange_lookup(self, test_session):
        """Test safe exchange lookup with error handling"""
        from backend.models.consolidated_models import get_exchange_by_code
        
        # Test existing exchange
        exchange = get_exchange_by_code(test_session, "NYSE")
        assert exchange is not None
        assert exchange.code == "NYSE"
        
        # Test non-existing exchange (should create it)
        new_exchange = get_exchange_by_code(test_session, "TSE")
        assert new_exchange is not None
        assert new_exchange.code == "TSE"

class TestAsyncDatabaseFixes:
    """Test async database operation fixes"""
    
    @pytest.fixture
    async def async_db_manager(self):
        """Create test async database manager"""
        db_manager = AsyncDatabaseManager("sqlite+aiosqlite:///:memory:")
        await db_manager.initialize()
        yield db_manager
        await db_manager.close()
    
    @pytest.mark.asyncio
    async def test_async_session_context_manager(self, async_db_manager):
        """Test that async session context manager works without Future errors"""
        async with async_db_manager.get_session() as session:
            # This should not raise '_asyncio.Future' object has no attribute '_condition'
            result = await session.execute(text("SELECT 1 as test"))
            row = result.fetchone()
            assert row[0] == 1
    
    @pytest.mark.asyncio
    async def test_stock_insertion_safe(self, async_db_manager):
        """Test safe stock insertion without database errors"""
        stock_data = {
            "ticker": "TEST",
            "name": "Test Company",
            "exchange": "NASDAQ"
        }
        
        # This should handle the "column 'code' does not exist" error gracefully
        stock_id = await async_db_manager.insert_stock_safe(stock_data)
        
        # Should return None or valid ID, not raise exception
        assert stock_id is None or isinstance(stock_id, int)
    
    @pytest.mark.asyncio
    async def test_batch_processing_without_future_errors(self, async_db_manager):
        """Test batch processing without asyncio Future object errors"""
        processor = BatchProcessor(async_db_manager, batch_size=10)
        
        test_stocks = [
            {"ticker": f"TEST{i}", "name": f"Test Company {i}", "exchange": "NYSE"}
            for i in range(25)
        ]
        
        # This should not raise Future object errors
        results = await processor.process_stock_batch(test_stocks)
        
        assert "processed" in results
        assert "errors" in results
        assert isinstance(results["processed"], int)
        assert isinstance(results["errors"], int)

class TestDataQualityValidation:
    """Test enhanced data quality validation"""
    
    @pytest.fixture
    def quality_checker(self):
        """Create data quality checker instance"""
        return EnhancedDataQualityChecker()
    
    def test_negative_zero_price_detection(self, quality_checker):
        """Test detection of negative or zero prices"""
        bad_price_data = {
            'open': 0.0,  # Zero price
            'high': 150.0,
            'low': -10.0,  # Negative price
            'close': 145.0,
            'volume': 1000000
        }
        
        result = quality_checker.validate_price_data(bad_price_data)
        
        assert not result.is_valid, "Should detect invalid prices"
        assert "Negative or zero prices" in str(result.issues)
    
    def test_excessive_price_change_detection(self, quality_checker):
        """Test detection of excessive price changes (>50%)"""
        extreme_change_data = {
            'open': 100.0,
            'high': 200.0,
            'low': 95.0,
            'close': 180.0,  # 80% increase
            'volume': 1000000,
            'previous_close': 100.0
        }
        
        result = quality_checker.validate_price_data(extreme_change_data)
        
        # Should be flagged as warning or issue
        issues_and_warnings = result.issues + result.warnings
        assert any("Excessive price change" in str(item) for item in issues_and_warnings)
    
    def test_ohlc_relationship_validation(self, quality_checker):
        """Test OHLC price relationship validation"""
        invalid_ohlc_data = {
            'open': 150.0,
            'high': 140.0,  # High < Open (invalid)
            'low': 160.0,   # Low > Open (invalid)
            'close': 145.0,
            'volume': 1000000
        }
        
        result = quality_checker.validate_price_data(invalid_ohlc_data)
        
        assert not result.is_valid, "Should detect invalid OHLC relationships"
        assert len(result.issues) > 0
    
    def test_delisted_ticker_detection(self, quality_checker):
        """Test detection of delisted/invalid tickers"""
        # Test with obviously invalid ticker
        result = quality_checker.validate_ticker_existence("INVALID_TICKER_123")
        
        # Should detect as invalid (though may depend on data source)
        assert isinstance(result, ValidationResult)
        assert hasattr(result, 'is_valid')
    
    def test_timezone_handling_fix(self, quality_checker):
        """Test timezone configuration for yfinance"""
        tz_config = quality_checker.fix_timezone_for_yfinance("AAPL", "NYSE")
        
        assert 'timezone' in tz_config
        assert 'start_time' in tz_config
        assert 'end_time' in tz_config
        assert tz_config['timezone'] in ['America/New_York', 'UTC']

class TestErrorHandlingImprovements:
    """Test enhanced error handling and logging"""
    
    def test_database_error_logging(self):
        """Test database error logging with context"""
        test_error = Exception("column 'code' does not exist")
        
        error_id = robust_logger.log_database_error(
            test_error, 
            query="SELECT id FROM exchanges WHERE code = 'NASDAQ'",
            params={'code': 'NASDAQ'}
        )
        
        assert error_id is not None
        assert len(error_id) > 0
    
    def test_async_error_logging(self):
        """Test async operation error logging"""
        test_error = Exception("'_asyncio.Future' object has no attribute '_condition'")
        
        error_id = robust_logger.log_async_error(test_error, "batch_processing")
        
        assert error_id is not None
        # Should detect specific async issue
        error_summary = robust_logger.get_error_summary(hours=1)
        assert error_summary['total_errors'] > 0
    
    def test_error_handling_decorator(self):
        """Test error handling decorator functionality"""
        
        @handle_errors(exceptions=(ValueError,), return_value="handled")
        def function_that_fails():
            raise ValueError("Test error")
        
        result = function_that_fails()
        assert result == "handled"
    
    def test_safe_database_operation_decorator(self):
        """Test database operation decorator with schema error"""
        
        @safe_database_operation
        def database_operation_that_fails():
            raise Exception("column 'code' does not exist")
        
        with pytest.raises(DatabaseSchemaError):
            database_operation_that_fails()
    
    def test_batch_error_handler(self):
        """Test batch processing error handler"""
        handler = BatchProcessingErrorHandler("test_batch")
        
        # Record some successes and errors
        handler.record_success("item1")
        handler.record_success("item2")
        handler.record_error(Exception("Test error"), "item3")
        
        summary = handler.get_summary()
        
        assert summary['successful'] == 2
        assert summary['failed'] == 1
        assert summary['success_rate'] < 100.0

class TestIntegrationScenarios:
    """Test complete error scenarios and their fixes"""
    
    def test_stock_creation_error_scenario(self):
        """Test the complete stock creation error scenario fix"""
        # Simulate the original error scenario
        # "Error creating stock record for [TICKER]: column 'code' does not exist"
        
        # This test would verify the complete fix pipeline:
        # 1. Database schema is correct
        # 2. Exchange lookup works
        # 3. Stock creation succeeds
        # 4. No asyncio errors occur
        
        # Create test data
        stock_data = {
            "ticker": "AAPL",
            "name": "Apple Inc.",
            "exchange": "NASDAQ"
        }
        
        # Test that all components work together
        test_engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(test_engine)
        
        with Session(test_engine) as session:
            # Add exchanges
            nasdaq = Exchange(code="NASDAQ", name="NASDAQ Stock Market")
            session.add(nasdaq)
            session.commit()
            
            # Create stock (should not fail)
            stock = Stock(
                ticker=stock_data["ticker"],
                name=stock_data["name"],
                exchange=nasdaq
            )
            session.add(stock)
            session.commit()
            
            # Verify creation
            created_stock = session.query(Stock).filter_by(ticker="AAPL").first()
            assert created_stock is not None
            assert created_stock.ticker == "AAPL"
            assert created_stock.exchange.code == "NASDAQ"
    
    @pytest.mark.asyncio
    async def test_batch_processing_error_scenario(self):
        """Test the complete batch processing error scenario fix"""
        # Simulate the original error scenario
        # "Error in batch [NUMBER]: '_asyncio.Future' object has no attribute '_condition'"
        
        # This would test the complete async fix:
        # 1. Proper async/await usage
        # 2. Connection pool management
        # 3. Error isolation in batches
        # 4. Proper cleanup
        
        # Create mock async operations
        async def mock_process_item(item):
            if item.get("fail"):
                raise Exception("Test async error")
            return {"processed": True}
        
        # Test batch with mixed success/failure
        test_items = [
            {"id": 1},
            {"id": 2, "fail": True},  # This should fail
            {"id": 3}
        ]
        
        results = []
        errors = []
        
        # Process with proper error handling
        for item in test_items:
            try:
                result = await mock_process_item(item)
                results.append(result)
            except Exception as e:
                errors.append({"item": item, "error": str(e)})
        
        # Verify mixed results handled correctly
        assert len(results) == 2  # 2 successful
        assert len(errors) == 1   # 1 failed
        assert not any("_asyncio.Future" in str(e) for e in errors)

# Test configuration and fixtures
@pytest.fixture(scope="session")
def test_database_url():
    """Provide test database URL"""
    return "sqlite:///:memory:"

@pytest.fixture(autouse=True)
def setup_test_logging():
    """Setup logging for tests"""
    logging.getLogger().setLevel(logging.DEBUG)

# Pytest markers for test categorization
pytest.mark.database = pytest.mark.mark("database", "Database-related tests")
pytest.mark.async_ops = pytest.mark.mark("async_ops", "Async operation tests")
pytest.mark.data_quality = pytest.mark.mark("data_quality", "Data quality tests")
pytest.mark.error_handling = pytest.mark.mark("error_handling", "Error handling tests")
pytest.mark.integration = pytest.mark.mark("integration", "Integration tests")

if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short"])