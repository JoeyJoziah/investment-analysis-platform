#!/usr/bin/env python3
"""
Simple ETL Test Script
Tests basic ETL functionality without running full pipeline
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all ETL modules can be imported"""
    try:
        from backend.etl.data_extractor import DataExtractor
        from backend.etl.data_transformer import DataTransformer
        from backend.etl.data_loader import DataLoader
        from backend.etl.etl_orchestrator import ETLOrchestrator
        
        logger.info("✅ All ETL modules imported successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Import error: {e}")
        return False

def test_database_connection():
    """Test database connection"""
    try:
        from backend.etl.data_loader import DataLoader
        loader = DataLoader()
        stats = loader.get_loading_stats()
        logger.info(f"✅ Database connected. Stats: {json.dumps(stats, indent=2, default=str)}")
        return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False

def test_data_extraction():
    """Test basic data extraction"""
    try:
        from backend.etl.data_extractor import DataExtractor
        import asyncio
        
        extractor = DataExtractor()
        
        # Test yfinance extraction (no API key needed)
        async def test():
            result = await extractor.fetch_yfinance_data('AAPL', period='5d')
            return result
        
        result = asyncio.run(test())
        
        if result and 'price_data' in result:
            logger.info(f"✅ Data extraction successful. Latest price: ${result['price_data']['close']}")
            return True
        else:
            logger.error("❌ No data extracted")
            return False
            
    except Exception as e:
        logger.error(f"❌ Extraction failed: {e}")
        return False

def test_data_transformation():
    """Test data transformation"""
    try:
        from backend.etl.data_transformer import DataTransformer
        import pandas as pd
        
        transformer = DataTransformer()
        
        # Create sample data
        sample_data = {
            'ticker': 'TEST',
            'sources': {
                'yfinance': {
                    'price_data': {
                        'history': [
                            {'Date': '2024-01-01', 'Open': 100, 'High': 105,
                             'Low': 99, 'Close': 103, 'Volume': 1000000}
                        ]
                    }
                }
            }
        }
        
        # Transform
        result = transformer.transform_price_data(sample_data)
        
        if not result.empty:
            logger.info(f"✅ Data transformation successful. Rows: {len(result)}")
            return True
        else:
            logger.error("❌ Transformation resulted in empty DataFrame")
            return False
            
    except Exception as e:
        logger.error(f"❌ Transformation failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("="*60)
    logger.info("ETL PIPELINE COMPONENT TESTS")
    logger.info("="*60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Database Connection", test_database_connection),
        ("Data Extraction", test_data_extraction),
        ("Data Transformation", test_data_transformation)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\nTesting: {test_name}")
        logger.info("-"*40)
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 All tests passed! ETL pipeline is ready.")
    else:
        logger.info(f"\n⚠️ {total - passed} tests failed. Please check the errors above.")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())