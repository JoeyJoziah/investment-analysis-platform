#!/usr/bin/env python3
"""
ETL Pipeline Activation Script
Activates and tests the complete ETL pipeline
"""

import asyncio
import sys
import os
import logging
from datetime import datetime
import json
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.etl.etl_orchestrator import ETLOrchestrator, ETLScheduler
from backend.etl.data_loader import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'etl_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def check_database_connection():
    """Check if database is accessible"""
    try:
        loader = DataLoader()
        stats = loader.get_loading_stats()
        logger.info("Database connection successful")
        logger.info(f"Current database stats: {json.dumps(stats, indent=2, default=str)}")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


def check_api_keys():
    """Check if required API keys are configured"""
    required_keys = [
        'FINNHUB_API_KEY',
        'ALPHA_VANTAGE_API_KEY', 
        'POLYGON_API_KEY',
        'NEWS_API_KEY'
    ]
    
    missing_keys = []
    for key in required_keys:
        if not os.getenv(key):
            missing_keys.append(key)
    
    if missing_keys:
        logger.warning(f"Missing API keys: {missing_keys}")
        logger.warning("Pipeline will use available APIs only")
    else:
        logger.info("All API keys configured")
    
    return len(missing_keys) == 0


async def test_single_ticker(ticker: str):
    """Test ETL pipeline with a single ticker"""
    logger.info(f"\n{'='*50}")
    logger.info(f"Testing ETL pipeline with ticker: {ticker}")
    logger.info(f"{'='*50}\n")
    
    orchestrator = ETLOrchestrator()
    
    # Configure for single ticker test
    orchestrator.config['batch_size'] = 1
    orchestrator.config['enable_ml'] = False  # Disable ML for quick test
    
    try:
        result = await orchestrator.run_realtime_update(ticker)
        
        logger.info(f"\nTest Results for {ticker}:")
        logger.info(json.dumps(result, indent=2, default=str))
        
        return result['status'] == 'success'
        
    except Exception as e:
        logger.error(f"Test failed for {ticker}: {e}")
        return False


async def test_batch_pipeline(tickers: list):
    """Test ETL pipeline with multiple tickers"""
    logger.info(f"\n{'='*50}")
    logger.info(f"Testing ETL pipeline with {len(tickers)} tickers")
    logger.info(f"Tickers: {tickers}")
    logger.info(f"{'='*50}\n")
    
    orchestrator = ETLOrchestrator()
    
    # Configure for batch test
    orchestrator.config['batch_size'] = 5
    orchestrator.config['enable_ml'] = True
    orchestrator.config['enable_recommendations'] = True
    
    try:
        result = await orchestrator.run_full_pipeline(tickers)
        
        logger.info("\n" + "="*50)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("="*50)
        
        duration = (result['end_time'] - result['start_time']).total_seconds() if result.get('end_time') else 0
        
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Stocks Processed: {result.get('stocks_processed', 0)}")
        logger.info(f"Stocks Failed: {result.get('stocks_failed', 0)}")
        logger.info(f"Recommendations Generated: {result.get('recommendations_generated', 0)}")
        
        if result.get('errors'):
            logger.error(f"Errors encountered: {len(result['errors'])}")
            for error in result['errors'][:5]:  # Show first 5 errors
                logger.error(f"  - {error}")
        
        return result.get('stocks_processed', 0) > 0
        
    except Exception as e:
        logger.error(f"Batch test failed: {e}")
        return False


async def run_full_pipeline():
    """Run the complete ETL pipeline"""
    logger.info(f"\n{'='*50}")
    logger.info("RUNNING FULL ETL PIPELINE")
    logger.info(f"{'='*50}\n")
    
    orchestrator = ETLOrchestrator()
    
    try:
        # Get all active tickers
        tickers = await orchestrator.get_active_tickers()
        logger.info(f"Processing {len(tickers)} active tickers")
        
        # Run pipeline
        result = await orchestrator.run_full_pipeline(tickers)
        
        # Display results
        logger.info("\n" + "="*50)
        logger.info("FULL PIPELINE RESULTS")
        logger.info("="*50)
        
        logger.info(json.dumps({
            'duration': str(result['end_time'] - result['start_time']) if result.get('end_time') else 'N/A',
            'stocks_processed': result.get('stocks_processed', 0),
            'stocks_failed': result.get('stocks_failed', 0),
            'recommendations_generated': result.get('recommendations_generated', 0),
            'error_count': len(result.get('errors', []))
        }, indent=2))
        
        # Check database stats
        loader = DataLoader()
        stats = loader.get_loading_stats()
        
        logger.info("\nDatabase Statistics After Pipeline:")
        logger.info(json.dumps(stats, indent=2, default=str))
        
        return result.get('stocks_processed', 0) > 0
        
    except Exception as e:
        logger.error(f"Full pipeline failed: {e}")
        return False


async def schedule_pipeline():
    """Schedule and run the ETL pipeline"""
    logger.info(f"\n{'='*50}")
    logger.info("SCHEDULING ETL PIPELINE")
    logger.info(f"{'='*50}\n")
    
    scheduler = ETLScheduler()
    
    logger.info("Running daily pipeline...")
    await scheduler.run_daily_pipeline()
    
    logger.info("\nRunning hourly update...")
    await scheduler.run_hourly_update()
    
    logger.info("\nScheduler test complete")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='ETL Pipeline Activation Script')
    parser.add_argument('--mode', choices=['test', 'batch', 'full', 'schedule'], 
                       default='test', help='Pipeline mode to run')
    parser.add_argument('--tickers', nargs='+', 
                       default=['AAPL', 'GOOGL', 'MSFT'],
                       help='Tickers to process')
    
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("ETL PIPELINE ACTIVATION SCRIPT")
    logger.info("="*70)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Timestamp: {datetime.now()}")
    logger.info("="*70 + "\n")
    
    # Pre-flight checks
    logger.info("Running pre-flight checks...")
    
    if not check_database_connection():
        logger.error("Database connection failed. Please check your database configuration.")
        return 1
    
    check_api_keys()
    
    # Run selected mode
    success = False
    
    if args.mode == 'test':
        # Test with single ticker
        success = await test_single_ticker(args.tickers[0])
        
    elif args.mode == 'batch':
        # Test with batch of tickers
        success = await test_batch_pipeline(args.tickers)
        
    elif args.mode == 'full':
        # Run full pipeline
        success = await run_full_pipeline()
        
    elif args.mode == 'schedule':
        # Test scheduler
        await schedule_pipeline()
        success = True
    
    # Final summary
    logger.info("\n" + "="*70)
    if success:
        logger.info("✅ ETL PIPELINE ACTIVATION SUCCESSFUL")
    else:
        logger.info("❌ ETL PIPELINE ACTIVATION FAILED")
    logger.info("="*70)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)