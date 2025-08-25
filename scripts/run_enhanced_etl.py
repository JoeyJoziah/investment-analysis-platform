#!/usr/bin/env python3
"""
Enhanced ETL Runner with Multi-Source Data Collection
Solves yfinance rate limiting by using multiple data sources
"""

import sys
import os
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.etl.etl_orchestrator import ETLOrchestrator
from backend.etl.multi_source_extractor import MultiSourceExtractor
from backend.etl.distributed_batch_processor import DistributedBatchProcessor
from backend.etl.data_validator import DataValidator, ValidationLevel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'backend/etl/logs/enhanced_etl_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedETLRunner:
    """Enhanced ETL Runner with multi-source support"""
    
    def __init__(self, mode='standard'):
        self.mode = mode
        self.orchestrator = ETLOrchestrator()
        self.extractor = MultiSourceExtractor()
        self.validator = DataValidator()
        self.distributed_processor = None
        
        if mode == 'distributed':
            self.distributed_processor = DistributedBatchProcessor()
    
    def run_sample_test(self, tickers=None):
        """Test with a small sample of tickers"""
        if not tickers:
            tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        
        logger.info(f"Testing with {len(tickers)} tickers: {tickers}")
        
        # Extract data
        logger.info("Extracting data from multiple sources...")
        data = self.extractor.extract_batch(tickers)
        
        # Validate data
        logger.info("Validating data quality...")
        for ticker, ticker_data in data.items():
            if ticker_data:
                quality = self.validator.calculate_quality_score(ticker_data)
                logger.info(f"{ticker}: Quality score = {quality:.1f}/100")
                
                if quality < 70:
                    logger.warning(f"{ticker}: Low quality score, checking issues...")
                    issues = self.validator.validate_financial_data(ticker_data)
                    if issues:
                        logger.warning(f"{ticker} issues: {issues}")
        
        # Display results
        logger.info("\nExtraction Results:")
        logger.info("-" * 50)
        for ticker, ticker_data in data.items():
            if ticker_data:
                logger.info(f"{ticker}:")
                logger.info(f"  Price: ${ticker_data.get('price', 'N/A')}")
                logger.info(f"  Volume: {ticker_data.get('volume', 'N/A'):,}")
                logger.info(f"  Change: {ticker_data.get('change_percent', 'N/A')}%")
                logger.info(f"  Source: {ticker_data.get('source', 'Unknown')}")
            else:
                logger.warning(f"{ticker}: No data extracted")
        
        return data
    
    def run_full_etl(self, resume=False):
        """Run full ETL for all stocks"""
        logger.info("="*60)
        logger.info("Starting Full ETL Pipeline with Multi-Source Extraction")
        logger.info("="*60)
        
        try:
            # Load all stock symbols
            symbols_df = self.orchestrator.data_loader.load_stock_symbols()
            all_tickers = symbols_df['symbol'].tolist()
            logger.info(f"Loaded {len(all_tickers)} stock symbols")
            
            # Use distributed processing for large datasets
            if len(all_tickers) > 1000:
                logger.info("Using distributed processing for large dataset")
                return self.run_distributed_etl(all_tickers, resume)
            
            # Standard processing for smaller datasets
            batch_size = 50
            total_batches = (len(all_tickers) + batch_size - 1) // batch_size
            
            success_count = 0
            failed_tickers = []
            
            for i in range(0, len(all_tickers), batch_size):
                batch = all_tickers[i:i+batch_size]
                batch_num = i // batch_size + 1
                
                logger.info(f"Processing batch {batch_num}/{total_batches}: {len(batch)} tickers")
                
                try:
                    # Extract data
                    data = self.extractor.extract_batch(batch)
                    
                    # Process successful extractions
                    for ticker, ticker_data in data.items():
                        if ticker_data:
                            # Validate quality
                            quality = self.validator.calculate_quality_score(ticker_data)
                            
                            if quality >= 70:
                                # Store in database
                                self.orchestrator.data_transformer.transform_and_store(ticker_data)
                                success_count += 1
                            else:
                                logger.warning(f"{ticker}: Quality too low ({quality:.1f})")
                                failed_tickers.append(ticker)
                        else:
                            failed_tickers.append(ticker)
                    
                    # Progress update
                    progress = ((i + len(batch)) / len(all_tickers)) * 100
                    logger.info(f"Progress: {progress:.1f}% - Success: {success_count}, Failed: {len(failed_tickers)}")
                    
                except Exception as e:
                    logger.error(f"Error processing batch {batch_num}: {e}")
                    failed_tickers.extend(batch)
                
                # Respect rate limits between batches
                time.sleep(2)
            
            # Final summary
            logger.info("="*60)
            logger.info("ETL Pipeline Completed")
            logger.info(f"Total Processed: {len(all_tickers)}")
            logger.info(f"Successful: {success_count} ({(success_count/len(all_tickers)*100):.1f}%)")
            logger.info(f"Failed: {len(failed_tickers)} ({(len(failed_tickers)/len(all_tickers)*100):.1f}%)")
            logger.info("="*60)
            
            # Save failed tickers for retry
            if failed_tickers:
                with open('backend/etl/failed_tickers.json', 'w') as f:
                    json.dump(failed_tickers, f)
                logger.info(f"Failed tickers saved to backend/etl/failed_tickers.json")
            
            return {
                'total': len(all_tickers),
                'success': success_count,
                'failed': len(failed_tickers),
                'failed_tickers': failed_tickers[:10]  # First 10 for review
            }
            
        except Exception as e:
            logger.error(f"ETL Pipeline failed: {e}")
            raise
    
    def run_distributed_etl(self, tickers, resume=False):
        """Run ETL using distributed processing"""
        logger.info("Initializing distributed batch processor...")
        
        if not self.distributed_processor:
            self.distributed_processor = DistributedBatchProcessor()
        
        # Create or resume jobs
        if resume:
            logger.info("Resuming previous ETL run...")
            jobs = self.distributed_processor.get_pending_jobs()
        else:
            logger.info("Creating new jobs for distributed processing...")
            jobs = self.distributed_processor.create_jobs(tickers)
        
        logger.info(f"Processing {len(jobs)} jobs with {len(tickers)} total tickers")
        
        # Process all jobs
        results = self.distributed_processor.process_all_jobs(jobs)
        
        # Print statistics
        self.distributed_processor.print_statistics()
        
        return results
    
    def run_daily_update(self):
        """Run daily update for existing stocks"""
        logger.info("Running daily update for existing stocks...")
        
        # Get stocks that need updating (e.g., haven't been updated today)
        # This would query your database for stocks needing updates
        
        # For now, update top 100 most active stocks
        priority_tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK.B',
            'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'BAC', 'PYPL',
            'NFLX', 'ADBE', 'CRM', 'PFE', 'TMO', 'CMCSA', 'ABT', 'NKE', 'CVX',
            'PEP', 'WMT', 'ABBV', 'KO', 'INTC', 'VZ', 'CSCO', 'XOM', 'MRK'
        ]
        
        logger.info(f"Updating {len(priority_tickers)} priority stocks")
        data = self.extractor.extract_batch(priority_tickers)
        
        success_count = sum(1 for d in data.values() if d)
        logger.info(f"Daily update completed: {success_count}/{len(priority_tickers)} successful")
        
        return data

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Enhanced ETL Runner with Multi-Source Support')
    parser.add_argument('--mode', choices=['test', 'full', 'distributed', 'daily'], 
                       default='test', help='ETL mode to run')
    parser.add_argument('--tickers', type=str, help='Comma-separated list of tickers for test mode')
    parser.add_argument('--resume', action='store_true', help='Resume previous run')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create runner
    runner = EnhancedETLRunner(mode='distributed' if args.mode == 'distributed' else 'standard')
    
    # Execute based on mode
    if args.mode == 'test':
        tickers = args.tickers.split(',') if args.tickers else None
        runner.run_sample_test(tickers)
    
    elif args.mode == 'full':
        runner.run_full_etl(resume=args.resume)
    
    elif args.mode == 'distributed':
        # Load all tickers
        orchestrator = ETLOrchestrator()
        symbols_df = orchestrator.data_loader.load_stock_symbols()
        all_tickers = symbols_df['symbol'].tolist()
        runner.run_distributed_etl(all_tickers, resume=args.resume)
    
    elif args.mode == 'daily':
        runner.run_daily_update()
    
    logger.info("ETL Runner completed successfully")

if __name__ == "__main__":
    main()