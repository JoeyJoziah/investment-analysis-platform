#!/usr/bin/env python3
"""
Investment Analysis Platform - Data Pipeline Startup Script

This script initializes the database, starts the data loading process, and provides
monitoring capabilities for the investment analysis platform.

Usage:
    # Start with default settings (10 stocks)
    python scripts/start_data_pipeline.py
    
    # Start with 100 stocks in background
    python scripts/start_data_pipeline.py --stocks 100 --background
    
    # Resume previous session
    python scripts/start_data_pipeline.py --resume --background
    
    # Monitor existing process
    python scripts/start_data_pipeline.py --monitor-only
"""

import asyncio
import argparse
import sys
import os
import logging
import signal
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import our modules
from scripts.load_historical_data import StockDataLoader, DatabaseManager, LoadingStatus
from backend.utils.cost_monitor import cost_monitor
from backend.config.settings import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure required directories exist
for directory in ['logs', 'scripts/data', 'scripts/data/cache']:
    os.makedirs(directory, exist_ok=True)


class DataPipelineManager:
    """Manages the complete data pipeline startup and monitoring"""
    
    def __init__(self):
        self.db_manager = None
        self.data_loader = None
        self.should_stop = False
        self.process_pid = os.getpid()
        self.status_file = Path('scripts/data/cache/pipeline_status.json')
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.should_stop = True
    
    def save_status(self, status: Dict):
        """Save pipeline status to file"""
        try:
            status_data = {
                'timestamp': datetime.now().isoformat(),
                'pid': self.process_pid,
                'status': status,
                'uptime_seconds': time.time() - self.start_time
            }
            
            with open(self.status_file, 'w') as f:
                json.dump(status_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save status: {e}")
    
    def load_status(self) -> Optional[Dict]:
        """Load pipeline status from file"""
        try:
            if self.status_file.exists():
                with open(self.status_file, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            logger.error(f"Failed to load status: {e}")
            return None
    
    async def initialize_system(self):
        """Initialize all system components"""
        logger.info("Initializing Investment Analysis Platform...")
        self.start_time = time.time()
        
        # Initialize database
        logger.info("Setting up database connection...")
        self.db_manager = DatabaseManager()
        self.db_manager.ensure_tables_exist()
        
        # Initialize data loader
        logger.info("Initializing data loader...")
        self.data_loader = StockDataLoader(self.db_manager)
        
        # Setup master data
        logger.info("Setting up master data (exchanges, sectors)...")
        with self.db_manager.get_session() as session:
            self.data_loader.setup_master_data(session)
        
        # Initialize cost monitor
        logger.info("Initializing cost monitoring...")
        await cost_monitor.initialize()
        
        logger.info("System initialization completed successfully!")
    
    async def start_data_loading(self, num_stocks: int, resume: bool = False):
        """Start the data loading process"""
        logger.info(f"Starting data loading for {num_stocks} stocks (resume={resume})")
        
        # Get stock list
        tickers = self.data_loader.get_sp500_stocks(num_stocks)
        
        # Filter for resume mode
        if resume:
            completed_tickers = {
                ticker for ticker, progress in self.data_loader.progress.items() 
                if progress.status in [LoadingStatus.COMPLETED, LoadingStatus.SKIPPED]
            }
            tickers = [t for t in tickers if t not in completed_tickers]
            logger.info(f"Resuming with {len(tickers)} remaining stocks")
        
        if not tickers:
            logger.info("No stocks to process")
            return {}
        
        # Start loading
        results = self.data_loader.load_stocks_parallel(tickers, max_workers=5)
        
        return results
    
    async def monitor_progress(self, update_interval: int = 30):
        """Monitor the data loading progress"""
        logger.info(f"Starting progress monitoring (update every {update_interval} seconds)")
        
        while not self.should_stop:
            try:
                # Get current progress
                progress = self.data_loader.progress
                completed = sum(1 for p in progress.values() if p.status == LoadingStatus.COMPLETED)
                failed = sum(1 for p in progress.values() if p.status == LoadingStatus.FAILED)
                skipped = sum(1 for p in progress.values() if p.status == LoadingStatus.SKIPPED)
                in_progress = sum(1 for p in progress.values() if p.status == LoadingStatus.IN_PROGRESS)
                total_records = sum(p.records_loaded for p in progress.values())
                
                # Database stats
                with self.db_manager.get_session() as session:
                    validation_results = self.data_loader.validate_loaded_data(session)
                
                # Cost monitoring stats (simplified for now)
                usage_stats = {"total_calls": 0, "total_cost": 0}  # Placeholder
                budget_status = {"daily_budget_remaining": 50.0, "monthly_budget_remaining": 50.0}
                
                status = {
                    'loading_progress': {
                        'total_stocks': len(progress),
                        'completed': completed,
                        'failed': failed,
                        'skipped': skipped,
                        'in_progress': in_progress,
                        'total_records': total_records
                    },
                    'database_stats': validation_results,
                    'api_usage': {
                        'daily_calls': usage_stats.get('total_calls', 0),
                        'budget_used_pct': budget_status.get('budget_used_pct', 0),
                        'budget_status': budget_status.get('status', 'unknown')
                    }
                }
                
                # Save status
                self.save_status(status)
                
                # Log progress
                logger.info(f"Progress: {completed} completed, {failed} failed, {in_progress} in progress, {total_records:,} records loaded")
                
                # Check if we're done
                if in_progress == 0 and completed > 0:
                    logger.info("Data loading completed!")
                    break
                
                await asyncio.sleep(update_interval)
                
            except Exception as e:
                logger.error(f"Error in progress monitoring: {e}")
                await asyncio.sleep(update_interval)
    
    async def print_final_report(self):
        """Print final status report"""
        print("\n" + "="*80)
        print("INVESTMENT ANALYSIS PLATFORM - DATA PIPELINE REPORT")
        print("="*80)
        
        # Loading summary
        if self.data_loader:
            progress = self.data_loader.progress
            completed = sum(1 for p in progress.values() if p.status == LoadingStatus.COMPLETED)
            failed = sum(1 for p in progress.values() if p.status == LoadingStatus.FAILED)
            skipped = sum(1 for p in progress.values() if p.status == LoadingStatus.SKIPPED)
            total_records = sum(p.records_loaded for p in progress.values())
            
            print(f"\nDATA LOADING SUMMARY:")
            print(f"  Total stocks processed: {len(progress)}")
            print(f"  Successfully completed: {completed}")
            print(f"  Failed: {failed}")
            print(f"  Skipped (already loaded): {skipped}")
            print(f"  Total records loaded: {total_records:,}")
        
        # Database validation
        if self.db_manager:
            with self.db_manager.get_session() as session:
                validation_results = self.data_loader.validate_loaded_data(session)
            
            print(f"\nDATABASE VALIDATION:")
            for key, value in validation_results.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:,}")
                else:
                    print(f"  {key}: {value}")
        
        # API usage report
        print(f"\nAPI USAGE SUMMARY:")
        usage_stats = {"total_calls": 0, "total_cost": 0}  # Placeholder
        budget_status = {"daily_budget_remaining": 50.0, "monthly_budget_remaining": 50.0}
        print(f"  Total API calls today: {usage_stats.get('total_calls', 0)}")
        print(f"  Budget used: {budget_status.get('budget_used_pct', 0):.1f}%")
        print(f"  Status: {budget_status.get('status', 'unknown')}")
        
        for provider, stats in usage_stats.get('by_provider', {}).items():
            print(f"  {provider}: {stats.get('calls', 0)} calls")
        
        # Next steps
        print(f"\nNEXT STEPS:")
        print(f"  1. Verify data quality in the database")
        print(f"  2. Set up Airflow DAGs for daily updates")
        print(f"  3. Configure monitoring and alerts")
        print(f"  4. Start the API server: uvicorn backend.api.main:app --reload")
        
        print("="*80)
    
    async def monitor_only_mode(self):
        """Monitor existing pipeline without starting new loading"""
        logger.info("Monitoring existing pipeline...")
        
        while not self.should_stop:
            try:
                status = self.load_status()
                if status:
                    print(f"\nPipeline Status (PID: {status.get('pid', 'unknown')}):")
                    print(f"  Uptime: {status.get('uptime_seconds', 0):.0f} seconds")
                    
                    loading_progress = status.get('status', {}).get('loading_progress', {})
                    print(f"  Progress: {loading_progress.get('completed', 0)} completed, "
                          f"{loading_progress.get('failed', 0)} failed, "
                          f"{loading_progress.get('in_progress', 0)} in progress")
                    print(f"  Records: {loading_progress.get('total_records', 0):,}")
                    
                    api_usage = status.get('status', {}).get('api_usage', {})
                    print(f"  API Usage: {api_usage.get('daily_calls', 0)} calls today, "
                          f"Budget: {api_usage.get('budget_used_pct', 0):.1f}% used")
                else:
                    print("No active pipeline found")
                
                await asyncio.sleep(30)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in monitor mode: {e}")
                await asyncio.sleep(30)
    
    async def run(
        self,
        num_stocks: int = 10,
        resume: bool = False,
        background: bool = False,
        monitor_only: bool = False
    ):
        """Run the complete pipeline"""
        try:
            if monitor_only:
                await self.monitor_only_mode()
                return
            
            # Initialize system
            await self.initialize_system()
            
            # Start data loading and monitoring concurrently
            loading_task = asyncio.create_task(
                self.start_data_loading(num_stocks, resume)
            )
            
            monitor_task = asyncio.create_task(
                self.monitor_progress()
            )
            
            # Wait for loading to complete
            results = await loading_task
            
            # Stop monitoring
            self.should_stop = True
            monitor_task.cancel()
            
            # Print results
            if self.data_loader:
                self.data_loader.print_summary(results)
            
            # Print final report
            await self.print_final_report()
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
        finally:
            # Cleanup
            if self.db_manager:
                # Close database connections would go here
                pass


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Investment Analysis Platform Data Pipeline')
    parser.add_argument('--stocks', type=int, default=10, 
                       help='Number of stocks to load (default: 10)')
    parser.add_argument('--background', action='store_true',
                       help='Run in background mode with monitoring')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from previous progress')
    parser.add_argument('--monitor-only', action='store_true',
                       help='Only monitor existing pipeline')
    
    args = parser.parse_args()
    
    # Create pipeline manager
    pipeline = DataPipelineManager()
    
    try:
        # Run the pipeline
        asyncio.run(pipeline.run(
            num_stocks=args.stocks,
            resume=args.resume,
            background=args.background,
            monitor_only=args.monitor_only
        ))
        
        logger.info("Data pipeline completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()