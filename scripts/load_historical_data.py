#!/usr/bin/env python3
"""
Historical Data Loader for Investment Analysis Platform

This script loads 1 year of historical data for S&P 500 stocks using Yahoo Finance.
Includes progress monitoring, error handling, and background execution capability.

Usage:
    python scripts/load_historical_data.py --stocks 10 --background
    python scripts/load_historical_data.py --stocks 100 --validate-only
    python scripts/load_historical_data.py --resume
"""

import asyncio
import argparse
import logging
import os
import sys
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
import signal
import threading
import time
from pathlib import Path

# Add the project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine, text, func
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError
import psycopg2
from tqdm import tqdm
import concurrent.futures
from dataclasses import dataclass
from enum import Enum

from backend.models.database import (
    Base, Stock, Exchange, Sector, Industry, 
    PriceHistory, Fundamentals, APIUsage
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_loading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)
os.makedirs('scripts/data/cache', exist_ok=True)


class LoadingStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class LoadingProgress:
    """Track loading progress for each stock"""
    ticker: str
    status: LoadingStatus
    records_loaded: int = 0
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class DatabaseManager:
    """Manages database connections and operations"""
    
    def __init__(self):
        self.engine = None
        self.Session = None
        self._connect()
    
    def _connect(self):
        """Establish database connection"""
        try:
            # Default connection string for PostgreSQL
            db_url = os.getenv(
                'DATABASE_URL', 
                'postgresql://postgres:9v1g^OV9XUwzUP6cEgCYgNOE@localhost:5432/investment_db'
            )
            
            self.engine = create_engine(
                db_url,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600
            )
            self.Session = sessionmaker(bind=self.engine)
            
            # Test connection
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                logger.info("Database connection established successfully")
                
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def get_session(self) -> Session:
        """Get a new database session"""
        return self.Session()
    
    def ensure_tables_exist(self):
        """Ensure all tables and indexes exist"""
        try:
            # Create tables using SQLAlchemy - this will also create indexes defined in models
            # Wrap in try-catch to handle existing indexes gracefully
            try:
                Base.metadata.create_all(self.engine)
                logger.info("Database tables and model-defined indexes ensured")
            except Exception as e:
                if "already exists" in str(e):
                    logger.info("Tables/indexes already exist, continuing...")
                else:
                    # Re-raise if it's not a duplicate index error
                    raise
            
            # Additional custom indexes not defined in models
            with self.engine.connect() as conn:
                # Function to check if index exists
                def index_exists(index_name):
                    query = text("""
                        SELECT COUNT(*) 
                        FROM pg_indexes 
                        WHERE indexname = :index_name
                    """)
                    result = conn.execute(query, {"index_name": index_name}).scalar()
                    return result > 0
                
                # Only define additional indexes not already in models
                # Skip indexes that are defined in the model files:
                # - idx_prediction_date (defined in model)
                # - idx_prediction_stock_model (defined in model)
                additional_indexes = [
                    ("idx_price_history_stock_date", "CREATE INDEX IF NOT EXISTS idx_price_history_stock_date ON price_history (stock_id, date DESC)"),
                    ("idx_price_history_date", "CREATE INDEX IF NOT EXISTS idx_price_history_date ON price_history (date DESC)"),
                    ("idx_technical_indicators_stock_date", "CREATE INDEX IF NOT EXISTS idx_technical_indicators_stock_date ON technical_indicators (stock_id, date DESC)"),
                    ("idx_prediction_stock_date", "CREATE INDEX IF NOT EXISTS idx_prediction_stock_date ON predictions (stock_id, prediction_date)"),
                    ("idx_stocks_ticker", "CREATE INDEX IF NOT EXISTS idx_stocks_ticker ON stocks (ticker)"),
                    ("idx_stocks_exchange", "CREATE INDEX IF NOT EXISTS idx_stocks_exchange ON stocks (exchange)")
                ]
                
                # Create additional indexes that don't exist
                created_count = 0
                for index_name, create_sql in additional_indexes:
                    try:
                        if not index_exists(index_name):
                            conn.execute(text(create_sql))
                            conn.commit()
                            logger.debug(f"Created additional index: {index_name}")
                            created_count += 1
                        else:
                            logger.debug(f"Index already exists: {index_name}")
                    except Exception as e:
                        # Log but don't fail if index already exists
                        if "already exists" in str(e):
                            logger.debug(f"Index {index_name} already exists")
                        else:
                            logger.debug(f"Could not create index {index_name}: {e}")
                
                if created_count > 0:
                    logger.info(f"Created {created_count} additional indexes")
            
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise


class StockDataLoader:
    """Loads historical stock data with progress tracking"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.progress_file = Path('scripts/data/cache/loading_progress.json')
        self.should_stop = False
        
        # Load existing progress
        self.progress: Dict[str, LoadingProgress] = self._load_progress()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.should_stop = True
    
    def _load_progress(self) -> Dict[str, LoadingProgress]:
        """Load existing progress from cache"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                
                progress = {}
                for ticker, info in data.items():
                    progress[ticker] = LoadingProgress(
                        ticker=ticker,
                        status=LoadingStatus(info['status']),
                        records_loaded=info.get('records_loaded', 0),
                        error_message=info.get('error_message'),
                        started_at=datetime.fromisoformat(info['started_at']) if info.get('started_at') else None,
                        completed_at=datetime.fromisoformat(info['completed_at']) if info.get('completed_at') else None
                    )
                
                logger.info(f"Loaded progress for {len(progress)} stocks from cache")
                return progress
                
            except Exception as e:
                logger.warning(f"Failed to load progress cache: {e}")
        
        return {}
    
    def _save_progress(self):
        """Save current progress to cache"""
        try:
            data = {}
            for ticker, progress in self.progress.items():
                data[ticker] = {
                    'status': progress.status.value,
                    'records_loaded': progress.records_loaded,
                    'error_message': progress.error_message,
                    'started_at': progress.started_at.isoformat() if progress.started_at else None,
                    'completed_at': progress.completed_at.isoformat() if progress.completed_at else None
                }
            
            with open(self.progress_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")
    
    def get_sp500_stocks(self, limit: Optional[int] = None) -> List[str]:
        """Get S&P 500 stock list"""
        # Top S&P 500 stocks by market cap
        sp500_top = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK.B', 'UNH', 'JNJ',
            'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'BAC', 'ABBV', 'PFE',
            'AVGO', 'KO', 'LLY', 'MRK', 'PEP', 'TMO', 'COST', 'WMT', 'DIS', 'ABT',
            'DHR', 'VZ', 'ADBE', 'NKE', 'CRM', 'MCD', 'T', 'BMY', 'NFLX', 'TXN',
            'NEE', 'ORCL', 'ACN', 'QCOM', 'PM', 'CMCSA', 'LIN', 'HON', 'WFC', 'UPS',
            'IBM', 'AMGN', 'RTX', 'LOW', 'SCHW', 'ELV', 'CAT', 'GS', 'DE', 'INTU',
            'BA', 'SPGI', 'BLK', 'MDT', 'AXP', 'BKNG', 'TJX', 'GILD', 'AMAT', 'SYK',
            'PLD', 'GE', 'MMM', 'CVS', 'AMT', 'CI', 'MO', 'SO', 'TMUS', 'ZTS',
            'CB', 'ISRG', 'DUK', 'CSX', 'CME', 'TGT', 'MU', 'CL', 'ITW', 'FIS',
            'NOC', 'USB', 'ICE', 'NSC', 'EQIX', 'APD', 'AON', 'FCX', 'PNC', 'BSX'
        ]
        
        if limit:
            return sp500_top[:limit]
        return sp500_top
    
    def setup_master_data(self, session: Session):
        """Setup exchanges, sectors, and basic stock records"""
        logger.info("Setting up master data...")
        
        # Ensure exchanges exist
        exchanges = [
            ('NYSE', 'New York Stock Exchange', 'America/New_York'),
            ('NASDAQ', 'NASDAQ', 'America/New_York'),
            ('AMEX', 'American Stock Exchange', 'America/New_York')
        ]
        
        for code, name, timezone in exchanges:
            existing = session.query(Exchange).filter(Exchange.code == code).first()
            if not existing:
                exchange = Exchange(code=code, name=name, timezone=timezone)
                session.add(exchange)
        
        # Ensure sectors exist
        sectors = [
            'Technology', 'Healthcare', 'Financial Services', 'Consumer Discretionary',
            'Communication Services', 'Industrials', 'Consumer Staples', 'Energy',
            'Utilities', 'Real Estate', 'Materials'
        ]
        
        for sector_name in sectors:
            existing = session.query(Sector).filter(Sector.name == sector_name).first()
            if not existing:
                sector = Sector(name=sector_name)
                session.add(sector)
        
        session.commit()
        logger.info("Master data setup completed")
    
    def ensure_stock_exists(self, session: Session, ticker: str) -> Optional[Stock]:
        """Ensure stock record exists in database"""
        try:
            stock = session.query(Stock).filter(Stock.ticker == ticker).first()
            if stock:
                return stock
            
            # Convert ticker for yfinance compatibility
            yf_ticker_symbol = self.convert_ticker_for_yfinance(ticker)
            
            # Get basic info from Yahoo Finance
            yf_ticker = yf.Ticker(yf_ticker_symbol)
            try:
                info = yf_ticker.info
                name = info.get('longName', info.get('shortName', ticker))
                market_cap = info.get('marketCap')
                sector_name = info.get('sector', 'Technology')
                
                # Get exchange
                exchange = session.query(Exchange).filter(Exchange.code == 'NASDAQ').first()
                
                # Get sector
                sector = session.query(Sector).filter(Sector.name == sector_name).first()
                if not sector:
                    sector = session.query(Sector).filter(Sector.name == 'Technology').first()
                
                stock = Stock(
                    ticker=ticker,
                    name=name,
                    exchange_id=exchange.id if exchange else 1,
                    sector_id=sector.id if sector else 1,
                    market_cap=market_cap,
                    is_active=True,
                    is_tradeable=True
                )
                
                session.add(stock)
                session.commit()
                logger.info(f"Created stock record for {ticker}")
                return stock
                
            except Exception as e:
                logger.warning(f"Failed to get info for {ticker}: {e}")
                # Create basic stock record
                exchange = session.query(Exchange).first()
                sector = session.query(Sector).first()
                
                stock = Stock(
                    ticker=ticker,
                    name=ticker,
                    exchange_id=exchange.id if exchange else 1,
                    sector_id=sector.id if sector else 1,
                    is_active=True,
                    is_tradeable=True
                )
                
                session.add(stock)
                session.commit()
                return stock
                
        except Exception as e:
            logger.error(f"Failed to ensure stock exists for {ticker}: {e}")
            session.rollback()
            return None
    
    def check_existing_data(self, session: Session, stock: Stock) -> int:
        """Check how many records already exist for this stock"""
        try:
            count = session.query(func.count(PriceHistory.id)).filter(
                PriceHistory.stock_id == stock.id
            ).scalar()
            return count or 0
        except Exception as e:
            logger.error(f"Failed to check existing data for {stock.ticker}: {e}")
            return 0
    
    def convert_ticker_for_yfinance(self, ticker: str) -> str:
        """Convert ticker symbol for yfinance API compatibility"""
        # Handle special cases where dots need to be converted to hyphens
        # This is needed for stocks like BRK.B -> BRK-B
        yf_ticker = ticker.replace('.', '-')
        
        # Log conversion if ticker was changed
        if yf_ticker != ticker:
            logger.info(f"Converted ticker {ticker} to {yf_ticker} for yfinance API")
        
        return yf_ticker
    
    def load_stock_data(self, ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """Load historical data for a single stock"""
        try:
            logger.info(f"Fetching data for {ticker}...")
            
            # Convert ticker for yfinance compatibility
            yf_ticker_symbol = self.convert_ticker_for_yfinance(ticker)
            
            # Create yfinance ticker
            yf_ticker = yf.Ticker(yf_ticker_symbol)
            
            # Fetch historical data
            hist = yf_ticker.history(period=period)
            
            if hist.empty:
                logger.warning(f"No data returned for {ticker} (yfinance symbol: {yf_ticker_symbol})")
                return None
            
            # Clean and validate data
            hist = hist.dropna()
            hist.reset_index(inplace=True)
            
            # Add calculated fields
            hist['intraday_volatility'] = (hist['High'] - hist['Low']) / hist['Low']
            hist['typical_price'] = (hist['High'] + hist['Low'] + hist['Close']) / 3
            hist['vwap'] = (hist['typical_price'] * hist['Volume']).cumsum() / hist['Volume'].cumsum()
            
            logger.info(f"Successfully loaded {len(hist)} records for {ticker}")
            return hist
            
        except Exception as e:
            logger.error(f"Failed to load data for {ticker}: {e}")
            return None
    
    def save_price_data(self, session: Session, stock: Stock, data: pd.DataFrame) -> int:
        """Save price data to database"""
        try:
            records_saved = 0
            
            for _, row in data.iterrows():
                try:
                    # Check if record already exists
                    existing = session.query(PriceHistory).filter(
                        PriceHistory.stock_id == stock.id,
                        PriceHistory.date == row['Date']
                    ).first()
                    
                    if existing:
                        continue
                    
                    price_record = PriceHistory(
                        stock_id=stock.id,
                        date=row['Date'],
                        open=float(row['Open']),
                        high=float(row['High']),
                        low=float(row['Low']),
                        close=float(row['Close']),
                        adjusted_close=float(row['Adj Close']) if 'Adj Close' in row else float(row['Close']),
                        volume=int(row['Volume']),
                        intraday_volatility=float(row['intraday_volatility']),
                        typical_price=float(row['typical_price']),
                        vwap=float(row['vwap'])
                    )
                    
                    session.add(price_record)
                    records_saved += 1
                    
                except Exception as e:
                    logger.error(f"Failed to save record for {stock.ticker} on {row['Date']}: {e}")
                    continue
            
            session.commit()
            logger.info(f"Saved {records_saved} price records for {stock.ticker}")
            return records_saved
            
        except Exception as e:
            logger.error(f"Failed to save price data for {stock.ticker}: {e}")
            session.rollback()
            return 0
    
    def load_single_stock(self, ticker: str) -> LoadingProgress:
        """Load data for a single stock"""
        progress = LoadingProgress(ticker=ticker, status=LoadingStatus.IN_PROGRESS)
        progress.started_at = datetime.now()
        
        try:
            with self.db_manager.get_session() as session:
                # Ensure stock exists
                stock = self.ensure_stock_exists(session, ticker)
                if not stock:
                    progress.status = LoadingStatus.FAILED
                    progress.error_message = "Failed to create stock record"
                    return progress
                
                # Check existing data (for incremental loading)
                existing_count = self.check_existing_data(session, stock)
                if existing_count > 200:  # Roughly 1 year of data
                    logger.info(f"Stock {ticker} already has {existing_count} records, skipping")
                    progress.status = LoadingStatus.SKIPPED
                    progress.records_loaded = existing_count
                    return progress
                
                # Load historical data
                hist_data = self.load_stock_data(ticker)
                if hist_data is None or hist_data.empty:
                    progress.status = LoadingStatus.FAILED
                    progress.error_message = "No historical data available"
                    return progress
                
                # Save to database
                records_saved = self.save_price_data(session, stock, hist_data)
                progress.records_loaded = records_saved
                
                if records_saved > 0:
                    progress.status = LoadingStatus.COMPLETED
                else:
                    progress.status = LoadingStatus.FAILED
                    progress.error_message = "No records saved"
                
                # Record API usage
                api_usage = APIUsage(
                    provider='yfinance',
                    endpoint='history',
                    calls_count=1,
                    data_points=len(hist_data),
                    success=True,
                    estimated_cost=0.0  # Yahoo Finance is free
                )
                session.add(api_usage)
                session.commit()
                
        except Exception as e:
            logger.error(f"Failed to load data for {ticker}: {e}")
            progress.status = LoadingStatus.FAILED
            progress.error_message = str(e)
        
        finally:
            progress.completed_at = datetime.now()
        
        return progress
    
    def load_stocks_parallel(self, tickers: List[str], max_workers: int = 5) -> Dict[str, LoadingProgress]:
        """Load multiple stocks in parallel"""
        logger.info(f"Starting parallel loading of {len(tickers)} stocks with {max_workers} workers")
        
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_ticker = {
                executor.submit(self.load_single_stock, ticker): ticker 
                for ticker in tickers
            }
            
            # Process completed tasks
            for future in tqdm(
                concurrent.futures.as_completed(future_to_ticker),
                total=len(tickers),
                desc="Loading stocks"
            ):
                if self.should_stop:
                    logger.info("Stop signal received, cancelling remaining tasks...")
                    executor.shutdown(wait=False)
                    break
                
                ticker = future_to_ticker[future]
                try:
                    progress = future.result(timeout=60)
                    results[ticker] = progress
                    self.progress[ticker] = progress
                    
                    # Save progress periodically
                    if len(results) % 5 == 0:
                        self._save_progress()
                    
                except concurrent.futures.TimeoutError:
                    logger.error(f"Timeout loading {ticker}")
                    progress = LoadingProgress(
                        ticker=ticker,
                        status=LoadingStatus.FAILED,
                        error_message="Timeout"
                    )
                    results[ticker] = progress
                    self.progress[ticker] = progress
                    
                except Exception as e:
                    logger.error(f"Exception loading {ticker}: {e}")
                    progress = LoadingProgress(
                        ticker=ticker,
                        status=LoadingStatus.FAILED,
                        error_message=str(e)
                    )
                    results[ticker] = progress
                    self.progress[ticker] = progress
                
                # Rate limiting - small delay between requests
                time.sleep(0.1)
        
        # Final progress save
        self._save_progress()
        return results
    
    def print_summary(self, results: Dict[str, LoadingProgress]):
        """Print loading summary"""
        completed = sum(1 for p in results.values() if p.status == LoadingStatus.COMPLETED)
        failed = sum(1 for p in results.values() if p.status == LoadingStatus.FAILED)
        skipped = sum(1 for p in results.values() if p.status == LoadingStatus.SKIPPED)
        total_records = sum(p.records_loaded for p in results.values())
        
        print("\n" + "="*60)
        print("DATA LOADING SUMMARY")
        print("="*60)
        print(f"Total stocks processed: {len(results)}")
        print(f"Successfully completed: {completed}")
        print(f"Failed: {failed}")
        print(f"Skipped (already loaded): {skipped}")
        print(f"Total records loaded: {total_records:,}")
        print("="*60)
        
        # Print failures
        failures = [p for p in results.values() if p.status == LoadingStatus.FAILED]
        if failures:
            print("\nFAILED STOCKS:")
            for failure in failures[:10]:  # Show first 10 failures
                print(f"  {failure.ticker}: {failure.error_message}")
            if len(failures) > 10:
                print(f"  ... and {len(failures) - 10} more")
    
    def validate_loaded_data(self, session: Session) -> Dict[str, int]:
        """Validate loaded data"""
        logger.info("Validating loaded data...")
        
        validation_results = {}
        
        try:
            # Count total records
            total_records = session.query(func.count(PriceHistory.id)).scalar()
            validation_results['total_price_records'] = total_records
            
            # Count stocks with data
            stocks_with_data = session.query(func.count(Stock.id.distinct())).join(PriceHistory).scalar()
            validation_results['stocks_with_data'] = stocks_with_data
            
            # Check for data quality issues
            invalid_prices = session.query(func.count(PriceHistory.id)).filter(
                (PriceHistory.high < PriceHistory.low) |
                (PriceHistory.close < 0) |
                (PriceHistory.volume < 0)
            ).scalar()
            validation_results['invalid_prices'] = invalid_prices
            
            # Recent data check
            recent_date = datetime.now() - timedelta(days=7)
            recent_records = session.query(func.count(PriceHistory.id)).filter(
                PriceHistory.date >= recent_date
            ).scalar()
            validation_results['recent_records'] = recent_records
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            validation_results['error'] = str(e)
        
        return validation_results


def run_background_monitor():
    """Monitor the loading process in a separate thread"""
    def monitor():
        progress_file = Path('scripts/data/cache/loading_progress.json')
        
        while True:
            try:
                if progress_file.exists():
                    with open(progress_file, 'r') as f:
                        data = json.load(f)
                    
                    completed = sum(1 for p in data.values() if p.get('status') == 'completed')
                    failed = sum(1 for p in data.values() if p.get('status') == 'failed')
                    in_progress = sum(1 for p in data.values() if p.get('status') == 'in_progress')
                    total_records = sum(p.get('records_loaded', 0) for p in data.values())
                    
                    print(f"\rProgress: {completed} completed, {failed} failed, {in_progress} in progress, {total_records:,} records loaded", end='', flush=True)
                
                time.sleep(10)  # Update every 10 seconds
                
            except KeyboardInterrupt:
                break
            except Exception:
                time.sleep(10)
                continue
    
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    return monitor_thread


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Load historical stock data')
    parser.add_argument('--stocks', type=int, default=10, help='Number of stocks to load (default: 10)')
    parser.add_argument('--background', action='store_true', help='Run in background mode')
    parser.add_argument('--validate-only', action='store_true', help='Only validate existing data')
    parser.add_argument('--resume', action='store_true', help='Resume from previous progress')
    parser.add_argument('--max-workers', type=int, default=5, help='Max parallel workers')
    
    args = parser.parse_args()
    
    logger.info("Starting historical data loading process")
    
    # Initialize database manager
    db_manager = DatabaseManager()
    db_manager.ensure_tables_exist()
    
    # Initialize data loader
    loader = StockDataLoader(db_manager)
    
    # Setup master data
    with db_manager.get_session() as session:
        loader.setup_master_data(session)
    
    # Validate only mode
    if args.validate_only:
        with db_manager.get_session() as session:
            validation_results = loader.validate_loaded_data(session)
            print("\nValidation Results:")
            for key, value in validation_results.items():
                print(f"  {key}: {value:,}")
        return
    
    # Get stock list
    tickers = loader.get_sp500_stocks(args.stocks)
    
    # Filter for resume mode
    if args.resume:
        completed_tickers = {
            ticker for ticker, progress in loader.progress.items() 
            if progress.status in [LoadingStatus.COMPLETED, LoadingStatus.SKIPPED]
        }
        tickers = [t for t in tickers if t not in completed_tickers]
        logger.info(f"Resuming with {len(tickers)} remaining stocks")
    
    if not tickers:
        logger.info("No stocks to process")
        return
    
    # Start background monitor if requested
    monitor_thread = None
    if args.background:
        monitor_thread = run_background_monitor()
    
    try:
        # Load data
        results = loader.load_stocks_parallel(tickers, args.max_workers)
        
        # Print summary
        loader.print_summary(results)
        
        # Validate loaded data
        with db_manager.get_session() as session:
            validation_results = loader.validate_loaded_data(session)
            print("\nValidation Results:")
            for key, value in validation_results.items():
                print(f"  {key}: {value:,}")
        
        logger.info("Data loading process completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Process failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()