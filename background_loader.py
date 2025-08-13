#!/usr/bin/env python3
"""
Background data loader - loads additional stocks incrementally
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import execute_batch
import logging
import sys
import time
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_loading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'investment_db',
    'user': 'postgres',
    'password': '9v1g^OV9XUwzUP6cEgCYgNOE'
}

# Extended S&P 500 list (next 80 stocks)
NEXT_STOCKS = [
    'CVX', 'PEP', 'ABBV', 'MRK', 'TMO', 'COST', 'ADBE', 'CRM', 'ACN', 'LLY',
    'NKE', 'TXN', 'MDT', 'UPS', 'BMY', 'NFLX', 'AMD', 'QCOM', 'HON', 'NEE',
    'RTX', 'LOW', 'INTU', 'AMT', 'SPGI', 'CAT', 'GS', 'IBM', 'BLK', 'SBUX',
    'CVS', 'AMAT', 'ISRG', 'MU', 'AXP', 'GILD', 'SCHW', 'MDLZ', 'PYPL', 'CI',
    'ZTS', 'PLD', 'CB', 'MO', 'ADI', 'REGN', 'NOW', 'FISV', 'CSX', 'SO',
    'MMC', 'DUK', 'CCI', 'BDX', 'ITW', 'BSX', 'EQIX', 'ICE', 'AON', 'HUM',
    'WM', 'PNC', 'TGT', 'APD', 'CL', 'SHW', 'FIS', 'CME', 'MCO', 'USB',
    'EL', 'SYK', 'KLAC', 'ECL', 'ADP', 'SNPS', 'GD', 'LHX', 'MCK', 'CDNS'
]

def get_existing_stocks(conn):
    """Get list of stocks already in database"""
    cursor = conn.cursor()
    cursor.execute("SELECT ticker FROM stocks")
    return {row[0] for row in cursor.fetchall()}

def get_or_create_stock(conn, ticker):
    """Get stock ID or create new stock record"""
    cursor = conn.cursor()
    
    # Check if stock exists
    cursor.execute("SELECT id FROM stocks WHERE ticker = %s", (ticker,))
    result = cursor.fetchone()
    
    if result:
        return result[0]
    
    # Get default exchange
    cursor.execute("SELECT id FROM exchanges WHERE name = 'NYSE'")
    exchange = cursor.fetchone()
    if not exchange:
        cursor.execute("INSERT INTO exchanges (name) VALUES ('NYSE') RETURNING id")
        exchange = cursor.fetchone()
    
    # Get default sector
    cursor.execute("SELECT id FROM sectors WHERE name = 'Technology'")
    sector = cursor.fetchone()
    if not sector:
        cursor.execute("INSERT INTO sectors (name) VALUES ('Technology') RETURNING id")
        sector = cursor.fetchone()
    
    # Create stock
    cursor.execute("""
        INSERT INTO stocks (ticker, name, exchange_id, sector_id)
        VALUES (%s, %s, %s, %s)
        RETURNING id
    """, (ticker, ticker, exchange[0], sector[0]))
    
    stock_id = cursor.fetchone()[0]
    conn.commit()
    return stock_id

def load_stock_data(conn, ticker, days=365):
    """Load historical data for a single stock"""
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Download data from Yahoo Finance
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            logger.warning(f"No data found for {ticker}")
            return 0
        
        # Get or create stock record
        stock_id = get_or_create_stock(conn, ticker)
        
        # Prepare data for insertion
        cursor = conn.cursor()
        records = []
        
        for date, row in df.iterrows():
            records.append((
                stock_id,
                date.date(),
                float(row['Open']),
                float(row['High']),
                float(row['Low']),
                float(row['Close']),
                int(row['Volume'])
            ))
        
        # Bulk insert with conflict handling
        execute_batch(
            cursor,
            """
            INSERT INTO price_history (stock_id, date, open, high, low, close, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (stock_id, date) DO UPDATE
            SET open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume
            """,
            records,
            page_size=100
        )
        
        conn.commit()
        logger.info(f"✓ Loaded {len(records)} records for {ticker}")
        return len(records)
        
    except Exception as e:
        logger.error(f"✗ Error loading {ticker}: {e}")
        conn.rollback()
        return 0

def save_progress(progress_file, loaded_stocks):
    """Save progress to file"""
    with open(progress_file, 'w') as f:
        json.dump({'loaded': list(loaded_stocks), 'timestamp': datetime.now().isoformat()}, f)

def load_progress(progress_file):
    """Load progress from file"""
    try:
        with open(progress_file, 'r') as f:
            data = json.load(f)
            return set(data.get('loaded', []))
    except:
        return set()

def main():
    """Main execution function"""
    logger.info("=" * 60)
    logger.info("Background Data Loader - Loading Extended Stock List")
    logger.info("=" * 60)
    
    progress_file = 'loading_progress.json'
    
    # Connect to database
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        logger.info("Database connected")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        sys.exit(1)
    
    try:
        # Get existing stocks
        existing = get_existing_stocks(conn)
        loaded_previously = load_progress(progress_file)
        
        # Filter stocks to load
        stocks_to_load = [s for s in NEXT_STOCKS if s not in existing and s not in loaded_previously]
        
        logger.info(f"Existing stocks in database: {len(existing)}")
        logger.info(f"New stocks to load: {len(stocks_to_load)}")
        
        if not stocks_to_load:
            logger.info("All stocks already loaded!")
            return
        
        # Load data for each stock
        total_records = 0
        successful = 0
        loaded_stocks = loaded_previously.copy()
        
        for i, ticker in enumerate(stocks_to_load, 1):
            logger.info(f"[{i}/{len(stocks_to_load)}] Loading {ticker}...")
            records = load_stock_data(conn, ticker)
            
            if records > 0:
                successful += 1
                total_records += records
                loaded_stocks.add(ticker)
                
                # Save progress every 5 stocks
                if i % 5 == 0:
                    save_progress(progress_file, loaded_stocks)
            
            # Rate limiting
            time.sleep(1.5)
            
            # Status update every 10 stocks
            if i % 10 == 0:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(DISTINCT stock_id) FROM price_history")
                total_stocks = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM price_history")
                total_prices = cursor.fetchone()[0]
                logger.info(f"Progress: {total_stocks} stocks, {total_prices:,} price records in database")
        
        # Final save
        save_progress(progress_file, loaded_stocks)
        
        # Print summary
        logger.info("=" * 60)
        logger.info(f"Batch Complete! Loaded {successful}/{len(stocks_to_load)} stocks")
        logger.info(f"Total new records: {total_records:,}")
        
        # Final database stats
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM stocks")
        stock_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM price_history")
        price_count = cursor.fetchone()[0]
        
        logger.info(f"Database totals: {stock_count} stocks, {price_count:,} price records")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    finally:
        conn.close()

if __name__ == "__main__":
    main()