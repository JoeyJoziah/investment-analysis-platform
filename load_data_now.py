#!/usr/bin/env python3
"""
Quick data loader for Investment Analysis Platform
Loads historical stock data using Yahoo Finance (free)
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import execute_batch
import logging
import sys
from tqdm import tqdm
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
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

# Top S&P 500 stocks to load
TOP_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
    'META', 'TSLA', 'BRK-B', 'JPM', 'JNJ',
    'V', 'UNH', 'WMT', 'MA', 'PG',
    'HD', 'DIS', 'BAC', 'XOM', 'AVGO'
]

def create_tables(conn):
    """Create necessary database tables if they don't exist"""
    cursor = conn.cursor()
    
    # Create exchanges table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS exchanges (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create sectors table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sectors (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create stocks table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stocks (
            id SERIAL PRIMARY KEY,
            ticker VARCHAR(20) UNIQUE NOT NULL,
            name VARCHAR(255),
            exchange_id INTEGER REFERENCES exchanges(id),
            sector_id INTEGER REFERENCES sectors(id),
            market_cap BIGINT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create price_history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS price_history (
            id SERIAL PRIMARY KEY,
            stock_id INTEGER REFERENCES stocks(id),
            date DATE NOT NULL,
            open DECIMAL(12,4),
            high DECIMAL(12,4),
            low DECIMAL(12,4),
            close DECIMAL(12,4),
            volume BIGINT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(stock_id, date)
        )
    """)
    
    # Create indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_history_stock_date ON price_history(stock_id, date DESC)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_history_date ON price_history(date DESC)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_stocks_ticker ON stocks(ticker)")
    
    conn.commit()
    logger.info("Database tables created/verified")

def get_or_create_stock(conn, ticker):
    """Get stock ID or create new stock record"""
    cursor = conn.cursor()
    
    # Check if stock exists
    cursor.execute("SELECT id FROM stocks WHERE ticker = %s", (ticker,))
    result = cursor.fetchone()
    
    if result:
        return result[0]
    
    # Get default exchange (NYSE)
    cursor.execute("SELECT id FROM exchanges WHERE name = 'NYSE'")
    exchange = cursor.fetchone()
    if not exchange:
        cursor.execute("INSERT INTO exchanges (name) VALUES ('NYSE') RETURNING id")
        exchange = cursor.fetchone()
    
    # Get default sector (Technology)
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
        
        logger.info(f"Downloading data for {ticker}...")
        
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
        logger.info(f"Loaded {len(records)} records for {ticker}")
        return len(records)
        
    except Exception as e:
        logger.error(f"Error loading {ticker}: {e}")
        conn.rollback()
        return 0

def main():
    """Main execution function"""
    logger.info("=" * 60)
    logger.info("Investment Analysis Platform - Data Loader")
    logger.info("=" * 60)
    
    # Connect to database
    try:
        logger.info("Connecting to database...")
        conn = psycopg2.connect(**DB_CONFIG)
        logger.info("Database connected successfully")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        logger.info("Make sure PostgreSQL is running: docker-compose up -d postgres")
        sys.exit(1)
    
    try:
        # Create tables
        create_tables(conn)
        
        # Load data for each stock
        total_records = 0
        successful_stocks = 0
        
        logger.info(f"Loading data for {len(TOP_STOCKS)} stocks...")
        
        for ticker in tqdm(TOP_STOCKS, desc="Loading stocks"):
            records = load_stock_data(conn, ticker)
            if records > 0:
                successful_stocks += 1
                total_records += records
            
            # Small delay to be respectful to Yahoo Finance
            time.sleep(1)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("DATA LOADING COMPLETE!")
        logger.info(f"Stocks loaded: {successful_stocks}/{len(TOP_STOCKS)}")
        logger.info(f"Total records: {total_records:,}")
        logger.info("=" * 60)
        
        # Verify data
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM stocks")
        stock_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM price_history")
        price_count = cursor.fetchone()[0]
        
        logger.info(f"Database now contains:")
        logger.info(f"  - {stock_count} stocks")
        logger.info(f"  - {price_count:,} price records")
        
    except Exception as e:
        logger.error(f"Error during data loading: {e}")
        sys.exit(1)
    finally:
        conn.close()

if __name__ == "__main__":
    main()