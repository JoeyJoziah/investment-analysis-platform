#!/usr/bin/env python3
"""
Simple Mock Data Generator - Adapts to existing schema
"""

import random
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import psycopg2
from psycopg2.extras import execute_batch
from dotenv import load_dotenv
import logging
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path)

def generate_mock_data(num_stocks=10):
    """Generate mock data for testing"""
    
    db_config = {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'port': int(os.getenv('POSTGRES_PORT', 5432)),
        'database': os.getenv('POSTGRES_DB', 'investment_db'),
        'user': os.getenv('POSTGRES_USER', 'postgres'),
        'password': os.getenv('POSTGRES_PASSWORD', '9v1g^OV9XUwzUP6cEgCYgNOE')
    }
    
    # Sample tickers
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V',
               'PG', 'UNH', 'HD', 'MA', 'DIS', 'BAC', 'ADBE', 'CRM', 'NFLX', 'KO'][:num_stocks]
    
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()
    
    try:
        logger.info("Starting mock data generation...")
        
        # Get existing IDs
        cursor.execute("SELECT id FROM exchanges LIMIT 1")
        result = cursor.fetchone()
        exchange_id = result[0] if result else 1
        
        cursor.execute("SELECT id FROM sectors LIMIT 1")
        result = cursor.fetchone()
        sector_id = result[0] if result else 1
        
        cursor.execute("SELECT id FROM industries LIMIT 1")
        result = cursor.fetchone()
        industry_id = result[0] if result else 1
        
        # Insert stocks
        logger.info(f"Inserting {num_stocks} stocks...")
        stock_ids = {}
        for ticker in tickers:
            cursor.execute("""
                INSERT INTO stocks (ticker, name, exchange_id, sector_id, industry_id, is_active)
                VALUES (%s, %s, %s, %s, %s, true)
                ON CONFLICT (ticker) DO UPDATE SET
                    is_active = true
                RETURNING id
            """, (ticker, f"{ticker} Corporation", exchange_id, sector_id, industry_id))
            stock_ids[ticker] = cursor.fetchone()[0]
        
        # Generate price history for each stock
        logger.info("Generating price history...")
        for ticker, stock_id in stock_ids.items():
            base_price = random.uniform(50, 500)
            for days_ago in range(30):
                date = datetime.now().date() - timedelta(days=days_ago)
                
                # Random walk
                change = random.uniform(-0.05, 0.05)
                open_price = base_price * (1 + change)
                close_price = open_price * (1 + random.uniform(-0.02, 0.02))
                high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.01))
                low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.01))
                volume = random.randint(1000000, 50000000)
                
                cursor.execute("""
                    INSERT INTO price_history (stock_id, date, open, high, low, close, volume)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (stock_id, date) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume
                """, (stock_id, date, round(open_price, 2), round(high_price, 2),
                     round(low_price, 2), round(close_price, 2), volume))
                
                base_price = close_price
        
        # Generate technical indicators
        logger.info("Generating technical indicators...")
        for ticker, stock_id in stock_ids.items():
            cursor.execute("""
                INSERT INTO technical_indicators 
                (stock_id, date, sma_20, sma_50, rsi_14, macd, macd_signal, 
                 bollinger_upper, bollinger_lower, bollinger_middle)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (stock_id, date) DO UPDATE SET
                    sma_20 = EXCLUDED.sma_20,
                    rsi_14 = EXCLUDED.rsi_14
            """, (
                stock_id,
                datetime.now(),
                round(random.uniform(100, 400), 2),  # sma_20
                round(random.uniform(100, 400), 2),  # sma_50
                round(random.uniform(30, 70), 2),    # rsi_14
                round(random.uniform(-5, 5), 2),     # macd
                round(random.uniform(-5, 5), 2),     # macd_signal
                round(random.uniform(110, 450), 2),  # bollinger_upper
                round(random.uniform(90, 390), 2),   # bollinger_lower
                round(random.uniform(100, 400), 2)   # bollinger_middle
            ))
        
        # Generate simple recommendations
        logger.info("Generating recommendations...")
        for ticker in random.sample(list(stock_ids.keys()), min(5, len(stock_ids))):
            stock_id = stock_ids[ticker]
            cursor.execute("""
                INSERT INTO recommendations
                (stock_id, action, confidence, target_price, stop_loss, 
                 reasoning, technical_score, fundamental_score, sentiment_score,
                 risk_level, time_horizon_days, is_active, created_at, priority)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, true, %s, %s)
            """, (
                stock_id,
                random.choice(['buy', 'hold', 'strong_buy']),
                round(random.uniform(0.6, 0.95), 2),
                round(random.uniform(100, 500), 2),
                round(random.uniform(80, 100), 2),
                'Generated by mock data generator for testing',
                round(random.uniform(0.5, 0.9), 2),
                round(random.uniform(0.5, 0.9), 2),
                round(random.uniform(0.4, 0.8), 2),
                random.choice(['LOW', 'MEDIUM', 'HIGH']),
                random.choice([30, 90, 365]),
                datetime.now(),
                random.randint(1, 10)
            ))
        
        conn.commit()
        
        # Print summary
        cursor.execute("SELECT COUNT(*) FROM stocks WHERE is_active = true")
        stock_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM price_history WHERE date >= CURRENT_DATE - INTERVAL '30 days'")
        price_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM recommendations WHERE is_active = true")
        rec_count = cursor.fetchone()[0]
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ Mock Data Generation Complete!")
        logger.info("=" * 60)
        logger.info(f"Active Stocks: {stock_count}")
        logger.info(f"Price Records (30 days): {price_count}")
        logger.info(f"Active Recommendations: {rec_count}")
        logger.info("=" * 60)
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Error: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate simple mock data')
    parser.add_argument('--stocks', type=int, default=10, help='Number of stocks')
    args = parser.parse_args()
    
    generate_mock_data(args.stocks)