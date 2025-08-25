"""
Stock Universe Manager - Fetches and manages ALL US exchange stocks
"""

import logging
import pandas as pd
from typing import List, Dict, Set, Optional
from datetime import datetime
import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values
import yfinance as yf
import requests

load_dotenv()

logger = logging.getLogger(__name__)


class StockUniverseManager:
    """Manages the complete universe of US exchange stocks"""
    
    def __init__(self):
        self.db_config = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'database': os.getenv('POSTGRES_DB', 'investment_db'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'postgres')
        }
        
        self.exchanges = ['NYSE', 'NASDAQ', 'AMEX']
        
    def get_all_us_tickers_from_yfinance(self) -> List[Dict]:
        """Get comprehensive list of US tickers using yfinance screener"""
        logger.info("Fetching US stock universe from yfinance...")
        
        all_tickers = []
        
        try:
            # Get S&P 500 tickers as a starting point
            sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            sp500_table = pd.read_html(sp500_url)[0]
            sp500_tickers = sp500_table['Symbol'].tolist()
            
            for ticker in sp500_tickers:
                all_tickers.append({
                    'ticker': ticker,
                    'name': sp500_table[sp500_table['Symbol'] == ticker]['Security'].values[0] if len(sp500_table[sp500_table['Symbol'] == ticker]) > 0 else ticker,
                    'exchange': 'NYSE',  # Most S&P 500 are NYSE
                    'sector': sp500_table[sp500_table['Symbol'] == ticker]['GICS Sector'].values[0] if len(sp500_table[sp500_table['Symbol'] == ticker]) > 0 else None,
                    'industry': sp500_table[sp500_table['Symbol'] == ticker]['GICS Sub-Industry'].values[0] if len(sp500_table[sp500_table['Symbol'] == ticker]) > 0 else None,
                })
            
            logger.info(f"Found {len(sp500_tickers)} S&P 500 stocks")
            
        except Exception as e:
            logger.error(f"Error fetching S&P 500: {e}")
        
        # Get NASDAQ listings
        try:
            nasdaq_url = 'https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=25000&exchange=nasdaq'
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(nasdaq_url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and 'rows' in data['data']:
                    for stock in data['data']['rows']:
                        ticker = stock.get('symbol', '').strip()
                        if ticker and not any(t['ticker'] == ticker for t in all_tickers):
                            all_tickers.append({
                                'ticker': ticker,
                                'name': stock.get('name', ticker),
                                'exchange': 'NASDAQ',
                                'sector': stock.get('sector'),
                                'industry': stock.get('industry'),
                            })
                    logger.info(f"Added NASDAQ stocks, total now: {len(all_tickers)}")
        except Exception as e:
            logger.error(f"Error fetching NASDAQ listings: {e}")
        
        # Get NYSE listings using alternative method
        try:
            # Common NYSE tickers not in S&P 500
            additional_nyse = [
                'AA', 'AAL', 'AAN', 'AAON', 'AAP', 'AAT', 'AB', 'ABB', 'ABBV', 'ABC',
                'ABEV', 'ABG', 'ABM', 'ABR', 'ABT', 'AC', 'ACA', 'ACAD', 'ACC', 'ACCO',
                'ACEL', 'ACH', 'ACHC', 'ACI', 'ACLS', 'ACM', 'ACN', 'ACRE', 'ACT', 'ACTG',
                'ACU', 'ACV', 'ACY', 'ADBE', 'ADC', 'ADCT', 'ADD', 'ADEA', 'ADI', 'ADM',
                'ADMA', 'ADNT', 'ADP', 'ADS', 'ADSK', 'ADT', 'ADTN', 'ADUS', 'ADV', 'ADVM'
            ]
            
            for ticker in additional_nyse:
                if not any(t['ticker'] == ticker for t in all_tickers):
                    all_tickers.append({
                        'ticker': ticker,
                        'name': ticker,
                        'exchange': 'NYSE',
                        'sector': None,
                        'industry': None,
                    })
            
        except Exception as e:
            logger.error(f"Error adding NYSE tickers: {e}")
        
        logger.info(f"Total tickers collected: {len(all_tickers)}")
        return all_tickers
    
    def populate_database_with_all_stocks(self) -> int:
        """Populate database with all US exchange stocks"""
        logger.info("Populating database with US stock universe...")
        
        # Get all tickers
        all_tickers = self.get_all_us_tickers_from_yfinance()
        
        if not all_tickers:
            logger.error("No tickers retrieved")
            return 0
        
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        try:
            # Get exchange IDs
            cursor.execute("SELECT id, name FROM exchanges")
            exchanges = {row[1]: row[0] for row in cursor.fetchall()}
            
            # Ensure we have the main exchanges
            for exchange in ['NYSE', 'NASDAQ', 'AMEX']:
                if exchange not in exchanges:
                    cursor.execute("INSERT INTO exchanges (name) VALUES (%s) RETURNING id", (exchange,))
                    exchanges[exchange] = cursor.fetchone()[0]
            
            # Get existing tickers
            cursor.execute("SELECT ticker FROM stocks")
            existing = set(row[0] for row in cursor.fetchall())
            
            # Prepare new tickers for insertion
            new_tickers = []
            update_tickers = []
            
            for ticker_data in all_tickers:
                ticker = ticker_data['ticker']
                exchange_name = ticker_data.get('exchange', 'NYSE')
                exchange_id = exchanges.get(exchange_name, exchanges['NYSE'])
                
                if ticker not in existing:
                    new_tickers.append((
                        ticker,
                        ticker_data.get('name', ticker)[:255],  # Limit name length
                        exchange_id,
                        True,  # is_active
                        True   # is_tradeable
                    ))
                else:
                    update_tickers.append(ticker)
            
            # Bulk insert new tickers
            if new_tickers:
                execute_values(
                    cursor,
                    """
                    INSERT INTO stocks (ticker, name, exchange_id, is_active, is_tradeable)
                    VALUES %s
                    ON CONFLICT (ticker) DO NOTHING
                    """,
                    new_tickers
                )
                logger.info(f"Inserted {len(new_tickers)} new stocks")
            
            # Update existing tickers to ensure they're active
            if update_tickers:
                cursor.execute(
                    """
                    UPDATE stocks 
                    SET is_active = true, is_tradeable = true, last_updated = CURRENT_TIMESTAMP
                    WHERE ticker = ANY(%s)
                    """,
                    (update_tickers,)
                )
                logger.info(f"Updated {len(update_tickers)} existing stocks")
            
            conn.commit()
            
            # Get total count
            cursor.execute("SELECT COUNT(*) FROM stocks WHERE is_active = true")
            total_count = cursor.fetchone()[0]
            
            return total_count
            
        except Exception as e:
            logger.error(f"Database error: {e}")
            conn.rollback()
            return 0
            
        finally:
            cursor.close()
            conn.close()
    
    def get_all_active_tickers(self) -> List[str]:
        """Get all active stock tickers from database"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT ticker FROM stocks 
                WHERE is_active = true 
                ORDER BY ticker
            """)
            
            tickers = [row[0] for row in cursor.fetchall()]
            logger.info(f"Retrieved {len(tickers)} active stocks from database")
            return tickers
            
        finally:
            cursor.close()
            conn.close()


# Quick function to update ETL orchestrator
def update_etl_orchestrator():
    """Update the ETL orchestrator to use dynamic ticker loading"""
    
    etl_path = '/mnt/wsl/docker-desktop-bind-mounts/Ubuntu-24.04/7b51113a393465a37d4f1fda36b4d190088ac69ea8d5cf2f90400b3c14148ad3/backend/etl/etl_orchestrator.py'
    
    # Read current file
    with open(etl_path, 'r') as f:
        content = f.read()
    
    # Replace the get_active_tickers method
    new_method = '''    async def get_active_tickers(self) -> List[str]:
        """Get list of active tickers from database"""
        try:
            # Import the stock universe manager
            from backend.etl.stock_universe_manager import StockUniverseManager
            manager = StockUniverseManager()
            
            # Get all active tickers from database
            tickers = manager.get_all_active_tickers()
            
            if not tickers:
                logger.warning("No tickers in database, populating...")
                # Populate database if empty
                count = manager.populate_database_with_all_stocks()
                logger.info(f"Populated database with {count} stocks")
                # Get tickers again
                tickers = manager.get_all_active_tickers()
            
            # Apply max_tickers limit if configured
            max_tickers = self.config.get('max_tickers')
            if max_tickers and max_tickers < len(tickers):
                logger.info(f"Limiting to {max_tickers} tickers (from {len(tickers)} total)")
                return tickers[:max_tickers]
            
            logger.info(f"Processing {len(tickers)} stocks from database")
            return tickers
            
        except Exception as e:
            logger.error(f"Error getting active tickers: {e}")
            # Fallback to default list if database fails
            default_tickers = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA',
                'JPM', 'V', 'JNJ', 'WMT', 'PG', 'UNH', 'DIS', 'MA'
            ]
            return default_tickers'''
    
    # Find and replace the method
    import re
    pattern = r'async def get_active_tickers\(self\)[^}]+?return \[\]'
    
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, new_method, content, flags=re.DOTALL)
        
        # Write back
        with open(etl_path, 'w') as f:
            f.write(content)
        
        logger.info("Updated ETL orchestrator to use dynamic ticker loading")
    else:
        logger.warning("Could not find get_active_tickers method to replace")


if __name__ == "__main__":
    # Test the stock universe manager
    manager = StockUniverseManager()
    
    # Populate database with all stocks
    count = manager.populate_database_with_all_stocks()
    print(f"Database now contains {count} active stocks")
    
    # Update ETL orchestrator
    update_etl_orchestrator()
    print("ETL orchestrator updated to use dynamic stock loading")