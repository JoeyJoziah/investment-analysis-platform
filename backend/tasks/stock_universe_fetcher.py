"""
Comprehensive Stock Universe Fetcher
Fetches ALL US exchange stocks (6000+) from multiple sources
"""

import os
import time
import logging
import requests
import pandas as pd
import yfinance as yf
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
from datetime import datetime
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StockInfo:
    symbol: str
    name: str
    exchange: str
    market_cap: float = None
    sector: str = None
    industry: str = None

class ComprehensiveStockFetcher:
    """
    Fetches comprehensive list of all US stocks from multiple sources
    to achieve 6000+ stock coverage across NYSE, NASDAQ, and AMEX
    """
    
    def __init__(self):
        self.stocks = {}  # Dict[str, StockInfo]
        self.session = requests.Session()
        self.finnhub_api_key = os.getenv('FINNHUB_API_KEY')
        self.polygon_api_key = os.getenv('POLYGON_API_KEY')
        
        # Database configuration
        self.db_config = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'database': os.getenv('POSTGRES_DB', 'investment_db'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'postgres')
        }
        
        # Rate limiting counters
        self.finnhub_calls = 0
        self.polygon_calls = 0
        self.last_finnhub_reset = time.time()
        self.last_polygon_reset = time.time()
    
    def rate_limit_finnhub(self):
        """Rate limit Finnhub API calls (60/minute)"""
        current_time = time.time()
        if current_time - self.last_finnhub_reset >= 60:
            self.finnhub_calls = 0
            self.last_finnhub_reset = current_time
        
        if self.finnhub_calls >= 55:  # Buffer of 5 calls
            sleep_time = 60 - (current_time - self.last_finnhub_reset)
            if sleep_time > 0:
                logger.info(f"Finnhub rate limit reached. Sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
                self.finnhub_calls = 0
                self.last_finnhub_reset = time.time()
    
    def rate_limit_polygon(self):
        """Rate limit Polygon API calls (5/minute)"""
        current_time = time.time()
        if current_time - self.last_polygon_reset >= 60:
            self.polygon_calls = 0
            self.last_polygon_reset = current_time
        
        if self.polygon_calls >= 4:  # Buffer of 1 call
            sleep_time = 60 - (current_time - self.last_polygon_reset)
            if sleep_time > 0:
                logger.info(f"Polygon rate limit reached. Sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
                self.polygon_calls = 0
                self.last_polygon_reset = time.time()
    
    def fetch_nasdaq_listed_stocks(self) -> List[StockInfo]:
        """Fetch NASDAQ listed stocks from official NASDAQ API"""
        logger.info("Fetching NASDAQ listed stocks...")
        stocks = []
        
        try:
            # NASDAQ provides free CSV downloads for all listed stocks
            nasdaq_url = "https://api.nasdaq.com/api/screener/stocks"
            params = {
                "tableonly": "true",
                "limit": "25000",
                "offset": "0",
                "download": "true"
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = self.session.get(nasdaq_url, params=params, headers=headers, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and 'rows' in data['data']:
                    for row in data['data']['rows']:
                        try:
                            symbol = row.get('symbol', '').strip()
                            name = row.get('name', '').strip()
                            
                            # Filter valid symbols
                            if symbol and len(symbol) <= 5 and not any(c in symbol for c in ['/', '.', '-']):
                                stocks.append(StockInfo(
                                    symbol=symbol,
                                    name=name[:255] if name else symbol,
                                    exchange='NASDAQ',
                                    sector=row.get('sector'),
                                    industry=row.get('industry')
                                ))
                        except Exception as e:
                            logger.debug(f"Error processing NASDAQ stock {row}: {e}")
                            continue
            
            logger.info(f"Fetched {len(stocks)} NASDAQ stocks")
            
        except Exception as e:
            logger.error(f"Error fetching NASDAQ stocks: {e}")
        
        return stocks
    
    def fetch_nyse_stocks(self) -> List[StockInfo]:
        """Fetch NYSE listed stocks"""
        logger.info("Fetching NYSE listed stocks...")
        stocks = []
        
        try:
            # Get S&P 500 companies (mostly NYSE)
            sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            sp500_df = pd.read_html(sp500_url)[0]
            
            for _, row in sp500_df.iterrows():
                try:
                    symbol = str(row['Symbol']).strip().replace('.', '-')
                    name = str(row['Security']).strip()
                    sector = str(row.get('GICS Sector', '')).strip()
                    industry = str(row.get('GICS Sub-Industry', '')).strip()
                    
                    if symbol and symbol != 'nan':
                        stocks.append(StockInfo(
                            symbol=symbol,
                            name=name[:255] if name else symbol,
                            exchange='NYSE',
                            sector=sector if sector != 'nan' else None,
                            industry=industry if industry != 'nan' else None
                        ))
                        
                except Exception as e:
                    logger.debug(f"Error processing S&P 500 stock: {e}")
                    continue
            
            # Add Dow Jones stocks
            try:
                dji_url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
                dji_tables = pd.read_html(dji_url)
                
                # Find the table with company data
                for table in dji_tables:
                    if 'Symbol' in table.columns:
                        for _, row in table.iterrows():
                            try:
                                symbol = str(row['Symbol']).strip().replace('.', '-')
                                if 'Company' in row:
                                    name = str(row['Company']).strip()
                                elif 'Name' in row:
                                    name = str(row['Name']).strip()
                                else:
                                    name = symbol
                                
                                if symbol and symbol != 'nan' and not any(s.symbol == symbol for s in stocks):
                                    stocks.append(StockInfo(
                                        symbol=symbol,
                                        name=name[:255] if name else symbol,
                                        exchange='NYSE'
                                    ))
                            except Exception as e:
                                continue
                        break
                        
            except Exception as e:
                logger.debug(f"Error fetching Dow Jones stocks: {e}")
            
            logger.info(f"Fetched {len(stocks)} NYSE stocks")
            
        except Exception as e:
            logger.error(f"Error fetching NYSE stocks: {e}")
        
        return stocks
    
    def fetch_finnhub_stocks(self) -> List[StockInfo]:
        """Fetch comprehensive stock list from Finnhub API"""
        logger.info("Fetching stocks from Finnhub...")
        stocks = []
        
        if not self.finnhub_api_key:
            logger.warning("Finnhub API key not found")
            return stocks
        
        try:
            self.rate_limit_finnhub()
            
            url = "https://finnhub.io/api/v1/stock/symbol"
            params = {
                'exchange': 'US',
                'token': self.finnhub_api_key
            }
            
            response = self.session.get(url, params=params, timeout=30)
            self.finnhub_calls += 1
            
            if response.status_code == 200:
                data = response.json()
                
                for stock_data in data:
                    try:
                        symbol = stock_data.get('symbol', '').strip()
                        name = stock_data.get('description', '').strip()
                        exchange = stock_data.get('mic', '').strip()
                        stock_type = stock_data.get('type', '').upper()
                        
                        # Skip non-common stock types
                        if stock_type not in ['COMMON STOCK', 'CS', '']:
                            continue
                        
                        # Map exchange codes to readable names
                        exchange_map = {
                            'XNYS': 'NYSE',
                            'XNAS': 'NASDAQ',
                            'ARCX': 'AMEX',
                            'XASE': 'AMEX',
                            'BATS': 'BATS',
                            'XNCM': 'NASDAQ',
                            'XNGS': 'NASDAQ'
                        }
                        
                        exchange = exchange_map.get(exchange, 'NYSE')
                        
                        # Filter valid symbols
                        if symbol and len(symbol) <= 5 and exchange in ['NYSE', 'NASDAQ', 'AMEX']:
                            stocks.append(StockInfo(
                                symbol=symbol,
                                name=name[:255] if name else symbol,
                                exchange=exchange
                            ))
                            
                    except Exception as e:
                        logger.debug(f"Error processing Finnhub stock: {e}")
                        continue
            
            logger.info(f"Fetched {len(stocks)} stocks from Finnhub")
            
        except Exception as e:
            logger.error(f"Error fetching Finnhub stocks: {e}")
        
        return stocks
    
    def fetch_polygon_stocks(self) -> List[StockInfo]:
        """Fetch stocks from Polygon.io API"""
        logger.info("Fetching stocks from Polygon...")
        stocks = []
        
        if not self.polygon_api_key:
            logger.warning("Polygon API key not found")
            return stocks
        
        try:
            self.rate_limit_polygon()
            
            url = "https://api.polygon.io/v3/reference/tickers"
            params = {
                'market': 'stocks',
                'active': 'true',
                'limit': 1000,  # Max limit for free tier
                'apikey': self.polygon_api_key
            }
            
            response = self.session.get(url, params=params, timeout=30)
            self.polygon_calls += 1
            
            if response.status_code == 200:
                data = response.json()
                
                if 'results' in data:
                    for stock_data in data['results']:
                        try:
                            symbol = stock_data.get('ticker', '').strip()
                            name = stock_data.get('name', '').strip()
                            exchange = stock_data.get('primary_exchange', '').strip()
                            stock_type = stock_data.get('type', '').upper()
                            
                            # Skip non-common stock types
                            if stock_type not in ['CS', '']:
                                continue
                            
                            # Map exchange codes
                            if 'NASDAQ' in exchange.upper():
                                exchange = 'NASDAQ'
                            elif 'NYSE' in exchange.upper():
                                exchange = 'NYSE'
                            elif 'AMEX' in exchange.upper() or 'ARCA' in exchange.upper():
                                exchange = 'AMEX'
                            else:
                                continue
                            
                            if symbol and len(symbol) <= 5:
                                stocks.append(StockInfo(
                                    symbol=symbol,
                                    name=name[:255] if name else symbol,
                                    exchange=exchange,
                                    market_cap=stock_data.get('market_cap')
                                ))
                                
                        except Exception as e:
                            logger.debug(f"Error processing Polygon stock: {e}")
                            continue
            
            logger.info(f"Fetched {len(stocks)} stocks from Polygon")
            
        except Exception as e:
            logger.error(f"Error fetching Polygon stocks: {e}")
        
        return stocks
    
    def fetch_comprehensive_stock_universe(self) -> Dict[str, StockInfo]:
        """
        Fetch comprehensive stock universe from all sources
        Returns dictionary with symbol as key and StockInfo as value
        """
        logger.info("Starting comprehensive stock universe fetch...")
        
        all_stocks = []
        
        # Fetch from all sources
        sources = [
            ("NASDAQ", self.fetch_nasdaq_listed_stocks),
            ("NYSE", self.fetch_nyse_stocks),
            ("Finnhub", self.fetch_finnhub_stocks),
            ("Polygon", self.fetch_polygon_stocks)
        ]
        
        for source_name, fetch_func in sources:
            try:
                logger.info(f"Fetching from {source_name}...")
                stocks = fetch_func()
                all_stocks.extend(stocks)
                logger.info(f"Got {len(stocks)} stocks from {source_name}")
            except Exception as e:
                logger.error(f"{source_name} fetch failed: {e}")
        
        # Deduplicate and merge information
        stock_universe = {}
        
        for stock in all_stocks:
            symbol = stock.symbol.upper()
            
            # Skip invalid symbols
            if not symbol or len(symbol) > 5 or any(c in symbol for c in ['/', '$', '^']):
                continue
            
            if symbol not in stock_universe:
                stock_universe[symbol] = stock
            else:
                # Merge information from multiple sources
                existing = stock_universe[symbol]
                
                # Update with more detailed information
                if stock.market_cap and not existing.market_cap:
                    existing.market_cap = stock.market_cap
                if stock.sector and not existing.sector:
                    existing.sector = stock.sector
                if stock.industry and not existing.industry:
                    existing.industry = stock.industry
                if len(stock.name) > len(existing.name):
                    existing.name = stock.name
        
        logger.info(f"Comprehensive stock universe: {len(stock_universe)} unique stocks")
        
        # Log distribution by exchange
        exchange_counts = {}
        for stock in stock_universe.values():
            exchange_counts[stock.exchange] = exchange_counts.get(stock.exchange, 0) + 1
        
        for exchange, count in exchange_counts.items():
            logger.info(f"  {exchange}: {count} stocks")
        
        return stock_universe
    
    def save_stocks_to_database(self, stock_universe: Dict[str, StockInfo]) -> int:
        """Save stocks to database and return count of new stocks added"""
        logger.info("Saving stocks to database...")
        
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
            
            # Get existing stocks
            cursor.execute("SELECT ticker FROM stocks")
            existing_symbols = set(row[0] for row in cursor.fetchall())
            
            logger.info(f"Found {len(existing_symbols)} existing stocks in database")
            
            # Prepare new stocks for insertion
            new_stocks = []
            
            for symbol, stock_info in stock_universe.items():
                if symbol not in existing_symbols:
                    exchange_id = exchanges.get(stock_info.exchange, exchanges['NYSE'])
                    
                    new_stocks.append((
                        symbol,
                        stock_info.name[:255] if stock_info.name else symbol,
                        exchange_id,
                        True,  # is_active
                        True   # is_tradeable
                    ))
            
            # Bulk insert new stocks
            if new_stocks:
                execute_values(
                    cursor,
                    """
                    INSERT INTO stocks (ticker, name, exchange_id, is_active, is_tradeable)
                    VALUES %s
                    ON CONFLICT (ticker) DO NOTHING
                    """,
                    new_stocks,
                    template="(%s, %s, %s, %s, %s)"
                )
                logger.info(f"Inserted {len(new_stocks)} new stocks")
            
            # Update existing stocks to ensure they're active
            cursor.execute(
                """
                UPDATE stocks 
                SET is_active = true, is_tradeable = true, last_updated = CURRENT_TIMESTAMP
                WHERE ticker = ANY(%s)
                """,
                (list(stock_universe.keys()),)
            )
            
            conn.commit()
            
            # Get total count
            cursor.execute("SELECT COUNT(*) FROM stocks WHERE is_active = true")
            total_count = cursor.fetchone()[0]
            
            logger.info(f"Total active stocks in database: {total_count}")
            
            return len(new_stocks)
            
        except Exception as e:
            logger.error(f"Database error: {e}")
            conn.rollback()
            return 0
            
        finally:
            cursor.close()
            conn.close()

def expand_stock_universe():
    """Main function to expand stock universe to 6000+ stocks"""
    logger.info("="*60)
    logger.info("Starting stock universe expansion to 6000+ stocks...")
    logger.info("="*60)
    
    fetcher = ComprehensiveStockFetcher()
    
    # Fetch comprehensive stock universe
    stock_universe = fetcher.fetch_comprehensive_stock_universe()
    
    if len(stock_universe) < 1000:
        logger.warning(f"Only fetched {len(stock_universe)} stocks, which seems low")
        logger.info("Attempting to fetch additional stocks...")
    
    # Save to database
    new_stocks_count = fetcher.save_stocks_to_database(stock_universe)
    
    logger.info("="*60)
    logger.info(f"Stock universe expansion completed!")
    logger.info(f"  Total stocks fetched: {len(stock_universe)}")
    logger.info(f"  New stocks added: {new_stocks_count}")
    logger.info("="*60)
    
    return len(stock_universe), new_stocks_count

if __name__ == "__main__":
    total, new = expand_stock_universe()
    print(f"\nSummary: {total} total stocks, {new} new stocks added")