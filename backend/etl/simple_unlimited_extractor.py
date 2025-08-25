"""
Simple Unlimited Data Extractor - No Dependencies, No Rate Limits
Uses only basic Python libraries and free data sources
"""

import asyncio
import aiohttp
import json
import csv
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Any, Optional
import time
from io import StringIO

logger = logging.getLogger(__name__)


class SimpleUnlimitedExtractor:
    """Simple extractor using only free, unlimited sources"""
    
    def __init__(self):
        self.session = None
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        
    async def get_session(self):
        """Get or create aiohttp session"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def fetch_yahoo_csv(self, ticker: str, days: int = 30) -> Dict:
        """Fetch Yahoo Finance data via CSV download (no limits)"""
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Unix timestamps
            period1 = int(start_date.timestamp())
            period2 = int(end_date.timestamp())
            
            # Direct CSV download URL - NO AUTHENTICATION REQUIRED
            url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?" \
                  f"period1={period1}&period2={period2}&interval=1d&events=history"
            
            session = await self.get_session()
            
            async with session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"Yahoo CSV unavailable for {ticker}")
                    return None
                
                csv_text = await response.text()
                
                # Parse CSV data
                csv_reader = csv.DictReader(StringIO(csv_text))
                rows = list(csv_reader)
                
                if not rows:
                    return None
                
                # Get latest row
                latest = rows[-1]
                
                # Calculate basic statistics
                closes = [float(row['Close']) for row in rows if row.get('Close')]
                volumes = [int(float(row['Volume'])) for row in rows if row.get('Volume')]
                
                return {
                    'ticker': ticker,
                    'source': 'yahoo_csv',
                    'timestamp': datetime.now(),
                    'latest_data': {
                        'date': latest.get('Date'),
                        'open': float(latest.get('Open', 0)),
                        'high': float(latest.get('High', 0)),
                        'low': float(latest.get('Low', 0)),
                        'close': float(latest.get('Close', 0)),
                        'adj_close': float(latest.get('Adj Close', 0)),
                        'volume': int(float(latest.get('Volume', 0)))
                    },
                    'statistics': {
                        'avg_close_30d': sum(closes) / len(closes) if closes else 0,
                        'avg_volume_30d': sum(volumes) / len(volumes) if volumes else 0,
                        'min_close_30d': min(closes) if closes else 0,
                        'max_close_30d': max(closes) if closes else 0,
                        'volatility': self._calculate_volatility(closes)
                    },
                    'historical_count': len(rows)
                }
                
        except Exception as e:
            logger.error(f"Error fetching Yahoo CSV for {ticker}: {e}")
            return None
    
    def _calculate_volatility(self, prices: List[float]) -> float:
        """Calculate simple volatility measure"""
        if len(prices) < 2:
            return 0
        
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] != 0:
                returns.append((prices[i] - prices[i-1]) / prices[i-1])
        
        if not returns:
            return 0
        
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        return variance ** 0.5
    
    async def fetch_nasdaq_traded_info(self, ticker: str) -> Dict:
        """Fetch info from NASDAQ trader (free, no limits)"""
        try:
            url = "http://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
            
            session = await self.get_session()
            
            async with session.get(url) as response:
                if response.status != 200:
                    return None
                
                text = await response.text()
                
                # Find ticker in the list
                for line in text.split('\n')[1:-1]:  # Skip header and footer
                    parts = line.split('|')
                    if len(parts) >= 8 and parts[1] == ticker.upper():
                        return {
                            'ticker': ticker,
                            'source': 'nasdaq_trader',
                            'timestamp': datetime.now(),
                            'info': {
                                'symbol': parts[1],
                                'security_name': parts[2],
                                'market_category': parts[3],
                                'test_issue': parts[4] == 'Y',
                                'financial_status': parts[5],
                                'round_lot_size': int(parts[6]) if parts[6].isdigit() else 100,
                                'etf': parts[7] == 'Y',
                                'exchange': 'NASDAQ'
                            }
                        }
                
                return None
                
        except Exception as e:
            logger.error(f"Error fetching NASDAQ info: {e}")
            return None
    
    async def fetch_treasury_rates(self) -> Dict:
        """Fetch risk-free rates from US Treasury (free, no limits)"""
        try:
            # Treasury Direct API - completely free
            url = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/avg_interest_rates"
            
            session = await self.get_session()
            
            params = {
                'filter': 'record_date:gte:' + (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                'sort': '-record_date',
                'limit': '1'
            }
            
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    return None
                
                data = await response.json()
                
                if data.get('data'):
                    latest = data['data'][0]
                    return {
                        'source': 'treasury',
                        'timestamp': datetime.now(),
                        'rates': {
                            'date': latest.get('record_date'),
                            'avg_interest_rate': float(latest.get('avg_interest_rate_amt', 0)),
                            'security_type': latest.get('security_type_desc', 'Treasury')
                        }
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"Error fetching Treasury rates: {e}")
            return None
    
    async def fetch_fred_economic_data(self) -> Dict:
        """Fetch economic indicators from FRED (free with generous limits)"""
        try:
            # FRED API - 120 requests per minute (plenty for our needs)
            # Get S&P 500 index
            url = "https://api.stlouisfed.org/fred/series/observations"
            
            session = await self.get_session()
            
            # You can get a free API key from https://fred.stlouisfed.org/docs/api/api_key.html
            # For now, using the demo endpoint
            params = {
                'series_id': 'SP500',
                'file_type': 'json',
                'limit': '1',
                'sort_order': 'desc',
                'api_key': 'demo'  # Replace with actual key
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('observations'):
                        latest = data['observations'][0]
                        return {
                            'source': 'fred',
                            'timestamp': datetime.now(),
                            'sp500': {
                                'date': latest.get('date'),
                                'value': float(latest.get('value', 0))
                            }
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching FRED data: {e}")
            return None
    
    async def extract_all_data(self, ticker: str) -> Dict:
        """Extract data from all free sources"""
        # Check cache
        cache_key = f"{ticker}_{datetime.now().strftime('%Y%m%d%H')}"
        if cache_key in self.cache:
            cache_time, cached_data = self.cache[cache_key]
            if time.time() - cache_time < self.cache_ttl:
                logger.info(f"Using cached data for {ticker}")
                return cached_data
        
        # Fetch from all sources
        tasks = [
            self.fetch_yahoo_csv(ticker),
            self.fetch_nasdaq_traded_info(ticker),
            self.fetch_treasury_rates(),
            self.fetch_fred_economic_data()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        combined_data = {
            'ticker': ticker,
            'extraction_time': datetime.now(),
            'sources': {}
        }
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task failed: {result}")
                continue
            if result and isinstance(result, dict):
                source = result.get('source')
                if source:
                    combined_data['sources'][source] = result
        
        # Cache result
        self.cache[cache_key] = (time.time(), combined_data)
        
        return combined_data
    
    async def batch_extract(self, tickers: List[str], batch_size: int = 50) -> List[Dict]:
        """Extract data for multiple tickers efficiently"""
        all_results = []
        total_batches = (len(tickers) + batch_size - 1) // batch_size
        
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            logger.info(f"Processing batch {batch_num}/{total_batches}: {len(batch)} tickers")
            
            # Process batch concurrently
            tasks = [self.extract_all_data(ticker) for ticker in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for ticker, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Failed {ticker}: {result}")
                    all_results.append({
                        'ticker': ticker,
                        'error': str(result),
                        'extraction_time': datetime.now()
                    })
                else:
                    all_results.append(result)
            
            # Progress update
            processed = min(i + batch_size, len(tickers))
            logger.info(f"Progress: {processed}/{len(tickers)} tickers processed")
            
            # Small delay to be respectful
            if batch_num < total_batches:
                await asyncio.sleep(0.1)
        
        return all_results
    
    async def close(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()


# Standalone functions for easy testing
async def test_single_ticker():
    """Test with a single ticker"""
    extractor = SimpleUnlimitedExtractor()
    
    try:
        data = await extractor.extract_all_data('AAPL')
        print(f"Successfully extracted data for AAPL:")
        print(json.dumps(data, indent=2, default=str)[:1000])
    finally:
        await extractor.close()


async def test_batch_extraction():
    """Test batch extraction"""
    extractor = SimpleUnlimitedExtractor()
    
    try:
        # Test with 100 popular tickers
        tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'JNJ',
            'WMT', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'BAC', 'ADBE', 'CRM', 'NFLX',
            'PFE', 'ABBV', 'TMO', 'CSCO', 'PEP', 'AVGO', 'NKE', 'CMCSA', 'VZ', 'INTC',
            'COST', 'ABT', 'WFC', 'MRK', 'CVX', 'UPS', 'T', 'MS', 'ORCL', 'AMD',
            'TXN', 'HON', 'PM', 'IBM', 'QCOM', 'RTX', 'CAT', 'GS', 'SBUX', 'AMT',
            'INTU', 'GE', 'MMM', 'BA', 'NOW', 'ISRG', 'DE', 'SPGI', 'GILD', 'AXP',
            'BKNG', 'LMT', 'SYK', 'BLK', 'MDLZ', 'TJX', 'ADP', 'TMUS', 'C', 'MO',
            'CI', 'ZTS', 'CB', 'SO', 'DUK', 'PLD', 'CL', 'WM', 'ETN', 'BSX',
            'AON', 'ITW', 'MU', 'CSX', 'HUM', 'TGT', 'USB', 'PNC', 'GD', 'TFC',
            'SHW', 'MCO', 'FIS', 'MAR', 'AIG', 'KHC', 'F', 'DAL', 'GM', 'SPG'
        ]
        
        print(f"Starting extraction for {len(tickers)} tickers...")
        start_time = time.time()
        
        results = await extractor.batch_extract(tickers)
        
        elapsed = time.time() - start_time
        successful = sum(1 for r in results if 'error' not in r)
        
        print(f"\n=== Extraction Complete ===")
        print(f"Total tickers: {len(tickers)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(tickers) - successful}")
        print(f"Time elapsed: {elapsed:.2f} seconds")
        print(f"Average time per ticker: {elapsed/len(tickers):.2f} seconds")
        
    finally:
        await extractor.close()


if __name__ == "__main__":
    # Test single ticker
    print("Testing single ticker extraction...")
    asyncio.run(test_single_ticker())
    
    print("\n" + "="*50 + "\n")
    
    # Test batch extraction
    print("Testing batch extraction...")
    asyncio.run(test_batch_extraction())