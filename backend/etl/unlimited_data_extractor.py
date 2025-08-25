"""
Unlimited Stock Data Extractor - No Rate Limits
Scrapes data directly from web sources without API limitations
"""

import asyncio
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Any, Optional
import json
import re
import time
from urllib.parse import quote
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os

logger = logging.getLogger(__name__)


class UnlimitedDataExtractor:
    """Extract stock data without rate limits using web scraping and free sources"""
    
    def __init__(self):
        # Configure Selenium for headless scraping
        self.chrome_options = Options()
        self.chrome_options.add_argument('--headless')
        self.chrome_options.add_argument('--no-sandbox')
        self.chrome_options.add_argument('--disable-dev-shm-usage')
        self.chrome_options.add_argument('--disable-gpu')
        self.chrome_options.add_argument('--window-size=1920x1080')
        self.chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        # Session management
        self.session = None
        self.driver = None
        
        # Cache for frequently accessed data
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        
    async def get_session(self):
        """Get or create aiohttp session"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    def get_driver(self):
        """Get or create Selenium driver"""
        if not self.driver:
            try:
                self.driver = webdriver.Chrome(options=self.chrome_options)
            except:
                # Fallback to requests if Selenium not available
                self.driver = None
        return self.driver
    
    async def scrape_yahoo_finance(self, ticker: str) -> Dict:
        """Scrape Yahoo Finance without API - no rate limits"""
        try:
            url = f"https://finance.yahoo.com/quote/{ticker}"
            session = await self.get_session()
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    logger.warning(f"Yahoo Finance returned {response.status} for {ticker}")
                    return None
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract price data
                price_data = self._extract_yahoo_price_data(soup, ticker)
                
                # Extract key statistics
                stats_data = await self._scrape_yahoo_statistics(ticker)
                
                # Extract historical data
                historical_data = await self._scrape_yahoo_historical(ticker)
                
                return {
                    'ticker': ticker,
                    'source': 'yahoo_scrape',
                    'timestamp': datetime.now(),
                    'price_data': price_data,
                    'statistics': stats_data,
                    'historical': historical_data
                }
                
        except Exception as e:
            logger.error(f"Error scraping Yahoo Finance for {ticker}: {e}")
            return None
    
    def _extract_yahoo_price_data(self, soup: BeautifulSoup, ticker: str) -> Dict:
        """Extract price data from Yahoo Finance HTML"""
        try:
            data = {}
            
            # Current price
            price_elem = soup.find('fin-streamer', {'data-symbol': ticker, 'data-field': 'regularMarketPrice'})
            if price_elem:
                data['current_price'] = float(price_elem.get('value', 0))
            
            # Price change
            change_elem = soup.find('fin-streamer', {'data-symbol': ticker, 'data-field': 'regularMarketChange'})
            if change_elem:
                data['change'] = float(change_elem.get('value', 0))
            
            # Percent change
            pct_elem = soup.find('fin-streamer', {'data-symbol': ticker, 'data-field': 'regularMarketChangePercent'})
            if pct_elem:
                data['change_percent'] = float(pct_elem.get('value', 0))
            
            # Volume
            volume_elem = soup.find('fin-streamer', {'data-symbol': ticker, 'data-field': 'regularMarketVolume'})
            if volume_elem:
                data['volume'] = int(float(volume_elem.get('value', 0)))
            
            # Day range
            for td in soup.find_all('td', {'data-test': 'DAYS_RANGE-value'}):
                range_text = td.text.strip()
                if ' - ' in range_text:
                    low, high = range_text.split(' - ')
                    data['day_low'] = float(low.replace(',', ''))
                    data['day_high'] = float(high.replace(',', ''))
            
            # Market cap
            for td in soup.find_all('td', {'data-test': 'MARKET_CAP-value'}):
                cap_text = td.text.strip()
                data['market_cap'] = self._parse_market_cap(cap_text)
            
            # PE ratio
            for td in soup.find_all('td', {'data-test': 'PE_RATIO-value'}):
                pe_text = td.text.strip()
                if pe_text and pe_text != 'N/A':
                    data['pe_ratio'] = float(pe_text.replace(',', ''))
            
            return data
            
        except Exception as e:
            logger.error(f"Error extracting Yahoo price data: {e}")
            return {}
    
    def _parse_market_cap(self, cap_text: str) -> float:
        """Parse market cap text to float"""
        try:
            multipliers = {'K': 1e3, 'M': 1e6, 'B': 1e9, 'T': 1e12}
            cap_text = cap_text.upper().replace(',', '')
            
            for suffix, multiplier in multipliers.items():
                if suffix in cap_text:
                    number = float(cap_text.replace(suffix, ''))
                    return number * multiplier
            
            return float(cap_text)
        except:
            return 0
    
    async def _scrape_yahoo_statistics(self, ticker: str) -> Dict:
        """Scrape key statistics page"""
        try:
            url = f"https://finance.yahoo.com/quote/{ticker}/key-statistics"
            session = await self.get_session()
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    return {}
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                stats = {}
                
                # Extract all statistics tables
                for table in soup.find_all('table'):
                    rows = table.find_all('tr')
                    for row in rows:
                        cols = row.find_all('td')
                        if len(cols) >= 2:
                            key = cols[0].text.strip()
                            value = cols[1].text.strip()
                            
                            # Parse common statistics
                            if 'Beta' in key:
                                try:
                                    stats['beta'] = float(value)
                                except:
                                    pass
                            elif '52 Week High' in key:
                                try:
                                    stats['52_week_high'] = float(value.replace(',', ''))
                                except:
                                    pass
                            elif '52 Week Low' in key:
                                try:
                                    stats['52_week_low'] = float(value.replace(',', ''))
                                except:
                                    pass
                            elif 'Avg Volume' in key:
                                stats['avg_volume'] = self._parse_volume(value)
                
                return stats
                
        except Exception as e:
            logger.error(f"Error scraping Yahoo statistics: {e}")
            return {}
    
    def _parse_volume(self, volume_text: str) -> int:
        """Parse volume text to integer"""
        try:
            multipliers = {'K': 1e3, 'M': 1e6, 'B': 1e9}
            volume_text = volume_text.upper().replace(',', '')
            
            for suffix, multiplier in multipliers.items():
                if suffix in volume_text:
                    number = float(volume_text.replace(suffix, ''))
                    return int(number * multiplier)
            
            return int(float(volume_text))
        except:
            return 0
    
    async def _scrape_yahoo_historical(self, ticker: str, days: int = 30) -> List[Dict]:
        """Scrape historical data using download link"""
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Unix timestamps
            period1 = int(start_date.timestamp())
            period2 = int(end_date.timestamp())
            
            # Direct download URL (no authentication needed)
            url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?" \
                  f"period1={period1}&period2={period2}&interval=1d&events=history"
            
            session = await self.get_session()
            
            async with session.get(url) as response:
                if response.status != 200:
                    return []
                
                csv_text = await response.text()
                
                # Parse CSV
                lines = csv_text.strip().split('\n')
                if len(lines) < 2:
                    return []
                
                headers = lines[0].split(',')
                data = []
                
                for line in lines[1:]:
                    values = line.split(',')
                    if len(values) == len(headers):
                        row = {}
                        for i, header in enumerate(headers):
                            try:
                                if header == 'Date':
                                    row[header.lower()] = values[i]
                                elif header == 'Volume':
                                    row[header.lower()] = int(values[i])
                                else:
                                    row[header.lower()] = float(values[i])
                            except:
                                row[header.lower()] = values[i]
                        data.append(row)
                
                return data
                
        except Exception as e:
            logger.error(f"Error scraping Yahoo historical data: {e}")
            return []
    
    async def fetch_sec_edgar_data(self, ticker: str) -> Dict:
        """Fetch fundamental data from SEC EDGAR (free, no limits)"""
        try:
            # SEC EDGAR API endpoint (free, no authentication)
            base_url = "https://data.sec.gov/submissions"
            
            # Get CIK (Central Index Key) for the ticker
            cik = await self._get_cik_from_ticker(ticker)
            if not cik:
                return None
            
            # Pad CIK to 10 digits
            cik_padded = str(cik).zfill(10)
            
            url = f"{base_url}/CIK{cik_padded}.json"
            
            session = await self.get_session()
            headers = {
                'User-Agent': 'YourCompany your-email@example.com'  # SEC requires identification
            }
            
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    return None
                
                data = await response.json()
                
                # Extract relevant financial data
                filings = data.get('filings', {}).get('recent', {})
                
                return {
                    'ticker': ticker,
                    'source': 'sec_edgar',
                    'timestamp': datetime.now(),
                    'company_name': data.get('name'),
                    'cik': cik,
                    'sic': data.get('sic'),
                    'sic_description': data.get('sicDescription'),
                    'fiscal_year_end': data.get('fiscalYearEnd'),
                    'recent_filings': self._parse_recent_filings(filings)
                }
                
        except Exception as e:
            logger.error(f"Error fetching SEC EDGAR data: {e}")
            return None
    
    async def _get_cik_from_ticker(self, ticker: str) -> Optional[str]:
        """Get CIK from ticker symbol"""
        try:
            url = "https://www.sec.gov/files/company_tickers.json"
            session = await self.get_session()
            
            async with session.get(url) as response:
                if response.status != 200:
                    return None
                
                data = await response.json()
                
                for company in data.values():
                    if company.get('ticker', '').upper() == ticker.upper():
                        return str(company.get('cik_str'))
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting CIK: {e}")
            return None
    
    def _parse_recent_filings(self, filings: Dict) -> List[Dict]:
        """Parse recent SEC filings"""
        try:
            forms = filings.get('form', [])
            dates = filings.get('filingDate', [])
            accession_numbers = filings.get('accessionNumber', [])
            
            recent = []
            for i in range(min(10, len(forms))):  # Get last 10 filings
                recent.append({
                    'form': forms[i] if i < len(forms) else None,
                    'filing_date': dates[i] if i < len(dates) else None,
                    'accession_number': accession_numbers[i] if i < len(accession_numbers) else None
                })
            
            return recent
            
        except Exception as e:
            logger.error(f"Error parsing filings: {e}")
            return []
    
    async def fetch_iex_cloud_data(self, ticker: str) -> Dict:
        """Fetch data from IEX Cloud (free tier, no limits for certain endpoints)"""
        try:
            # IEX Cloud free endpoints (no token required for some data)
            base_url = "https://api.iextrading.com/1.0"
            
            session = await self.get_session()
            
            # Get quote data (free)
            quote_url = f"{base_url}/stock/{ticker.lower()}/quote"
            
            async with session.get(quote_url) as response:
                if response.status != 200:
                    return None
                
                quote_data = await response.json()
            
            # Get company info (free)
            company_url = f"{base_url}/stock/{ticker.lower()}/company"
            
            async with session.get(company_url) as response:
                if response.status == 200:
                    company_data = await response.json()
                else:
                    company_data = {}
            
            # Get stats (free)
            stats_url = f"{base_url}/stock/{ticker.lower()}/stats"
            
            async with session.get(stats_url) as response:
                if response.status == 200:
                    stats_data = await response.json()
                else:
                    stats_data = {}
            
            return {
                'ticker': ticker,
                'source': 'iex_cloud',
                'timestamp': datetime.now(),
                'quote': quote_data,
                'company': company_data,
                'stats': stats_data
            }
            
        except Exception as e:
            logger.error(f"Error fetching IEX Cloud data: {e}")
            return None
    
    async def extract_stock_data(self, ticker: str) -> Dict:
        """Extract stock data from all unlimited sources"""
        # Check cache first
        cache_key = f"{ticker}_{datetime.now().strftime('%Y%m%d%H')}"
        if cache_key in self.cache:
            cache_time, cached_data = self.cache[cache_key]
            if time.time() - cache_time < self.cache_ttl:
                logger.info(f"Using cached data for {ticker}")
                return cached_data
        
        # Fetch from all sources concurrently
        tasks = [
            self.scrape_yahoo_finance(ticker),
            self.fetch_sec_edgar_data(ticker),
            self.fetch_iex_cloud_data(ticker)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine data from all sources
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
        
        # Cache the result
        self.cache[cache_key] = (time.time(), combined_data)
        
        return combined_data
    
    async def batch_extract(self, tickers: List[str], max_concurrent: int = 20) -> List[Dict]:
        """Extract data for multiple tickers with controlled concurrency"""
        results = []
        
        # Process in controlled batches to avoid overwhelming resources
        for i in range(0, len(tickers), max_concurrent):
            batch = tickers[i:i + max_concurrent]
            logger.info(f"Processing batch {i//max_concurrent + 1}: {len(batch)} tickers")
            
            tasks = [self.extract_stock_data(ticker) for ticker in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for ticker, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to extract {ticker}: {result}")
                    results.append({'ticker': ticker, 'error': str(result)})
                else:
                    results.append(result)
            
            # Small delay between batches to be respectful
            await asyncio.sleep(0.5)
        
        return results
    
    async def close(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
        if self.driver:
            self.driver.quit()


class BulkDataDownloader:
    """Download bulk data files for historical data"""
    
    @staticmethod
    async def download_nasdaq_symbols() -> pd.DataFrame:
        """Download complete NASDAQ symbol list"""
        url = "http://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    text = await response.text()
                    lines = text.strip().split('\n')
                    
                    # Parse pipe-delimited file
                    data = []
                    for line in lines[1:-1]:  # Skip header and footer
                        parts = line.split('|')
                        if len(parts) >= 8:
                            data.append({
                                'symbol': parts[1],
                                'name': parts[2],
                                'market': parts[5],
                                'test_issue': parts[3],
                                'financial_status': parts[4],
                                'round_lot_size': parts[6]
                            })
                    
                    return pd.DataFrame(data)
        
        return pd.DataFrame()
    
    @staticmethod
    async def download_yahoo_bulk_data(date: datetime) -> pd.DataFrame:
        """Download bulk EOD data from Yahoo Finance"""
        # Note: This would typically involve downloading a large CSV file
        # For demonstration, returning empty DataFrame
        return pd.DataFrame()


if __name__ == "__main__":
    async def test():
        extractor = UnlimitedDataExtractor()
        
        # Test single ticker
        data = await extractor.extract_stock_data('AAPL')
        print(f"Extracted data for AAPL: {json.dumps(data, indent=2, default=str)[:500]}...")
        
        # Test batch extraction
        tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        results = await extractor.batch_extract(tickers)
        print(f"Extracted data for {len(results)} tickers")
        
        # Clean up
        await extractor.close()
    
    asyncio.run(test())