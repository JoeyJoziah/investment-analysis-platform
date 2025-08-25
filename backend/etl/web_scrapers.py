"""
Web Scraping Utilities for Free Financial Data Sources
Implements robust scraping with rate limiting and error handling
"""

import asyncio
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from urllib.parse import quote
import random

logger = logging.getLogger(__name__)


class WebScraperBase:
    """Base class for web scrapers with common functionality"""
    
    def __init__(self, base_delay: float = 2.0):
        self.base_delay = base_delay
        self.session_headers = {
            'User-Agent': self._get_random_user_agent(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    def _get_random_user_agent(self) -> str:
        """Get a random user agent to avoid detection"""
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15'
        ]
        return random.choice(user_agents)
    
    async def _make_request(self, url: str, session: aiohttp.ClientSession, 
                           retry_count: int = 3) -> Optional[BeautifulSoup]:
        """Make HTTP request with retries and error handling"""
        for attempt in range(retry_count):
            try:
                # Add random delay to avoid being detected
                await asyncio.sleep(self.base_delay + random.uniform(0, 2))
                
                async with session.get(url, headers=self.session_headers, 
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        content = await response.text()
                        return BeautifulSoup(content, 'html.parser')
                    elif response.status == 429:  # Rate limited
                        wait_time = 2 ** attempt * 10  # Exponential backoff
                        logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"HTTP {response.status} for {url}")
                        
            except Exception as e:
                logger.error(f"Request error (attempt {attempt + 1}): {e}")
                if attempt == retry_count - 1:
                    return None
                await asyncio.sleep(2 ** attempt)
        
        return None


class YahooFinanceScraper(WebScraperBase):
    """Scraper for Yahoo Finance data"""
    
    def __init__(self):
        super().__init__(base_delay=3.0)
        self.base_url = "https://finance.yahoo.com"
    
    async def scrape_stock_data(self, ticker: str) -> Optional[Dict]:
        """Scrape basic stock data from Yahoo Finance"""
        url = f"{self.base_url}/quote/{ticker}"
        
        async with aiohttp.ClientSession() as session:
            soup = await self._make_request(url, session)
            
            if not soup:
                return None
            
            try:
                data = {
                    'ticker': ticker,
                    'source': 'yahoo_scraper',
                    'timestamp': datetime.now(),
                }
                
                # Extract current price
                price_elem = soup.find('fin-streamer', {'data-field': 'regularMarketPrice'})
                if price_elem:
                    data['current_price'] = float(price_elem.get('value', 0))
                
                # Extract change
                change_elem = soup.find('fin-streamer', {'data-field': 'regularMarketChange'})
                if change_elem:
                    data['price_change'] = float(change_elem.get('value', 0))
                
                # Extract percentage change
                change_pct_elem = soup.find('fin-streamer', {'data-field': 'regularMarketChangePercent'})
                if change_pct_elem:
                    pct_text = change_pct_elem.get('value', '0')
                    data['price_change_pct'] = float(re.sub(r'[^\d.-]', '', pct_text))
                
                # Extract volume
                try:
                    stats_table = soup.find('table', {'data-test': 'quote-statistics'})
                    if stats_table:
                        rows = stats_table.find_all('tr')
                        for row in rows:
                            cells = row.find_all('td')
                            if len(cells) >= 2:
                                label = cells[0].get_text().strip()
                                value = cells[1].get_text().strip()
                                
                                if 'Volume' in label:
                                    data['volume'] = self._parse_volume(value)
                                elif 'Market Cap' in label:
                                    data['market_cap'] = self._parse_market_cap(value)
                                elif 'P/E Ratio' in label:
                                    data['pe_ratio'] = self._parse_ratio(value)
                except Exception as e:
                    logger.debug(f"Error extracting stats for {ticker}: {e}")
                
                # Extract company name
                try:
                    name_elem = soup.find('h1', {'data-reactid': True})
                    if name_elem:
                        company_name = name_elem.get_text().strip()
                        if '(' in company_name:
                            data['company_name'] = company_name.split('(')[0].strip()
                        else:
                            data['company_name'] = company_name
                except Exception as e:
                    logger.debug(f"Error extracting company name for {ticker}: {e}")
                
                return data
                
            except Exception as e:
                logger.error(f"Error parsing Yahoo Finance data for {ticker}: {e}")
                return None
    
    async def scrape_historical_data(self, ticker: str, days: int = 30) -> Optional[Dict]:
        """Scrape historical price data"""
        # Calculate time range
        end_time = int(time.time())
        start_time = end_time - (days * 24 * 60 * 60)
        
        url = f"{self.base_url}/quote/{ticker}/history"
        
        async with aiohttp.ClientSession() as session:
            soup = await self._make_request(url, session)
            
            if not soup:
                return None
            
            try:
                # Find historical data table
                table = soup.find('table', {'data-test': 'historical-prices'})
                if not table:
                    return None
                
                rows = table.find('tbody').find_all('tr')
                historical_data = []
                
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 6:
                        try:
                            date_str = cells[0].get_text().strip()
                            open_price = self._parse_price(cells[1].get_text())
                            high_price = self._parse_price(cells[2].get_text())
                            low_price = self._parse_price(cells[3].get_text())
                            close_price = self._parse_price(cells[4].get_text())
                            volume = self._parse_volume(cells[6].get_text()) if len(cells) > 6 else 0
                            
                            historical_data.append({
                                'date': date_str,
                                'open': open_price,
                                'high': high_price,
                                'low': low_price,
                                'close': close_price,
                                'volume': volume
                            })
                        except Exception as e:
                            logger.debug(f"Error parsing row for {ticker}: {e}")
                            continue
                
                return {
                    'ticker': ticker,
                    'source': 'yahoo_scraper',
                    'timestamp': datetime.now(),
                    'historical_data': historical_data
                }
                
            except Exception as e:
                logger.error(f"Error scraping historical data for {ticker}: {e}")
                return None
    
    def _parse_price(self, price_str: str) -> float:
        """Parse price string to float"""
        try:
            clean_price = re.sub(r'[^\d.-]', '', price_str)
            return float(clean_price) if clean_price else 0.0
        except:
            return 0.0
    
    def _parse_volume(self, volume_str: str) -> int:
        """Parse volume string to integer"""
        try:
            clean_volume = re.sub(r'[^\d.KMB]', '', volume_str.upper())
            
            if 'K' in clean_volume:
                return int(float(clean_volume.replace('K', '')) * 1000)
            elif 'M' in clean_volume:
                return int(float(clean_volume.replace('M', '')) * 1000000)
            elif 'B' in clean_volume:
                return int(float(clean_volume.replace('B', '')) * 1000000000)
            else:
                return int(float(clean_volume.replace(',', '')))
        except:
            return 0
    
    def _parse_market_cap(self, cap_str: str) -> int:
        """Parse market cap string to integer"""
        return self._parse_volume(cap_str)
    
    def _parse_ratio(self, ratio_str: str) -> float:
        """Parse ratio string to float"""
        try:
            clean_ratio = re.sub(r'[^\d.-]', '', ratio_str)
            return float(clean_ratio) if clean_ratio and clean_ratio != 'N/A' else 0.0
        except:
            return 0.0


class MarketWatchScraper(WebScraperBase):
    """Scraper for MarketWatch data"""
    
    def __init__(self):
        super().__init__(base_delay=4.0)
        self.base_url = "https://www.marketwatch.com"
    
    async def scrape_stock_data(self, ticker: str) -> Optional[Dict]:
        """Scrape stock data from MarketWatch"""
        url = f"{self.base_url}/investing/stock/{ticker}"
        
        async with aiohttp.ClientSession() as session:
            soup = await self._make_request(url, session)
            
            if not soup:
                return None
            
            try:
                data = {
                    'ticker': ticker,
                    'source': 'marketwatch_scraper',
                    'timestamp': datetime.now(),
                }
                
                # Extract price data
                price_elem = soup.find('bg-quote', class_='value')
                if price_elem:
                    data['current_price'] = self._parse_price(price_elem.get_text())
                
                # Extract change data
                change_elem = soup.find('bg-quote', class_='change--point--q')
                if change_elem:
                    data['price_change'] = self._parse_price(change_elem.get_text())
                
                # Extract percentage change
                pct_elem = soup.find('bg-quote', class_='change--percent--q')
                if pct_elem:
                    data['price_change_pct'] = self._parse_ratio(pct_elem.get_text())
                
                # Extract key metrics
                key_data_section = soup.find('div', class_='element--list')
                if key_data_section:
                    items = key_data_section.find_all('li', class_='kv__item')
                    for item in items:
                        label_elem = item.find('small', class_='kv__label')
                        value_elem = item.find('span', class_='kv__value')
                        
                        if label_elem and value_elem:
                            label = label_elem.get_text().strip().lower()
                            value = value_elem.get_text().strip()
                            
                            if 'volume' in label:
                                data['volume'] = self._parse_volume(value)
                            elif 'market cap' in label:
                                data['market_cap'] = self._parse_market_cap(value)
                            elif 'p/e ratio' in label:
                                data['pe_ratio'] = self._parse_ratio(value)
                
                return data
                
            except Exception as e:
                logger.error(f"Error parsing MarketWatch data for {ticker}: {e}")
                return None
    
    def _parse_price(self, price_str: str) -> float:
        """Parse price string to float"""
        try:
            clean_price = re.sub(r'[^\d.-]', '', price_str)
            return float(clean_price) if clean_price else 0.0
        except:
            return 0.0
    
    def _parse_volume(self, volume_str: str) -> int:
        """Parse volume string to integer"""
        try:
            clean_volume = re.sub(r'[^\d.KMB]', '', volume_str.upper())
            
            if 'K' in clean_volume:
                return int(float(clean_volume.replace('K', '')) * 1000)
            elif 'M' in clean_volume:
                return int(float(clean_volume.replace('M', '')) * 1000000)
            elif 'B' in clean_volume:
                return int(float(clean_volume.replace('B', '')) * 1000000000)
            else:
                return int(float(clean_volume.replace(',', '')))
        except:
            return 0
    
    def _parse_market_cap(self, cap_str: str) -> int:
        """Parse market cap string to integer"""
        return self._parse_volume(cap_str)
    
    def _parse_ratio(self, ratio_str: str) -> float:
        """Parse ratio string to float"""
        try:
            clean_ratio = re.sub(r'[^\d.-]', '', ratio_str)
            return float(clean_ratio) if clean_ratio and clean_ratio != 'N/A' else 0.0
        except:
            return 0.0


class GoogleFinanceScraper(WebScraperBase):
    """Scraper for Google Finance data"""
    
    def __init__(self):
        super().__init__(base_delay=5.0)
        self.base_url = "https://www.google.com/finance"
    
    async def scrape_stock_data(self, ticker: str) -> Optional[Dict]:
        """Scrape stock data from Google Finance"""
        url = f"{self.base_url}/quote/{ticker}"
        
        async with aiohttp.ClientSession() as session:
            soup = await self._make_request(url, session)
            
            if not soup:
                return None
            
            try:
                data = {
                    'ticker': ticker,
                    'source': 'google_finance_scraper',
                    'timestamp': datetime.now(),
                }
                
                # Extract price data using more specific selectors
                price_elem = soup.find('div', {'class': re.compile(r'YMlKec.*fxKbKc')})
                if price_elem:
                    data['current_price'] = self._parse_price(price_elem.get_text())
                
                # Extract change data
                change_elem = soup.find('div', {'class': re.compile(r'JwB6zf')})
                if change_elem:
                    change_text = change_elem.get_text()
                    if '(' in change_text and ')' in change_text:
                        parts = change_text.split('(')
                        if len(parts) >= 2:
                            data['price_change'] = self._parse_price(parts[0])
                            pct_part = parts[1].replace(')', '').replace('%', '')
                            data['price_change_pct'] = self._parse_ratio(pct_part)
                
                return data
                
            except Exception as e:
                logger.error(f"Error parsing Google Finance data for {ticker}: {e}")
                return None
    
    def _parse_price(self, price_str: str) -> float:
        """Parse price string to float"""
        try:
            clean_price = re.sub(r'[^\d.-]', '', price_str)
            return float(clean_price) if clean_price else 0.0
        except:
            return 0.0
    
    def _parse_ratio(self, ratio_str: str) -> float:
        """Parse ratio string to float"""
        try:
            clean_ratio = re.sub(r'[^\d.-]', '', ratio_str)
            return float(clean_ratio) if clean_ratio and clean_ratio != 'N/A' else 0.0
        except:
            return 0.0


class FREDScraper(WebScraperBase):
    """Scraper for Federal Reserve Economic Data (FRED)"""
    
    def __init__(self):
        super().__init__(base_delay=2.0)
        self.base_url = "https://fred.stlouisfed.org"
    
    async def scrape_economic_indicators(self) -> Optional[Dict]:
        """Scrape key economic indicators"""
        indicators = {
            'GDPC1': 'gdp',  # Real GDP
            'UNRATE': 'unemployment',  # Unemployment Rate  
            'FEDFUNDS': 'fed_funds_rate',  # Federal Funds Rate
            'DGS10': 'treasury_10y',  # 10-Year Treasury Rate
            'VIXCLS': 'vix'  # VIX
        }
        
        data = {
            'source': 'fred_scraper',
            'timestamp': datetime.now(),
            'indicators': {}
        }
        
        async with aiohttp.ClientSession() as session:
            for series_id, name in indicators.items():
                try:
                    url = f"{self.base_url}/series/{series_id}"
                    soup = await self._make_request(url, session)
                    
                    if soup:
                        # Find the latest observation
                        obs_elem = soup.find('span', {'id': 'series-last-observation-value'})
                        if obs_elem:
                            value = self._parse_price(obs_elem.get_text())
                            data['indicators'][name] = value
                            
                        # Add small delay between requests
                        await asyncio.sleep(1)
                        
                except Exception as e:
                    logger.error(f"Error scraping FRED indicator {series_id}: {e}")
        
        return data if data['indicators'] else None
    
    def _parse_price(self, price_str: str) -> float:
        """Parse price string to float"""
        try:
            clean_price = re.sub(r'[^\d.-]', '', price_str)
            return float(clean_price) if clean_price else 0.0
        except:
            return 0.0


# Factory function to get appropriate scraper
def get_scraper(source_name: str) -> Optional[WebScraperBase]:
    """Factory function to get the appropriate scraper"""
    scrapers = {
        'yahoo_scraper': YahooFinanceScraper,
        'marketwatch_scraper': MarketWatchScraper,
        'google_finance_scraper': GoogleFinanceScraper,
        'fred_scraper': FREDScraper
    }
    
    scraper_class = scrapers.get(source_name)
    if scraper_class:
        return scraper_class()
    
    return None