"""
Insider Trading Analysis Engine

Analyzes insider trading patterns using free SEC EDGAR data:
- Form 4 filings (insider transactions)
- Form 3 filings (initial ownership)
- Form 5 filings (annual statements)
- Pattern recognition for unusual activity
- Sentiment scoring based on insider actions

All data sourced from SEC EDGAR API (completely free)
"""

import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import xml.etree.ElementTree as ET
import json
import re

import pandas as pd
import numpy as np
from sqlalchemy import text

from backend.utils.cache import CacheManager
from backend.utils.rate_limiter import RateLimiter
from backend.utils.cost_monitor import CostMonitor
from backend.models.database import SessionLocal

logger = logging.getLogger(__name__)

@dataclass
class InsiderTransaction:
    """Individual insider transaction data"""
    filing_date: datetime
    transaction_date: datetime
    ticker: str
    insider_name: str
    insider_title: str
    transaction_type: str  # Purchase, Sale, Gift, etc.
    shares: float
    price: Optional[float]
    total_value: Optional[float]
    shares_owned_after: Optional[float]
    form_type: str  # Form 3, 4, or 5
    filing_url: str
    
@dataclass
class InsiderSentiment:
    """Insider sentiment analysis result"""
    sentiment_score: float  # -1 (very bearish) to 1 (very bullish)
    confidence: float
    net_purchases: float
    net_sales: float
    unique_insiders: int
    total_transactions: int
    avg_transaction_size: float
    recent_activity_trend: str
    key_insights: List[str]

class InsiderTradingAnalyzer:
    """
    Comprehensive insider trading analysis using free SEC data
    
    Features:
    - Real-time Form 4 filing monitoring
    - Historical insider pattern analysis
    - Unusual activity detection
    - Insider sentiment scoring
    - Executive vs non-executive analysis
    - Cluster buying/selling detection
    - Performance correlation analysis
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.cache = CacheManager()
        self.cost_monitor = CostMonitor()
        
        # SEC API rate limit: 10 requests per second
        self.sec_limiter = RateLimiter(calls=10, period=1)
        
        # SEC EDGAR base URLs
        self.sec_base_url = "https://data.sec.gov"
        self.edgar_base_url = "https://www.sec.gov/Archives/edgar/data"
        
        # User agent required by SEC
        self.headers = {
            'User-Agent': 'Investment Analysis Platform contact@example.com',
            'Accept': 'application/json',
            'Host': 'data.sec.gov'
        }
        
    async def analyze_insider_activity(
        self,
        symbol: str,
        cik: Optional[str] = None,
        days_back: int = 90
    ) -> Dict:
        """
        Analyze insider trading activity for a stock
        
        Args:
            symbol: Stock ticker symbol
            cik: SEC Central Index Key (will lookup if not provided)
            days_back: Number of days to analyze
            
        Returns:
            Comprehensive insider analysis results
        """
        # Check cache first
        cache_key = f"insider_analysis:{symbol}:{days_back}"
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            return cached_result
            
        try:
            # Get CIK if not provided
            if not cik:
                cik = await self._get_cik_for_symbol(symbol)
                if not cik:
                    return {'error': f'Could not find CIK for symbol {symbol}'}
            
            # Get recent filings
            filings = await self._get_recent_filings(cik, days_back)
            
            # Parse insider transactions
            transactions = await self._parse_insider_transactions(filings)
            
            if not transactions:
                return {
                    'symbol': symbol,
                    'cik': cik,
                    'analysis_date': datetime.now(),
                    'transactions_analyzed': 0,
                    'sentiment': None,
                    'insights': ['No insider transactions found in the specified period']
                }
            
            # Analyze transactions
            sentiment = self._calculate_insider_sentiment(transactions)
            patterns = self._detect_trading_patterns(transactions)
            unusual_activity = self._detect_unusual_activity(transactions)
            
            results = {
                'symbol': symbol,
                'cik': cik,
                'analysis_date': datetime.now(),
                'period_days': days_back,
                'transactions_analyzed': len(transactions),
                'sentiment': sentiment,
                'patterns': patterns,
                'unusual_activity': unusual_activity,
                'recent_transactions': self._format_recent_transactions(transactions[:10]),
                'insights': self._generate_insider_insights(sentiment, patterns, unusual_activity)
            }
            
            # Cache results
            await self.cache.set(cache_key, results, expire=3600)  # 1 hour
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing insider activity for {symbol}: {e}")
            return {'error': str(e)}
    
    async def _get_cik_for_symbol(self, symbol: str) -> Optional[str]:
        """Get Central Index Key for a stock symbol"""
        await self.sec_limiter.acquire()
        
        try:
            # Use SEC company search API
            url = f"{self.sec_base_url}/submissions/CIK{symbol.upper()}.json"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('cik')
                    
            # Alternative: search through ticker lookup
            lookup_url = f"{self.sec_base_url}/files/company_tickers.json"
            async with aiohttp.ClientSession() as session:
                async with session.get(lookup_url, headers=self.headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        for entry in data.values():
                            if entry.get('ticker', '').upper() == symbol.upper():
                                return str(entry.get('cik_str', '')).zfill(10)
                                
            return None
            
        except Exception as e:
            logger.error(f"Error getting CIK for {symbol}: {e}")
            return None
    
    async def _get_recent_filings(self, cik: str, days_back: int) -> List[Dict]:
        """Get recent Form 3, 4, and 5 filings for a company"""
        await self.sec_limiter.acquire()
        
        try:
            # Format CIK with leading zeros
            cik_formatted = str(cik).zfill(10)
            url = f"{self.sec_base_url}/submissions/CIK{cik_formatted}.json"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status != 200:
                        logger.warning(f"SEC API returned status {response.status} for CIK {cik}")
                        return []
                        
                    data = await response.json()
            
            # Filter for insider trading forms
            filings = data.get('filings', {}).get('recent', {})
            if not filings:
                return []
                
            cutoff_date = datetime.now() - timedelta(days=days_back)
            recent_filings = []
            
            forms = filings.get('form', [])
            filing_dates = filings.get('filingDate', [])
            accession_numbers = filings.get('accessionNumber', [])
            
            for i, form in enumerate(forms):
                if form in ['3', '4', '5']:  # Insider trading forms
                    try:
                        filing_date = datetime.strptime(filing_dates[i], '%Y-%m-%d')
                        if filing_date >= cutoff_date:
                            recent_filings.append({
                                'form_type': form,
                                'filing_date': filing_date,
                                'accession_number': accession_numbers[i],
                                'cik': cik_formatted
                            })
                    except (IndexError, ValueError) as e:
                        logger.warning(f"Error parsing filing date: {e}")
                        continue
            
            return recent_filings
            
        except Exception as e:
            logger.error(f"Error getting recent filings for CIK {cik}: {e}")
            return []
    
    async def _parse_insider_transactions(self, filings: List[Dict]) -> List[InsiderTransaction]:
        """Parse insider transactions from SEC filings"""
        transactions = []
        
        # Process filings in parallel with rate limiting
        semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
        
        async def process_filing(filing):
            async with semaphore:
                return await self._parse_single_filing(filing)
        
        tasks = [process_filing(filing) for filing in filings]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Error parsing filing: {result}")
                continue
            if result:
                transactions.extend(result)
        
        # Sort by transaction date (most recent first)
        transactions.sort(key=lambda x: x.transaction_date, reverse=True)
        
        return transactions
    
    async def _parse_single_filing(self, filing: Dict) -> List[InsiderTransaction]:
        """Parse a single SEC filing for insider transactions"""
        await self.sec_limiter.acquire()
        
        try:
            # Construct filing URL
            accession = filing['accession_number'].replace('-', '')
            cik = filing['cik']
            url = f"{self.edgar_base_url}/{cik}/{accession}/{filing['accession_number']}.txt"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status != 200:
                        return []
                        
                    filing_text = await response.text()
            
            # Parse XML content from EDGAR filing
            transactions = []
            
            # Extract XML documents from the filing text
            xml_docs = re.findall(r'<XML>(.*?)</XML>', filing_text, re.DOTALL)
            
            for xml_content in xml_docs:
                try:
                    root = ET.fromstring(xml_content)
                    
                    # Find ownership document
                    if root.tag.endswith('ownershipDocument'):
                        parsed_transactions = self._parse_ownership_document(
                            root, filing['filing_date'], url
                        )
                        transactions.extend(parsed_transactions)
                        
                except ET.ParseError as e:
                    logger.warning(f"Error parsing XML: {e}")
                    continue
            
            return transactions
            
        except Exception as e:
            logger.error(f"Error parsing filing {filing.get('accession_number', 'unknown')}: {e}")
            return []
    
    def _parse_ownership_document(self, root: ET.Element, filing_date: datetime, url: str) -> List[InsiderTransaction]:
        """Parse ownership document XML for transaction details"""
        transactions = []
        
        try:
            # Get issuer ticker
            ticker = None
            issuer = root.find('.//issuer')
            if issuer is not None:
                ticker_elem = issuer.find('issuerTradingSymbol')
                if ticker_elem is not None:
                    ticker = ticker_elem.text
            
            # Get reporting owner info
            owner_name = None
            owner_title = None
            reporting_owner = root.find('.//reportingOwner')
            if reporting_owner is not None:
                owner_name_elem = reporting_owner.find('.//rptOwnerName')
                if owner_name_elem is not None:
                    owner_name = owner_name_elem.text
                    
                # Get title/relationship
                relationships = reporting_owner.findall('.//reportingOwnerRelationship')
                titles = []
                for rel in relationships:
                    if rel.find('isDirector') is not None and rel.find('isDirector').text == '1':
                        titles.append('Director')
                    if rel.find('isOfficer') is not None and rel.find('isOfficer').text == '1':
                        officer_title = rel.find('officerTitle')
                        if officer_title is not None:
                            titles.append(officer_title.text)
                    if rel.find('isTenPercentOwner') is not None and rel.find('isTenPercentOwner').text == '1':
                        titles.append('10% Owner')
                owner_title = ', '.join(titles) if titles else 'Unknown'
            
            # Parse transactions
            non_derivative_table = root.find('.//nonDerivativeTable')
            if non_derivative_table is not None:
                for transaction in non_derivative_table.findall('.//nonDerivativeTransaction'):
                    try:
                        # Transaction date
                        trans_date_elem = transaction.find('.//transactionDate/value')
                        if trans_date_elem is not None:
                            trans_date = datetime.strptime(trans_date_elem.text, '%Y-%m-%d')
                        else:
                            trans_date = filing_date
                        
                        # Transaction details
                        trans_code_elem = transaction.find('.//transactionCode')
                        trans_code = trans_code_elem.text if trans_code_elem is not None else 'Unknown'
                        
                        # Map transaction codes to readable types
                        code_mapping = {
                            'P': 'Purchase',
                            'S': 'Sale',
                            'A': 'Grant',
                            'D': 'Disposition',
                            'G': 'Gift',
                            'V': 'Transaction in equity swap',
                            'W': 'Acquisition or disposition by will or inheritance',
                            'M': 'Exercise or conversion of derivative security'
                        }
                        trans_type = code_mapping.get(trans_code, trans_code)
                        
                        # Shares and price
                        shares_elem = transaction.find('.//transactionShares/value')
                        shares = float(shares_elem.text) if shares_elem is not None and shares_elem.text else 0
                        
                        price_elem = transaction.find('.//transactionPricePerShare/value')
                        price = float(price_elem.text) if price_elem is not None and price_elem.text else None
                        
                        total_value = shares * price if price is not None else None
                        
                        # Shares owned after
                        owned_after_elem = transaction.find('.//sharesOwnedFollowingTransaction/value')
                        owned_after = float(owned_after_elem.text) if owned_after_elem is not None and owned_after_elem.text else None
                        
                        # Only include significant transactions
                        if shares > 0 and ticker:
                            transactions.append(InsiderTransaction(
                                filing_date=filing_date,
                                transaction_date=trans_date,
                                ticker=ticker,
                                insider_name=owner_name or 'Unknown',
                                insider_title=owner_title or 'Unknown',
                                transaction_type=trans_type,
                                shares=shares,
                                price=price,
                                total_value=total_value,
                                shares_owned_after=owned_after,
                                form_type=f"Form {root.find('.//documentType').text if root.find('.//documentType') is not None else 'Unknown'}",
                                filing_url=url
                            ))
                            
                    except Exception as e:
                        logger.warning(f"Error parsing individual transaction: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Error parsing ownership document: {e}")
        
        return transactions
    
    def _calculate_insider_sentiment(self, transactions: List[InsiderTransaction]) -> InsiderSentiment:
        """Calculate overall insider sentiment based on trading patterns"""
        if not transactions:
            return InsiderSentiment(
                sentiment_score=0,
                confidence=0,
                net_purchases=0,
                net_sales=0,
                unique_insiders=0,
                total_transactions=0,
                avg_transaction_size=0,
                recent_activity_trend='neutral',
                key_insights=[]
            )
        
        # Separate purchases and sales
        purchases = [t for t in transactions if t.transaction_type in ['Purchase', 'Grant', 'Award']]
        sales = [t for t in transactions if t.transaction_type in ['Sale', 'Disposition']]
        
        # Calculate monetary values
        purchase_value = sum(t.total_value for t in purchases if t.total_value is not None)
        sale_value = sum(t.total_value for t in sales if t.total_value is not None)
        
        # Calculate share volumes
        purchase_shares = sum(t.shares for t in purchases)
        sale_shares = sum(t.shares for t in sales)
        
        net_purchases = purchase_value
        net_sales = sale_value
        net_sentiment = (purchase_value - sale_value) / max(purchase_value + sale_value, 1)
        
        # Weight by insider importance (executives get higher weight)
        executive_titles = ['CEO', 'CFO', 'President', 'Chairman', 'COO', 'CTO', 'EVP']
        
        weighted_sentiment = 0
        total_weight = 0
        
        for transaction in transactions:
            # Assign weights based on title
            weight = 1.0
            for title in executive_titles:
                if title.lower() in transaction.insider_title.lower():
                    weight = 2.0
                    break
            if 'Director' in transaction.insider_title:
                weight = 1.5
            if '10% Owner' in transaction.insider_title:
                weight = 2.5
                
            # Transaction direction sentiment
            if transaction.transaction_type in ['Purchase', 'Grant', 'Award']:
                sentiment_value = 1.0
            elif transaction.transaction_type in ['Sale', 'Disposition']:
                sentiment_value = -1.0
            else:
                sentiment_value = 0.0
                
            # Weight by transaction size
            if transaction.total_value:
                size_weight = min(transaction.total_value / 100000, 5.0)  # Cap at 5x for very large transactions
            else:
                size_weight = 1.0
                
            weighted_sentiment += sentiment_value * weight * size_weight
            total_weight += weight * size_weight
        
        final_sentiment = weighted_sentiment / max(total_weight, 1)
        
        # Calculate confidence based on consistency and volume
        consistency = self._calculate_sentiment_consistency(transactions)
        volume_factor = min(len(transactions) / 10, 1.0)  # Max confidence at 10+ transactions
        confidence = consistency * volume_factor
        
        # Analyze recent trend
        recent_trend = self._analyze_recent_trend(transactions)
        
        # Calculate average transaction size
        transaction_values = [t.total_value for t in transactions if t.total_value is not None]
        avg_transaction_size = np.mean(transaction_values) if transaction_values else 0
        
        # Generate key insights
        key_insights = []
        
        if abs(final_sentiment) > 0.3 and confidence > 0.5:
            if final_sentiment > 0:
                key_insights.append("Strong insider buying detected")
            else:
                key_insights.append("Strong insider selling detected")
        
        if len(set(t.insider_name for t in transactions)) > 5:
            key_insights.append("Multiple insiders trading simultaneously")
            
        executive_transactions = [t for t in transactions 
                                if any(title.lower() in t.insider_title.lower() 
                                      for title in executive_titles)]
        if len(executive_transactions) > 2:
            key_insights.append("Significant executive-level trading activity")
        
        return InsiderSentiment(
            sentiment_score=final_sentiment,
            confidence=confidence,
            net_purchases=purchase_value,
            net_sales=sale_value,
            unique_insiders=len(set(t.insider_name for t in transactions)),
            total_transactions=len(transactions),
            avg_transaction_size=avg_transaction_size,
            recent_activity_trend=recent_trend,
            key_insights=key_insights
        )
    
    def _calculate_sentiment_consistency(self, transactions: List[InsiderTransaction]) -> float:
        """Calculate how consistent the insider sentiment is"""
        if not transactions:
            return 0
            
        sentiments = []
        for transaction in transactions:
            if transaction.transaction_type in ['Purchase', 'Grant', 'Award']:
                sentiments.append(1)
            elif transaction.transaction_type in ['Sale', 'Disposition']:
                sentiments.append(-1)
            else:
                sentiments.append(0)
        
        if not sentiments:
            return 0
            
        # Calculate how aligned the sentiments are
        mean_sentiment = np.mean(sentiments)
        consistency = 1 - (np.std(sentiments) / 2)  # Normalize standard deviation
        
        return max(0, consistency)
    
    def _analyze_recent_trend(self, transactions: List[InsiderTransaction]) -> str:
        """Analyze the trend of recent insider activity"""
        if len(transactions) < 3:
            return 'neutral'
            
        # Sort by transaction date
        sorted_transactions = sorted(transactions, key=lambda x: x.transaction_date)
        
        # Split into first half and second half
        mid_point = len(sorted_transactions) // 2
        first_half = sorted_transactions[:mid_point]
        second_half = sorted_transactions[mid_point:]
        
        def calculate_period_sentiment(period_transactions):
            purchases = sum(1 for t in period_transactions 
                          if t.transaction_type in ['Purchase', 'Grant', 'Award'])
            sales = sum(1 for t in period_transactions 
                       if t.transaction_type in ['Sale', 'Disposition'])
            return purchases - sales
        
        first_sentiment = calculate_period_sentiment(first_half)
        second_sentiment = calculate_period_sentiment(second_half)
        
        if second_sentiment > first_sentiment + 1:
            return 'improving'
        elif second_sentiment < first_sentiment - 1:
            return 'declining'
        else:
            return 'stable'
    
    def _detect_trading_patterns(self, transactions: List[InsiderTransaction]) -> Dict:
        """Detect unusual trading patterns in insider activity"""
        patterns = {
            'cluster_buying': False,
            'cluster_selling': False,
            'executive_alignment': False,
            'unusual_volume': False,
            'timing_patterns': []
        }
        
        if not transactions:
            return patterns
        
        # Cluster detection (multiple insiders trading in same direction within short period)
        recent_transactions = [t for t in transactions 
                             if (datetime.now() - t.transaction_date).days <= 30]
        
        if len(recent_transactions) >= 3:
            purchases = [t for t in recent_transactions if t.transaction_type in ['Purchase', 'Grant']]
            sales = [t for t in recent_transactions if t.transaction_type in ['Sale', 'Disposition']]
            
            unique_buyers = len(set(t.insider_name for t in purchases))
            unique_sellers = len(set(t.insider_name for t in sales))
            
            if unique_buyers >= 3:
                patterns['cluster_buying'] = True
            if unique_sellers >= 3:
                patterns['cluster_selling'] = True
        
        # Executive alignment (multiple C-level executives trading in same direction)
        executive_titles = ['CEO', 'CFO', 'President', 'Chairman', 'COO', 'CTO']
        executive_transactions = [t for t in transactions 
                                if any(title.lower() in t.insider_title.lower() 
                                      for title in executive_titles)]
        
        if len(executive_transactions) >= 2:
            exec_purchases = [t for t in executive_transactions 
                            if t.transaction_type in ['Purchase', 'Grant']]
            exec_sales = [t for t in executive_transactions 
                        if t.transaction_type in ['Sale', 'Disposition']]
            
            if len(exec_purchases) >= 2 or len(exec_sales) >= 2:
                patterns['executive_alignment'] = True
        
        # Unusual volume detection
        if transactions:
            total_value = sum(t.total_value for t in transactions if t.total_value is not None)
            avg_transaction_value = total_value / len(transactions) if total_value else 0
            
            # Consider unusual if average transaction > $1M or total > $10M
            if avg_transaction_value > 1000000 or total_value > 10000000:
                patterns['unusual_volume'] = True
        
        return patterns
    
    def _detect_unusual_activity(self, transactions: List[InsiderTransaction]) -> Dict:
        """Detect unusual patterns that might indicate significant events"""
        unusual = {
            'activity_spike': False,
            'large_transactions': [],
            'concentrated_trading': False,
            'cross_insider_activity': False
        }
        
        if not transactions:
            return unusual
        
        # Activity spike detection (comparing to historical baseline)
        recent_30_days = [t for t in transactions 
                         if (datetime.now() - t.transaction_date).days <= 30]
        older_transactions = [t for t in transactions 
                            if (datetime.now() - t.transaction_date).days > 30]
        
        if len(older_transactions) > 0:
            recent_rate = len(recent_30_days) / 30
            historical_rate = len(older_transactions) / max(90, len(older_transactions) * 30)  # Assume 30 days per transaction period
            
            if recent_rate > historical_rate * 2:  # 2x normal rate
                unusual['activity_spike'] = True
        
        # Large transaction detection (>$5M or >1% of typical daily volume)
        large_transactions = [t for t in transactions 
                            if t.total_value and t.total_value > 5000000]
        unusual['large_transactions'] = [
            {
                'insider_name': t.insider_name,
                'transaction_date': t.transaction_date,
                'transaction_type': t.transaction_type,
                'value': t.total_value,
                'shares': t.shares
            } for t in large_transactions
        ]
        
        # Concentrated trading (many transactions by same insider)
        insider_counts = {}
        for transaction in transactions:
            insider_counts[transaction.insider_name] = insider_counts.get(transaction.insider_name, 0) + 1
        
        if any(count > 5 for count in insider_counts.values()):
            unusual['concentrated_trading'] = True
        
        # Cross-insider activity (multiple insiders from different categories)
        executive_insiders = set()
        director_insiders = set()
        owner_insiders = set()
        
        for transaction in transactions:
            title = transaction.insider_title.lower()
            if any(exec_title.lower() in title for exec_title in ['ceo', 'cfo', 'president', 'coo', 'cto']):
                executive_insiders.add(transaction.insider_name)
            elif 'director' in title:
                director_insiders.add(transaction.insider_name)
            elif '10% owner' in title:
                owner_insiders.add(transaction.insider_name)
        
        active_categories = sum([
            len(executive_insiders) > 0,
            len(director_insiders) > 0,
            len(owner_insiders) > 0
        ])
        
        if active_categories >= 2:
            unusual['cross_insider_activity'] = True
        
        return unusual
    
    def _format_recent_transactions(self, transactions: List[InsiderTransaction]) -> List[Dict]:
        """Format recent transactions for display"""
        formatted = []
        for transaction in transactions:
            formatted.append({
                'filing_date': transaction.filing_date.strftime('%Y-%m-%d'),
                'transaction_date': transaction.transaction_date.strftime('%Y-%m-%d'),
                'insider_name': transaction.insider_name,
                'insider_title': transaction.insider_title,
                'transaction_type': transaction.transaction_type,
                'shares': transaction.shares,
                'price': transaction.price,
                'total_value': transaction.total_value,
                'form_type': transaction.form_type
            })
        return formatted
    
    def _generate_insider_insights(
        self,
        sentiment: InsiderSentiment,
        patterns: Dict,
        unusual_activity: Dict
    ) -> List[str]:
        """Generate actionable insights from insider analysis"""
        insights = []
        
        # Sentiment insights
        if sentiment.confidence > 0.5:
            if sentiment.sentiment_score > 0.3:
                insights.append(f"Strong insider bullishness detected (confidence: {sentiment.confidence:.1%})")
            elif sentiment.sentiment_score < -0.3:
                insights.append(f"Strong insider bearishness detected (confidence: {sentiment.confidence:.1%})")
        
        # Pattern insights
        if patterns['cluster_buying']:
            insights.append("Multiple insiders buying simultaneously - potential bullish catalyst")
        if patterns['cluster_selling']:
            insights.append("Multiple insiders selling simultaneously - potential bearish signal")
        if patterns['executive_alignment']:
            insights.append("Executive team alignment in trading direction")
        if patterns['unusual_volume']:
            insights.append("Unusually large transaction volumes detected")
        
        # Unusual activity insights
        if unusual_activity['activity_spike']:
            insights.append("Significant spike in insider trading activity")
        if unusual_activity['large_transactions']:
            insights.append(f"{len(unusual_activity['large_transactions'])} large transactions (>$5M) detected")
        if unusual_activity['concentrated_trading']:
            insights.append("Concentrated trading by individual insiders")
        if unusual_activity['cross_insider_activity']:
            insights.append("Multiple insider categories (executives, directors, owners) active")
        
        # Volume insights
        if sentiment.unique_insiders > 5:
            insights.append(f"High insider participation ({sentiment.unique_insiders} unique insiders)")
        
        # Trend insights
        if sentiment.recent_activity_trend == 'improving':
            insights.append("Insider sentiment trend improving over time")
        elif sentiment.recent_activity_trend == 'declining':
            insights.append("Insider sentiment trend declining over time")
        
        # Value insights
        if sentiment.avg_transaction_size > 1000000:
            insights.append(f"Large average transaction size (${sentiment.avg_transaction_size:,.0f})")
        
        return insights or ["No significant insider trading patterns detected"]
    
    async def get_insider_alerts(self, symbol: str, thresholds: Dict) -> List[Dict]:
        """Generate insider trading alerts"""
        analysis = await self.analyze_insider_activity(symbol, days_back=30)
        
        alerts = []
        
        if 'error' in analysis:
            return alerts
            
        sentiment = analysis.get('sentiment')
        if not sentiment:
            return alerts
            
        # High confidence sentiment alerts
        if sentiment.confidence > thresholds.get('confidence_threshold', 0.7):
            if sentiment.sentiment_score > thresholds.get('bullish_threshold', 0.4):
                alerts.append({
                    'type': 'insider_bullish',
                    'message': f'Strong insider buying detected for {symbol}',
                    'sentiment_score': sentiment.sentiment_score,
                    'confidence': sentiment.confidence,
                    'net_purchases': sentiment.net_purchases,
                    'urgency': 'high'
                })
            elif sentiment.sentiment_score < thresholds.get('bearish_threshold', -0.4):
                alerts.append({
                    'type': 'insider_bearish',
                    'message': f'Strong insider selling detected for {symbol}',
                    'sentiment_score': sentiment.sentiment_score,
                    'confidence': sentiment.confidence,
                    'net_sales': sentiment.net_sales,
                    'urgency': 'high'
                })
        
        # Pattern alerts
        patterns = analysis.get('patterns', {})
        if patterns.get('cluster_buying'):
            alerts.append({
                'type': 'cluster_buying',
                'message': f'Multiple insiders buying {symbol} simultaneously',
                'urgency': 'medium'
            })
        
        if patterns.get('cluster_selling'):
            alerts.append({
                'type': 'cluster_selling',
                'message': f'Multiple insiders selling {symbol} simultaneously',
                'urgency': 'medium'
            })
        
        # Unusual activity alerts
        unusual = analysis.get('unusual_activity', {})
        if unusual.get('activity_spike'):
            alerts.append({
                'type': 'activity_spike',
                'message': f'Unusual spike in insider trading activity for {symbol}',
                'urgency': 'medium'
            })
        
        return alerts