"""
SEC EDGAR API Client - Unlimited free access to all SEC filings
This is our primary source for fundamental data
"""

import asyncio
import aiohttp
from typing import Dict, Optional, List, Any
from datetime import datetime, timedelta
import logging
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import re
import json

from backend.data_ingestion.base_client import BaseAPIClient
from backend.utils.cache import get_redis

logger = logging.getLogger(__name__)


class SECEdgarClient(BaseAPIClient):
    """
    SEC EDGAR client for unlimited fundamental data access
    """
    
    def __init__(self):
        # SEC requires user agent with contact info
        super().__init__("sec_edgar", api_key=None)
        self.headers = {
            'User-Agent': 'InvestmentAnalysisPlatform/1.0 (contact@example.com)'
        }
        self.cik_map = {}  # Cache CIK mappings
    
    def _get_base_url(self) -> str:
        return "https://data.sec.gov"
    
    async def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        method: str = "GET"
    ) -> Optional[Any]:
        """
        Override to add SEC-specific headers and no rate limiting
        """
        url = f"{self.base_url}/{endpoint}"
        
        try:
            async with self.session.request(
                method,
                url,
                params=params,
                headers=self.headers
            ) as response:
                if response.status == 200:
                    content_type = response.headers.get('Content-Type', '')
                    
                    if 'json' in content_type:
                        return await response.json()
                    else:
                        return await response.text()
                else:
                    logger.error(f"SEC EDGAR error {response.status}: {await response.text()}")
                    return None
                    
        except Exception as e:
            logger.error(f"SEC EDGAR request failed for {endpoint}: {e}")
            return None
    
    async def get_cik_mapping(self) -> Dict[str, str]:
        """
        Get mapping of tickers to CIK numbers
        """
        cache_key = "sec:cik_mapping"
        
        async def fetch():
            # Get the company tickers JSON file
            response = await self._make_request("submissions/CIK-listing.json")
            
            if response:
                # Build ticker to CIK mapping
                mapping = {}
                for item in response:
                    ticker = item.get('ticker')
                    cik = str(item.get('cik_str', '')).zfill(10)
                    if ticker and cik:
                        mapping[ticker.upper()] = cik
                
                return mapping
            return {}
        
        # Cache for 24 hours
        mapping = await self.get_cached_or_fetch(cache_key, fetch, ttl=86400)
        self.cik_map = mapping or {}
        return self.cik_map
    
    async def get_company_facts(self, ticker: str) -> Optional[Dict]:
        """
        Get comprehensive company facts from XBRL data
        """
        # Ensure we have CIK mapping
        if not self.cik_map:
            await self.get_cik_mapping()
        
        cik = self.cik_map.get(ticker.upper())
        if not cik:
            logger.warning(f"No CIK found for ticker {ticker}")
            return None
        
        cache_key = f"sec:facts:{ticker}"
        
        async def fetch():
            endpoint = f"api/xbrl/companyfacts/CIK{cik}.json"
            response = await self._make_request(endpoint)
            
            if response and 'facts' in response:
                facts = response['facts']
                
                # Extract key financial metrics
                financials = {
                    'ticker': ticker,
                    'cik': cik,
                    'entity_name': response.get('entityName'),
                    'sic': response.get('sic'),
                    'sic_description': response.get('sicDescription'),
                    'fiscal_year_end': response.get('fiscalYearEnd'),
                    'state_of_incorporation': response.get('stateOfIncorporation'),
                    'metrics': {}
                }
                
                # Process US-GAAP facts
                if 'us-gaap' in facts:
                    gaap = facts['us-gaap']
                    
                    # Income Statement items
                    income_items = [
                        'Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax',
                        'CostOfRevenue', 'GrossProfit', 'OperatingExpenses',
                        'OperatingIncomeLoss', 'NetIncomeLoss',
                        'EarningsPerShareBasic', 'EarningsPerShareDiluted'
                    ]
                    
                    # Balance Sheet items
                    balance_items = [
                        'Assets', 'AssetsCurrent', 'LiabilitiesAndStockholdersEquity',
                        'Liabilities', 'LiabilitiesCurrent', 'StockholdersEquity',
                        'CashAndCashEquivalentsAtCarryingValue', 'Debt',
                        'LongTermDebt', 'ShortTermBorrowings'
                    ]
                    
                    # Cash Flow items
                    cashflow_items = [
                        'CashProvidedByUsedInOperatingActivities',
                        'CashProvidedByUsedInInvestingActivities',
                        'CashProvidedByUsedInFinancingActivities',
                        'CapitalExpenditures', 'FreeCashFlow'
                    ]
                    
                    all_items = income_items + balance_items + cashflow_items
                    
                    for item in all_items:
                        if item in gaap:
                            units = gaap[item].get('units', {})
                            
                            # Get the most recent annual and quarterly values
                            annual_values = []
                            quarterly_values = []
                            
                            for unit_type, values in units.items():
                                for value in values:
                                    if value.get('form') in ['10-K', '10-K/A']:
                                        annual_values.append({
                                            'value': value.get('val'),
                                            'period': value.get('period'),
                                            'filed': value.get('filed')
                                        })
                                    elif value.get('form') in ['10-Q', '10-Q/A']:
                                        quarterly_values.append({
                                            'value': value.get('val'),
                                            'period': value.get('period'),
                                            'filed': value.get('filed')
                                        })
                            
                            # Sort by filing date
                            annual_values.sort(key=lambda x: x['filed'], reverse=True)
                            quarterly_values.sort(key=lambda x: x['filed'], reverse=True)
                            
                            financials['metrics'][item] = {
                                'label': gaap[item].get('label'),
                                'description': gaap[item].get('description'),
                                'latest_annual': annual_values[0] if annual_values else None,
                                'latest_quarterly': quarterly_values[0] if quarterly_values else None,
                                'annual_history': annual_values[:5],  # Last 5 years
                                'quarterly_history': quarterly_values[:8]  # Last 8 quarters
                            }
                
                financials['timestamp'] = datetime.utcnow().isoformat()
                return financials
            
            return None
        
        # Cache for 6 hours
        return await self.get_cached_or_fetch(cache_key, fetch, ttl=21600)
    
    async def get_recent_filings(
        self,
        ticker: str,
        form_types: List[str] = None
    ) -> Optional[List[Dict]]:
        """
        Get recent SEC filings for a company
        """
        if not self.cik_map:
            await self.get_cik_mapping()
        
        cik = self.cik_map.get(ticker.upper())
        if not cik:
            return None
        
        if not form_types:
            form_types = ['10-K', '10-Q', '8-K', 'DEF 14A', '13F-HR']
        
        cache_key = f"sec:filings:{ticker}:{','.join(form_types)}"
        
        async def fetch():
            endpoint = f"submissions/CIK{cik}.json"
            response = await self._make_request(endpoint)
            
            if response and 'filings' in response:
                recent_filings = []
                filings = response['filings'].get('recent', {})
                
                # Extract filing information
                for i in range(min(len(filings.get('form', [])), 100)):  # Last 100 filings
                    form = filings['form'][i]
                    
                    if not form_types or form in form_types:
                        recent_filings.append({
                            'form': form,
                            'filing_date': filings['filingDate'][i],
                            'reporting_date': filings['reportDate'][i],
                            'accession_number': filings['accessionNumber'][i],
                            'file_number': filings['fileNumber'][i],
                            'primary_document': filings['primaryDocument'][i],
                            'primary_doc_description': filings['primaryDocDescription'][i],
                            'is_xbrl': filings['isXBRL'][i] == 1,
                            'is_inline_xbrl': filings['isInlineXBRL'][i] == 1,
                            'size': filings['size'][i],
                            'url': f"https://www.sec.gov/Archives/edgar/data/{cik}/{filings['accessionNumber'][i].replace('-', '')}/{filings['primaryDocument'][i]}"
                        })
                
                return recent_filings
            
            return None
        
        # Cache for 1 hour
        return await self.get_cached_or_fetch(cache_key, fetch, ttl=3600)
    
    async def parse_10k_10q(self, ticker: str, filing_type: str = '10-K') -> Optional[Dict]:
        """
        Parse and extract data from 10-K or 10-Q filings
        """
        filings = await self.get_recent_filings(ticker, [filing_type])
        
        if not filings:
            return None
        
        # Get the most recent filing
        latest_filing = filings[0]
        
        cache_key = f"sec:parsed:{ticker}:{filing_type}:{latest_filing['accession_number']}"
        
        async def fetch():
            # Fetch the filing content
            filing_url = latest_filing['url']
            
            async with aiohttp.ClientSession() as session:
                async with session.get(filing_url, headers=self.headers) as response:
                    if response.status == 200:
                        content = await response.text()
                    else:
                        return None
            
            # Parse the filing
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract key sections
            sections = {
                'business': self._extract_section(soup, ['ITEM 1', 'BUSINESS']),
                'risk_factors': self._extract_section(soup, ['ITEM 1A', 'RISK FACTORS']),
                'properties': self._extract_section(soup, ['ITEM 2', 'PROPERTIES']),
                'legal_proceedings': self._extract_section(soup, ['ITEM 3', 'LEGAL PROCEEDINGS']),
                'mda': self._extract_section(soup, ['ITEM 7', "MANAGEMENT'S DISCUSSION"]),
                'financial_statements': self._extract_section(soup, ['ITEM 8', 'FINANCIAL STATEMENTS']),
                'controls': self._extract_section(soup, ['ITEM 9A', 'CONTROLS AND PROCEDURES'])
            }
            
            # Extract financial tables
            financial_data = self._extract_financial_tables(soup)
            
            return {
                'ticker': ticker,
                'filing_type': filing_type,
                'filing_date': latest_filing['filing_date'],
                'reporting_date': latest_filing['reporting_date'],
                'accession_number': latest_filing['accession_number'],
                'sections': sections,
                'financial_data': financial_data,
                'timestamp': datetime.utcnow().isoformat()
            }
        
        # Cache for 24 hours
        return await self.get_cached_or_fetch(cache_key, fetch, ttl=86400)
    
    def _extract_section(self, soup: BeautifulSoup, keywords: List[str]) -> Optional[str]:
        """
        Extract a specific section from the filing
        """
        text_content = soup.get_text()
        
        # Find the section
        for keyword in keywords:
            pattern = re.compile(rf'{keyword}.*?(?=ITEM|\Z)', re.IGNORECASE | re.DOTALL)
            match = pattern.search(text_content)
            if match:
                section_text = match.group(0)
                # Clean up the text
                section_text = re.sub(r'\s+', ' ', section_text)
                section_text = section_text.strip()
                
                # Limit to reasonable length
                if len(section_text) > 50000:
                    section_text = section_text[:50000] + "..."
                
                return section_text
        
        return None
    
    def _extract_financial_tables(self, soup: BeautifulSoup) -> Dict:
        """
        Extract financial data from tables
        """
        financial_data = {}
        
        # Find all tables
        tables = soup.find_all('table')
        
        for table in tables:
            # Try to identify the table type
            table_text = table.get_text().lower()
            
            if 'balance sheet' in table_text or 'financial position' in table_text:
                financial_data['balance_sheet'] = self._parse_financial_table(table)
            elif 'income statement' in table_text or 'operations' in table_text:
                financial_data['income_statement'] = self._parse_financial_table(table)
            elif 'cash flow' in table_text:
                financial_data['cash_flow'] = self._parse_financial_table(table)
        
        return financial_data
    
    def _parse_financial_table(self, table) -> List[Dict]:
        """
        Parse a financial table into structured data
        """
        rows = table.find_all('tr')
        
        if not rows:
            return []
        
        # Extract headers
        headers = []
        header_row = rows[0]
        for cell in header_row.find_all(['th', 'td']):
            headers.append(cell.get_text().strip())
        
        # Extract data
        data = []
        for row in rows[1:]:
            cells = row.find_all(['td', 'th'])
            if cells:
                row_data = {}
                for i, cell in enumerate(cells):
                    if i < len(headers):
                        value = cell.get_text().strip()
                        # Try to parse numbers
                        value = self._parse_number(value)
                        row_data[headers[i]] = value
                
                if any(row_data.values()):  # Skip empty rows
                    data.append(row_data)
        
        return data
    
    def _parse_number(self, value: str) -> Any:
        """
        Parse a string value to number if possible
        """
        if not value or value == '—' or value == '-':
            return None
        
        # Remove common formatting
        value = value.replace(',', '').replace('$', '').replace('%', '')
        value = re.sub(r'\((.*?)\)', r'-\1', value)  # Convert (123) to -123
        
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            return value
    
    async def get_insider_transactions(self, ticker: str) -> Optional[List[Dict]]:
        """
        Get insider trading data from Form 4 filings
        """
        filings = await self.get_recent_filings(ticker, ['4', '4/A'])
        
        if not filings:
            return None
        
        transactions = []
        
        # Process recent Form 4 filings
        for filing in filings[:20]:  # Last 20 transactions
            cache_key = f"sec:insider:{filing['accession_number']}"
            
            async def fetch_transaction():
                # This would parse the actual Form 4 XML
                # For now, return filing metadata
                return {
                    'filing_date': filing['filing_date'],
                    'accession_number': filing['accession_number'],
                    'url': filing['url']
                }
            
            transaction = await self.get_cached_or_fetch(
                cache_key,
                fetch_transaction,
                ttl=86400
            )
            
            if transaction:
                transactions.append(transaction)
        
        return transactions
    
    async def calculate_fundamental_ratios(self, ticker: str) -> Optional[Dict]:
        """
        Calculate financial ratios from SEC data
        """
        facts = await self.get_company_facts(ticker)
        
        if not facts or 'metrics' not in facts:
            return None
        
        metrics = facts['metrics']
        
        # Helper function to get latest value
        def get_latest(metric_name: str, period: str = 'annual') -> Optional[float]:
            if metric_name in metrics:
                if period == 'annual' and metrics[metric_name].get('latest_annual'):
                    return metrics[metric_name]['latest_annual']['value']
                elif period == 'quarterly' and metrics[metric_name].get('latest_quarterly'):
                    return metrics[metric_name]['latest_quarterly']['value']
            return None
        
        # Calculate ratios
        ratios = {}
        
        # Profitability Ratios
        revenue = get_latest('Revenues') or get_latest('RevenueFromContractWithCustomerExcludingAssessedTax')
        net_income = get_latest('NetIncomeLoss')
        total_assets = get_latest('Assets')
        equity = get_latest('StockholdersEquity')
        
        if revenue and net_income:
            ratios['net_margin'] = (net_income / revenue) * 100
        
        if total_assets and net_income:
            ratios['roa'] = (net_income / total_assets) * 100
        
        if equity and net_income:
            ratios['roe'] = (net_income / equity) * 100
        
        # Liquidity Ratios
        current_assets = get_latest('AssetsCurrent')
        current_liabilities = get_latest('LiabilitiesCurrent')
        
        if current_assets and current_liabilities:
            ratios['current_ratio'] = current_assets / current_liabilities
        
        # Leverage Ratios
        total_debt = get_latest('Debt') or (
            (get_latest('LongTermDebt') or 0) + (get_latest('ShortTermBorrowings') or 0)
        )
        
        if total_debt and equity:
            ratios['debt_to_equity'] = total_debt / equity
        
        # Per Share Metrics
        ratios['eps_basic'] = get_latest('EarningsPerShareBasic')
        ratios['eps_diluted'] = get_latest('EarningsPerShareDiluted')
        
        return {
            'ticker': ticker,
            'ratios': ratios,
            'source': 'SEC EDGAR',
            'timestamp': datetime.utcnow().isoformat()
        }