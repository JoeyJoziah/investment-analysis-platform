#!/usr/bin/env python3
"""
Load Initial Stock Data
Loads comprehensive list of stocks from major US exchanges
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from sqlalchemy.org import Session
from backend.models.unified_models import Stock, Exchange, Sector, Industry
from backend.utils.db_init import DatabaseInitializer
from backend.config.settings import settings

logger = logging.getLogger(__name__)


class StockDataLoader:
    """Load initial stock data into database"""
    
    def __init__(self):
        self.db_init = DatabaseInitializer()
        self.engine = self.db_init.engine
        
    def get_major_stocks(self):
        """Get list of major US stocks"""
        # Top stocks by market cap from each sector
        stocks = [
            # Technology
            {"ticker": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ", "sector": "Technology", "industry": "Consumer Electronics", "market_cap": 3000000000000},
            {"ticker": "MSFT", "name": "Microsoft Corporation", "exchange": "NASDAQ", "sector": "Technology", "industry": "Software", "market_cap": 2800000000000},
            {"ticker": "GOOGL", "name": "Alphabet Inc. Class A", "exchange": "NASDAQ", "sector": "Communication Services", "industry": "Internet Content", "market_cap": 1800000000000},
            {"ticker": "GOOG", "name": "Alphabet Inc. Class C", "exchange": "NASDAQ", "sector": "Communication Services", "industry": "Internet Content", "market_cap": 1800000000000},
            {"ticker": "AMZN", "name": "Amazon.com Inc.", "exchange": "NASDAQ", "sector": "Consumer Discretionary", "industry": "Internet Retail", "market_cap": 1700000000000},
            {"ticker": "NVDA", "name": "NVIDIA Corporation", "exchange": "NASDAQ", "sector": "Technology", "industry": "Semiconductors", "market_cap": 1500000000000},
            {"ticker": "META", "name": "Meta Platforms Inc.", "exchange": "NASDAQ", "sector": "Communication Services", "industry": "Internet Content", "market_cap": 1200000000000},
            {"ticker": "TSLA", "name": "Tesla Inc.", "exchange": "NASDAQ", "sector": "Consumer Discretionary", "industry": "Auto Manufacturers", "market_cap": 800000000000},
            
            # Financials
            {"ticker": "BRK.B", "name": "Berkshire Hathaway Inc. Class B", "exchange": "NYSE", "sector": "Financials", "industry": "Multi-Sector Holdings", "market_cap": 780000000000},
            {"ticker": "JPM", "name": "JPMorgan Chase & Co.", "exchange": "NYSE", "sector": "Financials", "industry": "Banks", "market_cap": 500000000000},
            {"ticker": "V", "name": "Visa Inc.", "exchange": "NYSE", "sector": "Financials", "industry": "Credit Services", "market_cap": 480000000000},
            {"ticker": "MA", "name": "Mastercard Incorporated", "exchange": "NYSE", "sector": "Financials", "industry": "Credit Services", "market_cap": 420000000000},
            {"ticker": "BAC", "name": "Bank of America Corporation", "exchange": "NYSE", "sector": "Financials", "industry": "Banks", "market_cap": 300000000000},
            {"ticker": "WFC", "name": "Wells Fargo & Company", "exchange": "NYSE", "sector": "Financials", "industry": "Banks", "market_cap": 200000000000},
            {"ticker": "GS", "name": "The Goldman Sachs Group Inc.", "exchange": "NYSE", "sector": "Financials", "industry": "Capital Markets", "market_cap": 150000000000},
            
            # Healthcare
            {"ticker": "UNH", "name": "UnitedHealth Group Incorporated", "exchange": "NYSE", "sector": "Healthcare", "industry": "Managed Healthcare", "market_cap": 550000000000},
            {"ticker": "JNJ", "name": "Johnson & Johnson", "exchange": "NYSE", "sector": "Healthcare", "industry": "Pharmaceuticals", "market_cap": 450000000000},
            {"ticker": "LLY", "name": "Eli Lilly and Company", "exchange": "NYSE", "sector": "Healthcare", "industry": "Pharmaceuticals", "market_cap": 580000000000},
            {"ticker": "PFE", "name": "Pfizer Inc.", "exchange": "NYSE", "sector": "Healthcare", "industry": "Pharmaceuticals", "market_cap": 280000000000},
            {"ticker": "ABBV", "name": "AbbVie Inc.", "exchange": "NYSE", "sector": "Healthcare", "industry": "Pharmaceuticals", "market_cap": 320000000000},
            {"ticker": "MRK", "name": "Merck & Co. Inc.", "exchange": "NYSE", "sector": "Healthcare", "industry": "Pharmaceuticals", "market_cap": 300000000000},
            
            # Consumer Staples
            {"ticker": "WMT", "name": "Walmart Inc.", "exchange": "NYSE", "sector": "Consumer Staples", "industry": "Hypermarkets", "market_cap": 450000000000},
            {"ticker": "PG", "name": "The Procter & Gamble Company", "exchange": "NYSE", "sector": "Consumer Staples", "industry": "Household Products", "market_cap": 400000000000},
            {"ticker": "KO", "name": "The Coca-Cola Company", "exchange": "NYSE", "sector": "Consumer Staples", "industry": "Soft Drinks", "market_cap": 280000000000},
            {"ticker": "PEP", "name": "PepsiCo Inc.", "exchange": "NASDAQ", "sector": "Consumer Staples", "industry": "Soft Drinks", "market_cap": 260000000000},
            {"ticker": "COST", "name": "Costco Wholesale Corporation", "exchange": "NASDAQ", "sector": "Consumer Staples", "industry": "Hypermarkets", "market_cap": 350000000000},
            
            # Energy
            {"ticker": "XOM", "name": "Exxon Mobil Corporation", "exchange": "NYSE", "sector": "Energy", "industry": "Oil & Gas", "market_cap": 450000000000},
            {"ticker": "CVX", "name": "Chevron Corporation", "exchange": "NYSE", "sector": "Energy", "industry": "Oil & Gas", "market_cap": 350000000000},
            {"ticker": "COP", "name": "ConocoPhillips", "exchange": "NYSE", "sector": "Energy", "industry": "Oil & Gas", "market_cap": 140000000000},
            {"ticker": "SLB", "name": "Schlumberger Limited", "exchange": "NYSE", "sector": "Energy", "industry": "Oil & Gas Equipment", "market_cap": 80000000000},
            
            # Industrials
            {"ticker": "UPS", "name": "United Parcel Service Inc.", "exchange": "NYSE", "sector": "Industrials", "industry": "Air Freight", "market_cap": 150000000000},
            {"ticker": "BA", "name": "The Boeing Company", "exchange": "NYSE", "sector": "Industrials", "industry": "Aerospace & Defense", "market_cap": 140000000000},
            {"ticker": "RTX", "name": "RTX Corporation", "exchange": "NYSE", "sector": "Industrials", "industry": "Aerospace & Defense", "market_cap": 140000000000},
            {"ticker": "CAT", "name": "Caterpillar Inc.", "exchange": "NYSE", "sector": "Industrials", "industry": "Farm & Heavy Machinery", "market_cap": 160000000000},
            {"ticker": "GE", "name": "General Electric Company", "exchange": "NYSE", "sector": "Industrials", "industry": "Industrial Conglomerates", "market_cap": 180000000000},
            
            # Real Estate
            {"ticker": "AMT", "name": "American Tower Corporation", "exchange": "NYSE", "sector": "Real Estate", "industry": "REITs", "market_cap": 100000000000},
            {"ticker": "PLD", "name": "Prologis Inc.", "exchange": "NYSE", "sector": "Real Estate", "industry": "REITs", "market_cap": 120000000000},
            {"ticker": "CCI", "name": "Crown Castle Inc.", "exchange": "NYSE", "sector": "Real Estate", "industry": "REITs", "market_cap": 60000000000},
            
            # Materials
            {"ticker": "LIN", "name": "Linde plc", "exchange": "NYSE", "sector": "Materials", "industry": "Industrial Gases", "market_cap": 200000000000},
            {"ticker": "APD", "name": "Air Products and Chemicals Inc.", "exchange": "NYSE", "sector": "Materials", "industry": "Industrial Gases", "market_cap": 70000000000},
            {"ticker": "SHW", "name": "The Sherwin-Williams Company", "exchange": "NYSE", "sector": "Materials", "industry": "Specialty Chemicals", "market_cap": 80000000000},
            
            # Utilities
            {"ticker": "NEE", "name": "NextEra Energy Inc.", "exchange": "NYSE", "sector": "Utilities", "industry": "Electric Utilities", "market_cap": 150000000000},
            {"ticker": "DUK", "name": "Duke Energy Corporation", "exchange": "NYSE", "sector": "Utilities", "industry": "Electric Utilities", "market_cap": 80000000000},
            {"ticker": "SO", "name": "The Southern Company", "exchange": "NYSE", "sector": "Utilities", "industry": "Electric Utilities", "market_cap": 75000000000},
            
            # Communication Services
            {"ticker": "DIS", "name": "The Walt Disney Company", "exchange": "NYSE", "sector": "Communication Services", "industry": "Entertainment", "market_cap": 200000000000},
            {"ticker": "CMCSA", "name": "Comcast Corporation", "exchange": "NASDAQ", "sector": "Communication Services", "industry": "Cable & Satellite", "market_cap": 180000000000},
            {"ticker": "VZ", "name": "Verizon Communications Inc.", "exchange": "NYSE", "sector": "Communication Services", "industry": "Telecom Services", "market_cap": 170000000000},
            {"ticker": "T", "name": "AT&T Inc.", "exchange": "NYSE", "sector": "Communication Services", "industry": "Telecom Services", "market_cap": 120000000000},
            {"ticker": "NFLX", "name": "Netflix Inc.", "exchange": "NASDAQ", "sector": "Communication Services", "industry": "Entertainment", "market_cap": 200000000000}
        ]
        
        return stocks
        
    def load_stocks(self):
        """Load stocks into database"""
        stocks = self.get_major_stocks()
        loaded = 0
        errors = 0
        
        with Session(self.engine) as session:
            # Get exchanges
            exchanges = {e.code: e for e in session.query(Exchange).all()}
            
            # Get sectors
            sectors = {s.name: s for s in session.query(Sector).all()}
            
            # Get industries
            industries = {i.name: i for i in session.query(Industry).all()}
            
            for stock_data in stocks:
                try:
                    # Check if stock already exists
                    existing = session.query(Stock).filter_by(ticker=stock_data['ticker']).first()
                    if existing:
                        logger.info(f"Stock {stock_data['ticker']} already exists")
                        continue
                        
                    # Get or create sector
                    sector = sectors.get(stock_data['sector'])
                    if not sector and stock_data['sector']:
                        sector = Sector(name=stock_data['sector'])
                        session.add(sector)
                        session.flush()
                        sectors[stock_data['sector']] = sector
                        
                    # Get or create industry
                    industry = industries.get(stock_data['industry'])
                    if not industry and stock_data['industry'] and sector:
                        industry = Industry(
                            name=stock_data['industry'],
                            sector_id=sector.id
                        )
                        session.add(industry)
                        session.flush()
                        industries[stock_data['industry']] = industry
                        
                    # Create stock
                    stock = Stock(
                        ticker=stock_data['ticker'],
                        name=stock_data['name'],
                        exchange_id=exchanges.get(stock_data['exchange']).id if stock_data['exchange'] in exchanges else None,
                        sector_id=sector.id if sector else None,
                        industry_id=industry.id if industry else None,
                        market_cap=stock_data.get('market_cap', 0),
                        is_active=True
                    )
                    
                    session.add(stock)
                    loaded += 1
                    logger.info(f"Loaded stock: {stock_data['ticker']} - {stock_data['name']}")
                    
                except Exception as e:
                    logger.error(f"Error loading stock {stock_data['ticker']}: {e}")
                    errors += 1
                    
            # Commit all changes
            session.commit()
            
        logger.info(f"Stock loading complete: {loaded} loaded, {errors} errors")
        return loaded, errors
        
    async def load_stock_list_from_api(self):
        """Load comprehensive stock list from free API (if available)"""
        # This would connect to a free API to get full stock list
        # For now, we use the curated list above
        pass


def main():
    """Main function"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    loader = StockDataLoader()
    loaded, errors = loader.load_stocks()
    
    print(f"\n✅ Loaded {loaded} stocks into database")
    if errors > 0:
        print(f"⚠️  {errors} errors occurred during loading")
        
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())