"""
Comprehensive Financial Data Fixtures and Mock Objects

This module provides realistic mock data for all external services and
financial scenarios, enabling complete offline testing.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, MagicMock
import json
import random
from dataclasses import dataclass, asdict
from decimal import Decimal

# Mock API Response Templates
@dataclass
class MockAPIResponse:
    """Base class for mock API responses"""
    status_code: int = 200
    data: Dict[str, Any] = None
    headers: Dict[str, str] = None
    
    def json(self):
        return self.data
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)


class AlphaVantageMocks:
    """Mock responses for Alpha Vantage API"""
    
    @staticmethod
    def daily_stock_data(ticker: str, days: int = 100) -> Dict[str, Any]:
        """Generate realistic daily stock data"""
        base_price = random.uniform(50, 300)
        dates = pd.date_range(end=date.today(), periods=days, freq='D')
        
        # Generate realistic price series
        returns = np.random.normal(0.001, 0.02, days)  # Daily returns
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        time_series = {}
        for i, (date_val, close_price) in enumerate(zip(dates, prices)):
            date_str = date_val.strftime('%Y-%m-%d')
            
            # Generate OHLC
            open_price = close_price * (1 + np.random.normal(0, 0.005))
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.01)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.01)))
            volume = int(np.random.lognormal(14, 0.5))  # Realistic volume distribution
            
            time_series[date_str] = {
                "1. open": f"{open_price:.2f}",
                "2. high": f"{high_price:.2f}",
                "3. low": f"{low_price:.2f}",
                "4. close": f"{close_price:.2f}",
                "5. volume": str(volume)
            }
        
        return {
            "Meta Data": {
                "1. Information": "Daily Prices (open, high, low, close) and Volumes",
                "2. Symbol": ticker,
                "3. Last Refreshed": dates[-1].strftime('%Y-%m-%d'),
                "4. Output Size": "Full size",
                "5. Time Zone": "US/Eastern"
            },
            "Time Series (Daily)": time_series
        }
    
    @staticmethod
    def company_overview(ticker: str) -> Dict[str, Any]:
        """Generate company overview data"""
        sectors = ["Technology", "Healthcare", "Finance", "Energy", "Consumer Discretionary", "Industrials"]
        
        # Generate correlated financial metrics
        market_cap = random.uniform(1e9, 500e9)  # $1B to $500B
        revenue = market_cap * random.uniform(0.5, 3.0)  # Revenue multiple
        net_margin = random.uniform(0.05, 0.30)
        net_income = revenue * net_margin
        
        return {
            "Symbol": ticker,
            "AssetType": "Common Stock",
            "Name": f"{ticker} Inc.",
            "Description": f"{ticker} is a technology company focused on innovation.",
            "CIK": str(random.randint(1000000, 9999999)),
            "Exchange": random.choice(["NASDAQ", "NYSE"]),
            "Currency": "USD",
            "Country": "USA",
            "Sector": random.choice(sectors),
            "Industry": "Software",
            "Address": "123 Tech Street, San Francisco, CA 94105",
            "MarketCapitalization": str(int(market_cap)),
            "EBITDA": str(int(net_income * 1.5)),
            "PERatio": str(round(random.uniform(15, 35), 2)),
            "PEGRatio": str(round(random.uniform(0.8, 2.5), 2)),
            "BookValue": str(round(random.uniform(10, 50), 2)),
            "DividendPerShare": str(round(random.uniform(0, 5), 2)),
            "DividendYield": str(round(random.uniform(0, 0.06), 4)),
            "EPS": str(round(net_income / (market_cap / 100), 2)),  # Assuming $100 share price
            "RevenuePerShareTTM": str(round(revenue / (market_cap / 100), 2)),
            "ProfitMargin": str(round(net_margin, 4)),
            "OperatingMarginTTM": str(round(net_margin * 1.2, 4)),
            "ReturnOnAssetsTTM": str(round(random.uniform(0.05, 0.20), 4)),
            "ReturnOnEquityTTM": str(round(random.uniform(0.10, 0.35), 4)),
            "RevenueTTM": str(int(revenue)),
            "GrossProfitTTM": str(int(revenue * random.uniform(0.6, 0.9))),
            "DilutedEPSTTM": str(round(net_income / (market_cap / 100), 2)),
            "QuarterlyEarningsGrowthYOY": str(round(random.uniform(-0.2, 0.4), 4)),
            "QuarterlyRevenueGrowthYOY": str(round(random.uniform(-0.1, 0.3), 4)),
            "AnalystTargetPrice": str(round(random.uniform(80, 200), 2)),
            "TrailingPE": str(round(random.uniform(15, 35), 2)),
            "ForwardPE": str(round(random.uniform(12, 30), 2)),
            "PriceToSalesRatioTTM": str(round(random.uniform(2, 15), 2)),
            "PriceToBookRatio": str(round(random.uniform(1, 8), 2)),
            "EVToRevenue": str(round(random.uniform(3, 20), 2)),
            "EVToEBITDA": str(round(random.uniform(10, 25), 2)),
            "Beta": str(round(random.uniform(0.5, 2.0), 2)),
            "52WeekHigh": str(round(random.uniform(120, 250), 2)),
            "52WeekLow": str(round(random.uniform(60, 150), 2)),
            "50DayMovingAverage": str(round(random.uniform(90, 180), 2)),
            "200DayMovingAverage": str(round(random.uniform(100, 200), 2)),
            "SharesOutstanding": str(int(random.uniform(1e9, 10e9))),
            "DividendDate": "2024-03-15",
            "ExDividendDate": "2024-02-28"
        }
    
    @staticmethod
    def technical_indicators(ticker: str, indicator: str) -> Dict[str, Any]:
        """Generate technical indicator data"""
        dates = pd.date_range(end=date.today(), periods=100, freq='D')
        
        if indicator.upper() == "RSI":
            data = {
                date_val.strftime('%Y-%m-%d'): {"RSI": str(round(random.uniform(20, 80), 4))}
                for date_val in dates
            }
            meta_data = {
                "1. Information": "Relative Strength Index (RSI)",
                "2. Symbol": ticker,
                "3. Indicator": "RSI",
                "4. Time Period": 14,
                "5. Series Type": "close"
            }
        elif indicator.upper() == "MACD":
            data = {
                date_val.strftime('%Y-%m-%d'): {
                    "MACD": str(round(random.uniform(-2, 2), 4)),
                    "MACD_Hist": str(round(random.uniform(-1, 1), 4)),
                    "MACD_Signal": str(round(random.uniform(-2, 2), 4))
                }
                for date_val in dates
            }
            meta_data = {
                "1. Information": "Moving Average Convergence/Divergence (MACD)",
                "2. Symbol": ticker,
                "3. Indicator": "MACD"
            }
        else:
            # Generic SMA/EMA
            data = {
                date_val.strftime('%Y-%m-%d'): {indicator: str(round(random.uniform(80, 150), 2))}
                for date_val in dates
            }
            meta_data = {
                "1. Information": f"{indicator}",
                "2. Symbol": ticker,
                "3. Indicator": indicator
            }
        
        return {
            "Meta Data": meta_data,
            f"Technical Analysis: {indicator}": data
        }


class FinnhubMocks:
    """Mock responses for Finnhub API"""
    
    @staticmethod
    def company_profile(ticker: str) -> Dict[str, Any]:
        """Generate company profile data"""
        return {
            "country": "US",
            "currency": "USD",
            "exchange": "NASDAQ NMS - GLOBAL MARKET",
            "ipo": "1980-12-12",
            "marketCapitalization": random.uniform(1000000, 3000000),  # Market cap in millions
            "name": f"{ticker} Inc",
            "phone": "14089961010",
            "shareOutstanding": random.uniform(15000, 17000),  # Shares outstanding in millions
            "ticker": ticker,
            "weburl": f"https://www.{ticker.lower()}.com/",
            "logo": f"https://static.finnhub.io/logo/{ticker.lower()}.png",
            "finnhubIndustry": "Technology"
        }
    
    @staticmethod
    def quote(ticker: str) -> Dict[str, Any]:
        """Generate real-time quote data"""
        base_price = random.uniform(100, 200)
        change = random.uniform(-10, 10)
        
        return {
            "c": round(base_price, 2),  # Current price
            "d": round(change, 2),  # Change
            "dp": round((change / base_price) * 100, 2),  # Percent change
            "h": round(base_price + abs(random.uniform(0, 5)), 2),  # High
            "l": round(base_price - abs(random.uniform(0, 5)), 2),  # Low
            "o": round(base_price + random.uniform(-3, 3), 2),  # Open
            "pc": round(base_price - change, 2),  # Previous close
            "t": int(datetime.now().timestamp())  # Timestamp
        }
    
    @staticmethod
    def basic_financials(ticker: str) -> Dict[str, Any]:
        """Generate basic financial metrics"""
        return {
            "symbol": ticker,
            "metricType": "all",
            "metric": {
                "10DayAverageTradingVolume": random.uniform(20000000, 100000000),
                "52WeekHigh": random.uniform(150, 200),
                "52WeekLow": random.uniform(80, 120),
                "52WeekLowDate": "2023-10-30",
                "52WeekHighDate": "2024-01-15",
                "beta": round(random.uniform(0.8, 1.5), 2),
                "marketCapitalization": random.uniform(1000000, 3000000),
                "peBasicExclExtraTTM": round(random.uniform(15, 30), 2),
                "peTTM": round(random.uniform(15, 30), 2),
                "roeTTM": round(random.uniform(0.15, 0.35), 4),
                "roaTTM": round(random.uniform(0.08, 0.20), 4),
                "grossMarginTTM": round(random.uniform(0.35, 0.65), 4),
                "netProfitMarginTTM": round(random.uniform(0.10, 0.30), 4),
                "operatingMarginTTM": round(random.uniform(0.20, 0.40), 4),
                "currentRatio": round(random.uniform(1.0, 3.0), 2),
                "quickRatio": round(random.uniform(0.8, 2.5), 2),
                "totalDebtToEquity": round(random.uniform(0.2, 1.5), 2),
                "totalDebtToTotalAsset": round(random.uniform(0.15, 0.45), 4),
                "totalDebtToTotalCapital": round(random.uniform(0.20, 0.50), 4),
                "freeCashFlowPerShareTTM": round(random.uniform(5, 15), 2),
                "bookValue": round(random.uniform(15, 40), 2),
                "priceToBookRatio": round(random.uniform(2, 8), 2),
                "priceToSalesRatio": round(random.uniform(3, 12), 2),
                "enterpriseValueOverEBIT": round(random.uniform(15, 35), 2),
                "evToSales": round(random.uniform(4, 15), 2)
            },
            "series": {
                "annual": {
                    "currentRatio": [
                        {"period": "2023-09-30", "v": random.uniform(1.0, 2.0)},
                        {"period": "2022-09-30", "v": random.uniform(1.0, 2.0)},
                        {"period": "2021-09-30", "v": random.uniform(1.0, 2.0)}
                    ],
                    "totalDebt": [
                        {"period": "2023-09-30", "v": random.uniform(100000, 150000)},
                        {"period": "2022-09-30", "v": random.uniform(90000, 140000)},
                        {"period": "2021-09-30", "v": random.uniform(80000, 130000)}
                    ]
                }
            }
        }
    
    @staticmethod
    def news_sentiment(ticker: str) -> Dict[str, Any]:
        """Generate news sentiment data"""
        articles = []
        for i in range(10):
            sentiment_score = random.uniform(-1, 1)
            articles.append({
                "id": f"news_{i}",
                "title": f"News about {ticker} - Article {i}",
                "url": f"https://example.com/news/{i}",
                "time_published": (datetime.now() - timedelta(days=random.randint(0, 30))).strftime('%Y%m%dT%H%M%S'),
                "authors": ["Financial Reporter"],
                "summary": f"Article discussing {ticker} business developments.",
                "banner_image": f"https://example.com/image_{i}.jpg",
                "source": "Financial News Network",
                "category_within_source": "Technology",
                "source_domain": "example.com",
                "overall_sentiment_score": sentiment_score,
                "overall_sentiment_label": "Bullish" if sentiment_score > 0.1 else "Bearish" if sentiment_score < -0.1 else "Neutral",
                "ticker_sentiment": [{
                    "ticker": ticker,
                    "relevance_score": random.uniform(0.5, 1.0),
                    "ticker_sentiment_score": sentiment_score,
                    "ticker_sentiment_label": "Bullish" if sentiment_score > 0.1 else "Bearish" if sentiment_score < -0.1 else "Neutral"
                }]
            })
        
        return {
            "items": str(len(articles)),
            "sentiment_score_definition": "x <= -0.35: Bearish; -0.35 < x <= -0.15: Somewhat-Bearish; -0.15 < x < 0.15: Neutral; 0.15 <= x < 0.35: Somewhat_Bullish; x >= 0.35: Bullish",
            "relevance_score_definition": "0 - 1, with higher numbers indicating higher relevance.",
            "feed": articles
        }


class PolygonMocks:
    """Mock responses for Polygon.io API"""
    
    @staticmethod
    def aggregates(ticker: str, timespan: str = "day", from_date: str = None, to_date: str = None) -> Dict[str, Any]:
        """Generate aggregates (OHLCV) data"""
        if not from_date:
            from_date = (date.today() - timedelta(days=100)).strftime('%Y-%m-%d')
        if not to_date:
            to_date = date.today().strftime('%Y-%m-%d')
        
        # Generate realistic stock data
        date_range = pd.date_range(start=from_date, end=to_date, freq='D')
        base_price = random.uniform(50, 300)
        
        results = []
        current_price = base_price
        
        for date_val in date_range:
            # Generate daily return
            daily_return = np.random.normal(0.001, 0.02)
            current_price *= (1 + daily_return)
            
            # Generate OHLC
            open_price = current_price * (1 + np.random.normal(0, 0.005))
            high_price = max(open_price, current_price) * (1 + abs(np.random.normal(0, 0.01)))
            low_price = min(open_price, current_price) * (1 - abs(np.random.normal(0, 0.01)))
            volume = int(np.random.lognormal(14, 0.5))
            
            results.append({
                "v": volume,  # Volume
                "vw": round(current_price, 2),  # Volume weighted average price
                "o": round(open_price, 2),  # Open
                "c": round(current_price, 2),  # Close
                "h": round(high_price, 2),  # High
                "l": round(low_price, 2),  # Low
                "t": int(date_val.timestamp() * 1000),  # Timestamp
                "n": random.randint(500, 2000)  # Number of transactions
            })
        
        return {
            "ticker": ticker,
            "queryCount": len(results),
            "resultsCount": len(results),
            "adjusted": True,
            "results": results,
            "status": "OK",
            "request_id": f"req_{random.randint(1000000, 9999999)}",
            "count": len(results)
        }
    
    @staticmethod
    def ticker_details(ticker: str) -> Dict[str, Any]:
        """Generate ticker details"""
        return {
            "results": {
                "ticker": ticker,
                "name": f"{ticker} Inc.",
                "market": "stocks",
                "locale": "us",
                "primary_exchange": "XNAS",
                "type": "CS",
                "active": True,
                "currency_name": "usd",
                "cik": str(random.randint(1000000, 9999999)),
                "composite_figi": f"BBG00{random.randint(1000000, 9999999)}",
                "share_class_figi": f"BBG00{random.randint(1000000, 9999999)}L11",
                "market_cap": random.uniform(1e9, 500e9),
                "phone_number": "14089961010",
                "address": {
                    "address1": "One Apple Park Way",
                    "city": "Cupertino",
                    "state": "CA",
                    "postal_code": "95014"
                },
                "description": f"{ticker} Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide.",
                "sic_code": "3571",
                "sic_description": "Electronic Computers",
                "ticker_root": ticker,
                "homepage_url": f"https://www.{ticker.lower()}.com",
                "total_employees": random.randint(50000, 200000),
                "list_date": "1980-12-12",
                "branding": {
                    "logo_url": f"https://api.polygon.io/v1/reference/company-branding/{ticker.lower()}/images/",
                    "icon_url": f"https://api.polygon.io/v1/reference/company-branding/{ticker.lower()}/images/"
                },
                "share_class_shares_outstanding": random.uniform(15e9, 18e9),
                "weighted_shares_outstanding": random.uniform(15e9, 18e9)
            },
            "status": "OK",
            "request_id": f"req_{random.randint(1000000, 9999999)}"
        }


class NewsAPIMocks:
    """Mock responses for News API"""
    
    @staticmethod
    def everything(query: str, language: str = "en", sort_by: str = "publishedAt") -> Dict[str, Any]:
        """Generate news articles"""
        articles = []
        sentiments = ["positive", "negative", "neutral"]
        sources = ["Reuters", "Bloomberg", "CNBC", "MarketWatch", "Yahoo Finance", "Financial Times"]
        
        for i in range(20):
            sentiment = random.choice(sentiments)
            
            if sentiment == "positive":
                headline = f"{query} shows strong performance in latest quarter"
                description = f"Positive developments for {query} continue to drive investor confidence"
            elif sentiment == "negative":
                headline = f"Concerns raised about {query} future prospects"
                description = f"Analysts express caution regarding {query} upcoming challenges"
            else:
                headline = f"{query} reports quarterly results"
                description = f"Regular business update from {query} provides market transparency"
            
            articles.append({
                "source": {
                    "id": None,
                    "name": random.choice(sources)
                },
                "author": "Financial Reporter",
                "title": headline,
                "description": description,
                "url": f"https://example.com/news/{i}",
                "urlToImage": f"https://example.com/image_{i}.jpg",
                "publishedAt": (datetime.now() - timedelta(hours=random.randint(1, 72))).isoformat() + "Z",
                "content": f"Full content about {query} would go here..."
            })
        
        return {
            "status": "ok",
            "totalResults": len(articles),
            "articles": articles
        }


class SECEdgarMocks:
    """Mock responses for SEC Edgar API"""
    
    @staticmethod
    def company_tickers() -> Dict[str, Any]:
        """Generate company ticker mapping"""
        return {
            "0": {
                "cik_str": 320193,
                "ticker": "AAPL",
                "title": "Apple Inc."
            },
            "1": {
                "cik_str": 789019,
                "ticker": "MSFT", 
                "title": "MICROSOFT CORP"
            },
            "2": {
                "cik_str": 1652044,
                "ticker": "GOOGL",
                "title": "Alphabet Inc."
            }
        }
    
    @staticmethod
    def company_facts(cik: str) -> Dict[str, Any]:
        """Generate company financial facts"""
        return {
            "cik": int(cik),
            "entityName": "Test Company Inc.",
            "facts": {
                "us-gaap": {
                    "Assets": {
                        "label": "Assets",
                        "description": "Total Assets",
                        "units": {
                            "USD": [
                                {
                                    "end": "2023-09-30",
                                    "val": random.uniform(300e9, 400e9),
                                    "accn": "0000320193-23-000106",
                                    "fy": 2023,
                                    "fp": "FY",
                                    "form": "10-K"
                                },
                                {
                                    "end": "2022-09-30", 
                                    "val": random.uniform(280e9, 380e9),
                                    "accn": "0000320193-22-000108",
                                    "fy": 2022,
                                    "fp": "FY",
                                    "form": "10-K"
                                }
                            ]
                        }
                    },
                    "Revenues": {
                        "label": "Revenues",
                        "description": "Total Revenues",
                        "units": {
                            "USD": [
                                {
                                    "end": "2023-09-30",
                                    "val": random.uniform(350e9, 450e9),
                                    "accn": "0000320193-23-000106", 
                                    "fy": 2023,
                                    "fp": "FY",
                                    "form": "10-K"
                                }
                            ]
                        }
                    }
                }
            }
        }


# Mock Database Fixtures
class MockDatabaseFixtures:
    """Database fixtures for testing"""
    
    @staticmethod
    def create_sample_stocks(count: int = 10) -> List[Dict[str, Any]]:
        """Create sample stock records"""
        sectors = ["Technology", "Healthcare", "Finance", "Energy", "Consumer Discretionary"]
        exchanges = ["NYSE", "NASDAQ", "AMEX"]
        
        stocks = []
        for i in range(count):
            ticker = f"TEST{i:03d}"
            stocks.append({
                "ticker": ticker,
                "company_name": f"Test Company {i}",
                "sector": random.choice(sectors),
                "industry": "Software",
                "exchange": random.choice(exchanges),
                "market_cap": random.uniform(1e9, 100e9),
                "shares_outstanding": random.uniform(100e6, 10e9),
                "is_active": True,
                "created_at": datetime.now() - timedelta(days=random.randint(1, 365)),
                "updated_at": datetime.now()
            })
        
        return stocks
    
    @staticmethod
    def create_sample_price_history(ticker: str, days: int = 252) -> List[Dict[str, Any]]:
        """Create sample price history"""
        base_price = random.uniform(50, 200)
        dates = pd.date_range(end=date.today(), periods=days, freq='D')
        
        price_history = []
        current_price = base_price
        
        for date_val in dates:
            # Generate daily return with some trend
            daily_return = np.random.normal(0.0005, 0.02)  # Slight upward bias
            current_price *= (1 + daily_return)
            
            # Generate OHLC
            open_price = current_price * (1 + np.random.normal(0, 0.005))
            high_price = max(open_price, current_price) * (1 + abs(np.random.normal(0, 0.01)))
            low_price = min(open_price, current_price) * (1 - abs(np.random.normal(0, 0.01)))
            volume = int(np.random.lognormal(14, 0.5))
            
            price_history.append({
                "ticker": ticker,
                "date": date_val.date(),
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(current_price, 2),
                "volume": volume,
                "adj_close": round(current_price, 2),  # Simplified
                "dividend_amount": 0.0,
                "split_coefficient": 1.0
            })
        
        return price_history


# Pytest Fixtures
@pytest.fixture
def mock_alpha_vantage_client():
    """Mock Alpha Vantage client"""
    client = Mock()
    client.get_daily_prices = AsyncMock(side_effect=lambda ticker, **kwargs: AlphaVantageMocks.daily_stock_data(ticker))
    client.get_company_overview = AsyncMock(side_effect=lambda ticker: AlphaVantageMocks.company_overview(ticker))
    client.get_technical_indicator = AsyncMock(side_effect=lambda ticker, indicator, **kwargs: AlphaVantageMocks.technical_indicators(ticker, indicator))
    return client

@pytest.fixture
def mock_finnhub_client():
    """Mock Finnhub client"""
    client = Mock()
    client.get_company_profile = AsyncMock(side_effect=lambda ticker: FinnhubMocks.company_profile(ticker))
    client.get_quote = AsyncMock(side_effect=lambda ticker: FinnhubMocks.quote(ticker))
    client.get_basic_financials = AsyncMock(side_effect=lambda ticker: FinnhubMocks.basic_financials(ticker))
    client.get_news_sentiment = AsyncMock(side_effect=lambda ticker: FinnhubMocks.news_sentiment(ticker))
    return client

@pytest.fixture
def mock_polygon_client():
    """Mock Polygon client"""
    client = Mock()
    client.get_aggregates = AsyncMock(side_effect=lambda ticker, **kwargs: PolygonMocks.aggregates(ticker, **kwargs))
    client.get_ticker_details = AsyncMock(side_effect=lambda ticker: PolygonMocks.ticker_details(ticker))
    return client

@pytest.fixture
def mock_news_api_client():
    """Mock News API client"""
    client = Mock()
    client.get_everything = AsyncMock(side_effect=lambda query, **kwargs: NewsAPIMocks.everything(query, **kwargs))
    return client

@pytest.fixture
def mock_sec_edgar_client():
    """Mock SEC Edgar client"""
    client = Mock()
    client.get_company_tickers = AsyncMock(return_value=SECEdgarMocks.company_tickers())
    client.get_company_facts = AsyncMock(side_effect=lambda cik: SECEdgarMocks.company_facts(cik))
    return client

@pytest.fixture
def sample_stock_data():
    """Sample stock data for testing"""
    return MockDatabaseFixtures.create_sample_stocks(10)

@pytest.fixture
def sample_price_history():
    """Sample price history for testing"""
    return MockDatabaseFixtures.create_sample_price_history("AAPL", 252)

@pytest.fixture
def mock_external_apis(mock_alpha_vantage_client, mock_finnhub_client, 
                      mock_polygon_client, mock_news_api_client, mock_sec_edgar_client):
    """Combined mock for all external APIs"""
    return {
        'alpha_vantage': mock_alpha_vantage_client,
        'finnhub': mock_finnhub_client,
        'polygon': mock_polygon_client,
        'news_api': mock_news_api_client,
        'sec_edgar': mock_sec_edgar_client
    }

@pytest.fixture
def market_scenarios():
    """Different market scenarios for testing"""
    return {
        'bull_market': {
            'trend': 'bullish',
            'volatility': 0.15,
            'expected_return': 0.12,
            'description': 'Strong upward trend with moderate volatility'
        },
        'bear_market': {
            'trend': 'bearish', 
            'volatility': 0.30,
            'expected_return': -0.15,
            'description': 'Declining market with high volatility'
        },
        'volatile_market': {
            'trend': 'neutral',
            'volatility': 0.45,
            'expected_return': 0.02,
            'description': 'High volatility with no clear direction'
        },
        'sideways_market': {
            'trend': 'neutral',
            'volatility': 0.10,
            'expected_return': 0.05,
            'description': 'Range-bound market with low volatility'
        }
    }

@pytest.fixture
def economic_indicators():
    """Mock economic indicators"""
    return {
        'interest_rates': {
            'federal_funds_rate': 5.25,
            '10_year_treasury': 4.20,
            '2_year_treasury': 4.80,
            'treasury_spread': -0.60  # Inverted yield curve
        },
        'inflation': {
            'cpi_yoy': 3.2,
            'core_cpi_yoy': 2.8,
            'pce_yoy': 2.9
        },
        'employment': {
            'unemployment_rate': 3.8,
            'non_farm_payrolls_change': 180000,
            'participation_rate': 62.9,
            'average_hourly_earnings_yoy': 4.1
        },
        'gdp': {
            'quarterly_growth_rate': 0.6,  # Annualized
            'annual_growth_rate': 2.4
        },
        'market_indicators': {
            'vix': 18.5,  # Market fear index
            'dollar_index': 103.2,
            'oil_price': 85.4,
            'gold_price': 2020.5,
            'bitcoin_price': 42000.0
        }
    }

# Mock response builders for different test scenarios
class ResponseBuilder:
    """Helper class to build mock responses for different scenarios"""
    
    @staticmethod
    def build_error_response(status_code: int, error_message: str):
        """Build error response"""
        mock_response = Mock()
        mock_response.status_code = status_code
        mock_response.json.return_value = {"error": error_message}
        mock_response.raise_for_status.side_effect = Exception(error_message)
        return mock_response
    
    @staticmethod
    def build_rate_limited_response():
        """Build rate limited response"""
        return ResponseBuilder.build_error_response(429, "Rate limit exceeded")
    
    @staticmethod
    def build_timeout_response():
        """Build timeout response"""
        mock_response = Mock()
        mock_response.side_effect = asyncio.TimeoutError("Request timeout")
        return mock_response
    
    @staticmethod
    def build_success_response(data: Dict[str, Any]):
        """Build successful response"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = data
        mock_response.raise_for_status.return_value = None
        return mock_response


# Utility functions for test data generation
def generate_correlated_financial_data(base_metrics: Dict[str, float]) -> Dict[str, float]:
    """Generate correlated financial metrics"""
    # Use base metrics to generate realistic correlations
    market_cap = base_metrics.get('market_cap', 100e9)
    
    # Generate revenue based on market cap (typical P/S ratios)
    ps_ratio = random.uniform(2, 15)
    revenue = market_cap / ps_ratio
    
    # Generate profit margins
    net_margin = random.uniform(0.05, 0.25)
    net_income = revenue * net_margin
    
    # Generate balance sheet items
    asset_turnover = random.uniform(0.5, 2.0)
    total_assets = revenue / asset_turnover
    
    equity_ratio = random.uniform(0.3, 0.7)
    total_equity = total_assets * equity_ratio
    total_debt = total_assets - total_equity - (total_equity * random.uniform(0.1, 0.3))  # Some current liabilities
    
    return {
        'market_cap': market_cap,
        'revenue': revenue,
        'net_income': net_income,
        'total_assets': total_assets,
        'total_equity': total_equity,
        'total_debt': total_debt,
        'pe_ratio': market_cap / net_income if net_income > 0 else None,
        'roe': net_income / total_equity if total_equity > 0 else None,
        'debt_to_equity': total_debt / total_equity if total_equity > 0 else None,
        'current_ratio': random.uniform(1.0, 3.0),
        'quick_ratio': random.uniform(0.8, 2.5),
        'gross_margin': random.uniform(0.3, 0.8),
        'operating_margin': net_margin * random.uniform(1.1, 1.5),
        'net_margin': net_margin
    }


if __name__ == "__main__":
    # Example usage
    av_mock = AlphaVantageMocks()
    sample_data = av_mock.daily_stock_data("AAPL", 30)
    print("Sample Alpha Vantage data keys:", list(sample_data.keys()))