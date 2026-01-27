"""
Data ingestion module for fetching market data from various sources.

This module provides comprehensive market data ingestion with multi-provider
support and intelligent fallback chains.

Providers:
- Yahoo Finance (yfinance) - Unlimited, primary for prices/fundamentals
- Finnhub (60/min) - Real-time quotes, news, sentiment
- Alpha Vantage (25/day, 5/min) - Historical, fundamentals, technical
- Polygon (5/min) - Market data, aggregates
- SEC EDGAR (Unlimited) - Fundamental data, filings
- Financial Modeling Prep - Fundamentals, DCF
- News API (100/day) - News headlines
- FRED (1000/day) - Economic data
"""

from backend.data_ingestion.market_scanner import (
    MarketScanner,
    DataProvider,
    ProviderHealth,
    StockQuote,
    StockFundamentals,
    get_market_scanner
)
from backend.data_ingestion.finnhub_client import FinnhubClient
from backend.data_ingestion.alpha_vantage_client import AlphaVantageClient
from backend.data_ingestion.polygon_client import PolygonClient
from backend.data_ingestion.sec_edgar_client import SECEdgarClient
from backend.data_ingestion.base_client import BaseAPIClient
from backend.data_ingestion.robust_api_client import RobustAPIClient

__all__ = [
    # Main Scanner
    'MarketScanner',
    'get_market_scanner',

    # Data Types
    'DataProvider',
    'ProviderHealth',
    'StockQuote',
    'StockFundamentals',

    # Clients
    'FinnhubClient',
    'AlphaVantageClient',
    'PolygonClient',
    'SECEdgarClient',
    'BaseAPIClient',
    'RobustAPIClient',
]