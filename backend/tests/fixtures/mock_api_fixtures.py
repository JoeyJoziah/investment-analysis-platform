"""
Mock API Fixtures
Comprehensive mock objects for external API services and dependencies.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, date, timedelta
import asyncio
import json
import random
from dataclasses import dataclass, asdict
from enum import Enum

import pandas as pd
import numpy as np


class APIResponseStatus(Enum):
    """API response status codes"""
    SUCCESS = "success"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    UNAUTHORIZED = "unauthorized"


@dataclass
class MockAPIResponse:
    """Mock API response structure"""
    status: APIResponseStatus
    data: Any
    message: str = ""
    headers: Dict[str, str] = None
    status_code: int = 200
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = {}


class MockAlphaVantageClient:
    """Mock Alpha Vantage API client"""
    
    def __init__(self, behavior: str = "normal"):
        """
        Initialize mock client
        Args:
            behavior: 'normal', 'rate_limited', 'error', 'timeout'
        """
        self.behavior = behavior
        self.call_count = 0
        self.daily_calls = 0
        self.max_daily_calls = 25  # Alpha Vantage free tier limit
    
    async def get_daily_prices(self, symbol: str, **kwargs) -> MockAPIResponse:
        """Mock daily price data"""
        self.call_count += 1
        self.daily_calls += 1
        
        if self.behavior == "rate_limited" or self.daily_calls > self.max_daily_calls:
            return MockAPIResponse(
                status=APIResponseStatus.RATE_LIMITED,
                data=None,
                message="API call frequency exceeded",
                status_code=429
            )
        
        if self.behavior == "error":
            return MockAPIResponse(
                status=APIResponseStatus.ERROR,
                data=None,
                message="Invalid API key",
                status_code=401
            )
        
        if self.behavior == "timeout":
            await asyncio.sleep(10)  # Simulate timeout
        
        # Generate mock price data
        dates = pd.date_range(end=datetime.now().date(), periods=100, freq='D')
        base_price = 100 + hash(symbol) % 200
        
        price_data = {}
        for i, date_val in enumerate(dates):
            price = base_price + random.uniform(-5, 5) + i * 0.1
            price_data[date_val.strftime('%Y-%m-%d')] = {
                '1. open': f"{price:.2f}",
                '2. high': f"{price + random.uniform(0, 3):.2f}",
                '3. low': f"{price - random.uniform(0, 3):.2f}",
                '4. close': f"{price + random.uniform(-2, 2):.2f}",
                '5. volume': str(random.randint(1000000, 10000000))
            }
        
        return MockAPIResponse(
            status=APIResponseStatus.SUCCESS,
            data={
                "Meta Data": {
                    "2. Symbol": symbol,
                    "3. Last Refreshed": datetime.now().strftime('%Y-%m-%d')
                },
                "Time Series (Daily)": price_data
            }
        )
    
    async def get_company_overview(self, symbol: str) -> MockAPIResponse:
        """Mock company overview data"""
        self.call_count += 1
        self.daily_calls += 1
        
        if self.daily_calls > self.max_daily_calls:
            return MockAPIResponse(
                status=APIResponseStatus.RATE_LIMITED,
                data=None,
                message="API call frequency exceeded",
                status_code=429
            )
        
        # Generate mock company data
        market_cap = random.randint(1000000000, 3000000000000)
        
        return MockAPIResponse(
            status=APIResponseStatus.SUCCESS,
            data={
                "Symbol": symbol,
                "Name": f"{symbol} Corporation",
                "Exchange": "NASDAQ",
                "Currency": "USD",
                "Country": "USA",
                "Sector": random.choice(["Technology", "Healthcare", "Finance", "Consumer", "Industrial"]),
                "Industry": f"{symbol} Industry",
                "MarketCapitalization": str(market_cap),
                "BookValue": f"{random.uniform(10, 50):.2f}",
                "DividendPerShare": f"{random.uniform(0, 5):.2f}",
                "EPS": f"{random.uniform(1, 10):.2f}",
                "RevenuePerShareTTM": f"{random.uniform(20, 100):.2f}",
                "ProfitMargin": f"{random.uniform(0.05, 0.25):.4f}",
                "ReturnOnAssetsTTM": f"{random.uniform(0.05, 0.20):.4f}",
                "ReturnOnEquityTTM": f"{random.uniform(0.10, 0.30):.4f}",
                "RevenueTTM": str(int(market_cap * random.uniform(0.5, 2.0))),
                "GrossProfitTTM": str(int(market_cap * random.uniform(0.3, 1.0)))
            }
        )


class MockFinnhubClient:
    """Mock Finnhub API client"""
    
    def __init__(self, behavior: str = "normal"):
        self.behavior = behavior
        self.call_count = 0
        self.minute_calls = 0
        self.max_minute_calls = 60  # Finnhub free tier limit
        self.last_reset = datetime.now()
    
    def _reset_rate_limit(self):
        """Reset rate limit counter if minute has passed"""
        now = datetime.now()
        if (now - self.last_reset).seconds >= 60:
            self.minute_calls = 0
            self.last_reset = now
    
    async def get_quote(self, symbol: str) -> MockAPIResponse:
        """Mock real-time quote"""
        self._reset_rate_limit()
        self.call_count += 1
        self.minute_calls += 1
        
        if self.minute_calls > self.max_minute_calls:
            return MockAPIResponse(
                status=APIResponseStatus.RATE_LIMITED,
                data=None,
                message="Rate limit exceeded",
                status_code=429
            )
        
        base_price = 100 + hash(symbol) % 200
        current_price = base_price + random.uniform(-10, 10)
        
        return MockAPIResponse(
            status=APIResponseStatus.SUCCESS,
            data={
                "c": current_price,  # Current price
                "h": current_price + random.uniform(0, 5),  # High
                "l": current_price - random.uniform(0, 5),  # Low
                "o": current_price + random.uniform(-2, 2),  # Open
                "pc": current_price - random.uniform(-1, 1),  # Previous close
                "t": int(datetime.now().timestamp())  # Timestamp
            }
        )
    
    async def get_company_news(self, symbol: str, from_date: str, to_date: str) -> MockAPIResponse:
        """Mock company news"""
        self._reset_rate_limit()
        self.call_count += 1
        self.minute_calls += 1
        
        if self.minute_calls > self.max_minute_calls:
            return MockAPIResponse(
                status=APIResponseStatus.RATE_LIMITED,
                data=None,
                message="Rate limit exceeded",
                status_code=429
            )
        
        # Generate mock news articles
        news_articles = []
        sentiments = ['positive', 'negative', 'neutral']
        
        for i in range(random.randint(3, 8)):
            sentiment = random.choice(sentiments)
            sentiment_score = {
                'positive': random.uniform(0.3, 0.8),
                'negative': random.uniform(-0.8, -0.3),
                'neutral': random.uniform(-0.2, 0.2)
            }[sentiment]
            
            news_articles.append({
                "category": "company",
                "datetime": int((datetime.now() - timedelta(hours=i)).timestamp()),
                "headline": f"{symbol} {sentiment} news headline {i}",
                "id": random.randint(100000, 999999),
                "image": f"https://example.com/image_{i}.jpg",
                "related": symbol,
                "source": random.choice(["Reuters", "Bloomberg", "CNBC", "MarketWatch"]),
                "summary": f"Mock {sentiment} news summary for {symbol}",
                "url": f"https://example.com/news/{symbol.lower()}_{i}",
                "sentiment_score": sentiment_score
            })
        
        return MockAPIResponse(
            status=APIResponseStatus.SUCCESS,
            data=news_articles
        )


class MockPolygonClient:
    """Mock Polygon.io API client"""
    
    def __init__(self, behavior: str = "normal"):
        self.behavior = behavior
        self.call_count = 0
        self.minute_calls = 0
        self.max_minute_calls = 5  # Polygon free tier limit
        self.last_reset = datetime.now()
    
    def _reset_rate_limit(self):
        """Reset rate limit counter"""
        now = datetime.now()
        if (now - self.last_reset).seconds >= 60:
            self.minute_calls = 0
            self.last_reset = now
    
    async def get_aggregates(
        self,
        ticker: str,
        multiplier: int,
        timespan: str,
        from_date: str,
        to_date: str
    ) -> MockAPIResponse:
        """Mock aggregate data"""
        self._reset_rate_limit()
        self.call_count += 1
        self.minute_calls += 1
        
        if self.minute_calls > self.max_minute_calls:
            return MockAPIResponse(
                status=APIResponseStatus.RATE_LIMITED,
                data=None,
                message="Rate limit exceeded",
                status_code=429
            )
        
        # Generate mock aggregate data
        base_price = 100 + hash(ticker) % 200
        results = []
        
        # Generate 30 days of data
        for i in range(30):
            price = base_price + random.uniform(-10, 10) + i * 0.1
            results.append({
                "c": price + random.uniform(-1, 1),  # Close
                "h": price + random.uniform(0, 3),   # High
                "l": price - random.uniform(0, 3),   # Low
                "n": random.randint(1000, 5000),     # Number of transactions
                "o": price + random.uniform(-1, 1),  # Open
                "t": int((datetime.now() - timedelta(days=29-i)).timestamp() * 1000),  # Timestamp
                "v": random.randint(1000000, 10000000),  # Volume
                "vw": price  # Volume weighted average
            })
        
        return MockAPIResponse(
            status=APIResponseStatus.SUCCESS,
            data={
                "ticker": ticker,
                "queryCount": len(results),
                "resultsCount": len(results),
                "adjusted": True,
                "results": results,
                "status": "OK",
                "request_id": f"req_{random.randint(100000, 999999)}"
            }
        )


class MockSECEdgarClient:
    """Mock SEC EDGAR API client"""
    
    def __init__(self, behavior: str = "normal"):
        self.behavior = behavior
        self.call_count = 0
    
    async def get_company_facts(self, cik: str) -> MockAPIResponse:
        """Mock company facts from SEC"""
        self.call_count += 1
        
        if self.behavior == "error":
            return MockAPIResponse(
                status=APIResponseStatus.ERROR,
                data=None,
                message="CIK not found",
                status_code=404
            )
        
        # Generate mock SEC filing data
        return MockAPIResponse(
            status=APIResponseStatus.SUCCESS,
            data={
                "cik": int(cik),
                "entityName": f"Mock Company {cik}",
                "facts": {
                    "us-gaap": {
                        "Assets": {
                            "label": "Assets",
                            "description": "Sum of current and noncurrent assets",
                            "units": {
                                "USD": [
                                    {
                                        "end": "2023-12-31",
                                        "val": random.randint(1000000000, 100000000000),
                                        "accn": "0000123456-23-000001",
                                        "fy": 2023,
                                        "fp": "FY",
                                        "form": "10-K"
                                    }
                                ]
                            }
                        },
                        "Revenues": {
                            "label": "Revenues",
                            "description": "Total revenues",
                            "units": {
                                "USD": [
                                    {
                                        "end": "2023-12-31",
                                        "val": random.randint(500000000, 50000000000),
                                        "accn": "0000123456-23-000001",
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
        )


class MockNewsAPIClient:
    """Mock News API client for sentiment analysis"""
    
    def __init__(self, behavior: str = "normal"):
        self.behavior = behavior
        self.call_count = 0
        self.daily_calls = 0
        self.max_daily_calls = 1000  # News API free tier limit
    
    async def get_everything(
        self,
        q: str,
        sources: str = None,
        domains: str = None,
        from_param: str = None,
        to: str = None,
        language: str = 'en',
        sort_by: str = 'publishedAt',
        page: int = 1,
        page_size: int = 20
    ) -> MockAPIResponse:
        """Mock news search"""
        self.call_count += 1
        self.daily_calls += 1
        
        if self.daily_calls > self.max_daily_calls:
            return MockAPIResponse(
                status=APIResponseStatus.RATE_LIMITED,
                data=None,
                message="Daily request limit exceeded",
                status_code=429
            )
        
        # Generate mock articles
        articles = []
        for i in range(page_size):
            sentiment = random.choice(['positive', 'negative', 'neutral'])
            articles.append({
                "source": {
                    "id": random.choice(['reuters', 'bloomberg', 'cnbc', None]),
                    "name": random.choice(['Reuters', 'Bloomberg', 'CNBC', 'MarketWatch'])
                },
                "author": f"Mock Author {i}",
                "title": f"Mock {sentiment} article about {q} - {i}",
                "description": f"This is a mock {sentiment} article description about {q}.",
                "url": f"https://example.com/article_{i}",
                "urlToImage": f"https://example.com/image_{i}.jpg",
                "publishedAt": (datetime.now() - timedelta(hours=i)).isoformat() + "Z",
                "content": f"Mock article content about {q}. This is {sentiment} news."
            })
        
        return MockAPIResponse(
            status=APIResponseStatus.SUCCESS,
            data={
                "status": "ok",
                "totalResults": page_size * 10,  # Simulate more results available
                "articles": articles
            }
        )


class MockMLModelManager:
    """Mock ML Model Manager for testing model predictions"""
    
    def __init__(self, behavior: str = "normal"):
        self.behavior = behavior
        self.models_loaded = True
        self.prediction_cache = {}
    
    def predict_price_movement(
        self,
        ticker: str,
        features: Dict[str, Any],
        horizon_days: int = 5
    ) -> Dict[str, Any]:
        """Mock price movement prediction"""
        
        if self.behavior == "error":
            raise ValueError("Model prediction failed")
        
        # Generate consistent predictions for same inputs
        cache_key = f"{ticker}_{horizon_days}_{hash(str(sorted(features.items())))}"
        
        if cache_key not in self.prediction_cache:
            current_price = features.get('current_price', 100.0)
            
            # Simulate model uncertainty
            confidence = random.uniform(0.5, 0.9)
            price_change_pct = random.uniform(-0.1, 0.1)  # ±10%
            predicted_price = current_price * (1 + price_change_pct)
            
            # Add confidence intervals
            ci_width = (1 - confidence) * current_price * 0.1
            
            self.prediction_cache[cache_key] = {
                'predicted_price': predicted_price,
                'predicted_return': price_change_pct,
                'confidence': confidence,
                'confidence_interval': (
                    predicted_price - ci_width,
                    predicted_price + ci_width
                ),
                'horizon_days': horizon_days,
                'model_name': f'mock_model_{horizon_days}d',
                'features_used': list(features.keys()),
                'prediction_timestamp': datetime.now().isoformat()
            }
        
        return self.prediction_cache[cache_key]
    
    def get_model_performance(self, model_name: str) -> Dict[str, Any]:
        """Mock model performance metrics"""
        return {
            'accuracy': random.uniform(0.55, 0.75),
            'precision': random.uniform(0.50, 0.70),
            'recall': random.uniform(0.50, 0.70),
            'f1_score': random.uniform(0.50, 0.70),
            'sharpe_ratio': random.uniform(0.5, 2.0),
            'max_drawdown': random.uniform(0.05, 0.20),
            'win_rate': random.uniform(0.45, 0.65),
            'avg_return': random.uniform(0.001, 0.02),
            'volatility': random.uniform(0.15, 0.35),
            'last_updated': datetime.now().isoformat()
        }


class MockCacheManager:
    """Mock cache manager for testing caching behavior"""
    
    def __init__(self):
        self.cache = {}
        self.hits = 0
        self.misses = 0
        self.sets = 0
    
    async def get(self, key: str) -> Optional[Any]:
        """Mock cache get"""
        if key in self.cache:
            self.hits += 1
            return self.cache[key]['value']
        else:
            self.misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Mock cache set"""
        self.cache[key] = {
            'value': value,
            'ttl': ttl,
            'created_at': datetime.now()
        }
        self.sets += 1
        return True
    
    async def delete(self, key: str) -> bool:
        """Mock cache delete"""
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    async def clear(self) -> bool:
        """Mock cache clear"""
        self.cache.clear()
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'sets': self.sets,
            'hit_rate': hit_rate,
            'total_keys': len(self.cache)
        }


# Pytest fixtures
@pytest.fixture
def mock_alpha_vantage_client():
    """Fixture for mock Alpha Vantage client"""
    return MockAlphaVantageClient()


@pytest.fixture
def mock_finnhub_client():
    """Fixture for mock Finnhub client"""
    return MockFinnhubClient()


@pytest.fixture
def mock_polygon_client():
    """Fixture for mock Polygon client"""
    return MockPolygonClient()


@pytest.fixture
def mock_sec_client():
    """Fixture for mock SEC EDGAR client"""
    return MockSECEdgarClient()


@pytest.fixture
def mock_news_api_client():
    """Fixture for mock News API client"""
    return MockNewsAPIClient()


@pytest.fixture
def mock_ml_model_manager():
    """Fixture for mock ML model manager"""
    return MockMLModelManager()


@pytest.fixture
def mock_cache_manager():
    """Fixture for mock cache manager"""
    return MockCacheManager()


@pytest.fixture
def rate_limited_clients():
    """Fixture providing rate-limited mock clients"""
    return {
        'alpha_vantage': MockAlphaVantageClient(behavior="rate_limited"),
        'finnhub': MockFinnhubClient(behavior="rate_limited"),
        'polygon': MockPolygonClient(behavior="rate_limited")
    }


@pytest.fixture
def error_clients():
    """Fixture providing error-prone mock clients"""
    return {
        'alpha_vantage': MockAlphaVantageClient(behavior="error"),
        'finnhub': MockFinnhubClient(behavior="error"),
        'polygon': MockPolygonClient(behavior="error"),
        'sec': MockSECEdgarClient(behavior="error")
    }


@pytest.fixture
def mock_external_dependencies(
    mock_alpha_vantage_client,
    mock_finnhub_client,
    mock_polygon_client,
    mock_ml_model_manager,
    mock_cache_manager
):
    """Fixture providing all mock external dependencies"""
    return {
        'alpha_vantage': mock_alpha_vantage_client,
        'finnhub': mock_finnhub_client,
        'polygon': mock_polygon_client,
        'ml_manager': mock_ml_model_manager,
        'cache': mock_cache_manager
    }


# Context managers for patching
@pytest.fixture
def patch_external_apis():
    """Context manager to patch all external API clients"""
    
    class APIPatcher:
        def __init__(self):
            self.patchers = {}
        
        def __enter__(self):
            self.patchers['alpha_vantage'] = patch('backend.data_ingestion.alpha_vantage_client.AlphaVantageClient')
            self.patchers['finnhub'] = patch('backend.data_ingestion.finnhub_client.FinnhubClient')
            self.patchers['polygon'] = patch('backend.data_ingestion.polygon_client.PolygonClient')
            
            self.mocks = {}
            for name, patcher in self.patchers.items():
                self.mocks[name] = patcher.start()
            
            return self.mocks
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            for patcher in self.patchers.values():
                patcher.stop()
    
    return APIPatcher()


# Utility functions for creating mock responses
def create_mock_api_response(
    status: APIResponseStatus = APIResponseStatus.SUCCESS,
    data: Any = None,
    message: str = "",
    status_code: int = 200
) -> MockAPIResponse:
    """Create a mock API response"""
    return MockAPIResponse(
        status=status,
        data=data,
        message=message,
        status_code=status_code
    )


def simulate_api_latency(min_delay: float = 0.1, max_delay: float = 2.0) -> Callable:
    """Decorator to simulate API latency"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            delay = random.uniform(min_delay, max_delay)
            await asyncio.sleep(delay)
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def create_batch_mock_responses(
    symbols: List[str],
    response_factory: Callable[[str], MockAPIResponse]
) -> Dict[str, MockAPIResponse]:
    """Create batch mock responses for multiple symbols"""
    return {symbol: response_factory(symbol) for symbol in symbols}