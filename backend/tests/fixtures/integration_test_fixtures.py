"""
Comprehensive Test Fixtures and Utilities for Integration Testing
Provides reusable fixtures, mock data generators, and testing utilities.
"""

import pytest
import asyncio
import json
import random
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional, Callable
from unittest.mock import AsyncMock, MagicMock, patch
from decimal import Decimal
import numpy as np
import pandas as pd
from faker import Faker

from backend.models.unified_models import (
    User, Stock, Portfolio, PortfolioPosition, Transaction,
    PriceData, Recommendation, Alert
)
from backend.config.database import get_async_db_session
from backend.auth.oauth2 import create_access_token
from backend.utils.comprehensive_cache import ComprehensiveCacheManager


fake = Faker()
Faker.seed(12345)  # For reproducible test data


class TestDataGenerator:
    """Generate realistic test data for various entities."""
    
    @staticmethod
    def generate_user(
        username: Optional[str] = None,
        email: Optional[str] = None,
        is_active: bool = True,
        **kwargs
    ) -> User:
        """Generate a realistic user."""
        return User(
            username=username or fake.user_name(),
            email=email or fake.email(),
            first_name=fake.first_name(),
            last_name=fake.last_name(),
            hashed_password="$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",
            is_active=is_active,
            is_verified=True,
            phone_number=fake.phone_number(),
            date_of_birth=fake.date_of_birth(minimum_age=18, maximum_age=80),
            created_at=datetime.utcnow() - timedelta(days=random.randint(1, 365)),
            last_login=datetime.utcnow() - timedelta(hours=random.randint(1, 24)),
            **kwargs
        )
    
    @staticmethod
    def generate_stock(
        symbol: Optional[str] = None,
        sector: Optional[str] = None,
        **kwargs
    ) -> Stock:
        """Generate a realistic stock."""
        sectors = [
            "Technology", "Healthcare", "Financial Services", "Consumer Cyclical",
            "Communication Services", "Industrials", "Consumer Defensive",
            "Energy", "Utilities", "Real Estate", "Basic Materials"
        ]
        
        industries = {
            "Technology": ["Software—Application", "Semiconductors", "Software—Infrastructure"],
            "Healthcare": ["Biotechnology", "Medical Devices", "Drug Manufacturers—Major"],
            "Financial Services": ["Banks—Regional", "Insurance—Life", "Asset Management"],
            "Consumer Cyclical": ["Auto Manufacturers", "Restaurants", "Retail—Apparel"],
            "Communication Services": ["Telecom Services", "Entertainment", "Internet Content & Information"],
            "Industrials": ["Aerospace & Defense", "Industrial Distribution", "Construction"],
            "Consumer Defensive": ["Grocery Stores", "Household & Personal Products", "Beverages—Non-Alcoholic"],
            "Energy": ["Oil & Gas E&P", "Oil & Gas Refining & Marketing", "Oil & Gas Equipment & Services"],
            "Utilities": ["Utilities—Regulated Electric", "Utilities—Renewable", "Utilities—Regulated Gas"],
            "Real Estate": ["REIT—Residential", "REIT—Retail", "Real Estate Services"],
            "Basic Materials": ["Steel", "Chemicals", "Gold"]
        }
        
        selected_sector = sector or random.choice(sectors)
        selected_industry = random.choice(industries[selected_sector])
        
        return Stock(
            symbol=symbol or fake.lexify(text="????", letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
            name=fake.company(),
            sector=selected_sector,
            industry=selected_industry,
            market_cap=random.randint(100000000, 3000000000000),  # $100M to $3T
            description=fake.text(max_nb_chars=500),
            country="US",
            currency="USD",
            exchange=random.choice(["NASDAQ", "NYSE", "AMEX"]),
            **kwargs
        )
    
    @staticmethod
    def generate_portfolio(
        user_id: int,
        name: Optional[str] = None,
        **kwargs
    ) -> Portfolio:
        """Generate a realistic portfolio."""
        strategies = ["aggressive", "growth", "balanced", "conservative", "income", "preservation"]
        
        return Portfolio(
            user_id=user_id,
            name=name or f"{fake.word().title()} {random.choice(['Growth', 'Value', 'Income'])} Portfolio",
            description=fake.text(max_nb_chars=200),
            strategy=random.choice(strategies),
            cash_balance=Decimal(random.uniform(1000, 500000)),
            target_allocation={
                "stocks": random.uniform(0.4, 0.9),
                "bonds": random.uniform(0.05, 0.3),
                "cash": random.uniform(0.05, 0.2)
            },
            created_at=datetime.utcnow() - timedelta(days=random.randint(1, 1095)),  # Up to 3 years ago
            **kwargs
        )
    
    @staticmethod
    def generate_position(
        portfolio_id: int,
        symbol: str,
        **kwargs
    ) -> PortfolioPosition:
        """Generate a realistic portfolio position."""
        quantity = random.uniform(1, 1000)
        avg_cost = random.uniform(10, 1000)
        current_price = avg_cost * random.uniform(0.7, 1.5)  # -30% to +50%
        
        return PortfolioPosition(
            portfolio_id=portfolio_id,
            symbol=symbol,
            quantity=Decimal(quantity),
            average_cost=Decimal(avg_cost),
            current_price=Decimal(current_price),
            cost_basis=Decimal(quantity * avg_cost),
            market_value=Decimal(quantity * current_price),
            unrealized_gain=Decimal(quantity * (current_price - avg_cost)),
            realized_gain=Decimal(random.uniform(-1000, 5000)),
            created_at=datetime.utcnow() - timedelta(days=random.randint(1, 365)),
            **kwargs
        )
    
    @staticmethod
    def generate_price_data(
        symbol: str,
        start_date: date,
        days: int = 252,
        base_price: float = 100.0,
        volatility: float = 0.2
    ) -> List[PriceData]:
        """Generate realistic price data using random walk."""
        prices = []
        current_price = base_price
        
        for i in range(days):
            price_date = start_date + timedelta(days=i)
            
            # Random walk with slight upward bias
            daily_return = np.random.normal(0.0005, volatility / np.sqrt(252))  # ~12.6% annual volatility
            current_price *= (1 + daily_return)
            
            # Ensure minimum price
            current_price = max(current_price, 1.0)
            
            # Generate OHLC
            high = current_price * (1 + abs(np.random.normal(0, volatility / 50)))
            low = current_price * (1 - abs(np.random.normal(0, volatility / 50)))
            open_price = current_price * np.random.uniform(0.99, 1.01)
            
            # Ensure OHLC relationships
            high = max(high, open_price, current_price)
            low = min(low, open_price, current_price)
            
            volume = int(np.random.lognormal(15, 1))  # Realistic volume distribution
            
            prices.append(PriceData(
                symbol=symbol,
                date=price_date,
                open=Decimal(round(open_price, 2)),
                high=Decimal(round(high, 2)),
                low=Decimal(round(low, 2)),
                close=Decimal(round(current_price, 2)),
                volume=volume,
                adjusted_close=Decimal(round(current_price, 2))
            ))
        
        return prices
    
    @staticmethod
    def generate_transaction(
        portfolio_id: int,
        symbol: str,
        transaction_type: str = None,
        **kwargs
    ) -> Transaction:
        """Generate a realistic transaction."""
        transaction_types = ["buy", "sell", "dividend", "transfer_in", "transfer_out"]
        selected_type = transaction_type or random.choice(transaction_types)
        
        quantity = random.uniform(1, 500) if selected_type in ["buy", "sell"] else 0
        price = random.uniform(10, 1000) if selected_type in ["buy", "sell"] else 0
        
        if selected_type == "dividend":
            total_amount = random.uniform(10, 500)
        else:
            total_amount = quantity * price
        
        return Transaction(
            portfolio_id=portfolio_id,
            symbol=symbol,
            transaction_type=selected_type,
            quantity=Decimal(quantity) if quantity > 0 else None,
            price=Decimal(price) if price > 0 else None,
            total_amount=Decimal(total_amount),
            fees=Decimal(random.uniform(0, 10)) if selected_type in ["buy", "sell"] else Decimal(0),
            notes=fake.text(max_nb_chars=100),
            created_at=datetime.utcnow() - timedelta(days=random.randint(1, 365)),
            **kwargs
        )


class MockDataFactory:
    """Factory for creating mock objects and data."""
    
    @staticmethod
    def create_mock_user(user_id: int = 1, **kwargs) -> User:
        """Create a mock user with default values."""
        defaults = {
            "id": user_id,
            "username": f"testuser{user_id}",
            "email": f"test{user_id}@example.com",
            "is_active": True,
            "is_verified": True,
            "created_at": datetime.utcnow()
        }
        defaults.update(kwargs)
        return User(**defaults)
    
    @staticmethod
    def create_mock_api_response(data: Any, status_code: int = 200) -> MagicMock:
        """Create a mock HTTP response."""
        response = MagicMock()
        response.status_code = status_code
        response.json.return_value = data
        response.text = json.dumps(data)
        response.headers = {"content-type": "application/json"}
        return response
    
    @staticmethod
    def create_mock_database_session() -> AsyncMock:
        """Create a mock database session."""
        session = AsyncMock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        session.close = AsyncMock()
        session.execute = AsyncMock()
        session.query = MagicMock()
        return session
    
    @staticmethod
    def create_mock_cache_manager() -> AsyncMock:
        """Create a mock cache manager."""
        cache = AsyncMock()
        cache.get = AsyncMock(return_value=None)
        cache.set = AsyncMock()
        cache.delete = AsyncMock()
        cache.clear = AsyncMock()
        cache.get_stats = AsyncMock(return_value={
            "hits": 100,
            "misses": 20,
            "hit_rate": 0.83
        })
        return cache


@pytest.fixture
def test_data_generator():
    """Provide test data generator."""
    return TestDataGenerator()


@pytest.fixture
def mock_data_factory():
    """Provide mock data factory."""
    return MockDataFactory()


@pytest.fixture
async def async_mock_db():
    """Provide async mock database session."""
    return MockDataFactory.create_mock_database_session()


@pytest.fixture
def mock_cache_manager():
    """Provide mock cache manager."""
    return MockDataFactory.create_mock_cache_manager()


@pytest.fixture
def sample_users(test_data_generator):
    """Generate sample users for testing."""
    return [test_data_generator.generate_user() for _ in range(5)]


@pytest.fixture
def sample_stocks(test_data_generator):
    """Generate sample stocks for testing."""
    stocks = []
    sectors = ["Technology", "Healthcare", "Finance"]
    for sector in sectors:
        for i in range(3):  # 3 stocks per sector
            stocks.append(test_data_generator.generate_stock(sector=sector))
    return stocks


@pytest.fixture
def sample_portfolios(test_data_generator, sample_users):
    """Generate sample portfolios for testing."""
    portfolios = []
    for user in sample_users:
        for i in range(random.randint(1, 3)):  # 1-3 portfolios per user
            portfolios.append(test_data_generator.generate_portfolio(user.id))
    return portfolios


@pytest.fixture
def historical_price_data(test_data_generator, sample_stocks):
    """Generate historical price data for testing."""
    price_data = {}
    start_date = date.today() - timedelta(days=365)
    
    for stock in sample_stocks[:5]:  # Only first 5 stocks to keep data manageable
        price_data[stock.symbol] = test_data_generator.generate_price_data(
            symbol=stock.symbol,
            start_date=start_date,
            days=252,  # One year of trading days
            base_price=random.uniform(20, 500)
        )
    
    return price_data


@pytest.fixture
def auth_headers():
    """Provide authentication headers."""
    token = create_access_token(data={"sub": "1", "username": "testuser"})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def mock_external_apis():
    """Mock all external API calls."""
    with patch('httpx.AsyncClient') as mock_client:
        mock_instance = AsyncMock()
        mock_client.return_value.__aenter__.return_value = mock_instance
        
        # Mock different API responses
        mock_responses = {
            "alpha_vantage": {
                "Global Quote": {
                    "01. symbol": "AAPL",
                    "05. price": "150.25",
                    "09. change": "2.15",
                    "10. change percent": "1.45%"
                }
            },
            "finnhub": {
                "c": 150.25,  # current price
                "d": 2.15,    # change
                "dp": 1.45,   # change percent
                "h": 152.00,  # high
                "l": 148.50,  # low
                "o": 149.00,  # open
                "pc": 148.10, # previous close
                "t": 1640995200  # timestamp
            },
            "polygon": {
                "ticker": "AAPL",
                "status": "OK",
                "results": {
                    "c": 150.25,
                    "h": 152.00,
                    "l": 148.50,
                    "o": 149.00,
                    "v": 75000000,
                    "vw": 150.12
                }
            }
        }
        
        def mock_get(*args, **kwargs):
            url = args[0] if args else kwargs.get('url', '')
            
            # Route to appropriate mock response based on URL
            if 'alphavantage' in url:
                return MockDataFactory.create_mock_api_response(mock_responses["alpha_vantage"])
            elif 'finnhub' in url:
                return MockDataFactory.create_mock_api_response(mock_responses["finnhub"])
            elif 'polygon' in url:
                return MockDataFactory.create_mock_api_response(mock_responses["polygon"])
            else:
                return MockDataFactory.create_mock_api_response({"error": "Unknown API"}, 404)
        
        mock_instance.get = AsyncMock(side_effect=mock_get)
        yield mock_instance


@pytest.fixture
def mock_websocket():
    """Provide mock WebSocket connection."""
    websocket = AsyncMock()
    websocket.accept = AsyncMock()
    websocket.close = AsyncMock()
    websocket.send_json = AsyncMock()
    websocket.send_text = AsyncMock()
    websocket.receive_json = AsyncMock()
    websocket.receive_text = AsyncMock()
    return websocket


class DatabaseTestUtils:
    """Utilities for database testing."""
    
    @staticmethod
    async def clean_database(session):
        """Clean test database."""
        # In order of foreign key dependencies
        tables = [
            "transactions",
            "portfolio_positions", 
            "portfolios",
            "recommendations",
            "alerts",
            "price_data",
            "stocks",
            "users"
        ]
        
        for table in tables:
            try:
                await session.execute(f"DELETE FROM {table}")
                await session.commit()
            except Exception:
                await session.rollback()
    
    @staticmethod
    async def seed_test_data(session, test_data_generator):
        """Seed database with test data."""
        try:
            # Create test users
            users = []
            for i in range(3):
                user = test_data_generator.generate_user(
                    username=f"testuser{i}",
                    email=f"test{i}@example.com"
                )
                # In real implementation, would use repository
                users.append(user)
            
            # Create test stocks
            stocks = []
            for i in range(10):
                stock = test_data_generator.generate_stock(
                    symbol=f"TEST{i:02d}"
                )
                stocks.append(stock)
            
            await session.commit()
            return {"users": users, "stocks": stocks}
            
        except Exception as e:
            await session.rollback()
            raise e


@pytest.fixture
def db_test_utils():
    """Provide database test utilities."""
    return DatabaseTestUtils()


class AssertionHelpers:
    """Helper functions for test assertions."""
    
    @staticmethod
    def assert_user_structure(user_data: Dict[str, Any]):
        """Assert user data has correct structure."""
        required_fields = ["id", "username", "email", "is_active"]
        for field in required_fields:
            assert field in user_data, f"Missing required field: {field}"
        
        # Should not expose sensitive data
        sensitive_fields = ["hashed_password", "password"]
        for field in sensitive_fields:
            assert field not in user_data, f"Sensitive field exposed: {field}"
    
    @staticmethod
    def assert_stock_structure(stock_data: Dict[str, Any]):
        """Assert stock data has correct structure."""
        required_fields = ["symbol", "name", "sector"]
        for field in required_fields:
            assert field in stock_data, f"Missing required field: {field}"
        
        # Validate data types
        if "market_cap" in stock_data:
            assert isinstance(stock_data["market_cap"], (int, float))
        if "price" in stock_data:
            assert isinstance(stock_data["price"], (int, float))
            assert stock_data["price"] > 0
    
    @staticmethod
    def assert_portfolio_structure(portfolio_data: Dict[str, Any]):
        """Assert portfolio data has correct structure."""
        required_fields = ["id", "name", "total_value", "positions_count"]
        for field in required_fields:
            assert field in portfolio_data, f"Missing required field: {field}"
        
        # Validate numeric fields
        numeric_fields = ["total_value", "total_gain", "cash_balance"]
        for field in numeric_fields:
            if field in portfolio_data:
                assert isinstance(portfolio_data[field], (int, float, str))
    
    @staticmethod
    def assert_recommendation_structure(recommendation_data: Dict[str, Any]):
        """Assert recommendation data has correct structure."""
        required_fields = ["symbol", "recommendation_type", "confidence_score"]
        for field in required_fields:
            assert field in recommendation_data, f"Missing required field: {field}"
        
        # Validate confidence score
        if "confidence_score" in recommendation_data:
            score = recommendation_data["confidence_score"]
            assert 0 <= score <= 1, f"Invalid confidence score: {score}"


@pytest.fixture
def assertion_helpers():
    """Provide assertion helpers."""
    return AssertionHelpers()


class PerformanceTimer:
    """Context manager for measuring performance."""
    
    def __init__(self, name: str, max_duration: float = None):
        self.name = name
        self.max_duration = max_duration
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = datetime.utcnow()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.utcnow()
        self.duration = (self.end_time - self.start_time).total_seconds()
        
        if self.max_duration and self.duration > self.max_duration:
            pytest.fail(
                f"{self.name} took {self.duration:.3f}s, "
                f"should be under {self.max_duration}s"
            )
    
    def get_duration(self) -> float:
        """Get measured duration."""
        return self.duration


@pytest.fixture
def performance_timer():
    """Provide performance timer."""
    return PerformanceTimer


class MockResponseBuilder:
    """Builder for creating mock API responses."""
    
    def __init__(self):
        self.responses = {}
    
    def add_alpha_vantage_quote(self, symbol: str, price: float, change: float):
        """Add Alpha Vantage quote response."""
        self.responses[f"alphavantage_quote_{symbol}"] = {
            "Global Quote": {
                "01. symbol": symbol,
                "05. price": str(price),
                "09. change": str(change),
                "10. change percent": f"{(change/price)*100:.2f}%"
            }
        }
        return self
    
    def add_finnhub_quote(self, symbol: str, price: float, change: float):
        """Add Finnhub quote response."""
        self.responses[f"finnhub_quote_{symbol}"] = {
            "c": price,
            "d": change,
            "dp": (change/price)*100,
            "h": price * 1.02,
            "l": price * 0.98,
            "o": price * 0.99,
            "pc": price - change
        }
        return self
    
    def get_response(self, key: str):
        """Get mock response by key."""
        return self.responses.get(key)
    
    def build_mock_client(self):
        """Build mock HTTP client with configured responses."""
        mock_client = AsyncMock()
        
        async def mock_get(url, **kwargs):
            # Route to appropriate response based on URL
            if 'alphavantage' in url and 'function=GLOBAL_QUOTE' in url:
                symbol = url.split('symbol=')[1].split('&')[0] if 'symbol=' in url else 'AAPL'
                key = f"alphavantage_quote_{symbol}"
            elif 'finnhub' in url and '/quote' in url:
                symbol = url.split('/quote?symbol=')[1].split('&')[0] if '/quote?symbol=' in url else 'AAPL'
                key = f"finnhub_quote_{symbol}"
            else:
                key = "default"
            
            response_data = self.get_response(key) or {"error": "No mock data configured"}
            return MockDataFactory.create_mock_api_response(response_data)
        
        mock_client.get = mock_get
        return mock_client


@pytest.fixture
def mock_response_builder():
    """Provide mock response builder."""
    return MockResponseBuilder()


# Global test configuration and utilities
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
async def cleanup_after_test():
    """Cleanup after each test."""
    yield
    # Cleanup code here (reset global state, clear caches, etc.)
    pass