"""
Async Testing Fixtures
Comprehensive async testing utilities and fixtures for database testing with proper isolation and cleanup.
"""

import asyncio
import pytest
import pytest_asyncio
from typing import AsyncGenerator, Generator, Dict, Any, List
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import text
from datetime import datetime, date
import tempfile
import os

from backend.config.database import AsyncDatabaseManager, DatabaseConfig, TransactionIsolationLevel
from backend.models.unified_models import Base, Stock, PriceHistory, User, Portfolio, Recommendation
from backend.repositories import stock_repository, user_repository, portfolio_repository
from backend.utils.async_locks import resource_lock_manager
from backend.utils.deadlock_handler import deadlock_handler


class AsyncTestDatabaseManager:
    """
    Test database manager that creates isolated test databases for each test.
    """
    
    def __init__(self, test_db_url: str = None):
        """Initialize test database manager"""
        if test_db_url is None:
            # Create in-memory SQLite database for testing
            test_db_url = "postgresql+asyncpg://postgres:password@localhost/test_investment_db"
        
        self.test_db_url = test_db_url
        self.engine = None
        self.sessionmaker = None
        self._test_data_cache: Dict[str, Any] = {}
    
    async def setup_test_database(self) -> AsyncDatabaseManager:
        """Set up test database with clean schema"""
        # Create async engine for testing
        self.engine = create_async_engine(
            self.test_db_url,
            echo=False,  # Disable SQL logging in tests unless needed
            pool_pre_ping=True,
            future=True
        )
        
        # Create all tables
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
        
        # Create session maker
        self.sessionmaker = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Create database manager with test configuration
        config = DatabaseConfig(
            url=self.test_db_url,
            pool_size=5,
            max_overflow=10,
            echo=False
        )
        
        db_manager = AsyncDatabaseManager(config)
        await db_manager.initialize()
        
        return db_manager
    
    async def cleanup_test_database(self):
        """Clean up test database"""
        if self.engine:
            # Drop all tables
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            
            # Dispose of engine
            await self.engine.dispose()
        
        # Clear cache
        self._test_data_cache.clear()
    
    async def get_test_session(self) -> AsyncSession:
        """Get test database session"""
        if not self.sessionmaker:
            raise RuntimeError("Test database not initialized")
        
        return self.sessionmaker()
    
    async def create_test_data(self) -> Dict[str, Any]:
        """Create comprehensive test data"""
        if 'test_data_created' in self._test_data_cache:
            return self._test_data_cache
        
        async with self.get_test_session() as session:
            # Create test stocks
            test_stocks = [
                Stock(
                    symbol="AAPL",
                    name="Apple Inc.",
                    exchange="NASDAQ",
                    sector="Technology",
                    industry="Consumer Electronics",
                    market_cap=3000000000000,
                    is_active=True,
                    is_tradable=True,
                    country="US",
                    currency="USD"
                ),
                Stock(
                    symbol="GOOGL",
                    name="Alphabet Inc.",
                    exchange="NASDAQ", 
                    sector="Technology",
                    industry="Internet Content & Information",
                    market_cap=1800000000000,
                    is_active=True,
                    is_tradable=True,
                    country="US",
                    currency="USD"
                ),
                Stock(
                    symbol="TSLA",
                    name="Tesla Inc.",
                    exchange="NASDAQ",
                    sector="Consumer Cyclical",
                    industry="Auto Manufacturers",
                    market_cap=800000000000,
                    is_active=True,
                    is_tradable=True,
                    country="US",
                    currency="USD"
                )
            ]
            
            for stock in test_stocks:
                session.add(stock)
            
            await session.flush()
            
            # Create test price history
            test_prices = [
                PriceHistory(
                    stock_id=test_stocks[0].id,
                    date=date(2024, 1, 1),
                    open=190.0,
                    high=195.0,
                    low=188.0,
                    close=193.5,
                    volume=50000000
                ),
                PriceHistory(
                    stock_id=test_stocks[0].id,
                    date=date(2024, 1, 2),
                    open=193.5,
                    high=197.0,
                    low=192.0,
                    close=196.2,
                    volume=45000000
                )
            ]
            
            for price in test_prices:
                session.add(price)
            
            # Create test user
            test_user = User(
                email="test@example.com",
                hashed_password="hashed_password_123",
                full_name="Test User",
                is_active=True,
                is_verified=True
            )
            
            session.add(test_user)
            await session.flush()
            
            # Create test portfolio
            test_portfolio = Portfolio(
                user_id=test_user.id,
                name="Test Portfolio",
                description="Test portfolio for unit tests",
                cash_balance=10000.0,
                is_default=True
            )
            
            session.add(test_portfolio)
            
            await session.commit()
            
            # Cache test data
            self._test_data_cache = {
                'test_data_created': True,
                'stocks': test_stocks,
                'prices': test_prices,
                'user': test_user,
                'portfolio': test_portfolio
            }
        
        return self._test_data_cache


# Global test database manager
test_db_manager = AsyncTestDatabaseManager()


# Pytest fixtures
@pytest_asyncio.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for entire test session"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def test_db() -> AsyncGenerator[AsyncTestDatabaseManager, None]:
    """Session-scoped test database fixture"""
    await test_db_manager.setup_test_database()
    yield test_db_manager
    await test_db_manager.cleanup_test_database()


@pytest_asyncio.fixture
async def async_session(test_db: AsyncTestDatabaseManager) -> AsyncGenerator[AsyncSession, None]:
    """Function-scoped async database session with transaction rollback"""
    async with test_db.get_test_session() as session:
        # Start transaction
        trans = await session.begin()
        
        try:
            yield session
        finally:
            # Always rollback to ensure test isolation
            await trans.rollback()


@pytest_asyncio.fixture
async def test_data(test_db: AsyncTestDatabaseManager) -> Dict[str, Any]:
    """Load comprehensive test data"""
    return await test_db.create_test_data()


@pytest_asyncio.fixture
async def test_stock(async_session: AsyncSession) -> Stock:
    """Create a single test stock"""
    stock = Stock(
        symbol="TEST",
        name="Test Corporation",
        exchange="NASDAQ",
        sector="Technology",
        industry="Software",
        market_cap=1000000000,
        is_active=True,
        is_tradable=True,
        country="US",
        currency="USD"
    )
    
    async_session.add(stock)
    await async_session.flush()
    await async_session.refresh(stock)
    
    return stock


@pytest_asyncio.fixture
async def test_user(async_session: AsyncSession) -> User:
    """Create a test user"""
    user = User(
        email="testuser@example.com",
        hashed_password="hashed_password",
        full_name="Test User",
        is_active=True,
        is_verified=True
    )
    
    async_session.add(user)
    await async_session.flush()
    await async_session.refresh(user)
    
    return user


@pytest_asyncio.fixture
async def test_portfolio(async_session: AsyncSession, test_user: User) -> Portfolio:
    """Create a test portfolio"""
    portfolio = Portfolio(
        user_id=test_user.id,
        name="Test Portfolio",
        description="Portfolio for testing",
        cash_balance=10000.0,
        is_default=True
    )
    
    async_session.add(portfolio)
    await async_session.flush()
    await async_session.refresh(portfolio)
    
    return portfolio


@pytest_asyncio.fixture
async def multiple_stocks(async_session: AsyncSession) -> List[Stock]:
    """Create multiple test stocks"""
    stocks = [
        Stock(
            symbol=f"TEST{i}",
            name=f"Test Corp {i}",
            exchange="NASDAQ",
            sector="Technology" if i % 2 == 0 else "Healthcare",
            industry=f"Industry {i}",
            market_cap=1000000000 * (i + 1),
            is_active=True,
            is_tradable=True,
            country="US",
            currency="USD"
        )
        for i in range(5)
    ]
    
    for stock in stocks:
        async_session.add(stock)
    
    await async_session.flush()
    
    for stock in stocks:
        await async_session.refresh(stock)
    
    return stocks


@pytest_asyncio.fixture
async def stock_with_price_history(
    async_session: AsyncSession,
    test_stock: Stock
) -> tuple[Stock, List[PriceHistory]]:
    """Create stock with price history"""
    prices = []
    base_price = 100.0
    
    for i in range(10):
        price = PriceHistory(
            stock_id=test_stock.id,
            date=date(2024, 1, i + 1),
            open=base_price + i,
            high=base_price + i + 2,
            low=base_price + i - 1,
            close=base_price + i + 1,
            volume=1000000 + (i * 100000)
        )
        prices.append(price)
        async_session.add(price)
    
    await async_session.flush()
    
    for price in prices:
        await async_session.refresh(price)
    
    return test_stock, prices


# Test utilities
class AsyncTestUtils:
    """Utilities for async testing"""
    
    @staticmethod
    async def wait_for_condition(
        condition_func,
        timeout: float = 5.0,
        check_interval: float = 0.1
    ) -> bool:
        """Wait for a condition to become true"""
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            if await condition_func():
                return True
            await asyncio.sleep(check_interval)
        
        return False
    
    @staticmethod
    async def assert_eventually(
        condition_func,
        timeout: float = 5.0,
        message: str = "Condition was not met within timeout"
    ):
        """Assert that condition becomes true within timeout"""
        success = await AsyncTestUtils.wait_for_condition(condition_func, timeout)
        assert success, message
    
    @staticmethod
    async def run_concurrent_operations(
        operations: List,
        max_concurrency: int = 10
    ) -> List[Any]:
        """Run multiple async operations concurrently with limited concurrency"""
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def run_with_semaphore(operation):
            async with semaphore:
                return await operation
        
        return await asyncio.gather(*[
            run_with_semaphore(op) for op in operations
        ])
    
    @staticmethod
    async def simulate_database_load(
        session: AsyncSession,
        operations_count: int = 100,
        concurrent_tasks: int = 10
    ):
        """Simulate database load for testing performance and concurrency"""
        async def db_operation():
            # Simple database operation
            result = await session.execute(text("SELECT 1"))
            return result.scalar()
        
        operations = [db_operation() for _ in range(operations_count)]
        
        return await AsyncTestUtils.run_concurrent_operations(
            operations, max_concurrency=concurrent_tasks
        )


# Mock fixtures for external dependencies
@pytest_asyncio.fixture
async def mock_external_api():
    """Mock external API responses"""
    class MockAPI:
        def __init__(self):
            self.call_count = 0
        
        async def get_stock_price(self, symbol: str) -> Dict[str, Any]:
            self.call_count += 1
            return {
                'symbol': symbol,
                'price': 100.0 + (self.call_count % 50),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    return MockAPI()


@pytest_asyncio.fixture
async def concurrent_test_setup():
    """Setup for concurrent testing"""
    # Reset lock manager state
    await resource_lock_manager._cleanup_expired_locks()
    
    # Reset deadlock handler metrics
    deadlock_handler.global_metrics = deadlock_handler.__class__().global_metrics
    deadlock_handler.circuit_breakers.clear()
    
    yield
    
    # Cleanup after concurrent tests
    await resource_lock_manager._cleanup_expired_locks()


# Performance testing fixtures
@pytest_asyncio.fixture
async def performance_test_data(
    async_session: AsyncSession
) -> Dict[str, Any]:
    """Create large dataset for performance testing"""
    # Create many stocks for performance testing
    stocks = []
    
    for i in range(1000):
        stock = Stock(
            symbol=f"PERF{i:04d}",
            name=f"Performance Test Stock {i}",
            exchange="NASDAQ" if i % 2 == 0 else "NYSE",
            sector=f"Sector {i % 10}",
            industry=f"Industry {i % 20}",
            market_cap=1000000 * (i + 1),
            is_active=True,
            is_tradable=True,
            country="US",
            currency="USD"
        )
        stocks.append(stock)
        async_session.add(stock)
        
        # Flush periodically to avoid memory issues
        if i % 100 == 0:
            await async_session.flush()
    
    await async_session.flush()
    
    return {'stocks': stocks, 'count': len(stocks)}


# Custom assertion helpers
def assert_stock_equal(actual: Stock, expected: Stock, ignore_fields: List[str] = None):
    """Assert that two stocks are equal, ignoring specified fields"""
    ignore_fields = ignore_fields or ['id', 'created_at', 'updated_at']
    
    for field in Stock.__table__.columns.keys():
        if field not in ignore_fields:
            actual_value = getattr(actual, field)
            expected_value = getattr(expected, field)
            assert actual_value == expected_value, f"Field '{field}' mismatch: {actual_value} != {expected_value}"


async def assert_database_state(
    session: AsyncSession,
    table_counts: Dict[str, int]
):
    """Assert expected counts in database tables"""
    for table_name, expected_count in table_counts.items():
        if table_name == 'stocks':
            result = await session.execute(text("SELECT COUNT(*) FROM stocks"))
        elif table_name == 'price_history':
            result = await session.execute(text("SELECT COUNT(*) FROM price_history"))
        elif table_name == 'users':
            result = await session.execute(text("SELECT COUNT(*) FROM users"))
        else:
            result = await session.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
        
        actual_count = result.scalar()
        assert actual_count == expected_count, f"Table {table_name}: expected {expected_count}, got {actual_count}"