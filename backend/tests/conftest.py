"""
Global Test Configuration for Investment Analysis Platform Integration Tests
Provides shared fixtures, test utilities, and configuration for all test modules.
"""

# CRITICAL: Set TESTING=True BEFORE any imports to disable middleware that blocks tests
import os
os.environ["TESTING"] = "True"
os.environ["DEBUG"] = "True"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"

import pytest
import pytest_asyncio
import asyncio
import os
from datetime import datetime
from typing import AsyncGenerator
from unittest.mock import patch, AsyncMock
import logging
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from httpx import AsyncClient, ASGITransport

from backend.api.main import app
from backend.config.database import get_async_db_session, initialize_database
from backend.auth.oauth2 import get_current_user, create_access_token
from backend.models.unified_models import User
from backend.utils.comprehensive_cache import get_cache_manager
from backend.config.settings import settings


# Configure test logging
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


# ============================================================================
# ApiResponse Wrapper Validation Helpers
# ============================================================================

def assert_success_response(response, expected_status=200):
    """
    Validate ApiResponse wrapper structure and return unwrapped data.

    Args:
        response: FastAPI TestClient response object
        expected_status: Expected HTTP status code (default: 200)

    Returns:
        Unwrapped data from response["data"]

    Example:
        data = assert_success_response(response)
        assert data["title"] == "My Title"
    """
    assert response.status_code == expected_status, \
        f"Expected status {expected_status}, got {response.status_code}"

    json_data = response.json()
    assert json_data["success"] == True, \
        f"Expected success=True, got success={json_data.get('success')}"
    assert "data" in json_data, \
        "Response missing 'data' field in ApiResponse wrapper"

    return json_data["data"]


def assert_api_error_response(response, expected_status, expected_error_substring=None):
    """
    Validate ApiResponse error structure.

    Args:
        response: FastAPI TestClient response object
        expected_status: Expected HTTP error status code
        expected_error_substring: Optional substring to verify in error message

    Returns:
        Full response JSON data

    Example:
        data = assert_api_error_response(response, 404, "not found")
        assert "error" in data
    """
    assert response.status_code == expected_status, \
        f"Expected status {expected_status}, got {response.status_code}"

    json_data = response.json()
    assert json_data["success"] == False, \
        f"Expected success=False for error response, got success={json_data.get('success')}"

    if expected_error_substring:
        error_msg = json_data.get("error", "")
        assert expected_error_substring.lower() in error_msg.lower(), \
            f"Expected '{expected_error_substring}' in error message, got: {error_msg}"

    return json_data


@pytest_asyncio.fixture(scope="function")
async def test_db_engine():
    """Create test database engine."""
    from backend.models.unified_models import Base

    # Use in-memory SQLite for tests (safe default)
    test_db_url = "sqlite+aiosqlite:///:memory:"

    engine = create_async_engine(
        test_db_url,
        echo=False,  # Set to True for SQL debugging
        pool_pre_ping=True
    )

    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine
    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def test_db_session_factory(test_db_engine):
    """Create test database session factory."""
    TestSessionLocal = sessionmaker(
        test_db_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    return TestSessionLocal


@pytest_asyncio.fixture
async def db_session(test_db_session_factory):
    """Provide database session for tests."""
    async with test_db_session_factory() as session:
        try:
            yield session
        finally:
            await session.rollback()  # Rollback any changes
            await session.close()


@pytest_asyncio.fixture
async def async_client():
    """Provide async HTTP client for API testing."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as client:
        yield client


@pytest_asyncio.fixture
async def client(async_client):
    """Alias for async_client for backward compatibility."""
    return async_client


@pytest.fixture
def test_user():
    """Provide test user."""
    return User(
        id=1,
        username="testuser",
        email="test@example.com",
        hashed_password="$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",
        is_active=True,
        is_verified=True,
        created_at=datetime.utcnow()
    )


@pytest_asyncio.fixture
async def nasdaq_exchange(db_session: AsyncSession):
    """Provide NASDAQ exchange for testing."""
    from backend.models.unified_models import Exchange

    exchange = Exchange(
        code="NASDAQ",
        name="NASDAQ Stock Market",
        timezone="America/New_York",
        country="US",
        currency="USD"
    )
    db_session.add(exchange)
    await db_session.commit()
    await db_session.refresh(exchange)
    return exchange


@pytest_asyncio.fixture
async def technology_sector(db_session: AsyncSession):
    """Provide Technology sector for testing."""
    from backend.models.unified_models import Sector

    sector = Sector(
        name="Technology",
        description="Technology sector"
    )
    db_session.add(sector)
    await db_session.commit()
    await db_session.refresh(sector)
    return sector


@pytest.fixture
def auth_token(test_user):
    """Provide authentication token."""
    return create_access_token(
        data={"sub": str(test_user.id), "username": test_user.username}
    )


@pytest.fixture
def auth_headers(auth_token):
    """Provide authentication headers."""
    return {"Authorization": f"Bearer {auth_token}"}


@pytest.fixture
def mock_current_user(test_user):
    """Mock current user dependency."""
    with patch.object(app, 'dependency_overrides') as mock_overrides:
        mock_overrides[get_current_user] = lambda: test_user
        yield test_user
        mock_overrides.clear()


@pytest.fixture
async def mock_cache():
    """Provide mock cache manager."""
    cache = AsyncMock()
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock(return_value=True)
    cache.delete = AsyncMock(return_value=True)
    cache.clear = AsyncMock(return_value=True)
    cache.get_stats = AsyncMock(return_value={
        "hits": 100,
        "misses": 20,
        "hit_rate": 0.83,
        "memory_usage": 1024 * 1024
    })
    
    with patch('backend.utils.comprehensive_cache.get_cache_manager', return_value=cache):
        yield cache


@pytest.fixture
def mock_external_apis():
    """Mock all external API calls."""
    alpha_vantage_response = {
        "Global Quote": {
            "01. symbol": "AAPL",
            "02. open": "149.00",
            "03. high": "152.00",
            "04. low": "148.50",
            "05. price": "150.25",
            "06. volume": "75000000",
            "07. latest trading day": "2024-01-15",
            "08. previous close": "148.10",
            "09. change": "2.15",
            "10. change percent": "1.45%"
        }
    }
    
    finnhub_response = {
        "c": 150.25,
        "d": 2.15,
        "dp": 1.45,
        "h": 152.00,
        "l": 148.50,
        "o": 149.00,
        "pc": 148.10,
        "t": 1640995200
    }
    
    polygon_response = {
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
    
    def mock_get(*args, **kwargs):
        url = args[0] if args else kwargs.get('url', '')
        
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        
        if 'alphavantage' in url:
            mock_response.json.return_value = alpha_vantage_response
        elif 'finnhub' in url:
            mock_response.json.return_value = finnhub_response
        elif 'polygon' in url:
            mock_response.json.return_value = polygon_response
        else:
            mock_response.json.return_value = {"error": "Unknown API"}
            mock_response.status_code = 404
        
        return mock_response
    
    with patch('httpx.AsyncClient.get', side_effect=mock_get):
        with patch('httpx.get', side_effect=mock_get):
            yield {
                "alpha_vantage": alpha_vantage_response,
                "finnhub": finnhub_response,
                "polygon": polygon_response
            }


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Setup test environment variables."""
    test_settings = {
        "DEBUG": "True",
        "TESTING": "True",
        "DATABASE_URL": "sqlite+aiosqlite:///:memory:",
        "REDIS_URL": "redis://localhost:6379/1",
        "SECRET_KEY": "test-secret-key-for-testing-only",
        "ALPHA_VANTAGE_API_KEY": "test_alpha_vantage_key",
        "FINNHUB_API_KEY": "test_finnhub_key",
        "POLYGON_API_KEY": "test_polygon_key",
        "NEWS_API_KEY": "test_news_key"
    }
    
    for key, value in test_settings.items():
        monkeypatch.setenv(key, value)


@pytest.fixture
def mock_ml_models():
    """Mock ML model responses."""
    mock_predictions = {
        "price_prediction": {
            "predicted_price": 155.50,
            "confidence": 0.85,
            "direction": "bullish",
            "volatility": 0.22
        },
        "recommendation": {
            "action": "buy",
            "confidence": 0.78,
            "target_price": 160.00,
            "stop_loss": 145.00
        },
        "technical_analysis": {
            "trend": "upward",
            "strength": 0.72,
            "support_levels": [148.00, 145.50, 142.00],
            "resistance_levels": [155.00, 158.50, 162.00]
        }
    }
    
    with patch('backend.ml.model_manager.get_model_manager') as mock_manager:
        mock_instance = AsyncMock()
        mock_instance.predict.return_value = mock_predictions["price_prediction"]
        mock_instance.analyze.return_value = mock_predictions
        mock_manager.return_value = mock_instance
        yield mock_predictions


@pytest.fixture
def performance_threshold():
    """Define performance thresholds for tests."""
    return {
        "api_response": 2.0,  # seconds
        "database_query": 1.0,  # seconds
        "cache_operation": 0.1,  # seconds
        "websocket_message": 0.5,  # seconds
        "bulk_operation": 10.0,  # seconds
    }


@pytest.fixture
async def cleanup_test_data():
    """Cleanup test data after tests."""
    yield
    # Cleanup operations would go here
    # In production, this would clean up test database, cache, etc.
    pass


class TestMetrics:
    """Track test execution metrics."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.test_counts = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0
        }
    
    def start_test_session(self):
        self.start_time = datetime.utcnow()
    
    def end_test_session(self):
        self.end_time = datetime.utcnow()
    
    def record_test_result(self, outcome: str):
        self.test_counts["total"] += 1
        if outcome in self.test_counts:
            self.test_counts[outcome] += 1
    
    def get_duration(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    def get_summary(self) -> dict:
        return {
            "duration": self.get_duration(),
            "test_counts": self.test_counts.copy(),
            "pass_rate": self.test_counts["passed"] / max(self.test_counts["total"], 1)
        }


@pytest.fixture(scope="session")
def test_metrics():
    """Provide test metrics tracking."""
    metrics = TestMetrics()
    yield metrics


# Pytest hooks for custom behavior
def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "api: marks tests as API tests"
    )
    config.addinivalue_line(
        "markers", "database: marks tests as database tests"
    )
    config.addinivalue_line(
        "markers", "security: marks tests as security tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add integration marker to integration test files
        if "test_integration" in item.fspath.basename:
            item.add_marker(pytest.mark.integration)
        
        # Add slow marker to tests with "slow" in name
        if "slow" in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Add performance marker to performance tests
        if "performance" in item.name or "load" in item.name:
            item.add_marker(pytest.mark.performance)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """Setup before each test."""
    # Skip tests based on environment
    if item.get_closest_marker("external_api"):
        if os.getenv("SKIP_EXTERNAL_API_TESTS", "false").lower() == "true":
            pytest.skip("External API tests disabled")


def pytest_runtest_teardown(item, nextitem):
    """Cleanup after each test."""
    # Clear any global state
    if hasattr(app, 'dependency_overrides'):
        app.dependency_overrides.clear()


# Custom assertions for better error messages
def assert_response_structure(response_data: dict, expected_fields: list):
    """Assert response has expected structure."""
    missing_fields = [field for field in expected_fields if field not in response_data]
    if missing_fields:
        pytest.fail(f"Response missing required fields: {missing_fields}")


def assert_performance_threshold(duration: float, threshold: float, operation: str):
    """Assert operation completed within performance threshold."""
    if duration > threshold:
        pytest.fail(
            f"{operation} took {duration:.3f}s, "
            f"exceeds threshold of {threshold}s"
        )


def assert_error_response(response, expected_status: int, expected_error_type: str = None):
    """Assert error response has correct structure."""
    assert response.status_code == expected_status
    
    if expected_status >= 400:
        error_data = response.json()
        assert "error" in error_data or "detail" in error_data
        
        if expected_error_type:
            error_message = error_data.get("error", error_data.get("detail", ""))
            assert expected_error_type.lower() in str(error_message).lower()


# Make custom assertions available globally
pytest.assert_response_structure = assert_response_structure
pytest.assert_performance_threshold = assert_performance_threshold
pytest.assert_error_response = assert_error_response