"""Simple async test to verify pytest-asyncio configuration"""
import pytest
import pytest_asyncio
import asyncio


@pytest_asyncio.fixture
async def async_value():
    """Simple async fixture"""
    await asyncio.sleep(0.001)
    return 42


@pytest.mark.asyncio
async def test_simple_async(async_value):
    """Simple async test"""
    assert async_value == 42
    await asyncio.sleep(0.001)
    assert True
