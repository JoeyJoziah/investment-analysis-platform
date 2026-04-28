"""
Pytest fixtures for security tests
"""

import pytest
import pytest_asyncio
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import StaticPool

from backend.models.unified_models import Base


# NOTE: Removed deprecated session-scope `event_loop` fixture (F-15-013, audit 2026-04).
# `asyncio_mode = auto` in pytest.ini handles loop creation; redefining
# event_loop here conflicts with newer pytest-asyncio and emits DeprecationWarnings.


@pytest_asyncio.fixture(scope="function")
async def db_engine():
    """Create async test database engine (in-memory SQLite)"""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        echo=False
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def db_session(db_engine):
    """Create async test database session"""
    async_session_maker = async_sessionmaker(
        db_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )

    async with async_session_maker() as session:
        yield session
        await session.rollback()
