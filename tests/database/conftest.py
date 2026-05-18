"""
Pytest fixtures for database transaction tests.

These fixtures create an in-memory SQLite engine + session factory and patch
``backend.repositories.base.get_db_session`` to yield from that test session.
This lets us assert real commit/rollback semantics on
``AsyncBaseRepository.transaction()`` without standing up Postgres.

Audit reference: 2026-04 G4 Phase 1 (F-07-002 fail-first commit-pair).
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

# Force testing posture before any backend imports.
# Must set ALL Settings()-required vars before importing backend.config.settings.
os.environ.setdefault("TESTING", "True")
os.environ.setdefault("DEBUG", "True")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-testing-only")
os.environ.setdefault("JWT_SECRET_KEY", "test-jwt-secret-key-for-testing-only")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/1")

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from backend.models.unified_models import Base, Exchange, Stock


@pytest_asyncio.fixture(scope="function")
async def tx_engine():
    """In-memory SQLite engine shared by a single test (StaticPool keeps it alive)."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
        echo=False,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def tx_session_factory(tx_engine):
    return async_sessionmaker(
        tx_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


@pytest_asyncio.fixture(scope="function")
async def patched_get_db_session(monkeypatch, tx_session_factory):
    """Patch ``backend.repositories.base.get_db_session`` to use the test engine.

    The real implementation goes through ``db_manager``, which requires Postgres
    credentials. For unit-level transaction semantics tests we substitute an
    asynccontextmanager that yields a session from the test factory.
    """

    @asynccontextmanager
    async def _fake_get_db_session(
        isolation_level=None, readonly: bool = False
    ) -> AsyncGenerator[AsyncSession, None]:
        # Mirror db_manager.get_session() semantics: commit on clean exit,
        # rollback on exception. This is what production
        # ``backend.config.database.get_db_session`` provides via
        # ``db_manager.get_session``.
        session = tx_session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    import backend.repositories.base as repo_base

    monkeypatch.setattr(repo_base, "get_db_session", _fake_get_db_session)
    yield _fake_get_db_session


@pytest_asyncio.fixture
async def seeded_exchange(tx_session_factory):
    """Seed a NASDAQ exchange so Stock rows can satisfy the FK."""
    async with tx_session_factory() as session:
        ex = Exchange(
            code="NASDAQ",
            name="NASDAQ Stock Market",
            timezone="America/New_York",
            country="US",
            currency="USD",
        )
        session.add(ex)
        await session.commit()
        await session.refresh(ex)
        return ex.id


@pytest.fixture
def sample_stock_factory(seeded_exchange):
    """Factory for valid Stock instances (FK to seeded NASDAQ exchange)."""

    def _make(symbol: str, **overrides) -> Stock:
        defaults = dict(
            symbol=symbol.upper(),
            name=overrides.pop("name", f"Test {symbol}"),
            exchange_id=seeded_exchange,
            asset_type="stock",
            country="US",
            currency="USD",
            is_active=True,
            is_tradable=True,
            is_delisted=False,
        )
        defaults.update(overrides)
        return Stock(**defaults)

    return _make


@pytest.fixture
def stock_repo(patched_get_db_session):
    """StockRepository with get_db_session patched onto the test engine."""
    from backend.repositories.stock_repository import StockRepository

    return StockRepository()
