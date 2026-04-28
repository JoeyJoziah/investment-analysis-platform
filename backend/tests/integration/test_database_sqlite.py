"""
Docker-free SQLite integration tests for repository CRUD + transaction paths.

Audit 2026-04, Cluster E Step 6 (workpaper:
docs/audits/2026-04/_synthesis/workpaper/E.md, F-15-020).

Why this file exists
--------------------
backend/tests/test_database_integration.py runs against a real Postgres via
testcontainers/Docker. In environments where Docker isn't available
(developer laptops, restricted CI), the entire suite was silently skipped
(F-15-008), giving the DB layer effectively zero integration coverage.

This module mirrors the most critical CRUD + transaction tests against an
in-memory SQLite database. It runs in any environment with no external
dependencies and serves as a Docker-free smoke test for the repository
layer's contract surface.

Scope
-----
The tests cover:
  1. User CRUD round-trip (insert + select).
  2. User email unique constraint.
  3. User required-field constraint (full_name NOT NULL).
  4. session.begin() commit persistence.
  5. session.begin() rollback on exception.

Tests that depend on Postgres-only features (LISTEN/NOTIFY, advisory locks,
JSONB operators, pg_stat_statements) and on the multi-table FK chain
(Stock -> Exchange/Sector/Industry) intentionally remain in the
testcontainers suite. This file is the always-on baseline, not a
replacement.
"""

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

# Skip cleanly if the optional aiosqlite driver isn't installed in the env.
pytest.importorskip("aiosqlite", reason="aiosqlite not installed")

from backend.models.unified_models import Base, User  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures: in-memory SQLite with the unified models' schema applied.
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def engine():
    """In-memory aiosqlite engine; StaticPool keeps a single shared connection."""
    eng = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
        echo=False,
    )
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        yield eng
    finally:
        await eng.dispose()


def _user_kwargs(**overrides):
    """Minimal valid User kwargs; tests can override any field."""
    base = dict(
        email="loki-e6@example.com",
        username="loki-e6",
        hashed_password="x" * 32,
        full_name="Loki E6 User",
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_user_crud_round_trip(engine):
    """Insert a User, then select it back; verify field round-trip."""
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as session:
        session.add(User(**_user_kwargs()))
        await session.commit()

    async with maker() as session:
        result = await session.execute(
            select(User).where(User.email == "loki-e6@example.com")
        )
        fetched = result.scalar_one()
        assert fetched.id is not None
        assert fetched.username == "loki-e6"
        assert fetched.full_name == "Loki E6 User"
        assert fetched.is_active is True


@pytest.mark.asyncio
async def test_user_email_unique_constraint(engine):
    """Two Users with the same email should raise IntegrityError on flush."""
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as session:
        session.add(User(**_user_kwargs(email="dupe@e6.test", username="dupe-1")))
        await session.commit()

    async with maker() as session:
        session.add(User(**_user_kwargs(email="dupe@e6.test", username="dupe-2")))
        with pytest.raises(IntegrityError):
            await session.commit()


@pytest.mark.asyncio
async def test_user_full_name_required(engine):
    """full_name is NOT NULL; missing it must raise IntegrityError."""
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as session:
        # Build kwargs without full_name -- SQLAlchemy will INSERT NULL.
        kwargs = _user_kwargs(email="nofullname@e6.test", username="nofn")
        kwargs.pop("full_name")
        kwargs["full_name"] = None
        session.add(User(**kwargs))
        with pytest.raises(IntegrityError):
            await session.commit()


@pytest.mark.asyncio
async def test_explicit_transaction_commits(engine):
    """Explicit BEGIN ... COMMIT block via session.begin() persists the row."""
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as session:
        async with session.begin():
            session.add(
                User(**_user_kwargs(email="commit@e6.test", username="commit-e6"))
            )

    async with maker() as session2:
        result = await session2.execute(
            select(User).where(User.email == "commit@e6.test")
        )
        assert result.scalar_one_or_none() is not None


@pytest.mark.asyncio
async def test_explicit_transaction_rolls_back_on_exception(engine):
    """When the body of session.begin() raises, the row must NOT persist."""
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    class _Boom(RuntimeError):
        pass

    with pytest.raises(_Boom):
        async with maker() as session:
            async with session.begin():
                session.add(
                    User(
                        **_user_kwargs(
                            email="rollback@e6.test", username="rollback-e6"
                        )
                    )
                )
                raise _Boom("simulated failure inside transaction")

    async with maker() as session2:
        result = await session2.execute(
            select(User).where(User.email == "rollback@e6.test")
        )
        assert result.scalar_one_or_none() is None
