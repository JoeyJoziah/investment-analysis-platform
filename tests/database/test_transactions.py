"""
F-07-002 — Fail-first proof that ``AsyncBaseRepository.transaction()`` is a no-op.

Audit reference: 2026-04 PRD §3 G4 Phase 1, workpaper
``docs/audits/2026-04/_synthesis/workpaper/G4_storage_security_residual.md``.

Bug summary
-----------
``backend/repositories/base.py:652-672`` decorates ``transaction`` with
``@asynccontextmanager`` but the body has no ``yield``. It defines an inner
async-generator function ``_execute_transaction`` and passes it to
``db_manager.execute_with_retry``. Net effect: ``async with
repo.transaction() as session: ...`` either errors with "generator didn't
yield" or silently produces no session and no commit/rollback.

These tests must FAIL on the unfixed code. They are intentionally written
WITHOUT ``xfail`` so CI records a red signal during the fail-first commit.
After the @asynccontextmanager rewrite they must turn GREEN.

Cascade contract
----------------
Workstream E (PR #146, commit 799abba) added two ``xfail(strict=True)``
tests in ``backend/tests/unit/test_repositories_unit.py`` covering this
exact bug. When the fix lands those markers MUST be removed in the same
commit, otherwise CI will flip them XPASS-strict and turn red.
"""
from __future__ import annotations

import pytest
from sqlalchemy import select, text


# ---------------------------------------------------------------------------
# Test 1 — RED: prove ``transaction()`` does not yield a real session today.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
@pytest.mark.database
async def test_transaction_yields_real_session(stock_repo):
    """``async with repo.transaction() as session`` must give a usable session.

    On unfixed code this raises ``RuntimeError: generator didn't yield`` or
    yields ``None`` because the outer @asynccontextmanager body has no
    ``yield``. After the fix this must execute a real ``SELECT 1``.
    """
    async with stock_repo.transaction() as session:
        assert session is not None, (
            "transaction() yielded None — async-generator bug (F-07-002). "
            "Outer @asynccontextmanager body never yields a session."
        )
        result = await session.execute(text("SELECT 1"))
        assert result.scalar() == 1


# ---------------------------------------------------------------------------
# Test 2 — RED: prove rollback semantics never run today.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
@pytest.mark.database
async def test_transaction_rolls_back_on_exception(
    stock_repo, sample_stock_factory, tx_session_factory
):
    """A raised exception inside the block must roll back the session.

    On unfixed code the body never executes (no yield), so neither the
    add() nor the rollback runs — and the test cannot even reach the
    inner ``raise``. After the fix, the row must be absent from the DB
    when the block re-raises.
    """

    class _Boom(RuntimeError):
        pass

    with pytest.raises(_Boom):
        async with stock_repo.transaction() as session:
            session.add(sample_stock_factory(symbol="ROLLBK"))
            await session.flush()
            raise _Boom("simulated failure inside transaction")

    # Verify with a fresh session that ROLLBK was rolled back.
    from backend.models.unified_models import Stock

    async with tx_session_factory() as verify_session:
        result = await verify_session.execute(
            select(Stock).where(Stock.symbol == "ROLLBK")
        )
        found = result.scalar_one_or_none()

    assert found is None, (
        "rollback failed — row 'ROLLBK' persisted after exception. "
        "Either transaction() did not roll back, or get_db_session "
        "auto-committed before the exception propagated."
    )


# ---------------------------------------------------------------------------
# Test 3 — RED: prove a clean exit commits.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
@pytest.mark.database
async def test_transaction_commits_on_clean_exit(
    stock_repo, sample_stock_factory, tx_session_factory
):
    """A clean exit from the block must commit the session.

    On unfixed code the body never executes so the row is never added,
    let alone committed. After the fix, the row must be visible from a
    fresh session.
    """
    async with stock_repo.transaction() as session:
        session.add(sample_stock_factory(symbol="CMTOK"))
        await session.flush()

    from backend.models.unified_models import Stock

    async with tx_session_factory() as verify_session:
        result = await verify_session.execute(
            select(Stock).where(Stock.symbol == "CMTOK")
        )
        found = result.scalar_one_or_none()

    assert found is not None, (
        "commit failed — row 'CMTOK' missing after clean transaction exit. "
        "transaction() either did not commit, or never executed the body."
    )
    assert found.symbol == "CMTOK"
