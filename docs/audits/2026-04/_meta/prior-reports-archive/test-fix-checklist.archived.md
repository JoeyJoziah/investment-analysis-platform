> **ARCHIVED 2026-04-27 by 15-test-suite**
> Original: docs/test-fix-checklist.md
> Validation summary: 2/5 claims still current.
> See `../../reports/15-test-suite.md` §2 for per-claim status.

Key claims validated: source file read at docs/test-fix-checklist.md.
1. "Session-scoped event_loop fixtures" — partially_stale: conftest.py (line count check = 0 event_loop references), but fixtures/integration_test_fixtures.py:650 and middleware/test_response_optimizer.py:403 still have deprecated event_loop overrides.
2. "asyncio_mode=auto missing from pytest.ini" — fully_stale: pytest.ini line 58 now has asyncio_mode = auto.
3. "async_fixtures.py event_loop at line 213" — fully_stale: async_fixtures.py does not exist (MISSING verified by filesystem check).
4. "integration_test_fixtures.py:650 event_loop" — current: grep confirms event_loop at line 650.
5. "middleware/test_response_optimizer.py event_loop" — current: grep confirms event_loop at line 403.
