> **ARCHIVED 2026-04-27 by 15-test-suite**
> Original: docs/test-fix-priority-matrix.md
> Validation summary: 3/5 claims still current.
> See `../../reports/15-test-suite.md` §2 for per-claim status.

[Claims from 2026-01-27 priority matrix]
1. "P0: Add asyncio_mode=auto" — fully_stale: pytest.ini:58 has asyncio_mode = auto.
2. "P0: Fix session-scoped event_loop" — partially_stale: conftest.py clean, but integration_test_fixtures.py:650 and test_response_optimizer.py:403 still have deprecated overrides.
3. "P1: Fix missing imports / broken modules" — current: test_performance_optimizations.py:34 still imports OptimizedRecommendationEngine outside the import guard (confirmed by grep).
4. "P1: pytest-asyncio version pinning discrepancy" — current: requirements.txt=0.23.8, requirements-dev.txt=0.23.3 (confirmed by grep).
5. "P2: Coverage reporting improvements" — partially_stale: README.md now sets 85% target, but pytest.ini comment still says 60%.
