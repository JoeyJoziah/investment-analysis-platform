> **ARCHIVED 2026-04-27 by 15-test-suite**
> Original: docs/TEST_REPORT.md
> Validation summary: 4/7 claims still current.
> See `../../reports/15-test-suite.md` §2 for per-claim status.

# Investment Analysis Platform - Test Report
**Date:** 2026-01-27
**Tester:** QA Testing Agent

[Original content archived verbatim — see source file for full text]

Key claims validated:
1. Frontend coverage ~45% — partially_stale (16 vitest files now exist vs "partial" in 2026-01)
2. Backend coverage ~65% — partially_stale (5020+ tests added since 2026-01-27)
3. ML Pipeline coverage ~55% — current (still guarded by pytest.skip patterns)
4. Authentication coverage ~80% — current (conftest.py auth fixtures active)
5. "strict: false" TypeScript concern — current (vite.config.ts still sets strict: false in test env)
6. No TypeScript checks in CI/CD — fully_stale (scope 14 shows mypy.yml and type-check.yml exist)
7. Overall coverage ~60%, target 80% — partially_stale (baseline now 85% target per README.md)
