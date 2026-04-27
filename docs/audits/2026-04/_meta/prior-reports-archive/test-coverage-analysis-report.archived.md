> **ARCHIVED 2026-04-27 by 15-test-suite**
> Original: docs/test-coverage-analysis-report.md
> Validation summary: 2/6 claims still current.
> See `../../reports/15-test-suite.md` §2 for per-claim status.

Key claims validated:
1. "10/38 tests passing (26.3%)" — fully_stale: TEST_BASELINE_REPORT (2026-03-04) confirms 5020+ passing; integration suite expanded from 38 to 113+ pytest files.
2. "Overall Coverage 10.51%" — fully_stale: coverage numbers from integration-only run on 2026-01-28; full suite significantly larger now.
3. "Schema Errors: 0 (all 7 fixed in Wave 4.5)" — current: conftest.py uses unified_models.Base without schema errors observed.
4. "28 tests failing due to missing endpoints and CSRF configuration" — partially_stale: many endpoints now implemented; CSRF fixtures in conftest.py; but some integration failures may persist (testcontainers guard still present).
5. "analytics/fundamental_analysis.py 8.56% coverage" — partially_stale: unit/ test files for analytics added (test_analytics_extended_agent4.py), coverage likely improved.
6. "Critical: 5 of 16 high-priority modules under 10% coverage" — partially_stale: unit tests added for most; exact current coverage unknown without a live run.
