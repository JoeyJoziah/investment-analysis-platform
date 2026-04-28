# tests/

This directory holds **legacy / cross-cutting** Python tests that pre-date
the canonical `backend/tests/` layout (currently: `tests/security/` and
`tests/test_database_fixes.py`).

For the primary test suite, infrastructure, fixtures, coverage policy, and
CI integration see:

- **[backend/tests/README.md](../backend/tests/README.md)** -- canonical guide

The five legacy stub files that previously lived here
(`TEST_SUMMARY.md`, `TEST_METRICS.md`, `E2E_AND_INTEGRATION_TESTS.md`,
`QUICK_START.md`, `FILE_MANIFEST.md`) were removed by audit 2026-04
(F-15-023). Their content was either out of date or duplicated by
`backend/tests/README.md`.
