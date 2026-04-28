# Cluster E — Test Exclusions & Mocked-Over-Real-Bug Patterns

**Worker:** Cluster Worker E (test-exclusion)
**Anchor:** F-15-003 (test exclusions in `pytest.ini`)
**Source scopes:** 06 (airflow-pipelines), 15 (test-suite)
**Sequencing role:** **Run FIRST or in parallel with A.** E un-excludes hidden tests, surfacing failures that become signal inputs for clusters B (jwt), C (csp), and D (random-data). E itself has **no hard prerequisites**.

---

## 1. Cluster Overview

Cluster E is the "test signal recovery" cluster. The audit finds that the project's pytest configuration **silently hides 1,356+ lines of security and database tests** (F-15-003), guards real-database integration tests behind a Docker check that fails silently in CI (F-15-008, F-15-020), and contains tests that **mock the very production code paths they are supposed to validate** (F-15-004, F-15-011), letting confirmed bugs in scopes 02, 07, 08, and 09 ship undetected.

The cluster also collects related test-infrastructure hygiene: collection-time errors (F-15-001), wrong test-runner flags (F-15-002), Airflow DAG test path issues (F-06-012), inconsistent coverage thresholds (F-15-009, F-15-019, F-15-027), hard-coded test discovery (F-15-021), an orphaned conftest fixture (F-15-013), and dead/misplaced files (F-15-023, F-15-026).

**Why E runs first:** Un-excluding the security/database directories will produce a cascade of failures from already-known bugs covered by other clusters. Capturing those failures *before* clusters B/C/D begin gives them concrete failing tests to drive their fixes (TDD-style), instead of relying solely on the audit text. The fix order matters: **infrastructure first, mock-removal second, real-bug fixes third (delegated to other clusters)**.

---

## 2. Member Findings (15)

| ID | Severity | Category | Title (short) |
|---|---|---|---|
| F-06-012 | medium | testing_gap | Airflow `test_technical_indicators.py` bare import only works from `utils/` |
| F-15-001 | critical | bug | `OptimizedRecommendationEngine` import outside `importorskip` guard kills 50+ tests at collection |
| F-15-002 | critical | bug | `run-tests.sh` passes Jest flags (`--watchAll`, `--passWithNoTests`) to Vitest 4.x |
| F-15-003 | critical | broken_import | `pytest.ini` excludes `tests/` — 1,356 lines of security/db tests never run |
| F-15-004 | critical | testing_gap | `FinnhubWebSocketClient.connect()` session-lifetime bug has no real test (mocks override) |
| F-15-008 | high | broken_dependency | Docker guard silently skips entire DB integration file in CI |
| F-15-009 | high | architecture | Three inconsistent coverage thresholds (60%/75%/85%) |
| F-15-011 | high | testing_gap | `AsyncBaseRepository.transaction()` async-generator bug has zero coverage |
| F-15-013 | medium | architecture | Orphaned `tests/security/conftest.py` defines deprecated session-scope `event_loop` |
| F-15-019 | medium | doc_drift | README coverage targets (85%/95%/100%) contradict enforced 75% |
| F-15-020 | medium | testing_gap | No SQLite fallback for DB integration tests |
| F-15-021 | medium | architecture | `run_integration_tests.py` hard-codes 6 files, ignores `backend/tests/integration/` (12 files) |
| F-15-023 | low | doc_drift | Five stub-redirect markdown files in `tests/` |
| F-15-026 | low | stale_code | `benchmark_n1_query_fix.py` lives in `backend/tests/` but is not a pytest file |
| F-15-027 | low | doc_drift | "Security code coverage: 100%" claim is unenforced and unachievable with mock-heavy tests |

**ID coverage assertion:** All 15 findings referenced. (F-06-012, F-15-001, F-15-002, F-15-003, F-15-004, F-15-008, F-15-009, F-15-011, F-15-013, F-15-019, F-15-020, F-15-021, F-15-023, F-15-026, F-15-027)

---

## 3. Sequenced Fix Steps

> **Fail-first protocol:** For each step that un-excludes or un-mocks a code path, run the affected test BEFORE the fix to confirm it now FAILS (or now collects). Capture that failure as the signal artifact for downstream clusters.

### Step 1 — Un-exclude `tests/security/` (anchor: F-15-003)

**File:** `pytest.ini`

1. **Verify path:** `pytest --collect-only tests/security/ 2>&1 | head` → expect "no tests collected" / not discovered.
2. **Edit `pytest.ini`:**
   - Add `tests` to `testpaths` (becomes `testpaths = backend/tests tests`).
   - Remove `tests` from `norecursedirs`.
3. **Pre-fix run (fail-first):** `pytest tests/security/ --collect-only` → must now collect items. Then `pytest tests/security/ -q --no-header` → **expect failures** (these are the cascade for B/C/D).
4. **Capture failures** to `docs/audits/2026-04/_synthesis/_meta/artifacts/E-step1-failures.txt`.

**Cross-scope cascade expected:**
- `tests/security/test_phase2_improvements.py` → admin API security findings → feeds **Cluster B** (jwt/auth) and **C** (csp).
- `tests/security/test_row_locking.py` → DB row-lock findings → feeds scope-07 cluster.
- `tests/test_database_fixes.py` → DB schema/fix findings → feeds scope-07.

### Step 2 — Un-exclude `tests/database/` and bare `tests/` files (continues F-15-003)

1. **Verify path:** `ls tests/` and `ls tests/database/ 2>/dev/null` to enumerate files now reachable.
2. **Same `pytest.ini` change as Step 1** also covers this; no additional config edit needed.
3. **Fail-first:** `pytest tests/test_database_fixes.py -q` → expect failures rooted in scope-07 (F-07-002 transaction bug, F-07-001-class).
4. Optional sub-fix (low-risk): clean orphaned `tests/security/conftest.py` event_loop fixture (**F-15-013**) — delete the deprecated session-scope `event_loop` fixture so it doesn't conflict with `asyncio_mode = auto` once the dir is discovered.

### Step 3 — Fix collection-time errors that block discovery (F-15-001, F-06-012)

These prevent Step-1/2 cascades from being captured cleanly.

1. **F-15-001:** In `backend/tests/test_performance_optimizations.py`, move line 34 (`from backend.analytics.recommendation_engine import OptimizedRecommendationEngine`) inside the `try`/`importorskip` block (lines 16–33), or wrap in its own `try/except ImportError: pytest.skip(..., allow_module_level=True)`.
   - **Verify:** `pytest --collect-only backend/tests/test_performance_optimizations.py` returns 50+ items, no `AttributeError`.
2. **F-06-012:** Add `data_pipelines/airflow/dags/utils/conftest.py` containing:
   ```python
   import os, sys
   sys.path.insert(0, os.path.dirname(__file__))
   ```
   **Verify:** `pytest data_pipelines/airflow/dags/utils/test_technical_indicators.py` from project root exits 0.

### Step 4 — Fix runner-flag and runner-discovery bugs (F-15-002, F-15-021)

1. **F-15-002:** In `run-tests.sh:84`, replace `npm test -- --watchAll=false --passWithNoTests` with `npm run test -- --run`. Vitest 4.x recognises `--run`.
   - **Verify:** `./run-tests.sh --frontend` exits 0 and reports vitest results for the 16 files in `src/`.
2. **F-15-021:** In `run_integration_tests.py:35-49`, replace the hardcoded 6-file list with dynamic discovery:
   ```python
   integration_dir = self.test_dir / "integration"
   self.test_categories["integration"] = sorted(integration_dir.glob("test_*.py"))
   self.test_categories["all"] = sorted(self.test_dir.rglob("test_*.py"))
   ```
   **Verify:** `python run_integration_tests.py --categories integration` discovers all 12 files in `backend/tests/integration/`.

### Step 5 — Surface mocked-over-real-bugs (F-15-004, F-15-011)

These are tests that *exist* but mock the bug under test, hiding scope-02 and scope-07 failures. **E does not fix the underlying code bugs** — that belongs to scope-02/07 follow-ups. E adds the failing tests so those clusters have signal.

1. **F-15-004:** Add a new `TestConnect` class to `backend/tests/unit/test_realtime_price_service.py` exercising `FinnhubWebSocketClient.connect()` with a minimally-mocked `aiohttp.ClientSession` (do **not** patch `connect` itself). Assert `client.ws is not None and not client.ws.closed` after `await client.connect()` returns. **Test must FAIL pre-fix** (verifies F-02-002 session-lifetime bug). Hand-off to scope-02 cluster.
2. **F-15-011:** Add unit tests in `backend/tests/unit/test_repositories_unit.py` covering `AsyncBaseRepository.transaction()`:
   - `test_transaction_commits_on_success`
   - `test_transaction_rolls_back_on_exception`
   These will FAIL because `transaction()` returns an async generator object, not a result (F-07-002). Hand-off to scope-07 cluster.

### Step 6 — Add Docker-optional / SQLite fallback (F-15-008, F-15-020)

1. **F-15-008:** In `backend/tests/test_database_integration.py:21-26`, replace silent `pytest.skip(...)` with a CI-visible reason and emit a `pytest.warns(...)` message. Add env-var override `INTEGRATION_REQUIRE_DOCKER=1` for CI to fail on skip.
2. **F-15-020:** Create new file `backend/tests/integration/test_database_sqlite.py` (NOT root, per workspace rules) with 3–5 critical CRUD/transaction tests using `sqlite+aiosqlite:///:memory:` (mirror `conftest.py`). No Docker dependency — runs everywhere.
   - **Verify:** `pytest backend/tests/integration/test_database_sqlite.py -v` passes in any env.

### Step 7 — Coverage threshold reconciliation (F-15-009, F-15-019, F-15-027)

1. Decide **single canonical threshold = 85%** (matches README intent).
2. In `pytest.ini:addopts`, uncomment / add `--cov=backend --cov-fail-under=85`.
3. In `run_integration_tests.py:205`, change `"--cov-fail-under=75"` → `"--cov-fail-under=85"`.
4. Update `backend/tests/README.md`: keep 85% as enforced minimum, mark 95%/100% targets as "aspirational, not gated" (F-15-019, F-15-027).
5. **Verify:** `pytest backend/tests/ --cov=backend --cov-fail-under=85` is the one canonical gate.

### Step 8 — Cleanup (F-15-023, F-15-026)

1. **F-15-023:** Delete `tests/TEST_SUMMARY.md`, `tests/TEST_METRICS.md`, `tests/E2E_AND_INTEGRATION_TESTS.md`, `tests/QUICK_START.md`, `tests/FILE_MANIFEST.md`. Add a single short `tests/README.md` pointing to `backend/tests/README.md`.
2. **F-15-026:** `git mv backend/tests/benchmark_n1_query_fix.py scripts/performance/benchmark_n1_query_fix.py`. Update its docstring `python -m ...` path.
   - **Verify:** `pytest backend/tests/ --collect-only | grep benchmark_n1` returns nothing.

---

## 4. Files Touched

**Configuration / runners (high traffic):**
- `pytest.ini` — testpaths, norecursedirs, addopts (Steps 1, 2, 7)
- `run-tests.sh` — line 84 vitest flag fix (Step 4)
- `run_integration_tests.py` — lines 35–49 (dynamic discovery), line 205 (threshold) (Steps 4, 7)

**Test files (additions / small edits):**
- `backend/tests/test_performance_optimizations.py:34` (F-15-001)
- `backend/tests/test_database_integration.py:21-26` (F-15-008)
- `backend/tests/unit/test_realtime_price_service.py` — add `TestConnect` (F-15-004)
- `backend/tests/unit/test_repositories_unit.py` — add transaction tests (F-15-011)
- `tests/security/conftest.py` — remove `event_loop` fixture (F-15-013)

**New files:**
- `data_pipelines/airflow/dags/utils/conftest.py` (F-06-012)
- `backend/tests/integration/test_database_sqlite.py` (F-15-020)

**Moves / deletes:**
- `backend/tests/benchmark_n1_query_fix.py` → `scripts/performance/benchmark_n1_query_fix.py` (F-15-026)
- Delete: `tests/TEST_SUMMARY.md`, `tests/TEST_METRICS.md`, `tests/E2E_AND_INTEGRATION_TESTS.md`, `tests/QUICK_START.md`, `tests/FILE_MANIFEST.md` (F-15-023)
- Add: `tests/README.md` (single redirect)

**Docs:**
- `backend/tests/README.md` — coverage section (F-15-019, F-15-027)

---

## 5. Acceptance Tests

Each is `pass/fail` runnable; failures from un-excluded real bugs are EXPECTED and tracked as cluster-cascade artifacts (not E acceptance failures).

| # | Command | Expected |
|---|---|---|
| AT-1 | `pytest --collect-only tests/security/ tests/` | Returns ≥3 test files, ≥ 1,356 LOC of tests collected; 0 collection errors |
| AT-2 | `pytest tests/security/ tests/ -q` | Runs without `ERROR` rows. Failures permitted (cascade to B/C/D); they MUST be captured to `_meta/artifacts/E-cascade-*.txt` |
| AT-3 | `pytest --collect-only backend/tests/test_performance_optimizations.py` | Reports 50+ items, no `AttributeError` |
| AT-4 | `./run-tests.sh --frontend` | Exits 0, vitest reports results for 16 src/ files |
| AT-5 | `pytest data_pipelines/airflow/dags/utils/test_technical_indicators.py` (from repo root) | Exits 0; no `ModuleNotFoundError` |
| AT-6 | `python run_integration_tests.py --categories integration --dry-run` | Lists all 12 files in `backend/tests/integration/` |
| AT-7 | `pytest backend/tests/integration/test_database_sqlite.py -v` | All tests pass without Docker |
| AT-8 | `grep -E '^addopts' pytest.ini` | Contains `--cov-fail-under=85` |
| AT-9 | `pytest backend/tests/ --collect-only \| grep -c benchmark_n1` | Returns `0` |
| AT-10 | `pytest backend/tests/unit/test_realtime_price_service.py::TestConnect -v` | Exists, fails pre-scope-02-fix (signal preserved) |
| AT-11 | `pytest backend/tests/unit/test_repositories_unit.py -k transaction` | Exists, fails pre-scope-07-fix |
| AT-12 | `ls tests/*.md \| wc -l` | `1` (only README.md) |
| AT-13 | `pytest tests/security/ --collect-only 2>&1 \| grep -c DeprecationWarning` | `0` (event_loop fixture removed) |

---

## 6. Rollback Plan

E is dominated by configuration changes; rollback is a single-line revert per file:

```bash
# Total cluster rollback in <2 min
git checkout HEAD~ -- pytest.ini run-tests.sh run_integration_tests.py backend/tests/README.md
git revert <e-step1-commit>..<e-step8-commit>      # for new files / moves
# Re-add original deletions if needed
```

**Per-step rollback granularity:**
- Step 1/2 (un-exclude): revert `pytest.ini` `testpaths`/`norecursedirs` lines.
- Step 3 (collection): revert single line in `test_performance_optimizations.py`; delete new airflow `conftest.py`.
- Step 4 (runners): revert `run-tests.sh:84`, `run_integration_tests.py:35-49`.
- Step 5 (new tests): delete `TestConnect` class and new `test_transaction_*` cases.
- Step 6 (Docker/SQLite): delete new `test_database_sqlite.py`; revert guard wording.
- Step 7 (thresholds): revert `pytest.ini` and `run_integration_tests.py` numbers.
- Step 8 (cleanup): `git mv` reverse + restore deleted stubs.

**No production code is modified by Cluster E**, so rollback risk is bounded to CI flakiness.

---

## 7. Dependencies

**Type:** `independent` — Cluster E has **no upstream blockers** and is a **prerequisite signal source for B, C, D**.

| Direction | Cluster | Relation |
|---|---|---|
| Upstream blockers | (none) | E can begin immediately |
| Parallel-safe with | A | A and E touch disjoint files (A = security/auth code; E = test config). May run concurrently. |
| Downstream consumers | B (jwt) | B uses failing tests in `tests/security/test_phase2_improvements.py` surfaced by E-step-1 as TDD anchor. |
| Downstream consumers | C (csp) | C uses admin-API/CSP failures from un-excluded suite. |
| Downstream consumers | D (random-data) | D uses test failures in DB & data-fixture tests un-excluded by E-step-2. |
| Downstream consumers | scope-02 follow-up | Consumes new `TestConnect` (F-15-004) failure. |
| Downstream consumers | scope-07 follow-up | Consumes new transaction tests (F-15-011) failure. |

**Recommended schedule:** **Run E first, in parallel with A.** Begin B/C/D after E-step-1 and E-step-2 commits land and the cascade artifact files exist.

---

## 8. Effort & Cost

| Step | Findings | Effort (hrs) | Risk |
|---|---|---|---|
| 1 | F-15-003 (a) | 1.0 | low (config) |
| 2 | F-15-003 (b), F-15-013 | 2.0 | low |
| 3 | F-15-001, F-06-012 | 2.0 | low |
| 4 | F-15-002, F-15-021 | 3.0 | low–medium (CI flow change) |
| 5 | F-15-004, F-15-011 | 7.0 | medium (new tests must reproduce known bugs) |
| 6 | F-15-008, F-15-020 | 7.0 | medium (new SQLite path) |
| 7 | F-15-009, F-15-019, F-15-027 | 4.0 | low (numbers + docs) |
| 8 | F-15-023, F-15-026 | 1.0 | low |
| **Total** | **15** | **~27 hrs** | mostly low |

**Cost (model routing assumption — Haiku for mechanical edits, Sonnet for new test authorship):** ~$0.80–$1.50 in agent runtime, dominated by Step 5/6 test authoring.

---

## 9. Loki-Actionable

**Largely yes, with one caveat:**

- ✅ Steps 1, 2, 3, 4, 7, 8 are mechanical config / move / threshold edits — fully Loki-automatable (single-file diffs with deterministic AT verification).
- ⚠️ Step 5 (new `TestConnect`, transaction tests) requires **judgement on minimal mocking** to expose the real bug rather than re-mock it. Loki can scaffold; a human/Sonnet review is recommended before merge.
- ⚠️ Step 6 (SQLite fallback) requires choosing which 3–5 CRUD/transaction tests are critical — minor design decision.
- ✅ Failures surfaced by Steps 1–2 are **not** decisions for E to make; they hand off as artifact files to clusters B/C/D.

All 15 findings carry `loki_actionable: true` in the slice.

---

## 10. Risks

1. **CI flood** — Un-exclusion will turn CI red (potentially dozens of failing tests overnight). **Mitigation:** Schedule Steps 1–2 during a planned window; mark new failures `xfail(strict=False, reason="cascade-to-cluster-X")` initially with a tracking issue per cluster, then progressively un-`xfail` as B/C/D land fixes. Coordinate with team via Slack before merging Step 1.
2. **Double-reporting bug** — Tests un-excluded by E may also be re-reported by clusters B/C/D as "newly found" bugs. **Mitigation:** E owns the cascade artifact files; B/C/D reference them in their workpapers as input.
3. **Coverage gate flip** — Bumping `--cov-fail-under` from 75 → 85 may break the next CI run on unrelated PRs. **Mitigation:** Land Step 7 in a dedicated PR after Steps 1–6 raise actual coverage; or stage at 80% then 85%.
4. **Mock-removal regressions** — Rewriting `TestConnect` (F-15-004) without `patch.object(ws_client, "connect", ...)` may make existing reconnect tests slower or flaky if `aiohttp.ClientSession` isn't fully isolated. **Mitigation:** Keep reconnect tests on the original mock path; add `TestConnect` as a new class only.
5. **Path coupling on Airflow conftest** — `sys.path.insert(0, ...)` in `conftest.py` (F-06-012) can shadow other modules if the airflow utils dir contains generic names. **Mitigation:** Prefer the relative-import variant (`from .technical_indicators_calculator import ...`) when feasible.
6. **Hard-coded test list users** — External CI scripts may rely on `run_integration_tests.py`'s 6-file list (F-15-021). **Mitigation:** Keep the old hardcoded names as a fallback `legacy` category; the dynamic list becomes the new default.
7. **Stub-redirect deletions break old links** — F-15-023 deletions may break wiki links. **Mitigation:** Keep `tests/README.md` redirect.

---

**Final assertion: All 15 cluster-E IDs are referenced in this workpaper:** F-06-012, F-15-001, F-15-002, F-15-003, F-15-004, F-15-008, F-15-009, F-15-011, F-15-013, F-15-019, F-15-020, F-15-021, F-15-023, F-15-026, F-15-027. ✓
