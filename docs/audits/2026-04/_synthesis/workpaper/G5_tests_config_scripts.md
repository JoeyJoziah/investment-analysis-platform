# G5 — Tests, Config & Scripts (Cluster Workpaper)

**Cluster Worker:** G5_tests_config_scripts
**Scopes:** 15-test-suite, 16-config-secrets, 17-scripts-tooling
**Findings:** 37
**Status:** Synthesized

---

## 1. Cluster Overview

This cluster captures the residual hygiene work in the test suite, build/runtime configuration, and developer-facing scripts after the secret-rotation work (Cluster A) and broad test-exclusion remediation (Cluster E) are landed. It groups three sub-themes:

- **Test infrastructure & coverage** (scope 15): incorrect mock specs that mask production bugs, stale `Base` imports in integration tests, deprecated `event_loop` fixtures and `asyncio.get_event_loop()` anti-patterns, redundant `@pytest.mark.asyncio` decorators, missing regression coverage for known security/architecture issues (F-08-001 PBKDF2, F-02-001 mixin divergence), and a frontend gap on `api.service.ts`.
- **Config consistency** (scope 16): Python version mismatch between `pyproject.toml` and `.mypy.ini`, dual mypy configurations with conflicting strictness, an undocumented redis 5→7 major version jump, an incomplete 2026-01-27 `requirements-ml.txt` split, rigid pinning vs. ranges (numpy), no `requirements-test.txt`, dead `backend/requirements-dev.txt`, non-standard `comment` field in `package.json`, and Makefile-script coupling without preflight checks.
- **Scripts consolidation** (scope 17): unbounded wait loops in `setup.sh`, bootstrap import failure in `install_dependencies.py`, legacy `docker-compose` v1 calls in deployment/migration scripts, credential-bearing duplicate test scripts, a 0-byte file in `scripts/scripts/`, committed `.sh.old` files, missing `set -e` and venv guards in `install_critical_deps.sh`, two competing secret-generation scripts at different entropy levels, an orphaned Airflow→Prefect migrator, 13 overlapping deployment scripts, claude-flow tooling shipped to production, missing idempotency in `init_database.sh`, bare `except:` clauses, zero automated script tests, and `SCRIPTS_REFERENCE.md` drift.

The unifying theme: this is the "post-consolidation cleanup" tier — none of these is a flag-day breaking issue, but together they cause cryptic failures, mask production bugs, and slow contributor onboarding. Most are surgically scoped and Loki-actionable.

---

## 2. Member Findings (all 37)

### Sub-theme A — Test infrastructure & coverage (9 findings)

| ID | Sev | Title |
|---|---|---|
| F-15-005 | high | `Mock(spec=StockRecommendation)` masks missing `ranking_score` field bug |
| F-15-010 | high | `test_integration_comprehensive.py:30` imports stale `Base` from `backend.models.database` |
| F-15-014 | medium | Deprecated session-scoped `event_loop` fixture in `integration_test_fixtures.py:650-654` |
| F-15-015 | medium | Deprecated `event_loop` fixture in `test_response_optimizer.py:402-407` |
| F-15-016 | medium | `asyncio.get_event_loop().run_until_complete()` anti-pattern in `test_ml_pipeline.py:180,191` |
| F-15-017 | medium | No tests cover `RecommendationCrudMixin`/`RecommendationAnalysisMixin` divergence (F-02-001) |
| F-15-018 | medium | No regression test for hardcoded PBKDF2 salt (F-08-001) |
| F-15-022 | medium | 1,622 redundant `@pytest.mark.asyncio` decorators across suite |
| F-15-025 | low | Frontend `api.service.ts` has no direct unit tests (401/refresh path) |

### Sub-theme B — Config consistency (10 findings)

| ID | Sev | Title |
|---|---|---|
| F-16-001 | critical | Python version mismatch: `pyproject.toml` 3.12 vs `.mypy.ini` 3.11 |
| F-16-002 | critical | Undocumented redis 5.0.7 → 7.0.0 major version jump |
| F-16-004 | high | Dual mypy configs (`pyproject.toml` strict vs `.mypy.ini` lenient) |
| F-16-005 | high | `requirements-ml.txt` never created; ~2GB ML deps still in main requirements |
| F-16-008 | medium | `numpy==1.24.0` rigid pin instead of `>=1.24.0,<2.0.0` |
| F-16-011 | medium | 136 `==` pins vs 1 `>=` — no documented versioning strategy |
| F-16-012 | medium | No `requirements-test.txt`; CI installs full dev tooling |
| F-16-013 | low | `package.json` non-standard `comment` field |
| F-16-014 | low | Makefile calls `setup.sh`/`start.sh`/`stop.sh` without preflight checks |
| F-16-015 | low | `backend/requirements-dev.txt` is dead (single line, redundant with root) |

### Sub-theme C — Scripts consolidation (18 findings)

| ID | Sev | Title |
|---|---|---|
| F-17-003 | high | `setup.sh:88-103` unbounded `until` wait loops with no timeout |
| F-17-004 | high | `install_dependencies.py:51-52` top-level imports of `packaging`/`requests` (bootstrap failure) |
| F-17-005 | high | `deploy_ml_production.sh` uses legacy `docker-compose` (v1) |
| F-17-006 | high | `migrate_to_optimized.sh` uses legacy `docker-compose` (v1) at 8 sites |
| F-17-008 | high | Credential-bearing duplicate test scripts in `scripts/testing/` (5 variants) |
| F-17-010 | medium | Zero-byte `scripts/scripts/simple_migrate.py` shadows real script |
| F-17-011 | medium | Stale `*.sh.old` files committed to `scripts/setup/` |
| F-17-012 | medium | `install_critical_deps.sh` lacks `set -e` and venv guard |
| F-17-013 | medium | Two secret-generation scripts at different entropy levels |
| F-17-014 | medium | `SCRIPTS_REFERENCE.md` missing `sync-all.sh` / board-sync suite |
| F-17-015 | medium | `migrate_airflow_to_prefect.py` orphaned (no Prefect installed) |
| F-17-016 | medium | 13 deployment scripts with overlapping responsibility |
| F-17-017 | medium | `phase1-consolidation.py` (claude-flow tooling) shipped in repo root scripts |
| F-17-018 | low | `.sh.old` agent-setup scripts target obsolete `$HOME/.config/claude-code` |
| F-17-019 | low | `init_database.sh` missing idempotency, unquoted `$DB_NAME` |
| F-17-020 | low | Bare `except:` clauses in `activate_pipeline.py:57-68` |
| F-17-021 | low | Zero automated tests for 150 scripts (no shellcheck, no pytest) |
| F-17-022 | low | `SCRIPTS_REFERENCE.md` lists non-existent root `test_performance.sh` |

**Total: 9 + 10 + 18 = 37.**

---

## 3. Sequenced Fix Steps (grouped by sub-theme)

Each step is small, safe, and ordered so later steps benefit from earlier signal.

### Step 1 — Config alignment first (unblocks consistent type-checking signal)

1. **F-16-001**: Edit `.mypy.ini` line 3 → `python_version = 3.12`.
2. **F-16-004**: Delete `.mypy.ini`; consolidate per-module overrides into `pyproject.toml [tool.mypy]` table; document strict vs. lenient sections inline.
3. **F-16-013**: Remove `comment` field from `package.json`; move rationale to `docs/CLAUDE_FLOW_V3_VERSION_ALIGNMENT.md` (already referenced).
4. **F-16-015**: Delete `backend/requirements-dev.txt` after grep-verifying no Dockerfile/CI references.

### Step 2 — Requirements file restructure

5. **F-16-012**: Create `requirements-test.txt` (pytest, pytest-asyncio, pytest-cov, pytest-mock, testcontainers, requests-mock, faker). Update `requirements-dev.txt` to `-r requirements-test.txt`. Update CI to use `requirements-test.txt`.
6. **F-16-005**: Create `requirements-ml.txt` (torch, transformers, huggingface_hub, datasets, shap, lime, optuna, plotly, matplotlib, seaborn). Remove from main `requirements.txt`. Update README install instructions and Dockerfile (conditional ML stage).
7. **F-16-008**: Loosen `numpy==1.24.0` → `numpy>=1.24.0,<2.0.0`. Run `pip install --dry-run` and the test suite.
8. **F-16-002**: Document redis 5→7 upgrade in `CHANGELOG` and `ENVIRONMENT.md`. Run integration suite (Celery broker + result backend) and record results.
9. **F-16-011**: Author `docs/VERSIONING.md` with three-tier policy (core `==`, data libs `>=,<`, optional `==`). Apply selectively.

### Step 3 — Script bootstrap & safety hardening (blocks infra issues from masking each other)

10. **F-17-004**: Add try/except guard for `packaging`/`requests` imports in `install_dependencies.py`; bootstrap-install if missing.
11. **F-17-003**: Add bounded counters + timeouts to all three `until` loops in `setup.sh:88-103` (60 iterations × 2s = 120s max).
12. **F-17-012**: Add `set -e` + `[ -d venv ] || python3 -m venv venv` to `install_critical_deps.sh`; replace `|| echo "Failed"` with proper exit codes.
13. **F-17-005, F-17-006**: Replace all `docker-compose` calls with `docker compose` in `deploy_ml_production.sh` and `migrate_to_optimized.sh` (8 sites in the latter, ~6 in the former).
14. **F-17-019**: Quote `"$DB_NAME"` and add `|| true` after `createdb` in `init_database.sh` for idempotency.
15. **F-17-020**: Replace bare `except:` in `activate_pipeline.py:57-68` with `except Exception as e:` + `logger.warning`.

### Step 4 — Script consolidation & cleanup (reduces surface area)

16. **F-17-010**: `rm -rf scripts/scripts/`.
17. **F-17-011, F-17-018**: Delete `scripts/setup/setup_global_agents.sh.old` and `scripts/setup/update_agents.sh.old` (and any other `*.sh.old`).
18. **F-17-008**: Pick one canonical `test_docker_connections.py` (env-var-only credentials). Delete the four `*_fixed.py` / `*_corrected.py` / `*_quick.py` variants. Cross-references Cluster A (secrets) and Cluster E (test exclusions).
19. **F-17-017**: Move `scripts/phase1-consolidation.py` to `.claude/tools/` and add `.dockerignore` rule.
20. **F-17-013**: Add deprecation banner + `exec` redirect in `scripts/generate_secrets.sh` pointing to `scripts/security/generate_secrets.sh`. Update `INSTALLATION_GUIDE.md`.
21. **F-17-015**: Decide on Prefect migration via ADR. Either remove `migrate_airflow_to_prefect.py` or add `requirements/prefect.txt` and migration plan. (Default: archive; soft-depends on scope 06 review.)
22. **F-17-016**: Author `scripts/deployment/README.md` mapping each of the 13 scripts to a use-case; deprecate or delete duplicates of root `start.sh`/`stop.sh`. Target ≤ 5 supported scripts.

### Step 5 — Docs sync

23. **F-17-014**: Add `sync-all.sh`, `sync-boards.sh`, `board-sync.sh`, `notion-sync.sh`, `scripts/github-board-sync.sh` to `docs/SCRIPTS_REFERENCE.md`.
24. **F-17-022**: Remove or correct the `test_performance.sh` entry in `SCRIPTS_REFERENCE.md`.
25. **F-16-014**: Add Makefile preflight `test -f ./setup.sh || (echo "Error: setup.sh missing" && exit 1)` for each referenced script.

### Step 6 — Test infrastructure hardening

26. **F-15-014**: Remove session-scoped `event_loop` fixture from `integration_test_fixtures.py:650-654`.
27. **F-15-015**: Remove function-scoped `event_loop` fixture from `test_response_optimizer.py:398-407`; preserve module-level `asyncio` import.
28. **F-15-016**: Convert `test_step_validate_input_returns_true` and `test_step_cleanup_noop` in `test_ml_pipeline.py` to `async def`; remove `run_until_complete`.
29. **F-15-022**: Codemod removal of `@pytest.mark.asyncio` decorators (1,622 occurrences) — single sweep with `ast`-based script (safer than `sed`).
30. **F-15-010**: Change `test_integration_comprehensive.py:30` from `from backend.models.database import Base` to `from backend.models.unified_models import Base` (consistent with `conftest.py:105`).

### Step 7 — Test-suite gap closure (regression coverage)

31. **F-15-005**: Replace `Mock(spec=StockRecommendation)` with real instances in `test_recommendation_engine.py:203-215` and `test_analytics_extended_agent4.py:839-846`. Test will fail until `ranking_score: float = 0.0` is added to the dataclass (links to F-09-003 in cluster G1).
32. **F-15-018**: Add `test_pbkdf2_uses_random_salt_and_sufficient_iterations` regression test (will fail until F-08-001 is fixed; intentional pending state — soft-depends on Cluster A).
33. **F-15-017**: Add smoke-import + signature-equivalence tests for `RecommendationCrudMixin`/`RecommendationAnalysisMixin` (links to F-02-001 in cluster G1).
34. **F-15-025**: Add `frontend/web/src/services/__tests__/api.service.test.ts` covering success GET, 401, network error, refresh path.

### Step 8 — Script test infrastructure

35. **F-17-021**: Add `tests/test_scripts/` directory; wire `shellcheck` in CI for `scripts/*.sh` and `scripts/deployment/*.sh`; add pytest import smoke tests for key Python scripts.

---

## 4. Files Touched

### Config

- `pyproject.toml`
- `.mypy.ini` (deleted)
- `package.json`
- `Makefile`
- `requirements.txt`
- `requirements-dev.txt`
- `requirements-test.txt` (new)
- `requirements-ml.txt` (new)
- `backend/requirements-dev.txt` (deleted)
- `Dockerfile` (conditional ML stage)
- `docs/VERSIONING.md` (new)
- `docs/CLAUDE_FLOW_V3_VERSION_ALIGNMENT.md` (extended)
- `docs/ENVIRONMENT.md`
- `CHANGELOG.md`

### Scripts

- `setup.sh`
- `install_dependencies.py`
- `install_critical_deps.sh`
- `scripts/init_database.sh`
- `scripts/data/activate_pipeline.py`
- `scripts/deploy_ml_production.sh`
- `scripts/migrate_to_optimized.sh`
- `scripts/scripts/` (deleted)
- `scripts/setup/setup_global_agents.sh.old` (deleted)
- `scripts/setup/update_agents.sh.old` (deleted)
- `scripts/testing/test_docker_connections.py` (canonical retained)
- `scripts/testing/test_docker_connections_fixed.py` (deleted)
- `scripts/testing/test_services_fixed.py` (deleted)
- `scripts/testing/test_services_corrected.py` (deleted)
- `scripts/testing/test_services_quick.py` (deleted)
- `scripts/phase1-consolidation.py` (moved to `.claude/tools/`)
- `scripts/generate_secrets.sh` (deprecation banner)
- `scripts/migrate_airflow_to_prefect.py` (archive or remove per ADR)
- `scripts/deployment/README.md` (new)
- `scripts/deployment/*.sh` (≤ 5 retained)
- `docs/SCRIPTS_REFERENCE.md`
- `INSTALLATION_GUIDE.md`
- `.dockerignore`
- `tests/test_scripts/` (new)
- `.github/workflows/*` (CI: requirements-test.txt + shellcheck)

### Tests

- `backend/tests/test_recommendation_engine.py`
- `backend/tests/test_analytics_extended_agent4.py`
- `backend/tests/test_integration_comprehensive.py`
- `backend/tests/fixtures/integration_test_fixtures.py`
- `backend/tests/middleware/test_response_optimizer.py`
- `backend/tests/unit/test_ml_pipeline.py`
- `backend/tests/security/` (new PBKDF2 regression test)
- `backend/tests/unit/test_recommendation_service.py` (mixin coverage)
- All test files containing `@pytest.mark.asyncio` (codemod sweep)
- `frontend/web/src/services/__tests__/api.service.test.ts` (new)

---

## 5. Acceptance Tests

Per finding (concise; matches each finding's `acceptance_test_hint`):

- F-15-005: `pytest backend/tests/test_recommendation_engine.py::TestRecommendationEngine::test_rank_recommendations -v` passes with real dataclass instances.
- F-15-010: `pytest --collect-only backend/tests/test_integration_comprehensive.py` collects with full schema.
- F-15-014, F-15-015, F-15-016: `pytest <path> -W error::DeprecationWarning` clean.
- F-15-017: `pytest backend/tests/unit/test_recommendation_service.py -k "mixin" -v` passes.
- F-15-018: `pytest backend/tests/security/ -k "pbkdf2" -v` (will fail intentionally until Cluster A fixes F-08-001).
- F-15-022: `grep -rn "pytest.mark.asyncio" backend/tests/ | wc -l` → 0.
- F-15-025: `vitest run src/services/__tests__/api.service.test.ts` passes.
- F-16-001: `grep python_version .mypy.ini` → 3.12 (or file deleted).
- F-16-002: `grep redis== requirements.txt` → `redis==7.0.0`; Celery integration tests green; CHANGELOG entry present.
- F-16-004: `ls -la .mypy.ini` → ENOENT; `mypy backend/` succeeds.
- F-16-005: `ls -la requirements-ml.txt` succeeds; main install footprint reduced.
- F-16-008: `grep numpy requirements.txt` → `numpy>=1.24.0,<2.0.0`; install green.
- F-16-011: `docs/VERSIONING.md` exists; ≥ 5 `>=` pins in `requirements.txt`.
- F-16-012: `pip install -r requirements-test.txt --dry-run` succeeds; CI uses it.
- F-16-013: `jq '.comment' package.json` → null; `npm install` succeeds.
- F-16-014: `make setup` either runs or errors with clear "setup.sh not found".
- F-16-015: `ls backend/requirements-dev.txt` → ENOENT; CI/Dockerfile grep clean.
- F-17-003: Stopping a service container produces `Timeout waiting for X` after ≤ 120s.
- F-17-004: `python3 -c "import sys; sys.modules['packaging']=None; sys.modules['requests']=None; import install_dependencies"` no `ImportError`.
- F-17-005, F-17-006: `./scripts/deploy_ml_production.sh --dry-run` and `migrate_to_optimized.sh` succeed under Compose v2 only.
- F-17-008: `ls scripts/testing/test_services*.py scripts/testing/test_docker*.py | wc -l` ≤ 2; remaining file uses env vars only.
- F-17-010: `ls scripts/scripts/` → ENOENT.
- F-17-011, F-17-018: `find scripts/ -name "*.old"` empty; no `$HOME/.config/claude-code` references.
- F-17-012: Re-running `install_critical_deps.sh` in clean dir creates venv and completes.
- F-17-013: `INSTALLATION_GUIDE.md` references only `scripts/security/generate_secrets.sh`.
- F-17-014: `grep "sync-all" docs/SCRIPTS_REFERENCE.md` returns ≥ 1.
- F-17-015: ADR recorded; script removed or `requirements/prefect.txt` exists.
- F-17-016: `scripts/deployment/README.md` exists; non-deprecated count ≤ 5.
- F-17-017: `phase1-consolidation.py` not in any Dockerfile `COPY`.
- F-17-019: Running `init_database.sh` twice succeeds.
- F-17-020: No bare `except:` in `activate_pipeline.py`.
- F-17-021: `shellcheck scripts/*.sh scripts/deployment/*.sh` returns 0; `pytest tests/test_scripts/` passes.
- F-17-022: `grep "test_performance" docs/SCRIPTS_REFERENCE.md` returns correct path or empty.

Cluster-level smoke gate: full `pytest backend/tests/ -W error::DeprecationWarning` clean; `mypy backend/` clean; `npm install` clean; `make setup` runs end-to-end.

---

## 6. Rollback Plan

Each step is a reversible commit. Recommended commits per sub-theme:

1. **Config**: `chore: align python version + consolidate mypy config + clean dead requirements`
2. **Requirements split**: `chore: split requirements into base/test/ml + loosen numpy + document redis 7 upgrade`
3. **Script safety**: `fix(scripts): add timeouts, set -e, venv guards, docker compose v2`
4. **Script cleanup**: `chore(scripts): remove .old files, dedupe testing/, move claude-flow tooling out of repo`
5. **Docs sync**: `docs: refresh SCRIPTS_REFERENCE + add VERSIONING.md + Makefile preflight`
6. **Test infra hardening**: `test: remove deprecated event_loop fixtures + asyncio anti-patterns + redundant decorators`
7. **Test gap closure**: `test: add regression coverage for ranking_score, PBKDF2 salt, mixin divergence, frontend api.service`
8. **Script test infra**: `ci: add shellcheck + script smoke-test scaffolding`

Rollback by reverting the relevant commit. Highest-risk commits (require careful CI signal):

- requirements split (#2): rollback restores `requirements.txt` superset; no broken installs.
- redundant `@pytest.mark.asyncio` codemod (#6): rollback restores decorators; tests still pass.
- mypy consolidation (#1): rollback restores `.mypy.ini`; both configs co-exist as before.

For each delete (`.mypy.ini`, `.sh.old`, dup test scripts, zero-byte files): files retained in git history, recoverable via `git restore --source <sha>`.

---

## 7. Dependencies

- **Soft-depends on Cluster A (secret rotation)**: F-17-008 dedup of credential-bearing test scripts requires the env-var-only credential pattern to be canonical (A's deliverable). F-17-013 (secret-gen consolidation) overlaps with A. F-15-018 (PBKDF2 regression test) intentionally fails until A fixes F-08-001.
- **Soft-depends on Cluster E (test exclusions un-stuck)**: Steps 6–7 (test-infra hardening, gap closure) yield best signal once previously-excluded tests run. F-15-022 codemod is safest after E's exclusion review (decorator removal could re-include broken tests).
- **Cross-scope to G1 (backend)**: F-15-005 (ranking_score) needs F-09-003's dataclass change; F-15-017 (mixin divergence) needs F-02-001 architectural decision.
- **Cross-scope to scope 06 (Airflow)**: F-17-015 needs an ADR on Prefect migration status.
- **Cross-scope to scope 13 (infra)**: F-17-005, F-17-006, F-17-016 deployment-script changes overlap with infra cluster.
- **Cross-scope to scope 18 (docs-health)**: F-17-014, F-17-022 are docs drift items.
- **No hard blockers**: every step in Sections 1–5 (config, scripts) is independent and can land before A/E.

---

## 8. Effort & Cost

Sum of `effort_hours` across all 37 findings: **48.5 hours**.

Breakdown by sub-theme:

- Tests (9 findings): 18 h (F-15-005:3, F-15-010:1, F-15-014:1, F-15-015:1, F-15-016:1, F-15-017:2, F-15-018:2, F-15-022:4, F-15-025:3)
- Config (10 findings): 10.0 h (F-16-001:0.5, F-16-002:3, F-16-004:1, F-16-005:1.5, F-16-008:0.5, F-16-011:2, F-16-012:0.5, F-16-013:0.25, F-16-014:0.5, F-16-015:0.25)
- Scripts (18 findings): 20.5 h (F-17-003:2, F-17-004:3, F-17-005:2, F-17-006:1, F-17-008:2, F-17-010:0.25, F-17-011:0.25, F-17-012:1, F-17-013:1, F-17-014:1, F-17-015:2, F-17-016:3, F-17-017:0.5, F-17-018:0.25, F-17-019:0.5, F-17-020:0.5, F-17-021:8, F-17-022:0.25)

Wall-clock estimate with one engineer in parallel-safe order: ~6 working days. With Loki batching of low-effort items (≤ 0.5 h each), ~12 of these collapse into a single PR (~3 h aggregate).

Cost: dominated by F-17-021 (8h) and F-15-022 (4h codemod). All others are surgical.

---

## 9. Loki-Actionable

**Loki-actionable: 35 of 37.**

Non-Loki:

- **F-15-025** (frontend api.service tests) — requires test design judgement about mocking strategy and refresh-flow ordering; better as a small spec'd PR than a Loki sweep.
- **F-16-011** (versioning policy doc) — requires design choice about the policy itself; doc authoring, not mechanical.
- **F-17-015** (Airflow→Prefect ADR) — requires a product/architecture decision; Loki cannot decide.

Highly Loki-friendly batches:

- All `*.old` deletions and `scripts/scripts/` removal (F-17-010, F-17-011, F-17-018).
- `docker-compose` → `docker compose` rewrites (F-17-005, F-17-006).
- `.mypy.ini` python version line, `package.json` comment field, `numpy` pin (F-16-001, F-16-013, F-16-008).
- `@pytest.mark.asyncio` codemod (F-15-022) — single mechanical sweep.
- `Base` import correction (F-15-010), bare-`except` rewrite (F-17-020).

---

## 10. Risks

- **R1 — Codemod over-removal (F-15-022):** A naive `sed` could munge multi-line decorators or string literals. Mitigation: AST-based codemod with dry-run + diff review; verify pytest collects same number of tests pre/post.
- **R2 — Redis 7 behavioral drift (F-16-002):** Major version bump may surface latent issues in Celery brokers or pub/sub flows. Mitigation: dedicated integration test pass + staging soak before documenting "stable".
- **R3 — Removing event_loop fixtures (F-15-014, F-15-015) breaks test isolation:** Some legacy tests may rely on session-scoped loops implicitly. Mitigation: run full suite with `-p no:cacheprovider` and verify no flaky failures.
- **R4 — Numpy `<2.0.0` upper bound traps a future SciPy/Pandas pull (F-16-008):** As ecosystem moves to numpy 2.x, this becomes the blocker. Mitigation: track and lift in next versioning sweep.
- **R5 — `requirements-ml.txt` split breaks Dockerfile build (F-16-005):** Conditional ML installation in Docker is non-trivial. Mitigation: add a `Dockerfile.ml` variant or build arg, with CI matrix coverage.
- **R6 — Deduping `scripts/testing/` (F-17-008) deletes a script CI silently relies on:** Mitigation: full repo grep + CI dry-run before deletion; preserve in git history.
- **R7 — `.mypy.ini` deletion (F-16-004) loses lenient overrides:** Mitigation: replicate per-module overrides into `[tool.mypy.overrides]` in `pyproject.toml`; run `mypy backend/` pre/post and diff error counts.
- **R8 — `docker compose` v2 not installed everywhere (F-17-005, F-17-006):** Some legacy CI runners still ship v1 only. Mitigation: bump runner image; add a preflight check at script top.
- **R9 — F-15-018 PBKDF2 regression test stays red:** Until Cluster A fixes F-08-001, this test is intentionally failing. Mitigation: mark `xfail` with explicit issue link until A lands; convert to plain pass after.
- **R10 — Deployment script consolidation (F-17-016) breaks an undocumented runbook:** Some operators may be using a now-removed script. Mitigation: keep deprecated wrappers that `exec` the canonical script for one release cycle; document in CHANGELOG.

---

**Final assertion:** This workpaper covers all 37 finding IDs:
F-15-005, F-15-010, F-15-014, F-15-015, F-15-016, F-15-017, F-15-018, F-15-022, F-15-025,
F-16-001, F-16-002, F-16-004, F-16-005, F-16-008, F-16-011, F-16-012, F-16-013, F-16-014, F-16-015,
F-17-003, F-17-004, F-17-005, F-17-006, F-17-008, F-17-010, F-17-011, F-17-012, F-17-013, F-17-014, F-17-015, F-17-016, F-17-017, F-17-018, F-17-019, F-17-020, F-17-021, F-17-022.
