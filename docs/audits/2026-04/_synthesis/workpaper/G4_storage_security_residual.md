# G4 — Storage / Security / Monitoring Residuals

**Cluster:** G4_storage_security_residual
**Scopes:** 07-database-persistence, 08-auth-security-compliance, 10-monitoring-observability
**Findings:** 41 (excluding clusters A=secret-rotation, B=jwt-auth, C=csp)
**Status:** READY (with 1 SEC item flagged for human ACK)

---

## 1. Cluster Overview

This cluster gathers the residual storage, security, and observability findings that did not fold into the high-profile A/B/C clusters. Three things stand out:

1. **The DB transactional API is currently broken (CRITICAL).** `F-07-002`: `BaseRepository.transaction()` is implemented as `async def` with `yield session` — i.e., an async generator — but is invoked through `execute_with_retry(operation)` which `await`s it. Awaiting the call returns the generator object; the function body never runs. **Every `async with repo.transaction(): ...` block in the codebase is a no-op with no rollback, no commit, no session.** This is a currently-broken correctness defect, not a hardening item, and it is the highest-priority fix in the cluster.
2. **Prometheus is effectively dark.** `F-10-001` + `F-10-003` + `F-10-002` + `F-10-014` together mean that (a) `/metrics` only serves a private custom registry, (b) every other monitoring submodule registers to the *default* registry which is never served, (c) the data-quality collector fails on a missing import every cycle, and (d) the dev Prometheus config scrapes the wrong path. Net effect: zero health/SLA/alert metrics ever leave the process. Dashboards & alerts that look "configured" are silently empty.
3. **Security residuals beyond JWT/CSP/secrets.** Hardcoded KDF salt + low PBKDF2 iters (`F-08-001`), in-process rate limiter that scales linearly with workers (`F-08-006`), single-column RBAC role storage (`F-08-007`), polluted secrets baseline (`F-08-010`), self-defeating gitleaks stopwords (`F-08-011`), file-upload polyglot risk (`F-08-014`), middleware that bypasses itself in tests (`F-08-016`), CODEMAPS drift (`F-08-018`), `auth/__init__.py` shim (`F-08-019`), and **untested SEC fiduciary code (`F-08-020`) — flagged for legal/compliance human ACK before adding tests that codify behavior.**

Cluster theme: things that *appear* to work in CI and dev because their failure modes are silent (no-op transactions, empty metrics, fail-open Redis, swallowed exceptions). Restoring observability of correctness is the dominant value here.

---

## 2. Member Findings (all 41)

### Scope 07 — Database & Persistence (17)
- **F-07-002** *critical bug* — `transaction()` async-generator misuse; every `async with repo.transaction():` is silently a no-op. **HIGHEST PRIORITY.**
- **F-07-003** *critical schema* — Migration 001 references `confidence_score` / `is_tradeable`; real cols are `confidence` / `is_tradable`. CREATE INDEX fails on clean DB.
- **F-07-004** *critical schema* — `optimizations.sql` uses MySQL-style inline `INDEX(...)` inside `CREATE TABLE` for `api_usage_log`; PostgreSQL parse error.
- **F-07-005** *high schema* — Migration `adba55bf7b52` creates `idx_watchlists_stock_id` on a column that lives in `watchlist_items`, not `watchlists`.
- **F-07-006** *high schema* — `add_performance_indexes.sql:18` repeats the phantom-column issue with `IF NOT EXISTS` masking the failure.
- **F-07-007** *high perf* — `Exchange.stocks`, `Sector.stocks`, `Sector.industries`, `Industry.stocks` all `lazy="selectin"`; every lookup eagerly fans out to thousands of stocks.
- **F-07-008** *high bug* — `stock_repository.get_by_sector()` filters on `field='sector'`; column is `sector_id` (FK). Raises `AttributeError`.
- **F-07-009** *high schema* — `Position` ORM lacks `version` column even though `add_row_locking_versions.sql` adds it. Optimistic locking non-functional.
- **F-07-010** *high dead_code* — `backend/models/ml_models.py` (1191 lines of PyTorch/sklearn) is not imported anywhere; pollutes ORM package import graph.
- **F-07-011** *medium quality* — Mutable `default={}` / `default=[...]` on JSON columns at lines 102, 842, 967, 1128 — shared mutable defaults across rows.
- **F-07-012** *medium dead_code* — `price_history_optimized` table created by migration 006 but never referenced by ORM/repo.
- **F-07-013** *medium perf* — `UserSession` missing index on `expires_at`.
- **F-07-014** *medium pattern* — `update()` strips `None` values, preventing intentional `NULL` writes.
- **F-07-015** *medium stale* — `database.py` and `tables.py` are duplicate shims over `unified_models`.
- **F-07-016** *medium drift* — Migration 002 silently swallows TimescaleDB exceptions with bare `except: pass`.
- **F-07-017** *low pattern* — `synchronous_commit = off` in postgres conf without per-table override for financial writes.
- **F-07-018** *low dead_code* — `db_timescale_init.py`, `deadlock_handler.py` zero non-test importers (cross-scope w/ scope 11; delegated).

### Scope 08 — Auth / Security / Compliance (10)
- **F-08-001** *critical security* — Hardcoded KDF salt `b"investment_analysis_salt"` + 100k PBKDF2 iters in `secrets_manager.py`. Re-encryption required.
- **F-08-006** *high security* — Per-process in-memory `RateLimiter` in `oauth2.py`; effective limit = configured × N workers.
- **F-08-007** *high security* — RBAC `assign_role` adds to in-memory set but persists `User.role` (single column); 2nd role overwrites on restart. **Default per handoff §6: declare single-role-only.**
- **F-08-010** *high security* — `.secrets.baseline` polluted with 1500+ NPM lockfile false positives.
- **F-08-011** *high quality* — `.gitleaks.toml` stopwords list contains the exact keywords (`password`, `bearer`, `token`, `api`, `key`, `secret`, `auth`) that should flag secrets.
- **F-08-014** *medium security* — File-upload allowlist permits `.txt` without magic-byte verification; polyglot risk.
- **F-08-016** *medium architecture* — `add_comprehensive_security_middleware` skips many middlewares when `is_testing=True`; test posture diverges from prod.
- **F-08-018** *medium drift* — CODEMAPS lists 7 of 20 security modules; codemap regen needed.
- **F-08-019** *low dead_code* — Empty `backend/auth/__init__.py` in single-file subpackage.
- **F-08-020** *low testing_gap* — **NO tests for `FiduciaryDutyChecker` (SEC-critical). `requires_human_ack: true` per handoff §6 — halt for legal review before encoding behavior in tests.**

### Scope 10 — Monitoring & Observability (14)
- **F-10-001** *critical architecture* — Custom `CollectorRegistry` isolation; `/metrics` serves only metrics_collector's registry. Health/SLA/alert/financial/db_perf metrics all on default registry, never served.
- **F-10-002** *critical bug* — `MimeText`/`MimeMultipart` import typo in `real_time_alerts.py`; correct names are `MIMEText`/`MIMEMultipart`. All email alerts broken.
- **F-10-003** *critical broken_import* — `from backend.monitoring.data_quality_metrics import get_quality_summary` — symbol does not exist; collection cycle silently fails every 10s.
- **F-10-005** *high bug* — `_value._value = ...` private-attr mutation on Counter; breaks atomicity, version-fragile, allows non-monotonic values.
- **F-10-006** *high bug* — Error budget recording rule uses 6h window for 30-day monthly budget; sustained low erosion never trips alerts.
- **F-10-008** *high bug* — `gc_collections.inc(gc.get_count()[i])` — `gc.get_count()` is **not** cumulative; should use `gc.get_stats()`.
- **F-10-009** *high stale_code* — Two divergent `prometheus.yml` (config/monitoring vs infrastructure/monitoring) with different jobs, paths, labels.
- **F-10-010** *high architecture* — `@app.on_event("startup"/"shutdown")` deprecated; removed in FastAPI ≥0.110. Metrics never start.
- **F-10-011** *medium architecture* — Duplicate `HealthStatus` enum in `health_checks.py` and `health_system.py` — `is`/`isinstance` cross-module fail.
- **F-10-012** *medium security* — `sla_tracker` consumes `get_redis()` which fail-opens (cross-scope F-08-008); SLA writes silently sink to no-op when Redis down.
- **F-10-014** *medium quality* — Dev `prometheus.yml` scrapes backend at `/api/metrics`; actual route is `/metrics`. `up{} == 0` permanently.
- **F-10-015** *low drift* — `sla_compliance_percent` Gauge defined twice with different label sets (1 vs 3 labels). Will collide if F-10-001 unifies the registry.
- **F-10-016** *low testing_gap* — `test_monitoring_api.py` calls `/api/health/metrics`; real endpoint is `/metrics`.
- **F-10-017** *low dead_code* — `_setup_elasticsearch()` runs on every `LogAnalysisSystem()` instantiation despite ES being permanently removed.

---

## 3. Sequenced Fix Steps

### Phase 1 — DB transactional bug (FAIL-FIRST, BLOCKING)

**Path verify before any edit:**
```bash
test -f backend/repositories/base.py && \
  sed -n '660,680p' backend/repositories/base.py
```

**Step 1.1 — Write failing test FIRST (`tests/database/test_transactions.py`).** This test must currently FAIL (proving F-07-002 is real) before the fix is written:

```python
import pytest
from sqlalchemy import text

@pytest.mark.asyncio
async def test_transaction_yields_real_session(stock_repo):
    """RED — currently passes-as-no-op; must execute SELECT 1."""
    async with stock_repo.transaction() as session:
        assert session is not None, "transaction() yielded None — async-generator bug"
        result = await session.execute(text("SELECT 1"))
        assert result.scalar() == 1

@pytest.mark.asyncio
async def test_transaction_rolls_back_on_exception(stock_repo, sample_stock_factory):
    """RED — rollback never runs because body never executes."""
    with pytest.raises(RuntimeError):
        async with stock_repo.transaction() as session:
            session.add(sample_stock_factory(symbol="ROLLBK"))
            await session.flush()
            raise RuntimeError("boom")
    found = await stock_repo.get_by_symbol("ROLLBK")
    assert found is None, "rollback failed — row persisted"
```

**Step 1.2 — Run the tests; confirm BOTH fail today.** If they pass on the current code, halt and re-investigate (means another path is in play).

**Step 1.3 — Apply fix.** Refactor `_execute_transaction` to use `asynccontextmanager` correctly. Recommended shape (matches handoff `recommendation`):

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def transaction(self):
    async with self.db_manager.get_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
```
Remove the `execute_with_retry` wrapping for the context-manager path; retries belong inside the wrapped operation, not around the `yield`.

**Step 1.4 — Re-run tests; both must turn GREEN.** Then run the full repository test pack to surface latent rollback bugs (Risk §10).

### Phase 2 — Schema / migration drift (parallel after Phase 1)
Order: F-07-003 → F-07-005 → F-07-006 → F-07-004 → F-07-009 → F-07-008 → F-07-013 (index) → F-07-011 (mutable defaults) → F-07-007 (lazy=select) → F-07-014 (UNSET sentinel) → F-07-016 (log warning) → F-07-012 + F-07-010 + F-07-015 + F-07-017 + F-07-018 (cleanup/doc).

Each migration fix: drop in feature branch, run `alembic upgrade head` against fresh container, then `alembic downgrade -1 && upgrade head` to verify reversibility.

### Phase 3 — Prometheus metrics restoration

3.1 Fix `F-10-002` (import typo) — single-line fix, immediate.
3.2 Fix `F-10-003` (add `get_quality_summary` wrapper).
3.3 Fix `F-10-014` (dev prom path).
3.4 Resolve `F-10-015` label-set divergence BEFORE unifying registries (collision blocker).
3.5 Apply `F-10-001` registry unification (Option (b): drop custom registry, use default; safer once F-10-015 resolved).
3.6 Fix `F-10-005` (Counter→Gauge for cumulative bytes), `F-10-008` (`gc.get_stats()`), `F-10-010` (lifespan), `F-10-011` (dedup HealthStatus), `F-10-009` (delete stale prometheus.yml), `F-10-006` (30d window — coordinate with SRE on tooling), `F-10-012` (cross-scope w/ F-08-008 fail-open), `F-10-016` (test path), `F-10-017` (delete ES setup).

### Phase 4 — Security residuals

4.1 `F-08-002` family — secrets baseline & gitleaks: `F-08-010` (lockfile excludes), `F-08-011` (remove stopwords). Cheap, high-signal.
4.2 `F-08-001` KDF: generate per-install salt, bump iters to ≥600k or migrate to Argon2id, write a one-shot re-encrypt script. **Rotates all stored secrets — coordinate with cluster A (secret-rotation) for combined window.**
4.3 `F-08-006` — delete in-process `RateLimiter`, route through Redis-backed `advanced_rate_limiter`.
4.4 `F-08-007` — **Default decision per handoff §6: declare single-role-only.** Update RBAC to remove the in-memory multi-role set, document the constraint, add a regression test that asserts only one role per user.
4.5 `F-08-014` — magic-byte verification + `Content-Disposition: attachment`.
4.6 `F-08-016` — split into `TestingSecurityProfile` so tests cannot diverge from prod posture.
4.7 `F-08-018`, `F-08-019` — codemap regen + auth shim consolidation.
4.8 **`F-08-020` HALT-AND-FLAG.** Do NOT write tests for `FiduciaryDutyChecker` until legal/compliance signs off on the behavioral spec the tests would lock in. `requires_human_ack: true`.

### Phase 5 — Redacted archives access control (handoff §6 default)

Default applied: **move out of repo**. Cross-tracked under D/F clusters; not re-implemented here. Referenced for completeness.

---

## 4. Files Touched

**Backend repositories / models / migrations:**
- `backend/repositories/base.py` (F-07-002, F-07-014)
- `backend/repositories/stock_repository.py` (F-07-008)
- `backend/models/unified_models.py` (F-07-007, F-07-009, F-07-011, F-07-013)
- `backend/models/database.py`, `backend/models/tables.py` (F-07-015)
- `backend/migrations/versions/001_add_critical_indexes.py` (F-07-003)
- `backend/migrations/versions/002_implement_partitioning.py` (F-07-016)
- `backend/migrations/versions/006_optimize_for_massive_loads.py` (F-07-012)
- `backend/migrations/versions/adba55bf7b52_*.py` (F-07-005)
- `backend/migrations/add_performance_indexes.sql` (F-07-006)
- `backend/migrations/add_row_locking_versions.sql` (F-07-009 alignment)
- `infrastructure/database/optimizations.sql` (F-07-004)
- `infrastructure/postgres/postgresql.conf` (F-07-017)
- `backend/models/ml_models.py` → move to `backend/ml/` (F-07-010)

**Security:**
- `backend/security/secrets_manager.py` (F-08-001)
- `backend/security/rbac.py` (F-08-007)
- `backend/security/security_config.py` (F-08-014, F-08-016)
- `backend/auth/oauth2.py` (F-08-006)
- `backend/auth/__init__.py` (F-08-019)
- `.secrets.baseline`, `.gitleaks.toml` (F-08-010, F-08-011)
- `docs/CODEMAPS/BACKEND.md` (F-08-018)
- *(F-08-020: NO files touched until human ACK)*

**Monitoring:**
- `backend/monitoring/metrics_collector.py` (F-10-001, F-10-005, F-10-008, F-10-010, F-10-015)
- `backend/monitoring/real_time_alerts.py` (F-10-002)
- `backend/monitoring/data_quality_metrics.py` (F-10-003)
- `backend/monitoring/health_checks.py` + `backend/monitoring/health_system.py` (F-10-011)
- `backend/monitoring/sla_tracker.py` (F-10-012)
- `backend/monitoring/log_analysis.py` (F-10-017)
- `infrastructure/monitoring/prometheus.yml` (F-10-014)
- `config/monitoring/prometheus.yml` → DELETE (F-10-009)
- `infrastructure/monitoring/alerts/slo-targets.yml` (F-10-006)
- `backend/tests/test_monitoring_api.py` (F-10-016)

**Tests added (NOT root):**
- `tests/database/test_transactions.py` (F-07-002 RED→GREEN)
- `tests/database/test_repository_filters.py` (F-07-008 regression)
- `tests/database/test_position_versioning.py` (F-07-009 regression)
- `tests/security/test_rbac_single_role.py` (F-08-007 regression)
- `tests/monitoring/test_metrics_endpoint.py` (F-10-001 unification, F-10-014 path)

---

## 5. Acceptance Tests

1. **F-07-002 fail-first proof**:
   ```
   pytest tests/database/test_transactions.py -v
   # On unfixed code: BOTH tests FAIL with AttributeError or assert session is not None.
   # After fix: BOTH pass; rollback test confirms row absent post-rollback.
   ```
2. **Migration sweep**: `alembic upgrade head` against fresh DB exits 0; `\d recommendations`, `\d positions`, `\d watchlists`, `\d api_usage_log` show expected indexes/columns; no phantom indexes.
3. **Prometheus non-empty**:
   ```
   curl -s http://localhost:8000/metrics | grep -E '^(health_check_status|sla_compliance_percent|data_quality_score|alert_)' | wc -l
   # Expect: > 5 lines (was: 0).
   ```
4. **Email alert import**: `python -c "from backend.monitoring.real_time_alerts import AlertSeverity"` exits 0.
5. **No private-attr mutation**: `grep -rn "_value._value" backend/monitoring/` → 0 hits.
6. **Lifespan migration**: `grep -rn "on_event(" backend/monitoring/` → 0 hits.
7. **Single HealthStatus**: `grep -rn "^class HealthStatus" backend/monitoring/` → 1 hit.
8. **Dev Prom path**: scrape config has `metrics_path: /metrics`; `up{job="investment-api"} == 1`.
9. **KDF**: two installs with same `MASTER_SECRET_KEY` produce different Fernet keys; PBKDF2 timing > 250ms.
10. **Rate limiter import sweep**: `grep -rn "from backend.auth.oauth2 import RateLimiter" backend/` → 0 hits.
11. **RBAC single-role regression**: `assign_role` twice → DB shows only the latest role; restart preserves it (F-08-007 default).
12. **Gitleaks signal**: planted dummy AWS key in a tracked file is detected by gitleaks; cleanup baseline < 100 entries.
13. **Lockfile noise gone**: `wc -l .secrets.baseline` < 100.
14. **HALT marker**: status JSON shows `F-08-020.requires_human_ack = true`; no test file `test_sec_fiduciary.py` written until ACK.

---

## 6. Rollback Plan

- Each phase lands as its own PR on a dedicated branch (`g4/phase-1-tx`, `g4/phase-2-schema`, `g4/phase-3-metrics`, `g4/phase-4-sec`). Single `git revert <merge-sha>` undoes any phase.
- **Phase 1 transaction fix**: highest-blast-radius. Hold for ≥48h on staging with full integration test suite + manual portfolio-mutation smoke before prod. Rollback = revert merge; no data migration needed (purely Python).
- **Phase 2 migrations**: every Alembic migration must have a working `downgrade()` that drops the index/column/table it adds. Test `upgrade head && downgrade -1 && upgrade head` in CI.
- **Phase 3 metrics registry unification**: temporarily expose BOTH old `/metrics` endpoint and new `/metrics-v2` for one scrape interval; switch Prometheus when v2 confirmed populated; then delete legacy.
- **Phase 4 KDF rotation**: write all-secrets re-encryption to a new key version with both old + new readable for 24h; flip read-default; then delete old. Rollback = flip read-default back.
- **F-08-007 RBAC**: behavior-preserving for current single-role users (DB already has only one role/user). Rollback = revert RBAC commit.

---

## 7. Dependencies

- **Independent of clusters A (secret-rotation), B (jwt-auth), C (csp).** No file overlaps.
- **Soft-depends on cluster E (test-suite)**: F-07-002 fix may cause previously-excluded `tests/database/*` to run for real (current no-op masked failures). E should un-exclude DB tests *after* G4 phase 1 lands, or G4 phase 1 should land while DB tests are still excluded and re-include them as part of acceptance.
- **F-10-012 cross-references F-08-008 (Redis fail-open)** — handled in cluster A or D depending on routing; G4 only adds the SLA-tracker-side guard.
- **F-08-001 KDF rotation** should be sequenced *with* cluster A's secret-rotation window to amortize the re-encryption pass.
- **F-07-018** dead-code deletion delegated to cluster covering scope-11.

---

## 8. Effort & Cost

| Phase | Findings | Sum hours |
|---|---|---|
| 1 — Tx fix (fail-first) | F-07-002 | 4.0 |
| 2 — Schema/migration drift | F-07-003..F-07-017 (15) | 24.5 |
| 3 — Prometheus restoration | F-10-001..F-10-017 (14) | 16.25 |
| 4 — Security residuals | F-08-001, 006, 007, 010, 011, 014, 016, 018, 019 (9) | 24.0 |
| 5 — SEC fiduciary tests | F-08-020 | (blocked, 4.0 after ACK) |
| **Total in-scope** | **40 actionable + 1 blocked** | **~68.75 h actionable** |

Cost (Sonnet @ $0.003/1K tokens, est. ~50K tokens per finding for plan+code+test+review): ~$6 in agent cost; rounding error vs engineer time.

---

## 9. Loki-Actionable Status

- **Loki-actionable (35/41):** F-07-002, F-07-003, F-07-004, F-07-005, F-07-006, F-07-007, F-07-008, F-07-009, F-07-010, F-07-011, F-07-012, F-07-013, F-07-014, F-07-015, F-07-016, F-07-018, F-08-001, F-08-006, F-08-010, F-08-011, F-08-014, F-08-018, F-08-020*, F-10-001, F-10-002, F-10-003, F-10-005, F-10-008, F-10-009, F-10-010, F-10-011, F-10-014, F-10-015, F-10-016, F-10-017
- **NOT loki-actionable / human review (6/41):** F-07-017 (pg durability tradeoff — needs human risk decision), F-08-007 (single-vs-multi-role policy decision), F-08-016 (testing-profile architecture), F-08-019 (package-shape preference), F-10-006 (SLO tooling selection), F-10-012 (depends on F-08-008 fix)
- **`requires_human_ack: true` (1):** **F-08-020** — SEC FiduciaryDutyChecker untested. Halt: legal/compliance must spec required behavior before tests are written, otherwise tests will encode whatever current code does, including bugs, as "correct."

\*F-08-020 is technically loki-actionable for *adding tests*, but content is gated.

---

## 10. Risks

1. **F-07-002 fix surfaces latent rollback bugs.** Today every `repo.transaction()` is a no-op — including failures that *should* roll back. Real transaction semantics will expose any code paths that relied on accidental persistence (e.g., expecting a partial write to "stick" after a logical exception). **Mitigation:** Phase 1 lands first, alone, on a long-lived staging soak; full integration test pack must run; portfolio-mutation paths should get manual smoke tests before prod cut.
2. **Prometheus metrics fix may flood dashboards / page on-call.** Once F-10-001 unifies registries, alerts that have been silently green for months will start firing on real data. SLO-error-budget alerts (F-10-006), `ServiceDown` (F-10-014), and health-check status alerts will all suddenly evaluate against actual series. **Mitigation:** silence Alertmanager during the cutover; review alert thresholds against first 24h of real data; keep on-call manually informed.
3. **F-07-007 lazy-loading change can break async paths.** Switching `lazy="selectin"` → `lazy="select"` re-introduces `MissingGreenlet` if any caller touches `.stocks` outside an async session. **Mitigation:** prefer `lazy="raise"` first to surface every offender, fix call sites with explicit `selectinload()`, then optionally relax to `lazy="select"`.
4. **F-08-001 KDF rotation is destructive if mis-sequenced.** Forgetting any stored ciphertext during re-encrypt = permanent data loss. **Mitigation:** dual-read window, exhaustive enumeration of stored secrets, dry-run report before flip.
5. **F-08-007 single-role default is a policy choice.** If the product later wants multi-role, this commits to a schema migration. **Mitigation:** document the choice in ADR; design the join-table path so it can be retrofitted without data loss.
6. **F-08-020 SEC test gap is now a known/recorded risk.** Until human ACK, the gap is acknowledged in writing and in the status JSON. Do not silently close it.

---

*All 41 findings referenced. SEC item F-08-020 halted per handoff §6.*
