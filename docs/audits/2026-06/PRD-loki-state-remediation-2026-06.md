---
prd_id: PRD-LOKI-2026-06-STATE-REMEDIATION
title: Investment Platform — State-Analysis Remediation (excl. secrets)
generated: 2026-06-23
source_analysis: 8-dimension multi-agent state analysis (2026-06-23)
executor: loki-mode (autonomous, RARV cycle)
verification: /verification-loop after EVERY task, phase, and tier (mandatory gate)
scope_excludes: [secrets-remediation]   # Recommendation #1 is handled by Devin out-of-band (rotation, history purge)
read_only_inputs: false                  # loki WILL modify source; this PRD is the source of truth
budget_total_usd: 60
budget_halt_usd: 110                      # ~2x — HALT + drop to Tier 0+1 only
default_model_tier: development           # opus default; fast=sonnet; --allow-haiku NOT set
tiers:
  - id: T0
    name: Deploy-blockers
    blocking: true
    workstreams: [W3-release, W3-migrations]
  - id: T1
    name: ML correctness & safety
    blocking: true
    workstreams: [W2-ml-honesty]
  - id: T2
    name: Truth & guardrails
    blocking: false
    workstreams: [W4-docs, W5-ci-gates]
  - id: T3
    name: Hygiene & consolidation
    blocking: false
    workstreams: [W7-sprawl]
  - id: T4
    name: Regulatory scaffolding (human-gated)
    blocking: false
    workstreams: [W6-regulatory]
sequencing:
  - T0 and T1 may run in parallel (no shared files); both MUST complete before T2 begins
  - T2 after T0+T1 (gate-tightening would otherwise fail on bugs T0/T1 fix)
  - T3 after T2 (consolidation references final canonical files)
  - T4 runs independently from T0 onward (drafting only; human-blocking on registration)
human_decisions: [D1, D2, D3, D4, D5, D6]   # all have recommended defaults — see §2
---

# PRD for Loki — Investment Platform State Remediation (2026-06)

> **TL;DR (5 lines)**
> 1. Source = the 2026-06-23 8-dimension state analysis. This PRD operationalizes recommendations **#2–#7** (secrets #1 is Devin's, out of band).
> 2. **Tier 0** fixes the two bugs that stop a clean release (dev-Dockerfile shipped to prod, crash-looping `cost_monitor`, alembic migration drift #216). **Tier 1** makes ML fail-loud so no random-weight prediction reaches a user.
> 3. **Tier 2** reconciles the lying status docs to one source of truth and tightens CI gates *differentially* (no 3,636-error wall). **Tier 3** kills Dockerfile/compose/ModelManager sprawl. **Tier 4** scaffolds the regulatory docs (registration stays human).
> 4. Every task, phase, and tier ends with a **`/verification-loop` gate** — Python+TS adapted (ruff diff-clean, changed-file mypy, targeted pytest, bandit no-new-HIGH, diff review). A red gate blocks advancement; loki self-heals or dead-letters after 5 tries.
> 5. All six human decisions (§2) ship with a **recommended default + cost of default**, so loki never blocks on Devin — it proceeds on defaults and logs the assumption.

---

## 0. Document Metadata

- **Source:** `/Users/devinmcgrath/projects/investment-analysis-platform` — 8-dimension multi-agent analysis, 2026-06-23. Per-dimension completeness: Backend 82 · Frontend 82 · Testing 72 · Security 72 · Infra 68 · Roadmap 55 · ML/Data 45 · Docs 45.
- **Predecessor:** `docs/audits/2026-04/PRD-for-loki.md` (374-finding remediation; wave-1 merged via #207, #211–#235). This PRD is the **post-wave-1 forward program**, derived from fresh code-level evidence, not a re-issue.
- **Excluded by request:** Recommendation **#1 — secrets remediation** (committed `.env*` files, credential rotation, history purge). Devin owns this. **loki MUST NOT** `git rm` tracked `.env*`, rotate creds, or rewrite history. loki MAY *reference* the secret-exposure risk in docs but takes no action on it.
- **Executor contract:** `loki-mode` autonomous RARV cycle. loki reads `.loki/state/orchestrator.json` + `.loki/queue/pending.json` (both pre-seeded from this PRD). Honors `.loki/PAUSE` and `.loki/STOP`.
- **Hard invariant — tests are sacred:** never delete, skip, or `xfail` a test to make a gate green. Fix the code. (Tier 2 W5 *re-tightens* xfail strictness; it does not loosen it.)

## 0.1 Quickstart for Devin (on return)

- **Watch:** `tail -f .loki/state/orchestrator.json` or the loki dashboard. Progress is also visible as atomic commits on branch `loki/state-remediation-2026-06`.
- **Stop/pause:** `touch .loki/STOP` (graceful end) or `touch .loki/PAUSE` (resume by deleting).
- **Decisions:** loki proceeds on the §2 defaults and logs each in `.loki/decision-log.jsonl`. To override, edit the decision row and `touch .loki/PAUSE` then resume — loki re-reads on resume.
- **Where work lands:** one branch `loki/state-remediation-2026-06`, one commit per task (finding/task ID in subject), PRs opened per-tier (T0, T1, T2, T3, T4) for your review. loki does **not** merge to `main` — merges are human-gated.
- **Budget:** soft cap $60, **HALT at $110** → drop to Tier 0+1 only. Track via `npx @claude-flow/cli@latest hooks metrics --v3-dashboard`.

## 1. Executive Summary

**Mission.** Take the platform from "advanced-beta that lies about being production-ready" to a state where a release deploys cleanly, no user can receive a fabricated ML recommendation, the docs tell the truth, and CI actually gates quality — without touching the secrets workstream Devin owns.

**Effort framing.** ~**42–66 engineer-hours** total across Tiers 0–3 (T4 drafting +6–10h, registration external). loki token cost est. **$30–55**. Tier 0+1 (the deploy-blockers + ML safety) is ~**14–20h** and is the "stop-the-bleeding" cut.

**Sequencing in one line:** `(T0 ∥ T1) → T2 → T3`, with `T4` drafting in parallel throughout.

**Definition of done (program):**
- `docker compose -f docker-compose.production.yml config` resolves with no missing-module service; the release pipeline builds the **nginx** frontend image; `alembic upgrade head` succeeds on a fresh DB.
- The recommendations path returns HTTP 503 `model_unavailable` when no trained weights exist — **never** a random-weight prediction. Zero `random.uniform`/`np.random` fabricators remain on any live request path.
- Exactly one authoritative roadmap/status entry point; every doc claiming "100% / production-ready" is corrected or quarantined.
- CI fails on *new* lint codes and *changed-file* type errors; the 8 previously-ignored test files run in a gating nightly job; `xfail strict` is restored.
- Each tier merged via its own reviewed PR; `main` untouched by loki.

## 2. Decision Log (defaults pre-applied; loki does NOT block)

> Every item has a **default loki executes immediately** and the **cost of that default**. Devin overrides only if desired (see §0.1).

| ID | Question | **Default (loki executes)** | Cost of default / why |
|----|----------|------------------------------|------------------------|
| **D1** | ML: train a real model now, or only make inference fail-loud? | **Fail-loud only.** Remove random fallback; gate recommendations behind "no trained model" → 503. Do **not** attempt a training run. | Recommendations endpoint returns 503 until a real model is trained (separate data-science effort). This immediately removes the liability; training is out of scope for a remediation pass. |
| **D2** | Docs: delete stale roadmap/status/index, or archive-in-place? | **Archive + redirect.** Move stale docs to `docs/_superseded/2026-06/` with a top-of-file STATUS banner; leave a 3-line pointer at the old path → README + this analysis. | Preserves git history (matches existing `SUPERSEDED.md` pattern); zero information loss; grep no longer surfaces "100% complete" as authoritative. |
| **D3** | CI: make mypy/lint blocking globally or differentially? | **Differential.** Gate on *new* ruff codes and *changed-line* mypy only (reuse the repo's differential-lint approach). Keep the 3,636-error mypy baseline non-blocking but frozen (no new errors). | Avoids an unpassable global wall; enforces a ratchet (debt can only shrink). Matches Devin's `differential-lint-validation` standard. |
| **D4** | Dockerfiles: which is canonical? | **Root `Dockerfile.backend` + `Dockerfile.frontend` (nginx) are canonical.** Delete `frontend/web/Dockerfile` (dev), `infrastructure/docker/{backend,frontend}/Dockerfile`, and `*.optimized` after repointing all references. | Removes the 4×/4× drift that caused the dev-server-to-prod bug. One image definition per service. |
| **D5** | Regulatory: engage counsel / self-register Form ADV? | **Draft only.** loki scaffolds Privacy Policy, ToS, and frontend investment-disclaimer components + a Form-ADV decision memo. loki does **NOT** self-register or assert fiduciary status; marks registration **HUMAN-BLOCKING**. | Unblocks the code/content portion now; the legal/regulatory act stays with Devin + counsel. No false compliance claims introduced. |
| **D6** | `cost_monitor` prod service: fix module path or drop the service? | **Fix.** Add a `python -m backend.utils.cost_monitor` entrypoint (`__main__`) wrapping the existing `CostMonitor` class; repoint the compose `command`. | Keeps the cost-guard feature (on-theme for the <$50/mo goal) for ~1h vs. silently dropping observability. |

## 3. Workstreams

> Task fields: **ID · title · files · acceptance · model tier · est · per-task verification gate.** Every task's gate is the §5 `/verification-loop` quick profile scoped to the touched files. Phase/tier gates are the full profile.

### Tier 0 — Deploy-blockers (BLOCKING)

**W3-release — Release pipeline correctness**
- **T0.1** Repoint prod frontend build to nginx Dockerfile.
  - files: `.github/workflows/production-deploy.yml`, `Dockerfile.frontend`, `frontend/web/Dockerfile`
  - acceptance: `production-deploy.yml` `build-production-images` builds the frontend from the **root `Dockerfile.frontend`** (node 20 → nginx static), not the dev Vite server. A `docker build -f Dockerfile.frontend` smoke step succeeds in CI (or is added behind the `docker-build` path) and serves a static `index.html`, not `npm start`.
  - tier: development · est: 1–2h
  - gate: workflow YAML lints (`actionlint` if present); `docker build -f Dockerfile.frontend frontend/web` succeeds locally if Docker available, else static-grep proof that no prod path references the dev Dockerfile.
- **T0.2** Fix the duplicated/overlapping nginx server config in `Dockerfile.frontend` (custom `nginx.conf` + echoed `default.conf` conflict).
  - acceptance: exactly one server block; `nginx -t` passes in the built image (or config-syntax grep proof).
  - tier: fast · est: 0.5h
- **T0.3 (D6)** Make `cost_monitor` runnable.
  - files: `backend/utils/cost_monitor.py` (add `if __name__ == "__main__"` runner), `docker-compose.production.yml`
  - acceptance: `python -m backend.utils.cost_monitor --help` (or a no-arg loop) exits 0 / starts cleanly; compose `command` repointed; `docker compose -f docker-compose.production.yml config` resolves with no nonexistent-module service.
  - tier: development · est: 1h

**W3-migrations — Alembic drift (#216 / #242)**
- **T0.4** Resolve the migration-drift bugs blocking fresh-DB deploy.
  - files: `backend/migrations/versions/*` (the `is_tradeable` typo, `CURRENT_TIMESTAMP` IMMUTABLE-predicate, `stocks(sector)` vs `sector_id` column mismatch noted in #242)
  - acceptance: from an empty database, `alembic upgrade head` completes with exit 0; `alembic check` clean; a downgrade→upgrade round-trip on the touched revisions succeeds.
  - tier: development · est: 2–4h
  - gate: spin disposable postgres (compose `db` service or container), run `alembic upgrade head` then `alembic downgrade -1 && alembic upgrade head` on affected revs.

**Tier 0 exit gate (full `/verification-loop`):** fresh-DB `alembic upgrade head` green · prod compose `config` resolves · frontend prod image builds nginx · all touched-file tests green · `git diff` reviewed · open `T0` PR.

### Tier 1 — ML correctness & safety (BLOCKING)

**W2-ml-honesty**
- **T1.1 (D1)** Make `runtime_models.ModelManager` fail-loud.
  - files: `backend/ml/runtime_models.py` (the bare `except` ~:329 that swallows missing-weights and runs random-init torch)
  - acceptance: when weight artifacts are absent at the load paths, the manager raises a typed `ModelUnavailableError` (no random-init model is ever served). The bare `except` is replaced with specific handling + structured log. A unit test asserts: missing weights ⇒ raises, never returns a prediction.
  - tier: development · est: 2–3h
- **T1.2** Gate the recommendations path on model availability.
  - files: recommendation engine/service + `backend/api/routers/recommendations.py`
  - acceptance: when models are unavailable, the endpoint returns **HTTP 503** with `{"detail":"model_unavailable"}`; an integration test asserts 503 (not a 200 with fabricated numbers).
  - tier: development · est: 2–3h
- **T1.3** Excise dead fabricators.
  - files: `recommendation_crud.py` (dead `random.uniform` mixin per #242), `backend/ml/backtesting.py` (`_get_market_data`/`_get_benchmark_data` `np.random` legacy default when no `data_provider`), `backend/ml/model_manager.py` dummy `_create_dummy_*` returning `np.random`.
  - acceptance: no live request/backtest path can reach an `np.random`/`random.uniform` price/return generator. Either delete, or guard behind an explicit `allow_synthetic=True` test-only flag that defaults False and raises on a live path. Grep proof: zero `np.random`/`random.uniform` reachable from router→service→engine.
  - tier: development · est: 2–3h
- **T1.4** Collapse the dual `ModelManager`.
  - files: `backend/ml/model_manager.py` vs `backend/ml/runtime_models.py`
  - acceptance: one canonical manager; the other removed or reduced to a thin re-export; all importers updated; tests green. (If full merge is risky, mark the non-runtime one `@deprecated` and ensure no live path imports it — log the deferral.)
  - tier: development · est: 2–4h

**Tier 1 exit gate (full):** targeted ML + recommendations tests green · grep proof zero live-path fabricators · 503 integration test green · diff reviewed · open `T1` PR.

### Tier 2 — Truth & guardrails (after T0+T1)

**W4-docs (D2)**
- **T2.1** Establish the single source of truth: a new `docs/STATUS.md` (or repoint `README` "Status") that links README + `docs/CODEMAPS/*` + this analysis as authoritative; one current milestone list.
  - tier: fast · est: 1–2h
- **T2.2** Archive + redirect the liars: move `IMPLEMENTATION_STATUS.md`, `PROJECT_ROADMAP.md`, `IMPLEMENTATION_TRACKER.md`, `DOCUMENTATION_INDEX.md`, and the 25 "100% complete" phase/wave reports to `docs/_superseded/2026-06/`; leave 3-line pointers; add a STATUS banner to each. Update any in-repo links.
  - acceptance: grep for `100% complete|Production-Ready` in `docs/` (excluding `_superseded/`) returns only correctly-qualified statements; `DOCUMENTATION_INDEX.md` router count corrected to 18-registered/19-files or redirected.
  - tier: development · est: 2–3h
- **T2.3** Quarantine process artifacts: move the 57/86 audit/phase/wave/checklist artifacts out of `docs/` root into `docs/_process/` so reference docs are findable. Update `DOCUMENTATION_INDEX`→new STATUS.
  - tier: fast · est: 1–2h

**W5-ci-gates (D3)**
- **T2.4** Differential lint/type gate: add a CI step that fails on **new** ruff codes and **changed-line** mypy errors (baseline frozen, not required-zero). Reuse the repo's differential-lint pattern.
  - acceptance: a PR introducing a new `F401`/type error fails CI; a PR touching only clean lines passes despite the 3,636 baseline.
  - tier: development · est: 2–3h
- **T2.5** Restore `xfail strict`: flip `strict=False`→`strict=True` in pytest config; for each currently-masked bug, either fix it or convert to a tracked `@pytest.mark.skip(reason="ISSUE-…")` with a filed issue — **never** silent-green. List every flipped marker in the PR body.
  - acceptance: `xfail_strict=true` in `pytest.ini`; no `xfail` hides a passing-but-unasserted bug; CI green.
  - tier: development · est: 2–4h
- **T2.6** Un-ignore the big suites in a gating nightly: add a `nightly-full-tests.yml` running the 8 `--ignore`d files (`test_comprehensive_units`, `test_integration_comprehensive`, `test_financial_model_validation`, `test_security_compliance`, `test_performance_load`, `test_database_integration`, …) with services up; failures open/append a rolling issue (reuse the dedupe pattern from #233).
  - acceptance: nightly workflow exists, runs the 8 files, and is red if they fail (not silently skipped).
  - tier: development · est: 2–3h
- **T2.7** Fix the import-time global-state pollution that breaks `--collect-only` (Prometheus `Duplicated timeseries: health_check_status`; module raising `ValueError: CRITICAL SECURITY` at import).
  - acceptance: `pytest --collect-only --noconftest` produces **0** collection errors; targeted ad-hoc runs no longer order-dependent.
  - tier: development · est: 2–4h

**Tier 2 exit gate (full):** docs grep clean · differential gate proven (red on new error, green on clean) · `xfail strict` on · nightly workflow present · `--collect-only` 0 errors · diff reviewed · open `T2` PR.

### Tier 3 — Hygiene & consolidation (after T2) — D4

- **T3.1** Delete redundant Dockerfiles after repointing every reference to the canonical root `Dockerfile.backend` / `Dockerfile.frontend`. Verify TA-Lib sha256 pin survives only in the canonical backend image.
  - tier: development · est: 2–3h
- **T3.2** Consolidate compose sprawl: keep `docker-compose.yml` (dev base) + `docker-compose.production.yml`; fold `performance`/`ml-production`/`e2e` into profiles or document why each stays; remove dead overlaps. No service loses functionality.
  - tier: development · est: 2–4h
- **T3.3** Consolidate the dual auth managers (post-T1 model dedupe pattern): ensure the HS256 `enhanced_auth` path is not reachable for verification on live routes (delegate to canonical RS256), or remove if unused. **Note:** does not touch secrets; this is dead-path removal only. Coordinate with Devin's #201 hardening — log if a live importer remains.
  - tier: development · est: 2–3h

**Tier 3 exit gate (full):** one Dockerfile per service · compose `config` resolves for dev+prod · auth-path grep proof · full touched-area tests green · diff reviewed · open `T3` PR.

### Tier 4 — Regulatory scaffolding (parallel, human-gated) — D5

- **T4.1** Draft `docs/legal/PRIVACY_POLICY.md` and `docs/legal/TERMS_OF_SERVICE.md` from the COMPLIANCE_ACTION_PLAN requirements (clearly marked DRAFT — counsel review required).
  - tier: development · est: 2–3h
- **T4.2** Scaffold frontend investment-disclaimer components and wire them on recommendation/analysis surfaces (close the ~30%→target coverage gap). Content marked DRAFT.
  - tier: development · est: 2–4h
- **T4.3** Write `docs/legal/FORM_ADV_DECISION_MEMO.md` laying out the registration question, options, and a recommended path — **flagged HUMAN-BLOCKING; loki does not decide or assert fiduciary status.**
  - tier: planning · est: 1–2h

**Tier 4 exit gate (full):** drafts present + marked DRAFT/HUMAN-BLOCKING · disclaimer components render in frontend build · no doc asserts actual regulatory compliance · open `T4` PR.

## 4. Sequencing (acyclic)

```
        ┌────────── T0 (deploy-blockers) ─────────┐
start ─►┤                                          ├─► T2 (truth+gates) ─► T3 (hygiene)
        └────────── T1 (ML safety) ───────────────┘
T4 (regulatory drafts) ───────── runs in parallel from start ─────────────────────►
```
- T0 ∥ T1: disjoint file sets (CI/compose/migrations vs. `backend/ml/*` + recommendations). Safe to parallelize (up to loki's 10-agent cap).
- T2 strictly after T0+T1 (gate-tightening on a fixed tree).
- T3 strictly after T2 (consolidation points at final canonical files).
- T4 independent (drafting only).

## 5. The Verification-Loop Gate (MANDATORY — Python+TS profile)

> Run after **every task** (quick, scoped) and after **every phase/tier** (full). This is the `/verification-loop` skill adapted to this repo. A red gate **blocks advancement**; loki enters self-heal (RARV REFLECT→retry), and after 5 failures logs to the dead-letter queue and continues — never deletes/skips tests to force green.

**Quick gate (per task — scope to touched files):**
1. **Build/import:** changed Python imports cleanly (`python -c "import <touched modules>"`); if TS touched, `npm --prefix frontend/web run build` (or `tsc --noEmit`).
2. **Types (differential):** `mypy <changed files>` — zero **new** errors vs. baseline.
3. **Lint (differential):** `ruff check <changed files>` — zero **new** codes; `ruff format --check`. (TS: `npm --prefix frontend/web run lint`.)
4. **Tests (targeted):** run the nearest test module(s) for the change with `--noconftest` + env stubs where the suite needs isolation; green.
5. **Security:** `bandit -r <changed dirs> -lll` — no new HIGH/CRITICAL. (Does **not** touch secrets workstream.)
6. **Diff review:** `git diff --stat` + read the diff for unintended changes / missing error handling. Commit atomically (task ID in subject) on green.

**Full gate (per phase/tier — repo-level):**
- All quick-gate phases at repo scope where feasible, PLUS the **tier-specific exit gate** listed in §3, PLUS `pytest --collect-only --noconftest` shows no *new* collection errors, PLUS the tier PR is opened (not merged).

**Evidence:** loki appends a `verification` record per gate to `.loki/state/verification-log.jsonl` (`{taskId, gate, phase, pass, commands, summary, ts}`) so Devin can audit what was actually run.

## 6. Budget & Safety Guardrails

- **Token budget:** soft $60; **HALT at $110** → cancel in-flight T2/T3/T4, finish only T0+T1, open their PRs, write a HALT report to `.loki/HALT-REPORT.md`. Track via `hooks metrics --v3-dashboard`.
- **Per-task cost:** anything >$5/task → split into smaller mechanical steps or downgrade tier.
- **Branch/merge:** all work on `loki/state-remediation-2026-06`; **loki never merges to `main`** — per-tier PRs are human-gated.
- **Out of bounds (hard stops):** do not touch tracked `.env*` files, do not rotate/print secrets, do not rewrite git history, do not `--force` push, do not delete or `xfail`-away a failing test, do not assert regulatory compliance.
- **Failure policy:** 3 failures → simpler approach; 5 → dead-letter (`.loki/queue/dead-letter.json`) + continue; never leave the tree un-buildable between commits.

## 7. Acceptance (program-level, machine-checkable)

```yaml
done_when:
  T0: ["alembic upgrade head exits 0 on empty DB",
       "docker compose -f docker-compose.production.yml config resolves",
       "production-deploy.yml builds nginx frontend image (not dev)"]
  T1: ["recommendations returns 503 model_unavailable when weights absent",
       "grep: zero np.random/random.uniform reachable on live request path",
       "single canonical ModelManager imported by live code"]
  T2: ["docs grep: no unqualified '100% complete'/'Production-Ready' outside _superseded",
       "CI red on new ruff/type error, green on clean diff",
       "pytest.ini xfail_strict=true",
       "nightly-full-tests.yml runs the 8 previously-ignored files",
       "pytest --collect-only --noconftest: 0 errors"]
  T3: ["one Dockerfile per service; dev variant deleted",
       "compose config resolves dev+prod",
       "no live route reaches HS256 enhanced_auth verifier"]
  T4: ["PRIVACY/ToS/FORM_ADV drafts exist, marked DRAFT/HUMAN-BLOCKING",
       "disclaimer components render in frontend build"]
  program: ["5 tier PRs opened against main, none merged by loki",
            ".loki/state/verification-log.jsonl has a passing full-gate record per tier"]
```
