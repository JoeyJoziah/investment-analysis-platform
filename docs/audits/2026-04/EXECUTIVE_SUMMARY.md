# Executive Summary — Production Code Audit 2026-04

**Bottom line:** the codebase has **48 critical findings** spanning **15 of 18 scopes**, several of which mean core product features are silently broken in production right now (recommendations are random, real-time prices never arrive, every login potentially crashes, every risk decision uses the wrong report). There is also a **persistent secret-rotation incident** — production passwords flagged in January 2026 are still hardcoded in 14+ scripts, a migration file, and committed docs as of today.

If you have 30 minutes, read sections **1**, **2**, and **5** below and stop.

---

## 1. Top 10 Cross-Codebase Critical Findings

Ranked by blast radius × business impact, not by scope.

| # | Finding | Where | Why it's #N |
|---|---|---|---|
| 1 | **Production credentials still committed in repo + 14 scripts + alembic.ini** | F-08-009 / F-07-001 / F-17-001 / F-05-003 | DB password, Redis pwd, ES pwd, Grafana pwd, JWT 64-char secrets, Fernet keys all in plaintext. SECRET_ROTATION_PLAN dated 2026-01-27 was never executed. Single biggest blast radius; touches scopes 05/07/08/16/17/18. |
| 2 | **Every investment recommendation in production is random** | F-03-003 | `ml_models/` has only JSON configs — no trained binaries. System silently falls back to `DummyLSTM`/`DummyXGBoost`/`DummyProphet` returning random values. SEC-relevant for a regulated investment platform. |
| 3 | **`random.uniform()` faking data in production endpoints** | F-02-003 | `admin_service`, `recommendation_service` (backtest, perf, alerts), `portfolio_rebalancing`, `socketio_service._stream_price_updates` all return random values to authenticated users. Correctness + SEC compliance hazard. |
| 4 | **JWT login crashes on every request** | F-01-001 + F-08-002 + F-08-005 | `auth.py` uses RS256 with a plain string key (RS256 requires RSA key objects). Combined with ephemeral `JWT_SECRET_KEY` fallback when env unset and HS256 algorithm-confusion fallback path. Login may silently degrade or fail outright. |
| 5 | **TradingAgents risk decisions use the wrong report** | F-04-002 | `risk_manager.py:14` assigns `state["news_report"]` to local `fundamentals_report`. Every risk-management decision uses news in place of fundamentals since this code was written. One-line fix. |
| 6 | **Every transactional DB write is silently a no-op** | F-07-002 | `AsyncBaseRepository.transaction()` is an async generator passed to `await`, which returns the generator object without executing the body. All `async with repo.transaction():` blocks silently do nothing. |
| 7 | **Real-time price feed never works** | F-02-002 | `FinnhubWebSocketClient.connect()` opens `aiohttp.ClientSession` in `async with`, creates background task, returns from inside the block → session closes → feed never receives data. |
| 8 | **Production Docker build broken** | F-13-* | `docker-compose.production.yml` builds reference `target: runtime` stage that does not exist in root `Dockerfile.backend`. Production image cannot be built without manual intervention. |
| 9 | **Zero metrics actually reach Prometheus** | F-10-001 + F-10-003 | `metrics_collector.py` uses a private registry but other modules register to default — only the empty private registry is served. Plus `get_quality_summary` ImportError silently swallowed every 10s. Health/SLA dashboards are empty. |
| 10 | **CI shell injection via untrusted issue body** | F-14-* | 4 GitHub Actions workflows interpolate `${{ github.event.issue.title/body }}` into `run:` blocks. Anyone who can open an issue can RCE the CI runner. Plus TA-Lib downloaded over plaintext HTTP without checksum across 7 workflows. |

**Honorable mentions** (each could be #11): broken token revocation (F-08-004 — logout doesn't actually log out), CSP `unsafe-inline 'unsafe-eval'` app-wide (F-08-003), `OptimizedRecommendationEngine` calls non-existent method (F-09-002), 1234-LOC service duplicates ~2,500 LOC across mixins (F-02-001), Airflow 1.x→2.x import breaks ML training DAG entirely (F-06-*), monitoring router never registered → 5 endpoints unreachable (F-01-002).

## 2. Severity Heatmap

| Scope | Crit | High | Med | Low | Total | Risk band |
|---|---:|---:|---:|---:|---:|---|
| 02-services-domain | **4** | 7 | 10 | 4 | 25 | 🔴 hot |
| 07-database | **4** | 6 | 6 | 2 | 18 | 🔴 hot |
| 08-auth-security | **4** | 8 | 6 | 2 | 20 | 🔴 hot |
| 15-test-suite | **4** | 9 | 9 | 5 | 27 | 🔴 hot |
| 01-backend-api | 3 | 6 | 8 | 3 | 20 | 🟠 warm |
| 03-ml-engine | 3 | 5 | 6 | 3 | 17 | 🟠 warm |
| 05-data-ingestion-etl | 3 | 6 | 7 | 4 | 20 | 🟠 warm |
| 06-airflow | 3 | 6 | 5 | 2 | 16 | 🟠 warm |
| 09-analytics | 3 | 6 | 8 | 4 | 21 | 🟠 warm |
| 10-monitoring | 3 | 7 | 5 | 2 | 17 | 🟠 warm |
| 12-frontend | 3 | 7 | 8 | 4 | 22 | 🟠 warm |
| 04-trading-agents | 2 | 7 | 9 | 4 | 22 | 🟡 watch |
| 14-ci-cd | 2 | 4 | 5 | 3 | 14 | 🟡 watch |
| 16-config-secrets | 2 | 4 | 6 | 3 | 15 | 🟡 watch |
| 17-scripts-tooling | 2 | 7 | 8 | 5 | 22 | 🟡 watch |
| 18-docs-health | 2 | 8 | 18 | 10 | 38 | 🟡 watch |
| 13-infra-deployment | 1 | 7 | 8 | 4 | 20 | 🟢 ok-ish |
| 11-backend-utils-shared | 0 | 4 | 11 | 5 | 20 | 🟢 ok-ish |
| **TOTAL** | **48** | **114** | **143** | **69** | **374** | |

## 3. Read These 5 Reports First

If you don't have time for all 18:

1. **[08-auth-security-compliance.md](reports/08-auth-security-compliance.md)** — covers credential rotation incident, JWT/CSP/secret-management architecture
2. **[02-backend-services-domain.md](reports/02-backend-services-domain.md)** — biggest correctness issues + the random-data-in-production findings
3. **[03-ml-engine.md](reports/03-ml-engine.md)** — random recommendations + ML API exposed without auth
4. **[07-database-persistence.md](reports/07-database-persistence.md)** — silently broken transactions + schema drift
5. **[15-test-suite.md](reports/15-test-suite.md)** — explains why these bugs survived: huge test exclusions, mocked-over-real-bug patterns, broken frontend test runner

## 4. Cross-Scope Clusters

Findings that cannot be remediated in isolation — synthesis swarm should plan these as coordinated change-sets:

- **Secret-rotation cluster** (F-08-009, F-07-001, F-17-001, F-05-003, F-16-*) — single rotation event must touch alembic.ini, 14+ scripts, .env templates, 4+ doc files, then `git filter-repo` to purge history.
- **JWT/auth cluster** (F-01-001, F-08-002, F-08-004, F-08-005) — fix all together; partial fix breaks login worse.
- **CSP cluster** (F-08-003, F-12-003, F-13-*) — app code, frontend Vite build, edge nginx all need `unsafe-inline`/`unsafe-eval` removed in lockstep with a nonce strategy.
- **Random-data cluster** (F-02-003, F-03-003) — product/policy decision required: ship "no recommendation available" rather than a fake one.
- **Test exclusion cluster** (F-15-003, F-15-* mocked-over-real-bug) — un-excluding the security/database test directories will surface dozens of failures from already-known bugs in scopes 02/07/08; fix order matters.
- **Frontend↔backend contract cluster** (F-12-001 `/api/*` vs `/api/v1/*`, F-12-002 wrong response field, F-01-* deprecation middleware misfires) — single coordinated refactor of API client + versioning middleware.

## 5. Aggregate Effort Estimate

Rough sum of agent effort estimates: **~110–135 working days** of remediation work, depending on how aggressively cross-scope clusters can be parallelized. This is the synthesis swarm's job to refine into a sequenced PRD.

## 6. What's NOT in This Audit

Surfaced explicitly so the synthesis swarm doesn't try to fill gaps:

- No live system testing (read-only audit; findings are from code/config inspection + test runs).
- No load/benchmark verification of performance claims (referenced existing PERFORMANCE_BENCHMARKS.md as input).
- No legal review of SEC compliance posture (references compliance code; `FiduciaryDutyChecker` advisor-registration status remains a legal Q).
- No git history mining for past incidents (out of scope).
- 11 of 20 backend security modules sampled rather than fully read (scope 08 — by risk score).
- 5 of 39 backend domain files only partially read (scope 02 — by token budget).
