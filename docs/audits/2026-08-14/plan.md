# Audit Context Building — Six-Project Pure-Context Pass

## Goal

Build a stable, evidence-backed mental model of six local finance/code projects **before** any vulnerability hunt. Output is orientation + per-function micro-analysis + global invariants/workflows/trust boundaries. This is **pure context**. No findings, no fixes, no PoCs, no severity.

Skill: Trail of Bits `audit-context-building` (ultra-granular, line-by-line, First Principles / 5 Whys / 5 Hows).

## Explicit exclusions

| Path | Reason |
|---|---|
| `tax-prep-2025/` | User excluded. Treat as **Case B black box** wherever another repo reads it. |
| `thesis-monitor/` | User excluded (other session). Do not open or analyze. |
| IAP `frontend/web/node_modules`, archives, `*.pyc`, logs, datasets | Noise / not live source |
| Live secrets, `.env` bodies, `secrets/*.enc` contents, live DBs | Do not open. Model as named secrets/state, not values. |

## Targets (in scope)

1. `investment-analysis-platform` — large FastAPI + Celery + ML product
2. `portfolio-bridge` — offline MCP-JSON → SQLite hub (~1.8k LOC)
3. `msos-options-monitor` — OI accrual + Polygon bars + detection
4. `wheel-analytics` — Schwab confirmations → lots / tax estimates / dashboard
5. `market-intel` — market-construction CLI (RESOLVE + GLEIF ENUMERATE only)
6. `efinancialmodels-workshop` — skeleton (errors + verify scripts; no compiler yet)

## Non-goals

- Vulnerability identification, classification, or severity
- Fix recommendations
- Exploit / PoC reasoning
- Editing product code (this plan is context-only; execution writes **analysis notes**, not refactors)
- Analyzing excluded repos even when they appear as DSN/file dependencies

---

## Phase 1 already completed (orientation)

Bottom-up scans of all six trees are done. The facts below are anchors. Phase 2 must cite line numbers and **update the model** if a later read contradicts them.

### Cross-project couplings (do not reset context at repo borders)

```
[Broker MCP servers — outside all six repos]
        | files
        v
[portfolio-bridge] --exports--> wheel_positions_schwab.json
        |                       portfolio_input*.json
        |                       latest.json / bridge.db
        +-------- IAP script sync_portfolio_from_bridge.py
        |
        v
[wheel-analytics] --reads--> bridge export (advisory)
                  --reads--> tax-prep-2025/.env.local  (EXCLUDED / Case B)
                  --reads--> tax_advisor.* on Neon     (EXCLUDED / Case B)
                  --writes--> wheel_analytics.* (optional)

[msos-options-monitor]
        --reads POLYGON_API_KEY from IAP .env   (path coupling)
        --claude -p + Robinhood MCP             (Case B)
        --HTTPS api.polygon.io                  (Case B)
        --writes agents-harness-data/msos-options/options_history.db

[market-intel] --HTTPS api.gleif.org (Case B); hub ~/market-data
[efm-workshop] --no I/O yet; isolated
[IAP] --HTTPS vendors, JWT users, Celery/Airflow, optional ML :8001
```

### Per-target Phase 1 snapshot

**investment-analysis-platform**
- Purpose: FastAPI investment analysis / recs / portfolio / WS for 6k+ tickers.
- Entrypoints: 18 mounted routers in `backend/api/main.py` (~L361–385); Socket.IO wrap `socket_app` L419 vs compose using `app` (unresolved); separate `ml_api_server.py` :8001; Celery beat; Airflow DAGs; CLI sync scripts.
- Actors: end user (`free_user`), admin, unauthenticated GETs, workers, vendors (AV/Finnhub/Polygon/FMP/News/SEC), Slack/Sentry outbound. Stripe path named in CSRF exempt list; **no webhook router found**.
- State: Postgres/Timescale (`unified_models.py`), Redis, `.env` keys, encrypted `secrets/`.
- Dual stacks to resolve in Phase 2: `auth/oauth2.py` vs `security/enhanced_auth.py`; `settings.ALGORITHM=HS256` vs `security_config`; tokenUrl `/api/auth/token` vs mounted `/api/v1/auth/token`.

**portfolio-bridge**
- Offline only. `python -m bridge` → `sync.run` → parse five sources → `insert_sync` → `latest.json` → `export_all`.
- State: `$FINANCE_DATA_DIR/bridge.db` schema v2; Decimal-as-TEXT; Kubera excluded from symbol-keyed exports.
- Documented invariant: parser `ok` ≠ non-zero rows; filename `as_of` two prefix forms; `run()` isolates `latest.json` beside `db_path`.

**msos-options-monitor**
- Three legs: Claude+RH MCP chain → Polygon bars → detect/report. No HTTP server.
- DB: `options_history.db` under `agents-harness-data`. Two OCC encodings (21-char vs `O:` ticker) meet at report.
- Key: Polygon key read from IAP `.env`. `list_contracts` defined, unused. `write_digest` in YAML unused.

**wheel-analytics**
- CLI + optional Streamlit + optional Neon migrator. Sole FIFO: `wheel.lots.engine.build_stock_lots`.
- Default paths point at Schwab Downloads + hardcoded TY2025 1099-B oracle in `cli.py`.
- Tax-prep / Neon: Case B. Do not open excluded repo; model `find_neon_dsn` and `LedgerReader` as callers of hostile/unknown SQL endpoints.

**market-intel**
- `mktintel` CLI; stages 1–2 only. Single HTTP path `http.py`. GLEIF only.
- Hub: `MARKET_DATA_DIR` or `~/market-data`. `build/lib` is a stale install copy missing `seam.py` — Phase 2 uses `src/mktintel` as truth.
- README records two failed adversarial reviews of `http.py` at 100% coverage. Context-build that module fully; do **not** convert those notes into findings.

**efinancialmodels-workshop**
- Implemented: `errors.py` + `__version__` + `scripts/verify.py` + skill stub.
- Declared `efm-workshop` console script points at missing `cli.py`. No DB writes.

---

## Phase 2 — Ultra-granular function analysis

Every analyzed function uses the skill template:

1. Purpose (2–3 sentences)
2. Inputs & Assumptions (type / source / trust; ≥5 assumptions across the writeup)
3. Outputs & Effects (returns, writes, calls, events, postconditions)
4. Block-by-block: What / Why here / Assumptions / Depends on / First Principles or 5 Whys/Hows
5. Cross-function dependencies + external-call risk considerations (≥3 when any external exists)

Quality bar per function: ≥3 invariants, ≥5 assumptions, ≥1 First Principles, ≥3 combined 5 Whys/Hows, line citations (`L45`), no “probably”. Unclear items stay `Unclear; need to inspect X` until resolved.

**Jump-into-callee rule:** internal calls continue in the same flow. External code in another **in-scope** repo is Case A (jump). `tax-prep-2025`, broker MCP servers, Polygon, GLEIF, Neon, `claude` CLI are Case B (adversarial outcomes: revert / wrong return / unexpected state / misbehavior).

### Wave A — complete small trees in parallel (full `src/`)

Spawn four isolated context-builders at once. Each returns micro-analyses + a project-local invariant list.

| Agent | Scope | Must-cover functions |
|---|---|---|
| A1 portfolio-bridge | all `src/bridge/**` (~1.8k LOC) | `sync.run/main`, `parse_source`, `_common` helpers, five `parse()`, `occ.*`, `db.init_db/insert_sync/write_latest_json`, `_add_missing_foreign_keys`, `export_*` + `_dense_tail` / `_periods_per_year` / prune |
| A2 msos-options-monitor | all `src/msos_monitor/**` + `scripts/run_daily.ps1` + `prompts/capture_chain.md` | `config.load/polygon_key`, `db.connect/start_run/finish_run`, `ingest_chain.*`, `PolygonClient._get/daily_bars`, `capture/select_targets`, `detect.*`, `report.build/persist`, `healthcheck.check`, orchestrator deny-list + staged-file success criterion |
| A3 market-intel | `src/mktintel/**` only (not `build/lib`) | `__main__.main`, `run_topic`, `execute_run`, `GuardedTransport.handle_request`, `redact/*`, `QuotaGuard.consume/refund`, `gleif.fetch_*`, `seam.*` writers, `run.save_run/load_run`, `paths.slugify/_safe_ident` |
| A4 efm-workshop | entire `src/` + `scripts/verify.py` + `scripts/verify_spike.py` | `WorkshopError` / `format_error`; verify scanners; spike openpyxl flags. Document planned-but-absent CLI/DB as **not present** (do not analyze design docs as if they were code). |

### Wave B — wheel-analytics (one agent, full money/ingest/DB surface)

All Phase 1 target functions: confirmation parser + fingerprint/dedupe/invariants; `build_stock_lots`; reconcile + bridge freshness; covered-call / pnl / wheel / equity-curve; tax `options_1234` / `year_summary` / `washsale_1091` / `estimate_tax`; `find_neon_dsn` + migration runner + `LedgerReader` (Case B to excluded tax-prep/Neon); CLI export writers; dashboard assembler (no DSN import — verify).

Hard constraints to treat as **candidate invariants** until the code confirms them:
- No second FIFO loop outside `lots.engine`
- Dashboard does not reimplement `build_wheel_report` / scorecard / equity-curve
- Money is `Decimal`
- Tax output labeled estimate, not advice

### Wave C — IAP (prioritized, not whole-tree)

IAP is too large for “every function” in one execution. Phase 2 covers the **trust-sensitive 25** from Phase 1, grouped into parallel sub-agents, then a merge pass.

| C-agent | Files | Why first |
|---|---|---|
| C1 Auth identity | `security/jwt_manager.py`, `auth/oauth2.py`, `api/routers/auth.py`, `security/enhanced_auth.py`, `utils/auth.py` | Two user-resolution stacks; token mint/verify; register/login/refresh |
| C2 Money / trading | `services/trading_service.py`, `api/routers/trading.py`, `services/portfolio_service.py`, `portfolio_rebalancing.py`, `api/routers/portfolio.py` | Orders, cash, positions, rebalance |
| C3 Privileged control | `api/routers/admin.py` (`execute_system_command`), `services/admin_service.py`, `api/routers/settings.py` (api-keys → `.env`), `security/rbac.py` | Admin command plane; key write |
| C4 Secrets / crypto / GDPR | `secrets_manager.py`, `secrets_vault.py`, `data_encryption.py`, `routers/gdpr.py`, `compliance/*` | Key material + erase/export |
| C5 Ingest / vendors | `data_ingestion/*` clients, `tasks/data_tasks.py`, `etl/etl_orchestrator.py` + scrapers | Untrusted vendor JSON |
| C6 Advice / ML / agents | `recommendation_engine.py`, `recommendation_service.py`, `compliance/sec.py`, `routers/ml.py`, `ml_api_server.py`, `agents_service.py` | Recs + second HTTP port + LLM spend |
| C7 Realtime + CSRF + bootstrap | `routers/websocket.py`, `websocket_security.py`, `socketio_service.py`, `csrf_protection.py`, `api/main.py` lifespan/middleware, compose vs `socket_app` | Session/auth on WS; what is actually mounted |
| C8 Cross-repo writers | `scripts/sync_portfolio_from_bridge.py`, `sync_portfolio_from_neon.py`, `tradingagents_bridge/persistence.py` | IAP as consumer of bridge/Neon |

**IAP deferred (say so explicitly in the Phase 3 note):** most of `backend/analytics` indicators, ML training loops, Airflow DAG bodies beyond credential/operator inventory, frontend, `backend/utils` long tail, Alembic revision internals.

If a C-agent hits an internal callee, it **jumps** (e.g. `get_current_user` → oauth2 → jwt_manager) rather than stopping at the router.

### Orchestrator merge after each wave

- Fold callee analyses into caller flows (continuity rule).
- Record assumption corrections as `Earlier I thought X; now Y (file:line)`.
- Maintain a running **global invariant register** (below).

---

## Phase 3 — Global system understanding

After Waves A–C, reconstruct **per project** then **across the six**:

1. **State & invariant reconstruction** — read/write map for every durable store (`bridge.db`, `options_history.db`, IAP `unified_models`, wheel in-memory lots, `market.db` / `quota.db`).
2. **Workflow reconstruction** — end-to-end:
   - Daily MSOS chain/bars/report
   - Bridge sync → exports → wheel `bridge-reconcile` / IAP `sync_portfolio_from_bridge`
   - IAP register → JWT → portfolio/order → Celery refresh
   - market-intel topic run (partial by design)
   - EFM: currently verify-only
3. **Trust boundary map** — actor → entrypoint → behavior. Mark untrusted file ingest (MCP JSON, Schwab PDFs, staged chain JSON, GLEIF pages) vs privileged workers vs Case B remotes.
4. **Complexity / fragility clusters** (for a later hunt phase, **not** findings): high-assumption functions, dual auth stacks, dual OCC encodings, aggregator double-count rules, IAP public-vs-auth route split, `http.py` as the only market-intel network path.

### Seed invariant register (confirm or revise in Phase 2)

These are **hypotheses from docs/Phase 1**. Phase 2 must keep, rewrite, or drop each with a citation.

| ID | Claim | Source of claim |
|---|---|---|
| PB-1 | This package performs no network I/O | README |
| PB-2 | Money/qty are `Decimal`, stored as strings | README / schema |
| PB-3 | Source `ok` does not imply rows > 0 | README |
| PB-4 | Kubera excluded from symbol-keyed exports; still in DB/`latest.json` | `export.AGGREGATOR_SOURCES` |
| PB-5 | Returns use aggregator net-worth series alone + dense tail | README / `export.py` |
| PB-6 | `latest.json` lives beside `db_path`, not `raw_root.parent` | README |
| MS-1 | OI history cannot be backfilled; missed chain day is a permanent hole | README |
| MS-2 | Quote-change is not a detection key; OI/volume/block/IV are | `detect.py` |
| MS-3 | Bar pull is OI-ranked, capped by `max_contracts_per_run` | README / capture |
| WA-1 | Only `lots.engine` may FIFO shares | phase2-design.md |
| WA-2 | No live market-data layer; CSV/bridge labeled as such | phase2-design.md |
| WA-3 | Dashboard is assembler, not a second cycle engine | phase2-design.md |
| MI-1 | Unimplemented stages stay `pending`; CLI exit ≠ market constructed | README |
| MI-2 | All adapters go through `http.build_client` | `sources/__init__.py` |
| IAP-1 | Canonical ORM is `unified_models.py` | README |
| IAP-2 | `monitoring.py` router is not mounted | `main.py` / README |
| IAP-3 | JWT/CSRF/rate-limit behavior differs in `development` | `main.py` comments |

---

## Deliverable

A single standing context document (session-produced, not a product-repo commit unless you ask) containing:

1. Phase 1 maps (condensed from this plan)
2. Per-function micro-analyses (skill format) for Waves A–C
3. Call-chain continuity notes (bridge → wheel / IAP; IAP `.env` → MSOS; wheel → excluded tax-prep)
4. Phase 3: state maps, workflows, trust boundaries, fragility clusters
5. Open `Unclear; need to inspect X` list (should be empty for in-scope functions; leftover = deferred IAP tail)

No code changes to the six products. No secret values in the writeup.

## Execution order after you confirm

1. Launch Wave A (4 parallel context agents).
2. Launch Wave B (wheel) in parallel with Wave A.
3. Launch Wave C (8 IAP agents) in parallel with A/B.
4. Orchestrator merge + Phase 3 synthesis.
5. Stop. Do **not** start a vulnerability phase unless you explicitly ask.

## Risk to the quality bar (stated up front)

IAP full-tree line-by-line is not achievable in one pass. Wave C is the honest subset. Small repos get complete coverage. If you want IAP frontend or Airflow DAG bodies in the same pass, say so; they are currently deferred.
