# Audit Context — Six Projects (Pure Context)

Skill: Trail of Bits `audit-context-building`.  
Date: 2026-08-14.  
Mode: **understanding only**. No vulnerability findings, no fixes, no severity, no exploit reasoning.

Excluded from analysis (per operator): `tax-prep-2025/`, `thesis-monitor/`. Those appear only as Case B black boxes where in-scope code calls them.

Secrets (`.env` bodies, `secrets/*.enc`, live DBs, DSN values) were not opened.

---

## 1. What this document is

A stable mental model for a later hunt phase:

1. Who talks to whom
2. What each process actually writes
3. Which invariants the code currently encodes (not prescriptions)
4. Where assumptions pile up (fragility clusters — **not findings**)
5. What is still `Unclear; need to inspect X`

Full per-function micro-analyses live in the Wave A/B/C agent transcripts. This file is the merge + Phase 3 reconstruction.

---

## 2. Cross-project coupling (verified from code)

```
Broker MCP servers  (Case B, outside all six repos)
        |  local JSON / Markdown under $FINANCE_DATA_DIR/raw/<source>/
        v
portfolio-bridge  (no network)
        |  bridge.db + latest.json + exports/
        +---- wheel-analytics reads exports/wheel_positions_schwab.json (advisory)
        +---- IAP script sync_portfolio_from_bridge.py reads latest.json
        |
        v
wheel-analytics
        |  Case B: tax-prep-2025/web/.env.local  (path constant; file not opened)
        |  Case B: Neon tax_advisor.* SELECT / wheel_analytics.* DDL
        |  local: Schwab PDFs + positions CSV

msos-options-monitor
        |  reads POLYGON_API_KEY from IAP .env  (path in config.yaml; file not opened)
        |  Case B: claude -p + Robinhood MCP → staging JSON
        |  Case B: HTTPS api.polygon.io
        |  writes agents-harness-data/msos-options/options_history.db

investment-analysis-platform
        |  FastAPI :8000, optional ML :8001, Celery, Airflow
        |  Case B: AV / Finnhub / Polygon / SEC / yfinance / NewsAPI
        |  scripts also read Neon (Case B) and latest.json (Case A)

market-intel     isolated except HTTPS api.gleif.org (Case B)
efm-workshop     isolated; no I/O in the package
```

**Earlier I thought** IAP `sync_portfolio_from_bridge` wrote ORM positions.  
**Now:** the script POSTs `/api/v1/portfolio/{id}/positions`; that router builds an in-memory Pydantic `Transaction` and does not call `PortfolioRepository.add_position` (`portfolio.py` add/remove handlers; C2 + C8).

**Earlier I thought** live HTTP tokens were HS256 because `settings.ALGORITHM = "HS256"`.  
**Now:** `settings.ALGORITHM` has zero readers. Live mint is RS256 hardcoded in `jwt_manager.create_access_token` (C1).

---

## 3. Seed invariant register (confirm / rewrite / drop)

| ID | Seed claim | Verdict | Citation |
|---|---|---|---|
| PB-1 | package performs no network I/O | **KEEP** | no urllib/requests/httpx in `src/bridge`; I/O is Path + sqlite3 + os.replace |
| PB-2 | Decimal stored as strings | **REWRITE** | ingest is Decimal; SQLite TEXT via `str_dec`; **export JSON uses `float` via `_num`** (`export.py` L70–72) |
| PB-3 | source `ok` ≠ rows > 0 | **KEEP** | `sync.py` L61–72: `"ok"` iff `parse_source` returns |
| PB-4 | Kubera excluded from symbol-keyed exports; still in DB/`latest.json` | **KEEP** (narrow) | `AGGREGATOR_SOURCES=("kubera",)`; still inserted; IAP aggregate does **not** apply this exclusion (C8 I26) |
| PB-5 | returns use aggregator net-worth + dense tail | **REWRITE** | series is `account_value_history` for aggregator sources, not table `net_worth` (`export.py` L183–202) |
| PB-6 | `latest.json` lives beside `db_path` | **KEEP** | `sync.py` L41–42 |
| MS-1 | missed chain day is a permanent OI hole | **REWRITE** | pipeline never backfills vendor OI; SQLite will still accept `--snap-date` write of *live* OI under an old date |
| MS-2 | quote-change is not a detection key | **KEEP** | `run_detection` only: oi_change, volume_oi, block_trade, iv_outlier, data_quality |
| MS-3 | bar pull OI-ranked and capped | **KEEP** | `select_targets` ORDER BY OI DESC LIMIT; fallback is volume-ranked still LIMITed |
| WA-1 | only `lots.engine` may FIFO shares | **KEEP** + note | share FIFO only in `build_stock_lots`; `year_summary._directional_long_call_in_year` is option-contract FIFO |
| WA-2 | no live market-data layer | **KEEP** | no yfinance/httpx under `src/wheel`; unrealized from CSV only |
| WA-3 | dashboard is assembler not second cycle engine | **KEEP** | `build_dashboard_data` calls the four engines |
| MI-1 | unimplemented stages stay pending; CLI exit ≠ market constructed | **KEEP** | seven-stage order; only resolve+enumerate have handlers; EXIT_OK unreachable until all seven complete |
| MI-2 | all adapters go through `http.build_client` | **REWRITE** | convention: `gleif` does not call `build_client`; default `run_topic` injects the client. Enforcement is at construction, not inside adapters |
| IAP-1 | canonical ORM is `unified_models.py` | **KEEP** | README + model files; DataLoader raw SQL is a **second** write path using `stocks.ticker` |
| IAP-2 | `monitoring.py` router is not mounted | **KEEP** | defined at `/api/monitoring`; `main.py` never `include_router`s it |
| IAP-3 | JWT/CSRF/rate-limit differ in development | **REWRITE** | CSRF + HTTP rate-limit omitted iff `os.getenv("ENVIRONMENT","development")=="development"`. Other middleware still runs. `Settings.ENVIRONMENT` defaults to `"production"` — two env stories |

**Added (discovered, not in seed):**

| ID | Statement | Where |
|---|---|---|
| PB-7 | `insert_sync` stamps `sync_runs.status='success'` even if every source is missing/error | `db.py` L273–276 |
| PB-8 | activities are cumulative (no `sync_run_id`); `latest.json` publishes the whole table | `db.py` L204, L334–336 |
| IAP-AUTH-1 | live HTTP identity is `User` ORM via `oauth2.get_current_user`; JWT claims are discarded after decode | C1 I-9 |
| IAP-AUTH-2 | `/refresh` requires a still-valid **access** token; minted refresh JWT is discarded at login | `auth.py` `_issue_access_token`, `refresh_token` |
| IAP-MONEY-1 | the only HTTP persist of fills is `POST /api/v1/trading/orders/{int_pk}` → `add_position` | C2 |
| IAP-MONEY-2 | `POST /portfolio/{uuid}/positions` does not persist | C2 + C8 |
| IAP-ADMIN-1 | admin gate is `User.is_admin` only; `User.role` and JWT `roles` are not consulted | C3 I-3 |
| IAP-ADMIN-2 | no production writer sets `User.is_admin=True` | C3 I-11 |
| IAP-ML-1 | `ml_api_server` :8001 has no auth Depends | C6 |
| IAP-ML-2 | HTTP generate-recommendation paths do not insert `recommendations` | C6 I-NO-WRITE-ON-GENERATE |

---

## 4. Per-project reconstruction

### 4.1 portfolio-bridge

**Purpose.** Offline normalize-and-export hub. Operator runs `python -m bridge`. No HTTP, no scheduler in-repo.

**Actors.** Local operator; upstream file producers (MCP, out of repo); downstream file readers (wheel, IAP scripts, other tools named in `export.py` header).

**Workflow.**

1. Resolve `FINANCE_DATA_DIR` → `raw/`, `bridge.db`, `exports/`, `latest.json` beside DB.
2. For each of `webull, robinhood, snaptrade, ibkr, kubera`: missing/empty dir → `missing`; parse raise → `error`; parse return → `ok` (even if zero rows).
3. `insert_sync` one run (always `'success'` if it returns) → `write_latest_json` (atomic) → `export_all`.

**Trust boundary.** Every JSON/MD cell is Case B. Paths are trusted as locations. No broker credentials in this process.

**State.**

| Store | Writers | Readers in this package |
|---|---|---|
| `bridge.db` | `init_db`, `insert_sync` | `write_latest_json`, all `export_*` |
| `latest.json` | `write_latest_json` | none here |
| `exports/*` | `export_all` + prune | none here |
| `net_worth` table | insert | **no Python reader** |
| `orders` table | insert path | **no parser fills it** |

**Fragility cluster (not a finding).** Filename last-4 account binding; `ok`≠rows; Kubera included in IAP aggregate but excluded from symbol exports; history dates must be `fromisoformat`-compatible or `export_returns_input` raises after DB commit.

---

### 4.2 msos-options-monitor

**Purpose.** Accrue MSOS open interest forward and detect positioning (OI / volume-OI / block / IV), not quote artifacts.

**Actors.** Operator / Task Scheduler; `claude -p` (Case B); Robinhood MCP (Case B); Polygon HTTPS (Case B).

**Workflow (orchestrator `run_daily.ps1`).**

| Leg | When (docs) | Success predicate |
|---|---|---|
| chain | 16:05 ET | **staged file exists**, then ingest exit 0. Claude text is ignored |
| bars | 09:15 ET | `capture_polygon` exit 0 (`partial` → 1) |
| report | 09:45 ET | config load; empty digest is still 0 |
| healthcheck | not in script | separate CLI |

`snap_date` is **local today**, not agent `captured_at`. Same-day recapture is DELETE+INSERT if coverage ≥ 80%.

**OCC join.** Snapshots store 21-char OCC; bars store `O:…`. Only join is `report._compact` (strip spaces + `O:`).

**Trust.** Staging JSON untrusted; Polygon Case B; env file path points at IAP `.env` (not opened). Data dir hardcoded in `run_daily.ps1` and also in `config.yaml` — same string today, not mechanically coupled.

**Fragility cluster.** Dual data-dir sources; two OCC encodings; morning bars rank **yesterday’s** OI; `list_contracts` unused; `price_signal_field` unused; missed weekday is unrecoverable from vendors.

---

### 4.3 wheel-analytics

**Purpose.** Schwab confirmation PDFs → frozen ledger → one share-FIFO engine → cash/tax **estimates** + optional Streamlit + optional Neon schema.

**Actors.** Local operator; Schwab PDFs/CSV (untrusted files treated as oracle); bridge JSON (advisory, 24h freshness); Neon / tax-prep (Case B).

**Workflow.**

```
*/*.PDF → parse_confirmation → fingerprint → dedupe
      → build_stock_lots (only share FIFO)
      → covered_call / pnl / wheel / equity_curve / tax estimate
      → CLI print | trades.csv | summary.xlsx | Streamlit
```

**Case B payloads this repo sends (files not opened):**

- DSN lookup: CLI → `NEON_DSN` → `DATABASE_URL` → `tax-prep-2025/web/.env.local`
- `LedgerReader`: `SET TRANSACTION READ ONLY` + pinned SELECTs on `tax_advisor.accounts|entities|import_runs|transactions`
- Migrator: DDL + `INSERT` into `wheel_analytics.schema_version` only

**Not on the operator CLI path:** `LedgerReader` (1099 compare uses a hardcoded oracle in `cli.py`); QCC module.

**Fragility cluster.** Default filesystem paths to Downloads; `YTD_START` hardcoded 2026-01-01; `cmd_tax` forces `short_term_proceeds=ZERO`; equity curve adds option cash **and** §1234 gain; first promoted wheel cycle receives all underlying premium.

---

### 4.4 market-intel

**Purpose.** Construct a market map from topic + scope. Implemented: RESOLVE + ENUMERATE (GLEIF only).

**Workflow.** `mktintel` CLI → `run_topic` → `execute_run` stage loop. Unhandled stages stay `pending`. Result `partial` → exit 1. `EXIT_OK` only if all seven stages complete.

**Trust.** Topic/scope argv; `MARKET_INTEL_USER_AGENT` required; GLEIF HTTPS Case B; `run.json` on disk is untrusted structure; path construction is a documented security boundary.

**State.** Hub `MARKET_DATA_DIR` or `~/market-data`: per-topic `run.json` + lock + `market.db` + `raw/`; hub-wide `quota.db`.

**Fragility cluster.** `http.py` is the only network path (already adversarially reviewed twice per README — context only). `build/lib` is a stale install missing `seam.py`. Per-record GLEIF malformation is a silent drop.

---

### 4.5 efinancialmodels-workshop

**Purpose (implemented).** Error contract + verify scripts. Not a compiler.

**Present.** `errors.py` (`WorkshopError` + BSD exit codes + `format_error`); `scripts/verify.py`; `scripts/verify_spike.py`; skill stub.

**Absent.** `cli.py` (but `pyproject.toml` declares `efm-workshop = efm_workshop.cli:main`); library; compiler; any write to `EFM_WORKSHOP_HOME`.

**State R/W in package: none.**

---

### 4.6 investment-analysis-platform (Wave C subset)

IAP is too large for whole-tree line-by-line. Wave C covered auth, money, admin, secrets/GDPR, ingest, advice/ML, bootstrap/WS, and cross-repo writers.

#### Actors

| Actor | How they enter |
|---|---|
| Browser SPA | CORS + JWT Bearer on some routes |
| Unauthenticated client | health, root, many stock/market/analysis GETs; `GET /api/v1/metrics`; `GET /agents/capabilities`; WS `/stream` optional; ML :8001 all routes |
| Active user | `get_current_user` (DB row) |
| Admin | `User.is_admin` True (no in-repo writer of that flag) |
| Celery / Airflow / in-process scheduler | privileged batch; no end-user JWT |
| Vendor HTTP | Case B |
| CLI sync scripts | `IAP_USERNAME` / `IAP_PASSWORD` / `FINANCE_DATA_DIR` / Neon DSN |

#### Dual / triple stacks (continuity facts)

| Concern | Stack A | Stack B | Notes |
|---|---|---|---|
| User resolve | `oauth2.get_current_user` → `User` | `utils.auth` dict wrapper | same engine; agents/monitoring use dict |
| JWT mint | `jwt_manager` RS256 + Redis session | `enhanced_auth` HS256 vault secret | HTTP uses A; WS `/stream` handshake uses B |
| CSRF/RL env | `os.getenv("ENVIRONMENT","development")` | `Settings.ENVIRONMENT` default `"production"` | can disagree in one process |
| Fills | `TradingService.execute_trade` → `add_position` | Celery `execute_order` | second unlocked implementation; no HTTP `Order` insert |
| Portfolio id | int `portfolios.id` on trading | UUID `portfolio_id` on `/portfolio/{id}` | must not interchange |
| Recs | in-memory generate | Celery / TradingAgents persist | HTTP generate does not write table |
| Inference | `/api/v1/ml` + `ModelManager` | `ml_api_server` :8001 joblib | two control planes |
| Secrets | `SecretsManager` PBKDF2 salt A | `SecretsVault` salt B | same env name, different ciphertext |
| GDPR erase | `process_deletion` (admin) | `POST /users/me/anonymize` | different field sets |

#### Auth workflow (as coded)

```
register/login
  → username := email
  → bcrypt
  → create_tokens (RS256 access + refresh)
  → HTTP returns access only (refresh discarded)
  → /refresh remints from still-valid access
  → /logout logs, does not revoke
```

Protected HTTP: decode RS256 (iss/aud/type/session if Redis) **or** HS256 fallback without those checks → load `User` by email|username == `sub` → require `is_active`. Admin: `User.is_admin`.

#### Money workflow (as coded)

```
POST /api/v1/trading/orders/{int_pk}   ← only HTTP persist
  → get_current_user (identity only; no ownership)
  → validate_order (SELL qty not checked; MARKET cash ≈ 0)
  → add_position FOR UPDATE
       cash ± qty*price, average-cost lot, Transaction insert
No Order row. No broker. No user_id on the write.
```

`GET /portfolio/summary` may **create** a default portfolio with $10,000 cash.

#### Advice / ML workflow

Generate endpoints assemble in-memory recs (rules-based or engine). `DEMO_MODE` default False refuses fabricators. Table writers: Celery `create_recommendation`, `persist_tradingagents_decision` if `TRADINGAGENTS_PERSIST` truthy (default off). `GET /recommendations/{id}` does not load the table.

`ml_api_server` :8001: predict / load / delete / retrain with no Depends.

#### Bootstrap

Compose/Dockerfile serve `backend.api.main:app` (**not** `socket_app`). Socket.IO wrapper exists at import; `/socket.io/` is 404 under default deploy. CSRF+HTTP RL omitted when getenv ENVIRONMENT is development (this repo `.env`).

---

## 5. End-to-end workflows (cross-repo)

### 5.1 Daily MSOS

Scheduler → `run_daily.ps1 -Leg chain|bars|report` → staging file / Polygon / digest. Healthcheck is a separate process. A weekday without a staging file is logged `OI for $Stamp is LOST`.

### 5.2 Bridge sync → consumers

```
MCP files → python -m bridge
         → latest.json + exports/
         → wheel: load_bridge_positions (24h) / bridge-reconcile
         → IAP: sync_portfolio_from_bridge [--apply] → REST that does not persist
         → IAP: sync_portfolio_from_neon [default apply] → same REST
         → compare_bridge_vs_neon: read-only report
```

### 5.3 IAP user session

Register → JWT access → some routes public, money/settings/gdpr/ml require user → trading fill on int PK → Celery refresh marks. CSRF/RL only if ENVIRONMENT ≠ development.

### 5.4 market-intel topic

UA required → checkpointed stages → GLEIF pages → companies in `market.db` + raw JSON. Partial by design.

### 5.5 EFM

`python scripts/verify.py --all` / `verify_spike.py`. No workshop home writes.

---

## 6. Trust-boundary map

| Boundary | Untrusted input | Privilege after crossing |
|---|---|---|
| MCP JSON / MD on disk | file contents, filenames | become Decimal/TEXT rows |
| Schwab PDF / CSV | geometry + CSV cells | become frozen Trade / Position oracle |
| Staged chain JSON | agent-written OCC/quotes | become `chain_snapshots` if identity fields pass |
| GLEIF / Polygon / AV / Finnhub / SEC | HTTPS JSON/HTML | parsed or None; empty `{}` often truthy |
| Neon / tax-prep `.env.local` | Case B | DSN string or SELECT rows |
| IAP Bearer token | attacker-controlled JWT | DB `User` if decode + lookup succeed |
| IAP trading path param | int PK | write cash/positions **without ownership check** |
| IAP PUT `/settings/api-keys` | any active user | writes `.env` if getenv ENVIRONMENT is development |
| `ml_api_server` | any HTTP client that can reach :8001 | load/predict/delete/retrain |
| `claude -p` | MCP session | file write only if deny-list holds |
| Admin `POST /command` | admin user | service returns a dict; no subprocess on this path |

---

## 7. Complexity / fragility clusters

These are **places a later hunt should start**, not conclusions.

1. **IAP dual JWT + dual ENVIRONMENT + dual user-resolve.** Three `get_current_user`s, two JWT engines, two ENVIRONMENT defaults.
2. **IAP money: three fill stories.** Trading persist / portfolio DTO no-op / Celery unlocked `execute_order` with no creator.
3. **IAP public vs auth split.** Many market/analysis routes have no `Depends`; trading has identity without ownership.
4. **IAP admin boolean with no granter.** `is_admin` is the only gate; no in-tree writer.
5. **IAP two secret stores + three crypto stacks** sharing `MASTER_SECRET_KEY`.
6. **IAP two GDPR erasers** with different field sets; consent is AuditLog append-only, not a table.
7. **IAP two ML control planes** (`/api/v1/ml` vs :8001).
8. **IAP ingest two stacks** (Celery ORM vs ETL DataLoader) writing different stock identity columns (`symbol` vs `ticker`); several client/consumer shape mismatches (AV remap vs store raw keys; FH ISO datetime vs `fromtimestamp`).
9. **Bridge `ok` vs rows vs Kubera aggregator** vs IAP aggregate that does not exclude Kubera.
10. **MSOS dual OCC + dual data-dir + clock authority** (local date vs agent `captured_at` vs UTC bar dates).
11. **Wheel first-cycle premium assignment + hardcoded tax year/oracle.**
12. **Socket.IO constructed but not served; native WS + trigger REST unauthenticated.**

---

## 8. IAP deferred (explicit)

Not line-by-line in this pass:

- Most of `backend/analytics` indicators (except recommendation engine)
- ML training loops / `minimal_training.py` internals
- Airflow DAG bodies beyond credential/operator inventory
- Frontend (`frontend/web/src`) beyond auth URL + Socket.IO event names
- `backend/utils` long tail
- Alembic revision internals
- `infrastructure/` Nginx/Terraform beyond the compose/ASGI and `/api/ws` vs `/api/v1/ws` fact

---

## 9. Open Unclear list (merged, still unresolved)

Must stay “unclear” rather than guessed:

1. Whether any host actually sets `ENVIRONMENT=production` (repo `.env` is development).
2. Whether any out-of-repo unit binds `socket_app` or runs the IAP sync scripts on a schedule.
3. How a live `User.is_admin=True` row is created.
4. Whether Redis is up in each IAP environment (session/blacklist/rate-limit).
5. Live GET `/portfolio/{id}` contract vs missing `get_user_portfolio` / `get_portfolio_positions` on the repository.
6. Whether `Position.version=1` corresponds to a column not on `unified_models.Position`.
7. Who inserts `orders` rows (Celery can fill them; no constructor found).
8. Whether `EncryptedType` / GDPR Fernet are leftover or future columns.
9. Whether SecretsManager `./secrets`, SecretsVault `/app/secrets/vault`, and repo `secrets/*.enc` are one store or three.
10. Typical live kubera `source_status` (ok vs missing) — IAP would double-count if ok.
11. Whether `alpaca_phf` currently has Neon rows (compare added exit 4 after a zero-row day).
12. Whether production sets `TRADINGAGENTS_PERSIST` or `initialize_hybrid_engine`.
13. `run_daily.ps1` data dir vs `config.yaml` staying equal.
14. Process TZ vs ET for MSOS healthcheck hour-17 grace.
15. GLEIF / Polygon / AV live response shapes vs assumed keys (Case B).
16. Whether `format_error` will become the EFM CLI boundary (CLI absent).
17. `LedgerReader` 1099 path vs hardcoded `cmd_tax` oracle — which operators use.
18. Whether IAP `buying_power = 2 * cash` is the book rule (validate uses cash only).

---

## 10. Deliverable index

| Artifact | Role |
|---|---|
| This file | Phase 1+3 merge, invariant register, workflows, trust, clusters |
| Plan | `plan.md` in this session |
| Wave A1 transcript | full bridge micro-analyses |
| Wave A2 transcript | full MSOS micro-analyses + OCC join |
| Wave A3 transcript | market-intel (summary + MI-1/MI-2) |
| Wave A4 transcript | EFM implemented surface |
| Wave B transcript | wheel ingest/money/DB |
| Wave C1–C8 transcripts | IAP auth, money, admin, secrets/GDPR, ingest, ML, bootstrap, writers |

No product-repo files were modified. No vulnerability phase was started.
