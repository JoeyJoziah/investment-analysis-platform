# G1_backend — Backend Residual Cluster Workpaper

**Cluster:** G1_backend (residual catch-all)
**Scopes covered:** 01-backend-api, 02-backend-services-domain, 11-backend-utils-shared
**Assigned findings:** 46
**Generated:** 2026-04-27

---

## 1. Cluster overview

G1 is the residual backend cluster: every backend finding NOT already absorbed into the topical clusters (A secret-rotation, B jwt-auth, C csp, D random-data-recommendations, E test-exclusion, F frontend-backend-contract). The 46 findings span FastAPI routers, service-layer logic, async task plumbing, and shared `backend/utils/` modules. They naturally decompose into six sub-themes:

- **T1 — Router & middleware wiring gaps** (F-01-002, F-01-010, F-01-011, F-01-019, F-11-008): things that exist but are never registered, or registered twice, or claim to be wired but aren't.
- **T2 — Real-time / websocket / Socket.IO lifecycle** (F-02-002, F-02-020 partial, F-02-021, F-02-022 partial, F-01-018): the price-feed pipeline, including the headline aiohttp `async with` bug.
- **T3 — Recommendation service mixin reconciliation & SEC duplication** (F-02-001, F-02-006, F-02-012): the 1234-LOC service that 100% duplicates two unused mixins, plus the timezone-naive date bug and duplicated SEC disclosure constants.
- **T4 — Async task / scheduler / Celery cleanup** (F-02-004, F-02-008, F-02-009, F-02-013, F-02-024): two competing scheduler systems, two competing Celery configs, and an orphan stock-universe fetcher.
- **T5 — Service-layer correctness & N+1 / synthetic-data bugs** (F-02-007, F-02-010, F-02-016, F-02-017, F-02-019, F-02-022, F-02-023, F-02-025, F-01-014, F-01-015): real bugs in shipping endpoints (portfolio sector heuristic, RSI/MACD stubs, agents-service synthetic prices, sync DB in async health probe, missing thesis service, etc.).
- **T6 — `backend/utils/` consolidation & hygiene** (F-11-002 through F-11-020 minus F-11-008, plus F-02-014): cache sprawl (13 files), DB-utils sprawl (7 files), three cost-monitors, dead modules, bare excepts, naive datetimes, oversized files, doc drift.

The workpaper remains one document; sub-themes are used to organise §2 and §3 only.

**Top-10 anchor findings present in this slice:** F-01-002 (monitoring router never registered), F-02-001 (1234-LOC service vs unused mixins), F-02-002 (websocket session closes immediately). F-07-002 (transactions) is **NOT** in this slice (G4). F-09-002 (OptimizedRecommendationEngine) is **NOT** in this slice (G2/G3) — confirmed not present.

---

## 2. Member findings (all 46)

### T1 — Router & middleware wiring gaps (5)

- **F-01-002** [critical] monitoring.py router never registered — 5 endpoints unreachable
- **F-01-010** [medium] `register_security_middleware` imported but never called
- **F-01-011** [medium] VERSION_REGISTRY lists V3 STABLE but no V3 routes exist
- **F-01-019** [low] MiddlewarePriority NORMAL(5000) collides with REQUEST_SIZE(5000)
- **F-11-008** [medium] `CorrelationIDMiddleware` defined but never registered

### T2 — Real-time / websocket / Socket.IO lifecycle (4)

- **F-02-002** [critical] aiohttp ClientSession closed before WebSocket can be used (Finnhub never works)
- **F-02-020** [medium] No tests for `FinnhubWebSocketClient.connect()` flow
- **F-02-021** [medium] `create_socketio_asgi_app` not invoked in main.py
- **F-01-018** [low] websocket.py docstring claims "thin layer" but contains 142-line message handler

### T3 — Recommendation service mixin reconciliation (3)

- **F-02-001** [critical] RecommendationService duplicates 100% of two unused mixins (1234 LOC vs ~250)
- **F-02-006** [high] `get_bulk_price_history` called with timezone-naive `datetime.now().date()`
- **F-02-012** [medium] Duplicate SEC disclosure constants drift risk

### T4 — Async task / scheduler / Celery cleanup (5)

- **F-02-004** [critical] Async scheduler is wired but every body is commented out (operators see "started" but nothing fires)
- **F-02-008** [high] `data_pipeline.py` references missing `backend.config.celery_config`
- **F-02-009** [high] `task_config.py` (592 LOC) parallel queue/routing config not used in production
- **F-02-013** [medium] `ComprehensiveStockFetcher` (514 LOC) orphaned, never imported
- **F-02-024** [low] Two scheduling systems both documented as canonical

### T5 — Service-layer correctness & ingestion bugs (10)

- **F-01-014** [medium] stocks.py inline ImportError fallback hides missing `enhanced_error_handling`
- **F-01-015** [medium] health.py uses sync SQLAlchemy engine — blocks async event loop
- **F-02-007** [high] `_build_position_list` N+1 query (30-stock portfolio = 30 round-trips)
- **F-02-010** [high] `agents_service.run_technical_analysis` runs on synthetic random data
- **F-02-016** [medium] `day_change` uses unexplained `* 0.01` magic number
- **F-02-017** [medium] `calculate_rsi`/`calculate_macd` are stub `return None`
- **F-02-019** [medium] Investment Thesis feature has no service layer
- **F-02-022** [medium] Sector approximation uses `pos.symbol[:2]` (AAPL≠AAL)
- **F-02-023** [low] Module-level engine singletons not thread-safe
- **F-02-025** [low] `trading_service` symbol validation rolls its own check

### T6 — `backend/utils/` consolidation & hygiene (19)

- **F-02-014** [medium] Duplicated in-memory TTL cache implementations across `news_service` & `market_data_service`
- **F-11-002** [high] `cache_warmer.py` has zero production importers
- **F-11-003** [high] `db_timescale_init.py` and `deadlock_handler.py` have no non-test importers
- **F-11-004** [high] Cache module sprawl: 13 cache files, ~6,300 LOC, several with ≤2 importers
- **F-11-005** [high] 13 bare `except:` clauses in `backend/utils/`
- **F-11-006** [medium] Pervasive timezone-naive `datetime.now()` (30+ sites in utils)
- **F-11-007** [medium] `enhanced_error_handling.py` (941 LOC) largely unwired; competes with `middleware/error_handler.py`
- **F-11-009** [medium] Three cost-monitor implementations (~1,500 LOC combined)
- **F-11-010** [medium] DB utility sprawl — 7 active database utility modules
- **F-11-011** [medium] `print()` statements in `database_optimized.py` & `performance_tester.py`
- **F-11-012** [medium] Three modules exceed project's 800-line max (`risk_manager`, `portfolio_optimizer`, `resilient_pipeline`)
- **F-11-013** [medium] `tests/test_cache_warming.py` imports the dead `cache_warmer` module
- **F-11-014** [medium] `backend/utils/__init__.py` re-exports only 2 of 60 utilities
- **F-11-015** [medium] `docs/architecture/error-handling-analysis.md` cites non-existent main.py line numbers
- **F-11-017** [low] Hard top-level `import structlog` even where stdlib logging suffices
- **F-11-018** [low] `enhanced_error_handling.py` uses `from .exceptions import *`
- **F-11-019** [low] `backend/utils/__init__.py` has trivial docstring
- **F-11-020** [low] Anonymized IP uses `XXX.XXX` instead of documented convention

**Absorbed duplicates:** F-02-024 (two-scheduler doc) is fully resolved by executing F-02-004 (delete `scheduler.py`); kept separate so the doc-update step is not lost. F-11-013 (test imports dead module) follows automatically from F-11-002 (delete the module); kept as explicit test-removal step. F-02-012 (duplicate SEC constants) is largely resolved by F-02-001 option (a); kept as a fallback if F-02-001 chooses option (b).

**Total referenced:** 46 / 46.

---

## 3. Sequenced fix steps

> Ordering rationale: T1 (cheap wiring fixes) → T3 (anchor 1234-LOC reconciliation that frees later steps) → T2 (websocket bug — depends on no other work) → T5 (service correctness — some depend on T3) → T4 (task cleanup, requires human ack) → T6 (utils consolidation, mostly mechanical with two human-ack items).

### Step 1 — Register the monitoring router (F-01-002)
- **File:** `backend/api/main.py`
- **Action:** Add `from backend.api.routers import monitoring` near the other router imports and `app.include_router(monitoring.router, tags=["monitoring"])` (router self-prefixes — do NOT add prefix).
- **Fail-first test:** `tests/integration/test_monitoring_router.py::test_health_endpoint` calls `GET /api/monitoring/health` and asserts status_code == 200 — currently returns 404 (silently dead).
- **Pass-after test:** same test passes.

### Step 2 — Remove the dead `register_security_middleware` import (F-01-010)
- **File:** `backend/api/main.py:143`
- **Action:** Delete the unused import (middleware stack is configured inline).
- **Pass-after test:** `grep -n "register_security_middleware" backend/api/main.py` returns 0 lines.

### Step 3 — Resolve V3 version-registry lie (F-01-011)
- **File:** `backend/api/versioning.py:303-350`
- **Action:** Set `LATEST = APIVersion.V2` and downgrade V3 to `VersionStatus.PLANNED` (or remove entirely until routes ship).
- **Fail-first test:** `tests/integration/test_versioning.py::test_latest_resolves` asserts `APIVersionManager.LATEST` resolves to a registered router prefix — currently fails because `/api/v3/*` returns 404.
- **Pass-after test:** same test passes.

### Step 4 — Disambiguate MiddlewarePriority NORMAL collision (F-01-019)
- **File:** `backend/middleware/stack.py:66-73`
- **Action:** Either remove `NORMAL = 5000` (unused) or change to `4500`.
- **Pass-after test:** `len({m.value for m in MiddlewarePriority}) == len(MiddlewarePriority)`.

### Step 5 — Register CorrelationIDMiddleware (F-11-008)
- **File:** `backend/api/main.py` (middleware section, ~line 320), `backend/utils/structured_logging.py:316`
- **Action:** Add `app.add_middleware(CorrelationIDMiddleware)`.
- **Fail-first test:** `tests/integration/test_correlation_id.py` — `curl /api/v1/health -i` currently never includes `X-Correlation-ID` response header.
- **Pass-after test:** `X-Correlation-ID` always present; `correlation_id_var` populated in log records.

### Step 6 — Recommendation mixin reconciliation **[REQUIRES HUMAN ACK]** (F-02-001, absorbs F-02-012)
- **Files:** `backend/services/recommendation_service.py`, `backend/services/recommendation_crud.py`, `backend/services/recommendation_analysis.py`
- **Action — DEFAULT (per synthesis-handoff §6):** keep `RecommendationService` (1234 LOC) as the canonical class; mark the two mixin files (`recommendation_crud.py`, `recommendation_analysis.py`) for deletion in PRD §2. **Do NOT auto-delete.** Halt for human decision: option (a) `class RecommendationService(RecommendationCrudMixin, RecommendationAnalysisMixin):` and remove duplicates → reduces service to ~250 LOC; option (b) delete both mixin files. SEC disclosure constants are extracted to `backend/services/sec_disclosures.py` regardless.
- **Fail-first test:** `tests/services/test_recommendation_dedup.py` — assert `grep -c "RECOMMENDATION_MODEL_VERSION =" backend/services/` equals 1; currently returns 2 (silent drift risk).
- **Pass-after test:** dedup test passes; `wc -l backend/services/recommendation_service.py` < 350 (option a) OR mixin files absent (option b); existing `pytest backend/tests/test_recommendation*.py` all pass.

### Step 7 — Fix timezone-naive `datetime.now().date()` in recommendation pipeline (F-02-006)
- **Files:** `backend/services/recommendation_service.py:304-305`, `backend/services/recommendation_analysis.py:126-127`
- **Action:** Replace with `datetime.now(timezone.utc).date()`.
- **Fail-first test:** `tests/services/test_recommendation_dates.py` — freezing time to UTC midnight in a non-UTC machine timezone, assert `start_date` is the UTC-relative -90 days. Currently fails because `datetime.now()` returns local time.
- **Pass-after test:** `grep -nE "datetime\.now\(\)\.(date\(\)|date\b)" backend/services/recommendation_*.py` returns 0 hits.

### Step 8 — Promote SEC constants to shared module (F-02-012)
- **Files:** new `backend/services/sec_disclosures.py`; remove duplicates from `recommendation_service.py:25-51` and `recommendation_crud.py:23-46`.
- **Action:** Single source of truth for `RECOMMENDATION_MODEL_VERSION`, `RECOMMENDATION_MODEL_TRAINING_DATE`, `SEC_RISK_WARNING`, `SEC_METHODOLOGY_DISCLOSURE_TEMPLATE`, `SEC_LIMITATIONS_STATEMENT`.
- **Pass-after test:** `grep -c "SEC_RISK_WARNING =" backend/services/` returns 1.

### Step 9 — Fix Finnhub websocket session lifetime (F-02-002)
- **File:** `backend/services/realtime_price_service.py:86-98`
- **Action:** Promote session to instance attribute. In `__init__`: `self._session: aiohttp.ClientSession | None = None`. In `connect()`: `self._session = aiohttp.ClientSession(); self.websocket = await self._session.ws_connect(...)`. In `disconnect()`: `await self._session.close()`. Remove the `async with aiohttp.ClientSession()`.
- **Fail-first test (currently-broken):** `tests/integration/test_realtime_price_service.py::test_receive_one_trade` — stubs an aiohttp ws server, registers a callback, sends one trade message. Asserts callback invoked within 1 second AND **no reconnect occurred**. The test currently fails: receive_loop sees `WSMsgType.CLOSED` immediately and triggers infinite reconnect.
- **Pass-after test:** same test passes.

### Step 10 — Add Finnhub WS test harness (F-02-020)
- **File:** new `backend/tests/integration/test_realtime_price_service.py` (created in Step 9 above; this step formalises it).
- **Action:** Solidify the aiohttp-stub test as a regression suite covering connect, subscribe, receive callback, disconnect, error reconnect.
- **Pass-after test:** `pytest -k test_realtime_price_service` passes 100%.

### Step 11 — Decide Socket.IO mount or delete (F-02-021)
- **File:** `backend/services/socketio_service.py:300-313` and `backend/api/main.py`
- **Action:** Decision needed (see also F-02-015 in scope 02 — note: not in our slice). Default action: mount per the docstring instructions. If sockets are intentionally disabled, delete the file in a follow-up.
- **Pass-after test:** `grep -n "create_socketio_asgi_app" backend/api/main.py` returns 1 OR file deleted.

### Step 12 — Move websocket message handler into service (F-01-18)
- **Files:** `backend/api/routers/websocket.py:204-346`, `backend/services/websocket_service.py`
- **Action:** Move `handle_secure_client_message` into `websocket_service`; update docstring.
- **Pass-after test:** `wc -l backend/api/routers/websocket.py` < 150.

### Step 13 — Replace sync DB in async health probe (F-01-015)
- **File:** `backend/api/routers/health.py:8`
- **Action:** Replace `with engine.connect() as conn` with `async with db_manager.get_session() as session: await session.execute(text("SELECT 1"))` (same pattern monitoring.py:51-52 uses).
- **Fail-first test:** load test with 50 concurrent `GET /api/health/readiness` requests asserts P95 < 200ms while a slow DB query holds a connection. Currently fails because the sync call blocks the event loop.
- **Pass-after test:** same test passes.

### Step 14 — Fix N+1 in `_build_position_list` (F-02-007)
- **Files:** `backend/services/portfolio_service.py:766-768`, `backend/repositories/stock_repository.py` (add `get_by_symbols_bulk` if absent)
- **Action:** Pre-fetch all `Stock` rows by symbol set, build dict, look up.
- **Fail-first test:** `tests/integration/test_portfolio_detail_queries.py` instruments SQLAlchemy event listener for `before_cursor_execute`; asserts ≤6 queries for a 30-position portfolio. Currently shows ~36.
- **Pass-after test:** same test passes.

### Step 15 — Replace `pos.symbol[:2]` sector heuristic (F-02-022)
- **File:** `backend/services/portfolio_service.py:386`
- **Action:** Use `pos.stock.sector` (load via the bulk fetch from Step 14).
- **Pass-after test:** integration test asserts `unique_sectors` equals count of distinct `Stock.sector`.

### Step 16 — Fix `day_change` magic number (F-02-016)
- **File:** `backend/services/portfolio_service.py:580-586`
- **Action:** Replace `(close - average_cost) * 0.01` with `price_update.change` from `RealtimePriceService` (already exposed at `realtime_price_service.py:42`).
- **Fail-first test:** `tests/services/test_portfolio_day_change.py` with known prices (open=100, close=102) and 10 shares asserts `day_change == 20.0`. Currently fails (gives nonsense based on entry price).
- **Pass-after test:** same test passes.

### Step 17 — Replace synthetic random prices in technical analysis (F-02-010)
- **File:** `backend/services/agents_service.py:118-141`
- **Action:** Replace `np.random.seed(...)/np.random.normal(...)` with `await price_repository.get_price_history(ticker, ...)`. Raise `NotImplementedError` if no real prices available rather than fabricating.
- **Fail-first test:** `tests/services/test_agents_technical.py` — call `run_technical_analysis("AAPL", "standard")` twice; assert outputs match because they used the same persisted fixture. Currently passes only due to deterministic seed (which is the bug — it returns synthetic output, not real-data output).
- **Pass-after test:** with real-fixture prices, RSI/SMA outputs match a hand-computed reference; `grep -n "np.random.seed\|np.random.normal" backend/services/agents_service.py` returns 0.

### Step 18 — Delete or implement RSI/MACD stubs (F-02-017)
- **File:** `backend/services/analysis_service.py:242-249`
- **Action:** Delete stub `calculate_rsi`/`calculate_macd` (callers should use the engine or alpha-vantage helper).
- **Pass-after test:** `grep -nE "def calculate_(rsi|macd)" backend/services/analysis_service.py` returns 0.

### Step 19 — Decide Investment Thesis service layer (F-02-019)
- **File:** new `backend/services/thesis_service.py` (or doc update in `docs/architecture/`)
- **Action:** Add thin service mirroring `watchlist_service.py` pattern, OR document the simple-CRUD exception in architecture docs.
- **Pass-after test:** `backend/services/thesis_service.py` exists OR architecture doc explicitly notes the exception.

### Step 20 — Lock thread-safe lazy engine init (F-02-023)
- **File:** `backend/services/agents_service.py:26-49`
- **Action:** Wrap module-level engine accessors with `functools.lru_cache(maxsize=None)` factories.
- **Pass-after test:** concurrent stress test with 50 threads constructs exactly 1 instance of each engine.

### Step 21 — Use canonical ticker validator in trading service (F-02-025)
- **File:** `backend/services/trading_service.py:96-99`
- **Action:** Import `InputValidator.PATTERNS["ticker"]` from `backend/utils/validation.py`.
- **Pass-after test:** trading-service ticker rejection set matches the canonical validator's.

### Step 22 — Hard-import `enhanced_error_handling` (F-01-014)
- **File:** `backend/api/routers/stocks.py:42-59`
- **Action:** Convert try/except ImportError to a hard import; if optional, log a startup warning when the fallback engages.
- **Pass-after test:** removing `enhanced_error_handling.py` causes startup to fail loudly with descriptive ImportError.

### Step 23 — Async scheduler dual-system cleanup **[REQUIRES HUMAN ACK]** (F-02-004, absorbs F-02-024)
- **Files:** `backend/tasks/scheduler.py`, `backend/api/main.py:92-93`
- **Action — DEFAULT:** delete `scheduler.py` and remove the `start_scheduler()` call from main.py; rely solely on Celery beat (canonical). Document in `docs/architecture/`. **Do NOT auto-delete** — halt for human decision: option (a) delete (preferred) vs option (b) implement bodies to call `celery_app.send_task(...)`.
- **Fail-first test:** `tests/integration/test_scheduler.py::test_periodic_market_data_actually_fires` — register a stub task, sleep 6s, assert it was enqueued. Currently fails silently because the scheduler bodies are commented out and operators see only "AsyncScheduler started" in logs.
- **Pass-after test (option a):** scheduler.py gone, beat-only canonical, integration test confirms beat enqueues task. (option b): scheduler bodies invoke real `send_task`, integration test confirms enqueue.

### Step 24 — Reconcile `data_pipeline.py` Celery app (F-02-008)
- **File:** `backend/tasks/data_pipeline.py:28-29`
- **Action:** Default — merge tasks into `data_tasks.py` under canonical `celery_app` and delete `data_pipeline.py`. Alternative: create `backend/config/celery_config.py` to fix the broken import.
- **Pass-after test:** `python -c "from backend.tasks.data_pipeline import app; app.conf"` does not raise (option b) OR file absent and `grep -rn data_pipeline backend/` returns no production refs (option a).

### Step 25 — Remove or unify `task_config.py` queue scheme **[REQUIRES HUMAN ACK]** (F-02-009)
- **File:** `backend/tasks/task_config.py`
- **Action:** Default — keep `celery_app.py` simpler scheme as canonical and delete `task_config.py` (592 LOC). Halt for human decision because it's a 592-LOC delete.
- **Pass-after test:** single `TaskPriority` source; routing integration test verifies tasks land on expected queues.

### Step 26 — Move or delete `ComprehensiveStockFetcher` (F-02-013)
- **File:** `backend/tasks/stock_universe_fetcher.py`
- **Action:** Move to `scripts/stock_universe_fetcher.py` with shebang and `scripts/README.md` entry. (No production users; audit doc reference in `CODEBASE_ARCHITECTURE_MAP.md:369`.)
- **Pass-after test:** file lives under `scripts/`; doc reference updated.

### Step 27 — Extract shared TTL cache util (F-02-014)
- **Files:** `backend/utils/local_ttl_cache.py` (new or alias to existing `cache.py::LRUCache`); update `backend/services/news_service.py:24-40` and `backend/services/market_data_service.py:24-40`.
- **Pass-after test:** `grep -rn "_mem_cache\|_price_cache" backend/services/` returns 0.

### Step 28 — Delete `cache_warmer.py` and its orphan test (F-11-002, absorbs F-11-013)
- **Files:** `backend/utils/cache_warmer.py`, `backend/tests/test_cache_warming.py`
- **Action:** Delete both; confirm `cache_management` router uses only `CacheWarmingStrategy`.
- **Pass-after test:** `grep -rn "from backend.utils.cache_warmer" backend/` returns 0; full suite passes.

### Step 29 — Delete `db_timescale_init.py` and `deadlock_handler.py` (F-11-003)
- **Action:** Verify no non-test importers; delete (or move to `scripts/`).
- **Pass-after test:** `grep -rn "from backend.utils.(db_timescale_init|deadlock_handler)" backend/` → 0 hits.

### Step 30 — Cache module consolidation **[REQUIRES HUMAN ACK]** (F-11-004)
- **Action:** 24h refactor — consolidate 13 cache files into `backend/utils/cache/` package: `core.py`, `decorators.py`, `warming.py`, `monitoring.py`, `policies.py`. Halt for human ack — large surface, cross-scope impact.
- **Pass-after test:** all cache callers import from `backend.utils.cache`; cache hit-rate metrics unchanged in staging.

### Step 31 — Fix bare `except:` clauses (F-11-005)
- **Files:** 13 sites listed in finding (api_cache_decorators.py, validation.py, advanced_cache.py, database_query_cache.py, query_cache.py, etc.).
- **Action:** Replace each with `except Exception:` (minimum) or narrower; add `logger.exception(...)`.
- **Pass-after test:** `ruff check --select E722 backend/utils/` returns 0 hits.

### Step 32 — Project-wide timezone-aware datetime sweep (F-11-006)
- **Files:** 30+ sites across `backend/utils/`.
- **Action:** Replace `datetime.now()` with `datetime.now(timezone.utc)` (or `time.monotonic()` for durations). Add `ruff` rule `DTZ005` to lock in.
- **Pass-after test:** `ruff check --select DTZ backend/utils/` passes.

### Step 33 — Decide `enhanced_error_handling.py` future **[REQUIRES HUMAN ACK]** (F-11-007)
- **File:** `backend/utils/enhanced_error_handling.py` (941 LOC)
- **Action:** Default — keep only `handle_api_error`, `validate_stock_symbol`, `@with_error_handling` (used by `resilient_pipeline.py`). Delete `ErrorClassifier`, `ErrorCorrelationEngine`, `ErrorHandlingManager`. Halt for human decision: option (a) execute phased rollout; option (b) trim to <300 LOC.
- **Pass-after test:** `wc -l backend/utils/enhanced_error_handling.py` < 300, OR `ENHANCED_ERROR_HANDLING_ENABLED` flag wired with passing integration test.

### Step 34 — Cost monitor consolidation **[REQUIRES HUMAN ACK]** (F-11-009)
- **Files:** `cost_monitor.py`, `enhanced_cost_monitor.py`, `persistent_cost_monitor.py`
- **Action:** Default — keep `cost_monitor.py` (actively imported by `base_client`, `market_scanner`, `monitoring` router); fold persistence/budgets features into it; delete the other two. Halt for human ack.
- **Pass-after test:** single `backend.utils.cost_monitor`; `pytest -k cost` passes.

### Step 35 — DB utility sprawl reshape **[REQUIRES HUMAN ACK]** (F-11-010)
- **Files:** 7 modules.
- **Action:** Default — keep `database.py` and `async_database.py` (heavy importers). Move `database_optimized.py`, `db_init.py`, `db_read_replicas.py`, `database_monitoring.py`, `optimized_queries.py` into `backend/utils/db/` package and trim. Halt for human ack — cross-scope (07-database-persistence).
- **Pass-after test:** `wc -l backend/utils/database*.py backend/utils/db*.py backend/utils/async_database.py backend/utils/optimized_queries.py` < 1500.

### Step 36 — Replace `print()` with logger (F-11-011)
- **Files:** `backend/utils/database_optimized.py:319,324`, `backend/utils/performance_tester.py:600`.
- **Action:** Replace with `logger.info(...)` or move `if __name__ == "__main__"` blocks to `scripts/`.
- **Pass-after test:** `grep -nE "^[^#]*\bprint\(" backend/utils/*.py` returns 0.

### Step 37 — Split oversized modules **[REQUIRES HUMAN ACK]** (F-11-012)
- **Files:** `risk_manager.py` (985), `portfolio_optimizer.py` (950), `resilient_pipeline.py` (1049).
- **Action:** Continue extraction pattern (risk_manager already extracted var/metrics/stress). Split the other two by stage/concern. Halt for human ack — multi-file refactor.
- **Pass-after test:** all `backend/utils/*.py` ≤ 800 LOC.

### Step 38 — Decide `backend/utils/__init__.py` public surface (F-11-014, absorbs F-11-019)
- **File:** `backend/utils/__init__.py:5-7`
- **Action:** Default — remove the `RiskManager`/`PortfolioOptimizer` re-exports and document submodules as the canonical import path. Update docstring (F-11-019).
- **Pass-after test:** `__init__.py` either empty (with explanatory docstring) OR has a deliberate public surface; consistent with codebase imports.

### Step 39 — Annotate or archive `error-handling-analysis.md` (F-11-015)
- **File:** `docs/architecture/error-handling-analysis.md`
- **Action:** Add deprecation banner pointing to `docs/audits/2026-04/`, OR move to `docs/archive/`.
- **Pass-after test:** doc has visible deprecation banner OR moved.

### Step 40 — Make `structlog` optional (F-11-017)
- **File:** `backend/utils/structured_logging.py:14`
- **Action:** Wrap `import structlog` in try/except; fall back to stdlib logging in non-structured paths.
- **Pass-after test:** module imports successfully in a slim venv without `structlog`.

### Step 41 — Replace wildcard import (F-11-018)
- **File:** `backend/utils/enhanced_error_handling.py:23`
- **Action:** Replace `from .exceptions import *` with explicit imports of names used.
- **Pass-after test:** `ruff check --select F403 backend/utils/` returns 0.

### Step 42 — Document anonymized-IP convention (F-11-020)
- **File:** `backend/utils/data_anonymization.py:116`
- **Action:** Either change to `0.0` / `*.*` per documented convention, or document the existing `XXX.XXX` choice in a docstring.
- **Pass-after test:** docstring describes convention; existing tests pass.

---

## 4. Files touched

- `backend/api/main.py`
- `backend/api/routers/monitoring.py` (no edit — just registered)
- `backend/api/routers/stocks.py`
- `backend/api/routers/health.py`
- `backend/api/routers/websocket.py`
- `backend/api/security_integration.py` (no edit)
- `backend/api/versioning.py`
- `backend/middleware/stack.py`
- `backend/services/recommendation_service.py`
- `backend/services/recommendation_crud.py` (delete or wire as mixin)
- `backend/services/recommendation_analysis.py` (delete or wire as mixin)
- `backend/services/sec_disclosures.py` (new)
- `backend/services/realtime_price_service.py`
- `backend/services/socketio_service.py`
- `backend/services/portfolio_service.py`
- `backend/services/agents_service.py`
- `backend/services/analysis_service.py`
- `backend/services/news_service.py`
- `backend/services/market_data_service.py`
- `backend/services/trading_service.py`
- `backend/services/thesis_service.py` (new — pending decision)
- `backend/services/websocket_service.py`
- `backend/repositories/stock_repository.py` (add `get_by_symbols_bulk`)
- `backend/tasks/scheduler.py` (delete)
- `backend/tasks/data_pipeline.py` (delete or fix)
- `backend/tasks/task_config.py` (delete pending ack)
- `backend/tasks/stock_universe_fetcher.py` → `scripts/stock_universe_fetcher.py`
- `backend/utils/__init__.py`
- `backend/utils/structured_logging.py`
- `backend/utils/cache_warmer.py` (delete)
- `backend/utils/db_timescale_init.py` (delete)
- `backend/utils/deadlock_handler.py` (delete)
- `backend/utils/local_ttl_cache.py` (new or alias)
- `backend/utils/enhanced_error_handling.py`
- `backend/utils/cost_monitor.py` (consolidate target)
- `backend/utils/enhanced_cost_monitor.py` (delete pending ack)
- `backend/utils/persistent_cost_monitor.py` (delete pending ack)
- `backend/utils/database_optimized.py`
- `backend/utils/performance_tester.py`
- `backend/utils/data_anonymization.py`
- `backend/utils/api_cache_decorators.py`, `validation.py`, `advanced_cache.py`, `database_query_cache.py`, `query_cache.py`, `cache_warming.py` (bare-except fixes)
- `backend/utils/cache/` (new package, pending ack)
- `backend/utils/db/` (new package, pending ack)
- Tests added: `tests/integration/test_monitoring_router.py`, `tests/integration/test_versioning.py`, `tests/integration/test_correlation_id.py`, `tests/integration/test_realtime_price_service.py`, `tests/integration/test_portfolio_detail_queries.py`, `tests/integration/test_scheduler.py`, `tests/services/test_recommendation_dedup.py`, `tests/services/test_recommendation_dates.py`, `tests/services/test_portfolio_day_change.py`, `tests/services/test_agents_technical.py`
- Tests removed: `backend/tests/test_cache_warming.py`
- `pyproject.toml` / `ruff.toml` (add `DTZ` and `E722` rules)
- `docs/architecture/` (scheduler canonical-doc, thesis-service exception, error-handling-analysis archive banner)
- `scripts/README.md`
- `docs/architecture/CODEBASE_ARCHITECTURE_MAP.md:369`

---

## 5. Acceptance tests (consolidated)

```bash
# T1 — wiring
pytest tests/integration/test_monitoring_router.py -k test_health_endpoint
pytest tests/integration/test_versioning.py -k test_latest_resolves
pytest tests/integration/test_correlation_id.py
grep -n "register_security_middleware" backend/api/main.py | wc -l   # → 0
python -c "from backend.middleware.stack import MiddlewarePriority; assert len({m.value for m in MiddlewarePriority}) == len(MiddlewarePriority)"

# T2 — websocket / Socket.IO
pytest -k test_realtime_price_service
grep -n "create_socketio_asgi_app" backend/api/main.py | wc -l   # → 1 (or file deleted)
test $(wc -l < backend/api/routers/websocket.py) -lt 150

# T3 — recommendation reconciliation
test $(grep -c "RECOMMENDATION_MODEL_VERSION =" backend/services/recommendation_service.py backend/services/recommendation_crud.py) -eq 0
grep -c "SEC_RISK_WARNING =" backend/services/sec_disclosures.py   # → 1
grep -nE "datetime\.now\(\)\.(date\(\)|date\b)" backend/services/recommendation_*.py | wc -l   # → 0
pytest backend/tests/test_recommendation*.py
test $(wc -l < backend/services/recommendation_service.py) -lt 350   # if option (a) chosen

# T4 — task / scheduler
pytest tests/integration/test_scheduler.py::test_periodic_market_data_actually_fires
python -c "from backend.tasks.data_pipeline import app; app.conf"   # or file absent
grep -rn "stock_universe_fetcher\|ComprehensiveStockFetcher" backend/ | wc -l   # → 0

# T5 — service correctness
pytest tests/integration/test_portfolio_detail_queries.py   # ≤6 queries
pytest tests/services/test_portfolio_day_change.py
pytest tests/services/test_agents_technical.py
grep -n "np.random.seed\|np.random.normal" backend/services/agents_service.py | wc -l   # → 0
grep -nE "def calculate_(rsi|macd)" backend/services/analysis_service.py | wc -l   # → 0
locust -f load/health_readiness.py --headless -u 50 -r 10 -t 30s   # P95 < 200ms

# T6 — utils hygiene
ruff check --select E722 backend/utils/   # 0 issues
ruff check --select DTZ backend/utils/   # 0 issues
ruff check --select F403 backend/utils/   # 0 issues
grep -nE "^[^#]*\bprint\(" backend/utils/*.py | wc -l   # → 0
grep -rn "from backend.utils.cache_warmer" backend/ | wc -l   # → 0
grep -rn "from backend.utils.(db_timescale_init|deadlock_handler)" backend/ | wc -l   # → 0
grep -rn "_mem_cache\|_price_cache" backend/services/ | wc -l   # → 0
for f in backend/utils/*.py; do test $(wc -l < "$f") -le 800 || echo "OVERSIZE: $f"; done   # no output
python -c "import sys; sys.modules.pop('structlog', None); import backend.utils.structured_logging"   # in slim venv
pytest -k cost   # cost-monitor consolidation

# Smoke
pytest backend/tests/   # full suite green
```

---

## 6. Rollback plan

- Each step is a separate commit; a single revert restores prior behaviour.
- For human-ack steps (Steps 6, 23, 25, 30, 33, 34, 35, 37) the deletions are gated; if any acceptance test regresses on staging, revert the delete commit and restore from `git restore --source=<sha>~1`.
- Cache and DB consolidation (Steps 30, 35) are done behind feature-flag imports for one release cycle; old modules are kept as deprecation shims emitting `DeprecationWarning` for two weeks before final removal.
- Websocket fix (Step 9) ships behind a `FINNHUB_REALTIME_ENABLED` flag default-on; flip off to fall back to polling if production telemetry shows session leaks.
- Scheduler delete (Step 23) is gated on a 24-hour staging soak; if Celery beat is not configured, revert and reinstate `scheduler.py` until ops adds beat.

---

## 7. Dependencies

- **Soft-depends-on B (jwt-auth):** F-01-018 (websocket message handler relocation) touches the same auth path the B cluster modifies; B should land first to avoid merge churn.
- **Soft-depends-on E (test-exclusion):** F-02-020, F-11-013, F-02-001 add or remove tests; E governs the test-suite signal — execute E first so we know which test failures are real.
- **Soft-depends-on D (random-data-recommendations):** F-02-010 (synthetic prices in agents_service) is a sibling of D's recommendation random-data work; align fixtures.
- **Cross-scope (informational):** F-01-015 → 07-database-persistence; F-02-007 → 07; F-11-010 → 07; F-02-008 → 16-config-secrets; F-02-014 → 11-utils (internal); F-02-013 → 17-scripts-tooling, 18-docs-health.

`depends_on: ["B_jwt-auth", "E_test-exclusion", "D_random-data-recommendations"]`

---

## 8. Effort & cost

- Estimated engineering effort: ~115 hours (sum of finding effort_hours: 1+1+2+2+2+3+1+6+2+3+4+0.5+3+3+6+4+1+2+4+2+1+1+1+0.5+2+2+24+3+4+16+8+12+1+12+1+1+0.5+0.5+2+1+1+0.5+1.0 ≈ 142 raw, but ~25h overlap among consolidation tasks → ~115h net).
- Loki-mode estimated cost: $5–10 (most steps are small-context Edits; the four consolidation steps with human ack are not Loki-actionable).
- Wall-clock: 2–3 weeks with one engineer, or 1 week with a 3-agent swarm executing parallel sub-themes.

---

## 9. Loki-actionable status

| Step | Finding | loki_actionable | requires_human_ack | Rationale |
|---|---|---|---|---|
| 1 | F-01-002 | yes | no | One-line wiring |
| 2 | F-01-010 | yes | no | Remove dead import |
| 3 | F-01-011 | yes | no | Enum value change |
| 4 | F-01-019 | yes | no | Constant tweak |
| 5 | F-11-008 | yes | no | Add middleware line |
| 6 | F-02-001 | yes | **yes** | Architectural choice (mixin vs in-class). Default = keep service, mark mixins for cleanup; halt for human (per synthesis-handoff §6) |
| 7 | F-02-006 | yes | no | Mechanical replacement |
| 8 | F-02-012 | yes | no | Constant extraction |
| 9 | F-02-002 | yes | no | Bug fix |
| 10 | F-02-020 | yes | no | New test |
| 11 | F-02-021 | yes | no | Mount or delete |
| 12 | F-01-018 | yes | no | Move handler |
| 13 | F-01-015 | yes | no | Sync→async swap |
| 14 | F-02-007 | yes | no | Bulk-fetch refactor |
| 15 | F-02-022 | yes | no | Lookup `pos.stock.sector` |
| 16 | F-02-016 | yes | no | Replace magic number |
| 17 | F-02-010 | yes | no | Replace synthetic with repo fetch |
| 18 | F-02-017 | yes | no | Delete stubs |
| 19 | F-02-019 | yes | no | New service file or doc note |
| 20 | F-02-023 | yes | no | Add lru_cache |
| 21 | F-02-025 | no | no | Already flagged loki_actionable=false in slice |
| 22 | F-01-014 | yes | no | Hard import |
| 23 | F-02-004 | yes | **yes** | Default = delete `scheduler.py`; halt for human (destructive, dual-system) |
| 24 | F-02-008 | yes | no | Delete or fix import |
| 25 | F-02-009 | no | **yes** | 592-LOC delete; halt for human |
| 26 | F-02-013 | yes | no | Move file |
| 27 | F-02-014 | yes | no | Extract util |
| 28 | F-11-002 (+F-11-013) | yes | no | Delete dead module + test |
| 29 | F-11-003 | yes | no | Delete dead modules |
| 30 | F-11-004 | no | **yes** | 13-file consolidation, ~24h refactor; halt for human |
| 31 | F-11-005 | yes | no | Mechanical |
| 32 | F-11-006 | yes | no | Mechanical sweep |
| 33 | F-11-007 | no | **yes** | Default = trim to <300 LOC; halt for human |
| 34 | F-11-009 | no | **yes** | Three-module consolidation; halt for human |
| 35 | F-11-010 | no | **yes** | Cross-scope DB-utils reshape; halt for human |
| 36 | F-11-011 | yes | no | Mechanical |
| 37 | F-11-012 | no | **yes** | Multi-file split; halt for human |
| 38 | F-11-014 (+F-11-019) | no | no | Doc/exports decision |
| 39 | F-11-015 | yes | no | Doc edit |
| 40 | F-11-017 | yes | no | Optional import |
| 41 | F-11-018 | yes | no | Replace wildcard |
| 42 | F-11-020 | yes | no | Doc convention |

**Human-ack items surfaced to PRD §2:** Steps 6, 23, 25, 30, 33, 34, 35, 37 (eight items).

---

## 10. Risks per sub-theme

- **T1 (wiring):** Low. Risk is registering a router/middleware that shadows an existing route; mitigated by the fail-first acceptance tests.
- **T2 (websocket):** Medium. The Finnhub session fix changes session lifetime semantics — risk is leaking sessions on disconnect; mitigated by feature flag and explicit `disconnect()` close.
- **T3 (recommendation reconciliation):** High if option (a) chosen — 1234 LOC of cut-and-paste must reconcile to a single inheritance tree; risk of subtle behaviour drift in SEC disclosure text. Mitigated by F-02-012 extraction first.
- **T4 (scheduler/celery):** High operational risk — deleting `scheduler.py` requires that ops actually run Celery beat in every environment; staging soak required. `task_config.py` deletion is reversible.
- **T5 (service correctness):** Medium. Replacing synthetic data in `agents_service` will surface previously-hidden missing-data paths; ensure callers handle `NotImplementedError`. The N+1 fix risks introducing a different N+1 if `get_by_symbols_bulk` is naïvely implemented.
- **T6 (utils consolidation):** High. Cache and DB consolidation touch many call sites; staged behind deprecation shims for two-week rollout. Bare-except sweep can surface latent bugs that were previously swallowed; expect a wave of new error logs immediately after deploy.

---
