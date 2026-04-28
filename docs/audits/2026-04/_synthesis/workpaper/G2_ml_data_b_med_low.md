# G2_ml_data_b_med_low — ML/Data Medium+Low Residual Cluster

**Worker:** G2_ml_data_b_med_low
**Findings:** 47 (medium + low) across scopes 03 (ml-engine), 04 (trading-agents), 05 (data-ingestion-etl), 06 (airflow-pipelines), 09 (analytics)
**Posture:** Bulk cleanup / maintainability batch — defer to end of program after critical/high (G2_a) lands.

---

## 1. Cluster Overview

This residual cluster aggregates the medium- and low-severity ML/data-quality findings that remain after the critical and high-priority slice (`G2_ml_data_a_crit_high`) has been carved off. Findings here are dominated by predictable cleanup categories rather than user-facing defects:

- **Code quality / lint hygiene** (~13 findings): bare excepts, unused imports, tuple-vs-string bugs, typos, dead methods, unused instance vars, asyncio anti-patterns.
- **Dead / superseded code** (~6 findings): duplicated extractors/engines, dead `extract_phase`, dead `_load_pytorch_model`, unused dataclass shim methods, `simple_training_pipeline.py`, `YFinanceUtils`.
- **Doc drift** (~6 findings): wrong model paths, undocumented rate-limit gap, EMA-as-SMA mislabel, ETL_ACTIVATION_SUCCESS staleness, Python version mismatch, sentiment API docstring drift.
- **Performance refactors** (~5 findings): O(n²) SMA loops, day-by-day stockstats, OBV/AD-line Python loops, feature-store per-ticker loops, server-side `tqdm` noise.
- **Architectural duplication / coupling** (~5 findings): two extraction architectures, recommendation-engine duplication, `Toolkit._config` shared state, drawdown duplicates across 3 files, parse-time pool creation.
- **Statistical correctness** (1 finding): ADF tau approximation in cointegration analyzer.
- **Testing gaps** (~4 findings): streaming/scanner, VaR calculator, Black-Litterman, FinancialSituationMemory + analyst nodes.
- **Incomplete features** (~3 findings): online-learner not wired in, hardcoded 100-stock scanner stub, PDF/Excel report stubs.

The cluster is intentionally low-risk: nearly all items are mechanical refactors safe to batch-process via Loki passes. A small subset (deletions of modules, statistical-method swaps) requires human acknowledgment.

---

## 2. Member Findings (all 47 IDs)

| Finding | Scope | Severity | Sub-theme |
|---|---|---|---|
| F-03-009 | 03 | medium | code_quality (bare except) |
| F-03-010 | 03 | medium | incomplete (online learner unwired) |
| F-03-011 | 03 | medium | stale_code (simple_training_pipeline) |
| F-03-012 | 03 | medium | performance (feature_store loops) |
| F-03-013 | 03 | medium | doc_drift (model path) |
| F-03-015 | 03 | low | code_quality (dead `_load_pytorch_model`) |
| F-03-016 | 03 | low | doc_drift (rate-limit doc lie) |
| F-04-010 | 04 | medium | code_quality (typo "Makrdown") |
| F-04-011 | 04 | medium | code_quality (tuple-as-string) |
| F-04-012 | 04 | medium | code_quality (filename typo `aggresive_debator.py`) |
| F-04-013 | 04 | medium | performance (per-day stockstats) |
| F-04-014 | 04 | medium | performance (server-side tqdm) |
| F-04-015 | 04 | medium | architecture (`Toolkit._config` class-level mutable) |
| F-04-016 | 04 | medium | testing_gap (memory + analyst nodes) |
| F-04-017 | 04 | medium | architecture (relative `eval_results/` path) |
| F-04-018 | 04 | medium | stale_code (unused `time`, `json` imports) |
| F-04-019 | 04 | low | doc_drift (python_requires 3.10 vs 3.11) |
| F-04-020 | 04 | low | dead_code (`YFinanceUtils` class) |
| F-04-021 | 04 | low | code_quality (discarded `strptime`) |
| F-04-022 | 04 | low | better_pattern (Reddit ValueError UX) |
| F-05-010 | 05 | medium | dead_code (`extract_phase`) |
| F-05-011 | 05 | medium | incomplete (scanner 100-stock stub) |
| F-05-013 | 05 | medium | doc_drift (Yahoo rate mismatch) |
| F-05-014 | 05 | medium | testing_gap (streaming + scanner) |
| F-05-015 | 05 | medium | architecture (parallel extractors) |
| F-05-016 | 05 | low | code_quality (unused `legacy_validator`/`aggregator`) |
| F-05-017 | 05 | low | code_quality (orphan asyncio task) |
| F-05-018 | 05 | low | doc_drift (ETL_ACTIVATION_SUCCESS stale) |
| F-05-019 | 05 | low | better_pattern (per-instance pool) |
| F-05-020 | 05 | low | code_quality (stale user-agents) |
| F-06-010 | 06 | medium | architecture (parse-time `ensure_pool_exists`) |
| F-06-011 | 06 | medium | code_quality (XCom non-JSON-serializable) |
| F-06-013 | 06 | medium | code_quality (`>= 20` vs `MIN_DATA_POINTS = 26`) |
| F-06-014 | 06 | medium | dead_code (BashOperator/ExternalTaskSensor imports) |
| F-06-015 | 06 | low | doc_drift (EMA-as-SMA columns) |
| F-06-016 | 06 | low | better_pattern (asyncio.new_event_loop anti-pattern) |
| F-09-011 | 09 | medium | architecture (RecommendationEngine duplication) |
| F-09-012 | 09 | medium | code_quality (hardcoded "Technology" sector) |
| F-09-013 | 09 | medium | code_quality (PDF/Excel stubs return None) |
| F-09-014 | 09 | medium | statistical (ADF tau approximation incorrect) |
| F-09-015 | 09 | medium | performance (O(n²) SMA) |
| F-09-016 | 09 | medium | performance (Python loops in OBV/AD) |
| F-09-017 | 09 | medium | testing_gap (VaRCalculator) |
| F-09-018 | 09 | medium | testing_gap (BlackLittermanOptimizer) |
| F-09-019 | 09 | low | doc_drift (sentiment API contract) |
| F-09-020 | 09 | low | code_quality (RecommendationEngine shim methods) |
| F-09-021 | 09 | low | better_pattern (duplicate `_calculate_max_drawdown`) |

Total: **47 findings** confirmed.

---

## 3. Sequenced Fix Steps

Steps grouped by sub-theme to maximize batch efficiency. Each step is independently reversible; sequencing inside a group is incidental except where noted.

### Step 1 — Lint sweep: bare except + unused imports + discarded values
Findings: **F-03-009, F-04-018, F-04-021, F-06-014, F-05-016**
- Replace bare `except:` with `except Exception as e:` (`backend/ml/ml_api_server.py:175-178`).
- Drop unused `import time`, `import json` from 12 TradingAgents modules (`market_analyst.py`, `fundamentals_analyst.py`, `news_analyst.py`, `social_media_analyst.py`, `risk_manager.py`, `research_manager.py`, `bull_researcher.py`, `bear_researcher.py`, `aggresive_debator.py`, `conservative_debator.py`, `neutral_debator.py`, `trader.py`).
- Remove discarded `datetime.strptime(...)` calls in `interface.py:634-635` (or convert to `datetime.fromisoformat`-style explicit validation).
- Remove `BashOperator`, `ExternalTaskSensor` imports from `ml_training_pipeline_dag.py:25-26`.
- Remove unused `self.legacy_validator`, `self.aggregator` from `etl_orchestrator.py:50-52`.
- **Verify:** `ruff check backend/ data_pipelines/ --select F401,E722` returns clean for touched files.

### Step 2 — Typo + filename normalization batch
Findings: **F-04-010, F-04-011, F-04-012**
- Fix `Makrdown → Markdown` in `news_analyst.py:21`.
- Remove trailing comma making `system_message` a tuple in `fundamentals_analyst.py:24-26`.
- Rename `aggresive_debator.py → aggressive_debator.py`; update import in `tradingagents/agents/__init__.py:13`.
- **Verify:** `grep -rn "Makrdown\|aggresive_debator" backend/TradingAgents/` returns 0 hits; `python -c "from tradingagents.agents.fundamentals_analyst import system_message; assert isinstance(system_message, str)"` passes.

### Step 3 — Doc drift batch (paths, rates, versions, contracts, EMA naming)
Findings: **F-03-013, F-03-016, F-04-019, F-05-013, F-05-018, F-06-015, F-09-019**
- Update `docs/ml/ML_OPERATIONS_GUIDE.md` and `docs/ml/ML_QUICKSTART.md`: `backend/ml_models/` → `ml_models/`.
- Remove rate-limiting section from `ML_API_REFERENCE.md` (or open a follow-up to implement `slowapi`; default to removal here).
- Bump `python_requires=">=3.11"` in `backend/TradingAgents/setup.py` and align `pyproject.toml`.
- Update `DATA_COLLECTION_SOLUTION.md:69` to reflect `yahoo_scraper` configured rate (5/min, max 300/hr) — or open a follow-up to align code.
- Mark `docs/reports/ETL_ACTIVATION_SUCCESS.md` as superseded; add front-matter note pointing to the post-G2_a validation report.
- Add column comment in `technical_indicators` DDL noting `ema_12`/`ema_26` are SMA approximations; reference `EMA_RECURSIVE_SQL` as the upgrade path.
- Update `SentimentAnalysisEngine` class docstring to document both `analyze_sentiment(text)` and `analyze_stock_sentiment(ticker, texts)`.
- **Verify:** `grep -rn "backend/ml_models" docs/ml/` empty; `grep -rn "Makrdown\|aggresive" docs/` empty; `pytest -k "test_setup_python_requires"` (new) passes.

### Step 4 — Dead-code deletions (module-level, requires human ack)
Findings: **F-03-011, F-03-015, F-04-020, F-05-010, F-09-020**
- Delete `backend/ml/simple_training_pipeline.py` (or downgrade to a docstring-only deprecation shim) once test references migrate.
- Delete `_load_pytorch_model` from `model_manager.py:218-226`.
- Delete `YFinanceUtils` class from `backend/TradingAgents/tradingagents/dataflows/yfin_utils.py` (confirm no external callers via cross-scope grep).
- Delete `extract_phase`, `validate_extracted_data`, related orphan attrs from `etl_orchestrator.py:374-423`.
- Remove the ~15 `_` shim methods from `RecommendationEngine` (`recommendation_engine.py:563-613`) and update any tests that patch shim methods to patch module-level functions.
- **Acceptance:** Each deletion gated on `grep` showing zero non-test, non-defining references. Flagged as `requires_human_ack: true` in §9.

### Step 5 — `OnlineLearner` decision: wire-or-deprecate
Findings: **F-03-010**
- Decision required (not Loki-actionable). Either:
  - (A) Wire `OnlineLearner.update()` into `ModelManager.predict()` post-inference cycle, OR
  - (B) Mark feature `EXPERIMENTAL` in code (`__init__.py` flag + module docstring) and add a roadmap entry.
- Default for residual sweep: option (B) — inexpensive, preserves code.
- **Verify:** Module docstring contains `EXPERIMENTAL`; CHANGELOG entry recorded.

### Step 6 — Daily scanner stub removal
Findings: **F-05-011**
- Replace hardcoded `all_symbols[:100]` in `daily_scanner.py:80-93` with `StockUniverseManager.get_all_active_tickers()` query.
- **Verify:** Integration test asserts `len(_get_all_stock_symbols()) > 500`.

### Step 7 — Performance vectorization batch
Findings: **F-03-012, F-04-013, F-09-015, F-09-016**
- `feature_store.py:240-269`: replace per-entity Python loops with `groupby` aggregations.
- `interface.py:526-535`: precompute full stockstats indicator series once, then slice by date range.
- `technical_analysis.py:487-489`: replace SMA list comprehensions with `pd.Series(close).rolling(k).mean()`.
- `technical_analysis.py:285-313`: vectorize `_calculate_obv` and `_calculate_ad_line` using `np.where` + `np.cumsum`.
- **Verify:** Microbenchmarks (added under `backend/tests/perf/`): feature-store 6000 tickers × 9 features < 5s; `_analyze_market_structure` 500-row < 10ms; `_calculate_volume_indicators` 500-row < 5ms; mocked `pd.read_csv` called once per 30-day window.

### Step 8 — Async / lifecycle hygiene
Findings: **F-05-017, F-06-016, F-04-014**
- `etl_orchestrator.py:146`: cancel `processing_task` in `finally` and gather with `return_exceptions=True`.
- `enhanced_stock_pipeline.py:60-76`: replace `new_event_loop`/`set_event_loop`/`run_until_complete` with `asyncio.run(...)`.
- `interface.py:329`: gate `tqdm` behind a `show_progress: bool = False` parameter on `get_reddit_global_news`.
- **Verify:** `asyncio.all_tasks()` empty after pipeline run; `grep -n "new_event_loop\|set_event_loop" data_pipelines/airflow/dags/enhanced_stock_pipeline.py` empty; server-side call with `show_progress=False` produces no stderr.

### Step 9 — Architectural state-leak fixes
Findings: **F-04-015, F-04-017, F-06-010, F-05-019**
- `Toolkit._config`: convert to instance attribute; promote `update_config` to instance method (`agent_utils.py:35`).
- `trading_graph.py:225-232`: rebase `eval_results/...` writes on `self.config["results_dir"]`.
- `daily_stock_pipeline.py:1572`: move `ensure_pool_exists()` out of module top-level into a setup task or init container.
- `data_loader.py:42-57`: hoist SQLAlchemy engine to module-level singleton or accept via constructor injection.
- **Verify:** Independent `Toolkit` instances retain independent configs; `_log_state` writes under `results_dir` regardless of CWD; DAG file parse does not invoke pool creation; `DataLoader._create_engine` invoked exactly once per process in integration test.

### Step 10 — XCom + threshold + scanner-stub corrections
Findings: **F-06-011, F-06-013**
- `ml_training_pipeline_dag.py:162`: serialize via `dataclasses.asdict(config)` or delete the unused `xcom_push`. Default: delete.
- `daily_stock_pipeline.py:666` and `:1356`: change `data_count >= 20` → `data_count >= 26` in both CTE WHERE clauses to match `MIN_DATA_POINTS`. Add unit test asserting EMA-26 NULL for stocks with fewer than 26 data points.
- **Verify:** `grep -n "data_count >= 20" data_pipelines/airflow/dags/daily_stock_pipeline.py` empty; XCom pull test absent.

### Step 11 — Recommendation-engine consolidation + report stubs + sector hardcode
Findings: **F-09-011, F-09-012, F-09-013**
- Make `RecommendationEngine` a thin facade; delegate to `OptimizedRecommendationEngine` (or vice versa) so daily-recommendation generation has a single code path.
- Add `sector: Optional[str] = None` to `StockRecommendation` dataclass; populate from market scanner; fix `get_top_sectors` (`recommendation_ranking.py:213-224`) to use real values.
- In `generate_report`, replace `pass` for `format='pdf'` and `format='excel'` with `raise NotImplementedError("PDF/Excel report not yet supported")`.
- **Verify:** Single import path used for daily generation; `get_top_sectors` returns variety on a mixed list; `generate_report(..., format='pdf')` raises.

### Step 12 — Statistical correctness: ADF tau (requires human ack)
Findings: **F-09-014**
- Replace custom ADF approximation (`cointegration_analyzer.py:152-168`) with `statsmodels.tsa.stattools.adfuller(spread, regression='nc')`.
- Update test fixtures and any callers expecting the old (incorrect) statistic.
- **Acceptance:** `test_cointegration` p-value matches `adfuller(spread, regression='nc')[1]` for the same residual series. Flagged `requires_human_ack: true` due to potential downstream signal change.

### Step 13 — Testing gap fills
Findings: **F-04-016, F-05-014, F-09-017, F-09-018**
- New test files (placed under `backend/tests/unit/`):
  - `test_trading_agents_memory.py` — mocked chromadb; cover `FinancialSituationMemory.add_situations`, `get_memories`; analyst node creation with mocked LLM; trader node.
  - `test_streaming.py` — `KafkaProducerClient.send_message` and consumer message handling, mocked `aiokafka`.
  - `test_scanner.py` — `DailyStockScanner._analyze_stock` with mocked data sources.
  - `test_var_calculator.py` — historical, parametric, Monte Carlo; Kupiec backtest.
  - `test_black_litterman.py` — zero-view, single-view, mixed-sign weight normalization.
- **Verify:** All new tests green; coverage ≥ 80% per the matching modules.

### Step 14 — Drawdown DRY extraction
Findings: **F-09-021**
- Create `backend/analytics/risk/calculators/drawdown.py` exposing `calculate_max_drawdown(series)`.
- Migrate `PortfolioOptimizer._calculate_max_drawdown`, `RecommendationEngine._calculate_risk_metrics` drawdown block, and `recommendation_scoring.py` to import this single function.
- **Verify:** `grep -rn "def _calculate_max_drawdown\|max_dd =" backend/analytics/` shows one definition.

### Step 15 — Reddit + user-agent UX
Findings: **F-04-022, F-05-020**
- `reddit_utils.py:68-71`: replace `ValueError` with `max(1, max_limit // n_files)` floor and a clearer log; or reword the exception to mention `max_limit_per_day`.
- `web_scrapers.py:37-46`: integrate `fake-useragent` (or `ua-generator`) and remove the 2021-era hardcoded list.
- **Verify:** `fetch_top_from_category("global_news", date, 1, ...)` no longer raises; user-agent strings include 2025+ Chrome/Firefox builds.

---

## 4. Files Touched

Source (read-only here, modified only when steps execute):

- `backend/ml/ml_api_server.py`
- `backend/ml/online_learning.py`
- `backend/ml/simple_training_pipeline.py`
- `backend/ml/feature_store.py`
- `backend/ml/model_manager.py`
- `backend/TradingAgents/tradingagents/agents/__init__.py`
- `backend/TradingAgents/tradingagents/agents/analysts/{news_analyst,fundamentals_analyst,market_analyst,social_media_analyst}.py`
- `backend/TradingAgents/tradingagents/agents/managers/{risk_manager,research_manager}.py`
- `backend/TradingAgents/tradingagents/agents/researchers/{bull_researcher,bear_researcher}.py`
- `backend/TradingAgents/tradingagents/agents/risk_mgmt/{aggresive_debator → aggressive_debator,conservative_debator,neutral_debator}.py`
- `backend/TradingAgents/tradingagents/agents/trader/trader.py`
- `backend/TradingAgents/tradingagents/agents/utils/agent_utils.py`
- `backend/TradingAgents/tradingagents/dataflows/{interface,reddit_utils,yfin_utils}.py`
- `backend/TradingAgents/tradingagents/graph/trading_graph.py`
- `backend/TradingAgents/setup.py`
- `backend/etl/etl_orchestrator.py`
- `backend/etl/data_loader.py`
- `backend/etl/web_scrapers.py`
- `backend/scanner/daily/daily_scanner.py`
- `backend/streaming/kafka_client.py` (tests only)
- `backend/analytics/recommendation_engine.py`
- `backend/analytics/recommendation_ranking.py`
- `backend/analytics/recommendation_scoring.py`
- `backend/analytics/sentiment_analysis.py`
- `backend/analytics/technical_analysis.py`
- `backend/analytics/portfolio/{black_litterman,modern_portfolio_theory}.py`
- `backend/analytics/risk/calculators/{var_calculator,drawdown.py [NEW]}`
- `backend/analytics/statistical/cointegration_analyzer.py`
- `data_pipelines/airflow/dags/{daily_stock_pipeline,ml_training_pipeline_dag,enhanced_stock_pipeline}.py`
- New tests under `backend/tests/unit/`: `test_trading_agents_memory.py`, `test_streaming.py`, `test_scanner.py`, `test_var_calculator.py`, `test_black_litterman.py` (+ optional perf microbenchmarks under `backend/tests/perf/`).
- `pyproject.toml`
- Docs: `docs/ml/{ML_OPERATIONS_GUIDE,ML_QUICKSTART,ML_API_REFERENCE}.md`, `docs/architecture/DATA_COLLECTION_SOLUTION.md`, `docs/reports/ETL_ACTIVATION_SUCCESS.md`.

---

## 5. Acceptance Tests

Mostly mechanical checks. Run after each step:

- **Lint:** `ruff check backend/ data_pipelines/ --select F401,E722,F841` clean.
- **Type:** `mypy backend/ml/ backend/TradingAgents/tradingagents/agents/ backend/etl/ backend/analytics/` no new errors vs baseline.
- **Path/string greps (zero-hit assertions):**
  - `grep -rn "backend/ml_models" docs/ml/`
  - `grep -rn "Makrdown\|aggresive_debator" backend/ docs/`
  - `grep -n "data_count >= 20" data_pipelines/airflow/dags/daily_stock_pipeline.py`
  - `grep -n "new_event_loop\|set_event_loop" data_pipelines/airflow/dags/enhanced_stock_pipeline.py`
  - `grep -n "_load_pytorch_model" backend/ml/model_manager.py` shows definition only — and after deletion, zero hits.
  - `grep -rn "YFinanceUtils" backend/TradingAgents/tradingagents/ --include="*.py" | grep -v yfin_utils.py` empty.
  - `grep -n "legacy_validator\|self\.aggregator" backend/etl/etl_orchestrator.py` empty.
  - `grep -rn "unlimited_data_extractor" backend/` empty if F-05-015 consolidation accepted.
- **Behavioral asserts:**
  - `python -c "from tradingagents.agents.fundamentals_analyst import system_message; assert isinstance(system_message, str)"`.
  - Two `Toolkit(config_a)` and `Toolkit(config_b)` instances retain independent `_config`.
  - `_get_all_stock_symbols()` returns > 500 entries (post F-05-011 fix).
  - `RecommendationEngine.generate_report(..., format='pdf')` raises `NotImplementedError`.
  - `get_top_sectors` returns more than `{"Technology"}` on a multi-sector list.
- **Perf microbenchmarks (informational, not gating):**
  - `_analyze_market_structure` 500-row < 10ms.
  - `_calculate_volume_indicators` 500-row < 5ms.
  - Feature-store 6000 × 9 features < 5s.
  - Mocked `pd.read_csv` called once per 30-day stockstats window.
- **Test suite:** `pytest backend/tests/unit/ -q` green, including new suites listed in Step 13.
- **Doc-link sanity:** `pytest tests/docs/test_links.py` (or equivalent) passes; superseded report carries front-matter notice.

---

## 6. Rollback Plan

All changes are confined to `git`-tracked files; rollback is per-step:

- Revert Step N via `git revert <sha>` or `git checkout HEAD~N -- <files>`.
- Renamed file (Step 2, `aggresive_debator.py`): rollback restores the misspelled name and import; both the rename commit and the `__init__.py` import update must be reverted together.
- Deletions in Step 4 are recoverable via `git show <sha>:<path>` if a regression surfaces; tag the cleanup SHA (`g2-medlow-cleanup-YYYYMMDD`) before deletions.
- Statistical change in Step 12 (ADF) ships behind a feature flag `USE_STATSMODELS_ADF` (default `True` in dev, `False` in prod for one release) so reverting is a single-flag flip while we observe cointegration signal drift.
- Performance vectorization (Step 7): rollback only affects latency, never correctness. Add wallclock baseline metric before merging so rollback decisions are data-driven.
- New test files (Step 13) are net-additive; rollback is `git rm`.

---

## 7. Dependencies

This residual cluster is **independent** of A–F and G2_a:

- No source-code coupling with the critical/high G2_a slice; all of those findings address pre-existing different code paths.
- Cross-scope crumbs (`15-test-suite`, `18-docs-health`, `07-database-persistence`, `11-backend-utils-shared`, `13-infra-deployment`, `01-backend-api`, `16-config-secrets`) are documented per-finding but do not block sequencing — the corresponding workpapers (F, G6) hold the canonical fixes; this slice contributes ancillary cleanup only.
- Recommendation: **defer to end of program**, after G2_a (critical/high) merges and stabilizes. Run as a single multi-PR sweep over 2 weeks.

---

## 8. Effort & Cost

Bulk work, dominated by mechanical edits and test authoring:

| Category | Findings | Hours |
|---|---|---|
| Lint + typo + filename | 8 | ~3.5 |
| Doc drift | 7 | ~5.5 |
| Dead-code deletions | 5 | ~7.5 |
| Performance vectorization | 4 | ~13 |
| Async lifecycle | 3 | ~2 |
| Architectural state-leaks | 4 | ~5 |
| XCom + threshold | 2 | ~1.5 |
| Engine consolidation + report stubs + sector | 3 | ~13 |
| Statistical (ADF) | 1 | ~4 |
| Testing gap fills | 4 | ~19 |
| DRY extractions + Reddit + UA | 4 | ~5 |
| Online-learner decision | 1 | ~2 (option B) |
| Scanner stub | 1 | ~3 |
| **Total** | **47** | **~84 h** |

Note: the slice header guidance suggested ~40–60h. Actual rollup is ~84h driven by F-05-015 (16h extractor consolidation), F-05-014 (8h streaming tests), F-04-016 (4h trading-agents tests), and F-09-011 (8h recommendation-engine refactor). If F-05-015 is descoped to a follow-up (it is one of two `requires_human_ack` items), residual lands at ~68h.

**Loki cost estimate:** ~$3–5 (most steps are <500 LOC mechanical edits well within Loki's strike zone; Step 7 perf rewrites and Step 13 test authoring contribute the bulk of token spend). Add ~$1 buffer if Step 4 deletions trigger broad codebase re-greps.

---

## 9. Loki-Actionable

| ID | Loki-actionable | Notes |
|---|---|---|
| F-03-009 | yes | mechanical |
| F-03-010 | **no** — `requires_human_ack: true` | wire vs. deprecate is a product decision |
| F-03-011 | yes (delete), `requires_human_ack: true` | module deletion |
| F-03-012 | yes | vectorization |
| F-03-013 | yes | doc-only |
| F-03-015 | yes (delete), `requires_human_ack: true` | dead method removal |
| F-03-016 | yes | doc-only (or impl decision deferred) |
| F-04-010 | yes | string fix |
| F-04-011 | yes | mechanical |
| F-04-012 | yes | rename + import update |
| F-04-013 | yes | refactor |
| F-04-014 | yes | flag-gate |
| F-04-015 | yes | scoped refactor |
| F-04-016 | yes | new test files |
| F-04-017 | yes | path rebase |
| F-04-018 | yes | import cleanup |
| F-04-019 | yes | metadata bump |
| F-04-020 | **no** — `requires_human_ack: true` | dead-class deletion |
| F-04-021 | yes | trivial |
| F-04-022 | **no** — `requires_human_ack: true` | UX-error-shape decision |
| F-05-010 | yes (delete), `requires_human_ack: true` | dead method block |
| F-05-011 | yes | DB query swap |
| F-05-013 | yes | doc-only |
| F-05-014 | yes | new test files |
| F-05-015 | **no** — `requires_human_ack: true` | engine consolidation; architectural |
| F-05-016 | yes | depends on Step 4 |
| F-05-017 | yes | mechanical |
| F-05-018 | yes | doc-only |
| F-05-019 | yes | small refactor |
| F-05-020 | yes | dependency add + remove list |
| F-06-010 | yes | move call site |
| F-06-011 | yes | delete unused push |
| F-06-013 | yes | constant fix |
| F-06-014 | yes | import cleanup |
| F-06-015 | **no** — `requires_human_ack: true` | doc-vs-code-fix decision; recursive SQL adoption |
| F-06-016 | yes | mechanical |
| F-09-011 | yes | scoped refactor |
| F-09-012 | yes | dataclass field add |
| F-09-013 | yes | replace `pass` with raise |
| F-09-014 | yes, **`requires_human_ack: true`** | statistical signal change |
| F-09-015 | yes | vectorization |
| F-09-016 | yes | vectorization |
| F-09-017 | yes | new test file |
| F-09-018 | yes | new test file |
| F-09-019 | yes | doc-only |
| F-09-020 | yes (delete), `requires_human_ack: true` | shim removal touches test patches |
| F-09-021 | yes | DRY extraction |

**Net:** 42/47 fully Loki-actionable; 5 require human ack (F-03-010, F-05-015, F-06-015, F-09-014, plus the deletion-gated ones flagged above for explicit go-ahead before module/class deletion lands).

---

## 10. Risks

Risk profile is intentionally low — this is residual cleanup behind G2_a:

- **Statistical drift (F-09-014):** Replacing the bespoke ADF tau approximation with `statsmodels.adfuller` will change cointegration p-values on existing pairs. Mitigation: feature-flag rollout, golden-dataset diff before merge.
- **Test patch breakage (F-09-020):** Removing `RecommendationEngine` shim methods can break tests that `mock.patch.object(RecommendationEngine, "_normalize_score")`. Mitigation: convert to module-level patches in the same PR; run full `pytest` before merge.
- **Engine consolidation (F-05-015, F-09-011):** Dual-implementation removal can mask latent callers. Mitigation: keep deprecation shim with `warnings.warn(DeprecationWarning)` for one release before deletion.
- **EMA-as-SMA disclosure (F-06-015):** Documenting the approximation may surprise downstream consumers. Mitigation: changelog entry; offer `EMA_RECURSIVE_SQL` opt-in toggle.
- **User-agent rotation (F-05-020):** Switching to `fake-useragent` introduces a runtime dependency that fetches definitions from the network. Mitigation: pin to `ua-generator` (offline) or vendor a 2025-refreshed static list.
- **Online-learner decision (F-03-010):** Marking EXPERIMENTAL is conservative; if the team expects the feature live, option (A) wires it in instead. No production exposure either way.
- **Async/event-loop (F-06-016):** `asyncio.run` raises if called inside an existing loop. Mitigation: caller analysis confirmed Airflow tasks invoke top-level — safe.
- **Performance regressions:** None expected; all perf changes are strict speedups. Mitigation: microbenchmarks added under `backend/tests/perf/`.

Aggregate residual risk: **low**. Land after G2_a stabilizes; expect 2–4 PRs over 2 weeks.

---

_47 finding IDs referenced (F-03-009, F-03-010, F-03-011, F-03-012, F-03-013, F-03-015, F-03-016, F-04-010, F-04-011, F-04-012, F-04-013, F-04-014, F-04-015, F-04-016, F-04-017, F-04-018, F-04-019, F-04-020, F-04-021, F-04-022, F-05-010, F-05-011, F-05-013, F-05-014, F-05-015, F-05-016, F-05-017, F-05-018, F-05-019, F-05-020, F-06-010, F-06-011, F-06-013, F-06-014, F-06-015, F-06-016, F-09-011, F-09-012, F-09-013, F-09-014, F-09-015, F-09-016, F-09-017, F-09-018, F-09-019, F-09-020, F-09-021)._
