# G2 — ML / Data Critical+High Residual

Cluster: `G2_ml_data_a_crit_high`
Severity: critical + high
Findings: 41
Scopes: 03-ml-engine, 04-trading-agents, 05-data-ingestion-etl, 06-airflow-pipelines, 09-analytics

---

## 1. Cluster overview

This cluster is the residual ML/data critical+high backlog NOT folded into clusters A–F (which absorbed cross-cutting themes such as test infra, secrets, CVEs, schema-cohesion, etc.). It groups by the actual fix surface, which falls into five sub-themes:

- **A. ML pipeline correctness (03-ml-engine)** — torch.load CVE class, broken XGBoost training (R²<0), path traversal in model loader, hardcoded log paths, feature-store scope drift.
  Findings: F-03-002, F-03-004, F-03-006, F-03-007, F-03-008.
- **B. Trading-agents correctness (04-trading-agents)** — the canonical "fundamentals=news" one-line bug, hardcoded developer path, FD leak, provider/embedding mismatch, OpenAI `store=True` data residency, wildcard imports, partial+positional signature, broken setup.py deps.
  Findings: F-04-001, F-04-002, F-04-003, F-04-004, F-04-005, F-04-006, F-04-007, F-04-008, F-04-009.
- **C. Ingestion correctness (05-data-ingestion-etl)** — selenium top-level import breaks the package, infinite-loop in distributed orchestrator, SmartDataFetcher entirely stubbed, dual `ExtractionResult` dataclasses, SEC EDGAR placeholder email, `self.extractor` AttributeError, sqlite leak, Kafka auto-commit on financial events.
  Findings: F-05-001, F-05-002, F-05-004, F-05-005, F-05-006, F-05-007, F-05-008, F-05-009.
- **D. Airflow DAG fixes (06-airflow-pipelines)** — Airflow 1.x→2.x removed imports, missing `numpy` import, non-existent `check_recent_data_quality` method, VACUUM in transaction, deprecated `schedule_interval`, removed `create_session`, deprecated dead code, evaluate-on-random, placeholder alert email.
  Findings: F-06-001, F-06-002, F-06-003, F-06-004, F-06-005, F-06-006, F-06-007, F-06-008, F-06-009.
- **E. Analytics correctness (09-analytics)** — sentiment_engine wrong signature, OptimizedRecommendationEngine calls non-existent `scan_market_streaming` (architectural decision required: revert vs fix-forward), DCF mutates self, dataclass field drift, Hybrid engine references missing fields, sentiment stubs, fake Johansen test, hardcoded risk-free rate, hardcoded portfolio size, MPT ignores targets.
  Findings: F-09-001, F-09-002, F-09-003, F-09-004, F-09-005, F-09-006, F-09-007, F-09-008, F-09-009, F-09-010.

Sub-theme D (Airflow) is the **broadest blast radius** — DAG parse failures block ML training entirely, which is upstream of sub-theme A (ML correctness). Sub-themes B/C/E are independent. The flagship one-liner is F-04-002.

## 2. Member findings (all 41)

**03-ml-engine (5):** F-03-002, F-03-004, F-03-006, F-03-007, F-03-008.
**04-trading-agents (9):** F-04-001, F-04-002, F-04-003, F-04-004, F-04-005, F-04-006, F-04-007, F-04-008, F-04-009.
**05-data-ingestion-etl (8):** F-05-001, F-05-002, F-05-004, F-05-005, F-05-006, F-05-007, F-05-008, F-05-009.
**06-airflow-pipelines (9):** F-06-001, F-06-002, F-06-003, F-06-004, F-06-005, F-06-006, F-06-007, F-06-008, F-06-009.
**09-analytics (10):** F-09-001, F-09-002, F-09-003, F-09-004, F-09-005, F-09-006, F-09-007, F-09-008, F-09-009, F-09-010.

Total = 5 + 9 + 8 + 9 + 10 = **41**.

## 3. Sequenced fix steps (root-cause first, fail-first for assertion-verb findings)

Steps within a sub-theme are ordered by dependency; sub-themes themselves can run in parallel except where noted. **Fail-first**: each assertion-verb finding gets a regression test that *fails on current main* before code change.

### Sub-theme B — Trading-agents (start here, contains the trivial flagship)

1. **F-04-002 (canonical 1-line bug, fail-first).** Path verified: `backend/TradingAgents/tradingagents/agents/managers/risk_manager.py:14` reads `fundamentals_report = state["news_report"]`.
   - Add unit test: build a `state` where `news_report="N"` and `fundamentals_report="F"`, call `risk_manager_node`, assert the LLM-prompt string contains `"F"` and is *not* `"N\n\nN"`. Run — must fail.
   - Change line 14 to `fundamentals_report = state["fundamentals_report"]`. Run — must pass.
2. **F-04-003 (FD leak).** Replace bare `open()` with `with` context manager; add pytest with `mock_open` asserting `__exit__` called.
3. **F-04-008 (trader_node signature).** Refactor to closure factory; add `sender` to `AgentState`. Mypy-strict gate.
4. **F-04-001 (hardcoded `/Users/yluo/...`).** Replace with `os.getenv("TRADINGAGENTS_DATA_DIR", ...)`; assert in fresh-env smoke test.
5. **F-04-004 (provider-incompatible embeddings).** Gate `OpenAI` client on provider; raise `NotImplementedError` for unsupported providers.
6. **F-04-005 (OpenAI `store=True`).** Add config flag `openai_store_responses` defaulting `False`; thread through three callsites.
7. **F-04-006 (OpenRouter/Ollama header config).** Add `default_headers` for OpenRouter, `model_kwargs` for Ollama.
8. **F-04-007 (wildcard imports).** Replace `from tradingagents.agents import *` in `setup.py:8`, `trading_graph.py:15`, `agent_states.py:5` with explicit symbol lists; mypy-strict gate.
9. **F-04-009 (setup.py deps).** Add `langchain-anthropic>=0.1.0`, `langchain-google-genai>=1.0.0`; pin `langchain-openai>=0.1.0`. Verify in fresh venv.

### Sub-theme D — Airflow (do EARLY: blocks ML training)

10. **F-06-001 (Airflow 1.x imports, broken_import, fail-first).** `airflow dags list` will surface ImportError. Replace `airflow.operators.python_operator` → `airflow.operators.python`, `bash_operator` → `bash`, `external_task_sensor` → `external_task`, retain `airflow.utils.dates.days_ago` (still valid in 2.x; consider `pendulum`/`datetime` per Airflow 2.4+ deprecation).
11. **F-06-002 (`np` undefined NameError, fail-first).** Add regression test importing `evaluate_models` and calling it — assert no `NameError`. Add `import numpy as np` at top of `ml_training_pipeline_dag.py`.
12. **F-06-003 (`check_recent_data_quality` missing AttributeError, fail-first).** Path verified: `data_pipelines/airflow/dags/ml_training_pipeline_dag.py:169` calls method absent on `DataQualityChecker`. Decision: **rewrite DAG to call existing `generate_quality_report`** (cheaper than implementing). Acceptance: DAG `check_data_quality` task succeeds in test run.
13. **F-06-006 (`create_session` removed, broken_import).** Replace runtime pool creation with deployment-time `airflow pools set ...`; remove call entirely.
14. **F-06-005 (`schedule_interval` deprecated).** Find/replace across all 3 DAGs.
15. **F-06-004 (VACUUM in transaction, runtime bug).** Use raw `psycopg2` connection with `set_isolation_level(0)`.
16. **F-06-008 (evaluate-on-random).** Persist held-out test set during training; load in evaluate. Tied to F-03-004 fix.
17. **F-06-009 (`ml-alerts@company.com`).** Replace with Airflow Variable.
18. **F-06-007 (dead code cleanup).** Delete `DEPRECATED` legacy block.

### Sub-theme C — Ingestion

19. **F-05-001 (selenium top-level import, broken_import, fail-first).** `python3 -c "from backend.etl.data_extractor import DataExtractor"` currently `ModuleNotFoundError`. Wrap import in `try/except` with `SELENIUM_AVAILABLE = False`. Acceptance command must exit 0.
20. **F-05-007 (`self.extractor` AttributeError, fail-first).** Path verified: `backend/etl/etl_orchestrator.py:387,633`. Add `self.extractor = self.legacy_extractor` in `__init__`. Test via `pytest -k realtime`.
21. **F-05-005 (dual `ExtractionResult`).** Consolidate into `backend/etl/types.py`; both extractors re-export.
22. **F-05-002 (orchestrator infinite loop).** Add `max_wait_seconds`; count failed jobs toward exit; cancel `processing_task` in `finally`.
23. **F-05-008 (sqlite leak).** Replace 8 raw connections with `with sqlite3.connect(...) as conn:`.
24. **F-05-004 (SmartDataFetcher stub).** Implement real fetch by delegating to existing `FinnhubClient`/`AlphaVantageClient`/`SECEdgarClient`/`PolygonClient`.
25. **F-05-006 (SEC EDGAR placeholder email).** Read `SEC_EDGAR_CONTACT_EMAIL` env; fail loudly at startup if empty.
26. **F-05-009 (Kafka auto-commit).** Set `enable_auto_commit=False`; manual commit after success. Note: marked `loki_actionable=false` — coordinate with on-call/streaming owner.

### Sub-theme A — ML engine

27. **F-03-002 (torch.load CVE).** Add `weights_only=True` to all 5 callsites; for full-model loads use `torch.serialization.add_safe_globals`. Acceptance grep returns empty.
28. **F-03-006 (path traversal in model loader, security).** Validate `model_name` against regex `[a-zA-Z0-9_-]+`; assert resolved path within `models_path`. Add curl-based test (returns 400 not 500).
29. **F-03-007 (relative log path).** Replace with `Path(__file__).parent.parent.parent / "ml_logs" / ...`.
30. **F-03-004 (XGBoost R²<0, all importances zero, fail-first).** Add gate test asserting `r2 > 0.0` AND at least one non-zero feature importance — currently fails. Audit pipeline: target construction, train/test split look-ahead, feature preprocessing. Likely cross-cuts F-05-004 (mock data) and F-06-008.
31. **F-03-008 (feature-store scope drift).** Either implement MACD + Bollinger Bands compute methods, or update `ML_PIPELINE_DOCUMENTATION.md` to match actual surface. Default = implement (closes the gap).

### Sub-theme E — Analytics

32. **F-09-001 (sentiment wrong signature, fail-first).** Add pytest that calls `_run_sentiment_analysis` with realistic args — currently raises `AttributeError`. Replace with `await self.sentiment_engine.analyze_stock_sentiment(ticker, [item['text'] for item in text_data])`.
33. **F-09-002 (`scan_market_streaming` does not exist) — ARCHITECTURAL DECISION.** Per synthesis-handoff §6, **default = revert `OptimizedRecommendationEngine` to bundled engine**. This is destructive (removes optimization work) → `requires_human_ack: true` (see §9). Alternative fix-forward: implement `scan_market_streaming` as async generator wrapping existing `scan_market` (single-yield) — low-risk patch that keeps the optimized engine alive. **Recommend fix-forward (single-yield wrapper)** as it has the same code surface as a revert but preserves the optimized path. Human ack still required to choose.
34. **F-09-004 (ranking_score field drift).** Add `ranking_score: float = field(default=0.0)` to `StockRecommendation`. Assert `dataclasses.asdict(rec)` contains the key.
35. **F-09-005 (`overall_score`/`recommendation` missing on `EnhancedStockRecommendation`).** Remove unsupported kwargs in `_create_error_recommendation`; replace `self.overall_score` with `self.confidence` or computed combination. Add error-path unit test.
36. **F-09-003 (DCF mutates self).** Make `terminal_growth_rate` a method parameter; restore `self` not modified after `sensitivity_analysis`.
37. **F-09-008 (risk-free rate hardcoded local).** Promote to class attribute / config; sync with `FundamentalAnalysisEngine.risk_free_rate`.
38. **F-09-009 (portfolio size hardcoded $100K).** Accept `portfolio_size` parameter; thread through.
39. **F-09-007 (Johansen → Engle-Granger).** Implement via `statsmodels.tsa.vector_ar.vecm.coint_johansen` OR remove `JOHANSEN` enum. Default = remove enum (cheaper, honest).
40. **F-09-010 (MPT ignores targets).** Replace with `scipy.optimize` mean-variance solver, or delegate to existing `backend/utils/portfolio_optimizer.py`.
41. **F-09-006 (sentiment stubs).** Marked `loki_actionable=false` — depends on data ingestion layer. At minimum log loud warning + raise `NotImplementedError` until F-05-004 lands.

## 4. Files touched

ML (A): `backend/ml/model_manager.py`, `backend/ml/model_versioning.py`, `backend/ml/artifact_manager.py`, `backend/ml/training/evaluate_models.py`, `backend/ml/ml_api_server.py`, `backend/ml/training_pipeline.py`, `backend/ml/simple_training_pipeline.py`, `backend/ml/feature_store.py`.

Trading-agents (B): `backend/TradingAgents/tradingagents/agents/managers/risk_manager.py`, `backend/TradingAgents/tradingagents/dataflows/finnhub_utils.py`, `backend/TradingAgents/tradingagents/agents/utils/memory.py`, `backend/TradingAgents/tradingagents/dataflows/interface.py`, `backend/TradingAgents/tradingagents/graph/trading_graph.py`, `backend/TradingAgents/tradingagents/graph/setup.py`, `backend/TradingAgents/tradingagents/agents/agent_states.py`, `backend/TradingAgents/tradingagents/agents/trader/trader.py`, `backend/TradingAgents/tradingagents/default_config.py`, `backend/TradingAgents/setup.py`.

Ingestion (C): `backend/etl/unlimited_data_extractor.py`, `backend/etl/etl_orchestrator.py`, `backend/etl/multi_source_extractor.py`, `backend/etl/data_extractor.py`, `backend/etl/types.py` (new), `backend/data_ingestion/smart_data_fetcher.py`, `backend/data_ingestion/sec_edgar_client.py`, `backend/etl/distributed_batch_processor.py`, `backend/streaming/kafka_client.py`.

Airflow (D): `data_pipelines/airflow/dags/ml_training_pipeline_dag.py`, `data_pipelines/airflow/dags/daily_stock_pipeline.py`, `data_pipelines/airflow/dags/enhanced_stock_pipeline.py`.

Analytics (E): `backend/analytics/recommendation_engine.py`, `backend/analytics/recommendation_optimized.py`, `backend/analytics/recommendation_ranking.py`, `backend/analytics/recommendation_scoring.py`, `backend/analytics/recommendation_types.py`, `backend/analytics/agents/hybrid_engine.py`, `backend/analytics/sentiment_analysis.py`, `backend/analytics/fundamental/valuation/dcf_model.py`, `backend/analytics/statistical/cointegration_analyzer.py`, `backend/analytics/portfolio/modern_portfolio_theory.py`, `backend/analytics/scanner.py` (MarketScanner, for F-09-002 fix-forward).

Tests touched: matching test modules under `backend/tests/...` and `data_pipelines/airflow/tests/...`.

## 5. Acceptance tests (consolidated)

Organized by hint-style; each maps to one or more findings.

- **Greps that must return empty:**
  - `grep -rn "torch.load" backend/ml/ | grep -v weights_only` (F-03-002)
  - `grep -n "DEPRECATED\|Legacy" data_pipelines/airflow/dags/daily_stock_pipeline.py` (F-06-007)
  - `grep "company.com" data_pipelines/airflow/dags/ml_training_pipeline_dag.py` (F-06-009)
  - `grep -n "schedule_interval" data_pipelines/airflow/dags/*.py` (F-06-005)
- **Import smoke tests (must exit 0):**
  - `python3 -c "from backend.etl.data_extractor import DataExtractor"` (F-05-001)
  - `python -c "from data_pipelines.airflow.dags.daily_stock_pipeline import dag"` (F-06-006)
  - `python -c "from data_pipelines.airflow.dags.ml_training_pipeline_dag import evaluate_models"` (F-06-002)
  - `python -c "from tradingagents.graph.trading_graph import TradingAgentsGraph"` in fresh venv (F-04-009)
- **Pytest gates:**
  - `risk_manager_node` returns prompt with fundamentals (F-04-002, fail-first)
  - `_run_sentiment_analysis` returns SentimentResult (F-09-001, fail-first)
  - `pytest backend/tests/unit/test_etl_modules.py -k realtime` passes (F-05-007)
  - `dataclasses.asdict(rec)` contains `ranking_score` (F-09-004)
  - Error fallback path of hybrid engine raises no `TypeError` (F-09-005)
  - `model.terminal_growth_rate` unchanged after `sensitivity_analysis` (F-09-003)
  - `engine.generate_daily_recommendations(max_recommendations=5)` no `AttributeError` (F-09-002)
- **Quality gates / metrics:**
  - `r2 > 0.0` and ≥1 non-zero feature importance (F-03-004)
  - Efficient frontier monotonically increasing volatility w/ return (F-09-010)
  - Mock-fail-all integration test returns within 60s (F-05-002)
  - FD count stable over 100 iterations (F-04-003, F-05-008)
  - SEC `User-Agent` does not contain `example.com` (F-05-006)
- **Security gates:**
  - `curl -X POST .../models/../../../etc/passwd/load` → 400 (F-03-006)
  - Config-driven `store` kwarg matches setting (F-04-005)
- **CLI/airflow gates:**
  - `airflow dags list` shows all DAGs without RemovedInAirflow3Warning or ImportError (F-06-001, F-06-005, F-06-006)
  - `cleanup_and_optimize` task success (F-06-004)
  - `check_data_quality` task success (F-06-003)
- **Mypy/static:**
  - `mypy --strict graph/setup.py` passes (F-04-007)
  - `mypy` passes on `trader.py`; `AgentState.__annotations__` contains `sender` (F-04-008)
- **Behavioural:**
  - Anthropic-config `reflect_and_remember` no crash (F-04-004)
  - OpenRouter mocked endpoint shows required headers (F-04-006)
  - `analyze_comprehensive_sentiment("AAPL")["sources_analyzed"] > 0` (F-09-006, deferred)
  - `Sharpe`/`max_position_size` use config sources (F-09-008, F-09-009)
  - Johansen returns clearly-different result OR enum removed (F-09-007)
  - `feature_store.builtin_features` includes MACD + Bollinger (F-03-008)

## 6. Rollback plan per sub-theme

- **A (ML):** Revert per-finding commits. F-03-004 (XGBoost) — keep training data pipeline branch behind a feature flag `ML_R2_GATE_ENABLED` so a regression can fall back to legacy training without re-deploying. Pickle/torch loader hardening (F-03-002, F-03-006) — purely additive guards; revert is safe.
- **B (Trading-agents):** All changes are isolated within `backend/TradingAgents/tradingagents/...`. Rollback = `git revert <commit>`. F-04-002 is a one-character revert. F-04-009 (setup.py) — pin reverts cleanly.
- **C (Ingestion):** F-05-001 selenium-guard rollback would re-break the package, but original was already broken — never roll back. F-05-002, F-05-008, F-05-009 — wrap in feature flag `ETL_HARDENED_LOOP` so rollback to legacy loop is a config flip.
- **D (Airflow):** Each DAG is independent — pause failing DAG in Airflow UI, revert single DAG file. Pool-creation removal (F-06-006) requires re-applying `airflow pools set` at deployment; document in runbook.
- **E (Analytics):** F-09-002 — see §9 (architectural decision). All other E fixes are local; revert per commit. F-09-004 (dataclass field) — additive default, safe.

## 7. Dependencies

- **Independent of A/B/C clusters** (those handled secrets, schema cohesion, test infra, etc. — none rewrite the same files).
- **Soft-depends on E (test signal)** — once E lands testing infrastructure, the fail-first tests in this cluster (F-04-002, F-09-001, F-05-007, F-06-002, F-06-003, F-03-004) become enforceable in CI rather than ad-hoc.
- **Internal sequencing inside G2:**
  - Sub-theme D (Airflow) gates sub-theme A (ML) work that requires the training DAG to load, particularly F-06-008 ↔ F-03-004 ↔ F-06-002 (real eval data).
  - F-09-006 (sentiment stubs) waits on F-05-004 (SmartDataFetcher stub) — both rely on the ingestion clients.
  - F-09-001 should land before F-09-002 (sentiment correctness is a precondition for end-to-end recommendation integration test).
- **Cross-scope edges noted on findings:** F-03-006↔08-auth-security, F-04-001/05/09↔16-config-secrets, F-04-005↔08-auth-security, F-05-004↔02-backend/09-analytics, F-05-006↔08-auth-security, F-05-009↔10-monitoring, F-06-002↔03-ml-engine, F-06-003↔11-backend-utils-shared, F-06-008↔03-ml-engine, F-09-002↔05-ingestion, F-09-006↔05-ingestion, F-09-009↔02-backend.

## 8. Effort & cost

Sum of `effort_hours` across the 41 findings:

- A (ML): 3+8+2+1+8 = **22h**
- B (Trading-agents): 1+0.5+0.5+3+1+2+2+1+1 = **12h**
- C (Ingestion): 2+4+8+3+1+1+4+8 = **31h**
- D (Airflow): 1+0.5+2+1+0.5+1+2+4+0.5 = **12.5h**
- E (Analytics): 4+6+2+2+3+16+8+2+2+16 = **61h**

**Total: 138.5 engineer-hours** (~3.5 engineer-weeks at 40h/week).

Cost notes:
- Trivial wins (≤1h): F-04-002 (0.5), F-04-003 (0.5), F-06-002 (0.5), F-06-005 (0.5), F-06-009 (0.5), F-04-008 (1), F-04-001 (1), F-03-007 (1), F-04-005 (1), F-04-009 (1), F-05-006 (1), F-05-007 (1), F-06-001 (1), F-06-004 (1), F-06-006 (1) = ~12.5h for 15 findings → **fast PR slate**.
- Highest-cost stragglers (≥8h): F-03-004 (8), F-03-008 (8), F-05-004 (8), F-05-009 (8), F-09-006 (16), F-09-007 (8), F-09-010 (16) = 72h, ~half the cluster.

## 9. Loki-actionable

- **Loki-actionable (no human ack required), 38 findings:** F-03-002, F-03-004, F-03-006, F-03-007, F-03-008, F-04-001, F-04-002, F-04-003, F-04-004, F-04-005, F-04-006, F-04-007, F-04-008, F-04-009, F-05-001, F-05-002, F-05-004, F-05-005, F-05-006, F-05-007, F-05-008, F-06-001, F-06-002, F-06-003, F-06-004, F-06-005, F-06-006, F-06-007, F-06-008, F-06-009, F-09-001, F-09-003, F-09-004, F-09-005, F-09-007, F-09-008, F-09-009, F-09-010.
- **Loki-actionable=false (per source) — needs human-in-loop:** F-05-009 (Kafka auto-commit — coordinate w/ streaming/on-call owner), F-09-006 (sentiment stubs — depends on ingestion product decision).
- **`requires_human_ack: true` even though `loki_actionable=true`:** **F-09-002** (`OptimizedRecommendationEngine` revert vs fix-forward). The default in synthesis-handoff §6 is *revert to bundled engine*, which is destructive (removes optimization work). This workpaper recommends the fix-forward (single-yield async-generator wrapper around existing `scan_market`) as same blast-radius and preserves the optimized path. Human owner must choose between revert and fix-forward before Loki proceeds.

## 10. Rollout risks

- **Airflow re-parse risk:** Replacing `schedule_interval`→`schedule` (F-06-005) and removing `create_session` (F-06-006) can cause silent re-schedule from epoch on new DAG ID hash. Mitigation: deploy DAG changes during quiet window; verify last-success timestamp pre/post.
- **Pickle-loader tightening (F-03-002):** `weights_only=True` rejects checkpoints saved with custom classes. If existing `.pth` artifacts contain non-tensor objects, loading will fail until `add_safe_globals` is configured. Inventory checkpoints before rollout.
- **F-09-002 (architectural):** Either path (revert or fix-forward) changes recommendation engine behavior. Production-side smoke-test required before traffic switch — see §9.
- **F-05-004 (SmartDataFetcher real impl):** Currently every consumer receives zeros; flipping to real data will change downstream metrics, ML training inputs, and dashboards. Coordinate with cluster A (XGBoost retraining) since real features may further shift R². Treat as a coupled change with F-03-004.
- **F-04-009 (setup.py deps):** New package pins (`langchain-anthropic`, `langchain-google-genai`) increase install surface; CI must verify fresh-venv install.
- **Kafka auto-commit flip (F-05-009):** Switching to manual commits requires consumer code to commit explicitly. Misordered commits can cause duplicate processing on first deploy. Roll out behind feature flag and shadow-consume for 24h.
- **F-05-001 (selenium guard):** After fix, ETL package imports succeed but selenium-dependent paths now silently fall back. Add health endpoint that surfaces `SELENIUM_AVAILABLE` so ops sees degraded mode.
- **F-06-003 / F-06-008 chain:** Real eval data and real `check_recent_data_quality` may surface latent data-quality issues that previously hid behind the AttributeError/random-data masks. Expect a wave of Airflow alerts on first real run — pre-warn on-call.
- **Test load:** Fail-first regressions add ~6 new tests to CI; ensure test runtime budget is acceptable.

---

**Assertion (final):** All 41 findings referenced — F-03-002, F-03-004, F-03-006, F-03-007, F-03-008, F-04-001, F-04-002, F-04-003, F-04-004, F-04-005, F-04-006, F-04-007, F-04-008, F-04-009, F-05-001, F-05-002, F-05-004, F-05-005, F-05-006, F-05-007, F-05-008, F-05-009, F-06-001, F-06-002, F-06-003, F-06-004, F-06-005, F-06-006, F-06-007, F-06-008, F-06-009, F-09-001, F-09-002, F-09-003, F-09-004, F-09-005, F-09-006, F-09-007, F-09-008, F-09-009, F-09-010 — count = 41.
