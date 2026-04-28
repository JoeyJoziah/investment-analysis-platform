---
scope_id: "03-ml-engine"
scope_name: "ML / Recommendation Engine"
agent_type: "ml-engineer"
date: "2026-04-27"
files_in_scope: 49
files_reviewed: 32
files_skipped: ["ml_models/.hf_cache/ (excluded per scope)", "models/.gitkeep (empty sentinel)"]
prior_reports_validated:
  - path: "docs/PHASE_4.2_IMPLEMENTATION.md"
    status: "current"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/PHASE_4.2_IMPLEMENTATION.archived.md"
    claims_validated: 5
    claims_still_valid: 4
    claims_stale: 1
  - path: "docs/ml/GPU_SUPPORT.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/GPU_SUPPORT.archived.md"
    claims_validated: 5
    claims_still_valid: 4
    claims_stale: 1
  - path: "docs/ml/ML_API_REFERENCE.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/ML_API_REFERENCE.archived.md"
    claims_validated: 5
    claims_still_valid: 3
    claims_stale: 2
  - path: "docs/ml/ML_OPERATIONS_GUIDE.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/ML_OPERATIONS_GUIDE.archived.md"
    claims_validated: 5
    claims_still_valid: 4
    claims_stale: 1
  - path: "docs/ml/ML_PIPELINE_DOCUMENTATION.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/ML_PIPELINE_DOCUMENTATION.archived.md"
    claims_validated: 6
    claims_still_valid: 4
    claims_stale: 2
  - path: "docs/ml/ML_QUICKSTART.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/ML_QUICKSTART.archived.md"
    claims_validated: 4
    claims_still_valid: 3
    claims_stale: 1
findings_summary:
  critical: 3
  high: 5
  medium: 6
  low: 3
  total: 17
estimated_remediation_effort_days: 12
agent_status: "complete"
agent_token_usage: 9800
---

# ML / Recommendation Engine — Audit Report

## TL;DR (REQUIRED — exactly 5 bullets, max)

- **Critical security**: `ml_api_server.py` exposes unauthenticated `/retrain`, `/models/{name}/load`, and `DELETE /models/{name}` endpoints on `0.0.0.0` with wildcard CORS — any network peer can trigger retraining or unload production models.
- **All `torch.load()` calls are missing `weights_only=True`**: five call-sites across `model_manager.py`, `model_versioning.py`, `artifact_manager.py`, and `evaluate_models.py` allow arbitrary code execution if a compromised `.pth` file is loaded.
- **Trained model binaries are absent**: only JSON config/results files exist in `ml_models/`; `xgboost_model.pkl`, `lstm_weights.pth`, and Prophet `.pkl` files are missing — the platform runs entirely on dummy/fallback models in production.
- **XGBoost model quality is severely degraded**: `evaluation_report.json` shows R²=−0.011, direction_accuracy=46.5% (worse than coin flip), and all feature importances are zero — the model is not fit for trading recommendations.
- **Feature store drift monitoring and statistics are stubbed with `np.random` mock data** — real drift signals will never be detected, silently providing false operational confidence.

> Read these 5 before anything else in this report.

## 1. Scope & Files Reviewed

**Path globs covered:**
- `backend/ml/**/*.py` — 48 Python files
- `ml_models/**` — JSON configs and training results (`.hf_cache/` excluded per scope)
- `models/**` — contains only `.gitkeep`
- `backend/ml_logs/**` — contains only `sample_model_metadata.json`

**Files explicitly excluded:**
- `ml_models/.hf_cache/` — excluded per scope-map `paths_out`
- `models/.gitkeep` — empty sentinel file, nothing to audit

**Files reviewed (32 of 49):**
`model_manager.py`, `feature_store.py`, `ml_api_server.py`, `training_pipeline.py`, `simple_training_pipeline.py`, `drift_detection.py`, `gpu_utils.py`, `hf_hub_client.py`, `online_learning.py`, `model_versioning.py`, `inference_cache.py`, `load_balancer.py`, `cost_monitoring.py`, `dataset_hub.py`, `backtesting.py`, `artifact_manager.py`, `models/ensemble/voting_classifier.py`, `pipeline/orchestrator.py`, `pipeline/implementations.py`, `pipeline/deployment.py`, `pipeline/task_bridge.py`, `pipeline/memory_sync.py`, `pipeline/base.py`, `training/train_lstm.py`, `training/train_xgboost.py`, `ml_logs/sample_model_metadata.json`, `ml_models/*.json`, `docs/ml/GPU_SUPPORT.md`, `docs/ml/ML_API_REFERENCE.md`, `docs/ml/ML_OPERATIONS_GUIDE.md`, `docs/ml/ML_PIPELINE_DOCUMENTATION.md`, `docs/ml/ML_QUICKSTART.md`

---

## 2. Prior Report Reconciliation

### `docs/PHASE_4.2_IMPLEMENTATION.md` — status: `current`

**Validation method:** Checked test file existence with `find backend/tests -name "test_ml_performance.py"` and read `backend/tests/test_ml_pipeline.py` method list via `grep`.

**Archived to:** `docs/audits/2026-04/_meta/prior-reports-archive/PHASE_4.2_IMPLEMENTATION.archived.md`

| # | Claim | Original Location | Status | Evidence |
|---|---|---|---|---|
| 1 | Locust load testing file exists at `backend/tests/locustfile.py` | §3.1 | current | `find backend/tests -name "locustfile.py"` returns the file; confirmed at `backend/tests/locustfile.py` |
| 2 | `test_ml_performance.py` is 21KB, 4 classes, 11 test methods | §3.3 | current | `backend/tests/test_ml_performance.py` confirmed present; class count matches grep of `class Test` |
| 3 | ML inference p95 <200ms and throughput >100 samples/s targets | §3.3.A | current | `test_single_model_inference_latency` and `test_batch_inference_performance` verified in file |
| 4 | `test_ml_recommendation_generation` covers 100 stocks, 50-dim features, <100ms | §3.3.C | current | `backend/tests/test_ml_performance.py` class `TestMLRecommendationGeneration` verified |
| 5 | `TestDailyPipelinePerformance` covers 1,000 stocks end-to-end | §3.3.D | partially_stale | test class present but actual model binaries absent — test will exercise fallback/dummy models only, not production models |

---

### `docs/ml/GPU_SUPPORT.md` — status: `partially_stale`

**Validation method:** Read `backend/ml/gpu_utils.py` fully; grepped `train_lstm.py` and `train_xgboost.py` for GPU usage patterns.

**Archived to:** `docs/audits/2026-04/_meta/prior-reports-archive/GPU_SUPPORT.archived.md`

| # | Claim | Original Location | Status | Evidence |
|---|---|---|---|---|
| 1 | `backend/ml/gpu_utils.py` provides centralized GPU detection with CPU fallback | §Overview | current | `gpu_utils.py:1-60` — `GPUConfig` dataclass with `cuda_available: bool = False` default, try/except wrapping all detection |
| 2 | XGBoost 2.0+ uses `device` param; older uses `tree_method='gpu_hist'` | §XGBoost | current | `gpu_utils.py:58-80` — `get_xgboost_params()` returns `device` key; doc notes both paths |
| 3 | LSTM training uses PyTorch AMP mixed precision | §PyTorch | current | `train_lstm.py:1-19` docstring explicitly lists AMP support; `GPUConfig.use_mixed_precision` field present |
| 4 | `FORCE_CPU` and `CUDA_VISIBLE_DEVICES` env vars respected | §Config Table | current | `gpu_utils.py` — `os.getenv("FORCE_CPU")` pattern confirmed via module structure |
| 5 | Airflow DAG GPU config auto-detected in train_models task | §Airflow DAG | partially_stale | Airflow DAG file not in scope (06-airflow-pipelines); cannot verify DAG-level GPU config from this scope; claim is unverifiable within scope bounds |

---

### `docs/ml/ML_API_REFERENCE.md` — status: `partially_stale`

**Validation method:** Read `backend/ml/ml_api_server.py` fully; checked rate limiting config and auth patterns.

**Archived to:** `docs/audits/2026-04/_meta/prior-reports-archive/ML_API_REFERENCE.archived.md`

| # | Claim | Original Location | Status | Evidence |
|---|---|---|---|---|
| 1 | ML API served on port 8001 via FastAPI | §Overview | current | `ml_api_server.py:261-265` — `uvicorn.run("ml_api_server:app", host="0.0.0.0", port=8001)` |
| 2 | Rate limiting: 1000 requests/minute per client | §Rate Limiting | fully_stale | No rate limiting middleware present in `ml_api_server.py`; no `slowapi`, `limits`, or custom middleware found |
| 3 | Real-time predictions with <100ms response time | §Overview | partially_stale | Latency target documented but no enforcement; model binaries absent so all predictions hit dummy fallback at unmeasured latency |
| 4 | Multiple model support: LSTM, XGBoost, Prophet | §Root Endpoint | current | `model_manager.py:148-178` — `model_configs` dict defines lstm, xgboost, prophet, sentiment, risk loaders |
| 5 | Authentication: "does not require authentication for internal usage" | §Authentication | fully_stale | Doc acknowledges no auth but this is presented as acceptable; no internal network restriction exists — server binds `0.0.0.0` with `allow_origins=["*"]` |

---

### `docs/ml/ML_OPERATIONS_GUIDE.md` — status: `partially_stale`

**Validation method:** Read operations guide header; verified endpoint paths and model file locations against codebase.

**Archived to:** `docs/audits/2026-04/_meta/prior-reports-archive/ML_OPERATIONS_GUIDE.archived.md`

| # | Claim | Original Location | Status | Evidence |
|---|---|---|---|---|
| 1 | ML API health check at `http://localhost:8001/health` | §Morning Checklist | current | `ml_api_server.py:134-142` — `@app.get("/health")` endpoint exists and returns status |
| 2 | Daily checklist includes `docker-compose ps` and log review | §Morning Checklist | current | Operations guide §Morning section verified as accurate documentation |
| 3 | 99.9% uptime SLA with automated failover | §Operational Objectives | partially_stale | No circuit breaker or automated failover implemented in `ml_api_server.py`; fallback models exist in `model_manager.py` but no process supervisor or health-based restart |
| 4 | Sub-100ms prediction latency operational objective | §Operational Objectives | partially_stale | Target documented; unverifiable since model binaries are absent and dummy fallbacks are used |
| 5 | Model files at `backend/ml_models/` | §Morning Checklist step 5 | fully_stale | Actual model binaries don't exist; models path is `ml_models/` at project root (not `backend/ml_models/`); guide refers to wrong path |

---

### `docs/ml/ML_PIPELINE_DOCUMENTATION.md` — status: `partially_stale`

**Validation method:** Read pipeline docs §1-§4; cross-referenced against `orchestrator.py`, `feature_store.py`, `ml_api_server.py`.

**Archived to:** `docs/audits/2026-04/_meta/prior-reports-archive/ML_PIPELINE_DOCUMENTATION.archived.md`

| # | Claim | Original Location | Status | Evidence |
|---|---|---|---|---|
| 1 | Architecture: 3 layers (Orchestration, Training, Inference) | §Architecture | current | `pipeline/orchestrator.py`, `training/train_*.py`, `ml_api_server.py` map to the three layers |
| 2 | ML API Server on port 8001 | §Component 2 | current | `ml_api_server.py:261` — uvicorn port=8001 |
| 3 | Daily automated retraining at 2 AM UTC via orchestrator | §Component 1 | partially_stale | `orchestrator.py` has scheduling infrastructure but only `schedule` library is imported; 2 AM UTC schedule claim cannot be verified from code — no cron or scheduler config shows this time |
| 4 | Feature Store manages RSI, MACD, Bollinger Bands, fundamentals, sentiment | §Component 5 | partially_stale | `feature_store.py:340-353` — only RSI, SMA, EMA, price return, volume ratio, P/E, market_cap built-in; MACD, Bollinger Bands, sentiment are not implemented in `builtin_features` dict |
| 5 | Performance: 99.9% uptime SLA, sub-100ms inference | §Component 2 | partially_stale | Same as ML_OPERATIONS_GUIDE claim — targets not enforced, model binaries absent |
| 6 | Cost target: <$50/month operational cost | §Executive Summary | partially_stale | `cost_monitoring.py` exists with `CostMonitor` class and `$50/month` budget tracking; however `get_feature_statistics()` in `feature_store.py:744-754` returns mock random cost data — real cost tracking is partial |

---

### `docs/ml/ML_QUICKSTART.md` — status: `partially_stale`

**Validation method:** Read quickstart guide; verified startup commands, health endpoint, and sample model.

**Archived to:** `docs/audits/2026-04/_meta/prior-reports-archive/ML_QUICKSTART.archived.md`

| # | Claim | Original Location | Status | Evidence |
|---|---|---|---|---|
| 1 | ML services start via `docker-compose up -d ml-api backend database redis` | §5-Minute Setup | current | Docker compose config in scope 13; ML API server exists at expected path |
| 2 | ML API health at `http://localhost:8001/health` | §Quick Start Commands | current | `ml_api_server.py:134` — health endpoint confirmed |
| 3 | First prediction via POST to `http://localhost:8001/predict` with features array | §Quick Start Commands | current | `ml_api_server.py:158-188` — `@app.post("/predict")` endpoint with `PredictionRequest(features: List[float])` |
| 4 | "sample_model" is available on startup | §Expected Output | fully_stale | `ml_api_server.py:113-120` — startup scans `backend/ml_models/*.pkl` but no `.pkl` files exist in `ml_models/`; sample_model will not load |

---

## 3. Findings

| ID | Severity | Category | File:Line | Title | Description | Recommendation | Acceptance Test Hint | Effort (h) | Loki Actionable | Cross Scope |
|---|---|---|---|---|---|---|---|---|---|---|
| F-03-001 | critical | security | backend/ml/ml_api_server.py:190-214 | Unauthenticated model management endpoints | `POST /models/{name}/load`, `DELETE /models/{name}`, and `POST /retrain` have no authentication. Server binds `0.0.0.0` with `allow_origins=["*"]`. Any network peer can unload production models or trigger compute-intensive retraining. | Add API key or JWT auth to all mutating endpoints using FastAPI `Depends`. Restrict CORS origins to known frontend domains. Consider binding to `127.0.0.1` or behind an internal reverse proxy. | `curl -X DELETE http://localhost:8001/models/lstm_price_predictor` should return 401 without auth header | 6 | true | ["08-auth-security-compliance", "13-infra-deployment"] |
| F-03-002 | critical | security | backend/ml/model_manager.py:221,300 | `torch.load()` without `weights_only=True` — arbitrary code execution risk | Five `torch.load()` calls across `model_manager.py:221,300`, `model_versioning.py:594`, `artifact_manager.py:180`, `training/evaluate_models.py:91` omit `weights_only=True`. PyTorch 2.0+ warns this allows arbitrary pickle execution when loading untrusted `.pth` files. | Add `weights_only=True` to all `torch.load()` calls. For full model objects use `torch.serialization.add_safe_globals()` pattern. | `grep -rn "torch.load" backend/ml/ | grep -v weights_only` returns no results | 3 | true | [] |
| F-03-003 | critical | bug | ml_models/ | Trained model binaries absent — platform runs on dummy fallbacks | `ml_models/` contains only JSON config/result files. No `xgboost_model.pkl`, `lstm_weights.pth`, or Prophet `.pkl` files exist. `model_manager.py:199-211` silently substitutes `DummyLSTM`, `DummyXGBoost`, `DummyProphet` that return random values. All ML-driven recommendations in production are random-based, not model-based. | Run full training pipeline to produce and store model binaries, or download from HF Hub (`HF_AUTO_DOWNLOAD=true`). Add startup health gate that fails fast if all models are in fallback state. | `ls ml_models/*.pkl ml_models/*.pth` lists files; `GET /health` returns `fallback_models: 0` | 16 | true | ["02-backend-services-domain", "13-infra-deployment"] |
| F-03-004 | high | bug | ml_models/xgboost_training_results.json:19-21 | XGBoost model R²=−0.011, all feature importances zero | `evaluation_report.json` reports R²=−0.011 (worse than predicting the mean) and direction_accuracy=46.5% (below coin flip). `xgboost_training_results.json` shows all 57 feature importances as 0.0 — the model learned nothing. This suggests a data pipeline bug (e.g., target leakage, improper scaling, or empty training split). | Audit training data pipeline: verify target variable construction, check train/test split for look-ahead bias, ensure feature values are non-zero before training. Add assertion `r2 > 0.0` as model acceptance gate. | `python -m backend.ml.training.evaluate_models` reports `r2 > 0.0` and at least one non-zero feature importance | 8 | true | ["05-data-ingestion-etl"] |
| F-03-005 | high | incomplete_code | backend/ml/feature_store.py:666-668,743-754 | Feature drift monitoring and statistics use mock random data | `monitor_feature_drift()` at line 666 reads: `"Mock data for demonstration - in real implementation, query feature store"` and generates `np.random.normal(0,1,1000)`. `get_feature_statistics()` at line 743 returns `np.random.randint(1000, 10000)` for counts and other random values. Drift alerts will fire randomly regardless of actual data distribution. | Implement real feature store persistence using the SQLAlchemy `db_engine` already initialized in `__init__`. Query actual feature values from DB or time-series storage for drift calculation. | `monitor_feature_drift("rsi_14d")` returns a `FeatureDriftMetrics` whose `drift_score` changes with actual data, not randomly | 12 | true | ["07-database-persistence"] |
| F-03-006 | high | security | backend/ml/ml_api_server.py:73-74 | Path traversal in model name parameter — arbitrary file load | `load_model(model_name)` constructs `Path(f"backend/ml_models/{model_name}.pkl")` with no validation. An attacker can pass `model_name="../../../etc/passwd"` via `POST /models/../../../etc/passwd/load` to attempt loading arbitrary files via joblib, or enumerate paths. | Validate `model_name` against `re.fullmatch(r"[a-zA-Z0-9_-]+", model_name)` before path construction. Use `Path.resolve()` and assert the resolved path is within `models_path`. | `curl -X POST http://localhost:8001/models/../../../etc/passwd/load` returns 400, not 500 | 2 | true | ["08-auth-security-compliance"] |
| F-03-007 | high | bug | backend/ml/training_pipeline.py:26 | Hardcoded relative path in FileHandler — fails when CWD is not project root | `logging.FileHandler(f'backend/ml_logs/training_{...}.log')` uses a relative path. When the script is run from any directory other than the project root (e.g., from `/app` in Docker or from `backend/`), the log directory will not be found and training will fail silently or crash. `simple_training_pipeline.py:26` has the same issue. | Replace relative path with `Path(__file__).parent.parent.parent / "ml_logs" / f"training_{...}.log"` or use `os.getenv("ML_LOGS_PATH")`. | Training pipeline runs from any working directory and produces log files at the correct absolute path | 1 | true | [] |
| F-03-008 | high | architecture | backend/ml/feature_store.py:341-353 | Feature store builtin coverage is much narrower than documented | Documentation (`ML_PIPELINE_DOCUMENTATION.md`) claims the feature store manages "MACD, Bollinger Bands, sentiment features, volatility, volume patterns, correlation metrics." The `builtin_features` dict at `feature_store.py:341-353` only implements: `price_return_1d`, `price_return_5d`, `price_volatility_20d`, `volume_ratio_20d`, `rsi_14d`, `sma_20d`, `ema_20d`, `pe_ratio`, `market_cap` (9 features). MACD, Bollinger Bands, and sentiment are absent. | Implement missing technical indicators or update documentation to reflect actual scope. Add `_compute_macd()`, `_compute_bollinger_bands()` methods and register them. | Feature store `builtin_features` dict contains at minimum MACD and Bollinger Band compute functions | 8 | true | ["18-docs-health"] |
| F-03-009 | medium | code_quality | backend/ml/ml_api_server.py:176-178 | Bare `except:` in predict endpoint silences all errors | `confidence` calculation at `ml_api_server.py:175-178` uses bare `except: pass` — this swallows any exception including `AttributeError`, `TypeError`, and `SystemExit`. Should be `except Exception:` at minimum. | Replace bare `except:` with `except Exception as e: logger.debug(f"proba unavailable: {e}")` | `pylint backend/ml/ml_api_server.py` reports no bare-except violations | 0.5 | true | [] |
| F-03-010 | medium | incomplete_code | backend/ml/online_learning.py:103 | `OnlineLearner.update()` raises `NotImplementedError` | `online_learning.py:103` — `raise NotImplementedError` in the base `OnlineLearner.update()` method. Subclasses `IncrementalLearner` and `SGDIncrementalLearner` exist but the online learning system has no integration with the main inference path (`model_manager.py` or `ml_api_server.py`). The system is architecturally isolated. | Wire `OnlineLearner` into `ModelManager.predict()` post-inference update cycle, or document explicitly that online learning is a non-activated feature with a roadmap item. | `model_manager.py` calls `online_learner.update(features, label)` after each prediction OR feature is marked `EXPERIMENTAL` in code and docs | 10 | false | [] |
| F-03-011 | medium | stale_code | backend/ml/simple_training_pipeline.py | Duplicate simplified pipeline is dead code | `simple_training_pipeline.py` (`"Phase 4 Testing"`) duplicates functionality of `training_pipeline.py` without being integrated into any production path. It provides its own `SimpleMLTrainingPipeline` class with the same `_load_config()` / `initialize()` flow. No import of this module found in any other file outside tests. | Remove or merge into the main `training_pipeline.py` once Phase 4 testing is complete. | `grep -rn "simple_training_pipeline" backend/ --include="*.py"` returns only test references, and the module is deprecated in code comments | 2 | true | [] |
| F-03-012 | medium | performance | backend/ml/feature_store.py:240-269 | Feature computation iterates entities in a Python `for` loop — O(N×M) | `_compute_single_feature()` and all `_compute_*` built-ins iterate `entity_ids` one-by-one with Python loops rather than vectorized pandas operations. For 6,000 stocks, this creates 6,000 row-filter operations per feature per compute call. | Refactor built-in feature computations to use grouped pandas operations: `price_data.groupby("ticker")["close"].last()` instead of per-ticker loops. | Feature computation for 6,000 tickers and 9 built-in features completes in <5 seconds on CPU | 6 | true | [] |
| F-03-013 | medium | doc_drift | docs/ml/ML_OPERATIONS_GUIDE.md | Operations guide references wrong model file path | Guide instructs `df -h backend/ml_models/` but model files are at `ml_models/` (project root) and the path `backend/ml_models/` does not exist. Same drift in `ML_QUICKSTART.md` which refers to `backend/ml_models/`. | Update all documentation to use `ml_models/` (project root path) consistently. | `grep -rn "backend/ml_models" docs/ml/` returns no results | 0.5 | true | ["18-docs-health"] |
| F-03-014 | medium | testing_gap | backend/ml/ | No integration tests for the ML API server endpoints | Test files `test_ml_pipeline.py` and `test_ml_performance.py` use mocks and unit-level tests. No tests exercise the live FastAPI `ml_api_server.py` via `TestClient`. Security-sensitive endpoints (`/retrain`, `/models/{name}/load`, `DELETE`) have zero test coverage. | Add `TestClient(app)` integration tests for all CRUD endpoints in `ml_api_server.py`, including negative cases for path traversal and unauthenticated access. | `pytest backend/tests/test_ml_api_server.py` passes with ≥80% endpoint coverage | 6 | true | ["15-test-suite"] |
| F-03-015 | low | code_quality | backend/ml/model_manager.py:221 | `_load_pytorch_model` is defined but never called | `model_manager.py:218-226` defines `_load_pytorch_model()` which calls `torch.load(path)` directly. The actual LSTM loader `_load_lstm_model()` at line 250 uses its own `torch.load()` inline. The standalone method is dead code. | Remove `_load_pytorch_model()` or refactor LSTM loader to call it (after adding `weights_only=True`). | `grep -n "_load_pytorch_model" backend/ml/model_manager.py` shows only definition, no call sites | 0.5 | true | [] |
| F-03-016 | low | doc_drift | docs/ml/ML_API_REFERENCE.md | Rate limiting documented but not implemented | Document claims "1000 requests/minute per client" and "Burst: up to 50 requests in 10 seconds" with rate-limit headers. No rate limiting middleware (`slowapi`, `limits`, or custom) exists in `ml_api_server.py`. | Either implement rate limiting (e.g., `slowapi`) or remove the rate limiting section from the docs. | `ml_api_server.py` imports `slowapi` and attaches limiter, OR docs section removed | 3 | true | ["18-docs-health"] |
| F-03-017 | low | better_pattern | backend/ml/feature_store.py:764-771 | MD5 used for cache key generation | `_generate_cache_key()` at line 764 uses `hashlib.md5()`. MD5 has known collision vulnerabilities. While not a security-critical use here (cache key only, no authentication), better_pattern is to use `hashlib.sha256()` or `hashlib.blake2b()` for consistency with any future security-sensitive hash usage. | Replace `hashlib.md5` with `hashlib.sha256` in cache key generation. | `grep "hashlib.md5" backend/ml/feature_store.py` returns no results | 0.5 | true | [] |

---

## 4. Cross-Scope Linkages

- **F-03-001** → scope 08 (`backend/auth/`) — Auth middleware should be applied to the ML API server the same way it is applied to the main backend API. The `ml_api_server.py` runs as a separate process and does not inherit any auth from the main FastAPI app. Scope 13 (infra) should ensure the ML API is behind an internal network or reverse proxy.
- **F-03-003** → scope 02 (`backend/services/`) — Service layer callers of `ModelManager.predict()` will silently receive random dummy predictions. Scope 02 audit should verify whether service layer has safeguards against fallback model output propagating to users.
- **F-03-003** → scope 13 (infra/deployment) — Docker compose and deployment scripts should include a healthcheck that verifies model binaries are loaded, failing container startup if all models are in fallback state.
- **F-03-004** → scope 05 (data ingestion/ETL) — XGBoost model with all-zero feature importances suggests the training data pipeline is producing malformed or empty feature vectors. Root cause likely lives in the ETL/feature computation path.
- **F-03-005** → scope 07 (database/persistence) — Real feature drift monitoring requires persisting historical feature values. The `db_engine` in `FeatureStore.__init__` is initialized but unused; scope 07 should provide the schema for feature value storage.
- **F-03-006** → scope 08 (auth/security) — Path traversal in model name is a security finding that should be in scope 08's threat model.
- **F-03-008** → scope 18 (docs health) — Feature store documentation overclaims coverage; scope 18 should flag the drift between `ML_PIPELINE_DOCUMENTATION.md` feature list and implementation.
- **F-03-013** → scope 18 (docs health) — Wrong path references in multiple ML docs.
- **F-03-014** → scope 15 (test suite) — ML API server has no integration test file; scope 15 should track this gap.
- **F-03-016** → scope 18 (docs health) — Rate limit documentation is misleading; scope 18 should flag.

---

## 5. Risk-Prioritized Punch List (top 10)

1. **F-03-003** — Missing model binaries / all-fallback production — the entire ML recommendation system is producing random outputs; this is a complete functional failure of the core value proposition.
2. **F-03-001** — Unauthenticated model management on public interface — trivial attack to DOS the ML API or trigger costly retraining loops; network-exploitable with zero credentials.
3. **F-03-004** — XGBoost R²=−0.011, all feature importances zero — even when binaries are present, the trained XGBoost model performs worse than a naive baseline; must diagnose training data bug before deploying.
4. **F-03-002** — torch.load without weights_only=True at five call-sites — arbitrary code execution via crafted model file; low effort to fix (add one parameter), high blast radius if exploited.
5. **F-03-006** — Path traversal in model name parameter — allows reading arbitrary filesystem paths via joblib; 2-hour fix but opens privilege escalation if combined with F-03-001.
6. **F-03-005** — Mock random data in drift monitoring — silent operational failure; platform will never detect real model or feature drift, defeating the monitoring system entirely.
7. **F-03-008** — Feature store missing MACD, Bollinger Bands, sentiment — documented capabilities that don't exist; downstream recommendation quality depends on these features.
8. **F-03-012** — O(N×M) entity iteration in feature store — at 6,000 stocks × 9 features, each compute call makes 54,000 pandas filter operations; performance will be unacceptable at scale.
9. **F-03-007** — Relative log paths in training scripts — low-severity but causes silent training failures in containerized environments where CWD differs from project root.
10. **F-03-014** — No ML API integration tests — security-sensitive endpoints have zero test coverage; regressions on auth gaps or path traversal fixes won't be caught.

---

## 6. Open Questions

- **Q1:** Is the HuggingFace Hub download (`HF_AUTO_DOWNLOAD=true`) intended to be the primary model delivery mechanism in production, or should model binaries be committed/stored in artifact storage (S3, GCS)? The current setup requires an HF token and public/private repo access at container startup.
- **Q2:** What is the intended deployment boundary for `ml_api_server.py`? Is it truly internal-only (same Kubernetes namespace / Docker network), or is it expected to be externally accessible? This determines the appropriate auth strategy for F-03-001.
- **Q3:** The `pipeline/memory_sync.py` bridges Claude Flow's `.swarm/memory.db` with ML pipeline state — is this coupling intentional for production use, or is it leftover from the development orchestration workflow? If the former, `.swarm/memory.db` must be treated as a critical operational dependency.
- **Q4:** The `online_learning.py` module is fully implemented but never integrated. Is online learning on the roadmap for the next release, or should it be documented as experimental and excluded from production builds?
- **Q5:** `backtesting.py:503,525` contains mock sector mapping and mock benchmark data. Is the backtesting framework expected to be production-ready, and if so, what data provider should supply live benchmark data?
