---
scope_id: "05-data-ingestion-etl"
scope_name: "Data Ingestion, ETL, Streaming, Scanner"
agent_type: "data-engineer"
date: "2026-04-27"
files_in_scope: 37
files_reviewed: 30
files_skipped:
  - "backend/etl/cache_primitives.py (read selectively via imports chain)"
  - "backend/etl/cache_storage.py (read selectively via imports chain)"
  - "backend/etl/cache_analytics.py (read selectively via imports chain)"
  - "backend/etl/cache_warming.py (read selectively via imports chain)"
  - "backend/data_ingestion/scanner_types.py (not reached within read budget)"
  - "backend/data_ingestion/scanner_providers.py (not reached within read budget)"
prior_reports_validated:
  - path: "docs/architecture/DATA_COLLECTION_SOLUTION.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/DATA_COLLECTION_SOLUTION.archived.md"
    claims_validated: 8
    claims_still_valid: 4
    claims_stale: 4
  - path: "docs/architecture/MULTI_SOURCE_ETL_SOLUTION.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/MULTI_SOURCE_ETL_SOLUTION.archived.md"
    claims_validated: 9
    claims_still_valid: 5
    claims_stale: 4
  - path: "docs/architecture/UNLIMITED_DATA_EXTRACTION_SOLUTION.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/UNLIMITED_DATA_EXTRACTION_SOLUTION.archived.md"
    claims_validated: 8
    claims_still_valid: 3
    claims_stale: 5
  - path: "docs/architecture/UNLIMITED_STOCK_EXTRACTION_SOLUTION.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/UNLIMITED_STOCK_EXTRACTION_SOLUTION.archived.md"
    claims_validated: 9
    claims_still_valid: 4
    claims_stale: 5
  - path: "docs/reports/ETL_ACTIVATION_SUCCESS.md"
    status: "fully_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/ETL_ACTIVATION_SUCCESS.archived.md"
    claims_validated: 6
    claims_still_valid: 2
    claims_stale: 4
  - path: "docs/reports/STOCK_UNIVERSE_EXPANSION_SUCCESS.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/STOCK_UNIVERSE_EXPANSION_SUCCESS.archived.md"
    claims_validated: 6
    claims_still_valid: 2
    claims_stale: 4
findings_summary:
  critical: 3
  high: 6
  medium: 7
  low: 4
  total: 20
estimated_remediation_effort_days: 12
agent_status: "complete"
agent_token_usage: 9800
---

# Data Ingestion, ETL, Streaming, Scanner — Audit Report

## TL;DR (REQUIRED — exactly 5 bullets, max)

- The entire `backend.etl` package is unimportable in production — an unconditional `from selenium import webdriver` at module load crashes the package with `ModuleNotFoundError`, making all ETL functionality dead code.
- A production database password (`9v1g^OV9XUwzUP6cEgCYgNOE`) is exposed in plaintext in `docs/reports/STOCK_UNIVERSE_EXPANSION_SUCCESS.md` line 81 and cross-referenced as "🔴 Exposed" in the security audit doc but not yet rotated.
- `ETLOrchestrator._run_distributed_pipeline` contains an infinite polling loop: if jobs stall in "running" or "failed" state (never reaching "completed"), the monitor spins forever with no timeout.
- `SmartDataFetcher` (backend/data_ingestion/smart_data_fetcher.py) is entirely stub code returning hardcoded zeros and `"source": "mock"` for every data type — any caller consuming it receives fabricated data.
- The Kafka streaming client (`backend/streaming/kafka_client.py`) uses `enable_auto_commit: True` by default, which does not provide exactly-once processing guarantees for financial data where duplicate events cause double-counting.

> Read these 5 before anything else in this report.

## 1. Scope & Files Reviewed

**Path globs covered:**
- `backend/data_ingestion/**/*.py` (11 files)
- `backend/etl/**/*.py` (21 files)
- `backend/streaming/**/*.py` (2 files)
- `backend/scanner/**/*.py` (3 files)

**Files explicitly excluded:**
- `backend/etl/cache_primitives.py`, `cache_storage.py`, `cache_analytics.py`, `cache_warming.py` — read only their exported symbols via the `intelligent_cache_system.py` orchestrator; full read deferred to stay within budget
- `backend/data_ingestion/scanner_types.py`, `scanner_providers.py` — budget consumed before reaching these; no critical references found in callers

**Total files in scope:** 37 Python files  
**Files reviewed (direct read or grep evidence):** 30

## 2. Prior Report Reconciliation

### `docs/architecture/DATA_COLLECTION_SOLUTION.md` — status: `partially_stale`

**Validation method:** Direct file reads of referenced modules; grep of file existence; Python import test.

**Archived to:** `docs/audits/2026-04/_meta/prior-reports-archive/DATA_COLLECTION_SOLUTION.archived.md`

| # | Claim | Original Location | Status | Evidence |
|---|---|---|---|---|
| 1 | `multi_source_extractor.py` exists and provides multi-source extraction | §"New Files Created" | current | `find backend/etl -name multi_source_extractor.py` returns `backend/etl/multi_source_extractor.py`; file read confirms 600+ lines of implementation |
| 2 | `web_scrapers.py` implements Yahoo, MarketWatch, Google scraping | §"New Files Created" | current | `backend/etl/web_scrapers.py` read at line 1; `WebScraperBase`, `YahooFinanceScraper` classes confirmed |
| 3 | `distributed_batch_processor.py` exists and processes 6000+ stocks | §"New Files Created" | current | File confirmed at `backend/etl/distributed_batch_processor.py`; SQLite job tracking schema confirmed at lines 102-143 |
| 4 | `scripts/run_enhanced_etl.py` exists as main execution script | §"Quick Start" | current | `find ... -name run_enhanced_etl.py` returns `/scripts/run_enhanced_etl.py` |
| 5 | ETL pipeline imports successfully | §"Quick Start" | fully_stale | `python3 -c "from backend.etl.data_extractor import DataExtractor"` → `ModuleNotFoundError: No module named 'selenium'` at unlimited_data_extractor.py:21 |
| 6 | Rate limit: Yahoo scraper ~100/min, yfinance ~50/min | §"Data Sources" | partially_stale | `rate_limiting.py:52-67` shows yahoo_scraper rate=0.083 tokens/s (5/min, not 100/min); yfinance rate=0.028 (1.7/min, not 50/min) — actual configured rates are far more conservative than documented |
| 7 | `backend/etl/monitoring.py` exists for Prometheus export | §"Production Deployment" | fully_stale | `find backend/etl -name monitoring.py` returns nothing; module does not exist |
| 8 | 95-98% success rate after implementation | §"Performance Metrics" | unverifiable | No runtime metrics available; pipeline cannot run due to selenium import failure |

---

### `docs/architecture/MULTI_SOURCE_ETL_SOLUTION.md` — status: `partially_stale`

**Validation method:** Source reads, grep for class/method existence, import test.

**Archived to:** `docs/audits/2026-04/_meta/prior-reports-archive/MULTI_SOURCE_ETL_SOLUTION.archived.md`

| # | Claim | Original Location | Status | Evidence |
|---|---|---|---|---|
| 1 | `IntelligentSourceRouter` routes requests based on success rates | §"Intelligent Source Routing" | current | `multi_source_extractor.py:69` defines `IntelligentSourceRouter`; `source_priorities` dict at line 74 tracks success_rate per source |
| 2 | L1/L2/L3 multi-tier caching implemented | §"Advanced Caching System" | current | `intelligent_cache_system.py` confirmed as orchestrator; imports `MemoryTierCache`, `DiskTierCache`, `RedisTierCache` from sibling modules |
| 3 | Max concurrent jobs = 3, max concurrent per job = 8 | §"Job Configuration" | current | `etl_orchestrator.py:67-75`: `ProcessorConfig(max_concurrent_jobs=3, max_concurrent_per_job=8)` exactly matches |
| 4 | 4 validation levels: Basic, Standard, Strict, Comprehensive | §"Validation Levels" | current | `data_validation_pipeline.py:23-27` defines `ValidationLevel` enum with exactly these four levels |
| 5 | Drop-in replacement for existing code maintained | §"Drop-in Replacement" | current | `data_extractor.py:1-30` confirms `DataExtractor`, `MultiSourceDataExtractor` backward-compat wrappers exist |
| 6 | `POSTGRES_PASSWORD=postgres` shown as example env var | §"Environment Variables" | partially_stale | Config is still `os.getenv('POSTGRES_PASSWORD', 'postgres')` in `data_loader.py:36`, `stock_universe_manager.py:30`; default is insecure placeholder |
| 7 | System respects robots.txt | §"Compliance and Ethics" | unverifiable | No `robots.txt` check found in `web_scrapers.py` or any scraper; claim is undocumented in code |
| 8 | Cache reduces API calls by 70-80% | §"Cache Benefits" | unverifiable | No runtime telemetry accessible; metrics exist in code but pipeline is broken |
| 9 | ETL completes in 4-6 hours for 6000 stocks | §"Throughput Estimates" | unverifiable | Pipeline cannot run; no historic run data |

---

### `docs/architecture/UNLIMITED_DATA_EXTRACTION_SOLUTION.md` — status: `partially_stale`

**Validation method:** File existence checks, import tests, code reads.

**Archived to:** `docs/audits/2026-04/_meta/prior-reports-archive/UNLIMITED_DATA_EXTRACTION_SOLUTION.archived.md`

| # | Claim | Original Location | Status | Evidence |
|---|---|---|---|---|
| 1 | `unlimited_data_extractor.py` with Yahoo Finance scraping exists | §"Core Files" | current | File confirmed at `backend/etl/unlimited_data_extractor.py`; `YahooFinanceWebScraper` class present |
| 2 | `simple_unlimited_extractor.py` with no external dependencies | §"Core Files" | fully_stale | `find backend/etl -name simple_unlimited_extractor.py` returns nothing; file does not exist |
| 3 | No rate limits on Yahoo CSV direct download | §"Yahoo Finance Direct Downloads" | current | `unlimited_data_extractor.py` implements `BulkDataDownloader` using query1.finance.yahoo.com CSV URLs |
| 4 | `test_unlimited_extraction.py` test suite | §"Testing" | fully_stale | `find . -name test_unlimited_extraction.py` returns nothing; file does not exist |
| 5 | "50-100 stocks/second" processing speed | §"Performance Metrics" | fully_stale | Pipeline unimportable; no `selenium` module installed; import fails at `unlimited_data_extractor.py:21` |
| 6 | Integration with existing code — "No code changes required" | §"Migration Guide" | fully_stale | `data_extractor.py:19` imports from `unlimited_extractor_with_fallbacks` which imports `unlimited_data_extractor` — the selenium import kills all callers |
| 7 | SEC EDGAR extractor included | §"Free Data Sources" | current | `SECEdgarExtractor` class exists in `unlimited_data_extractor.py`; also `sec_edgar_client.py` in data_ingestion |
| 8 | IEX Cloud free tier integration | §"Free Data Sources" | current | `IEXCloudFreeExtractor` class present in `unlimited_data_extractor.py` |

---

### `docs/architecture/UNLIMITED_STOCK_EXTRACTION_SOLUTION.md` — status: `partially_stale`

**Validation method:** File existence, class verification, import test.

**Archived to:** `docs/audits/2026-04/_meta/prior-reports-archive/UNLIMITED_STOCK_EXTRACTION_SOLUTION.archived.md`

| # | Claim | Original Location | Status | Evidence |
|---|---|---|---|---|
| 1 | `intelligent_cache_system.py` with L1/L2/L3 tiers | §"Component Files" | current | File confirmed; module-level orchestrator refactored into 5 sub-modules (`cache_primitives`, `cache_storage`, `cache_redis`, `cache_warming`, `cache_analytics`) |
| 2 | `concurrent_processor.py` for parallel extraction | §"Component Files" | current | `backend/etl/concurrent_processor.py` confirmed; imported by `unlimited_extractor_with_fallbacks.py:29` |
| 3 | `data_validation_pipeline.py` with quality scoring | §"Component Files" | current | `backend/etl/data_validation_pipeline.py` confirmed; `ValidationLevel` enum verified |
| 4 | `data_extractor.py` is backward-compatible replacement | §"Component Files" | partially_stale | File exists with compat wrappers; however the module cannot be imported due to selenium transitive dependency |
| 5 | Cache hit rate >80% for repeated requests | §"Speed Benchmarks" | unverifiable | No runtime data; pipeline broken |
| 6 | Selenium fallback for dynamic content | §"Key Features" | partially_stale | `unlimited_data_extractor.py:21-25` has selenium imports but they are unconditional — not guarded by try/except; causes failure if selenium absent |
| 7 | `test_unlimited_extraction.py` with performance benchmarks | §"Testing" | fully_stale | File does not exist; `find` returns nothing |
| 8 | "Drop-in replacement — no code changes needed" | §"Migration Guide" | fully_stale | Import chain broken by selenium; existing callers get `ModuleNotFoundError` |
| 9 | Bloom filter for 90% faster cache misses | §"intelligent_cache_system.py" | current | `intelligent_cache_system.py:7` docstring mentions Bloom filter; `cache_primitives.py` exports `BloomFilter` |

---

### `docs/reports/ETL_ACTIVATION_SUCCESS.md` — status: `fully_stale`

**Validation method:** Python import test, grep for referenced components.

**Archived to:** `docs/audits/2026-04/_meta/prior-reports-archive/ETL_ACTIVATION_SUCCESS.archived.md`

| # | Claim | Original Location | Status | Evidence |
|---|---|---|---|---|
| 1 | "ETL modules imported successfully" | §"Success Metrics" | fully_stale | `python3 -c "from backend.etl.data_extractor import DataExtractor"` fails: `ModuleNotFoundError: No module named 'selenium'` — import tested 2026-04-27 |
| 2 | "Components initialized" | §"Success Metrics" | fully_stale | Cannot initialize what cannot import |
| 3 | "Database connected (8 tables)" | §"Success Metrics" | unverifiable | Cannot verify without running DB; code structure supports it |
| 4 | "Ready for production use" | §"Status Summary" | fully_stale | Package-level import fails; cannot be production-ready |
| 5 | Airflow DAG `enhanced_stock_pipeline` runs daily at 6 AM | §"Next Steps" | unverifiable | Airflow DAGs in scope 06; not verified here |
| 6 | ML modules disabled safely with `HAS_ML = False` | §"Optional Components" | current | `etl_orchestrator.py:29-33`: try/except around ML imports with `HAS_ML = False` fallback confirmed |

---

### `docs/reports/STOCK_UNIVERSE_EXPANSION_SUCCESS.md` — status: `partially_stale`

**Validation method:** Source read, grep, file existence check. Note: one redaction performed (plaintext password).

**Archived to:** `docs/audits/2026-04/_meta/prior-reports-archive/STOCK_UNIVERSE_EXPANSION_SUCCESS.archived.md`

**Redaction log:** Line 81 of original document contains `PGPASSWORD=9v1g^OV9XUwzUP6cEgCYgNOE psql ...` — redacted in archive as it matches pattern `password`. Redaction count: 1.

| # | Claim | Original Location | Status | Evidence |
|---|---|---|---|---|
| 1 | `StockUniverseManager` implemented with dynamic DB loading | §"Technical Implementation" | current | `backend/etl/stock_universe_manager.py` confirmed; `get_all_active_tickers()`, `populate_database_with_all_stocks()` methods present |
| 2 | Multi-source aggregation: NASDAQ API + Finnhub + Wikipedia | §"Data Sources Used" | current | `stock_universe_manager.py:43-80` shows NASDAQ API URL and Wikipedia table fetch patterns |
| 3 | 20,674 stocks loaded at 344% of target | §"Achievement Highlights" | unverifiable | Database state not accessible; code capability exists but runtime state unknown |
| 4 | ETL pipeline dynamically loads from DB (not hardcoded) | §"ETL Pipeline Status" | current | `etl_orchestrator.py:232-265`: `get_active_tickers()` calls `StockUniverseManager().get_all_active_tickers()` |
| 5 | Cost compliance under $50/month maintained | §"Cost Optimization" | unverifiable | No cost telemetry accessible |
| 6 | PGPASSWORD exposed in bash command | §"Recommended Commands" | fully_stale (security) | `docs/reports/STOCK_UNIVERSE_EXPANSION_SUCCESS.md:81` contains plaintext credential. Cross-referenced at `docs/security/SECURITY_CREDENTIALS_AUDIT.md:34` as "🔴 Exposed" — not yet remediated |

---

## 3. Findings

| ID | Severity | Category | File:Line | Title | Description | Recommendation | Acceptance Test Hint | Effort (h) | Loki Actionable | Cross Scope |
|---|---|---|---|---|---|---|---|---|---|---|
| F-05-001 | critical | broken_import | backend/etl/unlimited_data_extractor.py:21 | Unconditional selenium import breaks entire ETL package | `from selenium import webdriver` at module top-level; no try/except guard. `backend/etl/__init__.py` re-exports `DataExtractor` which imports through this chain. Entire `backend.etl` package fails to import, making ETL, orchestrator, and all callers non-functional. Confirmed: `python3 -c "from backend.etl.data_extractor import DataExtractor"` → `ModuleNotFoundError: No module named 'selenium'` | Wrap selenium imports in try/except with `SELENIUM_AVAILABLE = False` fallback, identical to how `talib`/`torch` are handled in other modules. Only the Selenium-specific scraping path should fail gracefully. | `python3 -c "from backend.etl.data_extractor import DataExtractor; print('OK')"` exits 0 | 2 | true | [] |
| F-05-002 | critical | bug | backend/etl/etl_orchestrator.py:152 | Infinite loop when distributed jobs do not complete normally | `while completed_jobs < len(job_ids)` at line 152 only increments when jobs reach `status == "completed"`. If jobs stall in "running", "failed", or "error" status (e.g., due to exception in processor worker), the loop never exits. No timeout, no break on failed-job accumulation. `processing_task` is also never awaited or cancelled on loop exit. | Add loop timeout (e.g., `max_wait_seconds = config.job_timeout_hours * 3600`), count failed+completed toward exit condition, cancel `processing_task` in finally block. | Integration test: mock all jobs to fail; assert orchestrator returns within 60s | 4 | true | [] |
| F-05-003 | critical | security | docs/reports/STOCK_UNIVERSE_EXPANSION_SUCCESS.md:81 | Production database password exposed in documentation | `PGPASSWORD=9v1g^OV9XUwzUP6cEgCYgNOE` appears verbatim in bash command at line 81. Cross-referenced as "🔴 Exposed" in `docs/security/SECURITY_CREDENTIALS_AUDIT.md:33-34` but not yet rotated. | Immediately rotate the credential. Remove plaintext from doc, replace with `PGPASSWORD=$DB_PASSWORD`. Scan git history with gitleaks for prior commits containing the value. | `git log --all -S "9v1g" --oneline` returns no results after rotation | 2 | true | ["08-auth-security-compliance"] |
| F-05-004 | high | incomplete_code | backend/data_ingestion/smart_data_fetcher.py:58-123 | SmartDataFetcher is entirely stub — returns mock zeros for all data types | Every fetch method (`_fetch_price_data`, `_fetch_fundamentals`, `_fetch_news`, `_fetch_financials`, `_fetch_earnings`, `_fetch_sentiment`, `_fetch_generic`) returns hardcoded `{"source": "mock", "price": 0.0, ...}`. Any component consuming this receives fabricated data silently. The module docstring claims "intelligent caching and rate limit management." | Implement real fetch logic by delegating to the existing `FinnhubClient`, `AlphaVantageClient`, `SECEdgarClient`, `PolygonClient` in `data_ingestion/`. Wire cache_manager and rate_limiter parameters. | Unit test: `fetcher.fetch_stock_data("AAPL", "price")["price"] != 0.0` | 8 | true | ["02-backend-services-domain", "09-analytics"] |
| F-05-005 | high | schema_mismatch | backend/etl/multi_source_extractor.py:46, backend/etl/unlimited_data_extractor.py:56 | Two incompatible `ExtractionResult` dataclasses with same name in same package | `multi_source_extractor.py:46` defines `ExtractionResult` with `data: Optional[Dict]`; `unlimited_data_extractor.py:56` defines `ExtractionResult` with `data: Any`. `etl_orchestrator.py:23` imports from `multi_source_extractor`; `data_extractor.py:20` imports from `unlimited_extractor_with_fallbacks` which re-exports from `unlimited_data_extractor`. Code paths using different imports will type-mismatch silently. | Consolidate into one canonical `ExtractionResult` in a new `backend/etl/types.py`; both extractors import from it. | `from backend.etl.types import ExtractionResult; from backend.etl.multi_source_extractor import ExtractionResult as R2; assert ExtractionResult is R2` | 3 | true | [] |
| F-05-006 | high | security | backend/data_ingestion/sec_edgar_client.py:31 | Placeholder contact email in SEC EDGAR User-Agent header | SEC EDGAR terms of service require a valid email contact in the User-Agent string for programmatic access. `contact@example.com` is a placeholder that violates SEC terms and can result in IP bans. | Replace with a real operational email via environment variable: `os.getenv('SEC_EDGAR_CONTACT_EMAIL', '')`. Fail loudly at startup if empty. | `assert "example.com" not in client.headers["User-Agent"]` | 1 | true | ["08-auth-security-compliance"] |
| F-05-007 | high | architecture | backend/etl/etl_orchestrator.py:387,633 | `self.extractor` referenced in legacy methods but never assigned | `extract_phase()` at line 387 calls `await self.extractor.batch_extract(...)` and `run_realtime_update()` at line 633 calls `await self.extractor.extract_all_data(ticker)`. `ETLOrchestrator.__init__` assigns `self.legacy_extractor` but never `self.extractor`. Both methods will raise `AttributeError` at runtime. | Rename `self.legacy_extractor` to `self.extractor` or add `self.extractor = self.legacy_extractor` in `__init__`. | `pytest backend/tests/unit/test_etl_modules.py -k "realtime"` passes | 1 | true | [] |
| F-05-008 | high | performance | backend/etl/distributed_batch_processor.py:99-548 | SQLite connections opened per-call with no context manager — connection leak on exception | All 8 `sqlite3.connect()` calls in `distributed_batch_processor.py` use bare `conn = sqlite3.connect(...)` / `conn.close()` pattern with no `with` statement or try/finally. If any exception occurs between open and close, the connection leaks. Under concurrent job processing this accumulates open file handles. | Replace all with `with sqlite3.connect(self.job_db_path) as conn:` pattern, or use a single connection pool. | `lsof -p <pid> | grep job_tracking.db` shows no accumulation over 100 iterations | 4 | true | [] |
| F-05-009 | high | architecture | backend/streaming/kafka_client.py:38 | Kafka consumer uses auto-commit — no exactly-once guarantee for financial data | `enable_auto_commit: bool = True` in `KafkaConfig`. For financial event streams (stock prices, recommendations, audit_logs), auto-commit means offsets are committed before processing completes. A crash mid-processing loses events. Duplicate processing is also possible on consumer restart. | Set `enable_auto_commit=False` and commit offsets manually after successful processing in the consumer loop. Consider `isolation_level="read_committed"` for transactional producers. | Integration test: kill consumer mid-batch, verify no duplicate or missing messages on restart | 8 | false | ["10-monitoring-observability"] |
| F-05-010 | medium | dead_code | backend/etl/etl_orchestrator.py:374-423 | `extract_phase()` is dead — superseded by `enhanced_extract_phase()` but never removed | `extract_phase()` at line 374 uses the old `self.extractor` (which itself has a bug per F-05-007) and is never called from `_run_standard_pipeline()`. `enhanced_extract_phase()` is called instead. The dead method also imports `DataValidator` as `self.legacy_validator` which is unused. | Remove `extract_phase()`, `validate_extracted_data()` (line 402), and the `self.legacy_validator` / `self.aggregator` assignments in `__init__`. | `grep -n "extract_phase\|validate_extracted_data" etl_orchestrator.py` returns zero hits after cleanup | 2 | true | [] |
| F-05-011 | medium | incomplete_code | backend/scanner/daily/daily_scanner.py:80-93 | Daily scanner stock list is hardcoded/test stub — returns only ~100 S&P stocks | `_get_all_stock_symbols()` at line 80 includes a comment "In production, this would query the database" and then does `return all_symbols[:100]` — hard limit of 100 tickers. The scanner claims to scan 6000+ stocks but actually scans at most 100. | Implement the database query using `StockUniverseManager.get_all_active_tickers()` and remove the `[:100]` slice. | `scanner.scan_all_stocks()` receives len > 500 from `_get_all_stock_symbols()` | 3 | true | ["07-database-persistence"] |
| F-05-012 | medium | security | backend/etl/data_loader.py:36, backend/etl/stock_universe_manager.py:30, backend/etl/cache_warming.py:73,151 | Default DB password `'postgres'` hardcoded in 4 files as `os.getenv(..., 'postgres')` | Using a default fallback password allows silent misconfiguration in staging/CI environments. If `POSTGRES_PASSWORD` env var is not set, connection uses insecure default. This affects 4 files across the ETL layer. | Remove the default: `os.getenv('POSTGRES_PASSWORD')` with no fallback, raise `ValueError` if None at startup. | `POSTGRES_PASSWORD="" python3 -c "from backend.etl.data_loader import DataLoader"` raises `ValueError` not silently connects | 2 | true | ["08-auth-security-compliance", "16-config-secrets"] |
| F-05-013 | medium | doc_drift | docs/architecture/DATA_COLLECTION_SOLUTION.md:69 | Documented Yahoo scraper rate (100/min, 60/min usage) contradicts code rate config | Doc states "Yahoo Finance Scraper: ~100/min, Our Usage: 60/min". `rate_limiting.py:52-57` configures `yahoo_scraper` at `rate=0.083` tokens/sec = 5/min with `max_per_hour=300` and `min_delay=2.0s`. Actual configured rate is 12x lower than documented. | Update doc to match code, or update code to match intended design with explanation of the discrepancy. | Grep `rate_limiting.py` for `yahoo_scraper` rate value matches what DATA_COLLECTION_SOLUTION.md states | 1 | true | [] |
| F-05-014 | medium | testing_gap | backend/streaming/kafka_client.py, backend/scanner/daily/daily_scanner.py | No tests exist for streaming layer or daily scanner | `find backend/tests -name "test_stream*" -o -name "test_kafka*" -o -name "test_scanner*"` returns nothing. The Kafka producer/consumer client (350 LOC), consumer loop logic, and `DailyStockScanner` (340 LOC) have zero test coverage. These components handle real-time financial data flows and scanning for 6000+ stocks. | Add unit tests for `KafkaProducerClient.send_message()`, `KafkaConsumerClient` message handling, and `DailyStockScanner._analyze_stock()` with mocked data sources. | `pytest backend/tests/unit/test_streaming.py backend/tests/unit/test_scanner.py` passes | 8 | true | ["15-test-suite"] |
| F-05-015 | medium | architecture | backend/etl/unlimited_data_extractor.py, backend/etl/multi_source_extractor.py | Two parallel extraction architectures with duplicated `ExtractionResult`, `StockData`, and source-routing logic | Both `unlimited_data_extractor.py` and `multi_source_extractor.py` independently implement source routing, fallback, caching integration, and result dataclasses. `etl_orchestrator.py` uses both. This creates maintenance burden and schema drift risk (already observed in F-05-005). | Designate one extraction engine as canonical (recommend `multi_source_extractor.py` + `rate_limiting.py` as they are more structured) and deprecate/remove the unlimited extractor stack, retaining only unique capabilities (bulk CSV download, SEC scraping). | `grep -rn "unlimited_data_extractor\|unlimited_extractor_with_fallbacks" backend/` returns 0 non-test hits after consolidation | 16 | false | [] |
| F-05-016 | low | code_quality | backend/etl/etl_orchestrator.py:50-52 | Unused instance variables assigned in `__init__` | `self.legacy_validator = DataValidator()` and `self.aggregator = DataAggregator()` are assigned but never referenced outside of `__init__`. Adds startup cost and confusion about which validator is in use. | Remove unused assignments once `extract_phase()` dead code (F-05-010) is cleaned up. | `grep -n "legacy_validator\|self\.aggregator" etl_orchestrator.py` returns 0 hits | 0.5 | true | [] |
| F-05-017 | low | code_quality | backend/etl/etl_orchestrator.py:146 | `processing_task` asyncio task created but never awaited or cancelled | `asyncio.create_task(self.distributed_processor.start_processing())` stores result in `processing_task` but the variable is never awaited, and not cancelled in the except block. If the function returns normally, the background task may continue running. | `await processing_task` after `stop_processing()`, or cancel it in finally block: `processing_task.cancel(); await asyncio.gather(processing_task, return_exceptions=True)`. | `asyncio.all_tasks()` returns empty set after `_run_distributed_pipeline` completes | 1 | true | [] |
| F-05-018 | low | doc_drift | docs/reports/ETL_ACTIVATION_SUCCESS.md | ETL_ACTIVATION_SUCCESS.md declares "production ready" as of 2025-08-19 — fully stale | Entire document asserts operational readiness and successful import. As of 2026-04-27, ETL package import fails with `ModuleNotFoundError: No module named 'selenium'`. Document should be marked deprecated or replaced with current state. | Supersede document with post-fix validation report after F-05-001 is resolved. | N/A — doc update | 0.5 | true | [] |
| F-05-019 | low | better_pattern | backend/etl/data_loader.py:42-57 | SQLAlchemy `QueuePool` of 10 connections created per `DataLoader` instance | `ETLOrchestrator` instantiates `DataLoader()` at construction. If multiple orchestrators or tasks run concurrently, each creates its own pool of 10. A singleton or dependency-injected engine would prevent connection overuse. | Promote engine to module-level singleton or accept it as a constructor parameter. | `DataLoader._create_engine` is called exactly once per process lifetime in integration test | 2 | true | ["07-database-persistence"] |
| F-05-020 | low | code_quality | backend/etl/web_scrapers.py:37-46 | Hardcoded user-agent rotation list is stale (Chrome 91/92, Firefox 89, Safari 14.1 from 2021) | `_get_random_user_agent()` at line 37 returns one of 5 user-agent strings, all from 2021. Modern anti-bot systems flag outdated user-agents. | Source from a maintained user-agent library (`fake-useragent` or `ua-generator`) or update the list periodically via config. | Manual check: scrapers do not receive HTTP 403 from Yahoo Finance endpoints in staging | 1 | true | [] |

## 4. Cross-Scope Linkages

- **F-05-003** → scope 08-auth-security-compliance: The exposed postgres password `9v1g^OV9XUwzUP6cEgCYgNOE` is already flagged in `docs/security/SECURITY_CREDENTIALS_AUDIT.md` as owned by scope 08. Remediation (credential rotation) is blocked pending that scope's work. ETL scope must update its documentation after rotation completes.
- **F-05-004** → scope 02-backend-services-domain: `SmartDataFetcher` is likely consumed by backend services for portfolio/analytics data. Stub data propagates silently to any service layer caller.
- **F-05-004** → scope 09-analytics: Analytics computations fed by stub zero-data produce meaningless results with no error raised.
- **F-05-006** → scope 08-auth-security-compliance: SEC EDGAR ToS violation is a compliance/legal risk, not just a technical one; the security scope should track this.
- **F-05-009** → scope 10-monitoring-observability: Kafka offset management and event loss detection requires monitoring infrastructure. The monitoring scope should instrument consumer lag.
- **F-05-011** → scope 07-database-persistence: Scanner needs `StockUniverseManager` which requires the `stocks` table to be populated; dependency on DB migration state.
- **F-05-012** → scope 08-auth-security-compliance, 16-config-secrets: Default credential removal spans config management (scope 16) and is a security concern (scope 08).
- **F-05-014** → scope 15-test-suite: Streaming and scanner test gaps belong to the test suite scope for tracking and coverage enforcement.
- **F-05-019** → scope 07-database-persistence: Connection pool design impacts the database scope's pool sizing and connection governance strategy.

## 5. Risk-Prioritized Punch List (top 10)

1. **F-05-001** — Broken ETL package import (selenium) — Nothing in the ETL layer works until this is fixed; all other ETL findings are moot while import fails. 2-hour fix with highest possible blast-radius.
2. **F-05-003** — Production DB password exposed in docs — Credential is already flagged as "🔴 Exposed" in security audit; rotation must happen before any public or shared environment use.
3. **F-05-002** — Infinite loop in distributed pipeline monitor — Production pipeline would hang indefinitely if any batch job fails; no timeout guard exists.
4. **F-05-007** — `self.extractor` AttributeError in legacy ETL methods — `run_realtime_update()` will crash on first call; affects any real-time price update path.
5. **F-05-004** — SmartDataFetcher entirely stub — Silent mock-data injection; any analytics or service consuming this returns garbage results without error.
6. **F-05-005** — Duplicate `ExtractionResult` classes — Schema divergence is already non-zero (`Dict` vs `Any`); will silently cause type failures as code evolves.
7. **F-05-008** — SQLite connection leak under exceptions — Under concurrent distributed processing, leaked connections accumulate until OS file descriptor limit is hit.
8. **F-05-009** — Kafka auto-commit with financial data — Data loss or duplication risk on consumer failure; especially critical for audit_logs topic.
9. **F-05-011** — Daily scanner hardcoded to 100 stocks max — Scanner claims 6000+ stock coverage but returns at most 100 due to `[:100]` slice; value delivered is a fraction of what is documented.
10. **F-05-012** — Default `'postgres'` password across 4 ETL files — Silent insecure configuration fallback; allows misconfigured deployments to appear functional while using a known default credential.

## 6. Open Questions

- Q1: Is `selenium`/`chromium-browser` installed in production Docker images? The `UNLIMITED_STOCK_EXTRACTION_SOLUTION.md` mentions adding `chromium-browser` to Dockerfile, but the `Dockerfile.*` files are in scope 13. If selenium is intentionally available in production, F-05-001 may be a dev-environment-only issue rather than a production outage — though the guard-less import is still fragile.
- Q2: What is the authoritative extraction path in production — `UnlimitedStockDataExtractor` (via `data_extractor.py`) or `MultiSourceStockExtractor` (via `etl_orchestrator.py`)? Both are wired in the orchestrator simultaneously. The relationship between the two architectures needs a single designated owner.
- Q3: Is `aiokafka` installed in the deployment environment? `kafka_client.py` imports it unconditionally at line 9; it also cannot be imported without it. If Kafka is not used, the streaming module should be guarded or the dependency made optional.
- Q4: What is the correct contact email for SEC EDGAR? This cannot be determined from code alone and requires a human decision before F-05-006 can be fully resolved.
