---
scope_id: "10-monitoring-observability"
scope_name: "Monitoring & Observability"
agent_type: "sre-engineer"
date: "2026-04-27"
files_in_scope: 34
files_reviewed: 27
files_skipped:
  - "infrastructure/monitoring/grafana/dashboards/03-business-metrics.json — structure identical to 01/02; dashboard format verified, exprs sampled"
  - "infrastructure/monitoring/grafana/dashboards/04-database-performance.json — sampled"
  - "infrastructure/monitoring/grafana/dashboards/05-external-apis.json — sampled"
  - "infrastructure/monitoring/grafana/provisioning/dashboards/dashboards.yml — provisioning boilerplate"
  - "infrastructure/monitoring/grafana/provisioning/datasources/prometheus.yml — standard"
  - "infrastructure/monitoring/templates/email.tmpl — template content"
  - "config/monitoring/grafana-dashboards/investment-app-dashboard.json — sampled via json.load"
prior_reports_validated:
  - path: "docs/PERFORMANCE_BENCHMARKS.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/PERFORMANCE_BENCHMARKS.archived.md"
    claims_validated: 5
    claims_still_valid: 2
    claims_stale: 3
    redactions: 0
  - path: "docs/PERFORMANCE_OPTIMIZATION.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/PERFORMANCE_OPTIMIZATION.archived.md"
    claims_validated: 6
    claims_still_valid: 3
    claims_stale: 3
    redactions: 0
  - path: "docs/architecture/PERFORMANCE_OPTIMIZATIONS.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/PERFORMANCE_OPTIMIZATIONS.archived.md"
    claims_validated: 4
    claims_still_valid: 2
    claims_stale: 2
    redactions: 0
findings_summary:
  critical: 3
  high: 7
  medium: 5
  low: 2
  total: 17
estimated_remediation_effort_days: 5.5
agent_status: "complete"
agent_token_usage: 5800
---

# Monitoring & Observability — Audit Report

## TL;DR (REQUIRED — exactly 5 bullets, max)

- **CRITICAL registry isolation**: `metrics_collector.py` registers all app metrics into a private `CollectorRegistry` and `setup_metrics_endpoint` calls `generate_latest(registry)` with only that registry — but `health_checks.py`, `alerting_system.py`, `application_monitoring.py`, `financial_monitoring.py`, and `database_performance.py` all register metrics to the **default** Prometheus registry which is never served. Zero SLA / health-check Prometheus metrics reach the scrape target.
- **CRITICAL metric name mismatch**: All Prometheus alert rules (`investment-platform.yml`, `slo-targets.yml`) and all Grafana dashboards query `api_request_duration_seconds_bucket`, but the only histogram actually registered in the served custom registry is `api_latency_seconds` (metrics_collector.py:59). Every latency alert and latency dashboard panel returns `no data`.
- **CRITICAL broken import in `real_time_alerts.py`**: Line 25 imports `MimeText` from `email.mime.text` and line 26 imports `MimeMultipart` from `email.mime.multipart`. The correct stdlib class names are `MIMEText` and `MIMEMultipart` (all-caps MIME). Every email alert delivery attempt raises `ImportError` at module load, silencing all email alert channels.
- **HIGH `get_quality_summary` phantom import**: `metrics_collector.py:486` does `from backend.monitoring.data_quality_metrics import get_quality_summary`, but `data_quality_metrics.py` exports no such function (confirmed: only `DataQualityMetricsCollector` class and `check_data_quality_with_metrics` exist). The data-quality collection branch raises `ImportError` silently (caught by the broad `except Exception`) every 10 seconds, meaning `data_quality_score`, `data_missing_percent`, and `data_staleness` gauges are never populated.
- **HIGH error budget formula is meaningless**: `slo-targets.yml:187-193` computes the 30-day error budget remaining from a 6-hour availability window (`slo:api_availability:ratio_rate6h`). A 6-hour window cannot represent a monthly budget; a 6-hour outage resets the ratio and the "budget remaining" immediately recovers, making `SLOErrorBudgetLow` and `SLOErrorBudgetExhausted` alerts unfireable under sustained low-level degradation.

> Read these 5 before anything else in this report.

## 1. Scope & Files Reviewed

**Path globs covered:**
- `backend/monitoring/**/*.py` — 16 files
- `infrastructure/monitoring/**` — 16 files (YAML, JSON, tmpl)
- `config/monitoring/**` — 2 files

**Files explicitly reviewed (27 of 34):**
- `backend/monitoring/__init__.py`
- `backend/monitoring/alerting_system.py`
- `backend/monitoring/alertmanager_webhook.py`
- `backend/monitoring/api_performance.py`
- `backend/monitoring/application_monitoring.py`
- `backend/monitoring/auto_scaler.py`
- `backend/monitoring/data_quality_dashboard.py`
- `backend/monitoring/data_quality_metrics.py`
- `backend/monitoring/database_performance.py`
- `backend/monitoring/financial_monitoring.py`
- `backend/monitoring/health_checks.py`
- `backend/monitoring/health_system.py`
- `backend/monitoring/log_analysis.py`
- `backend/monitoring/metrics_collector.py`
- `backend/monitoring/real_time_alerts.py`
- `backend/monitoring/sla_tracker.py`
- `infrastructure/monitoring/alertmanager.yml`
- `infrastructure/monitoring/alerts/investment-platform.yml`
- `infrastructure/monitoring/alerts/slo-targets.yml`
- `infrastructure/monitoring/grafana/dashboards/01-system-overview.json`
- `infrastructure/monitoring/grafana/dashboards/02-api-performance.json`
- `infrastructure/monitoring/loki/loki-config.yaml`
- `infrastructure/monitoring/loki/promtail-config.yaml`
- `infrastructure/monitoring/prometheus.yml`
- `infrastructure/monitoring/prometheus.prod.yml`
- `infrastructure/monitoring/README.md`
- `config/monitoring/prometheus.yml`

**Files skipped (7):** Grafana provisioning boilerplate, additional dashboard JSON files sampled via `json.load` for PromQL expressions rather than full read, email template. No functional logic in skipped files.

**Wave 3 context consulted:**
- `06-airflow-pipelines` TL;DR: Airflow DAGs broken; `statsd_mapping.yml` reviewed for metric alignment.
- `13-infra-deployment` TL;DR: Docker-compose correct; nginx scrape target noted.
- `08-auth-security-compliance` TL;DR F-08-008: Redis fail-open noted; `sla_tracker.py` uses `get_redis()` from the same module.

---

## 2. Prior Report Reconciliation

### `docs/PERFORMANCE_BENCHMARKS.md` — status: `partially_stale`

**Validation method:** Read full document; cross-referenced performance targets against `infrastructure/monitoring/alerts/slo-targets.yml` and `backend/monitoring/metrics_collector.py`.

**Archived to:** `docs/audits/2026-04/_meta/prior-reports-archive/PERFORMANCE_BENCHMARKS.archived.md`

| # | Claim | Original Location | Status | Evidence |
|---|---|---|---|---|
| 1 | "API p95 latency target <500ms" | PERFORMANCE_BENCHMARKS.md §Targets | current | `slo-targets.yml:108` confirms `SLOLatencyFastBurn` fires when p95 > 0.5s; SLO target matches |
| 2 | "Cache hit rate target >85%" | PERFORMANCE_BENCHMARKS.md §Targets | current | `investment-platform.yml:65` alert threshold 80% (conservative); `PERFORMANCE_OPTIMIZATION.md` also cites 85% |
| 3 | "Error rate target <1%" | PERFORMANCE_BENCHMARKS.md §Targets | partially_stale | `slo-targets.yml:157` uses 0.5% threshold in production SLO (`SLOErrorRateFastBurn`), tighter than the 1% stated here |
| 4 | "Benchmark Results Template shows actual measured values (892s, 6.7 stocks/s)" | PERFORMANCE_BENCHMARKS.md §Results | fully_stale | These are template placeholder values ("Test Execution Date: [DATE]"); no actual test run output found in repo; `Current` column is "TBD" throughout the targets table |
| 5 | "Phase 4.4: Setup APM and establish performance SLOs" | PERFORMANCE_BENCHMARKS.md §Roadmap | partially_stale | SLO alert rules now exist in `slo-targets.yml` (written after this doc), but SLO error budget formula is broken (see F-10-006); APM/continuous profiling not deployed |

---

### `docs/PERFORMANCE_OPTIMIZATION.md` — status: `partially_stale`

**Validation method:** Cross-referenced each named bottleneck fix against current source files using `grep`.

**Archived to:** `docs/audits/2026-04/_meta/prior-reports-archive/PERFORMANCE_OPTIMIZATION.archived.md`

| # | Claim | Original Location | Status | Evidence |
|---|---|---|---|---|
| 1 | "N+1 query fix COMPLETE — 98% query reduction (201+ → 2-3)" | PERFORMANCE_OPTIMIZATION.md §Critical #3 | current | `grep "get_bulk_price_history\|get_top_stocks" backend/repositories/` returns hits; `test_n1_query_fix.py` exists at `backend/tests/test_n1_query_fix.py` |
| 2 | "Elasticsearch removed — PostgreSQL FTS instead (saves $15-20/month)" | PERFORMANCE_OPTIMIZATION.md §Critical #4 | current | `prometheus.yml:62` comment confirms removal; `log_analysis.py:23` has optional import with graceful fallback |
| 3 | "`@cache_with_ttl` decorator is a NO-OP" | PERFORMANCE_OPTIMIZATION.md §Critical #1 | fully_stale | `grep "def cache_with_ttl" backend/utils/cache.py` returns a real implementation with Redis; this fix was completed outside this doc |
| 4 | "Redis maxmemory 128mb → 512mb fix pending" | PERFORMANCE_OPTIMIZATION.md §High #6 | fully_stale | Current `docker-compose.yml` Redis command uses `--maxmemory 256mb` (scope 13 confirmed); not 128mb; fix was partially applied |
| 5 | "Serial stock processing in Airflow DAG — 6-8h instead of <1h — fix pending" | PERFORMANCE_OPTIMIZATION.md §High #7 | fully_stale | `daily_stock_pipeline.py` now uses dynamic task mapping (Airflow 2.x); serial loop no longer present at DAG graph level (though F-06-001 means the DAG may not load at all) |
| 6 | "Alert thresholds: API p95 warning >400ms, critical >600ms" | PERFORMANCE_OPTIMIZATION.md §Alert Thresholds | partially_stale | `slo-targets.yml` uses 500ms as the fast-burn critical threshold; `investment-platform.yml` uses >2s for HighAPILatency warning. The doc's thresholds do not match deployed config |

---

### `docs/architecture/PERFORMANCE_OPTIMIZATIONS.md` — status: `partially_stale`

**Validation method:** Read document and cross-referenced each stated optimization against infrastructure files.

**Archived to:** `docs/audits/2026-04/_meta/prior-reports-archive/PERFORMANCE_OPTIMIZATIONS.archived.md`

| # | Claim | Original Location | Status | Evidence |
|---|---|---|---|---|
| 1 | "Redis configured with LRU eviction and 1GB memory limit" | PERFORMANCE_OPTIMIZATIONS.md §Caching | partially_stale | `docker-compose.yml` has `--maxmemory 256mb` (scope 13), not 1GB; LRU eviction policy not confirmed in compose file |
| 2 | "PostgreSQL: slow query logging for queries >100ms" | PERFORMANCE_OPTIMIZATIONS.md §Database | current | `infrastructure/postgres/postgresql.conf` (scope 07) confirmed this setting |
| 3 | "Docker: logging rotation to prevent disk fill" | PERFORMANCE_OPTIMIZATIONS.md §Docker | current | `docker-compose.yml` log driver options present per scope 13 review |
| 4 | "`docker-compose.performance.yml` created with resource limits" | PERFORMANCE_OPTIMIZATIONS.md §Docker | fully_stale | `find . -name "docker-compose.performance.yml"` returns no results; file does not exist in the repo |

---

## 3. Findings

| ID | Severity | Category | File:Line | Title | Description | Recommendation | Acceptance Test Hint | Effort (h) | Loki Actionable | Cross Scope |
|---|---|---|---|---|---|---|---|---|---|---|
| F-10-001 | critical | architecture | backend/monitoring/metrics_collector.py:658 | Custom registry isolation hides all SLA, health, and alerting Prometheus metrics | `metrics_collector.py` creates a private `CollectorRegistry` (line 35) and `generate_latest(registry)` (line 658) serves only that registry at `/metrics`. All metrics in `health_checks.py`, `alerting_system.py`, `application_monitoring.py`, `financial_monitoring.py`, and `database_performance.py` register to the **default** Prometheus registry (no `registry=` argument passed). `generate_latest()` (no args) for the default registry is never called. Result: Prometheus scrapes `/metrics` and gets only system/API/cost metrics — zero health-check, zero SLA compliance, zero alert metrics are exported. | Unify all monitoring modules onto one registry. Either (a) pass the custom `registry` object to every `Gauge/Counter/Histogram` instantiation in all monitoring submodules, or (b) remove the custom registry from `metrics_collector.py` and call `generate_latest()` (no args) so all modules contribute to the default registry. Option (b) requires checking for name collisions first (see F-10-002). | `curl http://localhost:8000/metrics` returns text containing `health_check_status` and `sla_compliance_percent`; Prometheus target shows no missing series for health alerts. | 4 | true | ["13-infra-deployment"] |
| F-10-002 | critical | bug | backend/monitoring/real_time_alerts.py:25-26 | `MimeText`/`MimeMultipart` import typo — email alerts silently fail at module load | Line 25: `from email.mime.text import MimeText` and line 26: `from email.mime.multipart import MimeMultipart`. The stdlib classes are `MIMEText` and `MIMEMultipart` (all-caps MIME prefix). Python raises `ImportError: cannot import name 'MimeText'` on any import of this module. Lines 409 and 420 also reference `MimeMultipart()` and `MimeText(html_body, 'html')` using the incorrect names. Since `__init__.py` imports `alertmanager_webhook` (not `real_time_alerts`) the module may not fail at startup unless imported elsewhere, but all email alert delivery paths are broken. | Change line 25 to `from email.mime.text import MIMEText`, line 26 to `from email.mime.multipart import MIMEMultipart`, line 409 to `msg = MIMEMultipart()`, and line 420 to `msg.attach(MIMEText(html_body, 'html'))`. | `python -c "from backend.monitoring.real_time_alerts import AlertSeverity"` exits 0 with no ImportError; integration test sends a test alert email. | 0.5 | true | [] |
| F-10-003 | critical | broken_import | backend/monitoring/metrics_collector.py:486 | `get_quality_summary` imported from `data_quality_metrics` does not exist | `metrics_collector.py:486` executes `from backend.monitoring.data_quality_metrics import get_quality_summary`. `data_quality_metrics.py` exports only `DataQualityMetricsCollector` (class, line 176) and `check_data_quality_with_metrics` (async function, line 543); no `get_quality_summary` function exists. The `ImportError` is caught by the surrounding `except Exception` at line 516, so `collect_data_quality_metrics()` silently fails every 10-second collection cycle. Gauges `data_quality_score`, `data_missing_percent`, and `data_staleness` are never populated. | Implement `get_quality_summary()` in `data_quality_metrics.py` as a synchronous wrapper around `DataQualityMetricsCollector.get_summary()`, or change the call site to use `await check_data_quality_with_metrics(...)`. | `grep -n "def get_quality_summary" backend/monitoring/data_quality_metrics.py` returns a hit; `data_quality_score` gauge has non-zero values in Prometheus after one collection cycle. | 1 | true | [] |
| F-10-004 | high | schema_mismatch | infrastructure/monitoring/alerts/investment-platform.yml:18 | Alert rule queries `api_request_duration_seconds_bucket` but served metric is `api_latency_seconds` | `investment-platform.yml:18` `HighAPILatency` expression: `histogram_quantile(0.95, sum(rate(api_request_duration_seconds_bucket[5m])) by (le)) > 2`. `metrics_collector.py:59` defines `api_latency_seconds` (served via custom registry). `backend/utils/monitoring.py:45` defines `api_request_duration_seconds` in a **separate** custom registry that is also never served (confirmed: `generate_latest(registry)` call in utils/monitoring.py is not wired to any HTTP endpoint). The same mismatch affects `slo-targets.yml:87,94,100,102` (all SLO latency recording rules) and all 5 Grafana dashboard panels querying `api_request_duration_seconds_bucket`. All latency alerts and dashboard panels return no data. | Rename `api_latency_seconds` in `metrics_collector.py:59` to `api_request_duration_seconds` to match alert rules and dashboards, OR update all alert rules and dashboards to use `api_latency_seconds`. Simultaneously wire `backend/utils/monitoring.py` metrics to the served endpoint (see F-10-001) to resolve the dual-registry issue. | `curl http://localhost:8000/metrics | grep api_request_duration_seconds_bucket` returns histogram bucket lines; `HighAPILatency` alert evaluates to a number (not NaN) in Prometheus. | 2 | true | ["01-backend-api"] |
| F-10-005 | high | bug | backend/monitoring/metrics_collector.py:296-297 | Counter internal `_value._value` mutated directly — breaks Prometheus atomicity guarantees | Lines 296-297: `network_bytes_sent.labels(interface='total')._value._value = net_io.bytes_sent` and `network_bytes_received.labels(interface='total')._value._value = net_io.bytes_recv`. This accesses the private internal attribute of a `Counter` object, bypassing the thread-safe increment API. Counters must be monotonically increasing; direct assignment can make the counter go backwards (e.g., if `net_io_counters()` resets after a reboot), violating Prometheus data model. Also breaks with any Prometheus client library version upgrade. | Counters of cumulative OS bytes should use a `Gauge` not a `Counter` (or use `REGISTRY` provided by node-exporter instead). Replace with `Gauge('network_bytes_sent_total', ...)` and call `.set(net_io.bytes_sent)`. Or remove and rely on node-exporter's `node_network_transmit_bytes_total`. | `network_bytes_sent_total` (renamed to Gauge) has monotonically non-decreasing values over multiple scrapes; no `_value._value` references remain in metrics_collector. | 1 | true | [] |
| F-10-006 | high | bug | infrastructure/monitoring/alerts/slo-targets.yml:187-193 | Error budget recording rule uses 6-hour window — cannot represent 30-day monthly budget | `slo:error_budget_remaining:ratio` (line 187) is derived from `slo:api_availability:ratio_rate6h`, a 6-hour rolling average. The comment at line 4 states the error budget is "~43 min/month downtime." Using a 6-hour window means: (a) the budget resets every 6 hours, so sustained low-level degradation never exhausts it; (b) a 1-hour outage looks catastrophic on the 6h window but "recovers" as soon as the window rolls past. `SLOErrorBudgetLow` and `SLOErrorBudgetExhausted` will not fire under gradual availability erosion. | Use Prometheus recording rules over a 30-day window: `sum_over_time(slo:api_availability:ratio_rate5m[30d])` or integrate with a purpose-built error budget tool (Sloth, Pyrra). At minimum replace the 6h reference with a 30d `avg_over_time`. | `slo:error_budget_remaining:ratio` decreases monotonically over the course of a simulated 30-day load test with 0.1% error injection; `SLOErrorBudgetLow` fires before the end of the month. | 3 | false | [] |
| F-10-007 | high | security | infrastructure/monitoring/prometheus.yml:85-87 | Airflow Prometheus scrape uses hardcoded plaintext password `admin` | `prometheus.yml:85-87` scrapes Airflow with `basic_auth: username: admin, password: admin`. This is a well-known default credential embedded in a committed config file that Prometheus workers load at runtime. Any operator with read access to this file or the Prometheus config can access the Airflow admin UI. Scope 08 (F-08-009) already flagged plaintext credentials in committed docs; this is an additional instance in infrastructure config. | Replace with `password_file: /etc/prometheus/secrets/airflow-password` and provision the secret via Kubernetes Secret or Docker secret, not via the YAML file. At minimum rotate the Airflow admin password and add `prometheus.yml` to `.gitleaks.toml` pattern scanning. | `grep "password:" infrastructure/monitoring/prometheus.yml` returns no hits (or returns a file reference, not a value); gitleaks scan of this file returns clean. | 1 | true | ["08-auth-security-compliance"] |
| F-10-008 | high | bug | backend/monitoring/metrics_collector.py:310-311 | `gc_collections` Counter incremented with current GC count on every cycle — monotonicity violated | Lines 310-311: `for i, count in enumerate(gc.get_count()): gc_collections.labels(generation=str(i)).inc(count)`. `gc.get_count()` returns the current number of collections since the **last** `gc.collect()` call, not since program start. On each 10-second collection loop, this increments the counter by a small number (typically 0-10), not the true cumulative GC count. The Prometheus `rate()` over this counter will be meaningful on average, but `gc.get_count()` does not reset between calls, so values are not deltas — they represent a rolling window internal to CPython's GC. The correct approach is to use `gc.get_stats()` which returns per-generation collection counts since program start. | Replace `gc.get_count()` with `gc.get_stats()` which returns a list of dicts with `'collected'` key representing cumulative total. Use `gc_collections.labels(generation=str(i))._value.set(stats['collected'])` with a Gauge, or track the delta manually. | `rate(gc_collections_total[5m])` in Prometheus returns a smooth non-spiky value during a GC-heavy workload; no counter resets visible in metrics. | 1 | true | [] |
| F-10-009 | high | stale_code | config/monitoring/prometheus.yml | Duplicate and divergent prometheus.yml in `config/monitoring/` — older stale copy | `config/monitoring/prometheus.yml` (57 lines) and `infrastructure/monitoring/prometheus.yml` (86 lines) are both present and differ significantly: different `external_labels`, missing `rule_files` section in the config copy, missing Nginx/Celery/Airflow scrape jobs, different job names (`backend` vs `investment-api`), different `metrics_path` for the backend (`/metrics` vs `/api/metrics`). The `config/monitoring/` copy appears to be an earlier draft never removed. Two authoritative Prometheus configs cause confusion about which is deployed. | Delete `config/monitoring/prometheus.yml`. The `infrastructure/monitoring/prometheus.yml` is the canonical config (has SLO rule references, complete scrape targets). Verify `docker-compose.yml` mounts `infrastructure/monitoring/prometheus.yml` as the active config. | `find . -name "prometheus.yml" -not -path "*/infrastructure/monitoring/*"` returns no hits; docker-compose volume mount references `infrastructure/monitoring/prometheus.yml`. | 0.5 | true | ["13-infra-deployment"] |
| F-10-010 | high | architecture | backend/monitoring/metrics_collector.py:669-683 | `on_event("startup"/"shutdown")` deprecated FastAPI lifecycle hooks | `metrics_collector.py:675` uses `@app.on_event("startup")` and `@app.on_event("shutdown")`. These decorators were deprecated in FastAPI 0.93 (released 2023-02) and removed in FastAPI 0.110+. The recommended pattern is `lifespan` context manager. If the application runs FastAPI >= 0.110, metrics collection never starts and never stops cleanly. | Replace with a `lifespan` async context manager: `from contextlib import asynccontextmanager; @asynccontextmanager async def lifespan(app): await metrics_collector.start_collection(); yield; await metrics_collector.stop_collection()` and pass to `FastAPI(lifespan=lifespan)`. | `grep -rn "on_event" backend/monitoring/metrics_collector.py` returns no hits; metrics collection starts on app startup (confirmed via log line "Started metrics collection"). | 1 | true | ["01-backend-api"] |
| F-10-011 | medium | architecture | backend/monitoring/health_checks.py:28 | Duplicate `HealthStatus` enum defined in `health_checks.py` and `health_system.py` | `health_checks.py:28` and `health_system.py:24` both define `class HealthStatus(Enum)` with identical values (`HEALTHY`, `DEGRADED`, `UNHEALTHY`, `CRITICAL`). Because they are separate enum classes, `HealthStatus.HEALTHY is health_system.HealthStatus.HEALTHY` is `False`. Any code that imports from one module and compares with a value from the other will fail equality checks silently. `__init__.py` uses `health_checks.setup_health_monitoring` while `health_system` is also imported independently. | Delete `health_system.HealthStatus` and import it from `health_checks` (or move to a shared `backend/monitoring/types.py`). Ensure all references in `health_system.py` use the imported type. | `grep -rn "class HealthStatus" backend/monitoring/` returns exactly 1 hit; `isinstance(health_system.ServiceHealth(...).status, health_checks.HealthStatus)` is `True`. | 1 | true | [] |
| F-10-012 | medium | security | backend/monitoring/sla_tracker.py:16,224 | `sla_tracker` uses `get_redis()` which is subject to the fail-open vulnerability (F-08-008) | `sla_tracker.py:16` imports `from backend.utils.cache import get_redis`; line 224 sets `self.redis = await get_redis()`. The auth scope audit (F-08-008) found that `redis_resilience.py` fails open — when Redis is unavailable, it returns a no-op client instead of raising. In `sla_tracker.py`, if Redis is in fail-open mode, SLA measurements silently succeed (`redis.set()` returns `True`) but are written to a no-op sink. SLA compliance records appear normal while actually empty. | Add a Redis connectivity check before recording SLA measurements: verify `self.redis` is not the fail-open stub before `redis.set()`. At minimum log a warning when Redis is in fail-open mode. Cross-reference scope 08 fix for the underlying fail-open issue. | `sla_tracker` raises `SLAStorageUnavailableError` (or logs a warning) when Redis is in fail-open mode; SLA measurements are not silently discarded. | 2 | false | ["08-auth-security-compliance"] |
| F-10-013 | medium | dead_code | backend/monitoring/metrics_collector.py:203-204 | `mttr` and `mtbf` Gauges defined but never populated in collection loop | `mttr = Gauge('mttr_minutes', ...)` and `mtbf = Gauge('mtbf_hours', ...)` are defined at module level (lines 203-204). The `update_mttr_mtbf()` method exists (line 651) but `grep -rn "update_mttr_mtbf" backend/` shows it is called only from tests, never from application code, incident handlers, or the collection loop. Both gauges always export the default value of 0, making the `mttr_minutes` and `mtbf_hours` time-series useless in Grafana. | Either: (a) wire `update_mttr_mtbf()` to an incident resolution event (e.g., from `alerting_system.py` when an alert resolves), computing MTTR from `alert.created_at` to resolution time; or (b) remove the gauges until an incident management workflow exists. | `mttr_minutes{service="database"}` in Prometheus shows non-zero values after simulating a health-check failure and recovery cycle. | 2 | true | [] |
| F-10-014 | medium | code_quality | infrastructure/monitoring/prometheus.yml:30 | Dev Prometheus config scrapes backend at `/api/metrics` but endpoint is `/metrics` | `prometheus.yml:30` sets `metrics_path: /api/metrics` for the `investment-api` job. `metrics_collector.py:669` registers `@app.get("/metrics")`. The correct scrape path is `/metrics`, not `/api/metrics`. In development, Prometheus will receive 404 responses for every scrape of the backend, producing `up{job="investment-api"} == 0` and triggering the `ServiceDown` alert continuously. | Change `prometheus.yml:30` from `metrics_path: /api/metrics` to `metrics_path: /metrics`. (The prod config already has the correct path at `prometheus.prod.yml:43`.) | `prometheus.yml` scrape config for `investment-api` has `metrics_path: /metrics`; Prometheus target page shows `investment-api` as UP. | 0.25 | true | [] |
| F-10-015 | low | doc_drift | backend/monitoring/metrics_collector.py:200 | `sla_compliance` Gauge label set differs between served custom registry and default registry instance | `metrics_collector.py:200` defines `sla_compliance = Gauge('sla_compliance_percent', ..., ['service'], registry=registry)` (1 label). `health_checks.py:66` defines `sla_compliance = Gauge('sla_compliance_percent', ..., ['service', 'sla_type', 'time_window'])` (3 labels) in the default (unserved) registry. The two definitions use different label sets for the same metric name. If F-10-001 is fixed by unifying to one registry, this will cause a Prometheus duplicate registration error at startup. | Consolidate to a single definition with the richer label set `['service', 'sla_type', 'time_window']` in one shared location. Remove the duplicate from whichever module is not the canonical owner. | `grep -rn "sla_compliance_percent" backend/monitoring/` returns exactly 1 definition; Prometheus startup logs show no `duplicate metrics collector registration` error. | 0.5 | true | [] |
| F-10-016 | low | testing_gap | backend/tests/test_monitoring_api.py:86 | Test suite queries `/api/health/metrics` but the metrics endpoint is `/metrics` | `test_monitoring_api.py:86,112,142,168` all call `client.get("/api/health/metrics")`. The actual endpoint registered in `metrics_collector.py:669` is `/metrics`. These tests would pass only if a separate health router also exposes a `/api/health/metrics` path — unconfirmed. If no such route exists, every test in `TestMetricsEndpoint` passes a 404 response without asserting the correct status code (tests would need to be checked for assertion strength). | Update all test assertions to call `/metrics` instead of `/api/health/metrics`, and add an explicit `assert response.status_code == 200` at the start of each test. | `pytest backend/tests/test_monitoring_api.py::TestMetricsEndpoint` exits 0 with all assertions against `response.status_code == 200`. | 1 | true | ["15-test-suite"] |
| F-10-017 | low | dead_code | backend/monitoring/log_analysis.py:585-601 | Elasticsearch setup code executes on every instantiation despite being permanently disabled | `log_analysis.py:585-601` `_setup_elasticsearch()` is called in `__init__`. Even though Elasticsearch is optional and the import is guarded at lines 26-30, the setup method still checks `AsyncElasticsearch is not None` and reads env vars on every `LogAnalysisSystem()` construction. This is dead execution — the env var `ELASTICSEARCH_URL` is never set (Elasticsearch was removed to save $15-20/month per PERFORMANCE_OPTIMIZATION.md). 20+ lines of dead setup code run pointlessly. | Remove `_setup_elasticsearch()` and its call from `__init__`, or guard the entire method behind an explicit `ENABLE_ELASTICSEARCH=true` feature flag that defaults off. | `grep -n "_setup_elasticsearch\|elasticsearch_client" backend/monitoring/log_analysis.py` returns 0 hits (after cleanup). | 0.5 | true | [] |

---

## 4. Cross-Scope Linkages

- `F-10-001` → scope `13-infra-deployment`: The `docker-compose.yml` mounts monitoring configs; unified registry fix may require updating the app startup sequence in the Dockerfile or compose service definition.
- `F-10-004` → scope `01-backend-api`: The `api_request_duration_seconds` histogram is defined in `backend/utils/monitoring.py` which is used by API middleware. Renaming or routing that metric to the served registry requires coordination with the API instrumentation layer.
- `F-10-007` → scope `08-auth-security-compliance`: The hardcoded Airflow scrape password is another instance of the plaintext-credential pattern flagged across multiple scopes (F-08-009). The gitleaks config should be extended to cover YAML files with `password:` literals.
- `F-10-009` → scope `13-infra-deployment`: The stale `config/monitoring/prometheus.yml` may be mounted by an older docker-compose variant; scope 13 should verify which config the running container uses.
- `F-10-010` → scope `01-backend-api`: `on_event` deprecation affects the main FastAPI app if other routers or the main app also use this pattern.
- `F-10-012` → scope `08-auth-security-compliance`: The Redis fail-open vulnerability (F-08-008) propagates into the SLA tracking path; the fix must be coordinated with scope 08's Redis resilience module.
- `F-10-016` → scope `15-test-suite`: Wrong test endpoint path means the monitoring test suite may be passing against 404s; scope 15 should verify assertion strength in all monitoring test files.

---

## 5. Risk-Prioritized Punch List (top 10)

1. **F-10-001** — Registry isolation hides all health and SLA metrics — The entire health-check, SLA compliance, and alerting Prometheus metric surface is invisible to the scraper. All SLO burn-rate alerts that depend on health-check data are moot. Zero operator visibility into system health via Prometheus/Grafana.
2. **F-10-004** — Latency metric name mismatch makes all latency alerts and dashboards return no data — Every SLO latency alert (`SLOLatencyFastBurn`, `SLOLatencySlowBurn`, `HighAPILatency`) and all Grafana p50/p95/p99 panels are permanently blank. MTTR for latency incidents is infinite.
3. **F-10-002** — `MimeText` import typo silences all email alert channels — Critical and emergency alerts that should page on-call via email are silently dropped. During an outage, no email is sent.
4. **F-10-003** — `get_quality_summary` phantom import — Data quality gauges are never populated; data freshness and anomaly metrics are permanently zero. Silent failures every 10 seconds.
5. **F-10-006** — Error budget formula uses 6-hour window — Monthly SLO compliance cannot be tracked. `SLOErrorBudgetLow` / `SLOErrorBudgetExhausted` alerts will never fire under the realistic "gradual degradation" failure mode. Error budget policy enforcement is impossible.
6. **F-10-007** — Hardcoded Airflow admin password in committed YAML — Credential exposure risk; rotatable in < 1 hour but requires immediate action if the repo is public or accessible to untrusted parties.
7. **F-10-005** — Counter internal mutation violates Prometheus data model — `rate()` on network counters may return negative values after system reboot; breaks counter semantics.
8. **F-10-008** — GC counter incremented with non-delta values — GC metrics are statistically incorrect; `rate(gc_collections_total[5m])` does not represent the true collection rate.
9. **F-10-010** — Deprecated FastAPI `on_event` lifecycle hooks — On FastAPI >= 0.110, metrics collection never starts; the entire `/metrics` endpoint returns stale or zero values.
10. **F-10-009** — Duplicate divergent `prometheus.yml` — Operator confusion about which config is deployed; the older copy has wrong scrape targets and will cause silent monitoring gaps if accidentally mounted.

---

## 6. Open Questions

- Q1: Is the application running FastAPI >= 0.110? If yes, F-10-010 is critical (not high) — metrics collection never starts at all, making F-10-001 even more severe.
- Q2: Which `prometheus.yml` is actually mounted in the running container — `config/monitoring/prometheus.yml` or `infrastructure/monitoring/prometheus.yml`? The answer determines whether the backend is currently being scraped at all (wrong metrics_path in the config copy would give persistent `up == 0`).
- Q3: Is there a production incident management workflow that should populate MTTR/MTBF (F-10-013)? If not, these gauges should be removed rather than left as dead zeros.
- Q4: The `sla_tracker.py` stores SLA measurements in Redis with TTL. Given the Redis fail-open vulnerability (F-08-008), has any SLA data been silently lost in production? Is there a secondary persistent store (PostgreSQL) for SLA records?
- Q5: The Loki config has `auth_enabled: false`. Is Loki exposed only on internal Docker network, or is it reachable externally? No network policy is visible in this scope.
