# Workstream A: secret-rotation

## 1. Cluster overview

**Problem statement.** The repository contains the production Postgres password (`9v1g^OV9XUwzUP6cEgCYgNOE`), Redis password (`RsYque&Xh%TUD*Nv^7k7B8X3`), Elasticsearch password (`4Bx+UM1CdSiEbMlQueRVvda+A4fLzCRsyuHUHbv5wMw=`), Grafana password (`a2A5j4JQ0nF8aTLyIYwRgZnMLQpIu5lW9jYx6pB5Xdw=`), an Airflow Prometheus basic-auth `admin:admin`, plus JWT/Fernet/HF/Google API tokens — all in committed source files (`alembic.ini`, docs, scripts, infrastructure configs). They were flagged in an earlier `SECRET_ROTATION_PLAN` that was never executed. Compounding this, `.env*` files are baked into Docker images, default fallback passwords (`'postgres'`) silently mask misconfiguration, and the env-template documentation (`.env.example` 242 vars vs `docs/ENVIRONMENT.md` 138 documented) is 24+ days stale and fragmented across three sources.

**Root cause & scope.** The single root cause is that secrets entered git history before any rotation gate was enforced. Fixing the live codebase (rotation, env-driven config, `.env` exclusion) is necessary but insufficient — historic git objects must be purged with `git filter-repo` to satisfy auditors. The cluster spans 10 audit scopes (05, 07, 08, 10, 12, 13, 15, 16, 17, 18) covering Postgres, Redis, Grafana, Elasticsearch, Airflow, Prometheus, JWT, Fernet, ETL, Alembic, Dockerfiles, scripts, conftest fixtures, and frontend env-mode handling. F-12-005, F-12-010, F-12-016, F-10-004, F-10-013, F-15-024, F-18-005 are non-secret findings that traveled with this cluster (cross-scope env/config drift) and are addressed at the end as second-order cleanup.

## 2. Member findings

All 25 assigned IDs are listed below.

- **F-05-003**: Production Postgres password exposed in `docs/reports/STOCK_UNIVERSE_EXPANSION_SUCCESS.md:81`.
- **F-05-012**: Default `'postgres'` fallback password in 4 ETL files (`backend/etl/data_loader.py:36`, `stock_universe_manager.py:30`, `cache_warming.py:73,151`).
- **F-07-001**: Production Postgres password committed in `alembic.ini:53` `sqlalchemy.url`.
- **F-08-009** (anchor): Plaintext credentials (DB/JWT/Fernet/Google/HF) across `docs/security/*` and `docs/reports/security-*`. Requires git history purge.
- **F-08-012**: Hardcoded `ALLOWED_ORIGINS` in `backend/security/security_config.py:46-53` evaluated at class-body load time.
- **F-10-004**: Prometheus alert metric name mismatch (`api_request_duration_seconds_bucket` vs served `api_latency_seconds`). Non-secret config drift; included as it touches `infrastructure/monitoring/`.
- **F-10-007**: Hardcoded `password: admin` for Airflow scrape in `infrastructure/monitoring/prometheus.yml:85-87`.
- **F-10-013**: `mttr`/`mtbf` Gauges in `backend/monitoring/metrics_collector.py:203-204` defined but never populated. Non-secret; second-order cleanup.
- **F-12-005**: `frontend/web/src/pages/InvestmentThesis.tsx` bypasses central API service. Non-secret architecture finding; second-order.
- **F-12-010**: `ErrorBoundary.tsx:38` uses unreliable `process.env.NODE_ENV` in Vite. Second-order.
- **F-12-016**: Absorbed by F-12-005 — same file, same root cause (`API_BASE_URL` hardcode is fixed by migrating to central api service). Tracked in §3 step 13 alongside F-12-005.
- **F-13-004**: `Dockerfile.ml-api:20`, `Dockerfile.ml-monitoring:20`, `Dockerfile.ml-scheduler:21` `COPY .env* ./` bakes env into image.
- **F-13-020**: `.dockerignore:77-79` negation rules — informational, no change required (auditor verified). Documented in step 6.
- **F-15-012**: `test_performance.sh:15` hardcodes `PGPASSWORD=postgres`.
- **F-15-024**: `setup_test_environment` autouse fixture in `backend/tests/conftest.py:460-477` re-monkeypatches envvars per test. Non-secret performance; second-order.
- **F-16-003**: `.env.example` has 242 vars but `docs/ENVIRONMENT.md` documents only 138 (57% coverage).
- **F-16-006**: Three overlapping env doc sources (`.env.example`, `docs/ENVIRONMENT.md`, root `ENVIRONMENT.md`). Note: root `ENVIRONMENT.md` does NOT exist on disk (verified) — finding partially stale; addressed by consolidating to single source.
- **F-16-007**: `.env.airflow` (45 vars) vs `.env.example` (242) — fragmented, no cross-reference.
- **F-16-009**: `.env_backup_DONOTUSE/` directory referenced. Note: directory does NOT exist on current `main` (verified `ls .env_backup_DONOTUSE` returns no such file). Step 7 verifies absence in working tree and history.
- **F-16-010**: `docs/ENVIRONMENT.md` last updated 2026-03-04 (>30d stale).
- **F-17-001** (anchor): Hardcoded DB/Redis/Elasticsearch/Grafana passwords in `scripts/testing/test_docker_connections.py:20,42` and `test_all_passwords.py:94,127`.
- **F-17-002**: Hardcoded DB password in `scripts/data/background_loader.py:33`, `background_loader_enhanced.py:56`, `load_data_now.py:30`, `scripts/deployment/start_data_loading.sh:58`, `scripts/init_database_windows.ps1:74`. Note: `scripts/data/load_historical_data.py:96` cited but file NOT FOUND on disk (verified) — flagged in step 9 as advisory grep.
- **F-17-007**: `scripts/init_database.sh:20` uses unsafe `export $(xargs)`; `init_database_fixed.sh` already has the safe `set -a; source; set +a` pattern.
- **F-17-009**: 14 duplicate scripts (`activate_pipeline.py`, `verify_database.py`, `mock_data_generator.py`, `simple_mock_generator.py`, …) each managing DB connectivity independently with the same hardcoded fallback.
- **F-18-005**: `docs/DOCUMENTATION_HEALTH.md §1.2` metrics dashboard is template-only, not implemented. Non-secret; second-order.

## 3. Sequenced fix steps

> **CRITICAL ORDERING:** Steps 1–2 (rotate live secrets) must complete before any code/config replacement or git history rewrite. Step 11 (`git filter-repo`) is destructive and gated on human ack.

**Step 1: Rotate ALL exposed credentials in production**
- Files: none (operational)
- Action: Generate new credentials and roll them in production for: Postgres (`POSTGRES_PASSWORD`), Redis (`REDIS_PASSWORD`), Elasticsearch (`ELASTIC_PASSWORD`), Grafana admin, Airflow admin, JWT signing secret, Fernet key (`GDPR_ENCRYPTION_KEY`), Google API key, HuggingFace token. Distribute via secret manager (Vault/Doppler/AWS Secrets Manager/1Password).
- Pass-after test: Old DB password fails: `PGPASSWORD='9v1g^OV9XUwzUP6cEgCYgNOE' psql -h <prod-host> -U postgres -d investment_db -c '\q'` exits non-zero with auth error.
- Path verified: yes (operational, no path)
- requires_human_ack: **true** (production rotation, coordination)

**Step 2: Update `.env.example` and provision env in CI/staging/prod**
- Files: `.env.example`, `.env.airflow`, deployment configs (out of scope but must be coordinated)
- Action: Ensure `.env.example` lists every required var with placeholder `<set-in-secret-manager>`. Verify `.env.example` has no real secret values (it should not). Add cross-reference comment block in `.env.airflow` to `.env.example`.
- Pass-after test: `grep -E "9v1g|RsYque|4Bx\+UM|a2A5j4" .env.example .env.airflow` returns no matches.
- Path verified: yes

**Step 3 (F-07-001): Replace `alembic.ini` plaintext URL**
- Files: `alembic.ini:53`
- Action: Change `sqlalchemy.url = postgresql://postgres:9v1g...` to `sqlalchemy.url = ${DATABASE_URL}`. In `alembic/env.py`, set `config.set_main_option("sqlalchemy.url", os.environ["DATABASE_URL"])` before `engine_from_config`. Add comment forbidding hardcoded creds.
- Fail-first test: `grep -F '9v1g' alembic.ini` currently returns line 53 (proves bug).
- Pass-after test: `grep -F '9v1g' alembic.ini` returns no matches; `DATABASE_URL=postgresql://... alembic current` succeeds.
- Path verified: yes

**Step 4 (F-05-003): Redact production password from docs**
- Files: `docs/reports/STOCK_UNIVERSE_EXPANSION_SUCCESS.md:81`, `docs/security/SECURITY_CREDENTIALS_AUDIT.md:33-34`, all `docs/security/*` and `docs/reports/security-*` referenced by F-08-009.
- Action: Replace literal credentials with `<REDACTED>` or `$DB_PASSWORD`. Run `grep -rE "9v1g|RsYque|4Bx\+UM|a2A5j4|<jwt-hex>|<fernet>" docs/` and redact every hit.
- Fail-first test: `grep -rF '9v1g' docs/ | wc -l` currently > 0.
- Pass-after test: `grep -rE "9v1g|RsYque|4Bx\+UM|a2A5j4" docs/` returns 0 hits.
- Path verified: yes

**Step 5 (F-05-012, F-17-001, F-17-002, F-17-009): Remove default-fallback DB passwords from code**
- Files:
  - `backend/etl/data_loader.py:36`
  - `backend/etl/stock_universe_manager.py:30`
  - `backend/etl/cache_warming.py:73,151`
  - `scripts/testing/test_docker_connections.py:20,42`
  - `scripts/testing/test_all_passwords.py:94,127`
  - `scripts/data/background_loader.py:33`
  - `scripts/data/background_loader_enhanced.py:56`
  - `scripts/data/load_data_now.py:30`
  - `scripts/data/activate_pipeline.py:63`
  - `scripts/verify_database.py:34`
  - `scripts/data/mock_data_generator.py:38`
  - `scripts/data/simple_mock_generator.py:36`
  - `scripts/deployment/start_data_loading.sh:58`
  - `scripts/init_database_windows.ps1:74`
- Action: Replace every `os.getenv('POSTGRES_PASSWORD', '<default>')` with a strict accessor. Create new shared module `scripts/lib/db_connect.py` (per F-17-009) exposing `get_engine()` that reads `DATABASE_URL` and raises `RuntimeError` if missing. All scripts import from it. For ETL files, reuse `backend/config/settings.py` accessor.
- Fail-first test: `grep -rnE "(9v1g|getenv\([^)]*['\"]postgres['\"])" backend/etl/ scripts/` currently returns hits.
- Pass-after test: `grep -rnE "(9v1g|RsYque|4Bx\+UM|a2A5j4)" backend/ scripts/ | grep -v "audits/"` returns 0; `POSTGRES_PASSWORD= python3 -c "from scripts.lib.db_connect import get_engine; get_engine()"` raises `RuntimeError`.
- Path verified: yes (all files except `scripts/data/load_historical_data.py:96` cited in F-17-002 — file does not exist; mark as advisory grep, no edit needed).

**Step 6 (F-13-004, F-13-020): Stop baking `.env*` into Docker images**
- Files: `Dockerfile.ml-api:20`, `Dockerfile.ml-monitoring:20`, `Dockerfile.ml-scheduler:21`, `.dockerignore:77-79`
- Action: Delete `COPY .env* ./` lines. Update docker-compose to inject env via `env_file:` or `environment:`. Add an explanatory comment block in `.dockerignore` describing the negation precedence so future edits don't break the rule.
- Fail-first test: `grep -nE "^COPY \.env" Dockerfile.ml-*` currently returns 3 hits.
- Pass-after test: `grep -rnE "^COPY \.env" Dockerfile.ml-*` returns 0; built image: `docker run --rm <ml-api-image> sh -c "ls -la / | grep -c '\.env'"` returns 0.
- Path verified: yes (Dockerfile.ml-api, .dockerignore confirmed)

**Step 7 (F-15-012): Fix performance test script**
- Files: `test_performance.sh:15`
- Action: Replace `PGPASSWORD=postgres` with `PGPASSWORD="${TEST_DB_PASSWORD:?TEST_DB_PASSWORD must be set}"`. Add comment: `# Local-only — not for CI`.
- Fail-first test: `grep -F 'PGPASSWORD=postgres' test_performance.sh` currently matches.
- Pass-after test: `grep -F 'PGPASSWORD=postgres' test_performance.sh` returns 0 matches.
- Path verified: yes

**Step 8 (F-17-007): Replace unsafe `.env` xargs sourcing**
- Files: `scripts/init_database.sh:20`
- Action: Replace `export $(cat .env | grep -v '^#' | xargs)` with the `set -a; source "$ENV_FILE"; set +a` pattern already used in `scripts/init_database_fixed.sh:33`. Either remove `init_database.sh` or have it `exec` the fixed version.
- Fail-first test: With a `.env` containing `DB_PASSWORD='ab cd&ef'`, run `bash scripts/init_database.sh` — currently fails or mis-parses.
- Pass-after test: Same case succeeds; `bash -n scripts/init_database.sh` passes; `grep -F 'set -a' scripts/init_database.sh` matches.
- Path verified: yes

**Step 9 (F-10-007): Externalize Airflow Prometheus password**
- Files: `infrastructure/monitoring/prometheus.yml:85-87`
- Action: Replace `password: admin` with `password_file: /etc/prometheus/secrets/airflow-password` and provision via Kubernetes/Docker secret. Rotate the Airflow admin password (covered by Step 1) and add `infrastructure/monitoring/` to `.gitleaks.toml` scan paths.
- Fail-first test: `grep -nE '^[[:space:]]*password:' infrastructure/monitoring/prometheus.yml` currently matches.
- Pass-after test: `grep -nE '^[[:space:]]*password:' infrastructure/monitoring/prometheus.yml` returns no plaintext value (only `password_file:` entries).
- Path verified: yes

**Step 10 (F-08-012): Move `ALLOWED_ORIGINS` to env CSV with lazy eval**
- Files: `backend/security/security_config.py:46-53`
- Action: Replace class-body literal with a `@property`/`classmethod` or factory that reads `os.getenv("ALLOWED_ORIGINS", "").split(",")` at request time, ensuring env is evaluated when middleware initializes, not at import.
- Pass-after test: `ALLOWED_ORIGINS="https://a.com,https://b.com" python -c "from backend.security.security_config import SecurityConfig; print(SecurityConfig().allowed_origins)"` prints the env-driven list.
- Path verified: yes

**Step 11 (F-08-009, F-16-009): Purge git history with `git filter-repo`**
- Files: entire git history
- Action: After ALL above steps merge to `main` and the rotation in Step 1 is confirmed:
  1. Mirror clone the repo.
  2. Run `git filter-repo --replace-text replacements.txt` where `replacements.txt` lists every leaked secret literal.
  3. Run `git filter-repo --invert-paths --path .env_backup_DONOTUSE/` (defense-in-depth even though the dir is not on `main`).
  4. Force-push to all remotes and notify all developers to re-clone.
  5. Re-run `gitleaks detect --source . --log-opts="--all"`; confirm 0 findings.
- Fail-first test: `gitleaks detect --source . --log-opts="--all" --no-banner` currently reports >0 leaks.
- Pass-after test: `gitleaks detect --source . --log-opts="--all" --no-banner` reports 0; `git log --all -S '9v1g^OV9XUwzUP6cEgCYgNOE' --oneline` returns 0 commits.
- Path verified: yes (operation, not a path)
- requires_human_ack: **true** (DESTRUCTIVE — rewrites all history; coordinator must approve, schedule team re-clone window)

**Step 12 (F-16-003, F-16-006, F-16-007, F-16-010): Consolidate env documentation**
- Files: `docs/ENVIRONMENT.md`, `.env.example`, `.env.airflow`
- Action: Choose `docs/ENVIRONMENT.md` as single source. Auto-generate the variables table from `.env.example` using a small script (`scripts/lib/generate_env_docs.py`). Document Redis 7 / `redis.asyncio` migration. Update "Last updated" to current date. Add cross-reference between `.env.example` and `.env.airflow`. Confirm root `ENVIRONMENT.md` is absent (verified — F-16-006 partially stale).
- Pass-after test: `awk -F'`' '/\| `[A-Z_]+` /{c++} END{print c}' docs/ENVIRONMENT.md` returns >230; `grep -c "redis.asyncio" docs/ENVIRONMENT.md` >= 1; `grep -c "Last updated: 2026-04" docs/ENVIRONMENT.md` >= 1.
- Path verified: yes

**Step 13 (F-12-005, F-12-010, F-12-016): Frontend env-mode and central API migration**
- Files: `frontend/web/src/pages/InvestmentThesis.tsx:30,53,87,137`, `frontend/web/src/components/common/ErrorBoundary.tsx:38,99`, `frontend/web/src/services/api.service.ts`, `frontend/web/src/config/api.config.ts`
- Action: (a) In `ErrorBoundary.tsx` replace both `process.env.NODE_ENV` checks with `import.meta.env.PROD` (and `!import.meta.env.PROD` respectively). (b) In `InvestmentThesis.tsx`, remove raw axios import and `localStorage.getItem('access_token')`; route through central `api` service; add `thesis` endpoints to `api.config.ts`. F-12-016 absorbed.
- Pass-after test: `grep -nF 'process.env.NODE_ENV' frontend/web/src/components/common/ErrorBoundary.tsx` returns 0; `grep -nE "from ['\"]axios['\"]" frontend/web/src/pages/InvestmentThesis.tsx` returns 0; build: `cd frontend/web && npm run build` succeeds.
- Path verified: yes

**Step 14 (F-10-004): Reconcile Prometheus latency metric naming**
- Files: `backend/monitoring/metrics_collector.py:59`, `backend/utils/monitoring.py:45`, `infrastructure/monitoring/alerts/investment-platform.yml:18`, `infrastructure/monitoring/alerts/slo-targets.yml:87,94,100,102`, all 5 Grafana dashboard panels.
- Action: Rename `api_latency_seconds` → `api_request_duration_seconds` in `metrics_collector.py:59` (canonical Prometheus naming). Wire `backend/utils/monitoring.py` registry to the served `/metrics` endpoint or remove the orphan registry. Verify alert and dashboards now resolve.
- Pass-after test: `curl -s http://localhost:8000/metrics | grep -c '^api_request_duration_seconds_bucket'` > 0; `promtool check rules infrastructure/monitoring/alerts/*.yml` passes.
- Path verified: yes

**Step 15 (F-10-013): Wire or remove `mttr`/`mtbf` gauges**
- Files: `backend/monitoring/metrics_collector.py:203-204,651`
- Action: Either (a) call `update_mttr_mtbf()` from the alert resolution handler in `backend/monitoring/alerting_system.py`, or (b) remove the gauges and their tests until incident workflow exists. Recommend (a) if alerting_system has a resolved-event hook; otherwise (b).
- Pass-after test: After simulating an incident: `curl -s :8000/metrics | grep mttr_minutes` shows non-zero — OR the gauges and dead method are deleted and `pytest backend/tests/monitoring/` passes.
- Path verified: yes

**Step 16 (F-15-024): Fix conftest autouse env-fixture overhead**
- Files: `backend/tests/conftest.py:460-477`
- Action: Change `scope="function"` → `scope="session"` on `setup_test_environment`, OR delete the fixture entirely (the same envvars are already set at module-load time, lines 7-12).
- Pass-after test: `time pytest backend/tests/ --co -q` is at least 15% faster than baseline (record baseline first).
- Path verified: yes

**Step 17 (F-18-005): Clarify or implement docs-health metrics**
- Files: `docs/DOCUMENTATION_HEALTH.md:§1.2`
- Action: Minimum: prepend a notice "Example metrics — not real-time; see future-work plan." Optional: add `.reports/doc-health/latest.json` generation in CI.
- Pass-after test: `grep -i 'example metrics' docs/DOCUMENTATION_HEALTH.md` returns a hit.
- Path verified: yes

## 4. Files touched

- `alembic.ini`
- `.env.example`, `.env.airflow`, `.dockerignore`
- `Dockerfile.ml-api`, `Dockerfile.ml-monitoring`, `Dockerfile.ml-scheduler`
- `test_performance.sh`
- `backend/etl/data_loader.py`, `backend/etl/stock_universe_manager.py`, `backend/etl/cache_warming.py`
- `backend/security/security_config.py`
- `backend/monitoring/metrics_collector.py`, `backend/utils/monitoring.py`
- `backend/tests/conftest.py`
- `infrastructure/monitoring/prometheus.yml`
- `infrastructure/monitoring/alerts/investment-platform.yml`, `infrastructure/monitoring/alerts/slo-targets.yml`
- `frontend/web/src/components/common/ErrorBoundary.tsx`
- `frontend/web/src/pages/InvestmentThesis.tsx`
- `frontend/web/src/services/api.service.ts`, `frontend/web/src/config/api.config.ts`
- `scripts/init_database.sh`
- `scripts/testing/test_docker_connections.py`, `scripts/testing/test_all_passwords.py`
- `scripts/data/background_loader.py`, `scripts/data/background_loader_enhanced.py`, `scripts/data/load_data_now.py`, `scripts/data/activate_pipeline.py`, `scripts/data/mock_data_generator.py`, `scripts/data/simple_mock_generator.py`
- `scripts/verify_database.py`
- `scripts/deployment/start_data_loading.sh`
- `scripts/init_database_windows.ps1`
- `scripts/lib/db_connect.py` (NEW)
- `scripts/lib/generate_env_docs.py` (NEW, optional)
- `docs/reports/STOCK_UNIVERSE_EXPANSION_SUCCESS.md`
- `docs/security/SECURITY_CREDENTIALS_AUDIT.md` (and other `docs/security/*`, `docs/reports/security-*`)
- `docs/ENVIRONMENT.md`
- `docs/DOCUMENTATION_HEALTH.md`
- `replacements.txt` (transient, used for `git filter-repo`)
- `.gitleaks.toml`

## 5. Acceptance tests (consolidated)

```bash
# Step 1 — production rotation
PGPASSWORD='9v1g^OV9XUwzUP6cEgCYgNOE' psql -h <prod> -U postgres -d investment_db -c '\q'   # MUST FAIL

# Step 2 — env templates clean
grep -E "9v1g|RsYque|4Bx\+UM|a2A5j4" .env.example .env.airflow                              # 0 matches

# Step 3 — alembic
grep -F '9v1g' alembic.ini                                                                  # 0
DATABASE_URL=postgresql://test:test@localhost:5432/test alembic current                     # exit 0

# Step 4 — docs redacted
grep -rE "9v1g|RsYque|4Bx\+UM|a2A5j4" docs/                                                 # 0

# Step 5 — code clean + strict env
grep -rnE "(9v1g|RsYque|4Bx\+UM|a2A5j4)" backend/ scripts/ | grep -v audits/                # 0
POSTGRES_PASSWORD= python3 -c "from scripts.lib.db_connect import get_engine; get_engine()" # raises

# Step 6 — Dockerfiles
grep -nE "^COPY \.env" Dockerfile.ml-api Dockerfile.ml-monitoring Dockerfile.ml-scheduler   # 0
docker run --rm <ml-api-image> sh -c "ls -la / | grep -c '\.env'"                           # 0

# Step 7 — perf script
grep -F 'PGPASSWORD=postgres' test_performance.sh                                           # 0

# Step 8 — init_database.sh
bash -n scripts/init_database.sh                                                            # exit 0
grep -F 'set -a' scripts/init_database.sh                                                   # 1+

# Step 9 — prometheus
grep -nE '^[[:space:]]*password:[[:space:]]*[^\s]+' infrastructure/monitoring/prometheus.yml # 0

# Step 10 — ALLOWED_ORIGINS
ALLOWED_ORIGINS="https://a.com,https://b.com" python -c "from backend.security.security_config import SecurityConfig; print(SecurityConfig().allowed_origins)"

# Step 11 — git history (DESTRUCTIVE)
gitleaks detect --source . --log-opts="--all" --no-banner                                   # 0 leaks
git log --all -S '9v1g^OV9XUwzUP6cEgCYgNOE' --oneline                                       # 0 commits

# Step 12 — env docs
awk -F'`' '/\| `[A-Z_]+` /{c++} END{print c}' docs/ENVIRONMENT.md                           # >= 230
grep -c 'redis.asyncio' docs/ENVIRONMENT.md                                                 # >= 1

# Step 13 — frontend
grep -nF 'process.env.NODE_ENV' frontend/web/src/components/common/ErrorBoundary.tsx        # 0
grep -nE "from ['\"]axios['\"]" frontend/web/src/pages/InvestmentThesis.tsx                  # 0
cd frontend/web && npm run build                                                            # exit 0

# Step 14 — Prometheus latency metric
curl -s http://localhost:8000/metrics | grep -c '^api_request_duration_seconds_bucket'      # > 0
promtool check rules infrastructure/monitoring/alerts/*.yml                                 # exit 0

# Step 15 — mttr/mtbf
curl -s :8000/metrics | grep -c '^mttr_minutes '                                            # > 0  (option a) OR
grep -c 'mttr_minutes' backend/monitoring/metrics_collector.py                              # 0  (option b)

# Step 16 — conftest perf
pytest backend/tests/ --co -q                                                               # >=15% faster vs baseline

# Step 17 — docs-health notice
grep -i 'example metrics' docs/DOCUMENTATION_HEALTH.md                                      # >= 1
```

## 6. Rollback plan

| Step | Rollback |
|------|----------|
| 1 (rotation) | Re-issue previous credential to all consumers; revoke new ones. Plan worst-case 30-minute downtime window during rotation. |
| 2–10, 12–17 (code/config) | Revert merge commit; redeploy. All changes are normal git commits. |
| 11 (`git filter-repo`) | NOT REVERSIBLE in-place. Mitigation: keep a tagged mirror of pre-rewrite history at `archive/pre-history-purge` in a private S3 bucket for forensic/legal needs only. Developers re-clone from rewritten remote. |

## 7. Dependencies

- depends_on:
  - `[]` — Cluster A is the upstream root-cause cluster. No prerequisite cluster.
- blocks:
  - `[{workstream: "B (env-config)", type: blocks, reason: "Env-config consolidation depends on .env.example/docs/ENVIRONMENT.md being canonical (Step 12)."}]`
  - `[{workstream: "any compliance/SOC2 cluster", type: blocks, reason: "Audit attestation requires gitleaks-clean history (Step 11)."}]`
  - `[{workstream: "Docker/deployment cluster", type: soft, reason: "Dockerfile env injection (Step 6) coordinates with deployment env_file plumbing."}]`

## 8. Effort & cost estimate

- Effort range: **22–32 hours** for an experienced implementer (Loki).
  - Steps 1–2 (rotate + provision): 4–6h coordinated with ops
  - Steps 3–10 (code/config replacements): 10–14h
  - Step 11 (`git filter-repo`): 2–4h plus team re-clone window
  - Steps 12–17 (docs/frontend/metrics/cleanup): 6–8h
  - Excludes review, deploy, and team rotation-coordination overhead.
- Estimated Loki token cost: **~$3.50–$5.50** total across all steps assuming Sonnet for steps 5/13/14 and Haiku for the remaining mechanical edits.

## 9. Loki-actionable status

- Overall: **partial** — steps 1, 2, and 11 require human decision/coordination.
- requires_human_ack: **true** (Step 1 = production secret rotation; Step 11 = destructive `git filter-repo` rewrite — see PRD §2 destructive-ops gate).
- Per-step:
  - Step 1: **human_ack** (production rotation; ops/security must execute and confirm)
  - Step 2: **partial** (Loki can edit `.env.example` and `.env.airflow`; provisioning in CI/staging/prod is human)
  - Steps 3–10: **loki-actionable**
  - Step 11: **human_ack — HALT BEFORE EXEC.** Loki may prepare `replacements.txt` and a runbook but MUST NOT run `git filter-repo`.
  - Steps 12, 14, 15, 16, 17: **loki-actionable**
  - Step 13: **loki-actionable**

## 10. Risks (production rollout)

| Risk | Mitigation |
|------|------------|
| Production outage during password rotation if any consumer still holds old creds | Use blue/green credential window: provision new secret alongside old, update consumers, then revoke old. Verify with `pg_stat_activity` no clients on old auth path before revoking. |
| `git filter-repo` breaks every active developer's clone and any open PR | Schedule a known maintenance window; freeze merges; notify all contributors; provide re-clone runbook; coordinate with GitHub Actions cache invalidation. |
| External services (Grafana dashboards, Airflow webhooks) reference rotated tokens | Inventory all consumers in Step 1 before rotation; update each via secret manager. |
| `alembic.ini` change breaks CI/local dev that relied on hardcoded URL | Update `alembic/env.py` and `Makefile`/`docker-compose.yml` simultaneously; document the env var in `docs/ENVIRONMENT.md`. |
| Removing default-fallback passwords breaks staging that depended on `'postgres'` default | Provision `POSTGRES_PASSWORD` in staging before merging Step 5; add a one-line CI check that fails fast if env vars are missing. |
| Renaming `api_latency_seconds` (Step 14) breaks any Grafana queries currently using it | Search `infrastructure/monitoring/grafana/` for `api_latency_seconds` references and update in same PR. |
| `import.meta.env.PROD` change (Step 13) misbehaves in tests using JSDOM | Update Vitest config to set `import.meta.env` correctly; run full frontend test suite in PR. |
| Force-push to main triggers branch protection / signed-commit policies | Temporarily relax branch protection during the maintenance window; re-enable immediately after. |
| Secrets leaked to external mirrors (forks, GitHub caches, package registries, Sentry/log aggregators) | Step 1 rotation is what actually mitigates this — `git filter-repo` only addresses the canonical history. Rotation is the load-bearing mitigation; history rewrite is hygiene. |

---

**Final assertion:** All 25 assigned findings (F-05-003, F-05-012, F-07-001, F-08-009, F-08-012, F-10-004, F-10-007, F-10-013, F-12-005, F-12-010, F-12-016, F-13-004, F-13-020, F-15-012, F-15-024, F-16-003, F-16-006, F-16-007, F-16-009, F-16-010, F-17-001, F-17-002, F-17-007, F-17-009, F-18-005) are referenced in §2.
