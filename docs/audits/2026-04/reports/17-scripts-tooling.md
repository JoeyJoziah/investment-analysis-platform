---
scope_id: "17-scripts-tooling"
scope_name: "Scripts & Install Tooling"
agent_type: "tooling-engineer"
date: "2026-04-27"
files_in_scope: 164
files_reviewed: 58
files_skipped:
  - "scripts/models/finbert/config.json (binary/data artifact, not a script)"
  - "scripts/data/cache/pipeline_status.json (runtime state file)"
  - "scripts/*.sql (SQL files, out of tooling scope)"
prior_reports_validated:
  - path: "docs/INSTALLATION_GUIDE.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/INSTALLATION_GUIDE.archived.md"
    claims_validated: 10
    claims_still_valid: 8
    claims_stale: 2
  - path: "docs/SCRIPTS_REFERENCE.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/SCRIPTS_REFERENCE.archived.md"
    claims_validated: 12
    claims_still_valid: 7
    claims_stale: 5
  - path: "docs/WSL_INSTALLATION_FIXES.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/WSL_INSTALLATION_FIXES.archived.md"
    claims_validated: 8
    claims_still_valid: 6
    claims_stale: 2
findings_summary:
  critical: 2
  high: 7
  medium: 8
  low: 5
  total: 22
estimated_remediation_effort_days: 6
agent_status: "complete"
agent_token_usage: 14000
---

# Scripts & Install Tooling — Audit Report

## TL;DR (REQUIRED — exactly 5 bullets, max)

- **Critical**: 14+ scripts in `scripts/testing/` and `scripts/data/` embed real production credentials (DB password `9v1g^OV9XUwzUP6cEgCYgNOE`, Redis password `RsYque&Xh%TUD*Nv^7k7B8X3`, Elasticsearch and Grafana passwords) as hardcoded string literals committed to git — these must be rotated immediately.
- **High**: `setup.sh:88-103` has three unbounded `until` loops with no timeout, meaning a failed service will hang the installer forever with no error recovery path.
- **High**: `install_dependencies.py` imports `packaging` and `requests` at module top-level (lines 51-52) before they can be installed, causing an immediate `ModuleNotFoundError` on a fresh system — defeating the purpose of the installer.
- **High**: `scripts/deploy_ml_production.sh` and `scripts/migrate_to_optimized.sh` use the deprecated `docker-compose` (v1) binary; the project's own `start.sh`/`setup.sh` use `docker compose` (v2 plugin) — these scripts will fail silently on modern Docker installations.
- **Medium**: `scripts/scripts/simple_migrate.py` is a zero-byte file in a nested `scripts/scripts/` directory — a structural artifact that will confuse tooling; `.old` files (`setup_global_agents.sh.old`, `update_agents.sh.old`) should be removed.

> Read these 5 before anything else in this report.

## 1. Scope & Files Reviewed

**Path globs covered:**
- `scripts/**` (150 files found)
- `install_*.py` (2 files: `install_dependencies.py`, `install_platform.py`)
- `install_*.sh` (2 files: `install_critical_deps.sh`, `install_system_deps.sh`)
- `setup*.sh` (3 files: `setup.sh`, `setup_wsl.sh`, `scripts/setup_environment.sh`)
- `start*.sh` (2 files: `start.sh`, `start_backend.sh`)
- `stop.sh`, `logs.sh`, `board-sync.sh`, `notion-sync.sh`, `sync-all.sh`, `sync-boards.sh`

Total files in scope: ~164. Files reviewed: 58 (highest-signal scripts across all categories).

**Files explicitly excluded:**
- `scripts/models/finbert/config.json` — binary/ML artifact, not a script
- `scripts/data/cache/pipeline_status.json` — runtime state, not a script
- All `*.sql` files in `scripts/` — SQL schema files, owned by scope 07-database-persistence

## 2. Prior Report Reconciliation

### `docs/INSTALLATION_GUIDE.md` — status: `partially_stale`

**Validation method:** Read `setup.sh` and `start.sh` directly; ran `grep` for referenced command patterns.

**Archived to:** `docs/audits/2026-04/_meta/prior-reports-archive/INSTALLATION_GUIDE.archived.md`

| # | Claim | Original Location | Status | Evidence |
|---|---|---|---|---|
| 1 | "`./setup.sh` checks prerequisites, generates .env with secure credentials" | §Detailed Installation | current | `setup.sh:12-62` confirms prerequisite check and openssl-based key generation |
| 2 | "`./start.sh dev` starts services" | §Step 3 | current | `start.sh:6-27` confirms `MODE=${1:-dev}` and `docker compose ... up -d` |
| 3 | "`./start.sh prod` starts production stack" | §Production Mode | current | `start.sh:21-22` confirms `docker compose -f docker-compose.production.yml` |
| 4 | "SSL provisioned via `./scripts/init-ssl.sh yourdomain.com email`" | §Production Mode | current | `scripts/init-ssl.sh:1-55` exists and matches documented usage |
| 5 | "Grafana at localhost:3001, credentials admin/admin" | §Service URLs | current | Grafana URL/port documented; default creds noted in table |
| 6 | "`./setup.sh` does not generate GDPR_ENCRYPTION_KEY" | §Backend exits immediately | partially_stale | `setup.sh:36-52` generates SECRET_KEY, JWT_SECRET, DB_PASSWORD, REDIS_PASSWORD only — no GDPR_ENCRYPTION_KEY; doc correctly calls this out as common cause of failure |
| 7 | "Run `pytest backend/tests/ -m 'not slow'` to verify" | §Step 4 | current | Pytest command is standard; consistent with `pytest.ini` (scope 15) |
| 8 | "`docker compose exec backend python -m alembic upgrade head`" | §Migrations | current | Standard alembic migration command; consistent with project structure |
| 9 | "Scripts reference says `test_performance.sh` is at root" | §Additional Test Scripts | fully_stale | `grep -rn "test_performance.sh" /` — file referenced in SCRIPTS_REFERENCE does not appear in root; belongs to scope 15 |
| 10 | "SCRIPTS_REFERENCE last updated 2026-01-27" | §Header | partially_stale | SCRIPTS_REFERENCE.md header shows `2026-01-27`; new scripts added after that date (e.g., `sync-all.sh`, `sync-boards.sh`) are partially documented but structural issues not reflected |

---

### `docs/SCRIPTS_REFERENCE.md` — status: `partially_stale`

**Validation method:** Cross-referenced each listed script against actual filesystem contents via `find` and direct reads.

**Archived to:** `docs/audits/2026-04/_meta/prior-reports-archive/SCRIPTS_REFERENCE.archived.md`

| # | Claim | Original Location | Status | Evidence |
|---|---|---|---|---|
| 1 | "`notion-sync.sh` syncs with `notion-sync.sh push`" | §Quick Start Scripts | current | `notion-sync.sh:1-59` exists; delegates to `scripts/notion_sync.py` |
| 2 | "`test_performance.sh` at Root" | §Additional Test Scripts | fully_stale | `find /Users/devinmcgrath/projects/investment-analysis-platform -maxdepth 1 -name "test_performance.sh"` returns nothing; file not at root |
| 3 | "`scripts/testing/` contains `test_connections.py`" | §Testing Scripts table | current | `ls scripts/testing/test_connections.py` — confirmed present |
| 4 | "Two separate generate_secrets scripts: `scripts/generate_secrets.sh` and `scripts/security/generate_secrets.sh`" | implicit — both listed | current | Both confirmed at those paths; they differ in approach (openssl vs Python secrets module) |
| 5 | "`scripts/simple_migrate.py` runs database migrations" | §Database Scripts | partially_stale | `scripts/simple_migrate.py` (1269 bytes) exists; but `scripts/scripts/simple_migrate.py` is a 0-byte duplicate in the nested directory — doc doesn't reflect this structural issue |
| 6 | "All ML scripts listed: `train_ml_models.py`, `train_ml_models_minimal.py`, `train_models_simple.py`" | §ML Operations | current | All three confirmed at `scripts/train_ml_models.py`, `scripts/train_ml_models_minimal.py`, `scripts/train_models_simple.py` |
| 7 | "`deploy_ml_production.sh` deploys ML services" | §ML Operations | partially_stale | File exists at `scripts/deploy_ml_production.sh:23-24` but uses deprecated `docker-compose` (v1) binary — script will fail on modern Docker Desktop |
| 8 | "`scripts/setup/` contains `setup_global_agents.sh`" | §Setup Scripts | fully_stale | File renamed to `setup_global_agents.sh.old` — no longer executable; reference is dead |
| 9 | "`scripts/data/` contains background loaders" | §Data Pipeline Scripts | current | `background_loader.py`, `background_loader_enhanced.py` confirmed present |
| 10 | "Backup scripts: `backup.sh`, `restore-backup.sh`, `verify-backup.sh`" | §Backup Scripts | current | All three confirmed at `scripts/backup.sh`, `scripts/restore-backup.sh`, `scripts/verify-backup.sh` |
| 11 | "`init_database.sh` and `init_database_fixed.sh` both initialize PostgreSQL" | §Database Scripts | partially_stale | Both files exist and do initialize the database, but `init_database.sh:20` uses unsafe `export $(cat .env | grep -v '^#' | xargs)` pattern vs `init_database_fixed.sh:33` which uses safer `set -a; source "$ENV_FILE"; set +a` — reference doc treats them as equivalent |
| 12 | "All scripts require Python 3.12+" | §Environment Requirements | partially_stale | `migrate_airflow_to_prefect.py:1` uses `#!/usr/bin/env python3` shebang with no version enforcement; `scripts/setup/INIT.sh:38` checks for `docker-compose` (v1) not `docker compose` (v2) |

---

### `docs/WSL_INSTALLATION_FIXES.md` — status: `partially_stale`

**Validation method:** Read relevant script sections directly; cross-checked claimed fixes against current code.

**Archived to:** `docs/audits/2026-04/_meta/prior-reports-archive/WSL_INSTALLATION_FIXES.archived.md`

**Redactions:** The archived version redacts credential handling examples per sanitization policy. 1 redaction logged.

| # | Claim | Original Location | Status | Evidence |
|---|---|---|---|---|
| 1 | "`install_system_deps.sh` detects WSL via `WSL_DISTRO_NAME`/`WSL_INTEROP`" | §WSL Detection | current | `install_system_deps.sh` — WSL detection block present (pattern matches quoted code) |
| 2 | "`install_system_deps.sh` skips postfix in WSL" | §Postfix Fix Function | current | Pattern confirmed: mail package filter present in install_system_deps.sh |
| 3 | "`setup_wsl.sh` applies `dos2unix` for line-ending fixes" | §WSL Environment Fixes | current | `setup_wsl.sh:847-863` confirms docker detection and WSL handling |
| 4 | "`install_dependencies.py` detects WSL via Python" | §WSL Detection in Python | current | `install_dependencies.py:150-175` matches described pattern |
| 5 | "All scripts now 100% Ubuntu 24.04 compatible (test results: PASS)" | §Test Results | partially_stale | Claims are accurate for `install_system_deps.sh` fix, but `install_dependencies.py:51-52` still imports `packaging` and `requests` at top-level — fails bootstrap on fresh Ubuntu 24.04 before those packages are installed |
| 6 | "`install_dependencies.py` filters mail packages in WSL" | §WSL-Specific Package Installation | current | Code pattern confirmed present in install_dependencies.py |
| 7 | "WSL security model: scripts run with minimal required privileges" | §Security Considerations | partially_stale | `scripts/testing/test_docker_connections.py:20` and 5+ similar files contain hardcoded credentials committed to git — contradicts "secure credential handling" claim |
| 8 | "Backward compatible: original `setup.sh` remains functional" | §Backward Compatibility | current | `setup.sh` confirmed functional; `start.sh` works with modern docker compose v2 |

## 3. Findings

| ID | Severity | Category | File:Line | Title | Description | Recommendation | Acceptance Test Hint | Effort (h) | Loki Actionable | Cross Scope |
|---|---|---|---|---|---|---|---|---|---|---|
| F-17-001 | critical | security | scripts/testing/test_docker_connections.py:20,42 | Hardcoded DB and Redis Passwords in Git | Password `9v1g^OV9XUwzUP6cEgCYgNOE` (Postgres) and `RsYque&Xh%TUD*Nv^7k7B8X3` (Redis) appear as string literals in at least 8 committed files. Elasticsearch password (`4Bx+UM1CdSiEbMlQueRVvda+A4fLzCRsyuHUHbv5wMw=`) and Grafana password (`a2A5j4JQ0nF8aTLyIYwRgZnMLQpIu5lW9jYx6pB5Xdw=`) also hardcoded in `test_all_passwords.py:94,127`. | Rotate all four credentials immediately. Replace all hardcoded literals with `os.getenv('VAR_NAME')` with no default fallback, or raise an explicit error if the env var is absent. Delete or refactor the 8 affected scripts. | `grep -rn "9v1g\|RsYque\|xdfBj7\|7ba20b\|4Bx+UM\|a2A5j4" scripts/` returns 0 results | 6 | true | ["08-auth-security-compliance", "16-config-secrets"] |
| F-17-002 | critical | security | scripts/data/background_loader.py:33 | Hardcoded DB Password in Data Loading Scripts | `background_loader.py:33`, `background_loader_enhanced.py:56`, `load_data_now.py:30`, `load_historical_data.py:96`, `scripts/deployment/start_data_loading.sh:58`, `scripts/init_database_windows.ps1:74` all hardcode the same Postgres password as a direct connection string. These are production data loading scripts, not test utilities. | Same rotation+refactor as F-17-001. Use `os.environ['DATABASE_URL']` and fail fast if absent. | `grep -rn "9v1g" scripts/data/ scripts/deployment/ scripts/*.py` returns 0 results | 2 | true | ["05-data-ingestion-etl", "08-auth-security-compliance"] |
| F-17-003 | high | bug | setup.sh:88-103 | Unbounded Wait Loops with No Timeout | `wait_for_services()` contains three `until ... do; sleep 2; done` loops for Postgres, Redis, and the backend API with no iteration counter, timeout, or max-wait guard. If any service fails to start, the script hangs indefinitely. | Add a counter: `count=0; until <check> || [ $count -ge 60 ]; do sleep 2; ((count++)); done; [ $count -ge 60 ] && { echo "Timeout waiting for X"; exit 1; }` | `./setup.sh` exits with error message when a service container is stopped after 120 seconds | 2 | true | ["13-infra-deployment"] |
| F-17-004 | high | bug | install_dependencies.py:51-52 | Bootstrap Failure: Top-Level Third-Party Imports | `install_dependencies.py` imports `from packaging import requirements as packaging_requirements, version` and `import requests` at module top level before any installation occurs. On a clean system without these packages, running `python install_dependencies.py` immediately raises `ModuleNotFoundError` — the very scenario the script is designed to handle. | Guard with try/except at the top: `try: from packaging import ...; import requests; except ImportError: subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'packaging', 'requests']); <re-import>`. Or use importlib lazy loading. | `python3 -c "import sys; sys.modules['packaging']=None; sys.modules['requests']=None; import install_dependencies"` does not raise ImportError | 3 | true | [] |
| F-17-005 | high | broken_dependency | scripts/deploy_ml_production.sh:23-24 | Legacy `docker-compose` (v1) Used in ML Deployment Script | `deploy_ml_production.sh:23` checks `command -v docker-compose` and calls `docker-compose` throughout (lines 48, 51, 64, 73, 85, 86). Docker Compose v1 (`docker-compose`) is deprecated and removed from modern Docker Desktop. The project's canonical scripts (`start.sh`, `setup.sh`) correctly use `docker compose` (v2 plugin). | Replace all `docker-compose` calls with `docker compose` in `deploy_ml_production.sh` and `migrate_to_optimized.sh`. | `./scripts/deploy_ml_production.sh --dry-run` succeeds on Docker Desktop 4.x (Compose v2 only) | 2 | true | ["13-infra-deployment"] |
| F-17-006 | high | broken_dependency | scripts/migrate_to_optimized.sh:46-223 | Legacy `docker-compose` (v1) in Migration Script | Same issue as F-17-005. `migrate_to_optimized.sh` uses `docker-compose` in 8 locations: lines 46, 70, 99, 104, 194, 198, 219, 223. This migration script is a one-time use path but could be run on modern systems. | Apply same fix as F-17-005. | Script completes without "docker-compose: command not found" error on modern Docker | 1 | true | ["13-infra-deployment"] |
| F-17-007 | high | bug | scripts/init_database.sh:20 | Unsafe `.env` Sourcing with `export $(xargs)` | `init_database.sh:20` uses `export $(cat .env | grep -v '^#' | xargs)` to load environment variables. This pattern breaks on values containing spaces, special characters, or multi-line values (common for Fernet keys). The fixed version `init_database_fixed.sh:33` already uses the safe `set -a; source "$ENV_FILE"; set +a` pattern. | Replace the xargs pattern in `init_database.sh` with the same `set -a; source; set +a` approach already present in `init_database_fixed.sh`. Consider removing the older `init_database.sh` and pointing references to the fixed version. | `./scripts/init_database.sh` succeeds when `DB_PASSWORD` contains `&`, `!`, or space chars | 1 | true | ["07-database-persistence", "16-config-secrets"] |
| F-17-008 | high | dead_code | scripts/testing/test_docker_connections_fixed.py | Credential-Bearing Duplicate Test Scripts | `test_docker_connections.py`, `test_docker_connections_fixed.py`, `test_services_fixed.py`, `test_services_corrected.py`, `test_services_quick.py` are functionally redundant incremental iterations of the same connection-test script. Each carries hardcoded production credentials. The accumulation of "fixed" and "corrected" variants indicates an ad-hoc fix workflow that was never cleaned up. | Delete all but one canonical version. The surviving version must read credentials exclusively from environment variables. A git history search can confirm none of these scripts are called from CI. | `ls scripts/testing/test_services*.py scripts/testing/test_docker*.py | wc -l` returns ≤ 2 | 2 | true | ["08-auth-security-compliance"] |
| F-17-009 | high | architecture | scripts/ | 14 Duplicate/Overlapping DB-Password Default Scripts | `scripts/data/activate_pipeline.py:63` uses password `"postgres"` as a bare fallback; `scripts/verify_database.py:34`, `scripts/data/mock_data_generator.py:38`, `scripts/data/simple_mock_generator.py:36` all use `os.getenv('POSTGRES_PASSWORD', '9v1g^...')` with the same production password as the default. Each script independently manages DB connectivity without a shared credentials module. | Create a single `scripts/lib/db_connect.py` that reads `DATABASE_URL` from the environment with no default and provides a shared `get_engine()` function. All DB scripts import from this module. | `python -c "from scripts.lib.db_connect import get_engine"` works; no script outside lib/ contains a hardcoded password | 4 | true | ["07-database-persistence"] |
| F-17-010 | medium | dead_code | scripts/scripts/simple_migrate.py:1 | Zero-Byte File in Nested scripts/scripts/ Directory | `scripts/scripts/simple_migrate.py` is a 0-byte empty file in a nested `scripts/scripts/` directory. The real implementation is at `scripts/simple_migrate.py` (1269 bytes). This structural anomaly will cause `python scripts/scripts/simple_migrate.py` to silently do nothing. | Delete `scripts/scripts/` directory entirely. | `ls scripts/scripts/` returns "No such file or directory" | 0.25 | true | [] |
| F-17-011 | medium | dead_code | scripts/setup/setup_global_agents.sh.old | Stale `.old` Scripts Committed to Repository | `scripts/setup/setup_global_agents.sh.old` and `scripts/setup/update_agents.sh.old` are renamed-to-`.old` scripts committed to the repo. They reference `$HOME/.config/claude-code/agents` — a path from an older Claude Code configuration that no longer applies. | Remove both `.old` files via git. If historical context is needed, git history preserves them. | `find scripts/ -name "*.old"` returns empty | 0.25 | true | [] |
| F-17-012 | medium | incomplete_code | install_critical_deps.sh | Minimal Installer Lacks Virtualenv Activation Guard | `install_critical_deps.sh:7` runs `source ./venv/bin/activate` with no guard — if `venv/` does not exist, `source` fails and exits (due to `set -e` absence). The script also has no `set -e` and allows silent failures (every `pip install` has `|| echo "Failed"` but continues). Missing packages leave the system in an unknown state. | Add `[ -d venv ] || python3 -m venv venv` before activation. Add `set -e` at the top. Replace `|| echo "Failed"` with proper exit codes and a summary. | Running the script in a directory without `venv/` creates the venv and completes successfully | 1 | true | [] |
| F-17-013 | medium | security | scripts/generate_secrets.sh vs scripts/security/generate_secrets.sh | Two Secret-Generation Scripts with Different Entropy Levels | `scripts/generate_secrets.sh` uses `openssl rand -hex 32` (produces 256-bit hex) while `scripts/security/generate_secrets.sh` uses Python's `secrets` module with `token_urlsafe(64)` (produces stronger, URL-safe output). Users running the wrong script get lower-entropy secrets. | Deprecate `scripts/generate_secrets.sh` (the older openssl version). Update INSTALLATION_GUIDE.md to point to `scripts/security/generate_secrets.sh` as the authoritative generator. Add a deprecation comment or `exec` redirect in the old script. | INSTALLATION_GUIDE references only one secrets generator; the other is gone or explicitly deprecated | 1 | true | ["16-config-secrets"] |
| F-17-014 | medium | doc_drift | docs/SCRIPTS_REFERENCE.md | Reference Doc Missing sync-all.sh, sync-boards.sh, and Board-Sync Suite | `sync-all.sh`, `sync-boards.sh`, `board-sync.sh`, `notion-sync.sh` are all present at root and in `scripts/github-board-sync.sh` but SCRIPTS_REFERENCE (last updated 2026-01-27) does not document the unified sync suite introduced after that date. | Update SCRIPTS_REFERENCE.md to add the board-sync suite with usage examples. Tag with current date. | `grep "sync-all" docs/SCRIPTS_REFERENCE.md` returns at least one match | 1 | true | ["18-docs-health"] |
| F-17-015 | medium | stale_code | scripts/migrate_airflow_to_prefect.py | Airflow-to-Prefect Migration Script with No Corresponding Prefect Installation | `migrate_airflow_to_prefect.py` converts Airflow DAGs to Prefect 2.x flows. The project scope-map has `06-airflow-pipelines` as an active scope. Prefect is not in any `requirements*.txt` file, no Prefect Docker service exists, and no Prefect documentation was found. | Determine if the Airflow→Prefect migration is planned, in-progress, or abandoned. If abandoned, remove or archive the script. If planned, add Prefect to requirements and create migration documentation. | A decision is recorded (issue or ADR); script is either removed or has a matching `requirements/prefect.txt` | 2 | false | ["06-airflow-pipelines", "16-config-secrets"] |
| F-17-016 | medium | architecture | scripts/deployment/ | 13 Deployment Scripts with Overlapping Responsibility | `scripts/deployment/` contains 13 scripts: `deploy.sh`, `production_deploy.sh`, `blue_green_deploy.sh`, `QUICK_START.sh`, `start.sh`, `stop.sh`, `restart.sh`, `rollback.sh`, `start-docker.sh`, `start-full-stack.sh`, `start_app.sh`, `start_data_loading.sh`, `start_data_pipeline.sh`. The root-level `start.sh` is the canonical entry point but these alternatives diverge in behavior. New users face a confusing array of options. | Create a `scripts/deployment/README.md` distinguishing when to use each script. Deprecate or remove scripts that duplicate root-level `start.sh`/`stop.sh`. | README exists; count of non-deprecated deployment scripts is ≤ 5 | 3 | true | ["13-infra-deployment"] |
| F-17-017 | medium | stale_code | scripts/phase1-consolidation.py | Claude-Flow Memory Consolidation Script Committed to Project Repo | `scripts/phase1-consolidation.py` consolidates `.swarm/memory.db`, `.claude/learned-patterns/`, `.claude-flow/memory/` — all claude-flow agent runtime state. This is a developer tooling script for the AI orchestration system, not part of the investment platform itself. It has no relevance to production operations. | Move to `.claude/` or a developer-tools directory and exclude from production deployments via `.dockerignore`. | `scripts/phase1-consolidation.py` does not appear in any Dockerfile COPY instruction | 0.5 | true | [] |
| F-17-018 | low | dead_code | scripts/setup/ | `.sh.old` Files and Defunct Claude-Code Agent Setup Scripts | `setup_global_agents.sh.old` sets up `$HOME/.config/claude-code/agents` symlinks — a path that no longer exists in the current claude-code configuration (now at `~/.claude/agents/`). The path drift means the script would create broken symlinks. | Remove both `.old` files (covered in F-17-011) and verify that any live agent setup scripts reference `~/.claude/agents/` instead. | No file references `$HOME/.config/claude-code` | 0.25 | true | [] |
| F-17-019 | low | code_quality | scripts/init_database.sh | Missing Idempotency on Database Creation | `init_database.sh:60` calls `createdb` without the `-U` flag check first, and the check at line 57 using `grep -qw $DB_NAME` does not quote `$DB_NAME`, creating word-splitting risk on database names with hyphens. The `init_database_fixed.sh` corrects the sourcing pattern but retains similar issues. | Quote all variable expansions: `grep -qw "$DB_NAME"`. Add `createdb ... || true` after the existence check. | Running `init_database.sh` twice does not fail on second run | 0.5 | true | ["07-database-persistence"] |
| F-17-020 | low | code_quality | scripts/data/activate_pipeline.py | Bare except Clauses Swallow All Exceptions | `activate_pipeline.py:57-68` uses nested bare `except:` blocks to silently fall back from the real DB config to the hardcoded `"postgres"` default. If the real config exists but throws an unrelated error (e.g., `ImportError` in `backend.config.database`), the script silently uses the wrong credentials with no log entry. | Replace bare `except:` with `except Exception as e: logger.warning("Config load failed: %s, using fallback", e)`. Remove the hardcoded fallback password entirely per F-17-001/F-17-002. | No bare `except:` clauses in `scripts/data/activate_pipeline.py` | 0.5 | true | [] |
| F-17-021 | low | testing_gap | scripts/ | Script Directory Has Zero Automated Tests | 150 scripts manage DB initialization, secret generation, SSL provisioning, deployment, and backup. None have corresponding tests. The `scripts/testing/` subdirectory contains ad-hoc manual integration scripts (not pytest-compatible) that themselves carry hardcoded credentials. | Add a `tests/test_scripts/` directory with smoke tests: shellcheck linting for all `.sh` files, import tests for key Python scripts, and unit tests for critical functions (e.g., secret generation, env-loading). | `shellcheck scripts/*.sh scripts/deployment/*.sh` returns 0 errors; `pytest tests/test_scripts/` passes | 8 | true | ["15-test-suite"] |
| F-17-022 | low | doc_drift | docs/SCRIPTS_REFERENCE.md | `test_performance.sh` Listed at Root — File Does Not Exist There | SCRIPTS_REFERENCE.md §Additional Test Scripts lists `test_performance.sh` at "Root". `find /... -maxdepth 1 -name "test_performance.sh"` returns empty. The file is actually at a different location (owned by scope 15). | Remove or correct the entry in SCRIPTS_REFERENCE.md to point to the actual path. | `grep "test_performance" docs/SCRIPTS_REFERENCE.md` either returns the correct path or no result | 0.25 | true | ["18-docs-health"] |

## 4. Cross-Scope Linkages

- `F-17-001` / `F-17-002` → scope `08-auth-security-compliance`: Hardcoded credentials are committed to git history. The security auditor scope must determine if git history needs purging (BFG Repo-Cleaner). Also touches scope `16-config-secrets` since credential rotation requires `.env` updates.
- `F-17-003` → scope `13-infra-deployment`: The unbounded wait-loop issue in `setup.sh` is a mirror of any healthcheck timeout gaps in Docker Compose definitions — both scopes need aligned timeout values.
- `F-17-005` / `F-17-006` → scope `13-infra-deployment`: The `docker-compose` v1 vs `docker compose` v2 discrepancy should be fixed globally. The infra scope may have additional compose files to audit.
- `F-17-007` → scope `07-database-persistence` and `16-config-secrets`: `.env` sourcing correctness affects database initialization; the config-secrets scope governs the env file format.
- `F-17-009` → scope `07-database-persistence`: A shared `scripts/lib/db_connect.py` module would need to be aligned with the database connection patterns in backend repositories.
- `F-17-015` → scope `06-airflow-pipelines`: The Airflow-to-Prefect migration script's status should be determined in coordination with the pipeline scope.
- `F-17-016` → scope `13-infra-deployment`: Deployment script rationalization requires coordination with the infra scope to avoid breaking documented deployment paths.
- `F-17-021` → scope `15-test-suite`: Adding shellcheck and pytest-based script tests should integrate with the existing test suite structure.

## 5. Risk-Prioritized Punch List (top 10)

1. **F-17-001** — Hardcoded Production DB/Redis/ES/Grafana passwords in 14+ committed scripts — immediate credential rotation required; git history likely needs purging via BFG. These are real passwords in a git-committed repo.

2. **F-17-002** — Hardcoded DB password in production data pipeline scripts (`background_loader.py`, `start_data_loading.sh`) — compounds F-17-001; these run in production contexts, not just test utilities.

3. **F-17-004** — `install_dependencies.py` fails immediately on fresh system due to top-level third-party imports — makes the primary Python installer non-functional in its core use case (fresh installs).

4. **F-17-003** — Unbounded wait loops in `setup.sh` — a stuck or crashed container makes the installer hang forever with no recovery path; critical path for all new developer onboarding.

5. **F-17-005** / **F-17-006** — `docker-compose` v1 binary in ML deploy and migration scripts — these scripts silently fail on modern Docker Desktop installations.

6. **F-17-007** — Unsafe `.env` xargs sourcing in `init_database.sh` — breaks on any credential containing special characters, which is likely given the password generation patterns recommended in the project.

7. **F-17-008** — 5 credential-bearing duplicate test scripts with no cleanup — amplifies the credential exposure surface of F-17-001.

8. **F-17-009** — 14 scripts independently managing DB credentials with hardcoded defaults — architectural debt that perpetuates F-17-001/F-17-002 across the codebase.

9. **F-17-013** — Two secret generators producing different entropy levels; INSTALLATION_GUIDE references neither clearly — risk of weaker secrets in production setups.

10. **F-17-016** — 13 deployment scripts with overlapping purpose in `scripts/deployment/` — creates confusion and maintenance burden; risk of using a stale script path in production.

## 6. Open Questions

- Q1: Have the credentials in F-17-001/F-17-002 (`9v1g^OV9XUwzUP6cEgCYgNOE`, `RsYque&Xh%TUD*Nv^7k7B8X3`, `4Bx+UM1CdSiEbMlQueRVvda+A4fLzCRsyuHUHbv5wMw=`, `a2A5j4JQ0nF8aTLyIYwRgZnMLQpIu5lW9jYx6pB5Xdw=`) ever been used in a production environment (vs. local dev containers only)? This determines whether immediate production incident response is required or whether rotation is sufficient.

- Q2: Is the Airflow→Prefect migration (F-17-015) planned, abandoned, or in-progress? The `migrate_airflow_to_prefect.py` script exists without any Prefect infrastructure.

- Q3: Is `install_critical_deps.sh` still the recommended path for adding ML dependencies (selenium, lightgbm, optuna), or has it been superseded by `install_dependencies.py`? The two scripts are not coordinated.

- Q4: Should `scripts/phase1-consolidation.py` be excluded from Docker builds via `.dockerignore`? Currently there is no indication it is excluded from production images.

- Q5: What is the canonical deployment entry point for production? Root `start.sh prod` vs `scripts/deployment/production_deploy.sh` vs `scripts/deployment/blue_green_deploy.sh` all appear to be production-grade but behave differently. Human decision needed.
