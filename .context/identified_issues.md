# Identified Issues

**Last Updated**: 2026-02-25

## Issue Summary

| Priority | Count | Status |
|----------|-------|--------|
| Critical Blockers | 3 | Unchanged from Jan |
| Code Quality (NEW) | 8 | Newly identified |
| CI/CD Issues (NEW) | 5 | Newly identified |
| Testing Gaps (NEW) | 6 | Newly identified |
| Medium Priority | 4 | Ongoing |
| Previously Resolved | 15 | Docker fixes + more |

---

## CRITICAL BLOCKERS (Unchanged)

### 1. GDPR Encryption Key Missing
- **Severity**: CRITICAL BLOCKER
- **Location**: `backend/utils/data_anonymization.py:19`
- **Error**: `AttributeError: 'NoneType' object has no attribute 'encode'`
- **Fix**: `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"` then add to `.env`

### 2. Database Empty - 0 Stocks
- **Severity**: CRITICAL BLOCKER
- **Impact**: Core functionality cannot operate
- **Fix**: Run `scripts/data/load_stock_universe.py`

### 3. Database User Role Missing
- **Severity**: HIGH
- **Fix**: `CREATE USER investment_user WITH PASSWORD '...'`

---

## CODE QUALITY ISSUES (NEW - from architecture analysis)

### 4. Utils Directory Sprawl (87 files)
- **Severity**: CRITICAL
- **Location**: `backend/utils/`
- **Problem**: 87 files including 24 cache-related files, 6 database variants
- **Impact**: Unmaintainable, impossible to understand dependency graph
- **Files**: `cache.py`, `cache_manager.py`, `advanced_cache.py`, `comprehensive_cache.py`, `cache_monitoring.py`, `cache_optimization.py`, `cache_warming.py`, `cache_warmer.py`, `cache_hit_optimization.py`, `bounded_cache.py`, `intelligent_cache_policies.py`, `predictive_cache_warming.py`, `production_cache_optimizer.py`, `api_cache_decorators.py`, `query_cache.py`, `database_query_cache.py`, `tier_based_caching.py`, `redis_optimization.py`, `redis_cluster_optimization.py`, `redis_resilience.py`, `distributed_cache_coordination.py`, `enhanced_cache_config.py`, `cache_efficiency_reports.py`, `cache_monitoring_dashboard.py`
- **Fix**: Extract into `backend/cache/`, `backend/database/`, `backend/errors/` packages

### 5. Three Competing ORM Base Declarations
- **Severity**: HIGH
- **Location**: `backend/models/tables.py`, `backend/models/unified_models.py`, `backend/models/consolidated_models.py`
- **Problem**: Each file declares `Base = declarative_base()` independently
- **Impact**: Schema integrity risk - only one Base can be the actual DB schema
- **Fix**: Choose `unified_models.py` as canonical, delete others, ensure single Base

### 6. Dead ETL Extractors (6 variants)
- **Severity**: HIGH
- **Location**: `backend/etl/`
- **Dead files**: `data_extractor_original_backup.py` (712 lines), `data_extractor_unlimited.py` (508), `simple_unlimited_extractor.py` (375), `unlimited_data_extractor.py` (609)
- **Impact**: ~2,200 lines of dead code causing confusion
- **Fix**: Delete dead files, keep only `data_extractor.py` and `multi_source_extractor.py`

### 7. Oversized Files (19 files >800 lines)
- **Severity**: MEDIUM
- **Worst offenders**: `recommendation_engine.py` (1,301), `routers/recommendations.py` (1,114), `routers/analysis.py` (1,113), `monitoring/health_checks.py` (1,103), `routers/gdpr.py` (1,075), `routers/portfolio.py` (1,039)
- **Fix**: Split into sub-modules, move business logic to services

### 8. Thin Service Layer
- **Severity**: MEDIUM
- **Problem**: Most business logic lives in 1000+ line routers instead of services
- **Impact**: Routers are untestable monoliths, violating single responsibility
- **Fix**: Move business logic from routers to `backend/services/`

### 9. Dual get_current_user Functions
- **Severity**: MEDIUM
- **Locations**: `backend/auth/oauth2.py` (returns User ORM) vs `backend/utils/auth.py` (returns dict)
- **Impact**: Tests must mock both, confusing for developers
- **Fix**: Consolidate into single function

### 10. Duplicate Analytics Engines
- **Severity**: LOW
- **Files**: `recommendation_engine.py` (1,301 lines) + `recommendation_engine_optimized.py` (882 lines)
- **Fix**: Merge into single optimized engine

### 11. Triple-Nested Test Directory
- **Severity**: LOW
- **Location**: `backend/backend/backend/tests/integration/test_phase3_integration.py`
- **Fix**: Delete the nested copy

---

## CI/CD ISSUES (NEW - from DevOps analysis)

### 12. Missing Kubernetes Manifests
- **Severity**: HIGH
- **Problem**: `staging-deploy.yml` and `production-deploy.yml` reference `infrastructure/kubernetes/` but no manifests exist
- **Impact**: Both deploy workflows will fail at `kubectl apply` step
- **Fix**: Create K8s manifests or switch to Docker-based deployment

### 13. Pipeline Instability
- **Severity**: HIGH
- **Evidence**: All 25 recent commits are `ci:` or `fix(ci):` prefixed
- **Root causes**: TA-Lib cross-compile flakiness, docker-compose v1/v2 divergence, missing test dependencies
- **Fix**: Pre-build TA-Lib in cached base image, add missing deps to requirements-dev.txt

### 14. Advisory-Only Quality Gates
- **Severity**: MEDIUM
- **Problem**: All security scans, coverage checks, and linting use `continue-on-error: true`
- **Impact**: Defects and vulnerabilities can merge without blocking
- **Fix**: Enable hard gates for at least CRITICAL security findings and minimum coverage threshold

### 15. Dockerfile Reference Inconsistency
- **Severity**: MEDIUM
- **Problem**: CI uses `./Dockerfile.backend` (root), compose uses `./infrastructure/docker/backend/Dockerfile`
- **Impact**: Different Dockerfiles built in CI vs local development
- **Fix**: Standardize to single Dockerfile path

### 16. Duplicate Production Compose Files
- **Severity**: LOW - RESOLVED
- **Files**: `docker-compose.prod.yml` (deleted) vs `docker-compose.production.yml` (canonical, kept)
- **Fix**: Deleted `docker-compose.prod.yml`, updated all references to `docker-compose.production.yml`

---

## TESTING GAPS (NEW - from QA analysis)

### 17. Actual Coverage ~35-40% (Target: 80%)
- **Severity**: CRITICAL
- **Details**: Previous 60% estimate was based on file presence, not measured coverage
- **1,723 tests exist** but coverage is concentrated in security, middleware, and a few routers
- **Fix**: Prioritize ETL, tasks, monitoring, ML coverage

### 18. Zero Coverage Areas
- **Severity**: HIGH
- **ETL** (~5%): `data_extractor.py`, `data_loader.py`, `data_transformer.py`, `data_validator.py` - all untested
- **Tasks** (~5%): All 11 Celery task files untested
- **TradingAgents** (0%): All 30+ files completely untested
- **Monitoring** (~20%): 10 of 15 monitoring modules untested

### 19. 8 Module-Level Test Skips
- **Severity**: MEDIUM
- **Skipped files**: `test_data_pipeline_integration.py` (celery), `test_integration_comprehensive.py` (testcontainers), `test_database_integration.py` (testcontainers), `test_security_compliance.py` (requests_mock), `test_performance_load.py` (memory_profiler), `test_performance_optimizations.py` (objgraph), `test_financial_model_validation.py` (sqlparse), `test_resilience_integration.py` (psycopg2)
- **Fix**: Add missing packages to requirements-dev.txt

### 20. No E2E Tests
- **Severity**: MEDIUM
- **Problem**: No Playwright/Selenium tests despite `.claude/rules/testing.md` requirement
- **Fix**: Add E2E tests for auth, portfolio, stock analysis flows

### 21. Empty unit/ Directory
- **Severity**: LOW
- **Problem**: `backend/tests/unit/` has only `__init__.py`, all unit tests in root `tests/`
- **Fix**: Organize tests by type

### 22. 5 xfail Tests Masking Real Bugs
- **Severity**: LOW
- **Location**: `backend/tests/integration/test_stocks_router.py`
- **Bugs**: `StockResponse.from_orm` lazy-loading failure, `price_repository.get_previous_price` not implemented
- **Fix**: Fix the underlying application bugs

---

## Previously Resolved (Jan 2026)

- [x] 10 Docker configuration issues fixed
- [x] Python version standardized to 3.11
- [x] Redis health check secured
- [x] Resource limits added to all services
- [x] Restart policies standardized
- [x] docker-compose v1 replaced with v2
