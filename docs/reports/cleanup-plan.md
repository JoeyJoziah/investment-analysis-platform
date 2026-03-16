# Requirements and Documentation Cleanup Plan

**Date:** 2026-01-27
**Status:** PENDING USER APPROVAL
**Risk Level:** MEDIUM - Review migration commands before execution

---

## Executive Summary

This cleanup plan consolidates **10 requirements files** into **3 logical files** and organizes **38 root markdown files** into appropriate documentation directories. The plan also addresses duplicate entries in the existing `requirements/` directory structure.

---

## Part 1: Python Requirements Consolidation

### Current State Analysis

**Root Directory Requirements Files (8 files):**
| File | Lines | Purpose | Action |
|------|-------|---------|--------|
| `requirements.txt` | 182 | Main comprehensive deps | CONSOLIDATE -> production |
| `requirements-core.txt` | 17 | Minimal core deps | DELETE (superseded) |
| `requirements-minimal.txt` | 24 | Testing minimal | DELETE (superseded) |
| `requirements-clean.txt` | 141 | Cleaned version | DELETE (superseded) |
| `requirements-old.txt` | 149 | Old backup | DELETE |
| `requirements-py313.txt` | 236 | Python 3.13 version | ARCHIVE |
| `requirements.production.txt` | 203 | Production deps | CONSOLIDATE |
| `requirements-airflow.txt` | 165 | Airflow deps | MOVE to config/infrastructure |

**Other Locations (3 files):**
| File | Purpose | Action |
|------|---------|--------|
| `backend/TradingAgents/requirements.txt` | TradingAgents subproject | KEEP (subproject deps) |
| `config/infrastructure/docker/airflow/requirements-airflow.txt` | Docker Airflow | KEEP (container-specific) |
| `requirements/` directory (10 files) | Modular requirements | CONSOLIDATE |

### Dependency Conflicts Identified

| Package | Conflict Versions | Resolution |
|---------|-------------------|------------|
| `fastapi` | 0.104.1, 0.111.0, 0.115.0, 0.115.5 | Use 0.115.0 (latest stable) |
| `pydantic` | 2.5.0, 2.8.2, 2.10.0 | Use 2.8.2 (compatible with fastapi) |
| `numpy` | 1.24.3, 1.26.2, 1.26.4, 2.1.3 | Use <2.0.0 (many deps incompatible with 2.x) |
| `pandas` | 2.1.3, 2.2.2, 2.2.3 | Use 2.0.0-3.0.0 range |
| `torch` | 2.1.1, 2.4.0, 2.5.1 | Use 2.4.0 (stable) |
| `redis` | 5.0.1, 5.0.7, 5.2.0 | Use 5.0.7 |
| `scikit-learn` | 1.3.2, 1.5.1, 1.5.2 | Use 1.5.1 |
| `pytest` | 7.4.3, 8.3.2, 8.3.3 | Use 8.3.2 |
| `transformers` | 4.35.2, 4.43.3, 4.46.2 | Use 4.43.3 |

### Duplicate Packages Found

The following packages appear in multiple files with identical or similar versions:
- `python-jose[cryptography]==3.3.0` - 6 occurrences
- `passlib[bcrypt]==1.7.4` - 6 occurrences
- `python-dotenv` - 7 occurrences
- `requests` - 6 occurrences
- `aiohttp` - 5 occurrences

### Consolidated Structure (Target)

```
investment-analysis-platform/
  requirements.txt           # Production dependencies (core + all features)
  requirements-dev.txt       # Development/testing only
  requirements-ml.txt        # ML-specific heavy dependencies
  requirements/
    README.md               # KEEP - documents structure
    DEPRECATED.md           # NEW - explains migration
  config/infrastructure/docker/airflow/
    requirements-airflow.txt  # KEEP - container-specific
  backend/TradingAgents/
    requirements.txt         # KEEP - subproject specific
```

---

## Part 2: Root Markdown Files Cleanup

### Current State: 38 Markdown Files in Root

**Files to KEEP in Root (4):**
- `README.md` - Project readme (essential)
- `CLAUDE.md` - Claude configuration (essential)
- `TODO.md` - Task tracking (essential)
- `CHANGELOG.md` - Not present, consider creating

**Files to DELETE (Outdated/Superseded) (3):**
| File | Reason |
|------|--------|
| `CLAUDE-old.md` | Old backup, superseded by CLAUDE.md |
| `REFACTORING_PLAN.md` | Outdated, only 1KB |
| `REFACTORING_SUMMARY.md` | Outdated refactoring notes |

**Files to MOVE to `docs/reports/` (12):**
| File | Size | Content Type |
|------|------|--------------|
| `ETL_ACTIVATION_SUCCESS.md` | 2.7KB | Success report |
| `MODERNIZATION_SUMMARY.md` | 5.2KB | Summary report |
| `STOCK_UNIVERSE_EXPANSION_SUCCESS.md` | 3.6KB | Success report |
| `PHASE_3.3_IMPLEMENTATION_SUMMARY.md` | 9KB | Phase summary |
| `PHASE_3_2_IMPLEMENTATION_SUMMARY.md` | 11.9KB | Phase summary |
| `PHASE_4.2_COMPLETION_SUMMARY.md` | 17.9KB | Phase summary |
| `PHASE_4.2_QUICK_START.md` | 10.7KB | Quick start |
| `PHASE_4_1_COMPLETION.md` | 11.2KB | Phase completion |
| `PRODUCTION_LAUNCH_COMPLETE.md` | 18.6KB | Launch report |
| `QUICK_REFERENCE.md` | 7.1KB | Reference guide |
| `QUICK_WINS.md` | 12.2KB | Quick wins report |
| `WEBSOCKET_IMPLEMENTATION.md` | 11.1KB | Implementation report |

**Files to MOVE to `docs/architecture/` (6):**
| File | Size | Content Type |
|------|------|--------------|
| `COMPREHENSIVE_CACHING_SYSTEM.md` | 11KB | Architecture doc |
| `DATA_COLLECTION_SOLUTION.md` | 9.5KB | Architecture doc |
| `MULTI_SOURCE_ETL_SOLUTION.md` | 9.3KB | Architecture doc |
| `UNLIMITED_DATA_EXTRACTION_SOLUTION.md` | 6.3KB | Architecture doc |
| `UNLIMITED_STOCK_EXTRACTION_SOLUTION.md` | 17.1KB | Architecture doc |
| `PERFORMANCE_OPTIMIZATIONS.md` | 5.5KB | Architecture doc |

**Files to MOVE to `docs/security/` (2):**
| File | Size | Content Type |
|------|------|--------------|
| `SECURITY_CREDENTIALS_AUDIT.md` | 19.5KB | Security audit |
| `SECURITY_IMPLEMENTATION_SUMMARY.md` | 11.5KB | Security summary |

**Files to MOVE to `docs/` (Already has subdirectory) (9):**
| File | Target Directory | Content Type |
|------|------------------|--------------|
| `INSTALLATION_GUIDE.md` | `docs/` | Setup guide |
| `ML_API_REFERENCE.md` | `docs/ml/` | ML API reference |
| `ML_OPERATIONS_GUIDE.md` | `docs/ml/` | ML operations |
| `ML_PIPELINE_DOCUMENTATION.md` | `docs/ml/` | ML pipeline docs |
| `ML_QUICKSTART.md` | `docs/ml/` | ML quick start |
| `PRODUCTION_DEPLOYMENT_GUIDE.md` | `docs/` | Deployment guide |
| `IMPLEMENTATION_STATUS.md` | `docs/` | Status tracking |
| `INTEGRATION_SUMMARY.md` | `docs/` | Integration docs |
| `WSL_INSTALLATION_FIXES.md` | `docs/` | Platform-specific |

**Files to MOVE to `docs/investigation/` (3):**
| File | Size | Content Type |
|------|------|--------------|
| `INFRASTRUCTURE_ANALYSIS.md` | 40KB | Investigation |
| `INFRASTRUCTURE_FIXES_CHECKLIST.md` | 17KB | Checklist |
| `INFRASTRUCTURE_SUMMARY.md` | 16.4KB | Summary |

---

## Part 3: Migration Commands

### STEP 1: Backup Current State

```bash
# Create backup branch
cd /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform
git checkout -b backup/pre-cleanup-$(date +%Y%m%d)
git add -A
git commit -m "Backup before requirements and docs cleanup"
git checkout -
```

### STEP 2: Requirements Cleanup

```bash
# Archive Python 3.13 specific requirements
mkdir -p /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/requirements/archive
mv /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/requirements-py313.txt \
   /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/requirements/archive/

# Delete superseded requirements files
rm /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/requirements-core.txt
rm /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/requirements-minimal.txt
rm /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/requirements-clean.txt
rm /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/requirements-old.txt
rm /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/requirements.production.txt

# Move duplicate Airflow requirements (root copy)
mv /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/requirements-airflow.txt \
   /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/requirements/archive/requirements-airflow-root-backup.txt
```

### STEP 3: Create Consolidated Requirements Files

**Note:** These files will be created with resolved dependencies (see new file contents below)

### STEP 4: Documentation Cleanup

```bash
# Ensure target directories exist
mkdir -p /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/docs/reports
mkdir -p /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/docs/architecture
mkdir -p /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/docs/security
mkdir -p /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/docs/ml
mkdir -p /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/docs/investigation

# Delete outdated files
rm /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/CLAUDE-old.md
rm /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/REFACTORING_PLAN.md
rm /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/REFACTORING_SUMMARY.md

# Move to docs/reports/
mv /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/ETL_ACTIVATION_SUCCESS.md \
   /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/docs/reports/
mv /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/MODERNIZATION_SUMMARY.md \
   /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/docs/reports/
mv /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/STOCK_UNIVERSE_EXPANSION_SUCCESS.md \
   /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/docs/reports/
mv /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/PHASE_3.3_IMPLEMENTATION_SUMMARY.md \
   /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/docs/reports/
mv /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/PHASE_3_2_IMPLEMENTATION_SUMMARY.md \
   /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/docs/reports/
mv /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/PHASE_4.2_COMPLETION_SUMMARY.md \
   /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/docs/reports/
mv /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/PHASE_4.2_QUICK_START.md \
   /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/docs/reports/
mv /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/PHASE_4_1_COMPLETION.md \
   /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/docs/reports/
mv /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/PRODUCTION_LAUNCH_COMPLETE.md \
   /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/docs/reports/
mv /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/QUICK_REFERENCE.md \
   /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/docs/reports/
mv /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/QUICK_WINS.md \
   /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/docs/reports/
mv /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/WEBSOCKET_IMPLEMENTATION.md \
   /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/docs/reports/

# Move to docs/architecture/
mv /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/COMPREHENSIVE_CACHING_SYSTEM.md \
   /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/docs/architecture/
mv /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/DATA_COLLECTION_SOLUTION.md \
   /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/docs/architecture/
mv /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/MULTI_SOURCE_ETL_SOLUTION.md \
   /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/docs/architecture/
mv /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/UNLIMITED_DATA_EXTRACTION_SOLUTION.md \
   /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/docs/architecture/
mv /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/UNLIMITED_STOCK_EXTRACTION_SOLUTION.md \
   /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/docs/architecture/
mv /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/PERFORMANCE_OPTIMIZATIONS.md \
   /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/docs/architecture/

# Move to docs/security/
mv /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/SECURITY_CREDENTIALS_AUDIT.md \
   /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/docs/security/
mv /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/SECURITY_IMPLEMENTATION_SUMMARY.md \
   /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/docs/security/

# Move to docs/ml/
mv /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/ML_API_REFERENCE.md \
   /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/docs/ml/
mv /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/ML_OPERATIONS_GUIDE.md \
   /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/docs/ml/
mv /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/ML_PIPELINE_DOCUMENTATION.md \
   /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/docs/ml/
mv /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/ML_QUICKSTART.md \
   /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/docs/ml/

# Move to docs/investigation/
mv /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/INFRASTRUCTURE_ANALYSIS.md \
   /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/docs/investigation/
mv /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/INFRASTRUCTURE_FIXES_CHECKLIST.md \
   /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/docs/investigation/
mv /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/INFRASTRUCTURE_SUMMARY.md \
   /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/docs/investigation/

# Move to docs/ root
mv /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/INSTALLATION_GUIDE.md \
   /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/docs/
mv /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/PRODUCTION_DEPLOYMENT_GUIDE.md \
   /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/docs/
mv /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/IMPLEMENTATION_STATUS.md \
   /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/docs/
mv /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/INTEGRATION_SUMMARY.md \
   /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/docs/
mv /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/WSL_INSTALLATION_FIXES.md \
   /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/docs/
```

### STEP 5: Commit Changes

```bash
git add -A
git commit -m "refactor: consolidate requirements and organize documentation

- Consolidate 10 root requirements files into 3 logical files
- Move 31 markdown files from root to appropriate docs/ subdirectories
- Delete 3 outdated/superseded documentation files
- Archive Python 3.13 specific requirements
- Resolve dependency version conflicts

See docs/reports/cleanup-plan.md for full details.

Co-Authored-By: claude-flow <ruv@ruv.net>"
```

---

## Part 4: New Consolidated Requirements Files

### requirements.txt (Production)

This file will contain all production dependencies with resolved versions.

Key sections:
- Core Framework (fastapi, uvicorn, pydantic)
- Database (sqlalchemy, asyncpg, psycopg2-binary, alembic)
- Caching & Messaging (redis, celery)
- Data Processing (pandas, numpy, scipy)
- Machine Learning (torch, scikit-learn, transformers, xgboost)
- Financial Data APIs (yfinance, alpha-vantage, finnhub, polygon)
- Technical Analysis (ta-lib, statsmodels)
- Web & API Clients (aiohttp, httpx, requests)
- Security (python-jose, passlib, PyJWT, cryptography)
- NLP (nltk, textblob, tokenizers)
- Monitoring (prometheus-client, opentelemetry, sentry-sdk)
- Deployment (gunicorn, docker, websockets)
- LLM Integration (langchain-anthropic, langchain-openai, langgraph)

### requirements-dev.txt (Development Only)

Development-only dependencies:
- Testing (pytest, pytest-asyncio, pytest-cov, pytest-mock, faker)
- Code Quality (black, isort, flake8, mypy)
- Security Scanning (bandit)
- Additional test utilities (testcontainers, memory-profiler)

### requirements-ml.txt (ML-Specific Heavy Dependencies)

For environments that need full ML capabilities:
- PyTorch ecosystem (torch, transformers, tokenizers)
- Hugging Face Hub (huggingface_hub, datasets)
- Model interpretability (shap, lime)
- Hyperparameter tuning (optuna)
- Visualization (plotly, matplotlib, seaborn)

---

## Part 5: Impact Summary

### Files to Delete
- 6 requirements files (superseded/duplicate)
- 3 markdown files (outdated)

### Files to Move
- 31 markdown files (to appropriate docs/ subdirectories)
- 2 requirements files (to archive)

### Files to Create
- 1 new consolidated requirements-dev.txt
- 1 new consolidated requirements-ml.txt
- Update existing requirements.txt

### Estimated Cleanup Results
- Root directory: From 38 markdown files to 4
- Requirements: From 10 root files to 3
- Resolved version conflicts: 9 packages
- Eliminated duplicate entries: ~30 package duplicates

---

## Approval Checklist

Before executing migration commands, please verify:

- [ ] Backup branch will be created
- [ ] No active development depends on files being moved
- [ ] CI/CD pipelines reference correct requirements file locations
- [ ] Docker builds reference correct requirements files
- [ ] Documentation links will be updated if needed

---

## Next Steps After Approval

1. Create backup branch
2. Execute requirements cleanup commands
3. Create new consolidated requirements files
4. Execute documentation move commands
5. Run tests to verify nothing is broken
6. Update any import statements or references
7. Commit changes
8. Update CI/CD if needed

---

**Awaiting user approval before executing any commands.**
