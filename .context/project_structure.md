# Project Structure

**Last Updated**: 2026-02-25
**Previous Analysis**: 2026-01-25

## Directory Tree Overview

```
investment-analysis-platform/
├── .context/                    # Analysis reports (this directory)
├── .github/
│   └── workflows/               # 28 CI/CD workflows (+14 since Jan)
├── agents/                      # 5 YAML agent definitions
├── backend/
│   ├── analytics/               # 25 files - analysis engines
│   │   ├── agents/              # 4 files - cache-aware agents, hybrid engine
│   │   ├── fundamental/         # 2 files - valuation (DCF)
│   │   ├── portfolio/           # 2 files - Black-Litterman, MPT
│   │   ├── risk/                # 2 files - VaR, risk attribution
│   │   └── statistical/         # 1 file - cointegration
│   ├── api/
│   │   ├── main.py              # FastAPI app with 12-layer middleware stack
│   │   ├── routers/             # 17 active routers (~130 endpoints)
│   │   ├── security_integration.py
│   │   └── versioning.py        # V1 deprecation middleware
│   ├── auth/                    # 3 files - OAuth2, password validation
│   ├── compliance/              # 3 files - GDPR, SEC
│   ├── config/                  # 4 files - settings, database, monitoring
│   ├── data_ingestion/          # 9 files - API clients per provider
│   ├── domain/contracts/        # 7 files - domain contracts (unused in prod)
│   ├── etl/                     # 18 files - ETL pipeline (6 extractor variants!)
│   ├── middleware/              # 5 files - priority-based stack
│   ├── migrations/              # 15 files - Alembic migrations
│   ├── ml/                      # 36 files - ML models, training, pipeline
│   │   ├── pipeline/            # 9 files
│   │   ├── training/            # 5 files - LSTM, XGBoost, Prophet
│   │   └── models/              # 1 file - voting classifier
│   ├── models/                  # 10 files - 3 competing ORM Base declarations
│   ├── monitoring/              # 16 files - health, alerts, metrics
│   ├── repositories/            # 10 files - async CRUD pattern (excellent)
│   ├── scanner/                 # 1 file - daily scanner
│   ├── security/                # 22 files - comprehensive security
│   ├── services/                # 6 files - thin service layer
│   ├── streaming/               # 2 files - Kafka client
│   ├── tasks/                   # 11 files - Celery tasks
│   ├── tests/                   # 71 test files, 75K lines, 1,723 functions
│   │   ├── integration/         # 16 files
│   │   ├── security/            # 5 files (263 tests)
│   │   ├── middleware/          # 4 files (102 tests)
│   │   ├── unit/                # EMPTY (only __init__.py)
│   │   └── fixtures/            # 4 fixture files
│   ├── TradingAgents/           # 39 files - embedded LangGraph trading system
│   └── utils/                   # 87 files - CRITICAL sprawl
├── frontend/
│   └── web/                     # React 18 + TypeScript + Material-UI
│       └── src/
│           ├── components/      # 30+ components
│           ├── pages/           # 12 pages
│           ├── store/slices/    # 6 Redux slices
│           ├── services/        # API + WebSocket services
│           ├── hooks/           # Performance + utility hooks
│           └── utils/           # Accessibility utilities
├── infrastructure/
│   ├── docker/                  # 4 backend + 3 frontend Dockerfiles
│   ├── monitoring/              # Prometheus, Grafana configs
│   └── nginx/                   # Reverse proxy + security headers
├── scripts/                     # 50+ shell scripts
│   └── deployment/              # Blue-green deploy, rollback
├── docker-compose.yml           # Base (17 services)
├── docker-compose.production.yml # Production stack (canonical)
├── docker-compose.dev.yml       # Development overrides
└── docker-compose.test.yml      # Test overrides
```

## Key Metrics

| Metric | Jan 25 | Feb 25 | Change |
|--------|--------|--------|--------|
| Python source files (non-test) | ~400 | 365 | Measured accurately |
| Test files | 20 | 71 | +51 (counted properly) |
| Test functions | N/A | 1,723 | NEW metric |
| Test lines of code | N/A | 75,012 | NEW metric |
| Test results | N/A | 1543 pass, 8 skip, 5 xfail, 0 fail | Stable |
| API routers | 18 | 17 active + 1 legacy | Clarified |
| API endpoints | N/A | ~130 | NEW metric |
| CI/CD workflows | 14 | 28 | +14 |
| Docker services defined | 12 | 17 | +5 (exporters) |
| Security modules | 16 | 22 | +6 (counted properly) |
| Utils files | N/A | 87 | NEW (critical) |
| Oversized files (>800 lines) | N/A | 19 | NEW metric |
| Dead/duplicate files | N/A | ~30+ | NEW metric |

## Frontend Stack (from analysis)

| Technology | Version | Notes |
|------------|---------|-------|
| React | 18.2.0 | 12 pages, 30+ components |
| TypeScript | 5.3.3 | Strict mode enabled |
| Vite | 7.3.1 | 18 manual vendor chunks |
| Redux Toolkit | - | 6 domain slices |
| Material-UI | 5.14 | Full theming + design tokens |
| Recharts | - | Primary charting |
| Plotly, Chart.js, Lightweight Charts | - | 3 additional chart libs (overkill) |
| Playwright | 1.40 | E2E config (2 test files only) |
| Vitest | 4.0.16 | Unit testing (4 test files only) |

**Frontend Rating**: Strong MVP (8/10) - features complete, testing sparse

## Changes Since Last Analysis (2026-01-25)

### Commits (25+ since Jan 25)
- All CI/CD fixes (pipeline stabilization phase)
- Docker v1 -> v2 migration across workflows
- TA-Lib arm64 cross-compile fixes
- GDPR_ENCRYPTION_KEY None-safety
- Python 3.9 dropped from CI matrix
- Non-blocking frontend lint, coverage, security scans

### Structural Concerns Identified
1. `backend/backend/backend/tests/` - triple-nested directory copy
2. `backend/utils/` - 87 files (24 cache, 6 database variants)
3. `backend/etl/` - 6 data extractor variants (4-5 dead)
4. `backend/models/` - 3 competing `Base = declarative_base()`
5. `docker-compose.production.yml` - canonical production stack (consolidated from duplicate)
6. `infrastructure/kubernetes/` - referenced by CI, does not exist
7. Frontend: 4 charting libraries (should be 1-2)
