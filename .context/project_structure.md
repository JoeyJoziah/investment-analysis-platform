# Project Structure

**Last Updated**: 2026-02-26
**Previous Analysis**: 2026-02-25

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
│   ├── etl/                     # 13 files - ETL pipeline (dead extractors removed)
│   ├── middleware/              # 5 files - priority-based stack
│   ├── migrations/              # 15 files - Alembic migrations
│   ├── ml/                      # 36 files - ML models, training, pipeline
│   │   ├── pipeline/            # 9 files
│   │   ├── training/            # 5 files - LSTM, XGBoost, Prophet
│   │   └── models/              # 1 file - voting classifier
│   ├── models/                  # 9 files - unified ORM Base declaration
│   ├── monitoring/              # 16 files - health, alerts, metrics
│   ├── repositories/            # 10 files - async CRUD pattern (excellent)
│   ├── scanner/                 # 1 file - daily scanner
│   ├── security/                # 22 files - comprehensive security
│   ├── services/                # 11 files - service layer (NEW expanded)
│   │   ├── stocks_service.py
│   │   ├── portfolio_service.py
│   │   ├── recommendation_service.py
│   │   ├── analysis_service.py
│   │   ├── admin_service.py
│   │   ├── gdpr_service.py
│   │   ├── watchlist_service.py
│   │   ├── agents_service.py
│   │   ├── trading_service.py
│   │   ├── websocket_service.py
│   │   └── realtime_price_service.py
│   ├── streaming/               # 2 files - Kafka client
│   ├── tasks/                   # 11 files - Celery tasks
│   ├── tests/                   # 99+ test files, ~130K lines, 3,569+ functions
│   │   ├── integration/         # 16 files
│   │   ├── security/            # 5 files (263 tests)
│   │   ├── middleware/          # 4 files (102 tests)
│   │   ├── unit/                # 29 files (NEW - service/utils unit tests)
│   │   └── fixtures/            # 4 fixture files
│   ├── TradingAgents/           # 39 files - embedded LangGraph trading system
│   └── utils/                   # 55 files - reduced from 87
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

| Metric | Feb 25 | Feb 26 | Change |
|--------|--------|--------|--------|
| Python source files (non-test) | 365 | ~330 | -35 (dead code removed) |
| Test files | 71 | 99+ | +28 unit test files |
| Test functions | 1,723 | 3,569+ | +107% (comprehensive coverage) |
| Test lines of code | 75,012 | ~130,000+ | NEW metric |
| Test results | 1543 pass, 8 skip, 5 xfail, 0 fail | 3569+ pass, 8 skip | +131% |
| API routers | 17 active | 16 active | -1 (legacy deleted) |
| API endpoints | ~130 | ~128 | Stable |
| CI/CD workflows | 28 | 28 | Stable |
| Docker services defined | 17 | 17 | Stable |
| Security modules | 22 | 22 | Stable |
| Utils files | 87 | 55 | -32 (cleaned up) |
| Service files | 0 | 11 | NEW (expanded) |
| Oversized routers (>800 lines) | 9 | 0 | ALL resolved |
| Dead/duplicate files removed | 0 | ~35 | Loki remediation |

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

## Changes Since Last Analysis (2026-02-25)

### Loki Mode Remediation (Feb 26)
**Status**: COMPLETE - All structural issues resolved

Deletions:
- `backend/backend/` (triple-nested directory) - DELETED
- `docker-compose.prod.yml` - DELETED (consolidated)
- 4 dead ETL extractors - DELETED
- Legacy stocks router - DELETED
- ~30+ other dead/duplicate files

Expansions:
- `backend/services/` - Expanded from 6 to 11 files with clear domain separation
- `backend/tests/unit/` - NEW 29 test files providing 80%+ unit coverage
- Test suite growth: +107% functions, 1543 → 3569+ passing

Consolidations:
- `backend/models/` - Unified to single `Base` declaration (tables.py, unified_models.py, schemas.py)
- `backend/utils/` - Reduced from 87 to 55 files (removed cache variants, db clutter)
- `backend/etl/` - Reduced from 18 to 13 files (removed dead extractors)

Code Quality:
- 9 oversized routers (>800 lines) - ALL refactored to <500 lines
- Service layer architecture - CLARIFIED with dedicated service classes
- Test infrastructure - IMPROVED with comprehensive fixtures and async support

### Historical Concerns (RESOLVED)
1. ✅ `backend/backend/backend/tests/` - triple-nested directory copy - DELETED
2. ✅ `backend/utils/` - 87 files sprawl - REDUCED to 55 focused files
3. ✅ `backend/etl/` - 6 data extractor variants - REDUCED to 1 canonical extractor
4. ✅ `backend/models/` - 3 competing `Base = declarative_base()` - UNIFIED to 1
5. ✅ `docker-compose.production.yml` - canonical production stack - CONFIRMED
6. ⚠️ `infrastructure/kubernetes/` - referenced by CI, does not exist - STILL MISSING
7. ⚠️ Frontend: 4 charting libraries - STILL PRESENT (Recharts, Plotly, Chart.js, Lightweight Charts)
