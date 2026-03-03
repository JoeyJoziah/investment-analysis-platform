# Project Structure

**Last Updated**: 2026-03-03 (Refreshed with deep audit)
**Previous Analysis**: 2026-02-26

## Directory Tree Overview

```
investment-analysis-platform/
├── .context/                    # Analysis reports (this directory)
├── .github/
│   └── workflows/               # 29 CI/CD workflows
├── agents/                      # 5 YAML agent definitions
├── backend/                     # 493 Python files
│   ├── analytics/               # 26 files - analysis engines
│   │   ├── agents/              # Cache-aware agents, hybrid engine
│   │   ├── fundamental/         # DCF valuation, ratio analysis
│   │   ├── portfolio/           # Black-Litterman, MPT
│   │   ├── risk/                # VaR, risk attribution
│   │   └── statistical/         # Cointegration
│   ├── api/
│   │   ├── main.py              # FastAPI app with 12-layer middleware stack
│   │   └── routers/             # 18 routers (153 endpoints: 150 HTTP + 3 WS, 8,112 total lines)
│   ├── auth/                    # 6 files - OAuth2, JWT (RS256), enhanced auth
│   ├── compliance/              # 10 files - GDPR, consent, data export
│   ├── config/                  # 7 files - settings, database (632 lines)
│   ├── data_ingestion/          # 15 files - API clients per provider, scrapers
│   ├── domain/                  # 5 files - contracts, abstractions
│   ├── etl/                     # 24 files - pipeline, cache (6 cache modules), rate limiting
│   ├── middleware/              # 9 files - priority-based stack
│   ├── migrations/              # 13 Alembic versions
│   ├── ml/                      # 48 files - full ML pipeline
│   │   ├── data_prep/           # Training data generation
│   │   ├── models/              # Ensemble voting classifier
│   │   ├── pipeline/            # Deployment, optimization
│   │   └── training/            # LSTM, XGBoost, Prophet trainers
│   ├── models/                  # 12 files - unified_models.py (canonical ORM Base)
│   ├── monitoring/              # 19 files - Prometheus, alerts, dashboards
│   ├── repositories/            # 13 files - async CRUD pattern
│   ├── scanner/                 # 4 files - market scanning
│   ├── security/                # 20 files - comprehensive security stack (3 stubs: rbac, crypto, passwords)
│   ├── services/                # 20 files - business logic layer (10,241 total lines)
│   ├── streaming/               # 2 files - Kafka client
│   ├── tasks/                   # 14 files - Celery (5 queues, beat schedule)
│   ├── tests/                   # 110 test files, 4,931 tests
│   │   ├── unit/                # 41 files
│   │   ├── integration/         # 16 files
│   │   ├── security/            # 5 files
│   │   ├── middleware/          # 4 files
│   │   ├── fixtures/            # 4 fixture files
│   │   └── *.py                 # 49 top-level test files
│   ├── TradingAgents/           # 39 files - LangGraph trading system (3 test files, low coverage)
│   └── utils/                   # 61 files (55 active, reduced from 87)
├── frontend/
│   └── web/                     # React 18 + TypeScript + Vite + MUI v5
│       └── src/
│           ├── components/      # 55 files across 15 subdirectories (1 dead: EnhancedDashboard.tsx)
│           │   ├── alerts/      # AlertForm, AlertsList
│           │   ├── analysis/    # AnalysisCharts, AnalysisFilters, AnalysisTable
│           │   ├── cards/       # Recommendation cards, portfolio summary, news
│           │   ├── charts/      # MarketHeatmap, Sparkline, StockChart
│           │   ├── common/      # ErrorBoundary, LoadingSpinner, PageSkeleton
│           │   ├── dashboard/   # Layout, Holdings, Metrics, Performance
│           │   ├── market/      # MarketCharts, MarketSummary, MarketTickers
│           │   ├── monitoring/  # CostMonitor
│           │   ├── panels/      # Allocation, MarketOverview, NewsFeed, Recs
│           │   ├── portfolio/   # PortfolioActions, PortfolioChart, PortfolioTabs
│           │   ├── recommendations/ # Filter, List
│           │   ├── settings/    # SettingsForm, SettingsTabs
│           │   ├── watchlist/   # WatchlistActions, WatchlistTable
│           │   ├── Layout/      # App shell
│           │   ├── NotificationPanel/
│           │   ├── SearchModal/
│           │   └── WebSocketIndicator/
│           ├── pages/           # 14 pages (all lazy-loaded)
│           ├── store/slices/    # 6 Redux slices
│           ├── services/        # API (Axios) + WebSocket (Socket.IO)
│           ├── hooks/           # 13 custom hooks (performance-oriented)
│           ├── types/           # TypeScript definitions
│           ├── theme/           # MUI theming + design tokens
│           ├── utils/           # Accessibility, env helpers
│           ├── config/          # API endpoint registry
│           └── design/          # Design system
│       └── tests/e2e/           # 2 Playwright spec files
├── infrastructure/
│   ├── docker/                  # Dockerfiles (backend, frontend, ML)
│   ├── monitoring/              # Prometheus, Grafana, alerts configs
│   └── nginx/                   # Reverse proxy + SSL config + security headers
├── scripts/                     # 100+ scripts across 8 subdirectories
│   ├── deployment/              # Blue-green deploy, rollback
│   ├── testing/                 # Test execution, coverage
│   ├── monitoring/              # Performance, cost monitoring
│   ├── models/                  # ML model management
│   ├── optimization/            # Performance optimization
│   ├── security/                # Security scanning, validation
│   ├── setup/                   # Environment, dependency setup
│   └── data/                    # Data management, ETL scripts
├── data_pipelines/              # Airflow DAGs
├── datasets/                    # Dataset definitions
├── docs/                        # 18 documentation subdirectories
├── ml_models/                   # Trained model artifacts (XGBoost, Prophet x3)
├── docker-compose.yml           # Base (17 services)
├── docker-compose.production.yml # Production stack (canonical)
├── docker-compose.dev.yml       # Development overrides
├── docker-compose.test.yml      # Test overrides
└── docker-compose.ml-production.yml # ML-specific production
```

## Key Metrics

| Metric | Feb 26 | Mar 3 | Change |
|--------|--------|-------|--------|
| Total project files | ~27,000 | 27,580 | +580 |
| Python source files | ~330 | 493 | +163 (sub-module splits) |
| Frontend TSX components | ~30 | 55 | +25 (extracted, 1 dead code) |
| Frontend pages | 12 | 14 | +2 (auth flows) |
| Backend test files | 96 | 110 | +14 |
| Frontend test files | 4 | 12 | +8 |
| Backend tests (passing) | 3,569 | 4,929 | +1,360 |
| Frontend tests (passing) | 0 | 197 | +197 (NEW) |
| Frontend TS/TSX files (total) | ~60 | 106 | +46 |
| API endpoints | ~140 | 153 | +13 (78 GET, 48 POST, 8 PUT, 7 DEL, 2 PATCH, 3 WS) |
| Router files | 15 | 18 | +3 (8,112 total lines) |
| Service files | 12 | 20 | +8 (10,241 total lines) |
| Security modules | 22 | 20 | Consolidated |
| CI/CD workflows | 28 | 28 | Stable |
| Docker compose files | 4 | 5 | +1 (ML production) |
| ML modules | 36 | 48 | +12 |
| Utils files | 55 | 61 | +6 (sub-module splits) |
| Database migrations | 9 | 13 | +4 |

## Technology Stack

### Backend
| Technology | Version | Notes |
|------------|---------|-------|
| Python | 3.10/3.11/3.12 | CI matrix across 3 versions |
| FastAPI | Latest | 12-layer middleware stack |
| SQLAlchemy | 2.x | DeclarativeBase, async sessions |
| PostgreSQL | 15 | TimescaleDB 2.12.1 extension |
| Redis | 7.2-alpine | Cache + distributed rate limiting |
| Celery | Latest | 5 queues, beat scheduler, prefork pool |
| Alembic | Latest | 13 migration versions |
| Prometheus | Latest | Metrics + alerting |

### Frontend
| Technology | Version | Notes |
|------------|---------|-------|
| React | 18.2.0 | 14 pages, 54 components |
| TypeScript | 5.3.3 | Strict mode (15 errors remaining) |
| Vite | 7.3.1 | 18 manual vendor chunks |
| Redux Toolkit | Latest | 6 domain slices |
| Material-UI | 5.14 | Full theming + design tokens |
| Recharts | Latest | Primary charting |
| Plotly, Chart.js, Lightweight Charts | Latest | 3 additional chart libs |
| Vitest | 4.0.16 | 12 test files, 201 tests |
| Playwright | 1.40 | 2 E2E specs (not yet in CI) |

### ML
| Technology | Notes |
|------------|-------|
| XGBoost | Trained model on disk (690 KB) |
| Prophet | 3 models (AAPL, ADBE, AMZN) |
| LightGBM | NEW - integrated |
| TA-Lib | NEW - technical indicators |
| LSTM | Training code exists, weights not saved |
| FinBERT | Sentiment analysis framework |

## Changes Since Last Analysis (2026-02-26)

### 30 Commits, 130 Files Changed (+39,830/-17,434 lines)

**Backend Modularization (Waves 4-6)**:
- Wave 4: Split 5 largest backend files into sub-modules
- Wave 5: Split 7 large backend files into sub-modules
- Wave 6: Extracted frontend components + added 8 test files

**Testing Expansion (Waves 1-3)**:
- Wave 1: ETL layer extended coverage (+307 tests)
- Wave 2: Monitoring layer extended coverage (+331 tests)
- Wave 3: ML and analytics extended coverage (+564 tests)

**Feature Additions**:
- ML: LightGBM, TA-Lib indicators, Monte Carlo VaR
- Real-time: Socket.IO service, Celery optimization
- Frontend: UI overhaul, expanded pages, auth flows (Login, Register, ForgotPassword), seed script
- ORM: DeclarativeBase migration, legacy shims
- Security: P0 fixes — password verification, crypto stubs, JWT unification

**Frontend Extraction**:
- 14 pages -> all lazy-loaded with typed skeletons
- Monolithic pages split into 54 domain-organized components
- 13 custom performance hooks
- 8 new page-level test files

## Files Over 800 Lines (Verified by Deep Audit)

**30 backend files exceed 800 lines. Top 10:**

| File | Lines | Category |
|------|-------|----------|
| `services/recommendation_service.py` | 1,234 | Services |
| `data_ingestion/market_scanner.py` | 1,211 | Data |
| `models/ml_models.py` | 1,191 | Models |
| `models/unified_models.py` | 1,168 | Models |
| `services/portfolio_service.py` | 1,162 | Services |
| `security/security_config.py` | 1,140 | Security |
| `tasks/maintenance_tasks.py` | 1,111 | Tasks |
| `TradingAgents/cli/main.py` | 1,105 | TradingAgents |
| `monitoring/health_checks.py` | 1,102 | Monitoring |
| `ml/online_learning.py` | 1,082 | ML |

**Distribution**: Services (2), Security (6), ML (5), Monitoring (4), ETL (2), Utils (3), Data (1), Models (2), TradingAgents (1), API (1), Tasks (1), Config (1)

**Frontend files >500 lines:**
| File | Lines | Status |
|------|-------|--------|
| `EnhancedDashboard.tsx` | 746 | DEAD CODE (not imported) |
| `SettingsTabs.tsx` | 586 | Candidate for extraction |
| `PortfolioSummary.tsx` | 534 | Candidate for extraction |

Note: Most backend large files are cohesive domain code, not sprawl. Frontend `EnhancedDashboard.tsx` should be deleted or integrated.
