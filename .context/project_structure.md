# Project Structure

**Last Updated**: 2026-03-04 (Post P0-P5 completion)
**Previous Analysis**: 2026-03-03

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
│   │   └── routers/             # 19 routers (153+ endpoints, 8,112 total lines)
│   │       └── trading.py       # NEW: order validate/execute/impact
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
│   │   └── training/            # LSTM, XGBoost, Prophet, LightGBM trainers
│   ├── models/                  # 12 files - unified_models.py (canonical ORM Base)
│   ├── monitoring/              # 19 files - Prometheus, alerts, dashboards
│   ├── repositories/            # 13 files - async CRUD pattern
│   ├── scanner/                 # 4 files - market scanning
│   ├── security/                # 20 files - comprehensive security stack
│   │   ├── rbac.py              # COMPLETE: in-memory + DB-backed RBAC
│   │   ├── crypto_utils.py      # COMPLETE: Fernet AES + RSA-2048
│   │   └── password_manager.py  # COMPLETE: bcrypt + legacy PBKDF2 verify
│   ├── services/                # 20 files - business logic layer (10,241 total lines)
│   ├── streaming/               # 2 files - Kafka client
│   ├── tasks/                   # 14 files - Celery (5 queues, beat schedule)
│   ├── tests/                   # 71+ test files, 5,020 passing
│   │   ├── unit/                # 28 files (+ test_trading_router.py, test_ml_router_extended.py)
│   │   ├── integration/         # 16 files
│   │   ├── security/            # 5 files
│   │   ├── middleware/          # 4 files
│   │   ├── fixtures/            # 4 fixture files
│   │   └── *.py                 # 49 top-level test files
│   ├── TradingAgents/           # 39 files - LangGraph trading system (3 test files, ~8% coverage)
│   └── utils/                   # 61 files (reduced from 87, stabilized)
├── frontend/
│   └── web/                     # React 18 + TypeScript + Vite + MUI v5
│       ├── Dockerfile           # Node Alpine multi-stage build
│       └── src/
│           ├── components/      # 54 active files across 15+ subdirectories
│           │   │                # (EnhancedDashboard.tsx DELETED)
│           │   ├── alerts/      # AlertForm, AlertsList
│           │   ├── analysis/    # AnalysisCharts, AnalysisFilters, AnalysisTable
│           │   ├── cards/       # Recommendation cards, portfolio summary, news
│           │   ├── charts/      # MarketHeatmap, Sparkline, StockChart
│           │   ├── common/      # ErrorBoundary, LoadingSpinner, PageSkeleton
│           │   ├── dashboard/   # Layout, Holdings, Metrics, Performance
│           │   ├── market/      # MarketCharts, MarketSummary, MarketTickers
│           │   ├── monitoring/  # CostMonitor
│           │   ├── panels/      # Allocation, MarketOverview, NewsFeed, Recs
│           │   ├── portfolio/   # PortfolioActions, PortfolioChart, PortfolioTabs,
│           │   │                # CorrelationMatrix, EfficientFrontier, RiskDecomposition
│           │   ├── recommendations/ # Filter, List
│           │   ├── settings/    # SettingsForm, SettingsTabs
│           │   ├── watchlist/   # WatchlistActions, WatchlistTable
│           │   ├── Layout/      # App shell
│           │   ├── NotificationPanel/
│           │   ├── SearchModal/
│           │   └── WebSocketIndicator/
│           ├── pages/           # 14 pages (all lazy-loaded)
│           │   └── auth.test.tsx # 30 tests: Login, Register, ForgotPassword
│           ├── store/slices/    # 6 Redux slices
│           ├── services/        # API (Axios) + WebSocket (Socket.IO)
│           ├── hooks/           # 13 custom hooks (performance-oriented)
│           ├── types/           # TypeScript definitions
│           ├── theme/           # MUI theming + design tokens
│           ├── utils/           # Accessibility, env helpers
│           ├── config/          # API endpoint registry
│           └── design/          # Design system
│       └── tests/e2e/           # 2 Playwright spec files (not in CI yet)
├── infrastructure/
│   ├── docker/                  # Dockerfiles (backend, frontend, ML)
│   ├── monitoring/              # Prometheus, Grafana, Loki, SLO alerts configs
│   │   ├── alertmanager.yml
│   │   ├── prometheus.yml
│   │   ├── prometheus.prod.yml
│   │   ├── alerts/
│   │   │   └── slo-targets.yml  # NEW: SLO target definitions
│   │   └── loki/                # Loki + Promtail configs (referenced from compose)
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
├── ml_models/                   # Trained model artifacts (XGBoost 690KB, Prophet x3)
├── docker-compose.yml           # Base (17 services)
├── docker-compose.production.yml # Production stack — Loki, Promtail, certbot added
├── docker-compose.dev.yml       # Development overrides
├── docker-compose.test.yml      # Test overrides
└── docker-compose.ml-production.yml # ML-specific production
```

## Key Metrics

| Metric | Mar 3 | Mar 4 | Change |
|--------|-------|-------|--------|
| Python source files | 493 | 493 | Stable |
| Frontend TSX components | 55 (1 dead) | 54 (dead removed) | -1 (EnhancedDashboard deleted) |
| Frontend pages | 14 | 14 | Stable |
| Backend test files | 71+ | 71+ | Stable |
| Frontend test files | 12 | 13 | +1 (auth.test.tsx) |
| Backend tests (passing) | 4,929 | 5,020 | +91 |
| Frontend tests (passing) | 197 | 197 | Stable |
| API endpoints | 153 | 153+ | +3 trading, +6 ML |
| Router files | 18 | 19 | +1 (trading.py) |
| Service files | 20 | 20 | Stable |
| Security modules | 20 (3 stubs) | 20 (0 stubs) | All stubs resolved |
| CI/CD workflows | 29 | 29 | Stable |
| Docker compose files | 5 | 5 | Stable |
| ML modules | 48 | 48 | Stable |

## Technology Stack

### Backend
| Technology | Version | Notes |
|------------|---------|-------|
| Python | 3.10/3.11/3.12 | CI matrix across 3 versions |
| FastAPI | Latest | 12-layer middleware stack |
| SQLAlchemy | 2.x | DeclarativeBase, async sessions |
| PostgreSQL | 15 | TimescaleDB 2.12.1 extension |
| Redis | 7.2-alpine | 640MB maxmemory, allkeys-lru |
| Celery | Latest | 5 queues, beat scheduler, prefork pool |
| Alembic | Latest | 13 migration versions |
| Prometheus | Latest | Metrics + alerting |
| cryptography | Latest | Fernet, RSA-2048 (crypto_utils) |
| passlib | Latest | bcrypt work factor 12 (password_manager) |

### Frontend
| Technology | Version | Notes |
|------------|---------|-------|
| React | 18.2.0 | 14 pages, 54 components |
| TypeScript | 5.3.3 | Strict mode, zero @ts-ignore |
| Vite | 7.3.1 | 18 manual vendor chunks |
| Redux Toolkit | Latest | 6 domain slices |
| Material-UI | 5.14 | Full theming + design tokens |
| Recharts | Latest | Primary charting |
| Plotly, Chart.js, Lightweight Charts | Latest | 3 additional chart libs |
| Vitest | 4.0.16 | 13 test files, 201 tests |
| Playwright | 1.40 | 2 E2E specs (not yet in CI) |

### ML
| Technology | Notes |
|------------|-------|
| XGBoost | Trained model on disk (690 KB) |
| Prophet | 3 models (AAPL, ADBE, AMZN) |
| LightGBM | Integrated |
| TA-Lib | Technical indicators |
| LSTM | Training code exists, weights not saved |
| FinBERT | Sentiment analysis framework |

## Files Over 800 Lines (Top 10, verified by deep audit)

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

Note: Most are cohesive domain code. None are routers (all routers <753 lines post-extraction).

## Frontend Files >500 Lines

| File | Lines | Status |
|------|-------|--------|
| ~~EnhancedDashboard.tsx~~ | ~~746~~ | DELETED (dead code) |
| `SettingsTabs.tsx` | 586 | Candidate for extraction (non-urgent) |
| `PortfolioSummary.tsx` | 534 | Candidate for extraction (non-urgent) |
