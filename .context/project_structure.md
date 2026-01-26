# Project Structure

**Last Updated**: 2026-01-25 (Session 2 - Docker Fixes Applied)
**Total Files**: 26,524+
**Analysis Method**: Full codebase scan with exploration agents + quality swarm analysis

## Executive Summary

This Investment Analysis Platform is a comprehensive AI-powered stock analysis system designed to analyze 6,000+ publicly traded stocks from NYSE, NASDAQ, and AMEX exchanges. The system operates autonomously with a target operational cost under $50/month.

## Directory Tree

```
investment-analysis-platform/
├── .claude/                          # Claude Code agent configurations (128+ agents)
│   ├── agents/                       # Specialized agent swarm configs
│   │   ├── architecture-reviewer.md
│   │   ├── backend-api-swarm.md
│   │   ├── code-review-expert.md
│   │   ├── data-ml-pipeline-swarm.md
│   │   ├── data-science-architect.md
│   │   ├── deal-underwriter.md       # Investment underwriting
│   │   ├── financial-analysis-swarm.md
│   │   ├── financial-modeler.md      # Financial modeling
│   │   ├── godmode-refactorer.md
│   │   ├── infrastructure-devops-swarm.md
│   │   ├── investment-analyst.md     # Investment analysis
│   │   ├── portfolio-manager.md      # Portfolio management
│   │   ├── project-quality-swarm.md
│   │   ├── queen-investment-orchestrator.md  # Master orchestrator
│   │   ├── risk-assessor.md          # Risk assessment
│   │   ├── security-compliance-swarm.md
│   │   ├── team-coordinator.md
│   │   ├── ui-visualization-swarm.md
│   │   └── [80+ more agent configs...]
│   ├── commands/                     # Custom slash commands (164+)
│   ├── helpers/                      # Helper scripts (35)
│   ├── rules/                        # Coding rules (8)
│   ├── skills/                       # Agent skills (71)
│   │   ├── financial-modeling/       # DCF, LBO modeling
│   │   ├── deal-structuring/         # Security packages
│   │   ├── underwriting-analysis/    # Credit analysis
│   │   ├── sec-compliance/           # SEC 2025 compliance
│   │   ├── cost-monitor/             # Budget tracking
│   │   └── [66+ more skills...]
│   └── v3/                           # V3 implementation
│
├── .context/                         # Project documentation & status
│   ├── overall_project_status.md     # Current status report
│   ├── feature_checklist.md          # Feature completion tracking
│   ├── deployment_readiness.md       # Deployment checklist
│   ├── identified_issues.md          # Known issues & fixes
│   ├── recommendations.md            # Strategic recommendations
│   ├── project_structure.md          # This file
│   └── README.md
│
├── .devcontainer/                    # Dev container config
│   ├── Dockerfile
│   ├── devcontainer.json
│   └── post-create.sh
│
├── .github/                          # GitHub configuration
│   ├── workflows/                    # 14 CI/CD workflows
│   │   ├── ci.yml                    # Main CI pipeline
│   │   ├── claude.yml                # Claude GitHub integration
│   │   ├── cleanup.yml
│   │   ├── comprehensive-testing.yml
│   │   ├── daily-pipeline-validation.yml
│   │   ├── dependency-updates.yml
│   │   ├── migration-check.yml
│   │   ├── performance-monitoring.yml
│   │   ├── production-deploy.yml
│   │   ├── release-management.yml
│   │   ├── reusable-build.yml
│   │   ├── reusable-test.yml
│   │   ├── security-scan.yml
│   │   └── staging-deploy.yml
│   ├── ISSUE_TEMPLATE/
│   ├── codeql/
│   └── dependabot.yml
│
├── backend/                          # FastAPI Backend (~400+ Python files)
│   ├── api/                          # FastAPI endpoints
│   │   ├── main.py                   # App entry point with lifespan
│   │   └── routers/                  # 18 API routers
│   │       ├── admin.py              # Admin operations
│   │       ├── agents.py             # Trading agents API
│   │       ├── analysis.py           # Analysis endpoints
│   │       ├── auth.py               # Authentication
│   │       ├── cache_management.py   # Cache operations
│   │       ├── gdpr.py               # GDPR compliance
│   │       ├── health.py             # Health checks
│   │       ├── monitoring.py         # Metrics endpoints
│   │       ├── portfolio.py          # Portfolio management
│   │       ├── recommendations.py    # Recommendations API
│   │       ├── stocks.py             # Stock CRUD
│   │       ├── stocks_legacy.py      # Legacy stock endpoints
│   │       ├── watchlist.py          # Watchlist management
│   │       └── websocket.py          # Real-time updates
│   │
│   ├── analytics/                    # Analysis engines
│   │   ├── agents/                   # ML agent orchestration
│   │   ├── fundamental/              # Fundamental analysis
│   │   │   └── valuation/dcf_model.py
│   │   ├── portfolio/                # Portfolio optimization
│   │   │   ├── black_litterman.py
│   │   │   └── modern_portfolio_theory.py
│   │   ├── statistical/              # Statistical analysis
│   │   │   └── cointegration_analyzer.py
│   │   ├── finbert_analyzer.py       # FinBERT NLP
│   │   ├── fundamental_analysis.py   # P/E, ROE, etc.
│   │   ├── recommendation_engine.py
│   │   ├── sentiment_analysis.py
│   │   └── technical_analysis.py     # RSI, MACD, etc.
│   │
│   ├── compliance/                   # Regulatory compliance
│   │   ├── audit_logging.py
│   │   ├── gdpr.py
│   │   └── sec.py
│   │
│   ├── config/                       # Configuration
│   │   ├── __init__.py
│   │   ├── database.py               # Async DB config
│   │   ├── monitoring_config.py
│   │   └── settings.py               # Environment settings
│   │
│   ├── data_ingestion/               # Data collection
│   │   ├── alpha_vantage_client.py
│   │   ├── finnhub_client.py
│   │   ├── polygon_client.py
│   │   ├── sec_edgar_client.py
│   │   ├── robust_api_client.py
│   │   └── smart_data_fetcher.py
│   │
│   ├── etl/                          # ETL pipeline
│   │   ├── data_extractor.py
│   │   ├── data_loader.py
│   │   ├── data_transformer.py
│   │   ├── etl_orchestrator.py
│   │   └── stock_universe_manager.py
│   │
│   ├── migrations/                   # Alembic migrations
│   │   └── versions/                 # 8 migration files
│   │
│   ├── ml/                           # Machine Learning
│   │   ├── training/                 # Model training scripts
│   │   │   ├── train_lstm.py
│   │   │   ├── train_xgboost.py
│   │   │   ├── train_prophet.py
│   │   │   └── run_full_training.py
│   │   ├── backtesting.py
│   │   ├── feature_store.py
│   │   ├── model_manager.py
│   │   └── training_pipeline.py
│   │
│   ├── models/                       # Database models
│   │   ├── unified_models.py         # Primary ORM models
│   │   ├── schemas.py                # Pydantic schemas
│   │   ├── tables.py                 # SQLAlchemy tables
│   │   └── database.py
│   │
│   ├── repositories/                 # Data access layer
│   │   ├── base.py
│   │   ├── stock_repository.py
│   │   ├── portfolio_repository.py
│   │   └── watchlist_repository.py
│   │
│   ├── security/                     # Security (16 modules)
│   │   ├── oauth2.py
│   │   ├── jwt_manager.py
│   │   ├── data_encryption.py
│   │   ├── rate_limiter.py
│   │   ├── audit_logging.py
│   │   └── [11 more security modules...]
│   │
│   ├── tasks/                        # Celery tasks
│   │   ├── celery_app.py
│   │   ├── analysis_tasks.py
│   │   ├── data_pipeline.py
│   │   └── scheduler.py
│   │
│   └── utils/                        # Utilities
│       ├── data_anonymization.py     # GDPR encryption (CRITICAL)
│       ├── cache.py
│       └── database.py
│
├── frontend/                         # React frontend
│   └── web/                          # (~50+ TypeScript files)
│       ├── src/
│       │   ├── components/
│       │   │   ├── Layout/
│       │   │   ├── charts/
│       │   │   │   ├── MarketHeatmap.tsx
│       │   │   │   └── StockChart.tsx
│       │   │   ├── cards/
│       │   │   │   ├── RecommendationCard.tsx
│       │   │   │   ├── PortfolioSummary.tsx
│       │   │   │   └── NewsCard.tsx
│       │   │   └── EnhancedDashboard.tsx
│       │   ├── pages/                # 11 pages
│       │   │   ├── Dashboard.tsx
│       │   │   ├── Analysis.tsx
│       │   │   ├── Portfolio.tsx
│       │   │   ├── Recommendations.tsx
│       │   │   └── [7 more pages...]
│       │   ├── services/
│       │   │   ├── api.service.ts
│       │   │   └── websocket.service.ts
│       │   ├── store/slices/         # Redux slices
│       │   └── App.tsx
│       ├── package.json
│       ├── vite.config.ts
│       └── Dockerfile
│
├── infrastructure/                   # Docker & deployment
│   ├── docker/
│   │   ├── backend/
│   │   │   ├── Dockerfile
│   │   │   └── Dockerfile.prod
│   │   ├── frontend/
│   │   │   └── Dockerfile.prod
│   │   ├── nginx/
│   │   └── postgres/
│   └── monitoring/
│       ├── prometheus.yml
│       ├── alertmanager.yml
│       └── grafana/dashboards/
│
├── ml_models/                        # Trained ML models
│   ├── lstm_weights.pth              # 5.1 MB - LSTM neural network
│   ├── lstm_scaler.pkl               # 1.9 KB - Feature scaling
│   ├── xgboost_model.pkl             # 690 KB - Gradient boosting
│   ├── xgboost_scaler.pkl            # 1.9 KB - Scaling
│   ├── xgboost_config.json           # Model config
│   ├── xgboost_feature_importance.json
│   └── prophet/                      # Time-series models
│       ├── AAPL_model.pkl
│       ├── ADBE_model.pkl
│       ├── AMZN_model.pkl
│       └── trained_stocks.json
│
├── data_pipelines/                   # Airflow pipelines
│   └── airflow/
│       └── dags/
│           ├── daily_stock_pipeline.py
│           ├── enhanced_stock_pipeline.py
│           └── ml_training_pipeline_dag.py
│
├── scripts/                          # Utility scripts
│   ├── data/
│   ├── deployment/
│   ├── models/
│   └── setup/
│
├── tests/                            # Integration tests (20 files)
│   └── test_database_fixes.py
│
├── docker-compose.yml                # Main Docker composition
├── docker-compose.dev.yml
├── docker-compose.prod.yml
├── docker-compose.test.yml
├── requirements.txt                  # 170+ Python packages
├── CLAUDE.md                         # Claude Code instructions
├── start.sh                          # Start services
├── stop.sh                           # Stop services
├── setup.sh                          # Initial setup
└── logs.sh                           # View logs
```

## New/Modified Files Since Last Analysis

### NEW Files
- `.claude/agents/` - 80+ new agent configurations
- `.claude/skills/` - 71 skill modules
- `.claude/commands/` - 164 command files
- `.claude/helpers/` - 35 helper scripts
- `backend/migrations/versions/c849a2ab3b24_add_updated_at_to_alerts_table.py`
- `infrastructure/docker/nginx/conf.d/security-headers.conf`

### MODIFIED Files (from git status)
- `CLAUDE.md` - Agent swarm instructions updated
- `backend/config/__init__.py`
- `backend/migrations/versions/004_add_cache_storage_table.py`
- `backend/ml/training/train_prophet.py`
- `backend/ml/training/train_xgboost.py`
- `backend/tasks/celery_app.py`
- `backend/utils/async_database_fixed.py`
- `backend/utils/cache_monitoring.py`
- `backend/utils/robust_error_handling.py`
- `data/ml_training/` - Training data files
- `docker-compose.yml`
- `docker-compose.prod.yml`
- `infrastructure/docker/backend/Dockerfile`
- `infrastructure/docker/backend/Dockerfile.prod`
- `infrastructure/docker/frontend/Dockerfile.prod`
- `infrastructure/monitoring/prometheus.prod.yml`
- `ml_models/` - All model files updated
- `requirements.txt`

### DELETED Files
- `data/training/sample_training_data.csv`
- `frontend/frontend/web/` - Entire duplicate directory removed

## Technology Stack

### Backend
| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.12 | Runtime |
| FastAPI | 0.115+ | Web framework |
| Uvicorn | 0.30.1 | ASGI server |
| SQLAlchemy | 2.0+ | ORM (async) |
| Pydantic | 2.8.2 | Data validation |
| Celery | 5.4.0 | Task queue |
| Redis | 5.0.7 | Caching/broker |

### Frontend
| Technology | Version | Purpose |
|------------|---------|---------|
| React | 18.2.0 | UI framework |
| TypeScript | 5.3.3 | Type safety |
| Vite | 5.0.12 | Build tool |
| Redux Toolkit | 1.9.7 | State management |
| Material-UI | 5.14.20 | UI components |
| Recharts | 2.10.3 | Charts |

### Infrastructure
| Technology | Version | Purpose |
|------------|---------|---------|
| Docker | Latest | Containerization |
| PostgreSQL | 15 | Database |
| TimescaleDB | Latest | Time-series |
| Elasticsearch | 8.11.1 | Search |
| Prometheus | 2.48.0 | Metrics |
| Grafana | 10.2.2 | Dashboards |
| Airflow | 2.7.3 | Orchestration |

### ML/AI
| Technology | Version | Purpose |
|------------|---------|---------|
| PyTorch | 2.4.0 | Deep learning |
| XGBoost | 2.1.1 | Ensemble ML |
| Prophet | 1.1.5 | Time-series |
| scikit-learn | 1.5.1 | ML utilities |
| Transformers | 4.43.3 | FinBERT |
| Optuna | 3.6.1 | Hyperparameters |

## File Statistics

| Category | Count |
|----------|-------|
| Total Files | 26,524+ |
| Python Files | ~400+ |
| TypeScript/TSX Files | ~50+ |
| Test Files | 20 |
| API Routers | 18 |
| Docker Services | 12 |
| Database Tables | 22 |
| ML Models | 7 files |
| CI/CD Workflows | 14 |
| Security Modules | 16 |
| Agent Configs | 128+ |
| Skills | 71 |
| Commands | 164 |

## Running Docker Services (12 - All Healthy)

| Service | Container | Port | Health |
|---------|-----------|------|--------|
| PostgreSQL/TimescaleDB | investment_db | 5432 | ✅ healthy |
| Redis | investment_cache | 6379 | ✅ healthy |
| Elasticsearch | investment_search | 9200 | ✅ healthy |
| Celery Worker | investment_worker | 8000 | ✅ healthy |
| Celery Beat | investment_scheduler | 8000 | ✅ healthy |
| Apache Airflow | investment_airflow | 8080 | Up |
| Prometheus | investment_prometheus | 9090 | Up |
| Grafana | investment_grafana | 3001 | Up |
| AlertManager | investment_alertmanager | 9093 | Up |
| PostgreSQL Exporter | investment_postgres_exporter | 9187 | Up |
| Redis Exporter | investment_redis_exporter | 9121 | Up |
| Elasticsearch Exporter | investment_elasticsearch_exporter | 9114 | Up |

## Database Schema (22 Tables)

```
public.alerts
public.api_usage
public.audit_logs
public.cost_metrics
public.exchanges
public.fundamentals
public.industries
public.ml_predictions
public.news_sentiment
public.orders
public.portfolios
public.positions
public.price_history
public.recommendations
public.sectors
public.stocks
public.system_metrics
public.technical_indicators
public.transactions
public.user_sessions
public.users
public.watchlists
```

**Note**: Database has 22 tables created but **0 stocks loaded** - data import required.

## Key Entry Points

| Component | Entry Point | Purpose |
|-----------|-------------|---------|
| Backend API | `backend/api/main.py` | FastAPI application |
| Frontend | `frontend/web/src/App.tsx` | React application |
| ML Training | `backend/ml/training/run_full_training.py` | Model training |
| ETL Pipeline | `data_pipelines/airflow/dags/daily_stock_pipeline.py` | Data ingestion |

## Critical Configuration Files

| File | Purpose | Status |
|------|---------|--------|
| `.env` | Environment variables | GDPR key missing |
| `docker-compose.yml` | Service orchestration | Ready |
| `requirements.txt` | Python dependencies | Updated |
| `backend/config/settings.py` | App configuration | Ready |
