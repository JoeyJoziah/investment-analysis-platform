# Investment Analysis Platform

**Version: 1.0.0** · **Last Updated: 2026-08-11**

A comprehensive, AI-powered investment analysis and recommendation platform that analyzes 6,000+ publicly traded stocks from NYSE, NASDAQ, and AMEX exchanges.

**Status**: Advanced-beta (not production-ready) — see [docs/STATUS.md](docs/STATUS.md) for authoritative state | **Budget**: <$50/month target | **Tests**: ~5,298 collected | **Routers**: 18 registered in `backend/api/main.py` (plus `monitoring.py` router file present but not wired)

---

## Quick Start

```bash
# 1. Initial setup (run once)
./setup.sh

# 2. Start development environment
./start.sh dev

# 3. Access the application
# Frontend: http://localhost:3000
# API Docs: http://localhost:8000/docs
# Grafana:  http://localhost:3001
```

---

## Features

### Core Capabilities
- **Real-time Stock Analysis**: Technical, fundamental, and sentiment analysis
- **AI-Powered Recommendations**: ML models including LSTM, XGBoost, and Prophet
- **Portfolio Management**: Track and optimize investment portfolios
- **Watchlist Management**: Custom stock watchlists with alerts
- **Real-time Updates**: WebSocket-based live data streaming

### Technical Features
- **Cost Optimized**: Designed to run under $50/month using free API tiers
- **Fully Automated**: Daily analysis without manual intervention
- **Compliance Ready**: GDPR and SEC compliant architecture
- **Scalable**: Handles 6,000+ stocks with intelligent caching
- **Security Hardened**: CSRF protection, rate limiting, auth gates, SSL/TLS, GDPR encryption key, timezone-aware UTC

---

## Architecture

```
investment-analysis-platform/
├── backend/                    # FastAPI backend
│   ├── api/                    # REST API endpoints (18 routers; see /docs)
│   ├── models/                 # SQLAlchemy ORM models (unified_models.py is canonical)
│   ├── services/               # Extracted service layer (10 service modules)
│   ├── ml/                     # ML pipeline (LSTM, XGBoost, Prophet, FinBERT)
│   ├── etl/                    # ETL processors
│   ├── tasks/                  # Celery task queue
│   ├── utils/                  # Utilities
│   └── migrations/             # Alembic migrations
├── frontend/web/               # React + Vite application
│   ├── src/components/         # UI components (portfolio/ subdirectory for analytics)
│   ├── src/pages/              # Page components
│   ├── src/store/              # Redux state slices
│   └── src/hooks/              # Custom hooks
├── infrastructure/             # Docker, Nginx, monitoring configs
│   ├── docker/                 # Dockerfile helpers and init scripts
│   └── monitoring/             # Prometheus, Grafana, Loki, AlertManager configs
├── data_pipelines/airflow/     # Apache Airflow DAGs
├── ml_models/                  # Trained ML model artifacts
├── scripts/                    # Automation scripts
└── .github/workflows/          # CI/CD pipelines
```

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | FastAPI 0.115+, Python 3.12, Uvicorn/Gunicorn |
| **Frontend** | React 18, TypeScript 5, Vite, Redux Toolkit, Material-UI 5 |
| **Database** | PostgreSQL 15 + TimescaleDB (time-series) |
| **Cache** | Redis 7 (multi-layer caching, Celery broker/result backend) |
| **Search** | PostgreSQL Full-Text Search (pg_trgm) |
| **Task Queue** | Celery 5 + Redis (broker: DB 0, results: DB 1) |
| **Data Pipelines** | Apache Airflow 2.7 |
| **ML/AI** | PyTorch, XGBoost, Prophet, FinBERT |
| **Monitoring** | Prometheus + VictoriaMetrics, Grafana 10, Loki + Promtail, AlertManager |
| **Containerization** | Docker + Docker Compose |
| **SSL** | Certbot (Let's Encrypt, auto-renewing) |
| **CI/CD** | GitHub Actions |

---

## Available Commands

### Shell Scripts
```bash
./setup.sh         # Initial setup with secure credentials
./start.sh dev     # Start development environment
./start.sh prod    # Start production environment
./start.sh test    # Run tests
./stop.sh          # Stop all services
./stop.sh --clean  # Stop and clean volumes
./logs.sh          # View all logs
./logs.sh backend  # View specific service logs
```

### Backend Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn backend.api.main:app --reload

# Run tests
pytest backend/tests/ --cov=backend

# Run only fast tests (skip slow/integration)
pytest backend/tests/ -m "not slow"

# Format code
black backend/ --line-length 88
isort backend/ --profile black
```

### Frontend Development
```bash
# Install dependencies
cd frontend/web && npm install

# Start development server (Vite)
npm run dev

# Run tests
npm test

# Build for production
npm run build
```

---

## API Endpoints

### Main API (Port 8000)

All v1 endpoints are mounted under `/api/v1/...`. The live, authoritative list is at `/docs` (Swagger). Highlights:

| Prefix | Router | Purpose |
|--------|--------|---------|
| `/api/health` | `health.py` | Health, readiness, liveness probes |
| `/api/v1/auth` | `auth.py` | Login, refresh, registration |
| `/api/v1/stocks` | `stocks.py` | Stock list, details, search |
| `/api/v1/analysis` | `analysis.py` | Per-ticker analysis (technical, fundamental, sentiment) |
| `/api/v1/recommendations` | `recommendations.py` | AI buy/hold/sell recommendations |
| `/api/v1/portfolio` | `portfolio.py` | Portfolio CRUD, performance, rebalancing |
| `/api/v1/watchlists` | `watchlist.py` | Watchlist CRUD + alerts |
| `/api/v1/thesis` | `thesis.py` | Investment-thesis tracking |
| `/api/v1/news` | `news.py` | Aggregated news + sentiment |
| `/api/v1/ml` | `ml.py` | ML inference + model status |
| `/api/v1/trading` | `trading.py` | Order placement / paper trading |
| `/api/v1/agents` | `agents.py` | TradingAgents orchestration |
| `/api/v1/settings` | `settings.py` | User preferences |
| `/api/v1/cache` | `cache_management.py` | Cache inspection + warming |
| `/api/v1/admin` | `admin.py` | Admin-only operations |
| `/api/v1` (gdpr) | `gdpr.py` | GDPR export/delete/consent |
| `/api/v1/ws` | `websocket.py` | Real-time WebSocket stream |
| (versioning) | `backend/api/versioning.py` | V1 migration monitoring (self-prefixed) |
| `/docs` | — | Swagger UI |
| `/redoc` | — | ReDoc UI |

> **Note:** `backend/api/routers/monitoring.py` is present but currently not wired into `main.py`. Tracked as a follow-up — see `docs/ANALYSIS_2026-05-14.md`.

---

## ML Models

| Model | Purpose | File |
|-------|---------|------|
| LSTM | Neural network predictions | lstm_weights.pth |
| XGBoost | Gradient boosting | xgboost_model.pkl |
| Prophet | Time-series forecasting | prophet/ directory |
| FinBERT | Sentiment analysis | via HuggingFace |

### ML Features
- Automated daily retraining with performance monitoring
- Real-time inference (<100ms prediction latency)
- Model versioning and rollback
- Drift detection and alerting
- Backtesting and strategy validation

---

## Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Required API Keys (free tiers available)
ALPHA_VANTAGE_API_KEY=your_key     # 25 calls/day
FINNHUB_API_KEY=your_key           # 60 calls/minute
POLYGON_API_KEY=your_key           # 5 calls/minute
NEWS_API_KEY=your_key              # 100 requests/day

# Auto-generated by setup.sh
DB_PASSWORD=auto_generated
REDIS_PASSWORD=auto_generated
SECRET_KEY=auto_generated
JWT_SECRET_KEY=auto_generated

# Required for GDPR compliance (generate before first run)
GDPR_ENCRYPTION_KEY=<fernet-key>   # python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# Production-only
CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
PROMETHEUS_REMOTE_URL=http://victoriametrics:8428/api/v1/write
```

See [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) for the full reference.

---

## Monitoring & Observability

### Service URLs (Development)
| Service | URL | Purpose |
|---------|-----|---------|
| Frontend | http://localhost:3000 | Web application |
| Backend API | http://localhost:8000 | REST API |
| API Docs | http://localhost:8000/docs | Swagger UI |
| Grafana | http://localhost:3001 | Dashboards |
| Prometheus | http://localhost:9090 | Metrics |
| VictoriaMetrics | http://localhost:8428 | Long-term metric storage |
| Loki | http://localhost:3100 | Log aggregation |

### Production Monitoring Stack
- **Prometheus** collects metrics from all services (15-30s scrape intervals)
- **VictoriaMetrics** stores metrics long-term (90-day retention)
- **Grafana** provides dashboards (API, Database, System, Business)
- **Loki + Promtail** aggregates structured logs from all containers
- **AlertManager** routes alerts to Slack/email/PagerDuty

---

## Testing

```bash
# Run all tests
pytest backend/tests/ --cov=backend --cov-report=html

# By marker
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Skip tests marked @pytest.mark.slow
```

### Test Status
- **5026 tests passing**, 8 skipped, 2 xfailed, 0 failed
- 28 unit test files in `backend/tests/unit/`
- Integration tests covering registered routers
- Security tests for rate limiting and security modules

---

## Deployment

### Development
```bash
./start.sh dev
```

### Production
```bash
# Configure SSL first
./scripts/init-ssl.sh yourdomain.com admin@yourdomain.com

# Start production (uses docker-compose.production.yml overlay)
./start.sh prod

# Monitor services
docker compose logs -f
```

### Production Docker Services
| Service | Purpose |
|---------|---------|
| postgres (TimescaleDB 15) | Primary database |
| redis:7 | Cache + Celery broker/results |
| backend (FastAPI + Gunicorn) | API server |
| frontend (React + Nginx) | Web UI |
| celery_worker | Background task processing |
| celery_beat | Scheduled task triggers |
| nginx | TLS termination, reverse proxy |
| certbot | Automatic Let's Encrypt renewal |
| prometheus | Metrics collection |
| victoriametrics | Long-term metric storage |
| grafana | Dashboards |
| loki | Log aggregation |
| promtail | Log shipping agent |
| alertmanager | Alert routing |

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for the full production guide.

---

## Compliance

### SEC 2025
- Investment recommendation disclosures
- Audit logging for all recommendations
- Risk disclosure statements
- Suitability assessments

### GDPR
- Data export endpoints (right to portability)
- Right to be forgotten (deletion)
- Consent management
- PII encryption via `GDPR_ENCRYPTION_KEY`

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Production deployment guide |
| [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) | Environment variable reference |
| [docs/INSTALLATION_GUIDE.md](docs/INSTALLATION_GUIDE.md) | Step-by-step installation |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Common issues and solutions |
| [CLAUDE.md](CLAUDE.md) | Development guidelines and AI agent framework |
| [/docs](http://localhost:8000/docs) | Interactive API documentation (Swagger) |

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests first (TDD)
4. Make your changes
5. Ensure tests pass (`pytest backend/tests/`)
6. Submit a pull request

### Code Standards
- Python: Black (88 chars), isort, mypy strict mode
- TypeScript: ESLint, Prettier
- Test coverage: 80% minimum
- Conventional commits

---

## License

MIT License - see LICENSE file for details

---

*Last updated: 2026-03-04*
