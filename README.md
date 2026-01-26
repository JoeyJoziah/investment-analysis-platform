# Investment Analysis Platform

A comprehensive, AI-powered investment analysis and recommendation platform designed to analyze 6,000+ publicly traded stocks from NYSE, NASDAQ, and AMEX exchanges.

**Status**: 95% Production Ready | **Budget**: <$50/month | **Codebase**: ~1,550,000 LOC | **Quick Wins**: 60-80% Performance Improvement

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
- **Compliance Ready**: GDPR and SEC 2025 compliant architecture
- **Multi-Agent AI**: 134 specialized AI agents for various tasks
- **Scalable**: Handles 6,000+ stocks with intelligent caching
- **Performance Optimized**: Quick Wins implemented for 60-80% improvement

---

## Architecture

```
investment-analysis-platform/
├── backend/                    # FastAPI backend (32 directories)
│   ├── api/                    # REST API endpoints (13 routers)
│   ├── models/                 # SQLAlchemy ORM models
│   ├── ml/                     # ML pipeline (22 modules)
│   ├── etl/                    # ETL processors (17 modules)
│   ├── tasks/                  # Celery task queue (9 modules)
│   ├── utils/                  # Utilities (91 modules)
│   └── migrations/             # Alembic migrations
├── frontend/web/               # React application
│   ├── src/components/         # UI components (12 directories)
│   ├── src/pages/              # Page components (15 files)
│   ├── src/store/              # Redux state (6 slices)
│   └── src/hooks/              # Custom hooks
├── data_pipelines/airflow/     # Apache Airflow DAGs
├── infrastructure/             # Docker, monitoring configs
├── ml_models/                  # Trained ML model artifacts
├── scripts/                    # Automation scripts (72 files)
├── .claude/                    # AI Agent framework
│   ├── agents/                 # 134 AI agents
│   ├── skills/                 # 71 skills
│   ├── commands/               # 175+ commands
│   └── rules/                  # 8 coding rules
└── .github/workflows/          # CI/CD (14 workflows)
```

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | FastAPI 0.115+, Python 3.12, Uvicorn/Gunicorn |
| **Frontend** | React 18.2, TypeScript 5.3, Redux Toolkit, Material-UI 5.14 |
| **Database** | PostgreSQL 15 + TimescaleDB (time-series) |
| **Cache** | Redis 7.0 (multi-layer caching) |
| **Search** | PostgreSQL Full-Text Search (pg_trgm) |
| **Task Queue** | Celery 5.4 + Redis backend |
| **Data Pipelines** | Apache Airflow 2.7.3 |
| **ML/AI** | PyTorch 2.4, XGBoost 2.1, Prophet 1.1.5, FinBERT |
| **Monitoring** | Prometheus, Grafana 10.2, AlertManager |
| **Containerization** | Docker 7.1, docker-compose |
| **CI/CD** | GitHub Actions (14 workflows) |

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
./notion-sync.sh   # Sync with Notion tracker
```

### Backend Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn backend.api.main:app --reload

# Run tests
pytest backend/tests/ --cov=backend

# Format code
black backend/ --line-length 88
isort backend/ --profile black
```

### Frontend Development
```bash
# Install dependencies
cd frontend/web && npm install

# Start development server
npm start

# Run tests
npm test

# Build for production
npm run build
```

---

## API Endpoints

### Main API (Port 8000)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/stocks` | GET | List all stocks |
| `/api/stocks/{ticker}` | GET | Stock details |
| `/api/recommendations` | GET | AI recommendations |
| `/api/analysis/{ticker}` | GET | Detailed analysis |
| `/api/portfolio` | GET/POST | Portfolio management |
| `/api/watchlists` | GET/POST | Watchlist operations |
| `/api/ws` | WS | Real-time WebSocket |
| `/docs` | GET | Swagger documentation |

### ML API (Port 8001)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | ML service health |
| `/models` | GET | List ML models |
| `/predict` | POST | Make predictions |
| `/predict/stock_price` | POST | Stock predictions |
| `/retrain` | POST | Trigger retraining |

---

## ML Models

The platform includes a comprehensive ML pipeline for automated stock analysis:

### Trained Models
| Model | Purpose | File |
|-------|---------|------|
| LSTM | Neural network predictions | lstm_weights.pth (5.1MB) |
| XGBoost | Gradient boosting | xgboost_model.pkl (274KB) |
| Prophet | Time-series forecasting | prophet/ directory |

### ML Features
- **Automated Training**: Daily retraining with performance monitoring
- **Real-time Inference**: <100ms prediction latency
- **Model Versioning**: Automatic versioning and rollback
- **Performance Monitoring**: Drift detection and alerting
- **Backtesting**: Strategy validation engine

---

## Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Required API Keys (free tiers available)
ALPHA_VANTAGE_API_KEY=your_key     # 25 calls/day
FINNHUB_API_KEY=your_key           # 60 calls/minute
POLYGON_API_KEY=your_key           # 5 calls/minute
NEWS_API_KEY=your_key

# Auto-generated by setup.sh
DB_PASSWORD=auto_generated
REDIS_PASSWORD=auto_generated
SECRET_KEY=auto_generated
JWT_SECRET_KEY=auto_generated

# Optional
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Monitoring & Observability

### Service URLs
| Service | URL | Purpose |
|---------|-----|---------|
| Frontend | http://localhost:3000 | Web application |
| Backend API | http://localhost:8000 | REST API |
| API Docs | http://localhost:8000/docs | Swagger UI |
| ML API | http://localhost:8001 | ML service |
| Grafana | http://localhost:3001 | Dashboards |
| Prometheus | http://localhost:9090 | Metrics |

### Production Monitoring
- Prometheus metrics collection
- Grafana dashboards (API, Database, ML, System)
- AlertManager for notifications
- Real-time performance tracking

---

## Testing

```bash
# Run all tests
./start.sh test

# Backend tests with coverage
pytest backend/tests/ --cov=backend --cov-report=html

# Frontend tests
cd frontend/web && npm test

# Specific test markers
pytest -m "unit"        # Unit tests
pytest -m "integration" # Integration tests
pytest -m "financial"   # Financial model tests
```

### Coverage Requirements
- Minimum: 85%
- Backend: 86 tests passing
- Frontend: 84 tests passing

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

# Start production
./start.sh prod

# Monitor services
docker compose logs -f
```

### Docker Services
- PostgreSQL 15 + TimescaleDB
- Redis 7 (caching)
- PostgreSQL Full-Text Search (replaces Elasticsearch)
- Backend (FastAPI)
- Frontend (React + Nginx)
- Celery Worker & Beat
- Prometheus, Grafana, AlertManager (production)

---

## AI Agent Framework

The platform integrates a sophisticated multi-agent AI system:

### Statistics
| Category | Count |
|----------|-------|
| AI Agents | 134 |
| Skills | 71 |
| Commands | 175+ |
| Helper Scripts | 32 |

### Primary Swarms
- **infrastructure-devops-swarm**: Docker, CI/CD, deployment
- **data-ml-pipeline-swarm**: ETL, Airflow, ML training
- **financial-analysis-swarm**: Stock analysis, predictions
- **backend-api-swarm**: FastAPI, REST APIs
- **ui-visualization-swarm**: React, dashboards
- **project-quality-swarm**: Code review, testing
- **security-compliance-swarm**: SEC/GDPR compliance

### Custom Investment Agents
- queen-investment-orchestrator
- investment-analyst
- deal-underwriter
- financial-modeler
- risk-assessor
- portfolio-manager

See [CLAUDE.md](CLAUDE.md) for detailed agent documentation.

---

## Cost Optimization

Designed to operate under **$50/month**:

| Component | Estimated Cost |
|-----------|---------------|
| Database | ~$10 |
| Compute | ~$15 |
| Storage | ~$5 |
| APIs | ~$10 |
| **Total** | **~$40/month** |

### Strategies
- Free API tier optimization
- Multi-layer intelligent caching
- Batch processing during off-peak hours
- Auto-scaling with resource limits

---

## Compliance

### SEC 2025
- Investment recommendation disclosures
- Audit logging for all recommendations
- Risk disclosure statements
- Suitability assessments

### GDPR
- Data export endpoints
- Right to be forgotten (deletion)
- Consent management
- Data anonymization

---

## Documentation

| Document | Description |
|----------|-------------|
| [CLAUDE.md](CLAUDE.md) | Development guidelines & agent framework |
| [TODO.md](TODO.md) | Project task tracking |
| [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) | Detailed status report |
| [ML_QUICKSTART.md](ML_QUICKSTART.md) | ML quick start guide |
| [ML_PIPELINE_DOCUMENTATION.md](ML_PIPELINE_DOCUMENTATION.md) | ML technical reference |
| [/docs](http://localhost:8000/docs) | Interactive API documentation |

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests (`./start.sh test`)
5. Submit a pull request

### Code Standards
- Python: Black (88 chars), isort, mypy, flake8
- TypeScript: ESLint, Prettier
- Test coverage: 85% minimum
- Conventional commits

---

## License

MIT License - see LICENSE file for details

---

## Acknowledgments

- Alpha Vantage for market data
- Finnhub for real-time quotes
- Polygon.io for historical data
- NewsAPI for sentiment analysis
- Claude Code for AI agent framework

---

**Built for automated investment analysis**

*Last updated: 2026-01-26*
