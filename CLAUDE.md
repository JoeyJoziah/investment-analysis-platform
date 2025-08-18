# CLAUDE.md

This file provides guidance to Claude Code when working with this investment analysis platform.

## Project Overview

This is an investment analysis and recommendation application designed to analyze 6,000+ publicly traded stocks from NYSE, NASDAQ, and AMEX exchanges. The system operates autonomously, generating daily recommendations without user input.

Key requirements:
- Target operational cost: under $50/month
- Must use free/open-source tools and APIs with generous free tiers
- Fully automated daily analysis without manual intervention
- Compliance with 2025 SEC and GDPR regulations

## Development Guidelines

When working with this codebase, please follow these principles:
- Maintain the cost-optimization focus (under $50/month)
- Preserve existing API credentials in .env file
- Use the simplified deployment scripts (start.sh, stop.sh, setup.sh)
- Follow the clean architecture patterns established

## Quick Start Commands

Use these simplified commands to work with the platform:

```bash
# Initial setup
./setup.sh

# Start development environment
./start.sh dev

# Start production environment
./start.sh prod

# Run tests
./start.sh test

# View logs
./logs.sh

# Stop all services
./stop.sh
```

## Technology Stack

### Backend
- **Framework**: FastAPI (Python)
- **ML/AI**: PyTorch, scikit-learn, Prophet, Hugging Face Transformers (FinBERT)
- **Data Processing**: Apache Airflow, Kafka, Pandas, NumPy/SciPy
- **Database**: PostgreSQL with TimescaleDB for time-series data
- **Caching**: Redis for API response caching

### Frontend
- **Web**: React.js with Material-UI
- **Visualization**: Plotly Dash, React-based charting libraries

### Infrastructure
- **Containerization**: Docker and docker-compose
- **Monitoring**: Prometheus/Grafana stack
- **Data Pipeline**: Apache Airflow

## Project Structure

```
├── backend/              # Backend API and business logic
├── frontend/web/         # React web application
├── data_pipelines/       # Airflow DAGs for data processing
├── infrastructure/       # Docker configurations and deployment
├── config/              # Configuration files
├── scripts/             # Utility scripts
├── docs/                # Documentation
└── tools/               # Development tools and utilities
```

## Key Development Commands

### Docker Operations
```bash
# Start development environment
./start.sh dev

# Start production environment
./start.sh prod

# View service logs
./logs.sh [service-name]

# Stop all services
./stop.sh
```

### Backend Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run backend directly (for development)
cd backend
uvicorn backend.api.main:app --reload

# Run tests
pytest backend/tests/

# Format code
black backend/
isort backend/
```

### Frontend Development
```bash
# Install dependencies
cd frontend/web
npm install

# Start development server
npm start

# Run tests
npm test

# Build for production
npm run build
```

## Configuration Management

### Environment Variables
Key environment variables are stored in `.env` file:
- `ALPHA_VANTAGE_API_KEY` - Alpha Vantage API key (25 calls/day limit)
- `FINNHUB_API_KEY` - Finnhub API key (60 calls/minute)
- `POLYGON_API_KEY` - Polygon.io API key (5 calls/minute free tier)
- `NEWS_API_KEY` - NewsAPI key for sentiment analysis
- Database and security credentials

### Docker Compose Configurations
- `docker-compose.yml` - Base configuration
- `docker-compose.dev.yml` - Development overrides
- `docker-compose.prod.yml` - Production overrides
- `docker-compose.test.yml` - Testing configuration

## Cost Optimization Strategy

The platform is designed to operate under $50/month through:
- **Smart API Usage**: Batch requests, intelligent caching, rate limiting
- **Efficient Processing**: Optimized queries, parallel processing, data compression
- **Resource Management**: Auto-scaling, resource limits, spot instances
- **Open Source Stack**: PostgreSQL, Redis, Elasticsearch, Grafana

## API Endpoints

### Core Endpoints
- `GET /api/health` - Health check
- `GET /api/recommendations` - Daily stock recommendations
- `GET /api/analysis/{ticker}` - Comprehensive stock analysis
- `GET /api/portfolio` - Portfolio management
- `WS /api/ws` - Real-time updates

### Development URLs
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Grafana: http://localhost:3001
- PgAdmin: http://localhost:5050 (dev only)

## Testing Strategy

```bash
# Run all tests
./start.sh test

# Run backend tests only
pytest backend/tests/ --cov=backend

# Run frontend tests only
cd frontend/web && npm test

# Run specific test categories
pytest backend/tests/ -m "unit"        # Unit tests
pytest backend/tests/ -m "integration" # Integration tests
pytest backend/tests/ -m "financial"   # Financial model tests
```

## Deployment

### Development Deployment
```bash
./setup.sh      # Initial setup
./start.sh dev  # Start development stack
```

### Production Deployment
```bash
./setup.sh         # Initial setup
./start.sh prod    # Start production stack
```

### Environment-Specific Features

**Development**:
- Hot reloading for backend and frontend
- Debug tools (PgAdmin, Redis Commander, Flower)
- Detailed logging
- Source code mounting

**Production**:
- Optimized builds
- Security headers
- Resource limits
- Backup services
- Monitoring alerts

## Security Considerations

- OAuth2 authentication for user endpoints
- API keys stored in environment variables
- Rate limiting per user/IP
- Data anonymization for GDPR compliance
- Audit logging for SEC requirements
- SSL/TLS encryption in production

## Performance Optimization

### Database Optimization
- TimescaleDB for time-series data
- Proper indexing strategies
- Connection pooling
- Query optimization

### Caching Strategy
- Redis for API responses
- Multi-layer caching (L1: Memory, L2: Redis, L3: Database)
- Smart cache invalidation

### API Optimization
- Batch processing
- Asynchronous operations
- Rate limiting and throttling
- Connection pooling

This documentation reflects the current simplified architecture after refactoring to remove complexity while maintaining functionality.