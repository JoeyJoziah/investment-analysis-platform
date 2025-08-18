# 📈 Investment Analysis Platform

A comprehensive, AI-powered investment analysis and recommendation platform designed to analyze 6,000+ publicly traded stocks from NYSE, NASDAQ, and AMEX exchanges.

## 🚀 Quick Start

```bash
# 1. Initial setup (run once)
./setup.sh

# 2. Start development environment
./start.sh dev

# 3. Access the application
# Frontend: http://localhost:3000
# API Docs: http://localhost:8000/docs
```

## 📋 Features

- **Real-time Stock Analysis**: Technical, fundamental, and sentiment analysis
- **AI-Powered Recommendations**: ML models including LSTM, XGBoost, and Prophet
- **Portfolio Management**: Track and optimize investment portfolios
- **Cost Optimized**: Designed to run under $50/month using free API tiers
- **Fully Automated**: Daily analysis without manual intervention
- **Compliance Ready**: GDPR and SEC compliant architecture

## 🏗️ Architecture

```
├── backend/           # FastAPI backend with ML models
├── frontend/          # React web application
├── data_pipelines/    # Apache Airflow DAGs
├── infrastructure/    # Docker and Kubernetes configs
├── scripts/          # Utility scripts
├── models/           # Trained ML models
└── tests/            # Test suites
```

## 🛠️ Technology Stack

- **Backend**: FastAPI, Python 3.11
- **Frontend**: React, TypeScript, Material-UI
- **Database**: PostgreSQL, TimescaleDB
- **Cache**: Redis
- **ML/AI**: PyTorch, scikit-learn, Prophet
- **Orchestration**: Apache Airflow
- **Container**: Docker, Kubernetes

## 📝 Available Commands

### Using Make
```bash
make help          # Show all available commands
make setup         # Initial project setup
make up            # Start development environment
make down          # Stop all services
make test          # Run tests
make logs          # View logs
make clean         # Clean up everything
```

### Using Scripts
```bash
./setup.sh         # Initial setup with secure passwords
./start.sh dev     # Start development environment
./start.sh prod    # Start production environment
./start.sh test    # Run tests
./stop.sh          # Stop all services
./stop.sh --clean  # Stop and clean volumes
./logs.sh          # View all logs
./logs.sh backend  # View specific service logs
```

## 🔧 Development

### Backend Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn backend.api.main:app --reload

# Run tests
pytest backend/tests/

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

## 🗄️ Database

### Migrations
```bash
# Run migrations
make db-migrate

# Rollback migration
make db-rollback

# Access database
make db-shell
```

## 📊 API Endpoints

- `GET /api/health` - Health check
- `GET /api/stocks/{ticker}` - Stock data and analysis
- `GET /api/recommendations` - AI recommendations
- `POST /api/portfolio` - Portfolio management
- `GET /api/analysis/{ticker}` - Detailed analysis
- `WS /ws` - WebSocket for real-time updates

## 🔐 Environment Variables

Copy `.env.template` to `.env` and update with your API keys:

```bash
# Required API Keys (free tiers available)
ALPHA_VANTAGE_API_KEY=your_key
FINNHUB_API_KEY=your_key
POLYGON_API_KEY=your_key
NEWS_API_KEY=your_key

# Generated automatically by setup.sh
DB_PASSWORD=auto_generated
REDIS_PASSWORD=auto_generated
SECRET_KEY=auto_generated
JWT_SECRET_KEY=auto_generated
```

## 📈 Cost Optimization

The platform is designed to operate under $50/month by:
- Using free API tiers effectively
- Intelligent caching strategies
- Batch processing during off-peak hours
- Auto-scaling down during idle periods

## 🧪 Testing

```bash
# Run all tests
make test

# Run with coverage
pytest --cov=backend --cov-report=html

# View coverage report
open htmlcov/index.html
```

## 🚀 Deployment

### Production Deployment
```bash
# Start production environment
./start.sh prod

# Monitor services
docker-compose logs -f

# Access monitoring
http://localhost:3001  # Grafana dashboard
```

### Kubernetes Deployment
```bash
# Apply configurations
kubectl apply -f infrastructure/kubernetes/

# Check status
kubectl get pods -n investment-platform
```

## 📚 Documentation

- [API Documentation](http://localhost:8000/docs) - Interactive API docs
- [Architecture Guide](docs/architecture/README.md) - System architecture
- [Development Guide](docs/guides/development.md) - Development practices
- [Deployment Guide](docs/deployment/README.md) - Deployment instructions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Alpha Vantage for market data
- Finnhub for real-time quotes
- Polygon.io for historical data
- NewsAPI for sentiment analysis

---

**Built with ❤️ for automated investment analysis**