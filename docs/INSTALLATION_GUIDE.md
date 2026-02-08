# Installation Guide - Investment Analysis Platform

**Last Updated**: 2026-01-29
**Version**: 1.0.0
**Platform**: macOS, Linux, Windows (WSL2)
**Python**: 3.12+
**Node.js**: 18+

---

## Quick Start (5 Minutes)

```bash
# 1. Clone repository
git clone https://github.com/yourusername/investment-analysis-platform.git
cd investment-analysis-platform

# 2. Run setup script (automated, interactive)
./setup.sh

# 3. Start development environment
./start.sh dev

# 4. Access the application
# Frontend: http://localhost:3000
# API Docs: http://localhost:8000/docs
# Grafana: http://localhost:3001
```

---

## Prerequisites

### System Requirements
- **CPU**: 4+ cores recommended
- **RAM**: 8GB minimum, 16GB recommended
- **Disk**: 50GB free space (for Docker images and data)
- **Network**: Internet connection (for API calls and dependencies)

### Software Requirements

#### macOS
```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install required tools
brew install python@3.12 node git docker docker-compose postgresql redis

# Verify installations
python3 --version  # Should be 3.12+
node --version     # Should be 18+
docker --version
docker-compose --version
```

#### Linux (Ubuntu/Debian)
```bash
# Update package manager
sudo apt-get update
sudo apt-get upgrade -y

# Install required tools
sudo apt-get install -y \
  python3.12 \
  python3.12-venv \
  python3-pip \
  nodejs \
  npm \
  git \
  docker.io \
  docker-compose \
  postgresql-client

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify installations
python3.12 --version
node --version
docker --version
```

#### Windows (WSL2)
```bash
# Install WSL2 (Windows Store or PowerShell as Admin)
wsl --install

# In WSL2 terminal:
sudo apt-get update
sudo apt-get upgrade -y

# Install required tools
sudo apt-get install -y \
  python3.12 \
  python3.12-venv \
  python3-pip \
  nodejs \
  npm \
  git \
  docker.io

# Add user to docker group
sudo usermod -aG docker $USER
```

---

## Detailed Installation Steps

### Step 1: Clone Repository

```bash
# HTTPS (recommended for first time)
git clone https://github.com/yourusername/investment-analysis-platform.git

# OR SSH (if you have SSH key configured)
git clone git@github.com:yourusername/investment-analysis-platform.git

cd investment-analysis-platform
```

### Step 2: Environment Setup

#### Option A: Automated Setup (Recommended)
```bash
# Run interactive setup script
./setup.sh

# This will:
# ✓ Check prerequisites
# ✓ Create .env file with secure credentials
# ✓ Generate encryption keys
# ✓ Setup database
# ✓ Initialize Redis
# ✓ Install Python dependencies
# ✓ Install Node.js dependencies
# ✓ Build Docker images
# ✓ Verify all services
```

#### Option B: Manual Setup

**2.1 Create Environment File**
```bash
# Copy template
cp .env.example .env

# Edit with your values
nano .env  # or vim, code, etc.

# Required values:
# - SECRET_KEY (generate: python -c "import secrets; print(secrets.token_hex(32))")
# - JWT_SECRET_KEY (same as above)
# - FERNET_KEY (generate: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
# - Financial API keys (Alpha Vantage, Finnhub, Polygon, NewsAPI)
```

**2.2 Python Environment**
```bash
# Create virtual environment
python3.12 -m venv venv

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt

# Verify installation
python -c "import fastapi; import sqlalchemy; print('✓ Core packages installed')"
```

**2.3 Node.js Dependencies**
```bash
# Install frontend dependencies
cd frontend/web
npm install

# Verify installation
npm --version
node_modules/.bin/react-scripts --version

cd ../..
```

**2.4 Database Setup**
```bash
# Docker PostgreSQL (if not using external database)
docker run --name investment-db \
  -e POSTGRES_DB=investment_db \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=$DB_PASSWORD \
  -p 5432:5432 \
  -d postgres:15

# Wait for database to start
sleep 10

# Initialize database
python -m alembic upgrade head
```

**2.5 Redis Setup**
```bash
# Docker Redis
docker run --name investment-redis \
  -e REDIS_PASSWORD=$REDIS_PASSWORD \
  -p 6379:6379 \
  -d redis:7

# Test connection
redis-cli -p 6379 ping
# Should output: PONG
```

### Step 3: Start Services

#### Development Mode
```bash
# Using provided script (recommended)
./start.sh dev

# OR manually start each service

# Terminal 1: Backend API
source venv/bin/activate
uvicorn backend.api.main:app --reload --port 8000

# Terminal 2: Frontend
cd frontend/web
npm start  # Starts on http://localhost:3000

# Terminal 3: Celery Worker (for background tasks)
source venv/bin/activate
celery -A backend.tasks.celery_app worker -l info

# Terminal 4: Celery Beat (for scheduled tasks)
celery -A backend.tasks.celery_app beat -l info
```

#### Production Mode
```bash
# Configure SSL first (if needed)
./scripts/init-ssl.sh yourdomain.com admin@yourdomain.com

# Start production environment
./start.sh prod

# This starts:
# ✓ FastAPI with Gunicorn
# ✓ React with Nginx
# ✓ PostgreSQL
# ✓ Redis
# ✓ Celery Worker & Beat
# ✓ Prometheus & Grafana
# ✓ AlertManager
```

### Step 4: Verify Installation

```bash
# Check backend health
curl http://localhost:8000/health

# Check frontend
open http://localhost:3000  # macOS
xdg-open http://localhost:3000  # Linux
start http://localhost:3000  # Windows

# Check API documentation
open http://localhost:8000/docs

# Check database connection
python -c "from backend.config.database import SessionLocal; \
  session = SessionLocal(); \
  result = session.execute('SELECT 1'); \
  print('✓ Database connected')"

# Check Redis connection
redis-cli ping

# Run test suite
./start.sh test
# Or manually:
pytest backend/tests/ --cov=backend -v
```

---

## Service URLs (Development)

| Service | URL | Purpose | Username | Password |
|---------|-----|---------|----------|----------|
| Frontend | http://localhost:3000 | Web application | N/A | N/A |
| API | http://localhost:8000 | REST API | N/A | N/A |
| API Docs | http://localhost:8000/docs | Swagger UI | N/A | N/A |
| ML Service | http://localhost:8001 | ML predictions | N/A | N/A |
| Grafana | http://localhost:3001 | Dashboards | admin | admin |
| Prometheus | http://localhost:9090 | Metrics | N/A | N/A |
| PostgreSQL | localhost:5432 | Database | postgres | (from .env) |
| Redis | localhost:6379 | Cache | (none) | (from .env) |

---

## Installation Troubleshooting

### Python Version Issues

**Problem**: `python3 --version` shows < 3.12
```bash
# Solution: Install Python 3.12
# macOS
brew install python@3.12
python3.12 --version

# Linux
sudo apt-get install python3.12
python3.12 --version

# Windows
# Download from python.org and install, or use WSL2
```

### Dependency Conflicts

**Problem**: `pip install -r requirements.txt` fails
```bash
# Solution: Create fresh virtual environment
rm -rf venv
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

### Docker Issues

**Problem**: Docker daemon not running
```bash
# Solution: Start Docker
# macOS
open -a Docker

# Linux
sudo systemctl start docker

# Windows (WSL2)
sudo service docker start
```

### Database Connection Issues

**Problem**: PostgreSQL connection refused
```bash
# Solution: Check database is running
docker ps | grep postgres

# If not running, start it
docker run --name investment-db \
  -e POSTGRES_DB=investment_db \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=$DB_PASSWORD \
  -p 5432:5432 \
  -d postgres:15
```

### Port Already in Use

**Problem**: Address already in use (port 3000, 8000, etc.)
```bash
# Find process using port
lsof -i :3000  # macOS/Linux
netstat -ano | findstr :3000  # Windows

# Kill process
kill -9 <PID>  # macOS/Linux
taskkill /PID <PID> /F  # Windows

# OR use different port
uvicorn backend.api.main:app --port 8001
```

### NPM Package Issues

**Problem**: npm install fails
```bash
# Solution: Clear cache and reinstall
rm -rf node_modules package-lock.json
npm cache clean --force
npm install

# OR use alternative registry
npm install --registry https://registry.npmmirror.com
```

### Frontend Not Loading

**Problem**: Blank page at localhost:3000
```bash
# Solution: Check console for errors
# 1. Open browser DevTools (F12)
# 2. Check Console tab for errors
# 3. Check Network tab for failed requests

# Restart frontend
cd frontend/web
npm start

# Or rebuild from scratch
rm -rf node_modules
npm install
npm start
```

---

## API Keys & Credentials

### Financial Data APIs

#### Alpha Vantage
1. Go to https://www.alphavantage.co/
2. Click "GET FREE API KEY"
3. Enter your email
4. Check email and copy API key
5. Add to `.env`: `ALPHA_VANTAGE_API_KEY=your_key`

#### Finnhub
1. Go to https://finnhub.io/register
2. Create account
3. Copy API key from dashboard
4. Add to `.env`: `FINNHUB_API_KEY=your_key`

#### Polygon.io
1. Go to https://polygon.io/dashboard/signup
2. Create account
3. Copy API key from dashboard
4. Add to `.env`: `POLYGON_API_KEY=your_key`

#### NewsAPI
1. Go to https://newsapi.org/register
2. Create account
3. Copy API key from dashboard
4. Add to `.env`: `NEWS_API_KEY=your_key`

### Optional AI APIs

#### OpenAI (for ChatGPT features)
1. Go to https://platform.openai.com/api-keys
2. Create new API key
3. Add to `.env`: `OPENAI_API_KEY=sk-...`

#### Anthropic Claude (for advanced AI)
1. Go to https://console.anthropic.com/
2. Create new API key
3. Add to `.env`: `ANTHROPIC_API_KEY=sk-ant-...`

---

## Docker Compose Setup

Alternative to manual setup using Docker Compose:

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop all services
docker-compose down

# Full reset (removes volumes)
docker-compose down -v
```

### docker-compose.yml Structure
```yaml
services:
  postgres:        # Database
  redis:           # Cache/Celery broker
  backend:         # FastAPI application
  frontend:        # React application
  celery-worker:   # Background tasks
  celery-beat:     # Scheduled tasks
  prometheus:      # Metrics (production)
  grafana:         # Dashboards (production)
  alertmanager:    # Alerting (production)
```

---

## Production Deployment

### Pre-Deployment Checklist
- [ ] All environment variables set
- [ ] SSL/TLS certificates obtained
- [ ] Database backed up
- [ ] Security audit passed
- [ ] Tests pass (100%)
- [ ] Load testing completed
- [ ] Monitoring configured
- [ ] Incident response plan ready

### Deployment Steps

```bash
# 1. Configure SSL/TLS
./scripts/init-ssl.sh yourdomain.com admin@yourdomain.com

# 2. Update environment variables for production
# Edit .env and set:
# - ENVIRONMENT=production
# - DEBUG=false
# - SESSION_COOKIE_SECURE=true
# - FORCE_HTTPS=true

# 3. Build Docker images
docker-compose build --no-cache

# 4. Start production environment
./start.sh prod

# 5. Monitor services
docker-compose logs -f

# 6. Verify health
curl https://yourdomain.com/health
```

---

## Maintenance & Updates

### Regular Maintenance
```bash
# Update dependencies
pip install -r requirements.txt --upgrade
npm update

# Database backups
pg_dump -U postgres -d investment_db > backup.sql

# Clear old logs
find logs -name "*.log" -mtime +30 -delete

# Update Docker images
docker-compose pull
docker-compose up -d
```

### Version Updates
```bash
# Check for updates
git fetch origin
git status

# Pull latest changes
git pull origin main

# Run migrations
python -m alembic upgrade head

# Restart services
./stop.sh
./start.sh prod
```

---

## Getting Help

### Documentation
- [README.md](../README.md) - Project overview
- [CLAUDE.md](../CLAUDE.md) - Development guidelines
- [docs/SECURITY.md](SECURITY.md) - Security guidelines
- [docs/ENVIRONMENT.md](ENVIRONMENT.md) - Configuration reference

### Support Channels
- GitHub Issues: https://github.com/yourusername/investment-analysis-platform/issues
- Discussions: https://github.com/yourusername/investment-analysis-platform/discussions
- Email: support@yourdomain.com

### Common Issues
- See [Troubleshooting](#installation-troubleshooting) section above
- Check Docker logs: `docker-compose logs -f`
- Check application logs: `tail -f logs/app.log`

---

## Next Steps

1. **Read Documentation**
   - [README.md](../README.md) - Project overview
   - [CLAUDE.md](../CLAUDE.md) - Development guidelines

2. **Configure API Keys**
   - Add financial data API keys to `.env`
   - Test API connections

3. **Run Tests**
   - Backend: `pytest backend/tests/ --cov=backend`
   - Frontend: `npm test`

4. **Start Development**
   - Create feature branch: `git checkout -b feature/your-feature`
   - Write tests first (TDD)
   - Submit pull request

---

*Installation Guide - Last Updated: 2026-01-29*
*Version: 1.0.0*
*Status: Production Ready*
