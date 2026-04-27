> **ARCHIVED 2026-04-27 by 17-scripts-tooling**
> Original: docs/INSTALLATION_GUIDE.md
> Validation summary: 8/10 claims still current.
> See `../../reports/17-scripts-tooling.md` §2 for per-claim status.

# Installation Guide

**Last Updated**: 2026-03-04
**Platform**: macOS, Linux, Windows (WSL2)
**Python**: 3.12+
**Node.js**: 18+

---

## Quick Start (5 Minutes)

```bash
# 1. Clone repository
git clone https://github.com/yourusername/investment-analysis-platform.git
cd investment-analysis-platform

# 2. Run setup script
./setup.sh

# 3. Start development environment
./start.sh dev

# 4. Access the application
# Frontend: http://localhost:3000
# API Docs: http://localhost:8000/docs
# Grafana:  http://localhost:3001
```

---

## Prerequisites

### System Requirements
- **CPU**: 4+ cores recommended
- **RAM**: 8GB minimum, 16GB recommended
- **Disk**: 50GB free space (Docker images and database)
- **Network**: Internet connection required for API calls and dependencies

### Software Requirements

#### macOS
```bash
# Install Homebrew if not present
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install required tools
brew install python@3.12 node git
brew install --cask docker  # Docker Desktop

# Verify
python3.12 --version  # 3.12+
node --version        # 18+
docker --version
docker compose version
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update && sudo apt-get upgrade -y

sudo apt-get install -y \
  python3.12 python3.12-venv python3-pip \
  nodejs npm git

# Docker Engine (not docker.io)
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker
```

#### Windows (WSL2)
```bash
# In PowerShell (Admin):
wsl --install

# Then in your WSL2 terminal, follow the Linux (Ubuntu) steps above
```

---

## Detailed Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/investment-analysis-platform.git
cd investment-analysis-platform
```

### Step 2: Environment Setup

#### Option A: Automated Setup (Recommended)

```bash
./setup.sh
# Checks prerequisites, generates .env with secure credentials,
# installs dependencies, builds Docker images, verifies services.
```

#### Option B: Manual Setup

**2.1 Create `.env` file**

```bash
cp .env.example .env
# Edit .env with your values (see docs/ENVIRONMENT.md for full reference)
```

Minimum required values before first run:

```bash
SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
JWT_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
MASTER_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(64))")
FERNET_KEY=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
GDPR_ENCRYPTION_KEY=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
DB_PASSWORD=$(python3 -c "import secrets; print(secrets.token_urlsafe(24))")
REDIS_PASSWORD=$(python3 -c "import secrets; print(secrets.token_urlsafe(24))")
```

Also add your financial API keys:
- `ALPHA_VANTAGE_API_KEY` - [alphavantage.co](https://www.alphavantage.co/support/#api-key)
- `FINNHUB_API_KEY` - [finnhub.io](https://finnhub.io/register)
- `POLYGON_API_KEY` - [polygon.io](https://polygon.io/dashboard/signup)
- `NEWS_API_KEY` - [newsapi.org](https://newsapi.org/register)

**2.2 Python virtual environment (for local development without Docker)**

```bash
python3.12 -m venv venv
source venv/bin/activate   # macOS/Linux
pip install --upgrade pip
pip install -r requirements.txt
```

**2.3 Frontend dependencies**

```bash
cd frontend/web
npm install
cd ../..
```

### Step 3: Start Services

#### Development Mode (Docker Compose)

```bash
./start.sh dev
# OR
docker compose up -d
```

#### Running services individually (without Docker)

```bash
# Terminal 1: Backend API
source venv/bin/activate
uvicorn backend.api.main:app --reload --port 8000

# Terminal 2: Frontend (Vite dev server)
cd frontend/web
npm run dev   # starts on http://localhost:3000

# Terminal 3: Celery worker
source venv/bin/activate
celery -A backend.tasks.celery_app worker -l info

# Terminal 4: Celery beat (scheduler)
source venv/bin/activate
celery -A backend.tasks.celery_app beat -l info
```

#### Production Mode

```bash
# Configure SSL first
./scripts/init-ssl.sh yourdomain.com admin@yourdomain.com

# Start production stack
./start.sh prod
```

### Step 4: Verify Installation

```bash
# Backend health
curl http://localhost:8000/api/health
# Expected: {"status": "healthy", ...}

# Open frontend
open http://localhost:3000       # macOS
xdg-open http://localhost:3000   # Linux

# Run the test suite
pytest backend/tests/ -m "not slow" --tb=short
```

---

## Service URLs (Development)

| Service | URL | Credentials |
|---------|-----|-------------|
| Frontend | http://localhost:3000 | - |
| API | http://localhost:8000 | - |
| API Docs (Swagger) | http://localhost:8000/docs | - |
| Grafana | http://localhost:3001 | admin / admin |
| Prometheus | http://localhost:9090 | - |
| PostgreSQL | localhost:5432 | postgres / (from .env) |
| Redis | localhost:6379 | (from .env) |

---

## Docker Compose Overview

| File | Purpose |
|------|---------|
| `docker-compose.yml` | Base service definitions (dev/test) |
| `docker-compose.dev.yml` | Development overrides (hot reload, mounted source) |
| `docker-compose.production.yml` | Production overrides (SSL, resource limits, monitoring) |
| `docker-compose.test.yml` | Test environment (isolated DB, mocked APIs) |

Production stack is started via the overlay:
```bash
docker compose -f docker-compose.production.yml up -d
```

### Production Docker Services

- PostgreSQL 15 + TimescaleDB
- Redis 7 (cache, Celery broker DB 0, Celery results DB 1)
- FastAPI backend (Gunicorn)
- React frontend (Nginx static serving)
- Celery worker and beat scheduler
- Nginx (TLS termination, reverse proxy)
- Certbot (automatic Let's Encrypt renewal)
- Prometheus + VictoriaMetrics (metrics with 90-day retention)
- Grafana (dashboards)
- Loki + Promtail (log aggregation)
- AlertManager (alert routing)

---

## Getting API Keys

### Alpha Vantage
1. Go to https://www.alphavantage.co/
2. Click "GET FREE API KEY"
3. Enter your email and copy the key
4. Add to `.env`: `ALPHA_VANTAGE_API_KEY=your_key`

### Finnhub
1. Go to https://finnhub.io/register
2. Create account and copy API key from dashboard
3. Add to `.env`: `FINNHUB_API_KEY=your_key`

### Polygon.io
1. Go to https://polygon.io/dashboard/signup
2. Create account and copy API key
3. Add to `.env`: `POLYGON_API_KEY=your_key`

### NewsAPI
1. Go to https://newsapi.org/register
2. Create account and copy API key
3. Add to `.env`: `NEWS_API_KEY=your_key`

---

## Installation Troubleshooting

### Python version < 3.12

```bash
# macOS
brew install python@3.12
python3.12 --version

# Ubuntu
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get install python3.12 python3.12-venv
```

### Dependency conflicts

```bash
rm -rf venv
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

### Docker daemon not running

```bash
# macOS
open -a Docker

# Linux
sudo systemctl start docker

# WSL2
sudo service docker start
```

### Port already in use

```bash
# Find the process using the port
lsof -i :8000     # macOS/Linux
# Kill it
kill -9 <PID>
```

### Backend starts then exits immediately

Check for missing required environment variables:

```bash
docker compose logs backend | head -30
```

The most common cause is a missing `GDPR_ENCRYPTION_KEY`. See [docs/TROUBLESHOOTING.md](TROUBLESHOOTING.md) Issue 2.

### npm install fails

```bash
cd frontend/web
rm -rf node_modules package-lock.json
npm cache clean --force
npm install
```

---

## Maintenance & Updates

### Update dependencies

```bash
# Python
pip install -r requirements.txt --upgrade

# Node
cd frontend/web && npm update
```

### Run database migrations

```bash
# With Docker
docker compose exec backend python -m alembic upgrade head

# Without Docker
source venv/bin/activate
python -m alembic upgrade head
```

### Pull latest code

```bash
git pull origin main
python -m alembic upgrade head
./stop.sh && ./start.sh prod
```

---

## Documentation

- [README.md](../README.md) - Project overview
- [docs/ENVIRONMENT.md](ENVIRONMENT.md) - All environment variables
- [docs/DEPLOYMENT.md](DEPLOYMENT.md) - Production deployment
- [docs/TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues
- [CLAUDE.md](../CLAUDE.md) - Development guidelines

---

*Last Updated: 2026-03-04*
