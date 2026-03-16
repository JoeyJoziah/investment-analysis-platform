# Consolidated Requirements Files Content

This document contains the exact content for the new consolidated requirements files.
To be created after user approval of the cleanup plan.

---

## File 1: requirements.txt (Production Dependencies)

```txt
# Investment Analysis Platform - Production Dependencies
# Python 3.12+ compatible
# Last updated: 2026-01-27

# ============================================================================
# CORE FRAMEWORK
# ============================================================================
fastapi>=0.115.0
uvicorn[standard]==0.30.1
pydantic==2.8.2
pydantic-settings==2.4.0

# ============================================================================
# DATABASE
# ============================================================================
sqlalchemy[asyncio]==2.0.31
asyncpg==0.29.0
psycopg2-binary==2.9.9
alembic==1.13.2

# ============================================================================
# CACHING & MESSAGING
# ============================================================================
redis==5.0.7
celery==5.4.0

# ============================================================================
# DATA PROCESSING
# ============================================================================
pandas>=2.0.0,<3.0.0
numpy>=1.24.0,<2.0.0
scipy>=1.11.0

# ============================================================================
# FINANCIAL DATA & ANALYSIS
# ============================================================================
alpha-vantage==2.3.1
finnhub-python==2.4.20
polygon-api-client==1.14.1
yfinance==0.2.40
ta-lib==0.4.32
statsmodels==0.14.2

# ============================================================================
# WEB & API CLIENTS
# ============================================================================
aiohttp==3.10.2
httpx==0.27.0
requests==2.32.3
beautifulsoup4==4.12.3
lxml==5.2.2

# ============================================================================
# SECURITY & AUTHENTICATION
# ============================================================================
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
PyJWT==2.9.0
cryptography==43.0.0

# ============================================================================
# DATA PIPELINE & ORCHESTRATION
# ============================================================================
confluent-kafka==2.5.0
aiokafka==0.11.0

# ============================================================================
# NLP & SENTIMENT ANALYSIS
# ============================================================================
nltk==3.9.1
textblob==0.18.0
tokenizers==0.19.1

# ============================================================================
# MONITORING & OBSERVABILITY
# ============================================================================
prometheus-client==0.20.0
opentelemetry-api==1.26.0
opentelemetry-sdk==1.26.0
sentry-sdk[fastapi]==2.12.0
structlog==24.4.0

# ============================================================================
# UTILITIES & HELPERS
# ============================================================================
python-dotenv==1.0.1
python-multipart==0.0.9
pytz==2024.1
schedule==1.2.2
tqdm==4.66.5
backoff==2.2.1
tenacity>=8.1.0,<9.0.0
cachetools==5.4.0
aiofiles==24.1.0
psutil==6.0.0
pyotp>=2.9.0
qrcode>=7.4.0
PyYAML>=6.0
validators>=0.22.0
bleach>=6.1.0
rich>=13.7.0
typer>=0.12.0
python-dateutil>=2.8.2
pybreaker>=1.2.0
greenlet>=3.0.0

# ============================================================================
# MACHINE LEARNING (CORE)
# ============================================================================
scikit-learn==1.5.1
xgboost==2.1.1
lightgbm>=4.0.0
prophet==1.1.5

# ============================================================================
# TRADING AGENTS & LLM INTEGRATION
# ============================================================================
langchain-anthropic==0.1.23
langchain-openai==0.1.23
langgraph==0.2.16

# ============================================================================
# DEPLOYMENT & PRODUCTION
# ============================================================================
gunicorn==22.0.0
docker==7.1.0
websockets>=10.3,<13.0
email-validator>=2.0.0
```

---

## File 2: requirements-dev.txt (Development Dependencies)

```txt
# Investment Analysis Platform - Development Dependencies
# Testing, linting, formatting, and development tools
# Only needed in development environments
# Python 3.12+ compatible
# Last updated: 2026-01-27

# Include production dependencies
-r requirements.txt

# ============================================================================
# TESTING FRAMEWORK
# ============================================================================
pytest==8.3.2
pytest-asyncio==0.23.8
pytest-cov==5.0.0
pytest-mock==3.14.0
faker==28.4.1
testcontainers>=4.0.0
requests-mock>=1.12.0

# ============================================================================
# CODE QUALITY & FORMATTING
# ============================================================================
black==24.8.0
isort==5.13.2
flake8==7.1.1
mypy==1.11.1

# ============================================================================
# SECURITY SCANNING
# ============================================================================
bandit==1.7.9

# ============================================================================
# DEBUGGING & PROFILING
# ============================================================================
memory-profiler>=0.61.0
objgraph>=3.6.0

# ============================================================================
# TEST UTILITIES
# ============================================================================
itsdangerous==2.2.0
aiosqlite>=0.20.0
sqlparse>=0.5.0
```

---

## File 3: requirements-ml.txt (ML-Specific Heavy Dependencies)

```txt
# Investment Analysis Platform - Machine Learning Dependencies
# Heavy ML frameworks and related packages
# Requires system dependencies: build-essential, python3-dev
# Note: These packages are large (~2-5GB) and may take significant time to install
# Python 3.12+ compatible
# Last updated: 2026-01-27

# Include production dependencies
-r requirements.txt

# ============================================================================
# DEEP LEARNING FRAMEWORKS
# ============================================================================
# PyTorch ecosystem (large download ~2GB)
# GPU support: pip install torch --index-url https://download.pytorch.org/whl/cu118
torch==2.4.0
transformers==4.43.3
huggingface_hub>=0.20.0
datasets>=2.16.0

# ============================================================================
# MODEL INTERPRETABILITY
# ============================================================================
shap==0.46.0
lime==0.2.0.1

# ============================================================================
# HYPERPARAMETER TUNING
# ============================================================================
optuna==3.6.1

# ============================================================================
# VISUALIZATION
# ============================================================================
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0

# ============================================================================
# WEB SCRAPING (for data collection)
# ============================================================================
selenium==4.18.1
webdriver-manager==4.0.1
html5lib>=1.1

# ============================================================================
# ADDITIONAL DATA SOURCES
# ============================================================================
boto3>=1.34.0
```

---

## Migration Notes

### Version Resolution Decisions

| Package | Chosen Version | Reason |
|---------|---------------|--------|
| `numpy` | `>=1.24.0,<2.0.0` | Many packages incompatible with numpy 2.x |
| `pandas` | `>=2.0.0,<3.0.0` | Flexible range for compatibility |
| `fastapi` | `>=0.115.0` | Latest stable with security fixes |
| `pydantic` | `2.8.2` | Known compatible with fastapi 0.115 |
| `torch` | `2.4.0` | Stable release, good transformers compatibility |
| `transformers` | `4.43.3` | Compatible with torch 2.4.0 |

### Removed Packages (Commented in Original)

- `pandas-ta` - Removed due to dependency conflicts with numpy 2.x
- `elasticsearch` - Using PostgreSQL full-text search instead
- `apache-airflow` - Runs in separate container with its own environment
- `aioredis` - Deprecated, using redis.asyncio instead

### Special Notes

1. **ta-lib**: Requires system library installation (`brew install ta-lib` on macOS)
2. **torch GPU**: For CUDA support, install with: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
3. **LightGBM GPU**: Requires OpenCL and compilation with GPU flag
