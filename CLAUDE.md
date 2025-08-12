# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an investment analysis and recommendation application designed to analyze 6,000+ publicly traded stocks from NYSE, NASDAQ, and AMEX exchanges. The system operates autonomously, generating daily recommendations without user input, while supporting optional customizations.

Key requirements:
- Target operational cost: under $50/month
- Must use free/open-source tools and APIs with generous free tiers
- Fully automated daily analysis without manual intervention
- Compliance with 2025 SEC and GDPR regulations

## 🤖 AUTOMATIC AGENT USAGE PROTOCOL

**CRITICAL REQUIREMENT**: All prompts to Claude Code MUST automatically invoke appropriate specialized agents unless the user explicitly states "no agents" or specifies a different agent. This is a mandatory requirement for all interactions with this project.

### Agent Selection Rules

Claude Code will automatically analyze each user prompt and:

1. **Always assess task complexity** to determine if single agent or multi-agent approach is needed
2. **Automatically select and invoke appropriate agents** based on the task requirements
3. **Use the agent-organizer** for complex multi-domain tasks requiring coordination
4. **Default to specialized agents** over generic responses for all technical tasks

### Automatic Agent Invocation Categories

#### 🐍 Python Backend Development (Auto-invoke: `python-pro`)
**Triggers**: Python code, FastAPI, backend logic, API endpoints, data processing, ML implementation
- Any Python code writing, review, or optimization
- FastAPI endpoint development or modification  
- Backend architecture or design patterns
- Async/await implementation
- Python testing or debugging

#### 📊 Financial & Data Analysis (Auto-invoke: `quant-analyst` + `data-scientist`)
**Triggers**: Financial models, stock analysis, trading algorithms, market data, SQL queries
- Financial model development or backtesting
- Trading strategy implementation
- Risk metrics calculation (VaR, Sharpe ratio)
- Portfolio optimization
- Market data analysis or SQL queries
- Time series forecasting

#### 🏗️ Database & Schema Design (Auto-invoke: `database-schema-designer` + `postgres-pro`)
**Triggers**: Database schema, migrations, queries, PostgreSQL, TimescaleDB
- Database schema design or modifications
- Alembic migrations
- PostgreSQL performance optimization
- Query optimization or indexing
- Database architecture decisions

#### 🤖 Machine Learning (Auto-invoke: `ml-engineer` + `ai-engineer`)
**Triggers**: ML models, PyTorch, scikit-learn, AI features, model training
- ML model development or optimization
- PyTorch or scikit-learn implementation
- Feature engineering or selection
- Model training pipelines
- AI-powered features or RAG systems

#### 🔒 Security & Compliance (Auto-invoke: `security-auditor`)
**Triggers**: Security, authentication, GDPR, SEC compliance, vulnerabilities
- Security vulnerability assessment
- Authentication system implementation
- Compliance requirement implementation
- Code security review
- OWASP compliance checks

#### ⚡ Performance & Optimization (Auto-invoke: `performance-optimizer`)
**Triggers**: Performance issues, optimization, bottlenecks, caching, scaling
- Performance bottleneck analysis
- Code optimization strategies
- Caching implementation
- Database query optimization
- System performance tuning

#### 🧪 Testing & Quality (Auto-invoke: `test-suite-generator` + `code-reviewer`)
**Triggers**: Testing, pytest, quality assurance, code review, debugging
- Test suite creation or enhancement
- Pytest implementation
- Code quality review
- Debugging assistance
- Quality assurance processes

#### 🚀 DevOps & Deployment (Auto-invoke: `deployment-engineer` + `cloud-architect`)
**Triggers**: Docker, Kubernetes, CI/CD, deployment, infrastructure, AWS
- Docker or Kubernetes configuration
- CI/CD pipeline setup or modification
- Infrastructure as code
- Cloud deployment strategies
- Container orchestration

#### ⚛️ Frontend Development (Auto-invoke: `frontend-developer` + `react-pro`)
**Triggers**: React, frontend, UI, components, JavaScript, TypeScript
- React component development
- Frontend architecture decisions
- UI/UX implementation
- JavaScript/TypeScript code
- Frontend testing or optimization

#### 📋 API Design & Documentation (Auto-invoke: `api-designer` + `api-documenter`)
**Triggers**: API endpoints, OpenAPI, Swagger, API documentation, REST design
- API endpoint design or modification
- OpenAPI/Swagger documentation
- REST API architecture
- API testing or validation
- Integration documentation

#### 🗄️ Data Engineering (Auto-invoke: `data-engineer`)
**Triggers**: ETL pipelines, Airflow, data processing, streaming, data warehouses
- ETL pipeline development
- Airflow DAG creation or modification
- Data streaming implementation
- Data warehouse design
- Data processing optimization

### 🎯 Multi-Agent Coordination Rules

#### Complex Multi-Domain Tasks (Auto-invoke: `agent-organizer`)
**Always use agent-organizer for tasks involving:**
- Multiple technology domains (e.g., backend + frontend + database)
- System-wide architectural changes
- Cross-functional requirements (security + performance + testing)
- New feature implementation requiring multiple specialists
- Major refactoring or modernization efforts
- Complete component development (API + tests + docs + security)

#### Single-Domain Tasks (Direct Agent Invocation)
**Use specific agents directly for:**
- Focused single-technology tasks
- Code optimization within one domain
- Specific debugging or troubleshooting
- Single-component modifications
- Documentation-only tasks

### 📝 Task-to-Agent Mapping Matrix

| Task Type | Primary Agent | Secondary Agents | When to Use agent-organizer |
|-----------|--------------|------------------|---------------------------|
| Python Backend Development | `python-pro` | `test-suite-generator`, `security-auditor` | When involves API design + security + testing |
| Financial Model Development | `quant-analyst` | `data-scientist`, `ml-engineer` | When involves ML + data pipeline + backend |
| Database Schema Changes | `database-schema-designer` | `postgres-pro`, `performance-optimizer` | When affects multiple systems |
| ML Model Implementation | `ml-engineer` | `python-pro`, `data-engineer` | When involves data pipeline + API integration |
| API Development | `api-designer` | `python-pro`, `api-documenter`, `security-auditor` | Always for complete API development |
| Frontend Features | `frontend-developer` | `react-pro`, `ui-designer` | When involves backend integration |
| Performance Issues | `performance-optimizer` | `database-optimizer`, `cloud-architect` | When involves multiple system layers |
| Security Implementation | `security-auditor` | `python-pro`, `api-designer` | When involves architecture changes |
| Testing Implementation | `test-suite-generator` | `python-pro`, `qa-expert` | When involves multiple test types |
| DevOps & Deployment | `deployment-engineer` | `cloud-architect`, `security-auditor` | When involves security + monitoring |

### 🔄 Automatic Fallback Mechanisms

**Primary Selection Failure**:
- If primary agent not available → Use `agent-organizer` to recommend alternatives
- If task unclear → Use `agent-organizer` to analyze and delegate
- If multiple domains detected → Automatically use `agent-organizer`

**Quality Assurance Fallbacks**:
- All code changes → Auto-include `code-reviewer` unless explicitly excluded
- All security-related tasks → Auto-include `security-auditor`
- All API changes → Auto-include `api-documenter` for documentation updates

### 🎯 Implementation Examples

#### Example 1: "Fix the recommendation engine performance issue"
**Auto-selected agents**: `performance-optimizer` + `python-pro` + `ml-engineer`
**Reason**: Performance issue + Python code + ML model optimization

#### Example 2: "Add user authentication to the API"  
**Auto-selected approach**: Use `agent-organizer` to coordinate `backend-architect` + `security-auditor` + `api-documenter`
**Reason**: Multi-domain task requiring architecture + security + documentation

#### Example 3: "Optimize the PostgreSQL query in the stock analysis"
**Auto-selected agents**: `postgres-pro` + `database-optimizer`
**Reason**: Database-specific optimization task

#### Example 4: "Create comprehensive tests for the trading algorithm"
**Auto-selected agents**: `test-suite-generator` + `quant-analyst` + `python-pro`
**Reason**: Testing + financial domain expertise + Python implementation

### 🚫 User Override Options

Users can override automatic agent selection by:
- **"no agents"** - Disables all automatic agent usage
- **"use only [agent-name]"** - Forces specific agent usage
- **"no [agent-name]"** - Excludes specific agents from selection
- **"manual agent selection"** - Allows user to choose agents

### 💡 Automatic Agent Selection Process

For every user prompt, Claude Code will:

1. **Parse the prompt** for task type indicators
2. **Identify technology domains** involved (backend, frontend, database, ML, etc.)
3. **Assess complexity level** (single-domain vs. multi-domain)
4. **Select appropriate agents** using the mapping matrix above
5. **Apply fallback mechanisms** if primary selection fails
6. **Invoke agents automatically** unless user explicitly opts out
7. **Coordinate multi-agent workflows** using agent-organizer when needed

### 🔧 Complexity Assessment Matrix

**Complexity Scoring (1-5 scale):**
- **Level 1-2:** Single agent direct invocation
- **Level 3:** Primary + secondary agents
- **Level 4-5:** Agent-organizer coordination required

| Indicator | Complexity Points |
|-----------|------------------|
| Multiple files affected | +1 |
| Cross-system dependencies | +2 |
| Security implications | +1 |
| Performance critical | +1 |
| Multiple technology stacks | +2 |
| Architectural changes | +2 |
| User-facing changes | +1 |

### 🚀 Agent Cohort Patterns

**Common Agent Teams for This Project:**

1. **Data Pipeline Activation Team**
   - Lead: `data-engineer`
   - Support: `python-pro`, `database-schema-designer`, `deployment-engineer`
   - Use for: Activating Airflow DAGs, ETL implementation

2. **ML Model Development Team**
   - Lead: `ml-engineer`
   - Support: `quant-analyst`, `data-scientist`, `python-pro`
   - Use for: Training models, feature engineering, backtesting

3. **Frontend Completion Team**
   - Lead: `frontend-developer`
   - Support: `react-pro`, `ui-designer`, `api-designer`
   - Use for: Implementing missing pages, UI/UX improvements

4. **Production Deployment Team**
   - Lead: `deployment-engineer`
   - Support: `cloud-architect`, `security-auditor`, `devops-engineer`
   - Use for: K8s setup, CI/CD pipeline, production configuration

5. **Performance Optimization Team**
   - Lead: `performance-optimizer`
   - Support: `database-optimizer`, `python-pro`, `cache-expert`
   - Use for: System-wide performance improvements

### 📊 Current Project Priority Agent Assignments

Based on project status (72% complete):

**Immediate Priority Tasks (auto-invoke these teams):**
1. **"Complete frontend pages"** → Frontend Completion Team
2. **"Train ML models"** → ML Model Development Team
3. **"Activate data pipeline"** → Data Pipeline Activation Team
4. **"Set up CI/CD"** → Production Deployment Team
5. **"Optimize performance"** → Performance Optimization Team

This ensures that every interaction leverages the full power of the 300+ specialized agents available, providing expert-level assistance for all aspects of the investment analysis platform development.

## Technology Stack

### Backend
- **Framework**: FastAPI (Python)
- **ML/AI**: PyTorch, scikit-learn, Prophet, Hugging Face Transformers (FinBERT)
- **Data Processing**: Apache Airflow, Kafka, Pandas, NumPy/SciPy
- **Database**: PostgreSQL with Elasticsearch for fast queries
- **Caching**: Redis or similar for API response caching

### Frontend
- **Web**: React.js with Material-UI
- **Mobile**: React Native
- **Visualization**: Plotly Dash, React-based charting libraries

### Infrastructure
- **Containerization**: Docker and docker-compose
- **Orchestration**: Kubernetes (targeting DigitalOcean or AWS Free Tier)
- **Monitoring**: Prometheus/Grafana stack
- **CI/CD**: GitHub Actions or GitLab CI

## Common Development Commands

### Quick Start with Makefile
```bash
# Primary commands via Makefile
make help           # Show available commands
make build          # Build all Docker images
make up             # Start all services
make down           # Stop all services
make test           # Run all tests
make clean          # Clean up containers and volumes
make debug          # Run debug validation
make logs           # View logs
make init-db        # Initialize database
make install-deps   # Install all dependencies
```

### Backend Development
```bash
# Install Python dependencies
pip install -r requirements.txt

# Run FastAPI development server
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest backend/tests/
pytest backend/tests/test_recommendation_engine.py -v  # Single test file
pytest backend/tests/ -k "test_function_name"  # Single test function

# Format Python code (configured in pyproject.toml)
black backend/ --line-length 88
isort backend/ --profile black

# Lint Python code
flake8 backend/ --max-line-length 88
mypy backend/ --python-version 3.11
pylint backend/
```

### Frontend Development
```bash
# Install dependencies
cd frontend/web && npm install
cd frontend/mobile && npm install

# Run React development server
cd frontend/web && npm start

# Run React Native
cd frontend/mobile && npm run ios  # or npm run android

# Run tests
npm test
npm test -- --coverage  # With coverage report

# Lint and format
cd frontend/web && npm run lint
cd frontend/web && npm run format

# Build for production
npm run build
```

### Docker Commands
```bash
# Build containers
docker-compose build
docker-compose -f docker-compose.yml -f docker-compose.development.yml build  # Development
docker-compose -f docker-compose.yml -f docker-compose.production.yml build   # Production

# Run entire stack
docker-compose up
docker-compose up -d  # Detached mode

# Run specific service
docker-compose up backend
docker-compose up postgres redis  # Multiple services

# View logs
docker-compose logs -f [service-name]
docker-compose logs -f backend --tail=100  # Last 100 lines

# Execute commands in containers
docker-compose exec backend bash
docker-compose exec backend python -m backend.utils.load_initial_stocks
```

### Database Management
```bash
# Run database migrations
alembic upgrade head
alembic downgrade -1  # Rollback one migration

# Create new migration
alembic revision --autogenerate -m "description"

# Access PostgreSQL
docker exec -it investment_db psql -U postgres -d investment_db

# Database initialization scripts
python scripts/init_database.py
python scripts/setup_db_credentials.py
bash scripts/init_database.sh

# Load initial stock data
docker-compose exec backend python -m backend.utils.load_initial_stocks
```

## Architecture Overview

The application follows a microservices architecture with these key components:

### Data Flow
1. **Data Ingestion**: Apache Airflow orchestrates daily ETL pipelines that fetch data from free APIs (Alpha Vantage, Finnhub, FMP, Polygon.io)
2. **Processing**: Kafka handles real-time streaming data; batch processing occurs via scheduled Airflow DAGs
3. **Storage**: PostgreSQL stores structured data; Elasticsearch indexes for fast queries; Redis caches API responses
4. **Analytics**: ML models run on scheduled basis, generating predictions and recommendations
5. **API Layer**: FastAPI serves processed data and recommendations to frontend clients
6. **Frontend**: React web app and React Native mobile app consume the API

### Key Architectural Decisions
- **Cost Optimization**: All API calls are batched and cached to stay within free tier limits
- **Scalability**: Kubernetes allows horizontal scaling; database sharding by ticker symbol
- **Reliability**: Circuit breakers and retries for external API calls; fallback to cached data
- **Security**: OAuth2 authentication, end-to-end encryption, audit logging

### Critical Cost Controls
- API call tracking dashboard in Grafana to monitor usage against free tier limits
- Automatic fallback to cached data when approaching API limits
- Batch processing during off-peak hours to minimize compute costs
- Auto-scaling down to zero pods during idle periods

## Important Implementation Notes

1. **API Rate Limiting**: Each external API has specific free tier limits that must be respected:
   - Alpha Vantage: 25 API calls/day (5 calls/minute)
   - Finnhub: 60 calls/minute
   - Polygon.io: 5 API calls/minute on free tier
   
2. **Data Caching Strategy**: Historical data should be cached indefinitely; intraday data cached for 15-30 minutes

3. **ML Model Training**: Models should be trained offline and deployed as serialized artifacts to minimize compute costs

4. **Compliance Requirements**: All user data must be anonymized; audit logs required for SEC compliance

5. **Error Handling**: Implement graceful degradation - if real-time data unavailable, use cached data with clear indicators

## Development Timeline

Based on requirements, the project follows a 23-week timeline:
- Weeks 1-2: Planning and architecture
- Weeks 3-6: Data ingestion pipelines
- Weeks 7-12: Analytics and ML implementation
- Weeks 13-16: UI and API development
- Weeks 17-19: Compliance and security
- Weeks 20-23: Testing and deployment

## Project Structure

### Key Directories
```
backend/
├── analytics/          # Technical, fundamental, sentiment analysis engines
│   ├── fundamental/    # DCF models, valuation, quality scoring
│   ├── sentiment/      # News, social media, insider sentiment
│   └── technical/      # Indicators, patterns, market structure
├── api/               # FastAPI endpoints and routers
├── data_ingestion/    # API clients (Alpha Vantage, Finnhub, Polygon, SEC)
├── ml/                # Machine learning models and features
│   ├── models/        # Classification, time series, ensemble models
│   └── features/      # Feature engineering and selection
├── tasks/             # Celery tasks and Airflow scheduling
├── utils/             # Shared utilities (cache, rate limiter, monitoring)
└── tests/             # Test suites and fixtures

frontend/web/
├── src/
│   ├── components/    # Reusable React components
│   ├── pages/         # Page components (Dashboard, Portfolio, etc.)
│   ├── services/      # API service layer
│   └── config/        # Configuration and constants

data_pipelines/
└── airflow/
    └── dags/          # Daily market analysis DAGs
```

### Important Configuration Files
- `pyproject.toml` - Python tooling configuration (black, isort, mypy, pytest)
- `alembic.ini` - Database migration configuration
- `docker-compose.yml` - Main Docker services configuration
- `docker-compose.{development,production,test}.yml` - Environment-specific overrides
- `Makefile` - Common development commands
- `requirements.txt` - Python dependencies
- `frontend/web/package.json` - Frontend dependencies and scripts

## Critical API Endpoints

### Main API Routes (FastAPI)
- `/api/health` - Health check endpoint
- `/api/stocks/{ticker}` - Stock data and analysis
- `/api/recommendations` - AI-generated recommendations
- `/api/portfolio` - Portfolio management
- `/api/analysis/{ticker}` - Detailed analysis (technical, fundamental, sentiment)
- `/api/admin` - Admin operations (requires authentication)
- `/ws` - WebSocket for real-time updates

### Service Ports
- Frontend: `http://localhost:3000`
- Backend API: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`
- PostgreSQL: `localhost:5432`
- Redis: `localhost:6379`
- Elasticsearch: `localhost:9200`
- Grafana: `http://localhost:3001` (when configured)
- Airflow: `http://localhost:8080` (when configured)

## Environment Variables

Key environment variables (stored in `.env`):
- `ALPHA_VANTAGE_API_KEY` - Alpha Vantage API key (25 calls/day limit)
- `FINNHUB_API_KEY` - Finnhub API key (60 calls/minute)
- `POLYGON_API_KEY` - Polygon.io API key (5 calls/minute free tier)
- `NEWS_API_KEY` - NewsAPI key for sentiment analysis
- `DB_PASSWORD` - PostgreSQL password
- `REDIS_PASSWORD` - Redis password
- `SECRET_KEY` - Application secret key
- `JWT_SECRET_KEY` - JWT token signing key

## Testing Strategy

### Running Tests
```bash
# Backend tests with coverage
pytest backend/tests/ --cov=backend --cov-report=html

# Frontend tests
cd frontend/web && npm test -- --coverage

# Integration tests with Docker
docker-compose -f docker-compose.test.yml up --abort-on-container-exit

# Specific test categories
pytest backend/tests/ -m "unit"        # Unit tests only
pytest backend/tests/ -m "integration" # Integration tests
pytest backend/tests/ -m "slow"        # Slow tests
pytest backend/tests/ -m "financial"   # Financial model tests
pytest backend/tests/ -m "api"         # API endpoint tests
pytest backend/tests/ -m "security"    # Security tests
pytest backend/tests/ -m "performance" # Performance tests
```

### Test Configuration
- **Coverage target**: 85% minimum (configured in pyproject.toml)
- **Test markers**: unit, integration, performance, security, compliance, financial, slow, api, database, cache, external_api, flaky
- **Auto-fail**: Tests fail if coverage drops below 85%

### Key Test Files
- `backend/tests/test_recommendation_engine.py` - Core recommendation logic
- `backend/tests/fixtures/` - Test data and mocks

## Deployment Scripts

- `deploy.sh` - Main deployment script
- `start.sh`, `start-docker.sh` - Start services
- `stop.sh` - Stop all services
- `restart.sh` - Restart services
- `COMPLETE_IMPLEMENTATION.sh` - Full implementation setup
- `QUICK_START.sh` - Quick start for development
- `validate-deployment.py` - Validate deployment health
- `update_agents.sh` - Update claude-code-sub-agents repository

## Claude Code Sub-Agents

The project includes comprehensive agent collections from 7 specialized repositories, providing hundreds of expert agents for all development needs:

### Repository 1: claude-code-sub-agents
From https://github.com/dl-ezo/claude-code-sub-agents.git stored in `.claude/agents/claude-code-sub-agents/`:
- **api-designer** - Design and document API endpoints
- **code-reviewer** - Review code for quality and best practices
- **database-schema-designer** - Design database schemas and relationships
- **performance-optimizer** - Optimize code performance
- **security-analyzer** - Analyze code for security vulnerabilities
- **system-architect** - Design system architecture
- **test-suite-generator** - Generate comprehensive test suites
- And 30+ more specialized agents

### Repository 2: awesome-claude-code-agents
From https://github.com/hesreallyhim/awesome-claude-code-agents.git stored in `.claude/agents/awesome-claude-code-agents/`:
- **python-backend-engineer** - Python backend development specialist (perfect for FastAPI)
- **backend-typescript-architect** - TypeScript backend architecture
- **senior-code-reviewer** - Advanced code review capabilities
- **ui-engineer** - UI/UX engineering specialist

### Repository 3: wshobson-agents
From https://github.com/wshobson/agents.git stored in `.claude/agents/wshobson-agents/`:
- **python-pro** - Advanced Python development expertise
- **data-scientist** - ML/AI and data analysis specialist
- **security-auditor** - Security vulnerability assessment
- **performance-engineer** - Performance optimization
- **cloud-architect** - Cloud infrastructure design
- **devops-troubleshooter** - DevOps and deployment issues
- **quant-analyst** - Financial analysis and modeling (perfect for investment app)
- And 50+ more professional agents

### Repository 4: voltagent-subagents
From https://github.com/VoltAgent/awesome-claude-code-subagents.git stored in `.claude/agents/voltagent-subagents/`:
- Organized in categories (development, testing, deployment, etc.)
- 100+ specialized agents across multiple domains
- Enterprise-focused development patterns

### Repository 5: furai-subagents
From https://github.com/0xfurai/claude-code-subagents.git stored in `.claude/agents/furai-subagents/`:
- Advanced code analysis and optimization agents
- Security-focused development specialists
- Modern development workflow agents

### Repository 6: lst97-subagents
From https://github.com/lst97/claude-code-sub-agents.git stored in `.claude/agents/lst97-subagents/`:
- Organized by specialization (business, data-ai, development, infrastructure, security)
- **context-manager** - Advanced context management
- **agent-organizer** - Meta-agent for organizing other agents
- Comprehensive development lifecycle coverage

### Repository 7: nuttall-agents
From https://github.com/iannuttall/claude-agents.git stored in `.claude/agents/nuttall-agents/`:
- Specialized development agents
- Focus on modern development practices

### Updating All Agents
To update all 7 agent repositories to the latest version, run:
```bash
./update_agents.sh
```

This script will:
- Pull the latest updates from all 7 repositories
- Display agent counts from each repository
- Show total available agents (typically 300+ agents)
- Ensure you have access to the latest agent capabilities

## Critical Architecture Patterns

### Stock Processing Tiers
The system categorizes stocks into 5 priority tiers for efficient API usage:
- **Tier 1 (Critical)**: S&P 500, high volume stocks - Updated hourly via Finnhub
- **Tier 2 (High)**: Mid-cap active stocks - Updated every 4 hours via Alpha Vantage
- **Tier 3 (Medium)**: Small-cap stocks - Updated every 8 hours via Polygon
- **Tier 4 (Low)**: Inactive stocks - Daily updates from cached data
- **Tier 5 (Minimal)**: Delisted/low activity - Weekly updates only

### API Client Architecture
All API clients inherit from `BaseAPIClient` (backend/data_ingestion/base_client.py) which provides:
- Automatic rate limiting with exponential backoff
- Cost monitoring integration
- Multi-tier caching (regular, extended, stale)
- Fallback provider switching when rate limited
- Circuit breaker pattern for resilience

### Cost Monitoring System
The `CostMonitor` (backend/utils/cost_monitor.py) and `EnhancedCostMonitor` implement:
- Real-time API usage tracking against free tier limits
- Automatic fallback to alternative providers
- Emergency mode activation when approaching $50/month limit
- Smart data fetching with cache prioritization
- Linear programming optimization for API allocation

### Data Quality Framework
`DataQualityChecker` (backend/utils/data_quality.py) validates:
- Price consistency (high >= low, close within range)
- Volume anomalies and outliers
- Data staleness and gaps
- Suspicious patterns (manipulation detection)
- Quality scoring system (0-100 scale)

### Analysis Engine Integration
The `RecommendationEngine` orchestrates multiple analysis engines:
1. **TechnicalAnalysisEngine**: 200+ indicators, pattern recognition
2. **FundamentalAnalysisEngine**: DCF valuation, peer comparison
3. **SentimentAnalysisEngine**: FinBERT-powered news analysis
4. **ModelManager**: Ensemble of LSTM, XGBoost, Prophet models

Each engine returns standardized scores that are weighted and combined.

### Database Optimization Strategy
- TimescaleDB hypertables for time-series data (price_history, technical_indicators)
- Monthly partitioning with automatic compression after 7 days
- Materialized views for common queries (daily_stock_metrics, top_movers)
- Continuous aggregates for real-time metrics
- Optimized autovacuum settings for high-write tables

### Airflow DAG Structure
The main DAG (`daily_market_analysis.py`) implements:
1. Market calendar check
2. Stock prioritization into tiers
3. Parallel data ingestion by tier
4. Technical analysis computation
5. Recommendation generation
6. Cost metrics update

Each tier has different update frequencies and API assignments based on priority.

### Caching Strategy
Three-tier caching system:
- **Regular Cache**: Standard TTL (5 min for prices, 1 day for fundamentals)
- **Extended Cache**: 2x TTL for cost-saving mode
- **Stale Cache**: 7-day fallback for emergencies

Cache keys follow pattern: `{data_type}:{ticker}:{date}`

### Error Handling Patterns
- Circuit breakers prevent cascading failures
- Graceful degradation to cached data
- Automatic provider switching on rate limits
- Stale data indicators in responses
- Comprehensive logging with correlation IDs

### Security Considerations
- OAuth2 authentication for user endpoints
- API keys stored in environment variables
- Rate limiting per user/IP
- Data anonymization for GDPR compliance
- Audit logging for SEC requirements

## Performance Optimization

### Parallel Processing
- Use Ray for distributed stock analysis
- Batch processing with configurable chunk sizes
- Async/await patterns throughout the codebase
- Connection pooling for database and Redis

### Query Optimization
- Database indexes on (stock_id, date) for time-series queries
- Materialized views refreshed on schedule
- Query result caching with appropriate TTLs
- Prepared statements for common queries

### Memory Management
- Streaming large datasets instead of loading into memory
- Garbage collection tuning for Python processes
- Docker memory limits to prevent OOM
- Redis memory policies with LRU eviction