# CLAUDE.md

This file provides guidance to Claude Code when working with this investment analysis platform.

## ⚡ CRITICAL: MANDATORY AGENT SWARM UTILIZATION ⚡

**YOU MUST PROACTIVELY USE SPECIALIZED AGENT SWARMS FOR ALL RELEVANT TASKS. THE USER SHOULD NEVER HAVE TO REMIND YOU TO USE THEM.**

### When to Use Agent Swarms (ALWAYS - NOT OPTIONAL)

#### 🏗️ infrastructure-devops-swarm
**USE IMMEDIATELY FOR:**
- ANY Docker, docker-compose, or container issues
- CI/CD pipeline configuration or debugging
- Deployment problems or optimization
- Monitoring setup (Prometheus, Grafana, AlertManager)
- Service health checks, restarts, or orchestration
- Cost optimization and resource management
- Infrastructure architecture decisions
- Production deployment configuration
- ANY DevOps or infrastructure-related task

**DO NOT** manually debug Docker issues, health checks, or service configurations. ALWAYS delegate to this swarm.

#### 💾 data-ml-pipeline-swarm
**USE IMMEDIATELY FOR:**
- ETL pipeline design or debugging
- Airflow DAG creation, modification, or troubleshooting
- Data ingestion from APIs (Alpha Vantage, Finnhub, Polygon, etc.)
- ML model training pipeline issues
- Data quality checks and validation
- TimescaleDB optimization
- API rate limiting strategies
- Batch processing design
- Data pipeline performance issues
- ANY data engineering or ML pipeline task

**DO NOT** manually debug data pipelines or API integrations. ALWAYS delegate to this swarm.

#### 📊 financial-analysis-swarm
**USE IMMEDIATELY FOR:**
- Stock analysis algorithms
- Financial calculations (ratios, metrics, indicators)
- ML-based prediction models
- Fundamental/technical analysis implementation
- Portfolio optimization algorithms
- Risk assessment (VaR, Sharpe, Sortino)
- Trading signal generation
- FinBERT or sentiment analysis for finance
- SEC compliance for recommendations
- ANY financial analysis, modeling, or investment logic

**DO NOT** manually implement financial algorithms. ALWAYS delegate to this swarm.

#### 🔧 backend-api-swarm
**USE IMMEDIATELY FOR:**
- FastAPI endpoint creation or modification
- REST API design and implementation
- WebSocket implementation
- Database operations and queries
- Authentication/authorization logic
- Repository pattern implementation
- Async service implementation
- API performance optimization
- ANY backend API or service development

**DO NOT** manually write API endpoints or services. ALWAYS delegate to this swarm.

#### 🎨 ui-visualization-swarm
**USE IMMEDIATELY FOR:**
- React component development
- Dashboard design and implementation
- Financial charts and visualizations
- Real-time UI updates
- Frontend state management
- User experience optimization
- Responsive design implementation
- ANY frontend or visualization work

**DO NOT** manually write React components or dashboards. ALWAYS delegate to this swarm.

#### ✅ project-quality-swarm
**USE IMMEDIATELY FOR:**
- Code review of ANY substantial changes
- Test automation and coverage
- Architecture decision validation
- Documentation updates
- Code quality improvements
- Refactoring evaluation
- ANY quality assurance task

**DO NOT** skip code reviews or quality checks. ALWAYS delegate to this swarm after implementation.

#### 🔍 Explore Agent
**USE IMMEDIATELY FOR:**
- Understanding codebase structure
- Finding files, classes, or functions
- Answering "how does X work?" questions
- Locating error handlers or specific implementations
- ANY exploratory or investigative task

**DO NOT** manually grep/glob for complex searches. ALWAYS use Explore agent.

#### 🏛️ architecture-reviewer
**USE IMMEDIATELY FOR:**
- System design decisions
- Technology stack evaluation
- Scalability strategy review
- Database schema review
- Microservices vs monolithic decisions
- ANY architectural decision or review

**DO NOT** make architectural decisions alone. ALWAYS get architecture-reviewer validation.

#### 🛡️ security-compliance-swarm
**USE IMMEDIATELY FOR:**
- SEC regulatory compliance features
- GDPR data protection implementation
- Security audits and vulnerability assessment
- Authentication hardening
- Compliance documentation
- Audit logging implementation
- ANY security or compliance task

**DO NOT** implement security features without this swarm's expertise.

#### 💎 godmode-refactorer
**USE IMMEDIATELY FOR:**
- ANY refactoring task (simple to complex)
- Code restructuring
- Architectural overhauls
- Multi-file refactoring
- Legacy code modernization
- ANY code quality improvement task

**DO NOT** manually refactor code. ALWAYS use this swarm.

### Swarm Utilization Rules

1. **PROACTIVE, NOT REACTIVE**: Launch swarms IMMEDIATELY when tasks match their domain. Don't wait for user prompts.

2. **PARALLEL EXECUTION**: Launch multiple swarms concurrently when possible using a single message with multiple Task tool calls.

3. **COMPREHENSIVE DELEGATION**: Give swarms complete context and full autonomy to solve problems.

4. **TRUST SWARM OUTPUT**: Swarm results represent expert-level work. Trust their recommendations and implementations.

5. **NEVER WORK ALONE**: If a task matches a swarm's domain, you MUST use that swarm. Working solo on specialized tasks is PROHIBITED.

### Violation Prevention

❌ **WRONG**: "Let me debug this Docker health check issue..."
✅ **CORRECT**: Immediately launch infrastructure-devops-swarm

❌ **WRONG**: "Let me write this FastAPI endpoint..."
✅ **CORRECT**: Immediately launch backend-api-swarm

❌ **WRONG**: "Let me create this React component..."
✅ **CORRECT**: Immediately launch ui-visualization-swarm

❌ **WRONG**: "Let me implement this financial calculation..."
✅ **CORRECT**: Immediately launch financial-analysis-swarm

**IF THE USER EVER HAS TO REMIND YOU TO USE A SWARM, YOU HAVE FAILED.**

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

## Multi-Swarm Coordination Strategy

### For Complex Cross-Domain Tasks

When tasks span multiple domains, coordinate swarms efficiently:

1. **Sequential Coordination**:
   ```
   infrastructure-devops-swarm → fixes Docker issues
   ↓
   backend-api-swarm → implements API changes
   ↓
   project-quality-swarm → reviews and tests
   ```

2. **Parallel Coordination**:
   ```
   Launch simultaneously in one message:
   - backend-api-swarm (API endpoint)
   - ui-visualization-swarm (React component)
   - data-ml-pipeline-swarm (data processing)

   Then launch:
   - project-quality-swarm (reviews all)
   ```

3. **Expert Consultation**:
   ```
   architecture-reviewer → validates design
   ↓
   Specialized swarms → implement
   ↓
   security-compliance-swarm → validates security
   ↓
   project-quality-swarm → final review
   ```

### Examples of Optimal Swarm Utilization

**Example 1: "Add a new stock recommendation endpoint"**
```
1. Launch architecture-reviewer to validate approach
2. Launch backend-api-swarm to implement endpoint
3. Launch financial-analysis-swarm for recommendation logic
4. Launch project-quality-swarm for testing
5. Launch infrastructure-devops-swarm for deployment config
```

**Example 2: "Debug Celery worker issues"**
```
1. IMMEDIATELY launch infrastructure-devops-swarm
   (DO NOT debug manually - this is their specialty)
```

**Example 3: "Create portfolio dashboard"**
```
1. Launch ui-visualization-swarm for React components
2. Launch backend-api-swarm for API endpoints (parallel)
3. Launch financial-analysis-swarm for calculations (parallel)
4. Launch project-quality-swarm for review
```

**Example 4: "Optimize data pipeline performance"**
```
1. Launch data-ml-pipeline-swarm to analyze and optimize
2. Launch infrastructure-devops-swarm for infrastructure tuning
3. Launch project-quality-swarm for validation
```

### Success Metrics

✅ **You're doing it RIGHT when:**
- Swarms are launched within first 1-2 messages
- Multiple swarms work in parallel on complex tasks
- You provide comprehensive context to each swarm
- User never has to ask "why didn't you use X swarm?"

❌ **You're doing it WRONG when:**
- You attempt specialized work yourself
- User has to prompt swarm usage
- Only using one swarm when multiple are relevant
- Not giving swarms complete autonomy

## Session Completion Checklist

**IMPORTANT**: After completing work in each session, update tracking documents:

1. **Update `TODO.md`** with completed items:
   - Mark finished tasks with ~~strikethrough~~ and ✅ COMPLETE
   - Update the "Already Complete" section
   - Update status percentages in header

2. **Update `IMPLEMENTATION_STATUS.md`** if major milestones reached

3. **Commit tracking doc updates** along with the code changes

This ensures context persists across sessions and prevents duplicate work.

## Available Skills

Skills are modular capabilities that extend agent functionality. Skills are located in `.claude/skills/` and provide specialized knowledge and workflows.

### Project-Specific Skills (Investment Platform)

| Skill | Purpose | Use When |
|-------|---------|----------|
| **sec-compliance** | SEC 2025 compliance validation | Generating/reviewing recommendations, audit trails |
| **cost-monitor** | $50/month budget tracking | Reviewing costs, optimizing resources |
| **api-rate-limiter** | Manage API quotas | Data ingestion, rate limit issues |
| **stock-analysis** | Comprehensive stock analysis | Running analysis pipelines |

### General Skills (from clawdbot)

| Skill | Purpose | Use When |
|-------|---------|----------|
| **github** | GitHub CLI operations | PRs, CI/CD, issues, code management |
| **tmux** | Terminal session management | Background processes, parallel tasks |
| **1password** | Secure credential management | API keys, secrets, credential audits |
| **slack** | Team communication | Notifications, alerts, status updates |
| **trello** | Project management | Task tracking, board management |
| **notion** | Documentation | ADRs, design docs, knowledge bases |
| **summarize** | Content summarization | News analysis, document processing |
| **coding-agent** | AI coding assistants | Parallel coding, automated reviews |
| **session-logs** | Session history | Debugging, audit trails, context |
| **model-usage** | Cost tracking | API/model usage monitoring |
| **skill-creator** | Create new skills | Building custom skills |

### Skill Distribution by Agent

Each agent has access to relevant skills:

```
infrastructure-devops-swarm
├── github (CI/CD, deployments)
├── tmux (monitoring, background processes)
├── 1password (secrets management)
├── slack (deployment notifications)
├── model-usage (cost tracking)
└── cost-monitor (budget monitoring)

financial-analysis-swarm
├── summarize (financial news analysis)
├── github (model versioning)
├── notion (methodology documentation)
├── sec-compliance (regulatory compliance)
└── stock-analysis (analysis workflows)

backend-api-swarm
├── github (code management)
├── tmux (development/testing)
├── 1password (credential management)
└── api-rate-limiter (API quotas)

data-ml-pipeline-swarm
├── github (pipeline versioning)
├── tmux (DAG monitoring)
├── summarize (text processing)
├── model-usage (ML costs)
├── api-rate-limiter (data ingestion)
└── cost-monitor (budget tracking)

ui-visualization-swarm
├── github (component workflow)
└── notion (design documentation)

security-compliance-swarm
├── 1password (credential audits)
├── github (security reviews)
├── session-logs (audit trails)
└── sec-compliance (regulatory checks)

project-quality-swarm
├── github (PR reviews, CI)
├── coding-agent (automated testing)
└── session-logs (debugging history)

team-coordinator
├── github (project overview)
├── slack (team communication)
├── trello (project management)
├── notion (documentation)
├── model-usage (cost monitoring)
└── All other skills as needed
```

### Using Skills

Skills are automatically available to agents based on their configuration. Key patterns:

**1. GitHub Operations (all agents)**
```bash
gh pr create --title "Feature" --body "Description"
gh pr checks <PR_NUMBER>
gh issue list --label "bug"
```

**2. Background Process Management (tmux)**
```bash
SOCKET="${TMPDIR:-/tmp}/session.sock"
tmux -S "$SOCKET" new-session -d -s mywork
tmux -S "$SOCKET" send-keys "command" Enter
```

**3. Secure Credential Access (1password)**
```bash
# Sign in first, then:
op read "op://Vault/Item/field"
```

**4. Cost Monitoring**
```bash
python scripts/model_usage.py --mode all
# Monitor API usage and stay under $50/month
```

**5. Compliance Validation (sec-compliance)**
```python
from backend.services.compliance import SECComplianceValidator
validator.validate_recommendation(recommendation)
```

### Creating Custom Skills

Use the `skill-creator` skill to build new project-specific skills:

1. Define the skill's purpose and triggers
2. Create `SKILL.md` with frontmatter and instructions
3. Add scripts/references/assets as needed
4. Place in `.claude/skills/<skill-name>/`

See `.claude/skills/skill-creator/SKILL.md` for detailed guidance.