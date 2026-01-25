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

## ⚠️ MANDATORY: Session Completion Protocol ⚠️

**CRITICAL**: You MUST execute these steps at the END of EVERY Claude session. This is NOT optional.

### Step 1: Sync to Notion (REQUIRED)
```bash
./notion-sync.sh push
```
**This command MUST be run before ending ANY session.** It syncs project status to the Notion Product Development Tracker.

### Step 2: Update Tracking Documents
1. **Update `TODO.md`** with completed items:
   - Mark finished tasks with ~~strikethrough~~ and ✅ COMPLETE
   - Update the "Already Complete" section
   - Update status percentages in header

2. **Update `IMPLEMENTATION_STATUS.md`** if major milestones reached

### Step 3: Commit and Push
```bash
git add -A && git commit -m "chore: Update project status" && git push
```

### Verification
Before ending the session, confirm:
- [ ] `./notion-sync.sh push` executed successfully
- [ ] TODO.md updated with session changes
- [ ] Changes committed and pushed

**IF YOU FORGET TO RUN NOTION SYNC, THE USER'S PROJECT BOARD WILL BE OUT OF DATE.**

This ensures context persists across sessions, prevents duplicate work, and keeps the Notion board synchronized.

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

---

## Complete Claude Code Ecosystem Integration

This project integrates **two major Claude Code repositories** for enterprise-grade multi-agent orchestration:

1. **everything-claude-code** from [github.com/affaan-m/everything-claude-code](https://github.com/affaan-m/everything-claude-code)
2. **claude-flow** from [github.com/ruvnet/claude-flow](https://github.com/ruvnet/claude-flow)

### Integration Summary

| Category | Everything Claude Code | Claude Flow | Custom | Total |
|----------|------------------------|-------------|--------|-------|
| Agents | 9 | 100+ (in 25 dirs) | 6 | 128 |
| Skills | 11 | 37 | 3 | 71 |
| Commands | 15 | 23+ (in 19 dirs) | 4 | 164 |
| Helpers | 4 | 31 | - | 35 |
| Rules | 8 | - | - | 8 |

### Claude Flow Advanced Capabilities

The claude-flow integration adds:

#### Multi-Agent Orchestration
- **Queen Orchestrator**: Master agent coordinating all investment workflows
- **Swarm Topologies**: Mesh (collaborative), Hierarchical (orchestrated), Ring, Star
- **Consensus Mechanisms**: Raft, BFT, Gossip, CRDT for agent coordination

#### Agent Categories from Claude Flow
| Category | Purpose |
|----------|---------|
| `consensus/` | BFT, Raft, Gossip consensus agents |
| `hive-mind/` | Collective intelligence coordination |
| `flow-nexus/` | Flow orchestration and management |
| `swarm/` | Multi-agent swarm coordination |
| `sona/` | Self-optimizing neural architecture |
| `sparc/` | SPARC methodology agents |
| `github/` | GitHub integration agents |
| `optimization/` | Performance optimization agents |
| `reasoning/` | Advanced reasoning agents |

#### Advanced Skills from Claude Flow
- `agentdb-*`: Advanced memory, learning, vector search
- `flow-nexus-*`: Neural, platform, swarm orchestration
- `github-*`: Code review, multi-repo, release management
- `hive-mind-advanced`: Collective intelligence
- `sparc-methodology`: SPARC development methodology
- `swarm-*`: Advanced swarm coordination
- `v3-*`: V3 implementation skills

#### Helper Scripts
- `learning-service.mjs`: Continuous learning loop
- `swarm-hooks.sh`: Swarm coordination hooks
- `worker-manager.sh`: Background worker management
- `metrics-db.mjs`: Performance metrics tracking
- `checkpoint-manager.sh`: State persistence

#### V3 Implementation
Located in `.claude/v3/`:
- `@claude-flow/`: Core TypeScript implementation
- `mcp/`: MCP server (connection pool, session manager, tool registry)
- `src/`: Coordination, memory, task execution modules
- `swarm.config.ts`: Swarm configuration

### Custom Investment Analysis Commands

| Command | Description | Agent |
|---------|-------------|-------|
| `/underwrite` | Comprehensive deal underwriting | deal-underwriter |
| `/model` | Build DCF, LBO, financial models | financial-modeler |
| `/analyze-structure` | Review security packages & liens | - |
| `/scenario` | Scenario and sensitivity analysis | - |

### Custom Investment Analysis Agents

#### `queen-investment-orchestrator` (.claude/agents/queen-investment-orchestrator.md)
**Master orchestrator** for coordinating all investment analysis workflows.
- Routes tasks to optimal specialized agents
- Coordinates multi-phase deal analysis
- Manages context across agent handoffs
- Supports mesh, hierarchical, star topologies
- Performance targets: 32.3% token reduction, 2.8-4.4x speedup

#### `investment-analyst` (.claude/agents/investment-analyst.md)
Senior investment analyst for deal analysis, due diligence, and investment memos.
- Market analysis and industry research
- Risk-return evaluation
- Investment memo generation
- Cannabis and real estate specialization

#### `deal-underwriter` (.claude/agents/deal-underwriter.md)
Credit underwriter for loans and secured lending transactions.
- Credit analysis and scoring
- Collateral evaluation
- UCC lien analysis
- Covenant structuring
- Intercreditor review

#### `financial-modeler` (.claude/agents/financial-modeler.md)
Expert financial modeler for quantitative analysis.
- DCF valuations
- LBO modeling
- Scenario and sensitivity analysis
- Returns calculations (IRR, MOIC)
- Data consolidation automation

#### `risk-assessor` (.claude/agents/risk-assessor.md)
Portfolio and deal risk assessment specialist.
- Value at Risk (VaR) calculations
- Stress testing and scenario analysis
- Credit risk analysis (PD, LGD, EAD)
- Regulatory compliance (SEC, Basel III/IV)
- Concentration risk monitoring

#### `portfolio-manager` (.claude/agents/portfolio-manager.md)
Portfolio construction and management specialist.
- Asset allocation (strategic and tactical)
- Portfolio optimization (Mean-Variance, Risk Parity, Black-Litterman)
- Performance attribution analysis
- Rebalancing strategies
- Risk budgeting

### Custom Investment Skills

#### `financial-modeling` (.claude/skills/financial-modeling/)
- DCF valuation methodology
- LBO model framework
- Scenario analysis patterns
- Sensitivity table generation
- Cash flow modeling

#### `deal-structuring` (.claude/skills/deal-structuring/)
- Security package design
- UCC lien perfection
- Intercreditor agreements
- Covenant structures
- Collateral analysis

#### `underwriting-analysis` (.claude/skills/underwriting-analysis/)
- Credit scoring methodology
- Financial statement spreading
- Risk assessment framework
- Due diligence checklists

### Imported Development Commands

| Command | Description |
|---------|-------------|
| `/plan` | Create implementation plan before coding |
| `/tdd` | Test-driven development workflow |
| `/code-review` | Quality and security review |
| `/build-fix` | Fix build and compilation errors |
| `/e2e` | End-to-end test generation |
| `/refactor-clean` | Remove dead code |
| `/update-docs` | Sync documentation |
| `/verify` | Run verification loop |
| `/learn` | Extract patterns mid-session |
| `/checkpoint` | Save verification state |

### Investment Analysis Workflow

```
1. Receive new deal for analysis
   → /underwrite to perform credit analysis

2. Build financial projections
   → /model DCF to create valuation

3. Review security package
   → /analyze-structure for lien priority

4. Stress test assumptions
   → /scenario for sensitivity analysis

5. Generate investment memo
   → Use investment-analyst agent
```

### Integration Manifest

Full component registry: `.claude/integration-manifest.json`

### Orchestration Patterns

Use the Queen Investment Orchestrator for complex workflows:

```
PATTERN 1: New Deal Analysis
├─ PARALLEL: investment-analyst + deal-underwriter + financial-modeler
├─ SEQUENTIAL: credit scoring → projections → collateral valuation
├─ INTEGRATION: scenario analysis → security package → risk assessment
└─ DELIVERY: investment memo

PATTERN 2: Portfolio Risk Assessment
├─ financial-analysis-swarm: Calculate VaR, Sharpe, Sortino
├─ financial-modeler: Stress test scenarios
├─ risk-assessor: Generate risk report
└─ Validate SEC compliance

PATTERN 3: Due Diligence Deep Dive
├─ deal-underwriter: UCC search, lien analysis
├─ investment-analyst: Business & market due diligence
├─ financial-modeler: Validate projections
└─ deal-underwriter: Structure security package
```

### Updating from Upstream

**Sync with everything-claude-code:**
```bash
git clone --depth 1 https://github.com/affaan-m/everything-claude-code.git /tmp/ecc
cp /tmp/ecc/agents/*.md .claude/agents/
cp -r /tmp/ecc/skills/* .claude/skills/
cp /tmp/ecc/commands/*.md .claude/commands/
rm -rf /tmp/ecc
```

**Sync with claude-flow:**
```bash
git clone --depth 1 https://github.com/ruvnet/claude-flow.git /tmp/cf
cp -r /tmp/cf/.claude/agents/* .claude/agents/
cp -r /tmp/cf/.claude/skills/* .claude/skills/
cp -r /tmp/cf/.claude/commands/* .claude/commands/
cp -r /tmp/cf/.claude/helpers/* .claude/helpers/
rm -rf /tmp/cf
```

Custom investment-specific components will not be overwritten as they have unique names.