# Agent Swarm System

## Investment Analysis Platform - 7 Expert Team Swarms

This directory contains a streamlined agent management system organized into 7 specialized swarms (teams) that coordinate expert knowledge for the investment analysis platform.

## Quick Start

### Using a Swarm

Swarms are invoked automatically when you describe tasks in their domain, or you can explicitly request one:

```
"Use the financial-analysis-swarm to analyze AAPL stock"
"Have the backend-api-swarm review this endpoint"
"Ask the team-coordinator which team should handle this"
```

### Not Sure Which Team?

Use the **team-coordinator** - it analyzes your request and recommends the optimal team(s).

---

## The 7 Swarms

| Swarm | Purpose | Use When... |
|-------|---------|-------------|
| **financial-analysis-swarm** | Stock analysis, ML predictions, quant methods | Analyzing stocks, building models, SEC compliance |
| **data-ml-pipeline-swarm** | ETL, Airflow DAGs, ML operations | Building pipelines, data ingestion, model training |
| **backend-api-swarm** | FastAPI, REST APIs, database ops | Creating endpoints, query optimization, auth |
| **security-compliance-swarm** | SEC/GDPR compliance, security audits | Compliance reviews, security hardening |
| **infrastructure-devops-swarm** | Docker, deployment, monitoring | DevOps, CI/CD, cost optimization |
| **ui-visualization-swarm** | React, charts, dashboards | Frontend development, data visualization |
| **project-quality-swarm** | Code review, testing, architecture | Reviews, tests, documentation |

Plus: **team-coordinator** for routing and multi-team coordination

---

## Directory Structure

```
.claude/
├── agents/                          # All agent definitions
│   ├── # SWARM AGENTS (primary)
│   ├── financial-analysis-swarm.md
│   ├── data-ml-pipeline-swarm.md
│   ├── backend-api-swarm.md
│   ├── security-compliance-swarm.md
│   ├── infrastructure-devops-swarm.md
│   ├── ui-visualization-swarm.md
│   ├── project-quality-swarm.md
│   ├── team-coordinator.md
│   │
│   ├── # SPECIALIST AGENTS (direct invocation)
│   ├── data-science-architect.md    # Deep data/analytics expertise
│   ├── architecture-reviewer.md     # System design review
│   ├── code-review-expert.md        # Detailed code review
│   ├── godmode-refactorer.md        # Complex refactoring
│   └── ui_design.md                 # UI/UX design commands
│
├── commands/                        # Slash commands
│   ├── analyze_codebase.md
│   └── ui_design.md
│
├── archive/                         # Archived legacy files
│   ├── agent_catalog.json           # Old 397-agent catalog
│   ├── agent_invocation_framework.py
│   └── ...
│
└── README.md                        # This file
```

---

## Swarm Details

### Financial Analysis Swarm
**Specialties**: Fundamental analysis, technical indicators, ML/FinBERT, portfolio optimization, risk metrics (VaR, Sharpe), SEC compliance

**Example prompts**:
- "Analyze AAPL fundamentals and technicals"
- "Build a momentum indicator for the screener"
- "Calculate portfolio VaR and Sharpe ratio"
- "Review recommendation logic for SEC compliance"

### Data & ML Pipeline Swarm
**Specialties**: Airflow DAGs, ETL design, API rate limiting, TimescaleDB, ML training/serving, data quality

**Example prompts**:
- "Create an Airflow DAG for daily stock data"
- "Optimize the pipeline for 6,000 stocks"
- "Implement API rate limiting with caching"
- "Set up ML model retraining schedule"

### Backend & API Swarm
**Specialties**: FastAPI async patterns, REST API design, WebSocket, PostgreSQL/Redis, OAuth2/JWT, repository pattern

**Example prompts**:
- "Add endpoint for portfolio rebalancing"
- "Implement WebSocket for real-time prices"
- "Optimize the recommendations query"
- "Add OAuth2 authentication flow"

### Security & Compliance Swarm
**Specialties**: SEC 2025 regulations, GDPR data protection, OWASP security, vulnerability assessment, audit logging

**Example prompts**:
- "Implement SEC-compliant audit logging"
- "Review authentication for vulnerabilities"
- "Ensure GDPR compliance for user data export"
- "Conduct security audit of the API"

### Infrastructure & DevOps Swarm
**Specialties**: Docker Compose, Prometheus/Grafana, CI/CD (GitHub Actions), cost optimization (<$50/mo)

**Example prompts**:
- "Set up Grafana monitoring dashboard"
- "Optimize Docker for production"
- "Reduce infrastructure costs"
- "Configure CI/CD pipeline"

### UI & Visualization Swarm
**Specialties**: React/TypeScript, Material-UI, Plotly charts, real-time updates, responsive design, accessibility

**Example prompts**:
- "Create stock price chart component"
- "Design the portfolio dashboard"
- "Add real-time price updates"
- "Improve mobile responsiveness"

### Project & Quality Swarm
**Specialties**: Code review, pytest/Jest testing, architecture decisions (ADRs), documentation, coverage

**Example prompts**:
- "Review the new recommendation endpoint"
- "Add unit tests for portfolio service"
- "Evaluate the caching architecture"
- "Update API documentation"

---

## Multi-Team Coordination

For complex tasks spanning multiple domains, the **team-coordinator** recommends coordination strategies:

### Example: "Add a new stock screener feature"

```
Coordination Strategy:

Phase 1: Financial Analysis Swarm
- Define screening criteria and calculations
- Deliverable: Screening logic specification

Phase 2: Backend API Swarm
- Implement screening endpoints
- Deliverable: API with /screener endpoints

Phase 3: UI Visualization Swarm
- Build screener interface
- Deliverable: React components

Phase 4: Project Quality Swarm
- Review and test all components
- Deliverable: Approved, tested feature
```

---

## Preserved Specialist Agents

These agents remain available for direct, focused tasks:

| Agent | Best For |
|-------|----------|
| `data-science-architect` | Deep dive on data architecture |
| `architecture-reviewer` | System design decisions |
| `code-review-expert` | Detailed code review |
| `godmode-refactorer` | Large-scale refactoring |

---

## Investment Platform Context

All swarms understand the platform's constraints:

- **Budget**: Under $50/month operational cost
- **Scale**: 6,000+ stocks (NYSE, NASDAQ, AMEX)
- **APIs**: Alpha Vantage (25/day), Finnhub (60/min), Polygon (5/min)
- **Compliance**: SEC 2025 regulations, GDPR
- **Stack**: FastAPI, React, PostgreSQL/TimescaleDB, Redis, Docker

---

## Migration from Old System

The previous 397-agent catalog has been archived. The new swarm system:

- **Reduces complexity**: 7 teams vs 397 individual agents
- **Preserves expertise**: Each swarm synthesizes specialist knowledge
- **Improves routing**: Team-coordinator helps select the right team
- **Maintains flexibility**: Individual specialist agents still available

Archived files are in `.claude/archive/` for reference.
