# GitHub Swarm for Investment Analysis Platform

A comprehensive GitHub automation swarm for managing the investment analysis platform repository.

## Overview

This swarm provides automated repository management through 6 specialized agents:

| Agent | Role | Purpose |
|-------|------|---------|
| **Issue Triager** | Categorize | Auto-labels issues by type, priority, and component |
| **PR Reviewer** | Review | Reviews code quality, security, and best practices |
| **Test Agent** | Test | Identifies test gaps and coverage issues |
| **Security Agent** | Security | Scans for vulnerabilities and secrets |
| **Documentation Agent** | Document | Tracks documentation updates needed |
| **Infrastructure Agent** | Monitor | Monitors Docker, CI/CD, and costs |

## Configuration

**Focus**: Development (active development with many pending changes)
**Agents**: 6 specialized agents
**Topology**: Hierarchical (coordinator-based)
**Repository**: JoeyJoziah/investment-analysis-platform

### Features Enabled

- Auto PR enhancements (labeling, reviewer suggestions)
- Issue labeling (type, priority, component)
- Code review automation (quality, security, tests)
- Daily security scans
- Hourly infrastructure health checks

## Quick Start

```bash
# Initialize the swarm
/github-swarm init

# Check status
/github-swarm status

# Analyze a PR
/github-swarm analyze-pr 42

# Run security scan
/github-swarm security-scan

# Full repository assessment
/github-swarm full-scan
```

## File Structure

```
.claude/agents/github-swarm/
├── README.md                    # This file
├── swarm-config.json           # Swarm configuration
├── github-swarm-coordinator.md # Coordinator agent
├── issue-triager.md            # Issue triage agent
├── pr-reviewer.md              # PR review agent
├── test-agent.md               # Test analysis agent
├── security-agent.md           # Security scan agent
├── documentation-agent.md      # Documentation tracking agent
└── infrastructure-agent.md     # Infrastructure monitoring agent
```

## GitHub Actions Integration

The swarm integrates with GitHub Actions via `.github/workflows/github-swarm.yml`:

| Trigger | Agent | Action |
|---------|-------|--------|
| `issues.opened` | Issue Triager | Auto-triage and label |
| `pull_request.opened` | PR Reviewer, Test, Security | Full PR analysis |
| `push` (main) | Security, Docs, Infra | Post-merge checks |
| `schedule` (daily) | Security | Full security scan |
| `schedule` (hourly) | Infrastructure | Health check |

## Labels

### Type Labels
- `bug` - Something isn't working
- `feature` - New feature request
- `enhancement` - Improvement
- `documentation` - Doc updates
- `infrastructure` - DevOps tasks
- `security` - Security concerns
- `performance` - Performance issues
- `testing` - Testing improvements

### Component Labels
- `backend` - FastAPI backend
- `frontend` - React frontend
- `ml-models` - ML/AI models
- `database` - PostgreSQL/TimescaleDB
- `data-pipeline` - ETL pipelines
- `api` - API endpoints

### Priority Labels
- `P0-critical` - Immediate action
- `P1-high` - High priority
- `P2-medium` - Medium priority
- `P3-low` - Low priority

### Status Labels
- `needs-triage` - Awaiting triage
- `in-progress` - Work in progress
- `ready-for-review` - Ready for review
- `blocked` - Blocked by dependency

## Budget Monitoring

The swarm monitors costs against the $50/month target:

- **Alert threshold**: 80% ($40)
- **Critical threshold**: 90% ($45)

## Integration with Claude-Flow

```javascript
// Initialize swarm
mcp__claude-flow__swarm_init {
  topology: "hierarchical",
  maxAgents: 6,
  config: ".claude/agents/github-swarm/swarm-config.json"
}

// Spawn agents
mcp__claude-flow__agent_spawn { type: "issue-triager" }
mcp__claude-flow__agent_spawn { type: "pr-reviewer" }
mcp__claude-flow__agent_spawn { type: "test-agent" }
mcp__claude-flow__agent_spawn { type: "security-agent" }
mcp__claude-flow__agent_spawn { type: "documentation-agent" }
mcp__claude-flow__agent_spawn { type: "infrastructure-agent" }
```

## Metrics Tracked

- Issue triage latency (target: <5 minutes)
- PR review latency (target: <30 minutes)
- Security scan coverage
- Test coverage trends
- CI/CD success rate
- Monthly infrastructure costs

## Related Documentation

- [CLAUDE.md](/CLAUDE.md) - Main Claude Code instructions
- [Integration Manifest](/.claude/integration-manifest.json) - Full component registry
- [Infrastructure Swarm](/.claude/agents/infrastructure-devops-swarm.md) - DevOps swarm
