# GitHub Swarm Commands

Create and manage a specialized swarm for GitHub repository management for the investment analysis platform.

## Quick Reference

```bash
# Initialize swarm
/github-swarm init

# Check status
/github-swarm status

# Analyze PR
/github-swarm analyze-pr 42

# Security scan
/github-swarm security-scan

# Full repository scan
/github-swarm full-scan
```

## Usage

```bash
npx claude-flow github swarm [options]
```

Or with slash commands:
```bash
/github-swarm <command> [options]
```

## Options

- `--repository, -r <owner/repo>` - Target GitHub repository (default: JoeyJoziah/investment-analysis-platform)
- `--agents, -a <number>` - Number of specialized agents (default: 6)
- `--focus, -f <type>` - Focus area: maintenance, development, review, triage
- `--auto-pr` - Enable automatic pull request enhancements
- `--issue-labels` - Auto-categorize and label issues
- `--code-review` - Enable AI-powered code reviews

## Examples

### Basic GitHub swarm

```bash
npx claude-flow github swarm --repository JoeyJoziah/investment-analysis-platform
```

### Development-focused swarm (current configuration)

```bash
npx claude-flow github swarm -r JoeyJoziah/investment-analysis-platform \
  -f development -a 6 --auto-pr --issue-labels --code-review
```

### Full-featured triage swarm

```bash
npx claude-flow github swarm -r JoeyJoziah/investment-analysis-platform \
  -a 6 -f triage --issue-labels --auto-pr
```

## Swarm Configuration

Configuration file: `.claude/agents/github-swarm/swarm-config.json`

### Agent Types (6 Agents)

| Agent | Purpose | Trigger Events |
|-------|---------|----------------|
| **Issue Triager** | Categorize issues by type, priority, components | `issues.opened`, `issues.edited` |
| **PR Reviewer** | Code quality, security, best practices review | `pull_request.opened`, `pull_request.synchronize` |
| **Documentation Agent** | Track doc updates, changelog, API docs | `pull_request.merged`, `push` |
| **Test Agent** | Coverage analysis, test gap detection | `pull_request.opened`, `pull_request.synchronize` |
| **Security Agent** | Vulnerability scan, secret detection | `pull_request.opened`, `push`, `schedule.daily` |
| **Infrastructure Agent** | Docker, CI/CD, deployment monitoring | `workflow_run.completed`, `schedule.hourly` |

## Commands Reference

### Initialization & Status

```bash
# Initialize swarm for repository
/github-swarm init [--repo owner/repo]

# Check swarm health and agent status
/github-swarm status [--detailed]

# View performance metrics
/github-swarm metrics [--period 7d]
```

### Issue Management

```bash
# Triage specific issue
/github-swarm triage-issue <number> [--priority high] [--assign @user]

# List issues with swarm analysis
/github-swarm list-issues [--state open] [--label bug]
```

### Pull Request Management

```bash
# Full PR analysis (quality + tests + security)
/github-swarm analyze-pr <number> [--skip security] [--force]

# Approve PR after checks pass
/github-swarm approve-pr <number> [--message "LGTM"]

# Request changes
/github-swarm request-changes <number> --reason "Security issue"
```

### Security Operations

```bash
# Repository security scan
/github-swarm security-scan [--full] [--create-issues]

# Scan specific PR
/github-swarm security-scan --pr 42

# Check for exposed secrets
/github-swarm check-secrets [--path backend/]

# Dependency vulnerability audit
/github-swarm dependency-audit [--fix]
```

### Test Operations

```bash
# Analyze test coverage
/github-swarm test-coverage [--pr 42] [--detailed]

# Find missing tests
/github-swarm find-test-gaps [--component backend]
```

### Documentation Operations

```bash
# Audit documentation freshness
/github-swarm doc-audit [--path docs/api/]

# Generate changelog entry
/github-swarm update-changelog --version 1.2.0
```

### Infrastructure Operations

```bash
# Infrastructure health check
/github-swarm infra-check [--detailed] [--service backend]

# Cost report
/github-swarm cost-report [--period month]

# Validate Docker configurations
/github-swarm validate-docker [--file docker-compose.prod.yml]

# CI/CD health check
/github-swarm ci-health [--days 7]
```

### Reporting

```bash
# Daily activity summary
/github-swarm daily-summary [--date 2026-01-24]

# Weekly report
/github-swarm weekly-report [--week 4]

# Complete repository analysis
/github-swarm full-scan
```

## Workflows

### Issue Triage Workflow

1. Scan issue content and metadata
2. Classify by type (bug, feature, security, etc.)
3. Assess priority (P0-P3)
4. Identify affected components
5. Detect duplicates
6. Apply appropriate labels
7. Suggest assignees
8. Post triage summary comment

### PR Review Workflow

1. Fetch PR details and diff
2. **Parallel execution**:
   - PR Reviewer: Code quality analysis
   - Test Agent: Coverage verification
   - Security Agent: Vulnerability scan
3. Aggregate results
4. Generate comprehensive review
5. Make approval/change request decision
6. Apply labels

### Security Scan Workflow

1. Dependency vulnerability scan (pip-audit)
2. Secret detection (gitleaks patterns)
3. Code pattern analysis (SQL injection, XSS)
4. Authentication review
5. Generate security report
6. Create issues for findings (optional)

### Infrastructure Health Workflow

1. Docker configuration validation
2. Service health checks
3. CI/CD success rate analysis
4. Cost monitoring vs $50/month budget
5. Deployment readiness assessment
6. Generate health report

## Integration with Claude-Flow

```javascript
// Initialize swarm within claude-flow
mcp__claude-flow__swarm_init {
  topology: "hierarchical",
  maxAgents: 6,
  name: "github-swarm",
  config: ".claude/agents/github-swarm/swarm-config.json"
}

// Spawn all agents
mcp__claude-flow__agent_spawn { type: "issue-triager" }
mcp__claude-flow__agent_spawn { type: "pr-reviewer" }
mcp__claude-flow__agent_spawn { type: "test-agent" }
mcp__claude-flow__agent_spawn { type: "security-agent" }
mcp__claude-flow__agent_spawn { type: "documentation-agent" }
mcp__claude-flow__agent_spawn { type: "infrastructure-agent" }

// Orchestrate tasks
mcp__claude-flow__task_orchestrate {
  task: "Full repository analysis",
  strategy: "parallel"
}
```

## Labels Created

### Type Labels
- `bug` (red) - Something isn't working
- `feature` (blue) - New feature request
- `enhancement` (cyan) - Improvement to existing feature
- `documentation` (blue) - Documentation updates
- `infrastructure` (purple) - Docker, CI/CD, deployment
- `security` (red) - Security vulnerability
- `performance` (yellow) - Performance optimization
- `testing` (blue) - Testing improvements

### Component Labels
- `backend` - FastAPI backend
- `frontend` - React frontend
- `ml-models` - ML/AI models
- `database` - PostgreSQL/TimescaleDB
- `data-pipeline` - ETL and data processing
- `api` - API endpoints

### Priority Labels
- `P0-critical` - Immediate action required
- `P1-high` - High priority
- `P2-medium` - Medium priority
- `P3-low` - Low priority

### Status Labels
- `needs-triage` - Needs triage by maintainer
- `in-progress` - Work in progress
- `ready-for-review` - Ready for code review
- `blocked` - Blocked by dependency

## Budget Monitoring

The infrastructure agent monitors costs against the $50/month target:
- Alert at 80% ($40)
- Critical at 90% ($45)

```bash
# Check current cost status
/github-swarm cost-report
```

## See Also

- `.claude/agents/github-swarm/` - Agent definitions
- `.claude/agents/github-swarm/swarm-config.json` - Swarm configuration
- `repo analyze` - Deep repository analysis
- `pr enhance` - Enhance pull requests
- `issue triage` - Intelligent issue management
- `code review` - Automated reviews
