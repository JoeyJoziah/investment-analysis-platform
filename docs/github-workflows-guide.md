# GitHub Actions Workflow Coordination Guide

## Overview

The Investment Analysis Platform uses a comprehensive, coordinated GitHub Actions workflow system that provides:

- **Automated CI/CD pipelines** with intelligent routing
- **PR and issue management** with automated triaging
- **Release automation** with semantic versioning
- **Deployment coordination** across environments
- **Monitoring and notifications** for system health
- **Board synchronization** with GitHub Projects and Notion

## Architecture

### Workflow Hierarchy

```
┌─────────────────────────────────────────┐
│     Workflow Coordinator (Master)       │
│  - Orchestrates all major workflows     │
│  - Manages dependencies                 │
│  - Unified notifications                │
└─────────────────────────────────────────┘
           │
           ├──────────────┬──────────────┬──────────────┐
           │              │              │              │
           ▼              ▼              ▼              ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
    │   CI     │  │ Security │  │  Build   │  │  Deploy  │
    │ Pipeline │  │   Scan   │  │ Pipeline │  │ Pipeline │
    └──────────┘  └──────────┘  └──────────┘  └──────────┘
```

## Core Workflows

### 1. Workflow Coordinator (`workflow-coordinator.yml`)

**Purpose**: Master orchestration workflow that coordinates CI/CD operations

**Triggers**:
- Manual dispatch with workflow type selection

**Workflow Types**:
- `full-ci`: Complete CI pipeline with tests, security, and build
- `fast-ci`: Quick validation with tests only
- `release-candidate`: Full pipeline + deployment to staging/production
- `hotfix`: Emergency deployment with minimal checks
- `security-audit`: Security scanning only
- `performance-check`: Performance tests only

**Usage**:
```bash
# Trigger full CI pipeline
gh workflow run workflow-coordinator.yml \
  --field workflow_type=full-ci

# Deploy release candidate to production
gh workflow run workflow-coordinator.yml \
  --field workflow_type=release-candidate \
  --field environment=production

# Emergency hotfix
gh workflow run workflow-coordinator.yml \
  --field workflow_type=hotfix \
  --field skip_tests=true  # Use with extreme caution!
```

**Features**:
- Intelligent workflow routing based on type
- Conditional execution of test/security/build/deploy stages
- Multi-channel notifications (Slack, email, GitHub)
- Automatic board synchronization on success

### 2. PR Automation (`pr-automation.yml`)

**Purpose**: Automated PR management with intelligent classification

**Triggers**:
- PR opened, synchronized, labeled
- PR review submitted
- Comments on PRs

**Key Features**:

#### Intelligent Classification
Automatically labels PRs based on:
- **Components**: `component:backend`, `component:frontend`, `component:infrastructure`
- **Size**: `size:small` (<100 lines), `size:medium` (<500 lines), `size:large`
- **Type**: `tests`, `documentation`, `dependencies`, `ci-cd`

#### PR Health Checks
- Description completeness validation
- Conventional commit format checking
- Title length validation
- Linked issues detection

#### Auto-merge Support
PRs with the `auto-merge` label will automatically merge when:
- All checks pass
- At least 1 approval received
- PR is mergeable (no conflicts)

#### Stale PR Detection
- Marks PRs stale after 14 days of inactivity
- Sends reminder notifications
- Suggests actions to maintainer

**Example**:
```yaml
# Apply auto-merge label to enable automatic merging
gh pr edit 123 --add-label "auto-merge"
```

### 3. Issue Management (`issue-management.yml`)

**Purpose**: Automated issue triaging and lifecycle management

**Triggers**:
- Issue opened, edited, closed
- Issue comments
- Scheduled daily check for stale issues

**Key Features**:

#### Intelligent Classification
Automatically labels issues based on content:
- **Type**: `type:bug`, `type:feature`, `type:question`, `type:documentation`
- **Priority**: `priority:critical`, `priority:high`, `priority:medium`
- **Component**: `component:backend`, `component:frontend`, `component:database`
- **Special**: `security`, `performance`

#### Security Issue Handling
- Automatic escalation to critical priority
- Immediate team notification
- Guidance on responsible disclosure
- Security advisory creation workflow

#### Duplicate Detection
- Scans existing issues for similar titles
- Comments with potential duplicates
- Helps reduce duplicate work

#### Stale Issue Management
- Marks issues stale after 30 days of inactivity
- Auto-closes after 37 days (7 days post-stale marking)
- Configurable for different issue types

#### First-Time Contributor Welcome
- Detects first-time issue reporters
- Sends welcome message with guidelines
- Links to contributing documentation

### 4. Automated Release (`automated-release.yml`)

**Purpose**: Streamlined release process with semantic versioning

**Triggers**:
- Manual dispatch with release type
- Push to `main` with VERSION file change

**Release Types**:
- `patch`: Bug fixes (0.0.X)
- `minor`: New features (0.X.0)
- `major`: Breaking changes (X.0.0)
- `prerelease`: Alpha/beta/rc versions

**Process**:
1. **Version Calculation**: Automatically bumps version using semantic versioning
2. **Changelog Generation**: Creates release notes from commit history
3. **Pre-release Validation**: Runs quick smoke tests
4. **GitHub Release**: Creates tagged release with changelog
5. **Deployment Trigger**: Automatically deploys stable releases to production
6. **Notifications**: Multi-channel release announcements

**Usage**:
```bash
# Create patch release
gh workflow run automated-release.yml \
  --field release_type=patch

# Create prerelease
gh workflow run automated-release.yml \
  --field release_type=prerelease \
  --field prerelease_tag=beta

# Create major release (breaking changes)
gh workflow run automated-release.yml \
  --field release_type=major
```

**Changelog Format**:
The workflow automatically categorizes commits into:
- Features (feat:, feature:)
- Bug Fixes (fix:, bugfix:)
- Performance (perf:)
- Documentation (docs:)
- Tests (test:)
- Refactoring (refactor:)
- CI/CD (ci:)
- Maintenance (chore:)

### 5. Monitoring & Notifications (`monitoring-notifications.yml`)

**Purpose**: Centralized monitoring and health tracking

**Triggers**:
- Workflow completion (any major workflow)
- Hourly health checks
- Daily summary at 9 AM UTC
- Manual dispatch

**Key Features**:

#### Workflow Status Monitoring
- Tracks all workflow executions
- Immediate notifications on failures
- Links to failed workflow logs

#### Repository Health Checks
Calculates health score (0-100) based on:
- Critical issues (-10 points each)
- High priority issues (-5 points each)
- Recent workflow failures (-3 points each)
- Stale PRs (-2 points each)

**Health Status Levels**:
- 💚 **Healthy** (80-100): Repository in good state
- 💛 **Warning** (60-79): Attention needed
- ❤️  **Critical** (<60): Immediate action required

#### Daily Summary
Automated daily report including:
- Commits in last 24 hours
- PRs opened/merged
- Issues opened/closed
- Key metrics and trends

#### Deployment Monitoring
- Tracks all deployment workflows
- Environment-specific monitoring
- Success/failure tracking
- Rollback recommendations

## CI/CD Pipeline (`ci.yml`)

The main CI pipeline includes:

### Backend Pipeline
1. **Code Quality**: Black, isort, flake8, mypy, pylint
2. **Security Analysis**: Bandit, safety checks
3. **Testing**: Unit and integration tests with 85% coverage threshold
4. **Test Services**: PostgreSQL and Redis containers

### Frontend Pipeline
1. **Testing**: Vitest test runner
2. **Linting**: ESLint
3. **Formatting**: Prettier
4. **Coverage**: LCOV reports to Codecov

### Docker Build
- Multi-platform builds (amd64, arm64)
- Layer caching for fast builds
- Automatic image tagging

### Integration Tests
- Full stack testing with docker-compose
- End-to-end validation
- Only on main branch pushes

## Deployment Workflows

### Staging Deployment (`staging-deploy.yml`)
- Triggered on develop branch pushes
- Quick validation tests
- Blue-green deployment strategy
- Smoke tests post-deployment

### Production Deployment (`production-deploy.yml`)
- Triggered on release creation or manual dispatch
- Comprehensive pre-deployment tests
- Security scanning with vulnerability gates
- Database migration handling
- Blue-green deployment with health checks
- Automatic rollback on failure
- Post-deployment monitoring setup

## Board Synchronization

### GitHub Projects Sync (`board-sync.yml`)
- Real-time sync on issue/PR events
- Daily scheduled sync
- Automatic status updates
- Label-based organization

### Notion Integration (`notion-github-sync.yml`)
- Bidirectional sync with Notion database
- Conflict resolution strategies
- Custom field mapping
- Scheduled and event-triggered sync

## Notification Channels

### Slack
**Setup**:
```bash
# Add Slack webhook to repository secrets
gh secret set SLACK_WEBHOOK_URL --body "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
```

**Notifications sent for**:
- Workflow failures
- Deployment completions
- Release creations
- Critical health alerts
- Daily summaries

### Email
**Setup**:
```bash
# Configure SMTP credentials
gh secret set SMTP_USERNAME --body "your-email@gmail.com"
gh secret set SMTP_PASSWORD --body "your-app-password"
gh secret set NOTIFICATION_EMAIL --body "team@example.com"
gh secret set EMAIL_ENABLED --body "true"
```

### GitHub
- Native GitHub notifications
- PR/Issue comments
- Check run summaries
- Workflow summaries

## Best Practices

### 1. PR Guidelines
- Use conventional commit format in PR titles
- Link related issues with "Closes #123" or "Fixes #456"
- Add meaningful descriptions (>50 characters)
- Request reviews from appropriate teams
- Use `auto-merge` label for straightforward changes

### 2. Issue Management
- Use clear, descriptive titles
- Include reproduction steps for bugs
- Specify expected vs actual behavior
- Add relevant labels manually if auto-classification misses
- Reference related issues/PRs

### 3. Release Process
- Always run `release-candidate` workflow type before production releases
- Use prerelease for beta/rc versions
- Review automatically generated changelog before publishing
- Ensure all tests pass before triggering release

### 4. Security
- Never commit secrets to repository
- Use GitHub secrets for all sensitive data
- Mark security issues appropriately
- Follow responsible disclosure for vulnerabilities
- Rotate secrets regularly

### 5. Monitoring
- Review daily health summaries
- Address critical health alerts within 24 hours
- Keep stale PRs/issues under control
- Monitor workflow failure trends

## Troubleshooting

### Workflow Failures

**1. Test Failures**
```bash
# View test results
gh run view <run-id> --log-failed

# Re-run failed jobs
gh run rerun <run-id> --failed
```

**2. Build Failures**
- Check Docker layer cache
- Verify dependencies are up to date
- Review build logs for specific errors

**3. Deployment Failures**
- Check deployment logs
- Verify secrets are configured
- Review health check failures
- Use rollback workflow if needed

### Health Score Issues

**Low Health Score (<60)**:
1. Address critical issues first
2. Merge or close stale PRs
3. Fix failing workflows
4. Review and close outdated issues

**Stale Items**:
```bash
# Bulk close stale issues
gh issue list --label stale --state open --json number --jq '.[].number' | \
  xargs -I {} gh issue close {} --reason "stale"
```

## Advanced Configuration

### Custom Workflow Routing

Edit `workflow-coordinator.yml` to add custom workflow types:

```yaml
case "$WORKFLOW_TYPE" in
  "custom-type")
    RUN_TESTS="true"
    RUN_SECURITY="true"
    RUN_BUILD="false"
    RUN_DEPLOY="false"
    NOTIFICATION_CHANNELS="slack"
    ;;
esac
```

### Custom Labels

Add to repository settings or via API:
```bash
gh label create "priority:urgent" --color "d73a4a" --description "Urgent priority"
gh label create "component:ml-models" --color "0366d6" --description "ML models component"
```

### Custom Health Scoring

Modify `monitoring-notifications.yml` health calculation:

```bash
# Adjust point deductions
SCORE=$((SCORE - CRITICAL * 15))  # Increase critical impact
SCORE=$((SCORE - HIGH * 3))       # Decrease high impact
```

## Metrics and Analytics

### Key Metrics Tracked
- Workflow success/failure rates
- Average PR merge time
- Issue resolution time
- Deployment frequency
- MTTR (Mean Time To Recovery)
- Code coverage trends

### Accessing Metrics
```bash
# Recent workflow runs
gh run list --limit 50

# PR statistics
gh pr list --state all --json number,createdAt,mergedAt,closedAt

# Issue statistics
gh issue list --state all --json number,createdAt,closedAt
```

## Integration with External Tools

### Codecov
Automatic coverage reporting:
```bash
gh secret set CODECOV_TOKEN --body "your-codecov-token"
```

### Sentry
Error tracking integration:
```bash
gh secret set SENTRY_DSN --body "your-sentry-dsn"
```

### DataDog
Performance monitoring:
```bash
gh secret set DATADOG_API_KEY --body "your-datadog-key"
```

## Security

### Secrets Required

**Essential**:
- `GITHUB_TOKEN`: Auto-provided by GitHub
- `SLACK_WEBHOOK_URL`: For Slack notifications

**Optional**:
- `CODECOV_TOKEN`: For code coverage
- `NOTION_TOKEN`: For Notion sync
- `SMTP_USERNAME`, `SMTP_PASSWORD`: For email notifications
- `PRODUCTION_KUBECONFIG`: For Kubernetes deployments
- `PRODUCTION_DATABASE_URL`: For production DB

### Secret Rotation

Rotate secrets quarterly:
```bash
# Update secret
gh secret set SECRET_NAME --body "new-value"

# Verify update
gh secret list
```

## Support and Contributions

For issues with workflows:
1. Check workflow logs: `gh run view <run-id> --log`
2. Review this documentation
3. Create an issue with `component:ci-cd` label
4. Contact DevOps team

For improvements:
1. Fork repository
2. Modify workflow files
3. Test in your fork
4. Submit PR with clear description
5. Add `component:ci-cd` label

## Changelog

### v3.0.0 (2026-01-27)
- Added Workflow Coordinator for unified orchestration
- Implemented intelligent PR automation
- Enhanced issue management with stale detection
- Added automated release pipeline
- Implemented centralized monitoring and notifications
- Integrated health scoring system

### v2.5.0 (2026-01-23)
- Added GitHub Swarm automation
- Implemented board synchronization
- Enhanced security scanning

### v2.0.0 (2026-01-15)
- Complete CI/CD pipeline restructure
- Added production deployment workflow
- Implemented release management
