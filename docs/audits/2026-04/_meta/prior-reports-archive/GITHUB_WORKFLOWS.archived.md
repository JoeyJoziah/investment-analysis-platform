> **ARCHIVED 2026-04-27 by 14-ci-cd-workflows**
> Original: docs/GITHUB_WORKFLOWS.md
> Validation summary: see ../../reports/14-ci-cd-workflows.md §2 for per-claim status.

# GitHub Workflows Documentation

**Investment Analysis Platform - CI/CD Pipeline**

Last Updated: 2026-01-29

---

## Table of Contents

1. [Overview](#overview)
2. [Workflow Catalog](#workflow-catalog)
3. [Core Workflows](#core-workflows)
4. [Deployment Workflows](#deployment-workflows)
5. [Automation Workflows](#automation-workflows)
6. [Utility Workflows](#utility-workflows)
7. [Workflow Dependencies](#workflow-dependencies)
8. [Trigger Matrix](#trigger-matrix)
9. [Environment Variables](#environment-variables)
10. [Secrets Configuration](#secrets-configuration)
11. [Troubleshooting Guide](#troubleshooting-guide)
12. [Workflow Status Badges](#workflow-status-badges)

---

## Overview

The Investment Analysis Platform uses GitHub Actions for comprehensive CI/CD automation. The workflow system includes 26 workflows organized into 4 categories:

- **Core Workflows**: Build, test, and quality assurance
- **Deployment Workflows**: Staging, production, and release management
- **Automation Workflows**: Issue management, PR automation, board sync
- **Utility Workflows**: Monitoring, notifications, coordination

### Architecture Principles

- **Cost Optimization**: Smaller runners, job timeouts, intelligent caching
- **Fail-Fast Strategy**: Critical failures stop dependent jobs
- **Concurrency Control**: Prevents redundant builds
- **Reusability**: Shared workflows for common operations
- **Security-First**: Multiple security scans before deployment

---

## Workflow Catalog

### Core Workflows

| Workflow | File | Purpose | Triggers |
|----------|------|---------|----------|
| CI Pipeline | `ci.yml` | Main continuous integration | Push (main/develop), PR |
| Security Scanning | `security-scan.yml` | Comprehensive security checks | Daily, Push, PR, Manual |
| Comprehensive Testing | `comprehensive-testing.yml` | Full test suite | Push, PR, Daily |
| Type Checking | `type-check.yml` | TypeScript/Mypy validation | Push, PR |
| Migration Check | `migration-check.yml` | Database migration validation | Push, PR |

### Deployment Workflows

| Workflow | File | Purpose | Triggers |
|----------|------|---------|----------|
| Staging Deploy | `staging-deploy.yml` | Deploy to staging | Push (main), Manual |
| Production Deploy | `production-deploy.yml` | Production deployment | Release, Manual |
| Automated Release | `automated-release.yml` | Semantic versioning releases | Manual, VERSION file |
| Release Management | `release-management.yml` | Release coordination | Release events |

### Automation Workflows

| Workflow | File | Purpose | Triggers |
|----------|------|---------|----------|
| PR Automation | `pr-automation.yml` | Intelligent PR management | PR events |
| Issue Management | `issue-management.yml` | Issue triaging & lifecycle | Issue events, Daily |
| Board Sync | `board-sync.yml` | GitHub Projects sync | Issues, PRs, Daily |
| Notion-GitHub Sync | `notion-github-sync.yml` | Bi-directional Notion sync | Manual, Scheduled |
| Auto-Sync | `auto-sync.yml` | Automated synchronization | Various triggers |

### Utility Workflows

| Workflow | File | Purpose | Triggers |
|----------|------|---------|----------|
| Workflow Coordinator | `workflow-coordinator.yml` | Orchestrate workflows | Manual |
| Monitoring & Notifications | `monitoring-notifications.yml` | Alert system | Various events |
| Performance Monitoring | `performance-monitoring.yml` | Performance tracking | Scheduled |
| Daily Pipeline Validation | `daily-pipeline-validation.yml` | Health checks | Daily |
| Cleanup | `cleanup.yml` | Resource cleanup | Scheduled |
| Dependency Updates | `dependency-updates.yml` | Automated updates | Weekly |

---

## Core Workflows

### CI Pipeline (`ci.yml`)

**Purpose**: Main continuous integration pipeline with parallel job execution

**Triggers**:
- Push to `main` or `develop`
- Pull requests to `main` or `develop`

**Jobs**:
1. **backend-quality** (15 min timeout)
   - Black, isort, flake8, mypy, pylint
   - Bandit security scan, Safety vulnerability check
   - Uploads quality reports

2. **backend-test** (30 min timeout)
   - Matrix: Python 3.12, test suites (unit, integration)
   - PostgreSQL 15 + Redis 7 services
   - 85% coverage requirement
   - Codecov integration

3. **frontend-test** (15 min timeout)
   - Vitest unit tests with coverage
   - ESLint quality checks
   - Prettier formatting validation

4. **frontend-quality** (10 min timeout)
   - ESLint with JSON output
   - Prettier validation

5. **docker-build** (20 min timeout)
   - Multi-platform builds (linux/amd64, linux/arm64)
   - GitHub Actions cache
   - Only on push or with `docker-build` label

6. **integration-test** (25 min timeout)
   - Full docker-compose stack
   - Only on main branch pushes

7. **notify-failure** (5 min timeout)
   - Slack and email notifications on failure

8. **ci-summary** (5 min timeout)
   - Generates workflow summary

9. **sync-boards** (5 min timeout)
   - Syncs GitHub Projects and Notion

**Key Features**:
- Concurrency group prevents duplicate runs
- Cost optimization with timeouts and selective execution
- Comprehensive caching strategy
- Board synchronization after completion

**Example Manual Trigger**:
```bash
gh workflow run ci.yml
```

---

### Security Scanning (`security-scan.yml`)

**Purpose**: Multi-layered security scanning across all components

**Triggers**:
- Daily at 2 AM UTC (scheduled)
- Push to main/develop
- Pull requests
- Manual with scan type selection

**Scan Types** (Manual Dispatch):
- `all`: Complete security audit
- `code`: Static code analysis only
- `dependencies`: Dependency vulnerabilities
- `secrets`: Secret scanning
- `containers`: Docker image scanning

**Jobs**:

1. **code-security**
   - CodeQL (Python, JavaScript, TypeScript)
   - Bandit (Python security)
   - Semgrep (multi-language)
   - ESLint security plugin

2. **dependency-security**
   - Safety (Python)
   - pip-audit
   - npm audit
   - Snyk integration

3. **secret-scanning**
   - TruffleHog (verified secrets)
   - GitLeaks (patterns)
   - Custom patterns for API keys
   - Environment file detection

4. **container-security**
   - Trivy vulnerability scanner
   - Hadolint (Dockerfile best practices)
   - Dockle (container hardening)

5. **security-report**
   - Consolidated report generation
   - PR comments on findings
   - SARIF upload to Security tab

6. **sync-security-findings**
   - Creates critical security issues
   - Syncs to GitHub Projects
   - Notion integration

**Security Gates**:
- CRITICAL vulnerabilities block deployment
- HIGH vulnerabilities logged but don't block
- Automatic issue creation for failures

**Example Manual Trigger**:
```bash
gh workflow run security-scan.yml -f scan_type=all
```

---

### Comprehensive Testing (`comprehensive-testing.yml`)

**Purpose**: Full test suite including unit, integration, E2E, performance, and security tests

**Triggers**:
- Push to main/develop
- Pull requests
- Daily at 2 AM UTC

**Jobs**:

1. **security-scan**: Safety, Bandit, Semgrep
2. **code-quality**: Black, isort, flake8, mypy, pylint
3. **unit-tests**: Backend with 85% coverage requirement
4. **integration-tests**: API integration with services
5. **performance-tests**: Load testing (on schedule or `[run-performance-tests]`)
6. **financial-model-tests**: Model validation and backtesting
7. **security-compliance-tests**: Authentication, authorization tests
8. **frontend-tests**: React unit and E2E tests
9. **docker-build**: Multi-stage container builds
10. **e2e-tests**: Full application stack tests (main branch only)
11. **test-report**: Consolidated reporting with PR comments

**Performance Test Conditions**:
- Scheduled runs (daily)
- Manual trigger with commit message `[run-performance-tests]`

**Example Manual Trigger**:
```bash
# Trigger with performance tests
git commit -m "feat: new feature [run-performance-tests]"
git push
```

---

## Deployment Workflows

### Staging Deployment (`staging-deploy.yml`)

**Purpose**: Deploy to staging environment for testing

**Triggers**:
- Push to `main` branch
- Manual dispatch with environment selection

**Jobs**:

1. **build-images** (Multi-platform Docker builds)
   - Backend and frontend images
   - GitHub Container Registry (ghcr.io)
   - SBOM generation (SPDX format)
   - Image tags: `staging-latest`, `staging-<sha>`

2. **security-scan**
   - Trivy container scanning
   - SARIF reports to Security tab
   - Critical vulnerability gate

3. **deploy-staging**
   - Kubernetes deployment
   - Database migrations
   - Rolling update strategy
   - Health verification

4. **smoke-tests**
   - Health checks (API, DB, Cache)
   - Basic API tests
   - Frontend availability

5. **performance-test**
   - Locust load testing (5 min, 10 users)
   - Performance baseline tracking

6. **deployment-summary**
   - Slack notifications
   - GitHub summary
   - PR comments (if applicable)

7. **rollback** (On Failure)
   - Automatic rollback to previous version
   - Notification to team

**Environment Variables**:
- `STAGING_DATABASE_URL`
- `STAGING_REDIS_URL`
- `STAGING_API_URL`
- `STAGING_FRONTEND_URL`

**Example Manual Trigger**:
```bash
gh workflow run staging-deploy.yml -f environment=staging
```

---

### Production Deployment (`production-deploy.yml`)

**Purpose**: Production deployment with comprehensive safety checks

**Triggers**:
- GitHub release published
- Manual dispatch with tag specification

**Concurrency**: Single production deployment at a time

**Jobs**:

1. **validate-release**
   - Version format validation (X.Y.Z)
   - Pre-release detection
   - Release metadata extraction

2. **pre-deployment-tests** (Skippable for hotfixes)
   - Critical test suite
   - 85% coverage requirement
   - Fast fail on errors

3. **build-production-images**
   - Multi-platform builds
   - Production tags: `vX.Y.Z`, `X.Y`, `production-latest`, `stable`
   - SBOM generation and signing

4. **security-scan**
   - Comprehensive Trivy scans
   - SARIF report generation
   - **CRITICAL vulnerability gate** (blocks deployment)

5. **deploy-production**
   - Blue-green deployment strategy
   - Database migrations with retry logic
   - Kubernetes rollout with health checks
   - Production smoke tests
   - Deployment tagging

6. **post-deployment-monitoring**
   - Enhanced monitoring setup
   - Grafana/PagerDuty integration

7. **emergency-rollback** (On Failure)
   - Automatic rollback to previous images
   - Team notification

8. **post-deploy-sync**
   - Mark issues as deployed
   - Sync boards (GitHub Projects, Notion)

**Safety Features**:
- Backup of current deployment before rollout
- Smoke tests (health, DB, cache, frontend)
- Automatic rollback on failure
- Manual approval required (production environment)

**Example Manual Trigger**:
```bash
# Standard deployment
gh workflow run production-deploy.yml -f tag=v1.2.3

# Emergency hotfix (skip tests)
gh workflow run production-deploy.yml -f tag=v1.2.4 -f skip_tests=true
```

---

### Automated Release (`automated-release.yml`)

**Purpose**: Semantic versioning and automated changelog generation

**Triggers**:
- Manual dispatch with release type
- Push to `VERSION` file on main

**Release Types**:
- `patch`: X.Y.Z → X.Y.(Z+1)
- `minor`: X.Y.Z → X.(Y+1).0
- `major`: X.Y.Z → (X+1).0.0
- `prerelease`: X.Y.Z → X.Y.(Z+1)-rc.1

**Jobs**:

1. **calculate-version**
   - Semantic version calculation
   - Latest tag detection
   - Version bumping logic
   - Pre-release detection

2. **generate-changelog**
   - Conventional commit parsing
   - Categorized changes (Features, Bug Fixes, etc.)
   - Release notes generation

3. **pre-release-checks**
   - Version format validation
   - Quick smoke tests

4. **create-release**
   - Git tag creation
   - GitHub release with changelog
   - Draft/Prerelease flags

5. **trigger-deployment** (Stable releases only)
   - Triggers production-deploy.yml
   - Only for non-prerelease versions

6. **notify-release**
   - Slack notifications
   - Release summary

**Example Manual Trigger**:
```bash
# Minor version bump
gh workflow run automated-release.yml -f release_type=minor

# Pre-release candidate
gh workflow run automated-release.yml -f release_type=prerelease -f prerelease_tag=rc
```

---

## Automation Workflows

### PR Automation (`pr-automation.yml`)

**Purpose**: Intelligent PR management with classification and health checks

**Triggers**:
- PR opened, synchronized, ready_for_review, reopened, labeled/unlabeled
- PR review submitted/dismissed
- Issue comments created

**Jobs**:

1. **pr-classifier**
   - Automated labeling (component, size, type)
   - Size analysis (small/medium/large)
   - Comment with metrics

2. **assign-reviewers**
   - Intelligent reviewer assignment by component
   - Team routing (backend, frontend, devops, security)

3. **pr-health-check**
   - Description quality check
   - Linked issue detection
   - Conventional commit format validation
   - Title length check
   - Breaking change detection

4. **auto-merge-check**
   - Requires `auto-merge` label
   - Checks: All CI passed, 1+ approval, mergeable
   - Squash merge with branch deletion

5. **stale-pr-check**
   - 14+ days inactive → `stale` label
   - Reminder comment

6. **pr-summary**
   - GitHub Actions summary generation

**Label Categories**:
- **Component**: `component:backend`, `component:frontend`, `component:infrastructure`, `component:ci-cd`
- **Size**: `size:small` (<100 lines), `size:medium` (<500 lines), `size:large` (500+ lines)
- **Type**: `tests`, `documentation`, `dependencies`

**Example**: PR opened → Automatically classified, labeled, health checked, and reviewers assigned

---

### Issue Management (`issue-management.yml`)

**Purpose**: Intelligent issue triaging, classification, and lifecycle management

**Triggers**:
- Issue opened, edited, closed, reopened, labeled, assigned
- Issue comments
- Daily at 3 AM UTC (stale check)

**Jobs**:

1. **classify-issue**
   - Type detection (bug, feature, question, documentation)
   - Priority classification (critical, high, medium)
   - Component assignment (backend, frontend, database, CI/CD)
   - Security/performance tagging
   - Automated triage comment

2. **security-issue-handler**
   - Auto-escalates security issues to critical
   - Security team notification
   - Responsible disclosure guidance
   - 24-hour review SLA

3. **duplicate-detector**
   - Searches similar open issues
   - Comments with potential duplicates

4. **completion-checker**
   - Detects linked PRs on close
   - Resolution summary comment

5. **stale-issue-check** (Scheduled)
   - 30+ days inactive → `stale` label
   - 37+ days (7 after stale) → Auto-close
   - Reopening guidance

6. **welcome-contributor**
   - First-time contributor detection
   - Welcome message with resources

**Classification Keywords**:
- **Bug**: "bug", "error", "exception", "crash", "fail", "broken"
- **Feature**: "feature", "enhancement", "add", "new", "implement"
- **Question**: "question", "?", "how to", "help"
- **Critical**: "urgent", "critical", "production", "blocking", "emergency"

---

### Board Sync (`board-sync.yml`)

**Purpose**: Synchronize GitHub issues/PRs with GitHub Projects and Notion

**Triggers**:
- Issue events (opened, edited, closed, labeled, assigned)
- PR events (opened, closed, merged, labeled)
- Daily at 6 AM UTC
- Manual with sync type selection

**Sync Types** (Manual):
- `full`: Issues + PRs + Board
- `issues-only`: Only issues
- `board-only`: Only PRs
- `report`: Generate report only

**Jobs**:

1. **issue-sync**: Add new issues to project board
2. **pr-sync**: Add PRs, handle merged PRs
3. **scheduled-sync**: Full sync of all open items
4. **manual-sync**: User-requested sync with options
5. **notion-sync**: Bi-directional Notion database sync

**Notion Integration**:
- Syncs issues to Notion database
- Maps GitHub labels to Notion properties
- Creates pages for new issues
- Reads `database_id` from `.github/board-sync.yml`

**Example Manual Trigger**:
```bash
# Full sync
gh workflow run board-sync.yml -f sync_type=full

# Report only
gh workflow run board-sync.yml -f sync_type=report
```

---

## Utility Workflows

### Workflow Coordinator (`workflow-coordinator.yml`)

**Purpose**: Master orchestration for coordinated CI/CD operations

**Triggers**: Manual dispatch only

**Workflow Types**:
- `full-ci`: Complete CI with all checks
- `fast-ci`: Tests only (no security/build)
- `release-candidate`: Full pipeline + deployment
- `hotfix`: Emergency deployment to production
- `security-audit`: Security scans only
- `performance-check`: Performance tests only

**Decision Matrix**:

| Workflow Type | Tests | Security | Build | Deploy | Notifications |
|---------------|-------|----------|-------|--------|---------------|
| full-ci | ✓ | ✓ | ✓ | ✗ | Slack, Email |
| fast-ci | ✓ | ✗ | ✗ | ✗ | Slack |
| release-candidate | ✓ | ✓ | ✓ | ✓ | Slack, Email, GitHub |
| hotfix | ✓ | ✓ | ✓ | Prod | Slack, Email, PagerDuty |
| security-audit | ✗ | ✓ | ✗ | ✗ | Slack, Email |
| performance-check | ✓ | ✗ | ✗ | ✗ | Slack |

**Jobs**:

1. **orchestration-setup**: Configure workflow based on type
2. **coordinated-tests**: Reusable test workflow
3. **coordinated-security**: Trigger security-scan.yml
4. **coordinated-build**: Docker multi-platform builds
5. **coordinated-deployment**: Environment-specific deployment
6. **unified-notifications**: Multi-channel notifications
7. **post-workflow-sync**: Board synchronization

**Example Manual Trigger**:
```bash
# Release candidate to staging
gh workflow run workflow-coordinator.yml \
  -f workflow_type=release-candidate \
  -f environment=staging

# Emergency hotfix
gh workflow run workflow-coordinator.yml \
  -f workflow_type=hotfix \
  -f skip_tests=true
```

---

## Workflow Dependencies

### Dependency Graph

```mermaid
graph TD
    A[Push to main] --> B[CI Pipeline]
    B --> C{Tests Pass?}
    C -->|Yes| D[Staging Deploy]
    C -->|No| E[Notify Failure]

    D --> F[Smoke Tests]
    F -->|Pass| G[Board Sync]
    F -->|Fail| H[Auto Rollback]

    I[Manual Release] --> J[Automated Release]
    J --> K[Create Release]
    K --> L[Production Deploy]
    L --> M[Post-Deploy Monitoring]

    N[Issue Opened] --> O[Issue Management]
    O --> P[Board Sync]
    P --> Q[Notion Sync]

    R[PR Opened] --> S[PR Automation]
    S --> T[CI Pipeline]
    T --> U[Security Scan]
```

### Reusable Workflows

**Reusable Test Workflow** (`reusable-test.yml`):
- Called by: `workflow-coordinator.yml`
- Inputs: `test_type`, `coverage_threshold`
- Secrets: Inherited from caller

**Reusable Build Workflow** (`reusable-build.yml`):
- Called by: Various deployment workflows
- Inputs: `environment`, `docker_tag`
- Outputs: Image digests

---

## Trigger Matrix

### By Event Type

| Event | Workflows Triggered |
|-------|---------------------|
| Push (main) | CI Pipeline, Staging Deploy, Board Sync |
| Push (develop) | CI Pipeline |
| PR opened | CI Pipeline, PR Automation, Security Scan |
| PR review approved | Auto-merge Check |
| Issue opened | Issue Management, Board Sync |
| Issue labeled (security) | Security Issue Handler |
| Release published | Production Deploy, Release Management |
| Schedule (daily 2 AM) | Security Scan, Comprehensive Testing |
| Schedule (daily 3 AM) | Stale Issue Check |
| Schedule (daily 6 AM) | Board Sync |
| Manual | All workflows support manual dispatch |

### By Branch

| Branch | Workflows |
|--------|-----------|
| main | CI, Integration Tests, Staging Deploy, Board Sync |
| develop | CI, Unit Tests |
| feature/* | CI (on PR) |
| hotfix/* | Full CI, Fast-tracked to Production (manual) |

---

## Environment Variables

### Global Environment Variables

```yaml
# Python and Node versions
PYTHON_VERSION: '3.12'  # or '3.11' in some workflows
NODE_VERSION: '18'

# Container registry
REGISTRY: ghcr.io
IMAGE_NAME: ${{ github.repository }}

# Runner optimization
SMALL_RUNNER: ubuntu-latest
```

### Job-Specific Variables

**Backend Tests**:
```yaml
DATABASE_URL: postgresql://postgres:testpass@localhost:5432/test_db
REDIS_URL: redis://localhost:6379/0
JWT_SECRET_KEY: test-secret-key
APP_ENV: testing
```

**Frontend Tests**:
```yaml
CI: true
JEST_JUNIT_OUTPUT_DIR: ./test-results
JEST_JUNIT_OUTPUT_NAME: results.xml
```

**Docker Builds**:
```yaml
BUILDKIT_INLINE_CACHE: 1
BUILD_ENV: production|staging
```

---

## Secrets Configuration

### Required Secrets

**GitHub Tokens** (Auto-provided):
- `GITHUB_TOKEN`: Automatic token for GitHub API access

**API Keys** (Configure in Settings → Secrets):
```yaml
# External services
ALPHA_VANTAGE_API_KEY
FINNHUB_API_KEY
POLYGON_API_KEY
NEWS_API_KEY

# Third-party tools
CODECOV_TOKEN
SNYK_TOKEN
GITLEAKS_LICENSE

# Notifications
SLACK_WEBHOOK_URL
SMTP_USERNAME
SMTP_PASSWORD
NOTIFICATION_EMAIL
```

**Deployment Secrets**:
```yaml
# Staging
STAGING_KUBECONFIG          # Base64-encoded kubeconfig
STAGING_DATABASE_URL
STAGING_REDIS_URL
STAGING_JWT_SECRET
STAGING_API_URL
STAGING_FRONTEND_URL
STAGING_TEST_USER_EMAIL
STAGING_TEST_USER_PASSWORD

# Production
PRODUCTION_KUBECONFIG       # Base64-encoded kubeconfig
PRODUCTION_DATABASE_URL
PRODUCTION_REDIS_URL
PRODUCTION_JWT_SECRET
PRODUCTION_API_URL
PRODUCTION_FRONTEND_URL
PRODUCTION_TLS_CERT
PRODUCTION_TLS_KEY
```

**Notion Integration**:
```yaml
NOTION_TOKEN               # Notion API integration token
# Database ID configured in .github/board-sync.yml
```

### Environment-Specific Secrets

Configure in GitHub Settings → Environments:

**staging** environment:
- `STAGING_*` secrets
- Reviewers: Optional
- Wait timer: None

**production** environment:
- `PRODUCTION_*` secrets
- Reviewers: **Required** (recommend 2+)
- Wait timer: Optional (0-30 days)
- Protected branches: `main` only

---

## Troubleshooting Guide

### Common Issues

#### 1. CI Pipeline Failures

**Symptom**: Backend tests fail with database connection errors

**Solution**:
```bash
# Check PostgreSQL service health
# In workflow logs, verify:
# - PostgreSQL service is running
# - Health checks pass
# - Correct connection string

# Local reproduction:
docker run -d -p 5432:5432 \
  -e POSTGRES_PASSWORD=testpass \
  -e POSTGRES_DB=test_db \
  postgres:15-alpine

pytest backend/tests/ --tb=short
```

**Symptom**: Frontend tests timeout

**Solution**:
```yaml
# Increase timeout in workflow
- name: Run frontend tests
  run: npm test -- --testTimeout=10000
  timeout-minutes: 20  # Increase from 15
```

#### 2. Docker Build Failures

**Symptom**: "Layer cache not found" or slow builds

**Solution**:
```yaml
# Ensure cache configuration in workflow
- name: Build and push
  uses: docker/build-push-action@v5
  with:
    cache-from: type=gha,scope=backend-ci
    cache-to: type=gha,mode=max,scope=backend-ci
```

**Symptom**: Multi-platform build fails

**Solution**:
```bash
# Check if QEMU is set up
- name: Set up QEMU
  uses: docker/setup-qemu-action@v3

# Reduce platforms for testing
platforms: linux/amd64  # Remove linux/arm64 temporarily
```

#### 3. Deployment Failures

**Symptom**: Kubernetes deployment times out

**Solution**:
```bash
# Check cluster connectivity
kubectl cluster-info
kubectl get nodes

# Verify image is accessible
docker pull ghcr.io/<org>/<repo>/backend:latest

# Check resource quotas
kubectl describe resourcequota -n investment-analysis-production
```

**Symptom**: Database migration fails

**Solution**:
```bash
# Manually run migration
kubectl exec -it deploy/backend-deployment -n investment-analysis-staging -- \
  alembic upgrade head

# Check migration history
kubectl exec -it deploy/backend-deployment -n investment-analysis-staging -- \
  alembic current
```

#### 4. Security Scan False Positives

**Symptom**: Trivy reports vulnerabilities in base images

**Solution**:
```yaml
# Add .trivyignore file to repository
CVE-2024-XXXXX  # Reason: Not applicable to our use case

# Or adjust severity threshold in workflow
severity: 'CRITICAL,HIGH'  # Ignore MEDIUM and LOW
```

**Symptom**: Secret scanner detects test API keys

**Solution**:
```bash
# Add to .gitignore
.env*
!.env.example

# Update custom-secret-patterns.txt to exclude test patterns
# Or use GitLeaks allowlist in .gitleaks.toml
```

#### 5. Board Sync Issues

**Symptom**: Issues not appearing on GitHub Projects board

**Solution**:
```bash
# Verify project exists
gh project list --owner <owner>

# Check project permissions
# Ensure GITHUB_TOKEN has repository-projects: write permission

# Manual sync
gh workflow run board-sync.yml -f sync_type=full
```

**Symptom**: Notion sync fails

**Solution**:
```bash
# Verify Notion token
curl -H "Authorization: Bearer $NOTION_TOKEN" \
  -H "Notion-Version: 2022-06-28" \
  https://api.notion.com/v1/users/me

# Check database ID in .github/board-sync.yml
# Ensure Notion integration has access to database
```

### Debugging Tips

**Enable Workflow Debug Logging**:
```bash
# Repository Settings → Secrets → Add:
ACTIONS_RUNNER_DEBUG: true
ACTIONS_STEP_DEBUG: true
```

**Access Workflow Artifacts**:
```bash
# Download artifacts from workflow run
gh run download <run-id>

# List artifacts
gh run view <run-id> --log
```

**Re-run Failed Jobs**:
```bash
# Re-run specific job
gh run rerun <run-id> --job <job-id>

# Re-run all failed jobs
gh run rerun <run-id> --failed
```

**Check Workflow Status**:
```bash
# List recent runs
gh run list --workflow ci.yml --limit 10

# Watch active run
gh run watch
```

---

## Workflow Status Badges

Add these badges to your `README.md` to display workflow status:

### CI Status

```markdown
![CI Pipeline](https://github.com/<owner>/<repo>/actions/workflows/ci.yml/badge.svg)
![Security Scan](https://github.com/<owner>/<repo>/actions/workflows/security-scan.yml/badge.svg)
![Comprehensive Tests](https://github.com/<owner>/<repo>/actions/workflows/comprehensive-testing.yml/badge.svg)
```

### Deployment Status

```markdown
![Staging Deploy](https://github.com/<owner>/<repo>/actions/workflows/staging-deploy.yml/badge.svg)
![Production Deploy](https://github.com/<owner>/<repo>/actions/workflows/production-deploy.yml/badge.svg)
```

### Automation Status

```markdown
![Board Sync](https://github.com/<owner>/<repo>/actions/workflows/board-sync.yml/badge.svg)
![Issue Management](https://github.com/<owner>/<repo>/actions/workflows/issue-management.yml/badge.svg)
```

### All Workflows

```markdown
## Workflow Status

| Category | Workflow | Status |
|----------|----------|--------|
| **CI** | CI Pipeline | ![CI](https://github.com/<owner>/<repo>/actions/workflows/ci.yml/badge.svg) |
| **CI** | Security Scan | ![Security](https://github.com/<owner>/<repo>/actions/workflows/security-scan.yml/badge.svg) |
| **CI** | Comprehensive Tests | ![Tests](https://github.com/<owner>/<repo>/actions/workflows/comprehensive-testing.yml/badge.svg) |
| **Deploy** | Staging | ![Staging](https://github.com/<owner>/<repo>/actions/workflows/staging-deploy.yml/badge.svg) |
| **Deploy** | Production | ![Production](https://github.com/<owner>/<repo>/actions/workflows/production-deploy.yml/badge.svg) |
| **Automation** | PR Automation | ![PR](https://github.com/<owner>/<repo>/actions/workflows/pr-automation.yml/badge.svg) |
| **Automation** | Issue Management | ![Issues](https://github.com/<owner>/<repo>/actions/workflows/issue-management.yml/badge.svg) |
| **Automation** | Board Sync | ![Board](https://github.com/<owner>/<repo>/actions/workflows/board-sync.yml/badge.svg) |
```

Replace `<owner>` and `<repo>` with your GitHub organization/username and repository name.

---

## Quick Reference

### Common Commands

```bash
# List all workflows
gh workflow list

# View workflow details
gh workflow view ci.yml

# Run workflow manually
gh workflow run <workflow-name>.yml

# Cancel running workflow
gh run cancel <run-id>

# Download workflow logs
gh run download <run-id>

# Enable/disable workflow
gh workflow enable <workflow-name>.yml
gh workflow disable <workflow-name>.yml
```

### Workflow Timing

| Workflow | Typical Duration | Timeout |
|----------|-----------------|---------|
| CI Pipeline | 15-25 min | 90 min |
| Security Scan | 20-30 min | 60 min |
| Staging Deploy | 10-15 min | 30 min |
| Production Deploy | 15-25 min | 60 min |
| Board Sync | 2-5 min | 10 min |

### Cost Optimization Tips

1. **Use Caching**: Cache dependencies, Docker layers, and build artifacts
2. **Concurrency Limits**: Prevent duplicate runs with concurrency groups
3. **Conditional Jobs**: Skip jobs when not needed (`if` conditions)
4. **Matrix Strategy**: Run tests in parallel, but limit matrix size
5. **Timeout Settings**: Set appropriate timeouts to prevent runaway jobs
6. **Artifact Retention**: Set appropriate retention periods (default: 90 days)

---

## Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Workflow Syntax Reference](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)
- [GitHub Actions Best Practices](https://docs.github.com/en/actions/learn-github-actions/best-practices-for-workflows)
- [Security Hardening](https://docs.github.com/en/actions/security-guides/security-hardening-for-github-actions)

---

**Last Updated**: 2026-01-29
**Maintained By**: DevOps Team
**Questions**: Create an issue with label `component:ci-cd`
