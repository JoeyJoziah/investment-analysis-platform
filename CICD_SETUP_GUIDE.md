# Complete CI/CD Setup Guide for Investment Analysis App

This comprehensive guide walks you through setting up the complete CI/CD pipeline for the Investment Analysis App, including GitHub Actions, secrets management, branch protection, and deployment.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [GitHub Repository Setup](#github-repository-setup)
3. [GitHub Secrets Configuration](#github-secrets-configuration)
4. [Branch Protection Rules](#branch-protection-rules)
5. [Enable GitHub Actions](#enable-github-actions)
6. [Container Registry Setup](#container-registry-setup)
7. [Kubernetes Cluster Setup](#kubernetes-cluster-setup)
8. [Testing the Pipeline](#testing-the-pipeline)
9. [Monitoring and Notifications](#monitoring-and-notifications)
10. [Troubleshooting](#troubleshooting)
11. [Final Verification Checklist](#final-verification-checklist)

## Prerequisites

Before starting, ensure you have:
- [ ] GitHub repository created and code pushed
- [ ] DigitalOcean or AWS account (for Kubernetes cluster)
- [ ] API keys for external services (Alpha Vantage, Finnhub, Polygon.io, NewsAPI)
- [ ] Slack workspace (for notifications)
- [ ] Docker installed locally (for testing)

## GitHub Repository Setup

### 1. Repository Structure Verification

Ensure your repository has the following structure:
```
investment_analysis_app/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       ├── staging-deploy.yml
│       ├── production-deploy.yml
│       ├── security-scan.yml
│       └── other workflow files...
├── backend/
├── frontend/
├── infrastructure/
├── docker-compose.yml
└── README.md
```

### 2. Initial Repository Settings

**Step 1: Access Repository Settings**
1. Navigate to your GitHub repository
2. Click the **"Settings"** tab (located at the top right of the repository)
3. You should see a sidebar with various options

**Step 2: General Settings**
1. In the sidebar, click **"General"**
2. Scroll to **"Features"** section
3. Ensure the following are checked:
   - ✅ **Issues** (for bug tracking)
   - ✅ **Projects** (for project management)
   - ✅ **Actions** (for CI/CD)
   - ✅ **Packages** (for container registry)

## GitHub Secrets Configuration

GitHub Secrets store sensitive information like API keys and credentials securely.

### 3. Accessing Secrets Settings

**Step 1: Navigate to Secrets**
1. In your repository, click **"Settings"** tab
2. In the left sidebar, expand **"Secrets and variables"**
3. Click **"Actions"**
4. You'll see tabs: **"Repository secrets"**, **"Environment secrets"**, **"Variables"**

### 4. Required Repository Secrets

Click **"New repository secret"** button for each secret below:

#### Database Secrets
```
Secret Name: DB_PASSWORD
Secret Value: your_secure_postgres_password_here
Description: PostgreSQL database password
```

```
Secret Name: REDIS_PASSWORD
Secret Value: your_secure_redis_password_here
Description: Redis cache password
```

#### Application Secrets
```
Secret Name: SECRET_KEY
Secret Value: your_django_secret_key_here_minimum_50_characters_long
Description: Django/FastAPI secret key for signing
```

```
Secret Name: JWT_SECRET_KEY
Secret Value: your_jwt_secret_key_here_minimum_32_characters
Description: JWT token signing key
```

#### API Keys for Data Sources
```
Secret Name: ALPHA_VANTAGE_API_KEY
Secret Value: your_alpha_vantage_api_key
Description: Alpha Vantage API key (25 calls/day limit)
```

```
Secret Name: FINNHUB_API_KEY
Secret Value: your_finnhub_api_key
Description: Finnhub API key (60 calls/minute)
```

```
Secret Name: POLYGON_API_KEY
Secret Value: your_polygon_io_api_key
Description: Polygon.io API key (5 calls/minute free tier)
```

```
Secret Name: NEWS_API_KEY
Secret Value: your_news_api_key
Description: NewsAPI key for sentiment analysis
```

#### Container Registry
```
Secret Name: REGISTRY_USERNAME
Secret Value: your_github_username
Description: GitHub Container Registry username
```

```
Secret Name: REGISTRY_TOKEN
Secret Value: your_github_personal_access_token
Description: GitHub Personal Access Token with packages:write scope
```

#### Cloud Infrastructure
```
Secret Name: DIGITALOCEAN_ACCESS_TOKEN
Secret Value: your_digitalocean_api_token
Description: DigitalOcean API token for Kubernetes cluster
```

Or if using AWS:
```
Secret Name: AWS_ACCESS_KEY_ID
Secret Value: your_aws_access_key_id
Description: AWS Access Key ID
```

```
Secret Name: AWS_SECRET_ACCESS_KEY
Secret Value: your_aws_secret_access_key
Description: AWS Secret Access Key
```

#### Notifications
```
Secret Name: SLACK_WEBHOOK_URL
Secret Value: https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
Description: Slack webhook for deployment notifications
```

```
Secret Name: SLACK_BOT_TOKEN
Secret Value: xoxb-your-slack-bot-token
Description: Slack bot token for detailed notifications
```

#### Production Environment
```
Secret Name: PRODUCTION_DATABASE_URL
Secret Value: postgresql://username:password@host:port/database
Description: Production database connection string
```

```
Secret Name: PRODUCTION_REDIS_URL
Secret Value: redis://username:password@host:port
Description: Production Redis connection string
```

### 5. Creating Personal Access Token (for REGISTRY_TOKEN)

**Step 1: Generate PAT**
1. Click your profile picture (top right) → **"Settings"**
2. Scroll down to **"Developer settings"** (bottom left)
3. Click **"Personal access tokens"** → **"Tokens (classic)"**
4. Click **"Generate new token"** → **"Generate new token (classic)"**

**Step 2: Configure Token**
- **Note**: "Investment Analysis App CI/CD"
- **Expiration**: 90 days (or longer if preferred)
- **Scopes** (check these boxes):
  - ✅ `repo` (Full control of private repositories)
  - ✅ `write:packages` (Upload packages to GitHub Package Registry)
  - ✅ `read:packages` (Download packages from GitHub Package Registry)
  - ✅ `delete:packages` (Delete packages from GitHub Package Registry)

**Step 3: Copy Token**
1. Click **"Generate token"**
2. **IMPORTANT**: Copy the token immediately (you won't see it again!)
3. Use this as the value for `REGISTRY_TOKEN` secret

## Branch Protection Rules

### 6. Setting Up Branch Protection

**Step 1: Access Branch Settings**
1. Repository **"Settings"** → **"Branches"** (in sidebar)
2. Under **"Branch protection rules"**, click **"Add rule"**

**Step 2: Configure Main Branch Protection**

Fill in the form:

**Branch name pattern**: `main`

**Protect matching branches** - Check these options:
- ✅ **Restrict pushes that create files larger than 100MB**
- ✅ **Require a pull request before merging**
  - ✅ **Require approvals**: `1` (minimum)
  - ✅ **Dismiss stale reviews when new commits are pushed**
  - ✅ **Require review from code owners**
- ✅ **Require status checks to pass before merging**
  - ✅ **Require branches to be up to date before merging**
  - **Status checks** (add these as they become available):
    - `backend-quality`
    - `frontend-quality` 
    - `backend-tests`
    - `frontend-tests`
    - `security-scan`
- ✅ **Require conversation resolution before merging**
- ✅ **Require signed commits**
- ✅ **Include administrators** (applies rules to admins too)

**Step 3: Configure Develop Branch Protection**

Repeat for `develop` branch with slightly relaxed rules:
- ✅ **Require a pull request before merging**
- ✅ **Require status checks to pass before merging**
- Status checks: `backend-quality`, `backend-tests`, `security-scan`

## Enable GitHub Actions

### 7. GitHub Actions Configuration

**Step 1: Actions Settings**
1. Repository **"Settings"** → **"Actions"** → **"General"**
2. Under **"Actions permissions"**:
   - Select: **"Allow all actions and reusable workflows"**
3. Under **"Workflow permissions"**:
   - Select: **"Read and write permissions"**
   - ✅ **Allow GitHub Actions to create and approve pull requests**

**Step 2: Runner Settings**
1. Under **"Runner groups"**: Leave as default
2. Under **"Actions secrets and variables"**: Already configured above

## Container Registry Setup

### 8. GitHub Container Registry (GHCR)

GitHub Container Registry is automatically enabled. Your workflows will push images to:
- `ghcr.io/yourusername/investment_analysis_app/backend:latest`
- `ghcr.io/yourusername/investment_analysis_app/frontend:latest`

**Verify Container Registry**
1. Repository → **"Packages"** tab (should appear after first successful build)
2. Images will be listed here after CI/CD runs

## Kubernetes Cluster Setup

### 9. DigitalOcean Kubernetes

**Step 1: Create Kubernetes Cluster**
1. Log into [DigitalOcean](https://cloud.digitalocean.com/)
2. Click **"Create"** → **"Kubernetes Cluster"**
3. Choose:
   - **Kubernetes version**: Latest stable
   - **Datacenter region**: Closest to your users
   - **VPC network**: Default
   - **Node pool**: 
     - Node type: `s-2vcpu-4gb` (2 vCPU, 4GB RAM) - $24/month
     - Node count: 2 nodes
   - **Cluster name**: `investment-analysis-prod`
4. Click **"Create Cluster"** (takes 3-5 minutes)

**Step 2: Download kubeconfig**
1. Once cluster is ready, click **"Download Config File"**
2. Save as `kubeconfig-investment-analysis.yaml`

**Step 3: Add Kubeconfig to GitHub Secrets**
```
Secret Name: KUBECONFIG_CONTENT
Secret Value: [paste entire contents of kubeconfig file]
Description: Kubernetes cluster configuration
```

### 10. AWS EKS (Alternative)

If using AWS instead:

**Step 1: Create EKS Cluster**
1. AWS Console → **EKS** → **"Create cluster"**
2. Configure:
   - **Cluster name**: `investment-analysis-prod`
   - **Kubernetes version**: Latest
   - **Service role**: Create new or use existing EKS service role
   - **VPC**: Default or create new
   - **Subnets**: Select at least 2 in different AZs
3. **Node Groups**:
   - **Node group name**: `investment-analysis-nodes`
   - **Instance types**: `t3.medium`
   - **Scaling**: Min 1, Max 3, Desired 2

## Testing the Pipeline

### 11. Initial Pipeline Test

**Step 1: Trigger First Build**
1. Make a small change to README.md
2. Commit and push to `develop` branch:
   ```bash
   git checkout -b feature/test-ci
   git add README.md
   git commit -m "Test CI/CD pipeline"
   git push origin feature/test-ci
   ```
3. Create Pull Request from `feature/test-ci` to `develop`

**Step 2: Monitor Build**
1. Repository → **"Actions"** tab
2. You should see running workflows:
   - **CI Pipeline** (triggered by PR)
   - Watch each job: `backend-quality`, `frontend-quality`, `backend-tests`, etc.

**Step 3: Check Build Status**
Jobs should show:
- ✅ **Backend Code Quality**: Black, isort, flake8, mypy checks
- ✅ **Frontend Code Quality**: ESLint, Prettier checks  
- ✅ **Backend Tests**: pytest with coverage report
- ✅ **Frontend Tests**: Jest tests
- ✅ **Security Scan**: Bandit, safety, npm audit
- ✅ **Build Docker Images**: Backend and frontend images

### 12. Deployment Test

**Step 1: Merge to Develop**
1. If all checks pass, merge PR to `develop`
2. This triggers **Staging Deploy** workflow

**Step 2: Monitor Staging Deployment**
1. **Actions** → **Staging Deploy** workflow
2. Should show:
   - ✅ Build and push images
   - ✅ Deploy to staging namespace
   - ✅ Run smoke tests

**Step 3: Production Deployment**
1. Create PR from `develop` to `main`
2. Once merged, triggers **Production Deploy** workflow
3. Monitor production deployment

## Monitoring and Notifications

### 13. Slack Integration

**Step 1: Create Slack App**
1. Visit [Slack API](https://api.slack.com/apps)
2. Click **"Create New App"** → **"From scratch"**
3. **App Name**: "Investment Analysis CI/CD"
4. **Workspace**: Select your workspace

**Step 2: Configure Incoming Webhooks**
1. **Features** → **Incoming Webhooks**
2. Toggle **"Activate Incoming Webhooks"** to On
3. Click **"Add New Webhook to Workspace"**
4. Choose channel: `#deployments` or `#general`
5. Copy webhook URL for `SLACK_WEBHOOK_URL` secret

**Step 3: Bot Token (Optional)**
1. **Features** → **OAuth & Permissions**
2. **Scopes** → **Bot Token Scopes**:
   - `chat:write` (send messages)
   - `files:write` (upload files)
3. **Install App to Workspace**
4. Copy **Bot User OAuth Token** for `SLACK_BOT_TOKEN` secret

### 14. GitHub Actions Notifications

The workflows are configured to send notifications on:
- ✅ **Successful deployments**
- ❌ **Failed builds/deployments**
- ⚠️ **Security vulnerabilities found**
- 📊 **Performance regression detected**

### 15. Monitoring Dashboard

**Step 1: Access Actions Dashboard**
1. Repository → **"Actions"** tab
2. **Workflows** sidebar shows all workflows
3. Click workflow name to see run history

**Step 2: Workflow Insights**
1. Repository → **"Insights"** tab
2. **Dependency graph** → **"Actions"** tab
3. View success rates, run times, and trends

## Troubleshooting

### 16. Common Issues and Solutions

#### Issue: "Secret not found" Error
**Solution**: 
1. Check secret name spelling (case-sensitive)
2. Verify secret exists: Settings → Secrets and variables → Actions
3. For organization repos, check if secret is at org level

#### Issue: Docker Build Fails
**Solution**:
1. Check Dockerfile syntax
2. Verify base images are accessible
3. Check for missing dependencies in requirements.txt

#### Issue: Tests Failing
**Solution**:
1. Run tests locally first: `pytest backend/tests/`
2. Check test database configuration
3. Ensure test fixtures are available

#### Issue: Deployment Fails
**Solution**:
1. Verify Kubernetes cluster is accessible
2. Check kubeconfig secret is correct
3. Ensure namespace exists
4. Verify image registry credentials

#### Issue: API Rate Limits
**Solution**:
1. Verify API keys are correct
2. Check API quota usage
3. Implement proper caching
4. Add rate limiting logic

### 17. Debugging Steps

**Step 1: Enable Debug Logging**
Add this secret for verbose output:
```
Secret Name: ACTIONS_STEP_DEBUG
Secret Value: true
```

**Step 2: SSH into Runner (Emergency)**
Add this step to failing workflow:
```yaml
- name: Setup tmate session
  if: failure()
  uses: mxschmitt/action-tmate@v3
```

**Step 3: Check Logs**
1. Actions → Failed workflow → Failed job
2. Expand each step to see detailed logs
3. Look for error messages and stack traces

## Final Verification Checklist

### 18. Complete Verification

#### GitHub Setup ✓
- [ ] Repository has all workflow files
- [ ] All required secrets configured (25+ secrets)
- [ ] Branch protection rules active on `main` and `develop`
- [ ] GitHub Actions enabled with correct permissions
- [ ] Container registry accessible

#### Pipeline Functionality ✓
- [ ] Code quality checks pass (Black, ESLint, etc.)
- [ ] All tests pass with >85% coverage
- [ ] Security scans complete without high vulnerabilities
- [ ] Docker images build successfully
- [ ] Images push to GitHub Container Registry

#### Deployment ✓
- [ ] Kubernetes cluster created and accessible
- [ ] Staging deployment works
- [ ] Production deployment works
- [ ] Database migrations run successfully
- [ ] Health checks pass post-deployment

#### Monitoring ✓
- [ ] Slack notifications working
- [ ] GitHub Actions dashboard showing green
- [ ] Application accessible at expected URLs
- [ ] Logs flowing properly

#### API Integration ✓
- [ ] Alpha Vantage API key working (check quota)
- [ ] Finnhub API key working
- [ ] Polygon.io API key working
- [ ] NewsAPI key working
- [ ] All APIs within free tier limits

### 19. Final Test Sequence

**Step 1: End-to-End Test**
```bash
# Create feature branch
git checkout -b feature/final-test
echo "Final CI/CD test" >> README.md
git add README.md
git commit -m "Final end-to-end CI/CD test"
git push origin feature/final-test
```

**Step 2: Monitor Full Pipeline**
1. Create PR: `feature/final-test` → `develop`
2. Verify all CI checks pass
3. Merge to `develop`
4. Verify staging deployment
5. Create PR: `develop` → `main`
6. Verify production deployment

**Step 3: Application Health Check**
1. Visit application URLs
2. Check API health endpoints
3. Verify database connections
4. Test key functionality

### 20. Maintenance Tasks

#### Weekly Tasks
- [ ] Review workflow run history
- [ ] Check API quota usage
- [ ] Review security scan results
- [ ] Update dependencies if needed

#### Monthly Tasks
- [ ] Rotate secrets (especially API keys)
- [ ] Review and optimize workflows
- [ ] Check cloud infrastructure costs
- [ ] Update base Docker images

#### As Needed
- [ ] Scale Kubernetes nodes based on load
- [ ] Update workflow permissions
- [ ] Add new secrets for integrations
- [ ] Adjust branch protection rules

## Success Indicators

Your CI/CD pipeline is properly configured when:

1. **Green Builds**: All workflow runs showing ✅ status
2. **Fast Feedback**: PR checks complete in <10 minutes
3. **Automated Deployments**: Code reaches production without manual steps
4. **Proper Notifications**: Team gets notified of build/deploy status
5. **Cost Control**: Staying within $50/month operational budget
6. **Security**: No high-severity vulnerabilities in security scans
7. **Coverage**: Tests maintain >85% code coverage
8. **Performance**: Application responds within acceptable time limits

## Getting Help

If you encounter issues:

1. **GitHub Actions Documentation**: https://docs.github.com/en/actions
2. **DigitalOcean Kubernetes**: https://docs.digitalocean.com/products/kubernetes/
3. **Slack API Documentation**: https://api.slack.com/
4. **Project Issues**: Create issue in your repository with:
   - Error message
   - Workflow run link
   - Steps to reproduce

Remember: This is a complex setup with many moving parts. It's normal to encounter issues initially. Work through them systematically, and don't hesitate to reach out for help!

---

**Total Setup Time Estimate**: 3-4 hours for experienced users, 6-8 hours for beginners
**Maintenance Time**: ~30 minutes per week

Your Investment Analysis App will have enterprise-grade CI/CD once this setup is complete!