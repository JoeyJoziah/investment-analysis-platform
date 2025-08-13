# CI/CD Troubleshooting Guide

Quick solutions for common CI/CD pipeline issues in the Investment Analysis App.

## 🚨 Emergency Quick Fixes

### Pipeline Completely Broken
```bash
# 1. Check workflow status
git status
git log --oneline -5

# 2. Test locally first  
./test_cicd_setup.sh

# 3. Validate configuration
python3 scripts/validate_cicd.py

# 4. Check GitHub Actions logs
# Go to: Repository → Actions → Failed workflow → View logs
```

### Can't Push to Repository
```bash
# Check branch protection rules
git push origin feature/your-branch  # Push to feature branch first
# Then create PR to main/develop
```

## 🔧 Common Issues & Solutions

### 1. GitHub Actions Not Running

**Symptoms:**
- No workflows appear in Actions tab
- Workflows exist but never trigger

**Solutions:**
```yaml
# Check .github/workflows/ci.yml has correct triggers:
on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]
```

**GitHub Settings Check:**
1. Repository → Settings → Actions → General
2. Ensure "Allow all actions and reusable workflows" is selected
3. Check "Read and write permissions" is selected

### 2. Secret Not Found Errors

**Error Message:**
```
Error: Secret SECRET_KEY not found
```

**Solutions:**
1. **Check Secret Name** (case-sensitive):
   - Repository → Settings → Secrets and variables → Actions
   - Verify exact spelling: `SECRET_KEY` not `secret_key`

2. **Re-add Secret:**
   ```bash
   # Generate new secret key
   python3 -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
   ```

3. **Environment vs Repository Secrets:**
   - Use Repository secrets for most cases
   - Environment secrets only for production deployments

### 3. Docker Build Failures

**Error Message:**
```
Error: failed to solve: process "/bin/sh -c pip install -r requirements.txt" did not complete successfully
```

**Solutions:**
1. **Check Dockerfile syntax:**
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   ```

2. **Test locally:**
   ```bash
   docker build -t test-build .
   docker run --rm test-build python --version
   ```

3. **Check requirements.txt:**
   ```bash
   # Validate requirements file
   pip install -r requirements.txt --dry-run
   ```

### 4. Test Failures

**Error Message:**
```
FAILED backend/tests/test_api.py::test_health_endpoint - AssertionError
```

**Solutions:**
1. **Run tests locally:**
   ```bash
   # Backend tests
   pytest backend/tests/ -v
   
   # Frontend tests  
   cd frontend/web && npm test
   ```

2. **Check test database:**
   ```bash
   # Ensure test database is configured
   export DATABASE_URL="postgresql://test:test@localhost:5432/test_db"
   pytest backend/tests/
   ```

3. **Common test fixes:**
   ```python
   # backend/tests/conftest.py - ensure proper test setup
   import pytest
   from fastapi.testclient import TestClient
   
   @pytest.fixture
   def client():
       from backend.api.main import app
       return TestClient(app)
   ```

### 5. Database Connection Issues

**Error Message:**
```
psycopg2.OperationalError: could not connect to server
```

**Solutions:**
1. **Check connection string format:**
   ```bash
   # Correct format
   DATABASE_URL="postgresql://username:password@host:port/database"
   
   # Common mistakes:
   # ❌ postgres://... (wrong protocol)
   # ❌ Missing password
   # ❌ Wrong port (5432 is default)
   ```

2. **Test connection manually:**
   ```bash
   # Using psql
   psql "postgresql://username:password@host:port/database"
   
   # Using Python
   python3 -c "
   import psycopg2
   conn = psycopg2.connect('postgresql://username:password@host:port/database')
   print('Connection successful!')
   conn.close()
   "
   ```

### 6. API Key Limits Exceeded

**Error Message:**
```
API limit exceeded for Alpha Vantage
```

**Solutions:**
1. **Check API quotas:**
   ```bash
   # Test API keys
   curl "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey=YOUR_KEY"
   ```

2. **Implement caching:**
   ```python
   # Use Redis caching for API responses
   import redis
   
   r = redis.Redis()
   cache_key = f"alpha_vantage:{symbol}:{date}"
   
   # Check cache first
   cached_data = r.get(cache_key)
   if cached_data:
       return json.loads(cached_data)
   ```

3. **API limits overview:**
   - Alpha Vantage: 25 calls/day
   - Finnhub: 60 calls/minute
   - Polygon.io: 5 calls/minute (free)

### 7. Kubernetes Deployment Failures

**Error Message:**
```
Error: failed to create deployment: unauthorized
```

**Solutions:**
1. **Check kubeconfig secret:**
   ```bash
   # Test kubeconfig locally
   export KUBECONFIG=downloaded-kubeconfig.yaml
   kubectl get nodes
   ```

2. **Verify namespace exists:**
   ```bash
   kubectl create namespace investment-analysis-staging
   kubectl create namespace investment-analysis-production
   ```

3. **Check RBAC permissions:**
   ```yaml
   # Ensure service account has proper permissions
   apiVersion: rbac.authorization.k8s.io/v1
   kind: ClusterRoleBinding
   metadata:
     name: deployment-binding
   subjects:
   - kind: ServiceAccount
     name: default
     namespace: default
   roleRef:
     kind: ClusterRole
     name: cluster-admin
     apiGroup: rbac.authorization.k8s.io
   ```

### 8. Container Registry Issues

**Error Message:**
```
Error: failed to push image to ghcr.io
```

**Solutions:**
1. **Check GitHub token permissions:**
   - Token needs `write:packages` scope
   - Username should be exact GitHub username

2. **Test registry login:**
   ```bash
   echo $REGISTRY_TOKEN | docker login ghcr.io -u $REGISTRY_USERNAME --password-stdin
   ```

3. **Verify image name format:**
   ```bash
   # Correct format
   ghcr.io/username/investment_analysis_app/backend:latest
   
   # Common mistakes:
   # ❌ Missing username
   # ❌ Wrong repository name
   # ❌ Missing tag
   ```

### 9. Branch Protection Blocking PRs

**Error Message:**
```
Required status checks are not passing
```

**Solutions:**
1. **Check required status checks:**
   - Repository → Settings → Branches → Edit rule
   - Ensure status check names match workflow job names

2. **Wait for checks to complete:**
   - All required checks must pass before merge
   - Check Actions tab for running workflows

3. **Bypass for emergencies** (admins only):
   - Use "Merge without waiting for requirements to be met"
   - Only for critical hotfixes

### 10. Slack Notifications Not Working

**Error Message:**
```
Failed to send Slack notification
```

**Solutions:**
1. **Verify webhook URL:**
   ```bash
   # Test webhook manually
   curl -X POST -H 'Content-type: application/json' \
     --data '{"text":"Test message"}' \
     YOUR_SLACK_WEBHOOK_URL
   ```

2. **Check bot token scopes:**
   - `chat:write` for sending messages
   - `files:write` for uploading files

3. **Verify channel permissions:**
   - Bot must be invited to channel
   - Channel must exist and be accessible

## 🔍 Debugging Workflow

### Step 1: Identify the Problem
```bash
# Check recent commits
git log --oneline -10

# Check current status
git status

# View recent workflow runs
# GitHub → Actions → Recent runs
```

### Step 2: Gather Information
```bash
# Run local validation
./test_cicd_setup.sh

# Check comprehensive validation
python3 scripts/validate_cicd.py

# Export logs for analysis
# GitHub → Actions → Failed workflow → Download logs
```

### Step 3: Test Locally
```bash
# Test Docker build
docker build -t local-test .

# Run tests locally
make test  # or pytest/npm test

# Test API endpoints
curl http://localhost:8000/api/health
```

### Step 4: Apply Fix
```bash
# Make minimal changes
git checkout -b fix/issue-description
# ... make fixes ...
git commit -m "fix: resolve CI/CD issue"
git push origin fix/issue-description
# Create PR to test fix
```

## 🛡️ Prevention Strategies

### 1. Pre-commit Hooks
```bash
# Install pre-commit
pip install pre-commit
pre-commit install

# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: tests
        name: tests
        entry: pytest backend/tests/
        language: system
        pass_filenames: false
```

### 2. Local Development Environment
```bash
# Use docker-compose for consistency
docker-compose up -d postgres redis
docker-compose exec backend python manage.py test

# Match CI environment
export CI=true
python3 scripts/validate_cicd.py
```

### 3. Monitoring Setup
```bash
# Enable workflow notifications
# GitHub → Settings → Notifications → Actions

# Set up status badges in README
[![CI](https://github.com/username/repo/actions/workflows/ci.yml/badge.svg)](https://github.com/username/repo/actions/workflows/ci.yml)
```

## 📞 Getting Help

### 1. Check Documentation
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)

### 2. Enable Debug Mode
```yaml
# Add to workflow for verbose logging
env:
  ACTIONS_STEP_DEBUG: true
  ACTIONS_RUNNER_DEBUG: true
```

### 3. Community Resources
- [GitHub Community Forum](https://github.community/)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/github-actions)
- [Docker Community Slack](https://dockercommunity.slack.com/)

### 4. Emergency SSH Access
```yaml
# Add this step to failing workflow for debugging
- name: Setup tmate session
  if: failure()
  uses: mxschmitt/action-tmate@v3
  timeout-minutes: 15
```

## ⚡ Quick Reference Commands

```bash
# Test everything locally
./test_cicd_setup.sh

# Validate comprehensive setup  
python3 scripts/validate_cicd.py

# Check Docker setup
docker-compose config
docker-compose up --dry-run

# Test API keys
curl "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey=$ALPHA_VANTAGE_API_KEY"

# GitHub CLI commands (if installed)
gh workflow list
gh workflow view ci.yml
gh run list --limit 5
```

---

**Remember**: Most CI/CD issues are configuration problems, not code problems. Start with the basics (secrets, permissions, syntax) before diving deep into complex debugging.

**Success Metric**: When you can push a small change and see it automatically deploy to production without manual intervention, your CI/CD pipeline is working correctly! 🎉