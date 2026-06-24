# Phase 3 Deployment Checklist

**Version:** 1.0.0
**Created:** 2026-01-27
**Target Environment:** Production
**Deployment Type:** Phase 3.0 Integration (Security, Type Safety, Row Locking)

---

## Deployment Status: 🔴 NOT READY

**Prerequisites Status:** ❌ BLOCKERS PRESENT
**Estimated Deployment Date:** TBD (after blockers resolved)
**Rollback Plan:** ✅ PREPARED

---

## Pre-Deployment Blockers

### 🚨 MUST FIX BEFORE PROCEEDING

- [ ] **BLOCKER #1:** Fix test import error (`create_refresh_token`)
  - **File:** `backend/tests/integration/test_auth_to_portfolio_flow.py:21`
  - **Action:** Export function or refactor test
  - **ETA:** 2-4 hours
  - **Owner:** Backend Team

- [ ] **BLOCKER #2:** Resolve 86+ mypy type errors
  - **Priority:** SQLAlchemy Base class (18 errors)
  - **Priority:** Default parameter types (2 errors)
  - **Action:** Fix types and add CI/CD check
  - **ETA:** 8-12 hours
  - **Owner:** Backend Team

- [ ] **BLOCKER #3:** Set up CI/CD type checking
  - **Action:** Create `.github/workflows/type-check.yml`
  - **Action:** Configure pre-commit hooks
  - **ETA:** 4-6 hours
  - **Owner:** DevOps Team

**⚠️ DO NOT PROCEED UNTIL ALL BLOCKERS ARE RESOLVED**

---

## Phase 1: Pre-Deployment Preparation

### 1.1 Code Quality Validation

- [ ] Fix test import error
  ```bash
  # Verify fix
  python -c "from backend.auth.oauth2 import create_refresh_token; print('OK')"
  ```

- [ ] Run full test suite
  ```bash
  pytest backend/tests/ -v --tb=short
  # Target: ≥80% pass rate (659/823 tests)
  ```

- [ ] Check test coverage
  ```bash
  pytest backend/tests/ --cov=backend --cov-report=html
  # Target: ≥80% coverage
  ```

- [ ] Verify type checking
  ```bash
  pip install lxml  # Install missing dependency
  mypy backend/ --html-report ./mypy-report
  # Target: <10 errors
  ```

- [ ] Run linter
  ```bash
  flake8 backend/ --max-line-length=120 --exclude=migrations
  # Target: 0 critical issues
  ```

### 1.2 Security Validation

- [ ] Run security audit
  ```bash
  bandit -r backend/ -ll
  # Target: 0 high/critical issues
  ```

- [ ] Check for hardcoded secrets
  ```bash
  grep -r "sk-\|api_key.*=" backend/ --include="*.py" | grep -v "\.env"
  # Target: 0 matches
  ```

- [ ] Validate CSRF middleware
  ```bash
  # Manual test: POST without CSRF token should fail
  curl -X POST https://staging.domain.com/api/portfolios \
    -H "Content-Type: application/json" \
    -d '{"name":"test"}'
  # Expected: 403 Forbidden
  ```

- [ ] Verify security headers
  ```bash
  curl -I https://staging.domain.com/api/health
  # Expected headers:
  # X-Frame-Options: DENY
  # X-Content-Type-Options: nosniff
  # Content-Security-Policy: ...
  ```

- [ ] Test row locking (race condition prevention)
  ```python
  # Run concurrent update test
  pytest backend/tests/integration/test_auth_to_portfolio_flow.py::test_concurrent_portfolio_updates -v
  # Expected: No data corruption
  ```

### 1.3 Database Preparation

- [ ] Backup production database
  ```bash
  # Create backup directory
  mkdir -p backups/pre-phase3

  # PostgreSQL backup
  docker-compose exec investment_db pg_dump -U investment_user \
    -d investment_db > backups/pre-phase3/backup-$(date +%Y%m%d-%H%M%S).sql

  # Verify backup
  ls -lh backups/pre-phase3/
  ```

- [ ] Test migration in staging
  ```bash
  # Apply migration
  docker-compose exec investment_db psql -U investment_user \
    -d investment_db < backend/migrations/add_row_locking_versions.sql

  # Verify version columns added
  docker-compose exec investment_db psql -U investment_user \
    -d investment_db -c "\d portfolios"
  # Expected: version column present
  ```

- [ ] Test rollback script
  ```sql
  -- Create rollback script: backend/migrations/rollback_row_locking.sql
  DROP INDEX IF EXISTS idx_portfolio_version;
  DROP INDEX IF EXISTS idx_position_version;
  ALTER TABLE portfolios DROP COLUMN IF EXISTS version;
  ALTER TABLE positions DROP COLUMN IF EXISTS version;
  ALTER TABLE transactions DROP COLUMN IF EXISTS version;
  ```

- [ ] Verify rollback works
  ```bash
  docker-compose exec investment_db psql -U investment_user \
    -d investment_db < backend/migrations/rollback_row_locking.sql

  # Re-apply to leave in ready state
  docker-compose exec investment_db psql -U investment_user \
    -d investment_db < backend/migrations/add_row_locking_versions.sql
  ```

### 1.4 Environment Configuration

- [ ] Create production environment file
  ```bash
  cp .env.example .env.production
  ```

- [ ] Configure critical environment variables
  ```bash
  # Security
  JWT_SECRET_KEY=<generate: python3 -c "import secrets; print(secrets.token_urlsafe(32))">
  GDPR_ENCRYPTION_KEY=<generate: python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())">
  SECRET_KEY=<generate: python3 -c "import secrets; print(secrets.token_hex(32))">

  # Database
  DATABASE_URL=postgresql://investment_user:<password>@investment_db:5432/investment_db
  DB_HOST=investment_db
  DB_PORT=5432
  DB_NAME=investment_db
  DB_USER=investment_user
  DB_PASSWORD=<strong-random-password>

  # Redis
  REDIS_URL=redis://investment_cache:6379/0

  # Application
  ENVIRONMENT=production
  DEBUG=false
  LOG_LEVEL=info
  ```

- [ ] Validate environment variables
  ```bash
  # Check all required vars are set
  python3 -c "
  from backend.config.settings import settings
  print('Environment:', settings.ENVIRONMENT)
  print('Database:', settings.DATABASE_URL)
  print('Redis:', settings.REDIS_URL)
  print('JWT configured:', bool(settings.JWT_SECRET_KEY))
  "
  ```

- [ ] Configure CORS origins
  ```bash
  # In .env.production
  ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
  ```

### 1.5 CI/CD Setup

- [ ] Create type-check workflow
  ```bash
  mkdir -p .github/workflows
  ```

  Create `.github/workflows/type-check.yml`:
  ```yaml
  name: Type Check

  on:
    push:
      branches: [main, develop]
    pull_request:
      branches: [main]

  jobs:
    mypy:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v3

        - name: Set up Python
          uses: actions/setup-python@v4
          with:
            python-version: '3.12'

        - name: Install dependencies
          run: |
            pip install -r backend/requirements.txt
            pip install mypy lxml

        - name: Run mypy
          run: mypy backend/ --html-report ./mypy-report

        - name: Upload report
          if: always()
          uses: actions/upload-artifact@v3
          with:
            name: mypy-report
            path: mypy-report/
  ```

- [ ] Configure pre-commit hooks
  ```bash
  # Install pre-commit
  pip install pre-commit

  # Create .pre-commit-config.yaml
  cat > .pre-commit-config.yaml <<EOF
  repos:
    - repo: https://github.com/pre-commit/mirrors-mypy
      rev: v1.8.0
      hooks:
        - id: mypy
          args: [--config-file=mypy.ini]
          additional_dependencies: [types-all]

    - repo: https://github.com/pycqa/flake8
      rev: 7.0.0
      hooks:
        - id: flake8
          args: [--max-line-length=120]

    - repo: https://github.com/PyCQA/bandit
      rev: 1.7.5
      hooks:
        - id: bandit
          args: [-ll, -r, backend/]
  EOF

  # Install hooks
  pre-commit install

  # Test hooks
  pre-commit run --all-files
  ```

### 1.6 Documentation Review

- [ ] Update API documentation
  ```bash
  # Verify docs are current
  grep -r "TODO\|FIXME\|XXX" docs/ | wc -l
  # Target: <5 TODOs
  ```

- [ ] Review deployment guide
  ```bash
  cat docs/DEPLOYMENT.md
  # Verify: SSL setup, domain config, monitoring
  ```

- [ ] Update CHANGELOG
  ```bash
  cat >> CHANGELOG.md <<EOF

  ## [Phase 3.0] - $(date +%Y-%m-%d)

  ### Added
  - Enhanced security with CSRF protection
  - Row-level locking for race condition prevention
  - Type safety validation with mypy
  - JWT token blacklisting
  - Security headers middleware

  ### Changed
  - Database schema: Added version columns
  - Authentication: Upgraded to RS256 encryption

  ### Security
  - Implemented request size limits
  - Added GDPR-compliant data encryption
  - Enhanced session management
  EOF
  ```

---

## Phase 2: Staging Deployment

### 2.1 Deploy to Staging Environment

- [ ] Build Docker images
  ```bash
  docker-compose -f docker-compose.staging.yml build
  ```

- [ ] Start staging services
  ```bash
  docker-compose -f docker-compose.staging.yml up -d
  ```

- [ ] Verify all services running
  ```bash
  docker-compose -f docker-compose.staging.yml ps
  # Expected: All containers 'Up (healthy)'
  ```

- [ ] Run database migration
  ```bash
  docker-compose -f docker-compose.staging.yml exec investment_db \
    psql -U investment_user -d investment_db \
    < backend/migrations/add_row_locking_versions.sql
  ```

- [ ] Verify migration success
  ```bash
  docker-compose -f docker-compose.staging.yml exec investment_db \
    psql -U investment_user -d investment_db \
    -c "SELECT column_name FROM information_schema.columns
        WHERE table_name='portfolios' AND column_name='version';"
  # Expected: version
  ```

### 2.2 Staging Smoke Tests

- [ ] Health check endpoint
  ```bash
  curl https://staging.domain.com/api/health
  # Expected: {"status":"healthy","version":"3.0.0"}
  ```

- [ ] Database connectivity
  ```bash
  curl https://staging.domain.com/api/health/db
  # Expected: {"status":"connected"}
  ```

- [ ] Redis connectivity
  ```bash
  curl https://staging.domain.com/api/health/redis
  # Expected: {"status":"connected"}
  ```

- [ ] Authentication flow
  ```bash
  # Register user
  curl -X POST https://staging.domain.com/api/v1/auth/register \
    -H "Content-Type: application/json" \
    -d '{"email":"test@test.com","password":"Test123!","full_name":"Test User"}'

  # Login
  curl -X POST https://staging.domain.com/api/v1/auth/login \
    -H "Content-Type: application/json" \
    -d '{"email":"test@test.com","password":"Test123!"}'
  # Expected: access_token and refresh_token
  ```

- [ ] CSRF protection
  ```bash
  # Should fail without CSRF token
  curl -X POST https://staging.domain.com/api/portfolios \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer <token>" \
    -d '{"name":"test"}'
  # Expected: 403 Forbidden
  ```

- [ ] Security headers
  ```bash
  curl -I https://staging.domain.com/api/health | grep -E "X-Frame|X-Content|CSP"
  # Expected: All headers present
  ```

- [ ] Row locking test
  ```bash
  # Run concurrent update test
  pytest backend/tests/integration/test_auth_to_portfolio_flow.py::test_concurrent_portfolio_updates -v --staging
  # Expected: PASS (no data corruption)
  ```

### 2.3 Staging Load Testing

- [ ] Install load testing tool
  ```bash
  pip install locust
  ```

- [ ] Create load test script
  ```python
  # locustfile.py
  from locust import HttpUser, task, between

  class InvestmentPlatformUser(HttpUser):
      wait_time = between(1, 3)

      @task
      def health_check(self):
          self.client.get("/api/health")

      @task(3)
      def get_stocks(self):
          self.client.get("/api/stocks?limit=10")
  ```

- [ ] Run load test
  ```bash
  locust -f locustfile.py --host=https://staging.domain.com \
    --users 100 --spawn-rate 10 --run-time 5m
  # Target: <500ms p95 response time, <1% error rate
  ```

- [ ] Review load test results
  ```bash
  # Check metrics:
  # - RPS (requests per second)
  # - Response times (p50, p95, p99)
  # - Error rate
  # Target: 100 RPS sustained, <500ms p95
  ```

### 2.4 Staging Validation

- [ ] Run full integration tests
  ```bash
  pytest backend/tests/integration/ -v --staging
  # Target: ≥80% pass rate
  ```

- [ ] Check application logs
  ```bash
  docker-compose -f docker-compose.staging.yml logs investment_backend | grep ERROR
  # Target: 0 errors
  ```

- [ ] Monitor resource usage
  ```bash
  docker stats
  # CPU: <80%, Memory: <80%
  ```

- [ ] Verify database performance
  ```bash
  docker-compose -f docker-compose.staging.yml exec investment_db \
    psql -U investment_user -d investment_db -c "
    SELECT query, mean_time, calls
    FROM pg_stat_statements
    ORDER BY mean_time DESC LIMIT 10;"
  # Target: <100ms mean time for top queries
  ```

---

## Phase 3: Production Deployment

### 3.1 Pre-Deployment Final Checks

- [ ] **STOP** - Verify all blockers resolved
  ```bash
  # Checklist:
  # ✅ Test import error fixed
  # ✅ Test pass rate ≥80%
  # ✅ Mypy errors <10
  # ✅ Staging validation passed
  # ✅ Load tests passed
  # ✅ Team approval obtained
  ```

- [ ] Create deployment tag
  ```bash
  git tag -a v3.0.0 -m "Phase 3.0 Production Release"
  git push origin v3.0.0
  ```

- [ ] Notify team of deployment window
  ```bash
  # Send notifications:
  # - Slack/Teams: 30 minutes before deployment
  # - Email stakeholders
  # - Update status page
  ```

### 3.2 Production Database Migration

- [ ] **CRITICAL:** Create production backup
  ```bash
  # Full database backup
  docker-compose -f docker-compose.prod.yml exec investment_db \
    pg_dump -U investment_user -d investment_db \
    > backups/production-pre-phase3-$(date +%Y%m%d-%H%M%S).sql

  # Verify backup size
  ls -lh backups/production-pre-phase3-*.sql
  # Should be >100MB if data exists
  ```

- [ ] Enable maintenance mode
  ```bash
  # Update load balancer or Nginx config
  # Return 503 Service Unavailable
  ```

- [ ] Stop application servers (keep database up)
  ```bash
  docker-compose -f docker-compose.prod.yml stop investment_backend investment_worker
  ```

- [ ] Apply database migration
  ```bash
  docker-compose -f docker-compose.prod.yml exec investment_db \
    psql -U investment_user -d investment_db \
    < backend/migrations/add_row_locking_versions.sql

  # Verify success
  echo $?  # Should be 0
  ```

- [ ] Verify migration applied
  ```bash
  docker-compose -f docker-compose.prod.yml exec investment_db \
    psql -U investment_user -d investment_db -c "
    SELECT table_name, column_name
    FROM information_schema.columns
    WHERE column_name='version' AND table_schema='public';"
  # Expected: portfolios, positions, transactions
  ```

### 3.3 Production Application Deployment

- [ ] Pull latest code
  ```bash
  git checkout v3.0.0
  ```

- [ ] Build production images
  ```bash
  docker-compose -f docker-compose.prod.yml build --no-cache
  ```

- [ ] Update environment configuration
  ```bash
  # Verify .env.production has all required vars
  grep -E "JWT_SECRET_KEY|DATABASE_URL|REDIS_URL" .env.production
  ```

- [ ] Start updated services
  ```bash
  docker-compose -f docker-compose.prod.yml up -d
  ```

- [ ] Verify services healthy
  ```bash
  # Wait 30 seconds for startup
  sleep 30

  docker-compose -f docker-compose.prod.yml ps
  # All services should show 'Up (healthy)'
  ```

- [ ] Disable maintenance mode
  ```bash
  # Update load balancer/Nginx to route traffic
  ```

### 3.4 Production Smoke Tests

- [ ] **CRITICAL:** Health check
  ```bash
  curl https://yourdomain.com/api/health
  # Expected: {"status":"healthy","version":"3.0.0"}
  ```

- [ ] Database connectivity
  ```bash
  curl https://yourdomain.com/api/health/db
  # Expected: {"status":"connected"}
  ```

- [ ] Authentication test
  ```bash
  # Test login with existing user
  curl -X POST https://yourdomain.com/api/v1/auth/login \
    -H "Content-Type: application/json" \
    -d '{"email":"admin@yourdomain.com","password":"<admin-password>"}'
  # Expected: access_token returned
  ```

- [ ] Security headers
  ```bash
  curl -I https://yourdomain.com/api/health
  # Verify: X-Frame-Options, CSP, X-Content-Type-Options
  ```

- [ ] CSRF protection
  ```bash
  # POST without CSRF token should fail
  TOKEN=$(curl -s -X POST https://yourdomain.com/api/v1/auth/login \
    -H "Content-Type: application/json" \
    -d '{"email":"test@test.com","password":"test"}' | jq -r '.data.access_token')

  curl -X POST https://yourdomain.com/api/portfolios \
    -H "Authorization: Bearer $TOKEN" \
    -d '{"name":"test"}'
  # Expected: 403 Forbidden
  ```

- [ ] Critical user flow
  ```bash
  # 1. Login
  # 2. Get portfolios
  # 3. Create position
  # 4. Verify data consistency
  # (Use Postman collection or automated script)
  ```

---

## Phase 4: Post-Deployment Monitoring

### 4.1 Immediate Monitoring (First Hour)

- [ ] Monitor error rates
  ```bash
  # Check application logs
  docker-compose -f docker-compose.prod.yml logs -f investment_backend | grep -i error
  # Target: 0 errors
  ```

- [ ] Monitor response times
  ```bash
  # Check Grafana dashboard
  # https://yourdomain.com:3001
  # Target: <500ms p95
  ```

- [ ] Check database connections
  ```bash
  docker-compose -f docker-compose.prod.yml exec investment_db \
    psql -U investment_user -d investment_db -c "
    SELECT count(*) FROM pg_stat_activity WHERE state='active';"
  # Should be <50 connections
  ```

- [ ] Monitor CPU/Memory
  ```bash
  docker stats
  # CPU <80%, Memory <80%
  ```

- [ ] Verify no data corruption
  ```bash
  # Run data integrity check
  docker-compose -f docker-compose.prod.yml exec investment_db \
    psql -U investment_user -d investment_db -c "
    SELECT COUNT(*) FROM portfolios WHERE version IS NULL;"
  # Expected: 0 (all rows should have version)
  ```

### 4.2 Extended Monitoring (First 24 Hours)

- [ ] Monitor user activity
  ```bash
  # Check active sessions
  docker-compose -f docker-compose.prod.yml exec investment_cache \
    redis-cli DBSIZE
  # Should increase if users are active
  ```

- [ ] Check alert notifications
  ```bash
  # Verify no critical alerts fired
  # Check AlertManager: https://yourdomain.com:9093
  ```

- [ ] Review slow queries
  ```bash
  docker-compose -f docker-compose.prod.yml exec investment_db \
    psql -U investment_user -d investment_db -c "
    SELECT query, mean_time, calls
    FROM pg_stat_statements
    ORDER BY mean_time DESC LIMIT 10;"
  # Target: <100ms mean time
  ```

- [ ] Monitor cache hit rate
  ```bash
  docker-compose -f docker-compose.prod.yml exec investment_cache \
    redis-cli INFO stats | grep hit_rate
  # Target: >80%
  ```

### 4.3 Weekly Monitoring

- [ ] Review error logs
  ```bash
  # Export and analyze logs
  docker-compose -f docker-compose.prod.yml logs --since 7d > /tmp/week-logs.txt
  grep -i error /tmp/week-logs.txt | wc -l
  ```

- [ ] Check database growth
  ```bash
  docker-compose -f docker-compose.prod.yml exec investment_db \
    psql -U investment_user -d investment_db -c "
    SELECT pg_size_pretty(pg_database_size('investment_db'));"
  ```

- [ ] Verify backups running
  ```bash
  ls -lt backups/ | head -10
  # Should see daily backups
  ```

- [ ] Review performance trends
  ```bash
  # Check Grafana dashboards
  # - Response times trend
  # - Database query performance
  # - Cache hit rates
  ```

---

## Phase 5: Rollback Procedure (If Needed)

### 5.1 Rollback Decision Criteria

**Rollback if:**
- [ ] Error rate >5% sustained for >10 minutes
- [ ] Response time >2s p95 sustained for >10 minutes
- [ ] Database corruption detected
- [ ] Critical security vulnerability discovered
- [ ] Service crashes repeatedly (>3 times in 30 minutes)

### 5.2 Rollback Execution

- [ ] **STOP** - Assess situation
  ```bash
  # Document the issue:
  # - What failed?
  # - Error messages?
  # - Impact on users?
  ```

- [ ] Enable maintenance mode
  ```bash
  # Route traffic away from application
  ```

- [ ] Stop current services
  ```bash
  docker-compose -f docker-compose.prod.yml down
  ```

- [ ] Rollback database migration
  ```bash
  docker-compose -f docker-compose.prod.yml exec investment_db \
    psql -U investment_user -d investment_db \
    < backend/migrations/rollback_row_locking.sql

  # Verify rollback
  docker-compose -f docker-compose.prod.yml exec investment_db \
    psql -U investment_user -d investment_db -c "\d portfolios"
  # version column should be gone
  ```

- [ ] Restore previous application version
  ```bash
  git checkout <previous-stable-tag>
  docker-compose -f docker-compose.prod.yml build
  ```

- [ ] Restore database (if corruption detected)
  ```bash
  # ONLY if database is corrupted
  docker-compose -f docker-compose.prod.yml exec investment_db \
    psql -U investment_user -d investment_db \
    < backups/production-pre-phase3-<timestamp>.sql
  ```

- [ ] Start previous version
  ```bash
  docker-compose -f docker-compose.prod.yml up -d
  ```

- [ ] Verify rollback successful
  ```bash
  curl https://yourdomain.com/api/health
  # Should return previous version number
  ```

- [ ] Disable maintenance mode
  ```bash
  # Re-enable traffic
  ```

- [ ] Document rollback
  ```bash
  # Create incident report:
  # - Reason for rollback
  # - Steps taken
  # - Data loss (if any)
  # - Remediation plan
  ```

**Estimated Rollback Time:** 15-30 minutes
**Data Loss Risk:** LOW (full backup available)

---

## Troubleshooting Guide

### Issue: Services won't start

**Symptoms:**
- Docker containers exit immediately
- Health checks fail

**Diagnosis:**
```bash
# Check logs
docker-compose -f docker-compose.prod.yml logs investment_backend

# Check environment
docker-compose -f docker-compose.prod.yml exec investment_backend env | grep DATABASE_URL
```

**Solutions:**
1. Verify environment variables are set
2. Check database is running and accessible
3. Verify migrations applied successfully
4. Check disk space: `df -h`

---

### Issue: Database migration fails

**Symptoms:**
- Migration script returns error
- Version columns not created

**Diagnosis:**
```bash
# Check for existing columns
docker-compose -f docker-compose.prod.yml exec investment_db \
  psql -U investment_user -d investment_db -c "\d portfolios"

# Check for conflicting indexes
docker-compose -f docker-compose.prod.yml exec investment_db \
  psql -U investment_user -d investment_db -c "\di"
```

**Solutions:**
1. If columns exist, migration already applied
2. If indexes conflict, drop and recreate
3. Check database user has ALTER TABLE permissions

---

### Issue: High error rate after deployment

**Symptoms:**
- 5xx errors in logs
- Users reporting failures

**Diagnosis:**
```bash
# Check error logs
docker-compose -f docker-compose.prod.yml logs investment_backend | grep -i error

# Check database connections
docker-compose -f docker-compose.prod.yml exec investment_db \
  psql -U investment_user -d investment_db -c "SELECT * FROM pg_stat_activity;"
```

**Solutions:**
1. Check if database migration completed
2. Verify environment variables correct
3. Check for race conditions in concurrent updates
4. Consider rollback if errors persist

---

### Issue: CSRF protection blocking legitimate requests

**Symptoms:**
- 403 Forbidden on POST requests
- Frontend can't submit forms

**Diagnosis:**
```bash
# Check if CSRF token being sent
curl -v -X POST https://yourdomain.com/api/portfolios \
  -H "Authorization: Bearer <token>" \
  -H "X-CSRF-Token: <token>"
```

**Solutions:**
1. Verify frontend sends CSRF token in headers
2. Check CSRF middleware configuration
3. Verify token generation in authentication flow

---

## Sign-Off

### Deployment Team

- [ ] **Backend Lead:** _________________ Date: _______
- [ ] **DevOps Lead:** _________________ Date: _______
- [ ] **Security Lead:** _________________ Date: _______
- [ ] **QA Lead:** _________________ Date: _______

### Stakeholders

- [ ] **Product Owner:** _________________ Date: _______
- [ ] **Engineering Manager:** _________________ Date: _______

### Post-Deployment Confirmation

- [ ] **24h Status Check:** _________________ Date: _______
- [ ] **Week 1 Review:** _________________ Date: _______

---

## Appendix

### A. Critical Commands Reference

```bash
# Health check
curl https://yourdomain.com/api/health

# Database backup
docker-compose exec investment_db pg_dump -U investment_user -d investment_db > backup.sql

# View logs
docker-compose -f docker-compose.prod.yml logs -f investment_backend

# Restart service
docker-compose -f docker-compose.prod.yml restart investment_backend

# Rollback migration
docker-compose exec investment_db psql -U investment_user -d investment_db < rollback.sql
```

### B. Contact Information

```
On-Call Engineer: [phone/slack]
Backend Team Lead: [contact]
DevOps Team Lead: [contact]
Security Team: [contact]
Emergency Escalation: [phone]
```

### C. Reference Documentation

- Deployment Guide: `docs/DEPLOYMENT.md`
- Validation Report: `docs/reports/PHASE3_DEPLOYMENT_READINESS.md`
- API Documentation: `https://yourdomain.com/api/docs`
- Migration SQL: `backend/migrations/add_row_locking_versions.sql`

---

**Checklist Version:** 1.0.0
**Last Updated:** 2026-01-27
**Next Review:** After blocker resolution
