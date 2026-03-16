# SECRET ROTATION PLAN - PHASE 1 CRITICAL REMEDIATION

## Priority: CRITICAL
**Timeline**: Complete within 24-48 hours

---

## 1. EXPOSED SECRETS INVENTORY

### Files Containing Secrets (12 Total)
```
/Users/devinmcgrath/Documents/GitHub/investment-analysis-platform/
├── .env (381 lines - ACTIVE IN USE)
├── .env.secure
├── .env.airflow
├── .env.production.example
├── .env.template
├── .env.example
├── .env.airflow.template
└── .env_backup_DONOTUSE/
    ├── .env.secure
    ├── .env.production.example
    ├── .env.production.backup
    ├── .env.example
    └── .env.airflow.template
```

### Secret Categories Requiring Rotation

#### A. Application Secrets
- **SECRET_KEY** - Used for: FastAPI sessions, CSRF protection
  - Rotation: Generate new `secrets.token_hex(32)`
  - Impact: ALL user sessions invalidated
  - Command: `python -c "import secrets; print(secrets.token_hex(32))"`

- **JWT_SECRET_KEY** - Used for: Authentication tokens
  - Rotation: Generate new `secrets.token_urlsafe(64)`
  - Impact: ALL users must re-login
  - Command: `python -c "import secrets; print(secrets.token_urlsafe(64))"`

- **FERNET_KEY** - Used for: Data encryption at rest
  - Rotation: `from cryptography.fernet import Fernet; Fernet.generate_key().decode()`
  - Impact: Encrypted data needs re-encryption migration
  - **WARNING**: Requires data migration script

#### B. Database Credentials
- **DB_PASSWORD** (PostgreSQL)
  - Rotation: Update in PostgreSQL + all services
  - Impact: Database connection interruption
  - Steps:
    1. `ALTER USER postgres WITH PASSWORD 'NEW_PASSWORD';`
    2. Update .env, docker-compose.yml
    3. Restart all database-dependent services

#### C. Redis Credentials
- **REDIS_PASSWORD**
  - Rotation: Update in Redis config + all services
  - Impact: Cache/session connection interruption
  - Steps:
    1. `redis-cli CONFIG SET requirepass "NEW_PASSWORD"`
    2. `redis-cli CONFIG REWRITE`
    3. Update .env, docker-compose.yml

#### D. External API Keys
- **ANTHROPIC_API_KEY** - Claude AI API
  - Provider: https://console.anthropic.com/settings/keys
  - Rotation: Generate new key, test, rotate

- **OPENAI_API_KEY** - OpenAI GPT API
  - Provider: https://platform.openai.com/api-keys
  - Rotation: Generate new key, test, rotate

- **GOOGLE_API_KEY** / **GOOGLE_CLOUD_API_KEY** - Google Cloud services
  - Provider: https://console.cloud.google.com/apis/credentials
  - Rotation: Generate new key, restrict scope

- **ALPACA_API_KEY** / **ALPACA_SECRET_KEY** - Stock data
  - Provider: https://app.alpaca.markets/paper/dashboard/overview
  - Rotation: Generate new keypair

- **ALPHA_VANTAGE_API_KEY** - Market data
  - Provider: https://www.alphavantage.co/support/#api-key
  - Rotation: Generate new key

- **FINNHUB_API_KEY** - Financial data
  - Provider: https://finnhub.io/dashboard
  - Rotation: Generate new key

- **NEWS_API_KEY** - News aggregation
  - Provider: https://newsapi.org/account
  - Rotation: Generate new key

- **FRED_API_KEY** - Economic data
  - Provider: https://fred.stlouisfed.org/docs/api/api_key.html
  - Rotation: Generate new key

#### E. Monitoring & Infrastructure
- **GRAFANA_PASSWORD** - Monitoring dashboard
  - Rotation: Update in Grafana UI or config

- **PROMETHEUS_PASSWORD** (if configured)
  - Rotation: Update in prometheus.yml

- **AIRFLOW_ADMIN_PASSWORD** - Workflow orchestration
  - Rotation: Update via Airflow CLI
  - Command: `airflow users create --username admin --password NEW_PASS --role Admin`

---

## 2. SECRET REGENERATION COMMANDS

### Generate All New Secrets (Safe to Run)
```bash
#!/bin/bash
# Script: generate_new_secrets.sh
# Purpose: Generate cryptographically secure secrets for rotation

echo "=== SECRET GENERATION SCRIPT ==="
echo ""

echo "SECRET_KEY (FastAPI):"
python3 -c "import secrets; print(secrets.token_hex(32))"
echo ""

echo "JWT_SECRET_KEY (Authentication):"
python3 -c "import secrets; print(secrets.token_urlsafe(64))"
echo ""

echo "FERNET_KEY (Encryption):"
python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
echo ""

echo "DB_PASSWORD (PostgreSQL - 32 chars alphanumeric):"
python3 -c "import secrets, string; chars=string.ascii_letters+string.digits; print(''.join(secrets.choice(chars) for _ in range(32)))"
echo ""

echo "REDIS_PASSWORD (Redis - 32 chars alphanumeric):"
python3 -c "import secrets, string; chars=string.ascii_letters+string.digits; print(''.join(secrets.choice(chars) for _ in range(32)))"
echo ""

echo "SESSION_SECRET_KEY (Sessions):"
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
echo ""

echo "=== COPY THESE VALUES TO YOUR NEW .env FILE ==="
```

---

## 3. PROVIDER ROTATION URLS

| Provider | Rotation URL | Notes |
|----------|--------------|-------|
| **Anthropic Claude** | https://console.anthropic.com/settings/keys | Generate new, delete old |
| **OpenAI** | https://platform.openai.com/api-keys | Revoke old key immediately |
| **Google Cloud** | https://console.cloud.google.com/apis/credentials | Set usage quotas |
| **Alpaca Markets** | https://app.alpaca.markets/paper/dashboard/overview | Regenerate both API key and secret |
| **Alpha Vantage** | https://www.alphavantage.co/support/#api-key | Email support for new key |
| **Finnhub** | https://finnhub.io/dashboard | Generate new key |
| **NewsAPI** | https://newsapi.org/account | Generate new key |
| **FRED** | https://fred.stlouisfed.org/docs/api/api_key.html | Request new key |

---

## 4. GIT HISTORY CLEANUP SCRIPT

**WARNING**: This rewrites git history. Coordinate with all team members.

```bash
#!/bin/bash
# Script: cleanup_git_secrets.sh
# Purpose: Remove secrets from git history

echo "=== GIT HISTORY SECRET CLEANUP ==="
echo "WARNING: This rewrites git history!"
echo "Ensure all team members are notified!"
echo ""
read -p "Continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Aborted."
    exit 1
fi

# Backup current branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
git branch backup-before-secret-cleanup

# Remove .env files from history using git-filter-repo (recommended)
# Install: pip install git-filter-repo
echo "Removing .env files from history..."
git filter-repo --path .env --invert-paths --force
git filter-repo --path .env.secure --invert-paths --force
git filter-repo --path .env.production --invert-paths --force
git filter-repo --path .env_backup_DONOTUSE/ --invert-paths --force

# Alternative using BFG Repo-Cleaner (if git-filter-repo unavailable)
# Download: https://rtyley.github.io/bfg-repo-cleaner/
# java -jar bfg.jar --delete-files .env
# java -jar bfg.jar --delete-files .env.secure
# git reflog expire --expire=now --all && git gc --prune=now --aggressive

echo "Git history cleaned. Force push required:"
echo "  git push origin --force --all"
echo "  git push origin --force --tags"
echo ""
echo "IMPORTANT: All collaborators must re-clone the repository!"
```

---

## 5. ROTATION EXECUTION PLAN

### Phase 1: Preparation (Day 1, Hour 1-2)
1. **Generate new secrets** using script above
2. **Create new API keys** from all providers
3. **Test new credentials** in staging environment
4. **Prepare rollback plan**

### Phase 2: Rotation (Day 1, Hour 3-4)
1. **Schedule maintenance window** (2-4 hours)
2. **Backup databases** (PostgreSQL, Redis)
3. **Update .env.template** with new format (no secrets)
4. **Rotate secrets** in this order:
   - External API keys (no downtime)
   - Redis password (requires restart)
   - PostgreSQL password (requires restart)
   - Application secrets (requires restart)
5. **Restart all services**
6. **Verify functionality**

### Phase 3: Verification (Day 1, Hour 5-6)
1. **Test all API endpoints**
2. **Verify authentication flows**
3. **Check external API connections**
4. **Monitor error logs**

### Phase 4: Git Cleanup (Day 2)
1. **Run git history cleanup** (off-hours)
2. **Force push to remote**
3. **Notify team to re-clone**

---

## 6. .env.template SAFE STRUCTURE

```bash
# ============================================================================
# INVESTMENT ANALYSIS APP - ENVIRONMENT CONFIGURATION
# ============================================================================
# Copy this file to .env and fill in actual values
# NEVER commit .env files to version control!
# ============================================================================

# Environment
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Application Core (GENERATE FRESH VALUES)
SECRET_KEY=generate_with_python_secrets_module_64_chars
JWT_SECRET_KEY=generate_with_python_secrets_module_128_chars
FERNET_KEY=generate_with_cryptography_fernet_module

# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=investment_db
DB_USER=postgres
DB_PASSWORD=generate_strong_password_32_chars
DATABASE_URL=postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=generate_strong_password_32_chars
REDIS_URL=redis://:${REDIS_PASSWORD}@${REDIS_HOST}:${REDIS_PORT}/0

# External API Keys (REQUEST FROM PROVIDERS)
ANTHROPIC_API_KEY=sk-ant-api03-xxxxx
OPENAI_API_KEY=sk-xxxxx
GOOGLE_API_KEY=xxxxx
ALPACA_API_KEY=xxxxx
ALPACA_SECRET_KEY=xxxxx
ALPHA_VANTAGE_API_KEY=xxxxx
FINNHUB_API_KEY=xxxxx
NEWS_API_KEY=xxxxx
FRED_API_KEY=xxxxx

# Monitoring
GRAFANA_PASSWORD=generate_strong_password
AIRFLOW_ADMIN_PASSWORD=generate_strong_password

# Security Headers
FORCE_HTTPS=true
CORS_ALLOWED_ORIGINS=https://yourdomain.com,https://api.yourdomain.com
```

---

## 7. VERIFICATION CHECKLIST

After rotation, verify:
- [ ] All services start successfully
- [ ] Users can login with existing accounts
- [ ] External API calls succeed
- [ ] Database queries execute
- [ ] Redis cache operations work
- [ ] WebSocket connections establish
- [ ] Monitoring dashboards accessible
- [ ] No secrets in logs
- [ ] .env added to .gitignore
- [ ] Git history cleaned
- [ ] Team notified of changes

---

## 8. ROLLBACK PROCEDURE

If rotation fails:
1. **Stop all services**
2. **Restore from backup**:
   - Copy `.env.backup` to `.env`
   - Restore database from backup
   - Restore Redis snapshot
3. **Restart services**
4. **Investigate failure**
5. **Plan retry**

---

## 9. POST-ROTATION SECURITY MEASURES

1. **Enable secret scanning** (GitHub Advanced Security)
2. **Use secrets manager** (AWS Secrets Manager, HashiCorp Vault)
3. **Implement secret rotation schedule** (90 days)
4. **Monitor for secret exposure** (truffleHog, gitleaks)
5. **Enforce pre-commit hooks** (detect-secrets)

---

## EMERGENCY CONTACTS

- **Security Lead**: [Contact Info]
- **DevOps Lead**: [Contact Info]
- **Database Admin**: [Contact Info]

---

**Document Version**: 1.0
**Created**: 2026-01-27
**Classification**: CONFIDENTIAL - INTERNAL USE ONLY
