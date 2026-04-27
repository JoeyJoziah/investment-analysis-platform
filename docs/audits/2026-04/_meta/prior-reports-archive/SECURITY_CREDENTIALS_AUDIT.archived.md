> **ARCHIVED 2026-04-27 by 08-auth-security-compliance**
> Original: docs/security/SECURITY_CREDENTIALS_AUDIT.md
> Validation summary: 0/5 claims still current — overall status: fully_stale
> See `../../reports/08-auth-security-compliance.md` §2 for per-claim status.
> Redactions applied: 7

# Investment Analysis Platform - Database & Security Credentials Audit
**Generated:** 2025-08-18
**WARNING:** This file contains sensitive information. Keep secure and never commit to version control.

---

## Executive Summary

This comprehensive audit identifies all database-related security variables, credentials, and configuration settings for the investment analysis platform. The audit reveals a complex multi-environment setup with proper credential management patterns but some security concerns that need immediate attention.

---

## 🔴 CRITICAL SECURITY FINDINGS

### High Priority Issues:
1. **Live API Keys in Version Control** - Real financial API keys are present in `.env` file
2. **Weak Default Passwords** - Some credentials use predictable patterns  
3. **Multiple Credential Sources** - Credentials scattered across multiple files
4. **Missing Production Security** - Some placeholder credentials still in production configs

---

## 📊 CREDENTIAL INVENTORY

### 🔐 DATABASE CREDENTIALS

| Variable Name | Purpose | Current Value | Status | Used By | Notes |
|---------------|---------|---------------|--------|---------|-------|
| `DB_HOST` | PostgreSQL Host | localhost / postgres | ✅ Set | All services | Container: postgres |
| `DB_PORT` | PostgreSQL Port | 5432 | ✅ Set | All services | Standard PostgreSQL port |
| `DB_NAME` | Main Database Name | investment_db | ✅ Set | Backend, Workers | Primary application database |
| `DB_USER` | PostgreSQL User | postgres | ✅ Set | All services | Default superuser (needs change) |
| `DB_PASSWORD` | PostgreSQL Password | `[REDACTED-db-pwd]` | 🔴 Exposed | All services | **CHANGE IMMEDIATELY** |
| `POSTGRES_PASSWORD` | Container Postgres Password | `[REDACTED-db-pwd]` | 🔴 Exposed | Docker | Same as DB_PASSWORD |
| `DATABASE_URL` | Full Database Connection String | Composed from above | ✅ Set | Backend | PostgreSQL connection string |

#### TimescaleDB Configuration:
| Variable | Value | Notes |
|----------|--------|-------|
| `POSTGRES_SHARED_BUFFERS` | 256MB / 512MB (prod) | Memory allocation |
| `POSTGRES_EFFECTIVE_CACHE_SIZE` | 1GB / 2GB (prod) | Query planner setting |
| `POSTGRES_MAX_CONNECTIONS` | 100 / 200 | Connection limit |
| `POSTGRES_WORK_MEM` | 4MB / 8MB (prod) | Per-operation memory |

### 🔐 REDIS CREDENTIALS

| Variable Name | Purpose | Current Value | Status | Used By | Notes |
|---------------|---------|---------------|--------|---------|-------|
| `REDIS_HOST` | Redis Host | localhost / redis | ✅ Set | Backend, Workers | Container: redis |
| `REDIS_PORT` | Redis Port | 6379 | ✅ Set | Backend, Workers | Standard Redis port |
| `REDIS_PASSWORD` | Redis Authentication | `RsYque` | 🔴 Weak | All services | **TOO SHORT - CHANGE** |
| `REDIS_DB` | Redis Database Number | 0 | ✅ Set | Backend | Default database |
| `REDIS_URL` | Full Redis Connection String | Composed | ✅ Set | Celery, Backend | Complete connection string |
| `REDIS_MAXMEMORY` | Memory Limit | 256mb / 1gb (prod) | ✅ Set | Redis | Memory allocation |
| `REDIS_MAXMEMORY_POLICY` | Eviction Policy | allkeys-lru | ✅ Set | Redis | Cache eviction strategy |

### 🔐 ELASTICSEARCH CREDENTIALS

| Variable Name | Purpose | Current Value | Status | Used By | Notes |
|---------------|---------|---------------|--------|---------|-------|
| `ELASTICSEARCH_HOST` | ES Host | localhost / elasticsearch | ✅ Set | Backend | Container: elasticsearch |
| `ELASTICSEARCH_PORT` | ES Port | 9200 | ✅ Set | Backend | Standard ES port |
| `ELASTICSEARCH_USER` | ES Username | elastic | ✅ Set | Backend | Default ES user |
| `ELASTICSEARCH_PASSWORD` | ES Password | `CHANGE_THIS_ELASTICSEARCH_PASSWORD` | 🔴 Placeholder | Backend | **NOT SET PROPERLY** |
| `ELASTICSEARCH_URL` | Full ES Connection String | Composed | ⚠️ Partial | Backend | Missing proper auth |
| `ELASTICSEARCH_HEAP_SIZE` | Java Heap Size | 512m / 1g (prod) | ✅ Set | Elasticsearch | Memory allocation |

---

## 🔑 APPLICATION SECURITY CREDENTIALS

### Core Application Secrets

| Variable Name | Purpose | Current Value | Status | Used By | Notes |
|---------------|---------|---------------|--------|---------|-------|
| `SECRET_KEY` | Django/Flask Secret | `[REDACTED-hex64]` | 🔴 Exposed | Backend | **64-char hex - ROTATE** |
| `JWT_SECRET_KEY` | JWT Signing Key | `[REDACTED-hex64]` | 🔴 Exposed | Auth System | **64-char hex - ROTATE** |
| `JWT_ALGORITHM` | JWT Algorithm | HS256 | ✅ Set | Auth System | HMAC SHA-256 |
| `JWT_EXPIRATION_HOURS` | Token Expiry | 24 | ✅ Set | Auth System | 24 hours |
| `FERNET_KEY` | Encryption Key | `2kru2R-XINgBhRtSMDGIloV-X45TxnbSZQh7SAOGl4g=` | 🔴 Multiple | Backend | **INCONSISTENT VALUES** |
| `MASTER_SECRET_KEY` | Master Secret | `[REDACTED-hex64]` | 🔴 Exposed | Global | **ROTATE IMMEDIATELY** |

---

## 💰 FINANCIAL DATA API KEYS

### Primary Data Sources (FREE TIER)

| Provider | Variable Name | Current Value | Daily/Minute Limits | Status | Notes |
|----------|---------------|---------------|---------------------|--------|-------|
| **Alpha Vantage** | `ALPHA_VANTAGE_API_KEY` | `4265EWGEBCXVE3RP` | 25 calls/day | 🔴 **EXPOSED** | **ROTATE KEY** |
| **Finnhub** | `FINNHUB_API_KEY` | `d295ehpr01qhoena0ffgd295ehpr01qhoena0fg0` | 60 calls/minute | 🔴 **EXPOSED** | **ROTATE KEY** |
| **Polygon.io** | `POLYGON_API_KEY` | `lwi0HlBLeyuDwSAIX6H5gpM4jM4xqLgk` | 5 calls/minute | 🔴 **EXPOSED** | **ROTATE KEY** |
| **NewsAPI** | `NEWS_API_KEY` | `c2173d404c67434cbd4ed9f94a71ed67` | 100 requests/day | 🔴 **EXPOSED** | **ROTATE KEY** |

### Secondary Data Sources

| Provider | Variable Name | Current Value | Status | Notes |
|----------|---------------|---------------|--------|-------|
| **Financial Modeling Prep** | `FMP_API_KEY` | `ZefnIE1rkuFeCXo0P3ufyTpDhjmLXSQf` | 🔴 **EXPOSED** | **ROTATE KEY** |
| **MarketAux** | `MARKETAUX_API_KEY` | `t5xMMMiJ1X5hthqMNLAqxvr8OkM0GRnN7PjMoXUm` | 🔴 **EXPOSED** | **ROTATE KEY** |
| **FRED** | `FRED_API_KEY` | `19f1d0fabec0d7fea3d76bbe215bc02f` | 🔴 **EXPOSED** | **ROTATE KEY** |
| **OpenWeather** | `OPENWEATHER_API_KEY` | `714738978ce75d8a1a9e0e37677e4d1e` | 🔴 **EXPOSED** | **ROTATE KEY** |
| **Yahoo Finance** | `YAHOO_FINANCE_API_KEY` | Empty | ⚠️ Optional | Not configured |

---

## 🏗️ INFRASTRUCTURE CREDENTIALS

### Apache Airflow

| Variable Name | Purpose | Current Value | Status | Used By | Notes |
|---------------|---------|---------------|--------|---------|-------|
| `AIRFLOW_DB_HOST` | Airflow DB Host | postgres | ✅ Set | Airflow | Same as main DB |
| `AIRFLOW_DB_PORT` | Airflow DB Port | 5432 | ✅ Set | Airflow | Same as main DB |
| `AIRFLOW_DB_NAME` | Airflow Database | airflow | ✅ Set | Airflow | Separate database |
| `AIRFLOW_DB_USER` | Airflow DB User | airflow | ✅ Set | Airflow | Dedicated user |
| `AIRFLOW_DB_PASSWORD` | Airflow DB Password | `GT2qAeOUct1hMLSbN45CUn07CGJ4nr+mAsg8Qyo39AU=` | 🔴 Exposed | Airflow | **Base64 encoded - CHANGE** |
| `AIRFLOW_ADMIN_USERNAME` | Airflow Web User | admin | ✅ Set | Airflow Web | Default admin |
| `AIRFLOW_ADMIN_PASSWORD` | Airflow Web Password | `CHANGE_THIS_ADMIN_PASSWORD` | 🔴 Placeholder | Airflow Web | **NOT SET** |
| `AIRFLOW_FERNET_KEY` | Airflow Encryption | Multiple values | 🔴 Inconsistent | Airflow | **MULTIPLE DIFFERENT VALUES** |
| `AIRFLOW_SECRET_KEY` | Airflow Session Key | `[REDACTED-hex64]` | 🔴 Exposed | Airflow Web | **ROTATE** |
| `FLOWER_PASSWORD` | Celery Flower UI | `flower123` / `secure_flower_password_123` | 🔴 Weak | Flower | **WEAK PASSWORD** |

### Celery Configuration

| Variable Name | Purpose | Current Value | Status | Used By | Notes |
|---------------|---------|---------------|--------|---------|-------|
| `CELERY_BROKER_URL` | Message Broker | Redis URL | ✅ Set | Workers | Uses Redis |
| `CELERY_RESULT_BACKEND` | Result Store | Redis URL | ✅ Set | Workers | Uses Redis |
| `CELERY_WORKER_CONCURRENCY` | Worker Threads | 2 / 4 | ✅ Set | Workers | Performance setting |

### Monitoring & Observability

| Variable Name | Purpose | Current Value | Status | Used By | Notes |
|---------------|---------|---------------|--------|---------|-------|
| `GRAFANA_ADMIN_USER` | Grafana Username | admin | ✅ Set | Grafana | Default admin |
| `GRAFANA_ADMIN_PASSWORD` | Grafana Password | `6UgoVpszXkGLzaBzohoh` | 🔴 Exposed | Grafana | **CHANGE** |
| `GRAFANA_PASSWORD` | Grafana Password | `6UgoVpszXkGLzaBzohoh` | 🔴 Exposed | Grafana | **DUPLICATE VALUE** |
| `SENTRY_DSN` | Error Tracking | Empty | ⚠️ Optional | Backend | Not configured |
| `SLACK_WEBHOOK_URL` | Notifications | Empty | ⚠️ Optional | Alerts | Not configured |

---

## 🔐 SECURITY & COMPLIANCE CONFIGURATION

### SEC Compliance

| Variable Name | Purpose | Current Value | Status | Required | Notes |
|---------------|---------|---------------|--------|----------|-------|
| `SEC_EDGAR_USER_AGENT` | SEC API ID | `InvestmentApp admin@example.com` | ⚠️ Placeholder | ✅ Yes | **UPDATE EMAIL** |
| `SEC_COMPLIANCE_MODE` | Enable SEC Mode | enabled | ✅ Set | ✅ Yes | Compliance enabled |
| `AUDIT_LOG_ENABLED` | Audit Logging | true | ✅ Set | ✅ Yes | Required for SEC |
| `AUDIT_LOG_RETENTION_DAYS` | Log Retention | 2555 (7 years) | ✅ Set | ✅ Yes | SEC requirement |
| `TRANSACTION_LOGGING` | Transaction Logs | true | ✅ Set | ✅ Yes | Financial compliance |

### GDPR Compliance

| Variable Name | Purpose | Current Value | Status | Required | Notes |
|---------------|---------|---------------|--------|----------|-------|
| `GDPR_COMPLIANCE` | Enable GDPR | enabled | ✅ Set | ✅ Yes | EU compliance |
| `PII_ENCRYPTION` | Encrypt PII | enabled | ✅ Set | ✅ Yes | Data protection |
| `DATA_ANONYMIZATION` | Anonymize Data | enabled | ✅ Set | ✅ Yes | Privacy protection |
| `RIGHT_TO_BE_FORGOTTEN` | Data Deletion | enabled | ✅ Set | ✅ Yes | GDPR requirement |
| `COOKIE_CONSENT_REQUIRED` | Cookie Consent | true | ✅ Set | ✅ Yes | GDPR requirement |

---

## 🌐 DEPLOYMENT & INFRASTRUCTURE

### Docker & Container Settings

| Variable Name | Purpose | Current Value | Status | Used By | Notes |
|---------------|---------|---------------|--------|---------|-------|
| `DOCKER_REGISTRY` | Container Registry | docker.io | ✅ Set | Deployment | Docker Hub |
| `DOCKER_USERNAME` | Registry Username | Empty | ⚠️ Optional | Deployment | Not configured |
| `DOCKER_PASSWORD` | Registry Password | Empty | ⚠️ Optional | Deployment | Not configured |

### Cloud Provider (DigitalOcean)

| Variable Name | Purpose | Current Value | Status | Used By | Notes |
|---------------|---------|---------------|--------|---------|-------|
| `DIGITALOCEAN_ACCESS_TOKEN` | API Token | Empty | ⚠️ Optional | Deployment | Not configured |
| `DIGITALOCEAN_CLUSTER_ID` | K8s Cluster | Empty | ⚠️ Optional | Deployment | Not configured |
| `DIGITALOCEAN_SPACES_KEY` | Object Storage Key | Empty | ⚠️ Optional | Backup | Not configured |
| `DIGITALOCEAN_SPACES_SECRET` | Object Storage Secret | Empty | ⚠️ Optional | Backup | Not configured |

### AWS Alternative

| Variable Name | Purpose | Current Value | Status | Used By | Notes |
|---------------|---------|---------------|--------|---------|-------|
| `AWS_ACCESS_KEY_ID` | AWS Access Key | Empty | ⚠️ Optional | Backup | Not configured |
| `AWS_SECRET_ACCESS_KEY` | AWS Secret Key | Empty | ⚠️ Optional | Backup | Not configured |
| `AWS_S3_BUCKET` | S3 Bucket Name | Empty | ⚠️ Optional | Backup | Not configured |

---

## 📱 FRONTEND CONFIGURATION

### React Environment Variables

| Variable Name | Purpose | Current Value | Status | Used By | Notes |
|---------------|---------|---------------|--------|---------|-------|
| `REACT_APP_API_URL` | Backend API URL | http://localhost:8000 | ✅ Set | Frontend | Development URL |
| `REACT_APP_WS_URL` | WebSocket URL | ws://localhost:8000/api/ws | ✅ Set | Frontend | Development URL |
| `REACT_APP_ENVIRONMENT` | Frontend Environment | development | ✅ Set | Frontend | Environment flag |
| `REACT_APP_GOOGLE_ANALYTICS_ID` | Analytics ID | Empty | ⚠️ Optional | Frontend | Not configured |
| `REACT_APP_SENTRY_DSN` | Frontend Error Tracking | Empty | ⚠️ Optional | Frontend | Not configured |

---

## 🔧 PERFORMANCE & OPTIMIZATION

### Database Performance

| Variable Name | Purpose | Current Value | Status | Notes |
|---------------|---------|---------------|--------|-------|
| `DB_POOL_SIZE` | Connection Pool Size | 20 | ✅ Set | Backend connection pooling |
| `DB_POOL_TIMEOUT` | Pool Timeout | 30 | ✅ Set | Connection timeout |
| `DB_POOL_RECYCLE` | Pool Recycle Time | 3600 | ✅ Set | 1 hour recycle |

### Cache Configuration

| Variable Name | Purpose | Current Value | Status | Notes |
|---------------|---------|---------------|--------|-------|
| `CACHE_TTL_DEFAULT` | Default Cache TTL | 300 | ✅ Set | 5 minutes |
| `CACHE_TTL_STOCK_PRICES` | Price Cache TTL | 60 | ✅ Set | 1 minute |
| `CACHE_TTL_STOCK_FUNDAMENTALS` | Fundamentals Cache TTL | 3600 | ✅ Set | 1 hour |
| `CACHE_TTL_NEWS` | News Cache TTL | 1800 | ✅ Set | 30 minutes |
| `CACHE_TTL_RECOMMENDATIONS` | Recommendations Cache TTL | 900 | ✅ Set | 15 minutes |

---

## 🚨 SECURITY ISSUES IDENTIFIED

### Critical Issues (Fix Immediately)

1. **🔴 EXPOSED API KEYS IN VERSION CONTROL**
   - All financial API keys are visible in `.env` file
   - Keys are committed to git repository
   - **Action:** Rotate all API keys immediately

2. **🔴 WEAK DATABASE PASSWORD**
   - `DB_PASSWORD` is exposed: `[REDACTED-db-pwd]`
   - **Action:** Generate new strong password (32+ characters)

3. **🔴 EXPOSED APPLICATION SECRETS**
   - `SECRET_KEY` and `JWT_SECRET_KEY` are visible
   - **Action:** Regenerate all application secrets

4. **🔴 INCONSISTENT FERNET KEYS**
   - Multiple different `FERNET_KEY` values across files
   - **Action:** Use single, properly generated Fernet key

### High Priority Issues

1. **🟠 WEAK REDIS PASSWORD**
   - `REDIS_PASSWORD` is too short: `RsYque`
   - **Action:** Generate 32+ character password

2. **🟠 PLACEHOLDER PASSWORDS**
   - Elasticsearch password not set properly
   - Airflow admin password is placeholder
   - **Action:** Set proper passwords for all services

3. **🟠 DEFAULT USERNAMES**
   - Using default `postgres` superuser
   - **Action:** Create dedicated application users

### Medium Priority Issues

1. **🟡 MISSING PRODUCTION CONFIGURATIONS**
   - Many optional services not configured
   - **Action:** Configure production monitoring and backup

2. **🟡 HARDCODED DEVELOPMENT SETTINGS**
   - URLs pointing to localhost in production configs
   - **Action:** Environment-specific configurations

---

## 📋 RECOMMENDED IMMEDIATE ACTIONS

### 1. Emergency Security Response (Do Now)

```bash
# 1. Rotate all API keys at provider websites
# 2. Generate new application secrets
./scripts/generate_secrets.sh > .env.new
# 3. Update .env file with new secrets
# 4. Restart all services
docker-compose down && docker-compose up -d
```

### 2. Database Security Hardening

```sql
-- Create dedicated application users
CREATE USER investment_app WITH PASSWORD 'new_secure_password_32_chars';
CREATE USER investment_readonly WITH PASSWORD 'new_readonly_password_32_chars';

-- Grant minimal required permissions
GRANT CONNECT ON DATABASE investment_db TO investment_app;
GRANT USAGE ON SCHEMA public TO investment_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO investment_app;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO investment_readonly;
```

### 3. Environment Variable Management

1. **Create proper `.env.production`** with all required variables
2. **Remove `.env` from version control** (add to `.gitignore`)
3. **Use secrets management** for Kubernetes deployments
4. **Implement credential rotation schedule** (monthly)

### 4. Monitoring & Alerting Setup

1. **Configure proper monitoring credentials**
2. **Set up alerts for credential usage**
3. **Implement audit logging for all credential access**
4. **Monitor API key usage** to detect compromise

---

## 🔒 CREDENTIAL ROTATION SCHEDULE

| Credential Type | Rotation Frequency | Next Due | Automation |
|----------------|-------------------|----------|------------|
| Database Passwords | Monthly | Immediate | Manual |
| Application Secrets | Quarterly | Immediate | Script Available |
| API Keys | Bi-annually | Immediate | Manual |
| Monitoring Passwords | Monthly | Immediate | Manual |

---

## 📁 FILE LOCATIONS AUDIT

### Environment Files Found:
- ✅ `./.env` - **MAIN FILE** (contains live credentials)
- ✅ `./.env.example` - Template file (safe)
- ✅ `./.env.production.example` - Production template (safe)
- ✅ `./.env.airflow` - Airflow-specific vars (contains credentials)
- ✅ `./.env.airflow.template` - Airflow template (safe)
- ✅ `./scripts/.env` - Scripts environment (contains credentials)
- ✅ `./frontend/web/.env.production` - Frontend production (safe URLs)

### Docker Compose Files:
- ✅ `./docker-compose.yml` - Uses environment variables (secure)
- ✅ `./docker-compose.prod.yml` - Production overrides (secure)
- ✅ `./docker-compose.dev.yml` - Development overrides (secure)
- ✅ `./docker-compose.test.yml` - Testing configuration (secure)

### Configuration Files:
- ✅ `./backend/config/settings.py` - Reads from environment (secure)
- ✅ `./backend/config/database.py` - Database configuration (secure)
- ✅ `./config/infrastructure/kubernetes/secrets-sealed.yaml` - Template (secure)

### Management Scripts:
- ✅ `./scripts/generate_secrets.sh` - Secret generation utility
- ✅ `./scripts/manage-secrets.sh` - Kubernetes secret management

---

## 💡 BEST PRACTICES IMPLEMENTED

### ✅ Good Security Practices Found:
1. **Environment variable usage** - No hardcoded credentials in code
2. **Separate environment files** - Different configs per environment  
3. **Docker secrets support** - Compose files use variable substitution
4. **Kubernetes secrets** - Sealed secrets implementation available
5. **Secret generation scripts** - Automated secure credential generation
6. **Connection pooling** - Database connections properly managed
7. **Compliance settings** - SEC and GDPR configurations present

### ❌ Security Gaps Found:
1. **Credentials in version control** - `.env` files committed
2. **Weak passwords** - Some credentials too simple
3. **Inconsistent encryption keys** - Multiple Fernet keys
4. **Missing production setup** - Optional services not configured
5. **No rotation schedule** - Credentials not regularly rotated

---

## 🎯 NEXT STEPS

### Immediate (Today):
1. **Rotate all exposed credentials**
2. **Remove .env from git history**
3. **Generate new strong passwords**
4. **Update production configurations**

### Short Term (This Week):
1. **Implement secrets management**
2. **Set up monitoring alerts**
3. **Create backup strategies**
4. **Document credential procedures**

### Long Term (This Month):
1. **Automate credential rotation**
2. **Implement zero-trust security**
3. **Regular security audits**
4. **Staff security training**

---

**Document Classification:** CONFIDENTIAL - Internal Use Only
**Review Date:** 2025-08-25
**Next Audit:** 2025-09-01