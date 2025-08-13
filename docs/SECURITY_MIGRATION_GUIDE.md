# Security Migration Guide

This guide provides step-by-step instructions for migrating to the enhanced security system implemented for the Investment Analysis Application.

## Overview

The security enhancements include:
1. **Secure Secrets Management** - Encrypted storage of API keys and credentials
2. **Enhanced JWT Authentication** - RS256 algorithm with token blacklisting  
3. **SQL Injection Prevention** - Parameterized queries and input sanitization
4. **Database Security Hardening** - SSL/TLS encryption and audit logging
5. **Advanced Rate Limiting** - Multi-algorithm rate limiting with threat detection
6. **Admin Authentication Security** - JWT-based admin access instead of hardcoded tokens

## Pre-Migration Checklist

- [ ] Backup current database and configuration files
- [ ] Verify Redis is available for token blacklisting and rate limiting
- [ ] Ensure PostgreSQL supports SSL connections
- [ ] Have a rollback plan ready
- [ ] Schedule maintenance window for production deployment

## Migration Steps

### Step 1: Install Required Dependencies

Add the following Python packages to your requirements.txt:

```bash
# Security packages
cryptography>=41.0.0
pyotp>=2.8.0
redis>=4.5.0
bcrypt>=4.0.0

# JWT with RS256 support  
PyJWT[crypto]>=2.8.0
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### Step 2: Set Up Master Secret Key

The secrets management system requires a master encryption key.

**For Development:**
```bash
export MASTER_SECRET_KEY="your-super-secure-master-key-32-chars"
```

**For Production:**
Generate a secure master key:
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Add to your secure environment configuration:
```bash
export MASTER_SECRET_KEY="your-generated-key-here"
```

### Step 3: Migrate API Keys to Secrets Manager

Run the migration script to move API keys from environment variables to encrypted storage:

```python
# scripts/migrate_secrets.py
from backend.security.secrets_manager import get_secrets_manager, SecretType
import os

def migrate_api_keys():
    """Migrate API keys from environment to secrets manager"""
    secrets_manager = get_secrets_manager()
    
    api_keys = [
        ("ALPHA_VANTAGE_API_KEY", "alpha_vantage"),
        ("FINNHUB_API_KEY", "finnhub"),
        ("POLYGON_API_KEY", "polygon"),
        ("FMP_API_KEY", "fmp"),
        ("NEWS_API_KEY", "news_api"),
        ("FRED_API_KEY", "fred"),
    ]
    
    for env_var, provider in api_keys:
        api_key = os.getenv(env_var)
        if api_key:
            success = secrets_manager.store_secret(
                f"api_key_{provider}",
                api_key,
                SecretType.API_KEY,
                description=f"API key for {provider}"
            )
            if success:
                print(f"✅ Migrated {provider} API key")
            else:
                print(f"❌ Failed to migrate {provider} API key")

if __name__ == "__main__":
    migrate_api_keys()
```

Run the migration:
```bash
python scripts/migrate_secrets.py
```

### Step 4: Generate RSA Keys for JWT

The enhanced JWT system will automatically generate RSA keys on first startup. To manually generate:

```python
# scripts/generate_jwt_keys.py
from backend.security.jwt_manager import get_jwt_manager

def generate_keys():
    jwt_manager = get_jwt_manager()
    print("✅ RSA keys generated and stored in secrets manager")

if __name__ == "__main__":
    generate_keys()
```

### Step 5: Configure Database Security

Update your database connection to use SSL:

**docker-compose.yml additions:**
```yaml
services:
  postgres:
    environment:
      POSTGRES_SSL_MODE: require
    volumes:
      - ./certs:/var/lib/postgresql/certs:ro
    command: >
      postgres 
      -c ssl=on 
      -c ssl_cert_file=/var/lib/postgresql/certs/server.crt
      -c ssl_key_file=/var/lib/postgresql/certs/server.key
      -c ssl_ca_file=/var/lib/postgresql/certs/ca.crt
```

**Environment variables:**
```bash
# Database SSL configuration
DB_CLIENT_CERT_PATH=/app/certs/client.crt
DB_CLIENT_KEY_PATH=/app/certs/client.key  
DB_CA_CERT_PATH=/app/certs/ca.crt
```

### Step 6: Update Application Configuration

Update your settings to use the secure database engine:

```python
# backend/config/settings.py additions
from backend.security.database_security import create_secure_database_engine

# Replace your existing engine creation with:
engine = create_secure_database_engine(enable_ssl=True, enable_audit=True)
```

### Step 7: Update Admin Endpoints

The admin endpoints now require proper JWT authentication instead of hardcoded tokens.

**Before (INSECURE):**
```python
@router.get("/admin/health")  
async def get_health(admin_token: str = Query(...)):
    if admin_token != "admin-secret-token":
        raise HTTPException(status_code=403)
```

**After (SECURE):**
```python
@router.get("/admin/health")
async def get_health(current_user = Depends(get_current_admin_user)):
    # User is automatically validated as active admin
```

### Step 8: Database Migration

Create and run database migrations for audit logging:

```sql
-- V001__add_audit_tables.sql
CREATE TABLE IF NOT EXISTS audit_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    user_id INTEGER,
    username VARCHAR(255),
    session_id VARCHAR(255),
    query_hash VARCHAR(32),
    affected_tables TEXT[],
    row_count INTEGER,
    duration_ms FLOAT,
    client_ip INET,
    application_name VARCHAR(255),
    success BOOLEAN DEFAULT true,
    error_message TEXT,
    risk_score INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_audit_logs_timestamp ON audit_logs(timestamp);
CREATE INDEX idx_audit_logs_user_id ON audit_logs(user_id);  
CREATE INDEX idx_audit_logs_event_type ON audit_logs(event_type);
CREATE INDEX idx_audit_logs_risk_score ON audit_logs(risk_score);
```

Run the migration:
```bash
alembic upgrade head
```

### Step 9: Update Frontend Configuration

Update your frontend to handle the new JWT tokens:

```javascript
// frontend/src/services/api.service.ts
class ApiService {
  async login(credentials) {
    const response = await fetch('/api/auth/token', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(credentials)
    });
    
    const data = await response.json();
    
    // Store both access and refresh tokens
    localStorage.setItem('access_token', data.access_token);
    localStorage.setItem('refresh_token', data.refresh_token);
    
    return data;
  }
  
  async refreshToken() {
    const refreshToken = localStorage.getItem('refresh_token');
    const response = await fetch('/api/auth/refresh', {
      method: 'POST',
      headers: { 
        'Authorization': `Bearer ${refreshToken}`,
        'Content-Type': 'application/json' 
      }
    });
    
    if (response.ok) {
      const data = await response.json();
      localStorage.setItem('access_token', data.access_token);
      return data.access_token;
    }
    
    // Refresh failed, redirect to login
    this.logout();
    return null;
  }
}
```

### Step 10: Configure Rate Limiting

Add rate limiting configuration to your environment:

```bash
# Rate limiting settings
REDIS_URL=redis://redis:6379/1
RATE_LIMIT_ENABLED=true
RATE_LIMIT_STRICT_MODE=true
```

### Step 11: Update Docker Compose

Update your docker-compose.yml to include security enhancements:

```yaml
version: '3.8'

services:
  backend:
    environment:
      - MASTER_SECRET_KEY=${MASTER_SECRET_KEY}
      - REDIS_URL=redis://redis:6379/1
      - DB_CLIENT_CERT_PATH=/app/certs/client.crt
      - DB_CLIENT_KEY_PATH=/app/certs/client.key
      - DB_CA_CERT_PATH=/app/certs/ca.crt
    volumes:
      - ./certs:/app/certs:ro
      - ./secrets:/app/secrets:rw
    depends_on:
      - postgres
      - redis
  
  redis:
    image: redis:7-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
  
  postgres:
    environment:
      POSTGRES_SSL_MODE: require
    volumes:
      - ./certs:/var/lib/postgresql/certs:ro
    command: >
      postgres 
      -c ssl=on
      -c log_statement=all
      -c log_destination=stderr
      -c logging_collector=on

volumes:
  redis_data:
```

## Post-Migration Verification

### Step 1: Verify Secrets Management
```bash
python -c "
from backend.security.secrets_manager import get_secrets_manager
sm = get_secrets_manager()
print('✅ Secrets manager working' if sm.get_secret('api_key_alpha_vantage') else '❌ Secrets manager failed')
"
```

### Step 2: Verify JWT Authentication
```bash
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"testpass"}'
```

Should return:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

### Step 3: Verify Admin Authentication
```bash
# This should fail (no token)
curl http://localhost:8000/api/admin/health

# This should work (with valid admin JWT)
curl -H "Authorization: Bearer YOUR_ADMIN_JWT" \
     http://localhost:8000/api/admin/health
```

### Step 4: Verify Rate Limiting
```bash
# Make multiple rapid requests to trigger rate limiting
for i in {1..10}; do
  curl -X POST http://localhost:8000/api/auth/login \
    -H "Content-Type: application/json" \
    -d '{"email":"test@example.com","password":"wrong"}'
done
```

Should eventually return `429 Too Many Requests`.

### Step 5: Verify Database Audit Logging
```bash
python -c "
from backend.security.database_security import get_database_security_manager
dm = get_database_security_manager()
report = dm.generate_security_report()
print(f'✅ Audit logging: {report[\"summary\"][\"total_queries\"]} queries logged')
"
```

## Rollback Plan

If issues occur during migration:

### Step 1: Revert to Previous Version
```bash
git checkout previous-version
docker-compose down
docker-compose up -d
```

### Step 2: Restore Database
```bash
psql -h localhost -U postgres investment_db < backup.sql
```

### Step 3: Restore Environment Variables
```bash
# Restore original .env file
cp .env.backup .env
```

## Security Testing

After migration, run security tests:

### Test SQL Injection Prevention
```python
# This should be blocked
try:
    from backend.security.sql_injection_prevention import validate_user_input
    result = validate_user_input("'; DROP TABLE users; --")
    print("❌ SQL injection not detected")  
except HTTPException:
    print("✅ SQL injection blocked")
```

### Test Rate Limiting
```python
import asyncio
from backend.security.rate_limiter import get_rate_limiter, RateLimitCategory

async def test_rate_limiting():
    rate_limiter = get_rate_limiter()
    # Simulate 10 rapid requests
    for i in range(10):
        status = await rate_limiter.check_rate_limit(
            "127.0.0.1", RateLimitCategory.AUTHENTICATION
        )
        if not status.allowed:
            print(f"✅ Rate limited after {i} requests")
            break
    else:
        print("❌ Rate limiting not working")

asyncio.run(test_rate_limiting())
```

## Monitoring and Alerting

Set up monitoring for the new security features:

### Prometheus Metrics
```python
# Add to monitoring configuration
security_metrics = [
    'rate_limit_violations_total',
    'sql_injection_attempts_total', 
    'admin_access_attempts_total',
    'jwt_token_validation_failures_total',
    'database_audit_events_total'
]
```

### Log Monitoring
```bash
# Monitor security logs
tail -f /app/logs/security.log | grep -E "(CRITICAL|HIGH_RISK|VIOLATION)"
```

### Grafana Dashboard
Import the security dashboard from `monitoring/grafana/dashboards/security-dashboard.json`.

## Troubleshooting

### Common Issues

#### Redis Connection Failed
```bash
# Check Redis connectivity
redis-cli -h redis ping
```

#### Secrets Manager Not Working
```bash
# Verify master key
echo $MASTER_SECRET_KEY | wc -c  # Should be > 20
```

#### JWT Verification Failed
```bash
# Check if RSA keys exist
ls -la /app/secrets/
```

#### Database SSL Connection Failed
```bash
# Verify certificates
openssl x509 -in /app/certs/client.crt -text -noout
```

### Support

For additional support:
1. Check the application logs: `docker logs investment-app-backend`
2. Review security audit logs: `cat /app/logs/database_audit.jsonl`
3. Test with the provided verification scripts
4. Consult the security documentation in `docs/SECURITY_IMPLEMENTATION.md`

## Security Considerations

- **Never commit secrets to version control**
- **Regularly rotate API keys and certificates**
- **Monitor audit logs for suspicious activity**
- **Keep dependencies updated for security patches**
- **Use strong master keys (>32 characters)**
- **Implement proper backup procedures for encrypted secrets**
- **Test security measures in staging before production**

The migration is complete when all verification steps pass and the application operates normally with enhanced security features.