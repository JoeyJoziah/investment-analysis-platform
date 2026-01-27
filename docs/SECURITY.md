# Security Guidelines & Best Practices

**Last Updated**: 2026-01-27
**Version**: 1.0.0
**Classification**: Internal Use Only

---

## Table of Contents

1. [Security Overview](#security-overview)
2. [Authentication & Authorization](#authentication--authorization)
3. [Data Protection](#data-protection)
4. [API Security](#api-security)
5. [Database Security](#database-security)
6. [Infrastructure Security](#infrastructure-security)
7. [Secrets Management](#secrets-management)
8. [Compliance Requirements](#compliance-requirements)
9. [Incident Response](#incident-response)
10. [Security Checklist](#security-checklist)

---

## Security Overview

### Security Architecture

The Investment Analysis Platform implements **defense-in-depth** with multiple layers:

```
┌─────────────────────────────────────────────────┐
│         Frontend (React + HTTPS)                │
│    • Secure connections (TLS 1.2+)              │
│    • CSRF protection tokens                     │
│    • XSS prevention (Content-Security-Policy)   │
└───────────────┬─────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────┐
│      API Layer (FastAPI + Authentication)       │
│    • OAuth2/JWT tokens                          │
│    • Rate limiting (Redis)                      │
│    • Input validation (Pydantic)                │
│    • API key management                         │
└───────────────┬─────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────┐
│    Business Logic (Role-Based Access)           │
│    • 6 user roles with permissions              │
│    • Audit logging for all operations           │
│    • Data encryption at application level       │
└───────────────┬─────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────┐
│    Database Layer (PostgreSQL + Security)       │
│    • Row-level security                         │
│    • Prepared statements (SQL injection prevent)│
│    • Encrypted sensitive columns                │
│    • Connection pooling with auth               │
└─────────────────────────────────────────────────┘
```

### Security Standards Compliance

- ✅ **OWASP Top 10** - All major vulnerabilities addressed
- ✅ **SEC 2025 Compliance** - Investment recommendation disclosures
- ✅ **GDPR Compliance** - Data protection regulations
- ✅ **PCI DSS** - If handling payment data
- ✅ **SOC 2 Ready** - Audit-trail and monitoring
- ✅ **CWE Top 25** - Common weakness mitigation

---

## Authentication & Authorization

### OAuth2 & JWT Implementation

#### Token-Based Authentication

```bash
# Login with credentials
curl -X POST https://yourdomain.com/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "user@example.com",
    "password": "secure-password"
  }'

# Response:
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

#### Token Configuration

```python
# From backend/utils/auth.py
JWT_ALGORITHM = "HS256"
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_EXPIRATION = 3600  # 1 hour access token
JWT_REFRESH_EXPIRATION = 604800  # 7 days refresh token

# Key Security:
# - Never commit JWT_SECRET_KEY to version control
# - Use 256-bit+ random keys (32+ bytes)
# - Rotate keys periodically (every 90 days recommended)
# - Use different keys for dev/staging/production
```

#### Refresh Token Flow

```python
# Clients should use refresh tokens to get new access tokens
curl -X POST https://yourdomain.com/api/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{
    "refresh_token": "eyJhbGciOiJIUzI1NiIs..."
  }'

# Response:
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "expires_in": 3600
}

# Benefits:
# - Short-lived access tokens reduce exposure if compromised
# - Refresh tokens stored securely (httpOnly cookies)
# - Ability to revoke refresh tokens for logged-out users
```

### Role-Based Access Control (RBAC)

#### User Roles

```
1. GUEST (0)
   - No authenticated access
   - Public endpoints only

2. USER (1)
   - Read own data
   - Create watchlists
   - View recommendations
   - Scope: Own data only

3. PREMIUM_USER (2)
   - All USER permissions
   - Advanced analysis
   - Custom portfolios
   - API access
   - Scope: Own data only

4. ANALYST (3)
   - View all user data (for analysis)
   - Create shared reports
   - Scope: All user data

5. ADMIN (4)
   - Full system access
   - User management
   - System settings
   - Audit logs
   - Scope: Entire system

6. SUPERADMIN (5)
   - System configuration
   - Database access
   - Infrastructure changes
   - Scope: Complete system
```

#### Permission Checks

```python
# From backend/utils/auth.py
from fastapi import Depends, HTTPException, status

async def require_auth(
    current_user: User = Depends(get_current_user),
    required_role: UserRole = UserRole.USER
):
    if current_user.role < required_role:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    return current_user

# Usage in endpoints:
@router.get("/api/admin/users")
async def get_all_users(user = Depends(require_auth(required_role=UserRole.ADMIN))):
    return users
```

#### Multi-Factor Authentication (MFA)

```python
# TOTP-based MFA enabled by default
# Users can scan QR code with authenticator app

curl -X POST https://yourdomain.com/api/auth/enable-mfa \
  -H "Authorization: Bearer $JWT_TOKEN"

# Response includes QR code for scanning

# Login with MFA:
curl -X POST https://yourdomain.com/api/auth/login-mfa \
  -d '{
    "username": "user@example.com",
    "password": "password",
    "totp_code": "123456"
  }'
```

---

## Data Protection

### Encryption at Rest

#### Database Column Encryption

```python
# From backend/models/user.py
from cryptography.fernet import Fernet

class User(Base):
    __tablename__ = "users"

    id: int = Column(Integer, primary_key=True)
    email: str = Column(String, unique=True)

    # Encrypted columns
    ssn: str = Column(String)  # Encrypted using Fernet
    phone: str = Column(String)  # Encrypted
    address: str = Column(String)  # Encrypted

    @property
    def decrypted_ssn(self) -> str:
        if not self.ssn:
            return None
        cipher = Fernet(os.getenv("DATA_ENCRYPTION_KEY").encode())
        return cipher.decrypt(self.ssn.encode()).decode()
```

#### GDPR Data Encryption

```python
# From backend/utils/data_anonymization.py
from cryptography.fernet import Fernet

class DataAnonymizer:
    def __init__(self):
        key = os.getenv("GDPR_ENCRYPTION_KEY")
        if not key:
            raise ValueError("GDPR_ENCRYPTION_KEY not configured")
        self.cipher = Fernet(key.encode())

    def encrypt_personal_data(self, data: str) -> str:
        """Encrypt personal data for GDPR compliance"""
        return self.cipher.encrypt(data.encode()).decode()

    def decrypt_personal_data(self, encrypted: str) -> str:
        """Decrypt personal data"""
        return self.cipher.decrypt(encrypted.encode()).decode()

    def anonymize_user_data(self, user_id: int) -> bool:
        """Anonymize user data (right to be forgotten)"""
        # Replace personal data with hashes
        # Retain only aggregated, non-identifiable data
```

### Encryption in Transit

#### HTTPS/TLS Configuration

```nginx
# From nginx configuration
server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    # SSL certificates
    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    # TLS version (1.2 minimum for security)
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # HSTS (force HTTPS)
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # Security headers
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "DENY" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Permissions-Policy "geolocation=(), microphone=(), camera=()" always;
}
```

#### WebSocket Security

```python
# From backend/api/websocket.py
from fastapi import WebSocket, WebSocketDisconnect

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # Verify origin
    origin = websocket.headers.get("origin")
    if origin not in ALLOWED_ORIGINS:
        await websocket.close(code=1008, reason="Unauthorized origin")
        return

    # Verify authentication
    token = websocket.query_params.get("token")
    user = verify_token(token)
    if not user:
        await websocket.close(code=1008, reason="Unauthorized")
        return

    # Connection established
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()
            # Process authenticated message
    except WebSocketDisconnect:
        pass
```

---

## API Security

### Input Validation

#### Pydantic Models (Backend)

```python
# From backend/schemas/*.py
from pydantic import BaseModel, Field, EmailStr, validator
from typing import Optional

class CreateUserRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr  # Validates email format
    password: str = Field(..., min_length=12)  # Minimum 12 chars
    age: int = Field(..., ge=18, le=150)  # Age validation

    @validator('password')
    def password_strength(cls, v):
        # Check for complexity
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain uppercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain number')
        if not any(c in '!@#$%^&*' for c in v):
            raise ValueError('Password must contain special character')
        return v

    class Config:
        # Prevent arbitrary extra fields
        extra = 'forbid'
```

#### Zod Models (Frontend)

```typescript
// From frontend/src/schemas/*.ts
import { z } from 'zod'

export const createUserSchema = z.object({
  username: z.string().min(3).max(50),
  email: z.string().email(),
  password: z.string()
    .min(12)
    .regex(/[A-Z]/, 'Must contain uppercase')
    .regex(/[0-9]/, 'Must contain number')
    .regex(/[!@#$%^&*]/, 'Must contain special character'),
  age: z.number().int().min(18).max(150)
})

// Usage:
try {
  const validated = createUserSchema.parse(formData)
} catch (error) {
  console.error(error.issues)
}
```

### Rate Limiting

#### Redis-Based Rate Limiting

```python
# From backend/utils/rate_limiter.py
from redis import Redis
import time

class RateLimiter:
    def __init__(self, redis_client: Redis):
        self.redis = redis_client

    def is_allowed(self, key: str, limit: int, window: int) -> bool:
        """
        Check if request is within rate limit.

        Args:
            key: User ID or IP address
            limit: Max requests allowed
            window: Time window in seconds
        """
        now = int(time.time())
        window_start = now - window

        # Count requests in window
        count = self.redis.zcount(key, window_start, now)

        if count < limit:
            self.redis.zadd(key, {str(now): now})
            self.redis.expire(key, window + 1)
            return True
        return False

# Configuration by endpoint:
# Public endpoints: 1000 requests/hour
# Authenticated endpoints: 100 requests/minute
# Admin endpoints: 10 requests/minute
# File upload: 5 requests/hour
```

#### FastAPI Rate Limiting Middleware

```python
# From backend/middleware/rate_limit.py
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Get user ID or IP
        user_id = request.user.id if hasattr(request, 'user') else request.client.host

        # Check rate limit
        if not rate_limiter.is_allowed(user_id, limit=100, window=60):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": "60"}
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = "100"
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response
```

### CORS Configuration

```python
# From backend/main.py
from fastapi.middleware.cors import CORSMiddleware

allowed_origins = [
    "https://yourdomain.com",
    "https://www.yourdomain.com",
]

if DEBUG:
    allowed_origins.extend([
        "http://localhost:3000",
        "http://localhost:8000",
    ])

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    expose_headers=["X-Total-Count"],
)
```

### API Key Security

```python
# From backend/utils/api_key.py
import secrets
from datetime import datetime, timedelta

class APIKeyManager:
    def generate_key(self, user_id: int) -> str:
        """Generate secure API key"""
        key = secrets.token_urlsafe(32)  # 256 bits

        # Store hashed version
        hashed = bcrypt.hashpw(key.encode(), bcrypt.gensalt())

        db.create(
            APIKey,
            user_id=user_id,
            key_hash=hashed,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=90)
        )

        return key  # Return to user once

    def verify_key(self, api_key: str) -> Optional[User]:
        """Verify API key and return user"""
        api_key_record = db.query(APIKey).filter(
            APIKey.key_hash == bcrypt.hashpw(api_key.encode(), api_key_record.key_salt)
        ).first()

        if not api_key_record or api_key_record.is_expired:
            return None

        return api_key_record.user

# Usage in API:
@router.get("/api/data")
async def get_data(
    x_api_key: str = Header(...),
    user = Depends(verify_api_key)
):
    return {"data": ...}
```

---

## Database Security

### SQL Injection Prevention

#### Using Parameterized Queries

```python
# ✅ CORRECT - Parameterized query
user = db.query(User).filter(User.email == email).first()

# ❌ WRONG - SQL injection vulnerability
user = db.query(User).filter(f"email = '{email}'").first()

# ✅ CORRECT - Raw SQL with parameters
result = db.execute(
    text("SELECT * FROM users WHERE email = :email"),
    {"email": email}
).fetchone()
```

#### ORM Safety

```python
# From backend/models/*.py
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, String, Integer

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True)
    password_hash = Column(String)  # Never store plain passwords

# ORM automatically parameterizes queries
user = session.query(User).filter_by(email=user_input).first()
# Equivalent to: SELECT * FROM users WHERE email = %s
```

### Sensitive Data Handling

#### Password Hashing

```python
# From backend/utils/auth.py
import bcrypt
from typing import Tuple

class PasswordManager:
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password with bcrypt (cost factor: 12)"""
        salt = bcrypt.gensalt(rounds=12)
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

    @staticmethod
    def verify_password(password: str, hash: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hash.encode('utf-8'))

# Never:
# - Store passwords in plain text
# - Use MD5 or SHA1 for passwords
# - Use salt that's too short or reused
```

#### Handling Financial Data

```python
# Financial data requires extra care
class Transaction(Base):
    __tablename__ = "transactions"

    user_id: int = Column(Integer, ForeignKey("users.id"))
    amount: Decimal = Column(Numeric(10, 2))  # Use Decimal, not Float
    symbol: str = Column(String)
    timestamp: DateTime = Column(DateTime, default=datetime.utcnow)

    # Always audit financial transactions
    audit_log = relationship("AuditLog")

# Audit every transaction change
@event.listens_for(Transaction, "after_insert")
@event.listens_for(Transaction, "after_update")
def receive_after_insert(mapper, connection, target):
    log_audit_event(
        event_type="transaction",
        user_id=target.user_id,
        changes=get_changes(target),
        timestamp=datetime.utcnow()
    )
```

---

## Infrastructure Security

### Network Security

#### Firewall Rules

```bash
# Allow only necessary ports
sudo ufw allow 22/tcp     # SSH (from specific IPs only)
sudo ufw allow 80/tcp     # HTTP (for Let's Encrypt)
sudo ufw allow 443/tcp    # HTTPS (frontend)
sudo ufw deny 5432/tcp    # Block direct database access
sudo ufw deny 6379/tcp    # Block direct Redis access
sudo ufw deny 9200/tcp    # Block direct Elasticsearch

# Enable firewall
sudo ufw enable

# Check rules
sudo ufw status numbered
```

#### SSH Hardening

```bash
# Edit /etc/ssh/sshd_config
Port 2222  # Non-standard port
PermitRootLogin no  # Never allow root SSH
PasswordAuthentication no  # Force key-based auth
PubkeyAuthentication yes
X11Forwarding no
MaxAuthTries 3
MaxSessions 10

# Reload SSH
sudo systemctl reload ssh
```

### Container Security

#### Docker Image Scanning

```bash
# Scan for vulnerabilities
docker scan backend:latest

# Use distroless images for smaller attack surface
FROM python:3.11-slim

# Run as non-root user
RUN useradd -m appuser
USER appuser

# Read-only root filesystem
RUN chmod -R 555 /app
```

#### Docker Secrets Management

```yaml
# docker-compose.prod.yml
services:
  investment_backend:
    environment:
      - JWT_SECRET_KEY_FILE=/run/secrets/jwt_secret
      - DB_PASSWORD_FILE=/run/secrets/db_password
    secrets:
      - jwt_secret
      - db_password

secrets:
  jwt_secret:
    file: /secure/jwt_secret.txt
  db_password:
    file: /secure/db_password.txt
```

### Log Security

```python
# From backend/utils/logging.py
import logging
from pythonjsonlogger import jsonlogger

class SanitizedFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        # Don't log sensitive fields
        sensitive_keys = ['password', 'token', 'secret', 'key', 'ssn', 'credit_card']

        for key in sensitive_keys:
            if key in record.msg:
                record.msg = record.msg.replace(key, '[REDACTED]')

        super().add_fields(log_record, record, message_dict)

# Usage:
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = SanitizedFormatter()
handler.setFormatter(formatter)
logger.addHandler(handler)
```

---

## Secrets Management

### Environment Variables

#### Secure Configuration

```bash
# ✅ CORRECT - Use .env (not in version control)
# .env (local - never commit)
JWT_SECRET_KEY=<256-bit-random-key>
DB_PASSWORD=<strong-password>
GDPR_ENCRYPTION_KEY=<fernet-key>

# ✅ Use environment-specific configs
# .env.development (local development)
# .env.staging (staging environment)
# .env.production (production only)

# ❌ NEVER commit secrets
# Don't add keys to code:
JWT_SECRET_KEY = "hardcoded_key"  # WRONG!

# Use environment variables instead
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
```

#### .env File Management

```bash
# 1. Create from template
cp .env.example .env.production

# 2. Edit with actual values
nano .env.production

# 3. Set restrictive permissions
chmod 600 .env.production

# 4. Never commit
echo ".env*" >> .gitignore
echo "!.env.example" >> .gitignore

# 5. Verify not committed
git check-ignore .env.production  # Should output: .env.production

# 6. For production, use secrets management:
# - AWS Secrets Manager
# - HashiCorp Vault
# - Docker Secrets
# - Kubernetes Secrets
```

### API Key Rotation

```python
# Rotate external API keys every 90 days
class APIKeyRotation:
    def check_and_rotate(self):
        old_keys = db.query(ExternalAPIKey).filter(
            ExternalAPIKey.created_at < datetime.now() - timedelta(days=90)
        ).all()

        for key in old_keys:
            # 1. Generate new key
            new_key = generate_new_api_key(key.provider)

            # 2. Test new key
            if test_api_key(key.provider, new_key):
                # 3. Disable old key
                key.is_active = False

                # 4. Log rotation
                logger.info(f"Rotated {key.provider} API key")
            else:
                logger.error(f"Failed to rotate {key.provider} API key")
```

---

## Compliance Requirements

### SEC Compliance (Investment Recommendations)

#### Recommendation Disclosure

```python
# From backend/api/recommendations.py
@router.get("/api/recommendations")
async def get_recommendations(
    user = Depends(require_auth)
):
    recommendations = db.query(Recommendation).filter_by(user_id=user.id).all()

    return {
        "recommendations": [
            {
                "symbol": rec.symbol,
                "action": rec.action,
                "price_target": rec.price_target,
                # REQUIRED DISCLOSURES:
                "disclaimer": """
                    INVESTMENT DISCLAIMER: This recommendation is generated by AI analysis
                    and should not be considered as financial advice. Past performance is
                    not indicative of future results. Investments carry risk of loss.

                    Confidence Score: 75%
                    Based on analysis of: Technical Analysis, Fundamental Analysis

                    Please consult with a financial advisor before making investment decisions.
                """,
                "confidence_score": rec.confidence,
                "analysis_factors": rec.analysis_factors,
                "generated_at": rec.created_at,
                "expiration": rec.created_at + timedelta(days=30)
            }
        ],
        "audit_trail": generate_audit_trail(user.id)
    }
```

### GDPR Compliance (Data Privacy)

#### Data Export (DSAR)

```python
# From backend/api/gdpr.py
@router.get("/api/gdpr/export")
async def export_user_data(
    user = Depends(require_auth)
):
    """
    Data Subject Access Request (DSAR) endpoint
    Returns all user data in machine-readable format
    """

    # Collect all user data
    data = {
        "user": serialize(user),
        "portfolios": [serialize(p) for p in user.portfolios],
        "transactions": [serialize(t) for t in user.transactions],
        "recommendations": [serialize(r) for r in user.recommendations],
        "watchlists": [serialize(w) for w in user.watchlists],
        "audit_logs": [serialize(l) for l in user.audit_logs],
    }

    # Create JSON file
    export = json.dumps(data, default=str)

    # Log export
    log_audit_event(
        event_type="data_export",
        user_id=user.id,
        timestamp=datetime.utcnow()
    )

    # Return downloadable file
    return Response(
        content=export,
        media_type="application/json",
        headers={
            "Content-Disposition": f"attachment; filename=gdpr_export_{user.id}.json"
        }
    )
```

#### Data Deletion (Right to be Forgotten)

```python
# From backend/api/gdpr.py
@router.delete("/api/gdpr/delete-account")
async def delete_user_account(
    confirmation: str,  # Requires "DELETE MY ACCOUNT" confirmation
    user = Depends(require_auth)
):
    """
    Delete all user data (Right to be Forgotten)
    This action is irreversible
    """

    if confirmation != "DELETE MY ACCOUNT":
        raise HTTPException(status_code=400, detail="Confirmation required")

    # Start transaction
    async with db.begin_nested():
        try:
            # 1. Create backup for compliance (7-year retention)
            backup_data = serialize_all_user_data(user.id)
            store_encrypted_backup(user.id, backup_data)

            # 2. Delete personal data
            user.email = None
            user.name = None
            user.phone = None
            user.address = None

            # 3. Anonymize transactions
            for transaction in user.transactions:
                transaction.user_id = None  # Anonymize

            # 4. Keep aggregated data (legal requirement)
            # But disassociate from user

            # 5. Delete sessions/tokens
            db.query(Session).filter_by(user_id=user.id).delete()

            # 6. Log deletion
            log_audit_event(
                event_type="account_deletion",
                user_id=user.id,
                timestamp=datetime.utcnow()
            )

            # 7. Commit changes
            await db.commit()

            return {"status": "Account deleted", "backup_available": True}

        except Exception as e:
            await db.rollback()
            logger.error(f"Account deletion failed: {e}")
            raise HTTPException(status_code=500, detail="Deletion failed")
```

---

## Incident Response

### Security Incident Procedure

#### 1. Detection & Reporting

```
If you discover a security issue:

1. DO NOT publicly disclose
2. Contact security team immediately:
   - Email: security@yourdomain.com
   - Phone: [emergency-number]
   - Slack: #security-incidents

3. Include:
   - Type of vulnerability
   - Affected component
   - Impact assessment
   - Reproduction steps
   - Suggested remediation
```

#### 2. Incident Classification

```
Severity Level:
- CRITICAL: Active exploitation, data breach, system down
  → Response time: 1 hour
  → Notify: All leads, stakeholders

- HIGH: Vulnerability allowing data access
  → Response time: 4 hours
  → Notify: Engineering team, security

- MEDIUM: Bug potentially affecting security
  → Response time: 1 business day
  → Notify: Engineering team

- LOW: Minor security issue, low risk
  → Response time: 1 week
  → Notify: Security team
```

#### 3. Incident Response Steps

```
1. ASSESS (15 minutes)
   - Determine scope
   - Identify affected systems
   - Assess data exposure risk

2. CONTAIN (1 hour)
   - Isolate affected systems
   - Revoke compromised credentials
   - Block malicious IP addresses
   - Disable affected API keys

3. INVESTIGATE (2-24 hours)
   - Review audit logs
   - Determine entry point
   - Identify duration of exposure
   - Check for lateral movement

4. REMEDIATE (24-48 hours)
   - Deploy security patch
   - Rotate all credentials
   - Update firewall rules
   - Reset affected user passwords

5. COMMUNICATE (ongoing)
   - Notify affected users
   - Prepare incident report
   - Update incident log
   - Brief stakeholders

6. REVIEW (1 week)
   - Post-incident analysis
   - Update security policies
   - Implement preventive measures
   - Train team on lessons learned
```

### Credential Compromise Response

```bash
# If API key compromised:
1. Immediately revoke key:
   DELETE FROM api_keys WHERE key_id = 'xxx'

2. Rotate in external system (Finnhub, Alpha Vantage, etc.)

3. Force password reset for user:
   UPDATE users SET password_reset_required = true WHERE id = user_id

4. Audit logs:
   SELECT * FROM audit_logs WHERE user_id = xxx AND created_at > NOW() - '7 days'

5. Check for unauthorized usage:
   SELECT * FROM api_call_logs WHERE api_key = 'xxx' AND success = true

# If database credentials compromised:
1. Change database password immediately
2. Update all application configurations
3. Restart all services
4. Review who has access to .env files
```

---

## Security Checklist

### Pre-Deployment

- [ ] All dependencies up-to-date (no CVEs)
- [ ] Security scanning passed (Bandit, Safety)
- [ ] HTTPS/TLS configured with valid certificate
- [ ] JWT secret keys generated (32+ bytes)
- [ ] Database encryption key configured
- [ ] GDPR encryption key configured
- [ ] All API keys rotated within 90 days
- [ ] Firewall rules configured (whitelist approach)
- [ ] SSH hardened (no password auth, non-standard port)
- [ ] Secrets not in version control
- [ ] Rate limiting configured
- [ ] CORS properly scoped
- [ ] Authentication required for sensitive endpoints
- [ ] Audit logging enabled
- [ ] Error messages don't leak sensitive info
- [ ] All user inputs validated
- [ ] SQL injection prevention verified

### Post-Deployment

- [ ] SSL certificate valid and auto-renewing
- [ ] All services running as non-root user
- [ ] Read-only file systems where possible
- [ ] Regular security scans scheduled
- [ ] Log aggregation and monitoring enabled
- [ ] Backup encryption enabled
- [ ] Disaster recovery tested
- [ ] Security team trained on runbooks
- [ ] Incident response plan documented
- [ ] Penetration testing scheduled
- [ ] Third-party library audits scheduled
- [ ] Password policies enforced
- [ ] MFA enabled for admin accounts
- [ ] VPN/bastion host for admin access
- [ ] Secrets rotation scheduled (quarterly)

### Ongoing Maintenance

- [ ] Security updates applied monthly
- [ ] Penetration testing conducted quarterly
- [ ] Security audit logs reviewed weekly
- [ ] Failed login attempts monitored
- [ ] API usage monitored for anomalies
- [ ] Database backup integrity verified
- [ ] Encryption keys backed up securely
- [ ] Dependency updates checked weekly
- [ ] Security training conducted annually
- [ ] Incident drills conducted quarterly

---

## Additional Resources

### Security Tools

```bash
# Dependency vulnerability scanning
pip install safety
safety check

# Code security analysis
pip install bandit
bandit -r backend/

# OWASP dependency check
docker run --rm --volume $(pwd):/src owasp/dependency-check:latest scan -f JSON -s .

# SQL injection testing
sqlmap -u "https://yourdomain.com/api/endpoint?param=value" --dbs

# HTTPS configuration testing
ssl-test https://yourdomain.com
nmap --script ssl-cert yourdomain.com

# Load testing
ab -n 1000 -c 10 https://yourdomain.com/
```

### Compliance Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [GDPR Compliance Guide](https://gdpr-info.eu/)
- [SEC Investment Advisor Rules](https://www.sec.gov/rules/)
- [CWE Top 25](https://cwe.mitre.org/top25/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)

---

## Support & Questions

For security questions or to report vulnerabilities:

```
Email: security@yourdomain.com
GPG Key: [provide if applicable]
Bug Bounty Program: [link if applicable]
```

---

*Document Version: 1.0.0*
*Last Updated: 2026-01-27*
*Maintained by: Security Team*
*Classification: Internal Use Only*
