# Security Documentation

## Overview

This directory contains reference documentation for the security implementation of the Investment Analysis Platform.

**Status**: Production
**Last Updated**: 2026-03-04
**Classification**: Internal Use Only

---

## Documentation Files

| File | Purpose | Audience |
|------|---------|----------|
| [README.md](./README.md) | This file - navigation guide | All |
| [SECURITY_FEATURES.md](./SECURITY_FEATURES.md) | CSRF, security headers, request size limits | All developers |
| [SECURITY_IMPLEMENTATION_SUMMARY.md](./SECURITY_IMPLEMENTATION_SUMMARY.md) | Complete security architecture overview | Security team, architects |
| [ROW_LOCKING.md](./ROW_LOCKING.md) | Row-level locking patterns (optimistic + pessimistic) | Backend developers |

---

## Security Architecture Summary

The platform implements defense-in-depth with the following layers:

- **Authentication**: OAuth2 + JWT (RS256), bcrypt password hashing, refresh token rotation
- **Authorization**: Role-based access control (Admin, Analyst, Trader, Viewer, API_User, Guest)
- **Transport**: HTTPS enforced via HSTS, TLS 1.2+
- **Request Protection**: CSRF tokens (double-submit cookie, HMAC-signed), request size limits
- **Response Headers**: CSP, X-Frame-Options, X-Content-Type-Options, Referrer-Policy, Permissions-Policy
- **Injection Prevention**: SQLAlchemy ORM with parameterized queries, Pydantic input validation
- **Encryption**: Fernet symmetric encryption for sensitive fields, RSA asymmetric for key exchange
- **Audit Logging**: 7-year retention for SEC compliance, structured JSON format
- **Rate Limiting**: Redis-backed, per-IP and per-user with sliding window algorithm

---

## Quick Reference

### Environment Variables Required

```bash
# Core Security
SECRET_KEY=                    # 64+ char secret
JWT_SECRET_KEY=               # 64+ char secret (RS256 RSA private key path)
FERNET_KEY=                   # Fernet encryption key
CSRF_SECRET_KEY=              # CSRF token secret (32+ chars)

# Database
DB_PASSWORD=                  # Secure password
DATABASE_URL=                 # PostgreSQL connection string

# Redis
REDIS_PASSWORD=               # Redis password
REDIS_URL=                    # Redis connection string

# Compliance
AUDIT_LOG_RETENTION_DAYS=2555  # 7 years for SEC
GDPR_ENCRYPTION_KEY=           # GDPR data encryption key
```

### Security Scripts

Located in `/scripts/security/`:

| Script | Purpose |
|--------|---------|
| `generate_secrets.sh` | Generate new cryptographically secure secrets |

---

## Security Best Practices

### Secret Management
1. Never commit secrets to git
2. Always use environment variables
3. Rotate secrets every 90 days
4. Store in password manager or vault
5. Monitor for secret exposure (GitHub secret scanning)

### Code Security
1. Input validation on all user data (Pydantic schemas)
2. Output encoding for all rendered content (React auto-escaping)
3. Parameterized queries for all SQL (SQLAlchemy ORM)
4. HTTPS everywhere in production (HSTS enforced)
5. Security headers on all responses (SecurityHeadersMiddleware)

### Container Security
1. Pin all images to digests
2. Scan images before deployment (Trivy)
3. Run as non-root users
4. Drop all capabilities by default
5. Use read-only filesystems where possible

### Operational Security
1. Regular security audits (quarterly)
2. Penetration testing (annually)
3. Security training for all developers
4. Incident response plan documented
5. Regular backups and disaster recovery tests

---

## Compliance Status

| Standard | Status | Notes |
|----------|--------|-------|
| OWASP Top 10 (2021) | Covered | All 10 categories addressed |
| SEC 2025 | Covered | 7-year audit log retention, disclosure controls |
| GDPR | Covered | Data export, deletion, anonymization endpoints |
| CWE Top 25 | Covered | XSS, SQLi, CSRF, file upload, null dereference |
| NIST CSF | Covered | Identify, Protect, Detect, Respond, Recover |

---

## Monitoring & Alerts

### Security Events
- CSP violations - Slack #security-alerts
- Failed authentication (>10 in 5 min) - PagerDuty
- Rate limit violations (>20 in 5 min) - Grafana dashboard
- CSRF token failures (>5 in 5 min) - Alertmanager

### Health Checks
- Service health - Prometheus + Grafana
- API endpoints - Uptime monitoring
- Database connections - Grafana
- Redis connections - Grafana
- Loki log aggregation - active

---

## External Resources

- [OWASP Top 10 (2021)](https://owasp.org/Top10/)
- [OWASP CSP Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Content_Security_Policy_Cheat_Sheet.html)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [CWE Top 25](https://cwe.mitre.org/top25/)
- [SEC 2025 Guidelines](https://www.sec.gov/)

---

**Document Owner**: Security Team
**Last Updated**: 2026-03-04
**Next Review**: 2026-06-04 (Quarterly)

**CONFIDENTIAL - INTERNAL USE ONLY**
