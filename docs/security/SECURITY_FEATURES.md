# Security Features Documentation

**Investment Analysis Platform - Phase 3 Security Implementation**

## Overview

This document describes the comprehensive security features implemented in Phase 3 of the security remediation plan. All features follow industry best practices and OWASP recommendations.

## 1. CSRF Protection

### Description
Cross-Site Request Forgery (CSRF) protection using double-submit cookie pattern with HMAC signatures.

### Features
- **Token Generation**: Cryptographically secure tokens with HMAC signatures
- **Double-Submit Pattern**: Validates both cookie and header tokens
- **Configurable Exemptions**: Webhooks and public APIs can be exempted
- **Token Rotation**: Fresh tokens on every GET request
- **Expiration**: 24-hour token expiration (configurable)

### Configuration
```python
from backend.security.csrf_protection import add_csrf_protection

add_csrf_protection(
    app,
    secret_key=os.getenv("CSRF_SECRET_KEY"),
    exempt_paths=["/api/webhooks/stripe", "/api/public"]
)
```

### Protected Methods
- POST
- PUT
- DELETE
- PATCH

### Exempt Paths (Default)
- `/api/webhooks/*`
- `/api/health`
- `/health`
- `/metrics`
- `/api/auth/login`
- `/api/auth/register`

### Usage in Frontend
```javascript
// Get CSRF token from cookie or header
const csrfToken = getCookie('csrf_token');

// Include in POST request
fetch('/api/data', {
  method: 'POST',
  headers: {
    'X-CSRF-Token': csrfToken,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify(data)
});
```

### Error Response
```json
{
  "success": false,
  "error": "CSRF validation failed",
  "detail": "Missing or invalid CSRF token",
  "code": "CSRF_VALIDATION_FAILED"
}
```

**Status Code**: 403 Forbidden

---

## 2. Security Headers

### Description
Comprehensive security headers middleware implementing defense-in-depth protection against various web attacks.

### Headers Implemented

#### X-Content-Type-Options
- **Value**: `nosniff`
- **Protection**: Prevents MIME-type sniffing attacks
- **Purpose**: Forces browser to respect declared content types

#### X-Frame-Options
- **Value**: `DENY` (default) or `SAMEORIGIN`
- **Protection**: Clickjacking attacks
- **Purpose**: Prevents page from being loaded in frames

#### X-XSS-Protection
- **Value**: `1; mode=block`
- **Protection**: Legacy XSS filter (defense-in-depth)
- **Purpose**: Enables browser's built-in XSS filter

#### Strict-Transport-Security (HSTS)
- **Value**: `max-age=31536000; includeSubDomains`
- **Protection**: Forces HTTPS connections
- **Purpose**: Prevents SSL stripping attacks
- **Note**: Only sent over HTTPS connections

#### Content-Security-Policy (CSP)
**Default Policy**:
```
default-src 'self';
script-src 'self';
style-src 'self' 'unsafe-inline';
img-src 'self' data: https:;
font-src 'self' data:;
connect-src 'self';
frame-src 'none';
object-src 'none';
base-uri 'self';
form-action 'self';
frame-ancestors 'none';
upgrade-insecure-requests;
block-all-mixed-content
```

**Protection**: XSS, injection attacks, unauthorized resource loading

#### Referrer-Policy
- **Value**: `strict-origin-when-cross-origin` (default)
- **Protection**: Controls referrer information leakage
- **Purpose**: Prevents sensitive URL information leakage

#### Permissions-Policy
**Default Policy**: Denies all browser features
- camera=()
- microphone=()
- geolocation=()
- payment=()
- usb=()
- etc.

**Protection**: Prevents unauthorized access to browser features

### Configuration
```python
from backend.middleware.security_headers import add_security_headers

add_security_headers(
    app,
    csp_script_src=["'self'", "https://cdn.example.com"],
    csp_connect_src=["'self'", "https://api.example.com"],
    exclude_paths={"/metrics", "/health"}
)
```

### Custom CSP Configuration
```python
from backend.middleware.security_headers import (
    SecurityHeadersConfig,
    ContentSecurityPolicy,
    SecurityHeadersMiddleware
)

csp = ContentSecurityPolicy(
    script_src=["'self'", "https://trusted-cdn.com"],
    style_src=["'self'", "'unsafe-inline'"],
    img_src=["'self'", "data:", "https:"],
    connect_src=["'self'", "wss://websocket.example.com"],
    report_uri="https://csp-report.example.com"
)

config = SecurityHeadersConfig(
    csp=csp,
    hsts_enabled=True,
    hsts_max_age=31536000,
    hsts_include_subdomains=True,
    hsts_preload=True
)

app.add_middleware(SecurityHeadersMiddleware, config=config)
```

---

## 3. Request Size Limits

### Description
Request body size validation to prevent memory exhaustion and DoS attacks.

### Default Limits
- **JSON Payloads**: 1 MB
- **Form Data**: 1 MB
- **File Uploads**: 10 MB
- **Text Content**: 512 KB

### Features
- **Content-Type Detection**: Automatic limit selection based on content type
- **Path-Specific Limits**: Custom limits for specific endpoints
- **Human-Readable Errors**: Clear size information in error messages
- **Early Validation**: Checks Content-Length header before reading body

### Configuration
```python
from backend.middleware.request_size_limiter import add_request_size_limits

add_request_size_limits(
    app,
    json_limit_mb=1.0,
    file_upload_limit_mb=10.0,
    path_limits={
        "/api/uploads/large": 50 * 1024 * 1024  # 50 MB for specific endpoint
    }
)
```

### Path-Specific Limits
```python
from backend.middleware.request_size_limiter import RequestSizeLimits

config = RequestSizeLimits(
    json_limit=1_048_576,  # 1 MB
    file_upload_limit=10_485_760,  # 10 MB
    path_limits={
        "/api/reports/generate": 5_242_880,  # 5 MB
        "/api/uploads/bulk": 52_428_800  # 50 MB
    }
)
```

### Error Response
```json
{
  "success": false,
  "error": "Request payload too large",
  "detail": "Request body size (2.5 MB) exceeds maximum allowed size (1.0 MB)",
  "code": "PAYLOAD_TOO_LARGE",
  "max_size": "1.0 MB",
  "received_size": "2.5 MB"
}
```

**Status Code**: 413 Payload Too Large

### Exempt Paths
- `/health`
- `/metrics`

---

## Integration Example

### Complete Security Setup
```python
import os
from fastapi import FastAPI
from backend.security.csrf_protection import add_csrf_protection
from backend.middleware.security_headers import add_security_headers
from backend.middleware.request_size_limiter import add_request_size_limits

app = FastAPI()

# 1. Add security headers (first, affects all responses)
add_security_headers(
    app,
    csp_script_src=["'self'", "https://cdn.example.com"],
    csp_connect_src=["'self'", "wss://websocket.example.com"]
)

# 2. Add request size limits (before CSRF to catch oversized requests early)
add_request_size_limits(
    app,
    json_limit_mb=1.0,
    file_upload_limit_mb=10.0
)

# 3. Add CSRF protection (last, validates tokens)
add_csrf_protection(
    app,
    secret_key=os.getenv("CSRF_SECRET_KEY"),
    exempt_paths=["/api/webhooks/stripe"]
)
```

---

## Security Testing

### Test Coverage
- **CSRF Protection**: 10+ tests
- **Security Headers**: 8+ tests
- **Request Size Limits**: 6+ tests

### Running Tests
```bash
# Run all security tests
pytest backend/tests/security/test_csrf_protection.py -v
pytest backend/tests/middleware/test_security_headers.py -v
pytest backend/tests/middleware/test_request_size_limiter.py -v

# Run with coverage
pytest backend/tests/security/ backend/tests/middleware/ --cov=backend/security --cov=backend/middleware --cov-report=html
```

### Security Validation
```bash
# CSRF validation
curl -X POST http://localhost:8000/api/test
# Expected: 403 CSRF validation failed

# Security headers validation
curl -I http://localhost:8000/api/test
# Expected: X-Content-Type-Options, X-Frame-Options, CSP, etc.

# Request size validation
curl -X POST http://localhost:8000/api/test \
  -H "Content-Type: application/json" \
  -H "Content-Length: 10000000" \
  -d '{}'
# Expected: 413 Payload Too Large
```

---

## Environment Variables

### Required
```bash
# CSRF secret key (must be 32+ characters)
CSRF_SECRET_KEY=your-secure-random-key-here
```

### Optional
```bash
# HSTS configuration
HSTS_MAX_AGE=31536000
HSTS_INCLUDE_SUBDOMAINS=true
HSTS_PRELOAD=false

# Request size limits (in MB)
JSON_SIZE_LIMIT_MB=1.0
FILE_UPLOAD_LIMIT_MB=10.0
```

---

## Performance Impact

### Overhead Analysis
- **CSRF Token Generation**: ~0.1ms per request
- **CSRF Token Validation**: ~0.05ms per request
- **Security Headers**: ~0.01ms per response
- **Request Size Check**: ~0.02ms per request (Content-Length only)

**Total Overhead**: < 1ms per request

### Caching Recommendations
- CSRF tokens: 24-hour expiration
- Security headers: Static, no caching needed
- Size limits: Pre-computed, no overhead

---

## Compliance

### Standards Met
- **OWASP Top 10 2021**:
  - A01:2021 - Broken Access Control (CSRF)
  - A03:2021 - Injection (CSP, Input Validation)
  - A05:2021 - Security Misconfiguration (Security Headers)
  - A06:2021 - Vulnerable Components (DoS Prevention)

- **CWE Coverage**:
  - CWE-352: CSRF
  - CWE-79: XSS (CSP)
  - CWE-1021: Clickjacking (X-Frame-Options)
  - CWE-400: DoS (Request Size Limits)

- **NIST 800-53**:
  - SC-5: Denial of Service Protection
  - SC-8: Transmission Confidentiality (HSTS)
  - SI-10: Information Input Validation

---

## Troubleshooting

### CSRF Token Issues
**Problem**: CSRF validation fails for legitimate requests

**Solution**: Ensure frontend includes token in both cookie and header
```javascript
const token = getCookie('csrf_token');
headers['X-CSRF-Token'] = token;
```

### CSP Violations
**Problem**: Resources blocked by CSP

**Solution**: Add trusted origins to CSP configuration
```python
csp_script_src=["'self'", "https://trusted-cdn.com"]
```

### Request Size Rejections
**Problem**: Legitimate large uploads rejected

**Solution**: Configure path-specific limits
```python
path_limits={
    "/api/uploads/reports": 50 * 1024 * 1024  # 50 MB
}
```

---

## Future Enhancements

### Planned Features
1. **Rate Limiting Integration**: Combine with existing rate limiter
2. **CSP Reporting**: Implement CSP violation reporting endpoint
3. **Token Refresh**: Automatic CSRF token refresh before expiration
4. **Adaptive Limits**: Dynamic size limits based on user tier
5. **Security Monitoring**: Real-time security event dashboard

---

## References

- [OWASP CSRF Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Cross-Site_Request_Forgery_Prevention_Cheat_Sheet.html)
- [OWASP Secure Headers Project](https://owasp.org/www-project-secure-headers/)
- [Content Security Policy Reference](https://content-security-policy.com/)
- [MDN Security Headers](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers#security)

---

**Last Updated**: 2026-01-27
**Version**: 1.0.0
**Authors**: Security Team
