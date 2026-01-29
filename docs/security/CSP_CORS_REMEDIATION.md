# CSP & CORS SECURITY REMEDIATION - PHASE 1

## Priority: HIGH
**Timeline**: Week 1-2

---

## VULNERABILITIES IDENTIFIED

### 1. Content Security Policy (CSP) Issues

**Problem**: `unsafe-inline` and `unsafe-eval` present in 6 files
- Allows arbitrary script execution
- XSS vulnerability vector
- Violates security best practices

**Affected Files**:
```
backend/security/security_config.py:87-88
backend/security/security_headers.py:114-115, 496-497, 516-517
backend/security/injection_prevention.py:670-671
backend/security/data_encryption.py:776-777
```

### 2. CORS Configuration Issues

**Problem**: Overly permissive CORS in production
- Allows `allow_credentials=True` with wildcards
- Permits all methods and headers in development
- No origin validation in some paths

---

## REMEDIATION STRATEGY

### A. Implement Nonce-Based CSP

Replace `unsafe-inline` with cryptographic nonces.

#### Implementation Plan:
1. Generate unique nonce per request
2. Add nonce to all inline scripts/styles
3. Update CSP header with nonce
4. Remove `unsafe-inline` and `unsafe-eval`

### B. Strict CORS Configuration

Implement environment-specific CORS policies.

---

## IMPLEMENTATION PATCHES

### Patch 1: backend/security/security_config.py

**Current (INSECURE)**:
```python
"Content-Security-Policy": (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "  # ❌ INSECURE
    "style-src 'self' 'unsafe-inline'; "  # ❌ INSECURE
    "img-src 'self' data: https:; "
    "font-src 'self' data:; "
    "connect-src 'self' ws: wss:"
),
```

**Replacement (SECURE)**:
```python
# In middleware, generate nonce per request:
import secrets

def get_csp_header(nonce: str) -> str:
    """Generate CSP header with nonce"""
    return (
        "default-src 'self'; "
        f"script-src 'self' 'nonce-{nonce}'; "  # ✅ SECURE
        f"style-src 'self' 'nonce-{nonce}'; "   # ✅ SECURE
        "img-src 'self' data: https:; "
        "font-src 'self' data:; "
        "connect-src 'self' ws: wss:; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "form-action 'self'; "
        "frame-ancestors 'none'; "
        "upgrade-insecure-requests"
    )

# In request handler:
@app.middleware("http")
async def csp_nonce_middleware(request: Request, call_next):
    nonce = secrets.token_urlsafe(16)
    request.state.csp_nonce = nonce

    response = await call_next(request)
    response.headers["Content-Security-Policy"] = get_csp_header(nonce)

    return response
```

**Frontend Changes Required**:
```html
<!-- OLD (insecure) -->
<script>
    console.log('Hello');
</script>

<!-- NEW (secure) -->
<script nonce="{{ csp_nonce }}">
    console.log('Hello');
</script>
```

---

### Patch 2: backend/security/security_headers.py

**Lines 114-115** (Change):
```python
# BEFORE
CSPDirective.SCRIPT_SRC: ["'self'", "'unsafe-inline'", "'unsafe-eval'"],
CSPDirective.STYLE_SRC: ["'self'", "'unsafe-inline'"],

# AFTER
CSPDirective.SCRIPT_SRC: ["'self'"],  # Nonces added dynamically
CSPDirective.STYLE_SRC: ["'self'"],   # Nonces added dynamically
```

**Lines 496-497** (Development Config - Still needs improvement):
```python
# BEFORE
CSPDirective.SCRIPT_SRC: ["'self'", "'unsafe-inline'", "'unsafe-eval'", "localhost:*"],
CSPDirective.STYLE_SRC: ["'self'", "'unsafe-inline'", "localhost:*"],

# AFTER (Still relaxed for dev, but safer)
CSPDirective.SCRIPT_SRC: ["'self'", "'unsafe-eval'", "localhost:*"],  # Keep unsafe-eval for dev tools ONLY
CSPDirective.STYLE_SRC: ["'self'", "localhost:*"],  # Remove unsafe-inline even in dev
```

**Lines 516-517** (Production Config):
```python
# BEFORE
CSPDirective.SCRIPT_SRC: ["'self'", "'unsafe-inline'"],  # Remove unsafe-eval in production
CSPDirective.STYLE_SRC: ["'self'", "'unsafe-inline'"],

# AFTER
CSPDirective.SCRIPT_SRC: ["'self'"],  # ✅ Strict
CSPDirective.STYLE_SRC: ["'self'"],   # ✅ Strict
```

---

### Patch 3: backend/security/injection_prevention.py

**Lines 670-671**:
```python
# BEFORE
"script-src 'self' 'unsafe-inline'; "
"style-src 'self' 'unsafe-inline'; "

# AFTER
"script-src 'self'; "
"style-src 'self'; "
```

---

### Patch 4: backend/security/data_encryption.py

**Lines 776-777**:
```python
# BEFORE
"script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
"style-src 'self' 'unsafe-inline'; "

# AFTER
"script-src 'self'; "
"style-src 'self'; "
```

---

### Patch 5: CORS Configuration (backend/api/main.py)

**Current (Lines 149-155)**:
```python
# Fallback to basic CORS if comprehensive security fails
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],  # ❌ TOO PERMISSIVE
    allow_headers=["*"],  # ❌ TOO PERMISSIVE
)
```

**Replacement (SECURE)**:
```python
# Strict CORS configuration
import os

ENVIRONMENT = os.getenv("ENVIRONMENT", "production").lower()

if ENVIRONMENT == "production":
    allowed_origins = [
        "https://yourdomain.com",
        "https://app.yourdomain.com",
        "https://api.yourdomain.com"
    ]
    allowed_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    allowed_headers = [
        "Accept",
        "Authorization",
        "Content-Type",
        "X-Requested-With",
        "X-CSRF-Token"
    ]
else:
    # Development - still restricted
    allowed_origins = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000"
    ]
    allowed_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
    allowed_headers = [
        "Accept",
        "Authorization",
        "Content-Type",
        "X-Requested-With",
        "X-CSRF-Token",
        "X-Debug-Mode"  # Dev-only header
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # ✅ Explicit list
    allow_credentials=True,
    allow_methods=allowed_methods,  # ✅ Explicit list
    allow_headers=allowed_headers,  # ✅ Explicit list
    expose_headers=["X-RateLimit-Remaining", "X-RateLimit-Reset"],
    max_age=3600
)
```

---

### Patch 6: backend/security/security_config.py CORS Settings

**Lines 49-68**:
```python
# BEFORE
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "https://investment-analysis.com",
    "https://api.investment-analysis.com"
]

if settings.ENVIRONMENT == "development":
    ALLOWED_ORIGINS.extend([
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000"
    ])

# AFTER
def get_allowed_origins() -> List[str]:
    """Get CORS allowed origins based on environment"""
    env = settings.ENVIRONMENT.lower()

    if env == "production":
        # Production - HTTPS only
        return [
            "https://investment-analysis.com",
            "https://app.investment-analysis.com",
            "https://api.investment-analysis.com"
        ]
    elif env == "staging":
        # Staging - HTTPS preferred
        return [
            "https://staging.investment-analysis.com",
            "https://staging-api.investment-analysis.com"
        ]
    else:
        # Development - localhost only
        return [
            "http://localhost:3000",
            "http://localhost:8000",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8000"
        ]

ALLOWED_ORIGINS = get_allowed_origins()
```

---

## NONCE MIDDLEWARE IMPLEMENTATION

Create new file: `backend/security/csp_nonce.py`

```python
"""
CSP Nonce Middleware for Secure Inline Scripts/Styles
"""
import secrets
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable

class CSPNonceMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add CSP nonce to each request.
    Nonce is cryptographically random and unique per request.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        # Generate cryptographically secure nonce
        nonce = secrets.token_urlsafe(16)

        # Store nonce in request state for template access
        request.state.csp_nonce = nonce

        # Process request
        response = await call_next(request)

        # Add CSP header with nonce
        csp_policy = self._build_csp_policy(nonce)
        response.headers["Content-Security-Policy"] = csp_policy

        return response

    def _build_csp_policy(self, nonce: str) -> str:
        """Build CSP policy with nonce"""
        directives = [
            "default-src 'self'",
            f"script-src 'self' 'nonce-{nonce}'",
            f"style-src 'self' 'nonce-{nonce}'",
            "img-src 'self' data: https:",
            "font-src 'self' data:",
            "connect-src 'self' wss: ws:",
            "media-src 'self'",
            "object-src 'none'",
            "frame-src 'none'",
            "frame-ancestors 'none'",
            "form-action 'self'",
            "base-uri 'self'",
            "upgrade-insecure-requests"
        ]

        return "; ".join(directives)
```

**Add to main.py**:
```python
from backend.security.csp_nonce import CSPNonceMiddleware

# Add after other middleware
app.add_middleware(CSPNonceMiddleware)
```

---

## TESTING CHECKLIST

### CSP Testing
- [ ] Verify nonce appears in HTML source
- [ ] Inline scripts execute with nonce
- [ ] Inline styles apply with nonce
- [ ] External scripts load from 'self'
- [ ] CSP violations logged (check browser console)
- [ ] No `unsafe-inline` or `unsafe-eval` in headers

### CORS Testing
- [ ] Production allows only HTTPS origins
- [ ] Development allows localhost origins
- [ ] Credentials work with allowed origins
- [ ] Blocked origins receive 403
- [ ] Preflight OPTIONS requests succeed
- [ ] No wildcard (*) in production

### Browser Compatibility
- [ ] Test in Chrome/Chromium
- [ ] Test in Firefox
- [ ] Test in Safari
- [ ] Test in Edge
- [ ] Check CSP reports in console

---

## MIGRATION TIMELINE

### Week 1
- **Day 1-2**: Implement CSP nonce middleware
- **Day 3-4**: Update all inline scripts/styles with nonces
- **Day 5**: Test in staging environment

### Week 2
- **Day 1-2**: Implement strict CORS
- **Day 3**: Test all API endpoints
- **Day 4**: Deploy to production (low-traffic window)
- **Day 5**: Monitor and fix issues

---

## ROLLBACK PLAN

If issues occur:
1. **Temporary Fix**: Set CSP to `report-only` mode
2. **Emergency Rollback**: Revert to previous CSP/CORS config
3. **Debug**: Check browser console for CSP violations
4. **Fix**: Update affected inline scripts/styles
5. **Retry**: Gradual rollout with monitoring

---

## MONITORING

### CSP Violation Reporting

Add CSP report endpoint:
```python
@app.post("/api/csp-report")
async def csp_violation_report(request: Request):
    """Receive CSP violation reports"""
    body = await request.json()
    logger.warning(f"CSP Violation: {body}")
    # Store in database for analysis
    return {"status": "reported"}
```

Update CSP header:
```python
f"report-uri /api/csp-report; report-to csp-endpoint"
```

---

**Document Version**: 1.0
**Created**: 2026-01-27
**Status**: READY FOR IMPLEMENTATION
