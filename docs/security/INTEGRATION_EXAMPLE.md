# Phase 3 Security Middleware Integration Example

## Complete Integration

### Step 1: Update main.py

```python
"""
main.py - Updated with Phase 3 Security Middleware
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import Phase 3 security integration
from backend.api.security_integration import (
    register_security_middleware,
    validate_security_configuration
)

# Import existing routers
from backend.api.routers import (
    stocks, analysis, recommendations, portfolio,
    auth, health, websocket, admin
)

# Validate security configuration before startup
validate_security_configuration()

# Create FastAPI app
app = FastAPI(
    title="Investment Analysis Platform",
    description="AI-Powered Stock Analysis with Enhanced Security",
    version="2.0.0"
)

# Add CORS middleware (before security middleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Phase 3 security middleware
register_security_middleware(
    app,
    csrf_enabled=True,
    security_headers_enabled=True,
    request_size_limits_enabled=True,
    csrf_secret_key=os.getenv("CSRF_SECRET_KEY"),
    csrf_exempt_paths=[
        "/api/webhooks/stripe",
        "/api/webhooks/github",
        "/api/public/data"
    ],
    csp_script_src=[
        "'self'",
        "https://cdn.jsdelivr.net",
        "https://cdnjs.cloudflare.com"
    ],
    csp_connect_src=[
        "'self'",
        "wss://websocket.example.com",
        "https://api.example.com"
    ],
    json_limit_mb=1.0,
    file_upload_limit_mb=10.0,
    path_size_limits={
        "/api/reports/generate": 5 * 1024 * 1024,  # 5 MB
        "/api/uploads/bulk": 50 * 1024 * 1024  # 50 MB
    }
)

# Include routers
app.include_router(health.router, prefix="/api/health", tags=["health"])
app.include_router(auth.router, prefix="/api/auth", tags=["authentication"])
app.include_router(stocks.router, prefix="/api/stocks", tags=["stocks"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(recommendations.router, prefix="/api/recommendations", tags=["recommendations"])
app.include_router(portfolio.router, prefix="/api/portfolio", tags=["portfolio"])
app.include_router(websocket.router, prefix="/api/ws", tags=["websocket"])
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

### Step 2: Environment Variables

Create or update `.env` file:

```bash
# Phase 3 Security Configuration

# CSRF Protection (REQUIRED)
CSRF_SECRET_KEY=your-secure-32-plus-character-secret-key-here-change-this-in-production

# CSRF Configuration
CSRF_ENABLED=true
CSRF_TOKEN_EXPIRY_HOURS=24

# Security Headers Configuration
SECURITY_HEADERS_ENABLED=true
HSTS_MAX_AGE=31536000
HSTS_INCLUDE_SUBDOMAINS=true
HSTS_PRELOAD=false

# Request Size Limits
REQUEST_SIZE_LIMITS_ENABLED=true
JSON_SIZE_LIMIT_MB=1.0
FILE_UPLOAD_LIMIT_MB=10.0
FORM_SIZE_LIMIT_MB=1.0

# Development vs Production
ENVIRONMENT=production  # or development
DEBUG=false
```

---

### Step 3: Generate CSRF Secret Key

```bash
# Generate a secure CSRF secret key
python -c "import secrets; print(secrets.token_hex(32))"

# Example output:
# a7f3b2c1d8e4f9a6b3c7d2e8f4a9b6c3d7e2f8a4b9c6d3e7f2a8b4c9d6e3f7a2
```

Add this to your `.env` file:
```bash
CSRF_SECRET_KEY=a7f3b2c1d8e4f9a6b3c7d2e8f4a9b6c3d7e2f8a4b9c6d3e7f2a8b4c9d6e3f7a2
```

---

### Step 4: Frontend Integration

#### React Example

```javascript
// utils/csrf.js
export function getCookie(name) {
  const value = `; ${document.cookie}`;
  const parts = value.split(`; ${name}=`);
  if (parts.length === 2) return parts.pop().split(';').shift();
  return null;
}

export function getCSRFToken() {
  return getCookie('csrf_token');
}

// api/client.js
import axios from 'axios';
import { getCSRFToken } from '../utils/csrf';

const apiClient = axios.create({
  baseURL: 'http://localhost:8000',
  withCredentials: true,  // Important for CSRF cookies
});

// Add CSRF token to all state-changing requests
apiClient.interceptors.request.use((config) => {
  if (['post', 'put', 'delete', 'patch'].includes(config.method)) {
    const csrfToken = getCSRFToken();
    if (csrfToken) {
      config.headers['X-CSRF-Token'] = csrfToken;
    }
  }
  return config;
});

// Handle CSRF errors
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 403 &&
        error.response?.data?.code === 'CSRF_VALIDATION_FAILED') {
      // Refresh page to get new CSRF token
      console.error('CSRF token expired or invalid');
      // Optionally: redirect to login or refresh token
    }
    return Promise.reject(error);
  }
);

export default apiClient;

// Usage
import apiClient from './api/client';

// POST request with automatic CSRF token
async function createPost(data) {
  const response = await apiClient.post('/api/posts', data);
  return response.data;
}

// File upload with size limit awareness
async function uploadFile(file) {
  if (file.size > 10 * 1024 * 1024) {
    throw new Error('File too large. Maximum size: 10 MB');
  }

  const formData = new FormData();
  formData.append('file', file);

  const response = await apiClient.post('/api/upload', formData);
  return response.data;
}
```

#### Vanilla JavaScript Example

```javascript
// Get CSRF token from initial page load
async function getCSRFToken() {
  const response = await fetch('/api/health');
  return response.headers.get('X-CSRF-Token');
}

// Make protected POST request
async function createResource(data) {
  const csrfToken = await getCSRFToken();

  const response = await fetch('/api/resources', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-CSRF-Token': csrfToken
    },
    credentials: 'include',  // Important for cookies
    body: JSON.stringify(data)
  });

  if (response.status === 403) {
    const error = await response.json();
    if (error.code === 'CSRF_VALIDATION_FAILED') {
      console.error('CSRF validation failed');
      // Handle token refresh
    }
  }

  return response.json();
}
```

---

### Step 5: Testing the Integration

#### Test CSRF Protection

```bash
# Should fail (no CSRF token)
curl -X POST http://localhost:8000/api/test \
  -H "Content-Type: application/json" \
  -d '{"test": "data"}'

# Expected: 403 CSRF validation failed

# Get CSRF token first
curl -c cookies.txt http://localhost:8000/api/health

# Extract token from cookies and use it
TOKEN=$(grep csrf_token cookies.txt | awk '{print $7}')

curl -X POST http://localhost:8000/api/test \
  -b cookies.txt \
  -H "X-CSRF-Token: $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"test": "data"}'

# Expected: Success
```

#### Test Security Headers

```bash
# Check security headers
curl -I http://localhost:8000/api/test

# Expected headers:
# X-Content-Type-Options: nosniff
# X-Frame-Options: DENY
# X-XSS-Protection: 1; mode=block
# Content-Security-Policy: default-src 'self'; ...
# Referrer-Policy: strict-origin-when-cross-origin
# Permissions-Policy: camera=(), microphone=(), ...
```

#### Test Request Size Limits

```bash
# Test oversized JSON payload
curl -X POST http://localhost:8000/api/test \
  -H "Content-Type: application/json" \
  -H "Content-Length: 10000000" \
  -d '{}'

# Expected: 413 Payload Too Large
# Response:
# {
#   "success": false,
#   "error": "Request payload too large",
#   "detail": "Request body size (9.5 MB) exceeds maximum allowed size (1.0 MB)",
#   "code": "PAYLOAD_TOO_LARGE",
#   "max_size": "1.0 MB"
# }
```

---

### Step 6: Monitoring

#### Log Analysis

```bash
# Monitor CSRF violations
tail -f logs/app.log | grep "CSRF validation failed"

# Monitor size limit violations
tail -f logs/app.log | grep "Request size limit exceeded"
```

#### Prometheus Metrics (Optional)

```python
# Add to security_integration.py
from prometheus_client import Counter, Histogram

csrf_violations = Counter(
    'csrf_violations_total',
    'Total CSRF validation failures',
    ['path', 'method']
)

size_limit_violations = Counter(
    'size_limit_violations_total',
    'Total request size limit violations',
    ['path', 'content_type']
)

request_size_histogram = Histogram(
    'request_size_bytes',
    'Request size distribution',
    ['path', 'content_type']
)
```

---

### Step 7: Troubleshooting

#### CSRF Issues

**Problem**: "CSRF validation failed" for legitimate requests

**Solutions**:
1. Ensure cookies are enabled in browser
2. Check `withCredentials: true` in fetch/axios
3. Verify CSRF token is included in header
4. Check cookie `SameSite` attribute

#### CSP Violations

**Problem**: Resources blocked by CSP

**Solutions**:
1. Check browser console for CSP violations
2. Add trusted domains to CSP configuration
3. Use CSP report-only mode for testing:
   ```python
   csp.report_only = True
   ```

#### Size Limit Issues

**Problem**: Legitimate large uploads rejected

**Solutions**:
1. Increase limit for specific endpoints:
   ```python
   path_size_limits={
       "/api/uploads/reports": 50 * 1024 * 1024
   }
   ```
2. Check Content-Length header is set correctly
3. Verify content-type matches expected limit

---

## Production Checklist

- [ ] CSRF secret key set in production environment
- [ ] Secret key is 32+ characters
- [ ] Secret key is NOT committed to version control
- [ ] HSTS enabled for HTTPS sites
- [ ] CSP configured for production domains
- [ ] Size limits appropriate for use case
- [ ] Webhook paths added to CSRF exemptions
- [ ] Frontend updated to include CSRF tokens
- [ ] Monitoring configured for security events
- [ ] Error handling tested
- [ ] All tests passing (60/60)

---

## Quick Reference

### Middleware Order (Important!)
1. CORS Middleware (first)
2. Security Headers Middleware
3. Request Size Limiter Middleware
4. CSRF Protection Middleware (last)

### Default Limits
- JSON: 1 MB
- Files: 10 MB
- Forms: 1 MB
- Text: 512 KB

### Protected Methods
- POST, PUT, DELETE, PATCH (CSRF required)
- GET, HEAD, OPTIONS (no CSRF)

### Exempt Paths
- `/api/webhooks/*`
- `/api/health`
- `/health`
- `/metrics`
- `/api/auth/login`
- `/api/auth/register`

---

**Last Updated**: 2026-01-27
**Version**: 1.0.0
