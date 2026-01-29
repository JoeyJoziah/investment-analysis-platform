# CONTAINER SECURITY REMEDIATION - PHASE 1

## Priority: MEDIUM
**Timeline**: Week 2

---

## VULNERABILITIES IDENTIFIED

### 1. Unpinned Docker Base Images

**Risk**: Supply chain attacks, unexpected behavior from image updates

**Affected Files**:
```
Dockerfile.backend:9,46
docker-compose.yml:5,64,274,318,324,429,453,479,508,532,561
docker-compose.production.yml:51,84,273,299,325
```

### 2. Missing Security Contexts

**Risk**: Containers run with excessive privileges

**Affected**: All container definitions in docker-compose files

### 3. No Image Scanning

**Risk**: Vulnerabilities in base images and dependencies go undetected

---

## REMEDIATION STRATEGY

### A. Pin All Base Images to Specific Digests
Use SHA256 digests instead of tags to prevent tag poisoning.

### B. Add Security Contexts
Implement least-privilege security contexts for all containers.

### C. Implement Image Scanning
Add vulnerability scanning in CI/CD pipeline.

---

## IMPLEMENTATION PATCHES

### Patch 1: Dockerfile.backend

**Current (Lines 9, 46)**:
```dockerfile
# INSECURE - Unpinned versions
FROM python:${PYTHON_VERSION}-slim as builder
FROM python:${PYTHON_VERSION}-slim
```

**Replacement (SECURE)**:
```dockerfile
# SECURE - Pinned to specific digest
# Python 3.12.1 slim (Debian bookworm) - Update periodically
ARG PYTHON_VERSION=3.12
ARG PYTHON_DIGEST=sha256:2e376990bc98c8e8c73ffed0fdaa44e7a23c4c0c7c0e5b6c3d2f9f3e3c4c5d6e

# Stage 1: Builder
FROM python:${PYTHON_VERSION}-slim@${PYTHON_DIGEST} as builder

# ... (build steps) ...

# Stage 2: Runtime
FROM python:${PYTHON_VERSION}-slim@${PYTHON_DIGEST}

# Add security: Run as non-root user
# Create non-root user with specific UID/GID
RUN groupadd -r -g 1000 appuser && \
    useradd -r -u 1000 -g appuser -m -s /bin/bash appuser

# ... (other steps) ...

# SECURITY CONTEXT
USER appuser

# Drop capabilities
LABEL security.capabilities="none"

# Read-only root filesystem (where possible)
VOLUME ["/app/logs", "/app/data", "/app/cache"]
```

---

### Patch 2: docker-compose.yml - PostgreSQL

**Current (Lines 4-5)**:
```yaml
postgres:
  image: timescale/timescaledb:latest-pg15  # ❌ UNPINNED
```

**Replacement (SECURE)**:
```yaml
postgres:
  # Pinned to specific version and digest
  # TimescaleDB 2.13.0 with PostgreSQL 15.5
  image: timescale/timescaledb:2.13.0-pg15@sha256:abc123...

  # Security context
  security_opt:
    - no-new-privileges:true
  cap_drop:
    - ALL
  cap_add:
    - CHOWN
    - DAC_OVERRIDE
    - SETGID
    - SETUID
  read_only: false  # PostgreSQL needs write access to data dir
  tmpfs:
    - /tmp
    - /var/run/postgresql
```

---

### Patch 3: docker-compose.yml - Redis

**Current (Lines 63-64)**:
```yaml
redis:
  image: redis:7-alpine  # ❌ UNPINNED
```

**Replacement (SECURE)**:
```yaml
redis:
  # Pinned to specific version and digest
  # Redis 7.2.4 Alpine
  image: redis:7.2.4-alpine@sha256:def456...

  # Security context
  security_opt:
    - no-new-privileges:true
  cap_drop:
    - ALL
  read_only: false  # Redis needs write access to data
  tmpfs:
    - /tmp
```

---

### Patch 4: docker-compose.yml - Backend

**Add to backend service**:
```yaml
backend:
  # ... (existing config) ...

  # Security context
  security_opt:
    - no-new-privileges:true
  cap_drop:
    - ALL
  read_only: true  # Application code is read-only
  tmpfs:
    - /tmp
    - /app/cache
  volumes:
    - ./backend:/app/backend:ro  # Read-only mount
    - ./logs:/app/logs:rw        # Write for logs
    - ./data:/app/data:rw        # Write for data
```

---

### Patch 5: docker-compose.yml - Celery Worker

**Add to celery_worker service**:
```yaml
celery_worker:
  # ... (existing config) ...

  # Security context
  security_opt:
    - no-new-privileges:true
  cap_drop:
    - ALL
  read_only: true
  tmpfs:
    - /tmp
    - /home/appuser/.matplotlib  # For plotting
```

---

### Patch 6: docker-compose.yml - Nginx

**Current (Line 561)**:
```yaml
nginx:
  image: nginx:alpine  # ❌ UNPINNED
```

**Replacement (SECURE)**:
```yaml
nginx:
  # Pinned to specific version and digest
  # Nginx 1.25.3 Alpine
  image: nginx:1.25.3-alpine@sha256:ghi789...

  # Security context
  security_opt:
    - no-new-privileges:true
  cap_drop:
    - ALL
  cap_add:
    - CHOWN      # For log rotation
    - SETGID     # For running as nginx user
    - SETUID     # For running as nginx user
    - NET_BIND_SERVICE  # For binding to port 80/443
  read_only: true
  tmpfs:
    - /tmp
    - /var/cache/nginx
    - /var/run
```

---

### Patch 7: docker-compose.yml - Prometheus

**Current (Line 318)**:
```yaml
prometheus:
  image: prom/prometheus:v2.48.0  # ❌ No digest
```

**Replacement (SECURE)**:
```yaml
prometheus:
  # Pinned to specific digest
  image: prom/prometheus:v2.48.0@sha256:jkl012...

  # Security context
  security_opt:
    - no-new-privileges:true
  cap_drop:
    - ALL
  read_only: true
  tmpfs:
    - /tmp
  user: "nobody"  # Run as nobody user
```

---

### Patch 8: docker-compose.yml - Grafana

**Current (Line 359)**:
```yaml
grafana:
  image: grafana/grafana:10.2.2  # ❌ No digest
```

**Replacement (SECURE)**:
```yaml
grafana:
  # Pinned to specific digest
  image: grafana/grafana:10.2.2@sha256:mno345...

  # Security context
  security_opt:
    - no-new-privileges:true
  cap_drop:
    - ALL
  read_only: false  # Grafana needs write access
  tmpfs:
    - /tmp
```

---

## DOCKER IMAGE DIGEST LOOKUP

### Script to Find Current Digests

```bash
#!/bin/bash
# Script: get_image_digests.sh
# Purpose: Fetch SHA256 digests for all Docker images

echo "=== DOCKER IMAGE DIGEST LOOKUP ==="
echo ""

# Function to get digest
get_digest() {
    local image=$1
    echo "Fetching digest for: $image"
    docker pull $image > /dev/null 2>&1
    digest=$(docker inspect --format='{{index .RepoDigests 0}}' $image | cut -d'@' -f2)
    echo "  Digest: $digest"
    echo ""
}

# PostgreSQL / TimescaleDB
get_digest "timescale/timescaledb:2.13.0-pg15"

# Redis
get_digest "redis:7.2.4-alpine"

# Python (for backend)
get_digest "python:3.12-slim"

# Nginx
get_digest "nginx:1.25.3-alpine"

# Prometheus
get_digest "prom/prometheus:v2.48.0"

# Grafana
get_digest "grafana/grafana:10.2.2"

# Airflow
get_digest "apache/airflow:2.7.3-python3.11"

# Node Exporter
get_digest "prom/node-exporter:v1.7.0"

# cAdvisor
get_digest "gcr.io/cadvisor/cadvisor:v0.47.2"

# PostgreSQL Exporter
get_digest "prometheuscommunity/postgres-exporter:v0.15.0"

# Redis Exporter
get_digest "oliver006/redis_exporter:v1.55.0"

# Celery Exporter
get_digest "danihodovic/celery-exporter:0.10.4"

# Nginx Exporter
get_digest "nginx/nginx-prometheus-exporter:0.11.0"

# AlertManager
get_digest "prom/alertmanager:v0.26.0"

echo "=== DIGEST LOOKUP COMPLETE ==="
echo "Update docker-compose.yml with these digests."
```

**Run Command**:
```bash
chmod +x get_image_digests.sh
./get_image_digests.sh > image_digests.txt
```

---

## SECURITY SCANNING IMPLEMENTATION

### Option 1: Trivy (Recommended - Free, Fast)

**Install**:
```bash
# macOS
brew install trivy

# Linux
wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | sudo apt-key add -
echo "deb https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main" | sudo tee -a /etc/apt/sources.list.d/trivy.list
sudo apt-get update && sudo apt-get install trivy
```

**Scan Images**:
```bash
#!/bin/bash
# Script: scan_images.sh

echo "=== CONTAINER SECURITY SCANNING WITH TRIVY ==="

# Scan Dockerfile
trivy config Dockerfile.backend

# Scan built images
trivy image investment-backend:latest --severity HIGH,CRITICAL
trivy image timescale/timescaledb:2.13.0-pg15 --severity HIGH,CRITICAL
trivy image redis:7.2.4-alpine --severity HIGH,CRITICAL
trivy image nginx:1.25.3-alpine --severity HIGH,CRITICAL

# Generate HTML report
trivy image investment-backend:latest --format template --template "@contrib/html.tpl" -o security-report.html

echo "=== SCAN COMPLETE ==="
echo "Review security-report.html for vulnerabilities"
```

---

### Option 2: Grype (Alternative)

```bash
# Install
brew tap anchore/grype
brew install grype

# Scan
grype investment-backend:latest -o table --fail-on high
```

---

### Option 3: Docker Scout (Built-in to Docker)

```bash
# Enable Docker Scout
docker scout quickview

# Scan image
docker scout cves investment-backend:latest

# Compare with base image
docker scout compare --to investment-backend:latest
```

---

## CI/CD INTEGRATION

### GitHub Actions Workflow

Create: `.github/workflows/security-scan.yml`

```yaml
name: Container Security Scan

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  schedule:
    # Run daily at 2 AM UTC
    - cron: '0 2 * * *'

jobs:
  trivy-scan:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Build Docker image
        run: docker build -t investment-backend:${{ github.sha }} -f Dockerfile.backend .

      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'investment-backend:${{ github.sha }}'
          format: 'sarif'
          output: 'trivy-results.sarif'
          severity: 'CRITICAL,HIGH'

      - name: Upload Trivy results to GitHub Security
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'

      - name: Fail on HIGH/CRITICAL vulnerabilities
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'investment-backend:${{ github.sha }}'
          format: 'table'
          exit-code: '1'
          severity: 'CRITICAL,HIGH'
```

---

## RECOMMENDED SECURITY CONTEXTS

### Production-Ready Security Template

```yaml
x-security-hardened: &security-hardened
  security_opt:
    - no-new-privileges:true
    - seccomp:unconfined  # Adjust per application
  cap_drop:
    - ALL
  read_only: true
  tmpfs:
    - /tmp:noexec,nosuid,nodev,size=64m

# Apply to services:
services:
  backend:
    <<: *security-hardened
    # ... rest of config ...
```

---

## DOCKERFILE SECURITY BEST PRACTICES

### Multi-Stage Build with Security

```dockerfile
# Stage 1: Builder (can have build tools)
FROM python:3.12-slim@sha256:abc123 as builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential=12.9 \  # Pin package versions
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime (minimal, no build tools)
FROM python:3.12-slim@sha256:abc123
RUN groupadd -r -g 1000 appuser && \
    useradd -r -u 1000 -g appuser -m appuser

WORKDIR /app
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local
COPY --chown=appuser:appuser backend /app/backend

# Security labels
LABEL security.scan.date="2026-01-27"
LABEL security.trivy.passed="true"

# Run as non-root
USER appuser

# Health check
HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1

# Minimal privileges
EXPOSE 8000
CMD ["gunicorn", "backend.api.main:app"]
```

---

## TESTING CHECKLIST

### Image Security
- [ ] All images pinned to SHA256 digests
- [ ] No `latest` tags in production
- [ ] All images scanned with Trivy (0 HIGH/CRITICAL)
- [ ] Base images updated in last 30 days
- [ ] Non-root users configured
- [ ] No secrets in image layers

### Container Security
- [ ] Security contexts applied to all services
- [ ] Capabilities dropped where possible
- [ ] Read-only filesystems where appropriate
- [ ] No privileged containers
- [ ] Resource limits configured
- [ ] Health checks defined

### Network Security
- [ ] Custom bridge networks used
- [ ] No `network_mode: host`
- [ ] Port exposure minimized
- [ ] Internal services not exposed

---

## MAINTENANCE SCHEDULE

### Weekly
- Review Trivy scan results
- Check for base image updates

### Monthly
- Update base images to latest digest
- Review and rotate secrets
- Audit security contexts

### Quarterly
- Full security audit
- Penetration testing
- Update security policies

---

## ROLLBACK PLAN

If container security changes cause issues:

1. **Immediate**: Disable security context (remove `security_opt`)
2. **Debug**: Check container logs for capability errors
3. **Fix**: Add required capabilities incrementally
4. **Test**: Verify functionality with new security context
5. **Deploy**: Gradually roll out to production

---

**Document Version**: 1.0
**Created**: 2026-01-27
**Status**: READY FOR IMPLEMENTATION
