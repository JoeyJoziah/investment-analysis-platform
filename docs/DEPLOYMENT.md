# Production Deployment Guide

**Last Updated**: 2026-01-27
**Version**: 1.0.0
**Status**: Production-Ready

---

## Table of Contents

1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [SSL Certificate Setup](#ssl-certificate-setup)
3. [Domain Configuration](#domain-configuration)
4. [Production Environment](#production-environment)
5. [Database Setup](#database-setup)
6. [Service Startup](#service-startup)
7. [Smoke Testing](#smoke-testing)
8. [Monitoring & Verification](#monitoring--verification)
9. [Scaling Considerations](#scaling-considerations)
10. [Backup & Recovery](#backup--recovery)

---

## Pre-Deployment Checklist

Before deploying to production, ensure:

### Infrastructure Requirements
- [ ] Server with minimum 4GB RAM (8GB recommended)
- [ ] 50GB+ disk space for databases and models
- [ ] Docker Engine 20.10+
- [ ] Docker Compose 2.0+
- [ ] Public IP address or domain name
- [ ] Port 80 (HTTP) accessible for Let's Encrypt
- [ ] Port 443 (HTTPS) open for production traffic

### Configuration Files
- [ ] `.env.production` created from `.env.example`
- [ ] `GDPR_ENCRYPTION_KEY` set (generate: `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`)
- [ ] `JWT_SECRET_KEY` configured
- [ ] Database credentials set
- [ ] API keys configured (Finnhub, Alpha Vantage, etc.)

### Domain & SSL
- [ ] Domain name registered or assigned
- [ ] DNS A record pointing to server IP
- [ ] Email address for Let's Encrypt certificate

### Monitoring
- [ ] Slack webhook configured (optional)
- [ ] SMTP credentials set for alerts (optional)
- [ ] AlertManager rules reviewed

### Backups
- [ ] Backup strategy defined (daily recommended)
- [ ] S3 bucket created (if using AWS backups)
- [ ] Backup scripts tested

### Team Readiness
- [ ] Incident response plan documented
- [ ] On-call rotation established
- [ ] Runbooks reviewed by team

---

## SSL Certificate Setup

### Option 1: Let's Encrypt (Recommended for Production)

Let's Encrypt provides free, automatically-renewed SSL certificates.

#### Prerequisites
```bash
# Install certbot
sudo apt-get update
sudo apt-get install -y certbot python3-certbot-nginx

# Or with Docker:
docker pull certbot/certbot
```

#### Generate Certificate

```bash
# Replace with your actual domain
DOMAIN="yourdomain.com"
EMAIL="admin@yourdomain.com"

# Stop Nginx temporarily (if running)
sudo systemctl stop nginx

# Generate certificate (standalone mode)
certbot certonly --standalone \
  -d $DOMAIN \
  -d "www.$DOMAIN" \
  --email $EMAIL \
  --agree-tos \
  --non-interactive \
  --preferred-challenges http

# Certificate locations:
# /etc/letsencrypt/live/$DOMAIN/fullchain.pem (public certificate)
# /etc/letsencrypt/live/$DOMAIN/privkey.pem (private key)
```

#### Automatic Renewal

Let's Encrypt certificates expire after 90 days. Set up automatic renewal:

```bash
# Test renewal (dry run)
certbot renew --dry-run

# Enable auto-renewal timer
sudo systemctl enable certbot.timer
sudo systemctl start certbot.timer

# Check renewal status
sudo systemctl status certbot.timer
```

#### Copy Certificates to Application

```bash
# Create cert directory in project
mkdir -p ./certs

# Copy Let's Encrypt certificates
sudo cp /etc/letsencrypt/live/$DOMAIN/fullchain.pem ./certs/
sudo cp /etc/letsencrypt/live/$DOMAIN/privkey.pem ./certs/

# Fix permissions
sudo chown $USER:$USER ./certs/*.pem
chmod 600 ./certs/privkey.pem
```

### Option 2: Self-Signed Certificate (Development/Testing Only)

For testing purposes only:

```bash
# Generate self-signed certificate (valid 365 days)
openssl req -x509 -newkey rsa:4096 -nodes \
  -out ./certs/cert.pem \
  -keyout ./certs/key.pem \
  -days 365 \
  -subj "/CN=localhost"

# Or use the provided script:
./setup-ssl.sh localhost
```

**Warning**: Self-signed certificates will show security warnings in browsers. Use Let's Encrypt for production.

---

## Domain Configuration

### DNS Setup

1. **Get Server IP Address**
```bash
# Public IP
curl ifconfig.me

# Or check in cloud provider dashboard
```

2. **Update DNS A Record**

Log into your domain registrar (GoDaddy, Namecheap, etc.) and create:

```
Record Type: A
Name: @ (or yourdomain.com)
Value: <your-server-ip>
TTL: 3600

Optional: Create CNAME for www
Record Type: CNAME
Name: www
Value: yourdomain.com
TTL: 3600
```

3. **Verify DNS Resolution**

```bash
# Should resolve to your server IP
nslookup yourdomain.com

# Or with dig
dig yourdomain.com

# Check propagation globally
dig yourdomain.com @8.8.8.8
```

### Firewall Configuration

```bash
# Allow HTTP (for Let's Encrypt renewal)
sudo ufw allow 80/tcp

# Allow HTTPS
sudo ufw allow 443/tcp

# Allow specific backend port (optional, for testing)
sudo ufw allow 8000/tcp

# Check rules
sudo ufw status
```

---

## Production Environment

### Environment Variables

Create `.env.production`:

```bash
# Copy from template
cp .env.example .env.production

# Edit with production values
nano .env.production
```

**Required Configuration**:

```bash
# Environment
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=info

# Database
DB_HOST=investment_db
DB_PORT=5432
DB_NAME=investment_db
DB_USER=investment_user
DB_PASSWORD=<strong-random-password>
DATABASE_URL=postgresql://investment_user:<password>@investment_db:5432/investment_db

# Redis
REDIS_URL=redis://investment_cache:6379/0

# Security
JWT_SECRET_KEY=<generate-random-256-bit-key>
GDPR_ENCRYPTION_KEY=<fernet-key-from-above>
ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com

# Domain & HTTPS
SSL_DOMAIN=yourdomain.com
SSL_CERT_PATH=/app/certs/fullchain.pem
SSL_KEY_PATH=/app/certs/privkey.pem

# APIs (Configure as needed)
FINNHUB_API_KEY=<your-key>
ALPHA_VANTAGE_API_KEY=<your-key>
POLYGON_API_KEY=<your-key>
NEWSAPI_KEY=<your-key>

# Monitoring (Optional)
SLACK_WEBHOOK_URL=<optional>
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=<your-email>
SMTP_PASSWORD=<app-password>
ADMIN_EMAIL=<alert-recipient>

# AI Integration (Optional)
OPENAI_API_KEY=<optional>
ANTHROPIC_API_KEY=<optional>
HUGGING_FACE_TOKEN=<optional>
```

### Generate Secure Keys

```bash
# JWT Secret (256-bit)
python3 -c "import secrets; print(secrets.token_urlsafe(32))"

# GDPR Encryption Key (Fernet)
python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# Database Password (random)
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

---

## Database Setup

### Initialize Production Database

```bash
# Start database container only
docker-compose -f docker-compose.prod.yml up -d investment_db

# Wait for PostgreSQL to be ready (check logs)
docker-compose -f docker-compose.prod.yml logs -f investment_db | grep "database system is ready"

# Run migrations
docker-compose -f docker-compose.prod.yml exec investment_db \
  psql -U postgres -c "CREATE USER investment_user WITH PASSWORD '$DB_PASSWORD' CREATEDB;"

docker-compose -f docker-compose.prod.yml exec investment_db \
  psql -U investment_user -d investment_db -f /docker-entrypoint-initdb.d/01-init-schema.sql
```

### Create Application Role

```bash
# Create application user
docker-compose -f docker-compose.prod.yml exec investment_db \
  psql -U postgres -c "
    CREATE ROLE investment_user WITH LOGIN PASSWORD '$DB_PASSWORD';
    GRANT CONNECT ON DATABASE investment_db TO investment_user;
    GRANT USAGE ON SCHEMA public TO investment_user;
    GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO investment_user;
    GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO investment_user;
  "
```

### Load Stock Data

```bash
# If you have pre-exported stock data:
docker-compose -f docker-compose.prod.yml exec investment_db \
  psql -U investment_user -d investment_db < /path/to/stock_data.sql

# Or use the stock loader script:
docker-compose -f docker-compose.prod.yml exec investment_backend \
  python backend/scripts/load_stocks.py
```

### Verify Database

```bash
# Check connection
docker-compose -f docker-compose.prod.yml exec investment_db \
  psql -U investment_user -d investment_db -c "SELECT 1;"

# List tables
docker-compose -f docker-compose.prod.yml exec investment_db \
  psql -U investment_user -d investment_db -c "\dt"

# Count stocks
docker-compose -f docker-compose.prod.yml exec investment_db \
  psql -U investment_user -d investment_db -c "SELECT COUNT(*) FROM stocks;"
```

---

## Service Startup

### Production Deployment (All Services)

```bash
# Build production Docker images
docker-compose -f docker-compose.prod.yml build

# Start all services
docker-compose -f docker-compose.prod.yml up -d

# Verify all services started
docker-compose -f docker-compose.prod.yml ps

# Expected output (all healthy):
# investment_db          Up (healthy)
# investment_cache       Up (healthy)
# investment_search      Up (healthy)
# investment_backend     Up (healthy)
# investment_worker      Up (healthy)
# investment_scheduler   Up (healthy)
# investment_airflow     Up
# investment_prometheus  Up
# investment_grafana     Up
# investment_alertmanager Up
```

### Check Service Logs

```bash
# View logs for all services
docker-compose -f docker-compose.prod.yml logs -f

# View specific service logs
docker-compose -f docker-compose.prod.yml logs -f investment_backend

# View last 100 lines
docker-compose -f docker-compose.prod.yml logs -n 100 investment_backend

# Follow logs with grep filter
docker-compose -f docker-compose.prod.yml logs -f | grep ERROR
```

### Verify Service Health

```bash
# Check health status
docker-compose -f docker-compose.prod.yml exec investment_backend \
  curl http://localhost:8000/api/health

# Expected response:
# {
#   "status": "healthy",
#   "version": "1.0.0",
#   "timestamp": "2026-01-27T00:00:00Z"
# }
```

---

## Smoke Testing

### Health Checks

```bash
# Backend API health
curl https://yourdomain.com/api/health

# Expected response (200 OK):
# {"status": "healthy", "version": "1.0.0"}

# API documentation
curl https://yourdomain.com/api/docs

# Database health
curl https://yourdomain.com/api/health/db

# Redis connection
curl https://yourdomain.com/api/health/redis

# Elasticsearch
curl https://yourdomain.com/api/health/elasticsearch
```

### API Endpoint Testing

```bash
# Get available stocks (should be non-empty after data load)
curl -X GET https://yourdomain.com/api/stocks \
  -H "Authorization: Bearer $JWT_TOKEN"

# Get recommendations
curl -X GET https://yourdomain.com/api/recommendations \
  -H "Authorization: Bearer $JWT_TOKEN"

# Test authentication (should fail)
curl -X GET https://yourdomain.com/api/protected-endpoint

# WebSocket connection test
wscat -c wss://yourdomain.com/ws
```

### Database Connectivity

```bash
# Verify stock data loaded
curl -X GET https://yourdomain.com/api/stocks?limit=1 \
  -H "Authorization: Bearer $JWT_TOKEN"

# Verify count > 0
```

### Frontend Access

```bash
# Test frontend loads
curl -I https://yourdomain.com

# Should return: HTTP/1.1 200 OK
# Should serve index.html
```

### Monitoring Dashboards

```bash
# Access Prometheus
https://yourdomain.com:9090

# Access Grafana (default login: admin/admin)
https://yourdomain.com:3001

# Check CPU, Memory, Disk usage
# Check API response times
# Check database connections
# Verify alerts configured
```

---

## Monitoring & Verification

### Grafana Dashboards

1. **Navigate to**: `https://yourdomain.com:3001`
2. **Login**: admin/admin (change password immediately)
3. **Verify dashboards**:
   - Application Metrics
   - Database Metrics
   - System Metrics
   - Business Metrics

### Key Metrics to Monitor

```
API Performance:
- Request Rate: should be >0 for active users
- Response Time: should be <500ms p95
- Error Rate: should be <1%

Database:
- Connection Count: should scale with load
- Query Time: should be <100ms p95
- Disk Usage: should not exceed 80%

System:
- CPU Usage: should be <80% under normal load
- Memory Usage: should be <80%
- Disk I/O: should be <80%

Cache:
- Hit Rate: should be >80%
- Memory Usage: should not exceed configured limit
```

### Alert Configuration

Configure alerts in AlertManager for:

```
- High error rate (>5% of requests)
- High latency (>1s p95 response time)
- Database connection failures
- Redis connection failures
- Disk space low (<10% free)
- Memory usage high (>90%)
- CPU usage high (>90% for >5 minutes)
```

### Log Aggregation

```bash
# View application logs
docker-compose -f docker-compose.prod.yml logs investment_backend

# View error logs specifically
docker-compose -f docker-compose.prod.yml logs investment_backend | grep ERROR

# Export logs for analysis
docker-compose -f docker-compose.prod.yml logs > /tmp/production-logs.txt

# View database logs
docker-compose -f docker-compose.prod.yml logs investment_db
```

---

## Scaling Considerations

### Horizontal Scaling

For high traffic, scale services:

```bash
# Scale backend workers
docker-compose -f docker-compose.prod.yml up -d --scale investment_backend=3

# Scale Celery workers
docker-compose -f docker-compose.prod.yml up -d --scale investment_worker=5

# Load balancer configuration needed (Nginx/HAProxy)
```

### Database Optimization

```sql
-- Check index usage
SELECT schemaname, tablename, indexname, idx_scan
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;

-- Check slow queries
SELECT query, mean_time, calls
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;

-- Vacuum and analyze
VACUUM ANALYZE;
```

### Redis Optimization

```bash
# Monitor Redis memory
redis-cli INFO memory

# Clear cache if needed
redis-cli FLUSHDB

# Configure eviction policy
redis-cli CONFIG SET maxmemory-policy allkeys-lru
```

### Database Replication

For high availability, consider:

```bash
# PostgreSQL replication setup
# (Configuration in docker-compose.prod.yml with REPLICATION_MODE=replica)

# Elasticsearch replication
# (Configure in elasticsearch.yml with number_of_replicas: 2)
```

---

## Backup & Recovery

### Automated Backups

```bash
# Run database backup
./db-backup.sh

# Backup to S3 (if configured)
./db-backup.sh --s3

# Check backup status
ls -lah ./backups/

# List S3 backups
aws s3 ls s3://your-backup-bucket/
```

### Manual Backup

```bash
# PostgreSQL full backup
docker-compose -f docker-compose.prod.yml exec investment_db \
  pg_dump -U investment_user -d investment_db > backup-$(date +%Y%m%d-%H%M%S).sql

# Redis snapshot
docker-compose -f docker-compose.prod.yml exec investment_cache \
  redis-cli BGSAVE

# Elasticsearch snapshot
curl -X PUT "localhost:9200/_snapshot/backup" \
  -H 'Content-Type: application/json' \
  -d '{"type": "fs", "settings": {"location": "/mnt/backups/elasticsearch"}}'
```

### Recovery Procedures

#### PostgreSQL Recovery

```bash
# Stop application
docker-compose -f docker-compose.prod.yml down investment_backend investment_worker

# Restore from backup
docker-compose -f docker-compose.prod.yml exec investment_db \
  psql -U investment_user -d investment_db < /path/to/backup.sql

# Verify data restored
docker-compose -f docker-compose.prod.yml exec investment_db \
  psql -U investment_user -d investment_db -c "SELECT COUNT(*) FROM stocks;"

# Restart application
docker-compose -f docker-compose.prod.yml up -d investment_backend investment_worker
```

#### Full System Recovery

```bash
# 1. Restore database
./db-restore.sh backup-20260127-143000.sql

# 2. Restore Redis cache
./cache-restore.sh

# 3. Restart all services
docker-compose -f docker-compose.prod.yml restart

# 4. Verify health
curl https://yourdomain.com/api/health
```

### Backup Retention Policy

```bash
# Keep local backups for 7 days
find ./backups/ -name "*.sql" -mtime +7 -delete

# Keep S3 backups for 30 days (configure S3 lifecycle policy)
# AWS S3 > Bucket > Lifecycle rules > Set expiration to 30 days
```

---

## Troubleshooting Common Issues

### Service Won't Start

```bash
# Check logs
docker-compose -f docker-compose.prod.yml logs investment_backend

# Check resource availability
docker stats

# Restart service
docker-compose -f docker-compose.prod.yml restart investment_backend
```

### Database Connection Errors

```bash
# Verify database is running
docker-compose -f docker-compose.prod.yml exec investment_db psql -U postgres -c "SELECT 1;"

# Check credentials in .env.production
grep DB_ .env.production

# Restart database
docker-compose -f docker-compose.prod.yml restart investment_db
```

### SSL Certificate Issues

```bash
# Check certificate validity
openssl x509 -in ./certs/cert.pem -text -noout

# Check certificate expiration
openssl x509 -in ./certs/cert.pem -noout -dates

# Renew certificate
certbot renew --force-renewal
```

### High Memory Usage

```bash
# Check which service is using memory
docker stats

# Increase container limits (in docker-compose.prod.yml)
# services:
#   investment_backend:
#     mem_limit: 2g
#     mem_reservation: 1g

# Restart service
docker-compose -f docker-compose.prod.yml up -d
```

### Slow API Response Times

```bash
# Check database query performance
docker-compose -f docker-compose.prod.yml exec investment_db \
  psql -U investment_user -d investment_db -c "
    SELECT query, mean_time, calls
    FROM pg_stat_statements
    ORDER BY mean_time DESC LIMIT 5;
  "

# Check Redis cache hit rate
redis-cli INFO stats

# Optimize slow queries with indexes
CREATE INDEX idx_stock_ticker ON stocks(ticker);
```

---

## Post-Deployment Checklist

After successful deployment:

- [ ] All services running and healthy
- [ ] SSL certificate active and valid
- [ ] Domain name resolves correctly
- [ ] API endpoints responding correctly
- [ ] Frontend loads without errors
- [ ] Authentication working
- [ ] Monitoring dashboards active
- [ ] Alerts configured and tested
- [ ] Backups scheduled and tested
- [ ] Team trained on operations
- [ ] Incident response plan reviewed
- [ ] Performance baselines documented
- [ ] Security checklist completed

---

## Support & Escalation

### Common Contact Points

```
On-Call Engineer: [contact-info]
Backend Lead: [contact-info]
DevOps Lead: [contact-info]
Emergency Hotline: [phone-number]
```

### Escalation Procedure

1. **SEV 1** (System down): Immediate notification to all leads
2. **SEV 2** (Degraded): Notify on-call engineer within 15 minutes
3. **SEV 3** (Minor issue): File ticket for next business day

### Runbook Links

- Service Restart: See above
- Database Recovery: See Backup & Recovery section
- SSL Certificate Renewal: See SSL Certificate Setup section
- Scaling: See Scaling Considerations section

---

## Next Steps

1. Acquire domain name
2. Configure DNS A records
3. Generate SSL certificate using Let's Encrypt
4. Update `.env.production` with all credentials
5. Run `./start.sh prod`
6. Execute smoke tests
7. Enable monitoring and alerts
8. Schedule team training

---

*Document Version: 1.0.0*
*Last Updated: 2026-01-27*
*Maintainer: DevOps Team*
