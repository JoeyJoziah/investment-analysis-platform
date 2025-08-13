# Service Down Recovery Runbook

## Overview
This runbook provides procedures for recovering from service outages in the Investment Analysis Application. Use this when one or more services are completely down or unresponsive.

## Prerequisites
- SSH/terminal access to the deployment environment
- Docker and docker-compose access
- Basic knowledge of the application architecture

## Immediate Response (First 5 Minutes)

### 1. Assess the Situation
```bash
# Check overall system status
make status

# Check Docker containers
docker ps -a

# Check if services are accessible
curl -f http://localhost:8000/api/health || echo "API down"
curl -f http://localhost:3000 || echo "Frontend down"
```

### 2. Check System Resources
```bash
# Check disk space
df -h

# Check memory usage
free -h

# Check CPU usage
top -bn1 | grep "Cpu(s)"

# Check Docker disk usage
docker system df
```

### 3. Check Logs for Errors
```bash
# Check recent logs for all services
make logs | tail -100

# Check specific service logs
docker-compose logs --tail=50 backend
docker-compose logs --tail=50 postgres
docker-compose logs --tail=50 redis
```

## Recovery Procedures

### Scenario 1: Single Service Down

#### Backend API Down
```bash
# Check backend container status
docker-compose ps backend

# View backend logs
docker-compose logs backend | tail -100

# Restart backend service
docker-compose restart backend

# Wait for startup (30-60 seconds)
sleep 60

# Verify API is responding
curl -f http://localhost:8000/api/health

# Check database connectivity
curl -f http://localhost:8000/api/health/db
```

#### Database Down
```bash
# Check PostgreSQL container
docker-compose ps postgres

# Check disk space for database volume
docker volume inspect investment_analysis_app_postgres_data

# Restart database
docker-compose restart postgres

# Wait for database startup
sleep 30

# Test database connection
docker-compose exec postgres pg_isready -h localhost -p 5432
```

#### Redis Cache Down
```bash
# Check Redis container
docker-compose ps redis

# Restart Redis
docker-compose restart redis

# Test Redis connection
docker-compose exec redis redis-cli ping
```

### Scenario 2: Multiple Services Down

#### Complete Stack Restart
```bash
# Stop all services gracefully
make down

# Clean up any orphaned containers
docker-compose rm -f

# Start services in dependency order
docker-compose up -d postgres redis
sleep 30

docker-compose up -d backend
sleep 60

docker-compose up -d frontend

# Verify all services are running
make status
```

#### Network Issues
```bash
# Check Docker networks
docker network ls
docker network inspect investment_analysis_app_default

# Recreate network if needed
docker-compose down
docker network prune -f
docker-compose up -d
```

### Scenario 3: Database Corruption

#### Database Recovery
```bash
# Stop all services
make down

# Create backup of current database
docker run --rm -v investment_analysis_app_postgres_data:/data -v $(pwd):/backup ubuntu tar czf /backup/postgres_backup_$(date +%Y%m%d_%H%M%S).tar.gz /data

# Start only database
docker-compose up -d postgres

# Check database integrity
docker-compose exec postgres psql -U postgres -d investment_db -c "SELECT pg_database_size('investment_db');"

# If database is corrupted, restore from backup
# (Restore procedure depends on backup strategy)
```

### Scenario 4: Out of Disk Space

#### Immediate Actions
```bash
# Stop services to prevent further data corruption
make down

# Clean Docker system
docker system prune -a -f

# Remove old logs
find /var/log -name "*.log" -mtime +7 -delete 2>/dev/null || true

# Check if space is freed
df -h

# If still low on space, remove old database backups
find . -name "*.backup" -mtime +30 -delete

# Restart services
make up
```

## Verification Procedures

### 1. Service Health Checks
```bash
# API health check
curl -f http://localhost:8000/api/health
echo $? # Should return 0

# Database connectivity
curl -f http://localhost:8000/api/health/db
echo $? # Should return 0

# Frontend accessibility
curl -f http://localhost:3000
echo $? # Should return 0
```

### 2. Functional Testing
```bash
# Test API endpoints
curl -f "http://localhost:8000/api/stocks/AAPL"

# Test database queries
docker-compose exec postgres psql -U postgres -d investment_db -c "SELECT COUNT(*) FROM stocks;"

# Test cache functionality
docker-compose exec redis redis-cli set test_key test_value
docker-compose exec redis redis-cli get test_key
docker-compose exec redis redis-cli del test_key
```

### 3. Performance Verification
```bash
# Check response times
time curl -s http://localhost:8000/api/health

# Check database performance
docker-compose exec postgres psql -U postgres -d investment_db -c "SELECT pg_stat_database.datname, pg_stat_database.numbackends, pg_stat_database.xact_commit, pg_stat_database.xact_rollback FROM pg_stat_database WHERE datname='investment_db';"

# Monitor resource usage
docker stats --no-stream
```

## Post-Recovery Actions

### 1. Update Monitoring
```bash
# Reset any alert states
# (Depends on monitoring system used)

# Check metrics collection
curl -f http://localhost:8000/metrics || echo "Metrics endpoint not responding"
```

### 2. Data Integrity Checks
```bash
# Run data quality checks
docker-compose exec backend python -c "
from backend.utils.data_quality import DataQualityChecker
checker = DataQualityChecker()
print('Running data integrity checks...')
# Add your data integrity verification here
"

# Verify recent data ingestion
docker-compose exec postgres psql -U postgres -d investment_db -c "
SELECT 
    COUNT(*) as total_records,
    MAX(date) as latest_date,
    COUNT(DISTINCT stock_id) as unique_stocks
FROM price_history 
WHERE date >= CURRENT_DATE - INTERVAL '2 days';"
```

### 3. Documentation
```bash
# Log the incident
echo "$(date): Service recovery completed - $(whoami)" >> /var/log/incidents.log

# Document any configuration changes made
# Update this runbook if procedures were modified
```

## Prevention Measures

### 1. Enable Monitoring Alerts
- Set up alerts for service health endpoints
- Monitor disk space usage
- Set up database connection monitoring

### 2. Regular Health Checks
- Schedule automated health checks every 5 minutes
- Set up log rotation to prevent disk space issues
- Implement database backup verification

### 3. Resource Management
- Set up Docker container resource limits
- Implement automatic log cleanup
- Monitor API usage patterns

## Escalation Criteria

Contact the on-call engineer if:
- Services cannot be restored within 30 minutes
- Data corruption is suspected
- Multiple recovery attempts have failed
- System resources are critically low and cannot be freed

## Related Runbooks
- [Database Issues](./database-issues.md)
- [Memory/CPU Issues](./memory-cpu-issues.md)
- [API Rate Limit Exceeded](./api-rate-limit-exceeded.md)
- [Daily Health Checks](./daily-health-checks.md)