# Database Setup and Configuration Guide

This guide explains how to properly set up and configure the database infrastructure for the Investment Analysis Platform.

## Overview

The platform uses three main data stores:
1. **PostgreSQL** - Primary relational database for structured data
2. **Redis** - In-memory cache for API responses and session data
3. **Elasticsearch** - Full-text search engine for fast queries

## Development Setup

### Option 1: Docker Compose (Recommended)

The easiest way to set up all databases for development:

```bash
# Start all databases
docker-compose up -d postgres redis elasticsearch

# Verify they're running
docker-compose ps

# Check logs if needed
docker-compose logs postgres
```

### Option 2: Local Installation

If you prefer to install databases locally:

#### PostgreSQL
```bash
# macOS
brew install postgresql@15
brew services start postgresql@15

# Ubuntu/Debian
sudo apt update
sudo apt install postgresql-15 postgresql-client-15

# Create database and user
sudo -u postgres psql
CREATE DATABASE investment_db;
CREATE USER investment_user WITH ENCRYPTED PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE investment_db TO investment_user;
\q
```

#### Redis
```bash
# macOS
brew install redis
brew services start redis

# Ubuntu/Debian
sudo apt update
sudo apt install redis-server
sudo systemctl enable redis-server
sudo systemctl start redis-server
```

## Production Database Setup

### 1. PostgreSQL Production Setup

For production, you have several options:

#### Option A: Managed Database Service (Recommended)
- **DigitalOcean Managed Databases**: $15/month for basic cluster
- **AWS RDS**: Free tier available (750 hours/month)
- **Google Cloud SQL**: $7.67/month for basic instance
- **Supabase**: Free tier with 500MB storage

#### Option B: Self-Hosted on VPS
```bash
# On your production server (Ubuntu 22.04)
sudo apt update
sudo apt install postgresql-15 postgresql-client-15

# Secure PostgreSQL
sudo -u postgres psql

# Create production database
CREATE DATABASE investment_db;

# Create application user with limited privileges
CREATE USER investment_app WITH ENCRYPTED PASSWORD 'generate_strong_password_here';
GRANT CONNECT ON DATABASE investment_db TO investment_app;
GRANT USAGE ON SCHEMA public TO investment_app;
GRANT CREATE ON SCHEMA public TO investment_app;

# Create read-only user for analytics
CREATE USER investment_readonly WITH ENCRYPTED PASSWORD 'another_strong_password';
GRANT CONNECT ON DATABASE investment_db TO investment_readonly;
GRANT USAGE ON SCHEMA public TO investment_readonly;

\q
```

### 2. Configure PostgreSQL for Production

Edit `/etc/postgresql/15/main/postgresql.conf`:
```conf
# Connection settings
listen_addresses = 'localhost'  # Change to '*' if accessing from other servers
max_connections = 100
shared_buffers = 256MB

# Performance settings for our use case
effective_cache_size = 1GB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1

# Logging
log_min_duration_statement = 1000  # Log queries slower than 1 second
log_line_prefix = '%t [%p] %u@%d '
```

Edit `/etc/postgresql/15/main/pg_hba.conf`:
```conf
# TYPE  DATABASE        USER            ADDRESS                 METHOD
local   all             all                                     peer
host    all             all             127.0.0.1/32            scram-sha-256
host    investment_db   investment_app  your_app_server/32      scram-sha-256
```

### 3. Database Security Best Practices

1. **Use Strong Passwords**
   ```bash
   # Generate secure password
   openssl rand -base64 32
   ```

2. **Enable SSL/TLS**
   ```bash
   # In postgresql.conf
   ssl = on
   ssl_cert_file = 'server.crt'
   ssl_key_file = 'server.key'
   ```

3. **Set up Firewall Rules**
   ```bash
   # Only allow connections from your app servers
   sudo ufw allow from your_app_server_ip to any port 5432
   ```

4. **Regular Backups**
   ```bash
   # Set up automated backups
   pg_dump -U investment_app -h localhost investment_db | gzip > backup_$(date +%Y%m%d_%H%M%S).sql.gz
   ```

## Database Credentials Configuration

### Development Environment (.env)
```env
# PostgreSQL
DATABASE_URL=postgresql://postgres:password@localhost:5432/investment_db
DB_HOST=localhost
DB_PORT=5432
DB_NAME=investment_db
DB_USER=postgres
DB_PASSWORD=password

# Redis
REDIS_URL=redis://localhost:6379/0
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Elasticsearch
ELASTICSEARCH_URL=http://localhost:9200
```

### Production Environment (.env.production)
```env
# PostgreSQL (Managed Database Example)
DATABASE_URL=postgresql://investment_app:strong_password_here@db-cluster.example.com:5432/investment_db?sslmode=require
DB_HOST=db-cluster.example.com
DB_PORT=5432
DB_NAME=investment_db
DB_USER=investment_app
DB_PASSWORD=strong_password_here
DB_SSL_MODE=require

# Connection Pool Settings
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=40
DATABASE_POOL_TIMEOUT=30
DATABASE_POOL_RECYCLE=1800

# Redis (Managed or Self-Hosted)
REDIS_URL=redis://:redis_password@redis-cluster.example.com:6379/0
REDIS_PASSWORD=redis_password_here
REDIS_POOL_SIZE=10
REDIS_SOCKET_TIMEOUT=5
REDIS_SOCKET_CONNECT_TIMEOUT=5

# Elasticsearch
ELASTICSEARCH_URL=https://elastic:password@elasticsearch.example.com:9200
ELASTICSEARCH_USERNAME=elastic
ELASTICSEARCH_PASSWORD=elastic_password_here
```

## Database Initialization

### 1. Create Database Schema

```bash
# Run migrations
docker-compose exec backend alembic upgrade head

# Or directly
python -m alembic upgrade head
```

### 2. Create Initial Tables

The application will automatically create tables on first run, but you can also initialize manually:

```sql
-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search

-- Create main tables (example)
CREATE TABLE IF NOT EXISTS stocks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(10) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    exchange VARCHAR(50),
    sector VARCHAR(100),
    industry VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_stocks_symbol ON stocks(symbol);
CREATE INDEX idx_stocks_sector ON stocks(sector);

-- Create price data table with partitioning for performance
CREATE TABLE IF NOT EXISTS stock_prices (
    id BIGSERIAL,
    stock_id UUID REFERENCES stocks(id),
    date DATE NOT NULL,
    open DECIMAL(10, 2),
    high DECIMAL(10, 2),
    low DECIMAL(10, 2),
    close DECIMAL(10, 2),
    volume BIGINT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (stock_id, date)
) PARTITION BY RANGE (date);

-- Create partitions for each year
CREATE TABLE stock_prices_2024 PARTITION OF stock_prices
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
CREATE TABLE stock_prices_2025 PARTITION OF stock_prices
    FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');
```

### 3. Load Initial Data

```bash
# Load stock symbols
python backend/utils/load_initial_data.py

# Or use the management command
docker-compose exec backend python -m backend.utils.db_init
```

## Connection Pooling

For production, proper connection pooling is critical:

```python
# In your application configuration
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool, QueuePool

# Production settings
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,          # Number of persistent connections
    max_overflow=40,       # Maximum overflow connections
    pool_timeout=30,       # Timeout for getting connection
    pool_recycle=1800,     # Recycle connections after 30 minutes
    pool_pre_ping=True,    # Test connections before using
)
```

## Monitoring and Maintenance

### 1. Monitor Database Performance

```sql
-- Check slow queries
SELECT query, mean_exec_time, calls 
FROM pg_stat_statements 
WHERE mean_exec_time > 1000 
ORDER BY mean_exec_time DESC;

-- Check table sizes
SELECT 
    schemaname AS table_schema,
    tablename AS table_name,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Check connection count
SELECT count(*) FROM pg_stat_activity;
```

### 2. Regular Maintenance Tasks

```bash
# Vacuum and analyze tables (should be automated)
VACUUM ANALYZE;

# Reindex for performance
REINDEX DATABASE investment_db;

# Update statistics
ANALYZE;
```

### 3. Backup Strategy

```bash
# Daily backups with retention
#!/bin/bash
BACKUP_DIR="/backups/postgres"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DB_NAME="investment_db"

# Create backup
pg_dump -U investment_app -h localhost $DB_NAME | gzip > $BACKUP_DIR/backup_$TIMESTAMP.sql.gz

# Keep only last 30 days
find $BACKUP_DIR -name "backup_*.sql.gz" -mtime +30 -delete

# Upload to cloud storage (optional)
aws s3 cp $BACKUP_DIR/backup_$TIMESTAMP.sql.gz s3://your-backup-bucket/postgres/
```

## Cost Optimization Tips

1. **Use Connection Pooling**: Reduces connection overhead
2. **Implement Caching**: Use Redis to cache expensive queries
3. **Partition Large Tables**: Improves query performance
4. **Regular Maintenance**: VACUUM and ANALYZE regularly
5. **Monitor Disk Usage**: Set up alerts for disk space

## Troubleshooting

### Common Issues

1. **Connection Refused**
   ```bash
   # Check if PostgreSQL is running
   sudo systemctl status postgresql
   
   # Check listening ports
   sudo netstat -tlnp | grep 5432
   ```

2. **Authentication Failed**
   ```bash
   # Check pg_hba.conf settings
   # Ensure password is correct
   # Try connecting locally first
   psql -U investment_app -d investment_db -h localhost
   ```

3. **Performance Issues**
   ```sql
   -- Check for missing indexes
   SELECT schemaname, tablename, indexname, idx_scan
   FROM pg_stat_user_indexes
   WHERE idx_scan = 0
   ORDER BY schemaname, tablename;
   ```

## Security Checklist

- [ ] Strong passwords for all database users
- [ ] SSL/TLS enabled for connections
- [ ] Firewall rules configured
- [ ] Regular security updates applied
- [ ] Audit logging enabled
- [ ] Backup encryption enabled
- [ ] Principle of least privilege for users
- [ ] Connection limits configured
- [ ] Monitoring and alerting set up

Remember: Never commit database credentials to version control!