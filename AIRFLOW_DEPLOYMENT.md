# Airflow Deployment Guide

This guide provides instructions for deploying Apache Airflow with Docker Compose for the Investment Analysis Application, with specific focus on Windows compatibility and environment variable configuration.

## Overview

The Airflow deployment has been specifically configured to handle environment variable issues commonly encountered on Windows systems. This deployment includes:

- Proper environment variable handling for Windows PowerShell/Command Prompt
- Separated Airflow-specific configuration files
- Windows-compatible startup scripts
- Comprehensive error handling and debugging tools

## Prerequisites

1. **Docker Desktop** - Ensure Docker Desktop is installed and running
2. **Docker Compose** - Included with Docker Desktop
3. **Windows PowerShell 5.1+** or **Command Prompt**
4. **Minimum Resources**:
   - 4GB RAM
   - 2 CPU cores
   - 10GB disk space

## File Structure

```
investment_analysis_app/
├── .env                           # Main environment variables
├── .env.airflow                   # Airflow-specific variables
├── docker-compose.airflow.yml     # Airflow Docker Compose configuration
├── start-airflow.bat             # Windows batch script
├── start-airflow.ps1             # PowerShell script
├── debug-airflow-env.ps1         # Environment debugging script
└── scripts/
    └── init_airflow_db.sql       # Database initialization script
```

## Environment Files

### .env.airflow

This file contains all Airflow-specific environment variables that were causing Docker Compose warnings:

- `AIRFLOW_FERNET_KEY` - Encryption key for Airflow secrets
- `AIRFLOW_SECRET_KEY` - Flask session key
- `SMTP_USER`, `SMTP_PASSWORD` - Email configuration (optional)
- `FLOWER_PASSWORD` - Celery Flower monitoring password
- `_AIRFLOW_WWW_USER_USERNAME`, `_AIRFLOW_WWW_USER_PASSWORD` - Admin user credentials

### Main .env File Updates

The main `.env` file has been updated to include:
- `AIRFLOW_FERNET_KEY` - Same as in .env.airflow for compatibility
- `AIRFLOW_SECRET_KEY` - Flask secret key
- Additional Airflow variables for consistency

## Quick Start

### Option 1: PowerShell Script (Recommended)

```powershell
# Start Airflow with initialization (first time)
.\start-airflow.ps1 -Init

# Start Airflow (subsequent runs)
.\start-airflow.ps1

# Stop Airflow
.\start-airflow.ps1 -Stop

# View logs for all services
.\start-airflow.ps1 -Logs

# View logs for specific service
.\start-airflow.ps1 -Logs -Service airflow-webserver

# Restart all services
.\start-airflow.ps1 -Restart

# Restart specific service
.\start-airflow.ps1 -Restart -Service airflow-scheduler
```

### Option 2: Batch Script

```cmd
# Double-click start-airflow.bat or run from command prompt
start-airflow.bat
```

### Option 3: Manual Docker Compose

```bash
# Set environment variables (PowerShell)
$env:AIRFLOW_UID = "50000"
$env:COMPOSE_PROJECT_NAME = "investment-airflow"

# Initialize Airflow (first time only)
docker-compose -f docker-compose.airflow.yml up airflow-init

# Start services
docker-compose -f docker-compose.airflow.yml up -d
```

## Service Access

Once started, the following services will be available:

| Service | URL | Credentials |
|---------|-----|-------------|
| **Airflow Webserver** | http://localhost:8080 | admin / secure_admin_password_789 |
| **Flower (Celery Monitor)** | http://localhost:5555 | admin / secure_flower_password_123 |
| **StatsD Exporter** | http://localhost:9102 | No auth required |
| **PostgreSQL** | localhost:5432 | postgres / [DB_PASSWORD] |
| **Redis** | localhost:6379 | Password: [REDIS_PASSWORD] |

## Troubleshooting

### Environment Variable Issues

1. **Run the debug script**:
   ```powershell
   .\debug-airflow-env.ps1
   ```

2. **Check Docker Compose configuration**:
   ```bash
   docker-compose -f docker-compose.airflow.yml config
   ```

3. **Verify environment files exist**:
   - `.env` - Main application variables
   - `.env.airflow` - Airflow-specific variables

### Common Issues and Solutions

#### 1. "Variable not set" warnings

**Problem**: Docker Compose shows warnings about missing variables like `AIRFLOW_FERNET_KEY`, `SMTP_USER`, etc.

**Solution**: 
- Use the provided `.env.airflow` file
- Run startup scripts which set proper environment variables
- All variables now have default values in docker-compose.airflow.yml

#### 2. Permission denied errors

**Problem**: Files owned by root, cannot access logs/dags

**Solution**:
- Scripts automatically set `AIRFLOW_UID=50000`
- Run `docker-compose down` and restart with scripts

#### 3. Database connection errors

**Problem**: Cannot connect to Airflow database

**Solution**:
- Run initialization first: `.\start-airflow.ps1 -Init`
- Check PostgreSQL is running: `docker-compose -f docker-compose.airflow.yml ps postgres`
- Verify database credentials match in both .env files

#### 4. Services fail to start

**Problem**: Containers exit immediately or fail health checks

**Solution**:
```bash
# Check logs for specific service
docker-compose -f docker-compose.airflow.yml logs [service-name]

# Common services to check:
docker-compose -f docker-compose.airflow.yml logs postgres
docker-compose -f docker-compose.airflow.yml logs redis
docker-compose -f docker-compose.airflow.yml logs airflow-webserver
```

## Configuration Details

### Environment Variable Resolution Order

Docker Compose resolves environment variables in this order:

1. Variables set in shell environment
2. Variables from `.env.airflow` file
3. Variables from `.env` file
4. Default values in docker-compose.airflow.yml

### Security Considerations

**Change these passwords before production:**

1. `AIRFLOW_DB_PASSWORD` - Database password
2. `_AIRFLOW_WWW_USER_PASSWORD` - Web interface admin password
3. `FLOWER_PASSWORD` - Monitoring interface password
4. `AIRFLOW_FERNET_KEY` - Generate new key:
   ```python
   from cryptography.fernet import Fernet
   print(Fernet.generate_key().decode())
   ```

### Resource Limits

Services are configured with resource limits:

| Service | CPU Limit | Memory Limit | CPU Reservation | Memory Reservation |
|---------|-----------|--------------|------------------|-------------------|
| Webserver | 2.0 | 2GB | 0.5 | 512MB |
| Scheduler | 2.0 | 2GB | 1.0 | 1GB |
| Worker (API) | 1.5 | 1.5GB | 0.5 | 512MB |
| Worker (Compute) | 3.0 | 3GB | 1.0 | 1GB |
| Worker (Default) | 2.0 | 2GB | 0.5 | 512MB |

## Useful Commands

### Service Management

```bash
# View running containers
docker-compose -f docker-compose.airflow.yml ps

# Stop all services
docker-compose -f docker-compose.airflow.yml down

# Restart a specific service
docker-compose -f docker-compose.airflow.yml restart airflow-webserver

# View logs (follow mode)
docker-compose -f docker-compose.airflow.yml logs -f airflow-scheduler

# Execute command in container
docker-compose -f docker-compose.airflow.yml exec airflow-webserver bash
```

### Database Management

```bash
# Connect to PostgreSQL
docker-compose -f docker-compose.airflow.yml exec postgres psql -U postgres -d investment_db

# Connect to Airflow database
docker-compose -f docker-compose.airflow.yml exec postgres psql -U airflow_user -d airflow_db

# Run database migration
docker-compose -f docker-compose.airflow.yml exec airflow-webserver airflow db upgrade
```

### Airflow CLI Commands

```bash
# List DAGs
docker-compose -f docker-compose.airflow.yml exec airflow-webserver airflow dags list

# Test a task
docker-compose -f docker-compose.airflow.yml exec airflow-webserver airflow tasks test [dag_id] [task_id] [execution_date]

# Create user
docker-compose -f docker-compose.airflow.yml exec airflow-webserver airflow users create \
    --username admin \
    --firstname FIRST_NAME \
    --lastname LAST_NAME \
    --role Admin \
    --email admin@example.com
```

## Development Workflow

1. **Start Services**: Use `start-airflow.ps1` or `start-airflow.bat`
2. **Access Web UI**: Go to http://localhost:8080
3. **Develop DAGs**: Place files in `data_pipelines/airflow/dags/`
4. **Monitor**: Use Flower at http://localhost:5555 for Celery monitoring
5. **Debug**: Check logs with `docker-compose -f docker-compose.airflow.yml logs [service]`

## Production Considerations

1. **Change all default passwords**
2. **Use external database** for better performance
3. **Configure email notifications** (SMTP settings)
4. **Set up monitoring** and alerting
5. **Configure backup strategy** for Airflow metadata
6. **Use secrets management** instead of environment files
7. **Configure SSL/TLS** for web interface
8. **Set appropriate resource limits** based on workload

## Support

For issues specific to this Airflow deployment:

1. Run the debug script: `.\debug-airflow-env.ps1`
2. Check Docker Compose configuration: `docker-compose -f docker-compose.airflow.yml config`
3. Review service logs: `docker-compose -f docker-compose.airflow.yml logs [service-name]`
4. Verify all required files exist and have correct content

The deployment is specifically optimized for Windows environments while maintaining compatibility with Linux/WSL systems.