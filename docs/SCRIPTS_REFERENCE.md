# Scripts Reference

> Auto-generated from project scripts. Last updated: 2026-01-26

This document provides a comprehensive reference for all shell and Python scripts in the investment-analysis-platform.

---

## Quick Start Scripts (Root Level)

| Script | Description | Usage |
|--------|-------------|-------|
| `setup.sh` | Initial environment setup | `./setup.sh` |
| `start.sh` | Start development/production environment | `./start.sh [dev\|prod\|test]` |
| `stop.sh` | Stop all running services | `./stop.sh` |
| `logs.sh` | View service logs | `./logs.sh [service-name]` |
| `notion-sync.sh` | Sync project status to Notion | `./notion-sync.sh push` |

---

## Setup Scripts

Located in `scripts/setup/`

| Script | Description |
|--------|-------------|
| `INIT.sh` | Master initialization script |
| `SETUP_ENVIRONMENT.sh` | Environment configuration |
| `setup.sh` | General setup script |
| `setup_local.py` | Local development setup |
| `fix_python_deps.sh` | Fix Python dependency issues |
| `install_test_dependencies.sh` | Install test dependencies |
| `verify_dependencies.py` | Verify all dependencies installed |
| `diagnose_imports.py` | Diagnose import issues |
| `download_models.py` | Download ML models |
| `init_db.py` | Initialize database schema |
| `run_connection_tests.sh` | Test service connections |
| `test_cicd_setup.sh` | Validate CI/CD configuration |

### Additional Root Setup Scripts

| Script | Description |
|--------|-------------|
| `install_critical_deps.sh` | Install critical dependencies |
| `install_system_deps.sh` | Install system-level dependencies |
| `setup_wsl.sh` | WSL-specific setup (Windows) |
| `setup_environment.sh` | Configure environment variables |
| `setup_airflow_complete.sh` | Complete Airflow setup |

---

## Deployment Scripts

Located in `scripts/deployment/`

| Script | Description |
|--------|-------------|
| `QUICK_START.sh` | Quick deployment for testing |
| `deploy.sh` | Standard deployment |
| `production_deploy.sh` | Production deployment with safety checks |
| `blue_green_deploy.sh` | Blue-green deployment strategy |
| `start.sh` | Start services |
| `stop.sh` | Stop services |
| `restart.sh` | Restart services |
| `rollback.sh` | Rollback to previous version |
| `start-docker.sh` | Start Docker containers |
| `start-full-stack.sh` | Start complete application stack |
| `start_app.sh` | Start application only |
| `start_data_loading.sh` | Start data loading services |
| `start_data_pipeline.sh` | Start data pipeline |
| `start_monitoring.sh` | Start monitoring stack |

---

## Database Scripts

| Script | Location | Description |
|--------|----------|-------------|
| `init_database.sh` | `scripts/` | Initialize PostgreSQL database |
| `init_database_fixed.sh` | `scripts/` | Fixed database initialization |
| `init_database.py` | `scripts/` | Python database initialization |
| `fix_database_schema.py` | `scripts/` | Fix schema issues |
| `apply_database_optimizations.py` | `scripts/` | Apply performance optimizations |
| `setup_db_credentials.py` | `scripts/` | Configure DB credentials |
| `simple_migrate.py` | `scripts/` | Run database migrations |

---

## Data Pipeline Scripts

Located in `scripts/data/`

| Script | Description |
|--------|-------------|
| `activate_data_pipeline.py` | Activate the data pipeline |
| `activate_pipeline.py` | Alternative pipeline activation |
| `background_loader.py` | Background data loading service |
| `background_loader_enhanced.py` | Enhanced loader with better error handling |
| `load_data_now.py` | Immediate data loading |
| `mock_data_generator.py` | Generate mock data for testing |
| `simple_mock_generator.py` | Simple mock data generator |

### Additional Pipeline Scripts

| Script | Location | Description |
|--------|----------|-------------|
| `activate_etl_pipeline.py` | `scripts/` | Activate ETL pipeline |
| `run_enhanced_etl.py` | `scripts/` | Run enhanced ETL process |
| `start_data_pipeline.py` | `scripts/` | Start data pipeline service |
| `load_historical_data.py` | `scripts/` | Load historical stock data |

---

## ML Operations Scripts

| Script | Description |
|--------|-------------|
| `train_ml_models.py` | Train all ML models |
| `train_ml_models_minimal.py` | Minimal model training |
| `train_models_simple.py` | Simplified training script |
| `deploy_ml_models.py` | Deploy trained models |
| `deploy_ml_production.sh` | Production ML deployment |
| `download_ml_models.py` | Download pre-trained models |
| `download_models.py` | Generic model download |
| `load_trained_models.py` | Load models into memory |
| `create_model_artifacts.py` | Generate model artifacts |
| `ml_scheduler.py` | Schedule ML training jobs |
| `ml_services_startup.sh` | Start ML services |
| `start_ml_api.sh` | Start ML API server |

---

## Monitoring Scripts

| Script | Location | Description |
|--------|----------|-------------|
| `monitoring-health-check.sh` | `scripts/` | Check service health |
| `setup-monitoring.sh` | `scripts/` | Set up monitoring stack |
| `monitor_pipeline.py` | `scripts/monitoring/` | Monitor data pipeline |

---

## Testing Scripts

Located in `scripts/testing/`

| Script | Description |
|--------|-------------|
| `test_all_passwords.py` | Test credential configuration |
| `test_api_connections.py` | Test API connectivity |
| `test_api_final.py` | Final API validation |
| `test_api_quick.py` | Quick API tests |
| `test_connections.py` | Test all connections |
| `test_docker_connections.py` | Test Docker connectivity |
| `test_docker_connections_fixed.py` | Fixed Docker connection tests |
| `test_fixes.py` | Test fix implementations |
| `test_pipeline_components.py` | Test pipeline components |
| `test_services_corrected.py` | Corrected service tests |
| `test_services_fixed.py` | Fixed service tests |
| `test_services_quick.py` | Quick service validation |

### Validation Scripts

Located in `scripts/testing/validation/`

| Script | Description |
|--------|-------------|
| `validate-deployment.py` | Validate deployment |
| `validate_api_setup.py` | Validate API configuration |
| `validate_env_vars.py` | Validate environment variables |
| `validate_structure.py` | Validate project structure |

### Additional Test Scripts

| Script | Location | Description |
|--------|----------|-------------|
| `run_all_tests.py` | `scripts/` | Run complete test suite |
| `generate_test_report.py` | `scripts/` | Generate test reports |
| `coverage_analysis.py` | `scripts/` | Analyze test coverage |
| `test_performance.sh` | Root | Performance benchmarks |

---

## Backup & Recovery Scripts

| Script | Description |
|--------|-------------|
| `backup.sh` | Create database backup |
| `restore-backup.sh` | Restore from backup |
| `verify-backup.sh` | Verify backup integrity |

---

## Security Scripts

| Script | Description |
|--------|-------------|
| `generate_secrets.sh` | Generate secure secrets |
| `manage-secrets.sh` | Manage application secrets |
| `migrate_secrets.py` | Migrate secrets to new format |
| `security_validation.py` | Validate security configuration |
| `init-ssl.sh` | Initialize SSL certificates |

---

## Airflow Scripts

| Script | Description |
|--------|-------------|
| `init_airflow.sh` | Initialize Airflow |
| `check_airflow_status.sh` | Check Airflow status |
| `start_airflow_minimal.sh` | Start minimal Airflow |
| `setup_airflow_complete.sh` | Complete Airflow setup |
| `test_airflow_pipeline.py` | Test Airflow pipelines |
| `migrate_airflow_to_prefect.py` | Migration to Prefect |

---

## Optimization Scripts

| Script | Location | Description |
|--------|----------|-------------|
| `optimize_performance.sh` | `scripts/` | Performance optimization |
| `static_asset_optimizer.py` | `scripts/optimization/` | Optimize static assets |
| `migrate_to_optimized.sh` | `scripts/` | Migrate to optimized config |

---

## Utility Scripts

| Script | Description |
|--------|-------------|
| `create_admin_user.py` | Create admin user account |
| `quick_setup.py` | Quick project setup |
| `notion_sync.py` | Sync with Notion API |
| `validate_cicd.py` | Validate CI/CD pipeline |
| `validate_data.py` | Validate data integrity |

---

## Usage Examples

### Initial Setup
```bash
# Full setup
./setup.sh

# Start development environment
./start.sh dev

# View logs
./logs.sh backend
```

### Database Operations
```bash
# Initialize database
./scripts/init_database.sh

# Apply optimizations
python scripts/apply_database_optimizations.py

# Run migrations
python scripts/simple_migrate.py
```

### ML Operations
```bash
# Train models
python scripts/train_ml_models.py

# Deploy to production
./scripts/deploy_ml_production.sh

# Start ML API
./scripts/start_ml_api.sh
```

### Backup Operations
```bash
# Create backup
./scripts/backup.sh

# Verify backup
./scripts/verify-backup.sh

# Restore from backup
./scripts/restore-backup.sh
```

---

## Environment Requirements

Most scripts expect:
- Python 3.12+
- Docker and docker-compose
- PostgreSQL client tools
- Environment variables from `.env`

See [ENVIRONMENT.md](./ENVIRONMENT.md) for complete environment variable reference.
