#!/usr/bin/env python3
"""
Apply comprehensive database performance optimizations

This script applies all database optimizations including:
1. Connection pool configuration updates
2. Critical database indexes
3. Time-based partitioning
4. Materialized views
5. Performance monitoring setup
"""

import sys
import os
import logging
import json
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.utils.database import get_db_sync, get_connection_pool_status
from backend.utils.migration_manager import MigrationManager
from backend.utils.database_monitoring import DatabaseMonitor
from backend.utils.performance_tester import validate_database_optimizations
from sqlalchemy import text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_prerequisites() -> bool:
    """Check prerequisites for database optimization"""
    
    logger.info("Checking prerequisites...")
    
    try:
        # Test database connection
        db = get_db_sync()
        result = db.execute(text("SELECT version()")).fetchone()
        logger.info(f"Connected to PostgreSQL: {result[0]}")
        db.close()
        
        # Check if alembic is available
        import subprocess
        result = subprocess.run(['alembic', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"Alembic available: {result.stdout.strip()}")
        else:
            logger.error("Alembic not available")
            return False
        
        # Check disk space
        import shutil
        free_gb = shutil.disk_usage('.').free / (1024**3)
        logger.info(f"Available disk space: {free_gb:.2f} GB")
        
        if free_gb < 5:
            logger.error("Insufficient disk space (need at least 5GB)")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Prerequisites check failed: {e}")
        return False


def apply_connection_pool_optimizations() -> bool:
    """Apply connection pool optimizations"""
    
    logger.info("Applying connection pool optimizations...")
    
    try:
        # Connection pool is already optimized in backend/utils/database.py
        # Verify the settings are applied
        pool_status = get_connection_pool_status()
        logger.info(f"Connection pool status: {pool_status}")
        
        if pool_status['size'] >= 15:
            logger.info("✓ Connection pool size optimized")
            return True
        else:
            logger.warning("! Connection pool size not optimal")
            return False
        
    except Exception as e:
        logger.error(f"Connection pool optimization failed: {e}")
        return False


def apply_database_migrations() -> bool:
    """Apply database migrations for indexes and partitioning"""
    
    logger.info("Applying database migrations...")
    
    try:
        db = get_db_sync()
        migration_manager = MigrationManager(db)
        
        # Run safety checks
        safety_checks = migration_manager.validate_migration_safety()
        logger.info(f"Migration safety checks: {json.dumps(safety_checks, indent=2)}")
        
        if not safety_checks['safe_to_migrate']:
            logger.error("Migration safety checks failed")
            return False
        
        # Run migrations with backup
        migration_result = migration_manager.run_migration_with_safety_checks(
            target_revision="head",
            create_backup=True
        )
        
        logger.info(f"Migration result: {json.dumps(migration_result, indent=2)}")
        
        if migration_result['success']:
            logger.info("✓ Database migrations applied successfully")
            return True
        else:
            logger.error(f"Migration failed: {migration_result.get('error')}")
            return False
        
    except Exception as e:
        logger.error(f"Migration application failed: {e}")
        return False
    
    finally:
        db.close()


def setup_monitoring() -> bool:
    """Setup database monitoring"""
    
    logger.info("Setting up database monitoring...")
    
    try:
        db = get_db_sync()
        monitor = DatabaseMonitor(db)
        
        # Enable slow query logging
        monitor.enable_slow_query_logging(threshold_ms=1000)
        
        # Generate initial performance report
        performance_report = monitor.generate_performance_report()
        
        # Save performance report
        reports_dir = Path("db_reports")
        reports_dir.mkdir(exist_ok=True)
        
        report_file = reports_dir / f"initial_performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(performance_report, f, indent=2, default=str)
        
        logger.info(f"✓ Performance report saved to {report_file}")
        logger.info("✓ Database monitoring setup completed")
        
        return True
        
    except Exception as e:
        logger.error(f"Monitoring setup failed: {e}")
        return False
    
    finally:
        db.close()


def run_performance_validation() -> bool:
    """Run performance validation tests"""
    
    logger.info("Running performance validation...")
    
    try:
        validation_results = validate_database_optimizations()
        
        # Save validation results
        reports_dir = Path("db_reports")
        reports_dir.mkdir(exist_ok=True)
        
        validation_file = reports_dir / f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(validation_file, 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        
        logger.info(f"Validation results saved to {validation_file}")
        
        # Check validation status
        validations = validation_results.get('validations', {})
        
        success = True
        for validation_name, validation_data in validations.items():
            if isinstance(validation_data, dict):
                status = validation_data.get('status', 'OK')
                if status == 'OK':
                    logger.info(f"✓ {validation_name}: PASSED")
                else:
                    logger.warning(f"! {validation_name}: {status}")
                    success = False
        
        return success
        
    except Exception as e:
        logger.error(f"Performance validation failed: {e}")
        return False


def create_maintenance_script():
    """Create ongoing database maintenance script"""
    
    logger.info("Creating maintenance script...")
    
    maintenance_script = '''#!/usr/bin/env python3
"""
Database maintenance script - run daily for optimal performance
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.utils.database import get_db_sync
from backend.utils.database_monitoring import DatabaseMonitor
from backend.utils.migration_manager import MigrationManager
from sqlalchemy import text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_maintenance():
    """Run daily database maintenance"""
    
    logger.info("Starting database maintenance...")
    
    try:
        db = get_db_sync()
        monitor = DatabaseMonitor(db)
        
        # Log connection pool status
        from backend.utils.database import log_connection_pool_metrics
        log_connection_pool_metrics()
        
        # Check table health
        main_tables = ['price_history', 'technical_indicators', 'recommendations']
        for table in main_tables:
            health = monitor.check_table_health(table)
            logger.info(f"Table {table} health score: {health.get('health_score', 'unknown')}")
            
            # Run VACUUM ANALYZE if needed
            if health.get('dead_tuple_percentage', 0) > 10:
                logger.info(f"Running VACUUM ANALYZE on {table}")
                db.execute(text(f"VACUUM ANALYZE {table}"))
                db.commit()
        
        # Check for slow queries
        slow_queries = monitor.get_slow_queries(5)
        if slow_queries:
            logger.warning(f"Found {len(slow_queries)} slow queries")
            for sq in slow_queries:
                logger.warning(f"Slow query: {sq.query[:100]}... ({sq.avg_duration_ms:.2f}ms avg)")
        
        # Clean up old backups
        migration_manager = MigrationManager(db)
        cleaned = migration_manager.cleanup_old_backups(keep_days=30)
        logger.info(f"Cleaned up {cleaned} old backup files")
        
        logger.info("Database maintenance completed")
        
    except Exception as e:
        logger.error(f"Maintenance failed: {e}")
        return False
    
    finally:
        db.close()
    
    return True


if __name__ == "__main__":
    run_maintenance()
'''
    
    scripts_dir = Path("scripts")
    maintenance_file = scripts_dir / "db_maintenance.py"
    
    with open(maintenance_file, 'w') as f:
        f.write(maintenance_script)
    
    # Make script executable
    maintenance_file.chmod(0o755)
    
    logger.info(f"✓ Maintenance script created: {maintenance_file}")


def main():
    """Main optimization application function"""
    
    logger.info("=" * 60)
    logger.info("APPLYING DATABASE PERFORMANCE OPTIMIZATIONS")
    logger.info("=" * 60)
    
    # Check prerequisites
    if not check_prerequisites():
        logger.error("Prerequisites check failed. Aborting.")
        sys.exit(1)
    
    success_steps = []
    total_steps = 5
    
    # Step 1: Apply connection pool optimizations
    logger.info(f"\n[1/{total_steps}] Applying connection pool optimizations...")
    if apply_connection_pool_optimizations():
        success_steps.append("Connection Pool")
    
    # Step 2: Apply database migrations
    logger.info(f"\n[2/{total_steps}] Applying database migrations...")
    if apply_database_migrations():
        success_steps.append("Database Migrations")
    
    # Step 3: Setup monitoring
    logger.info(f"\n[3/{total_steps}] Setting up database monitoring...")
    if setup_monitoring():
        success_steps.append("Monitoring Setup")
    
    # Step 4: Run performance validation
    logger.info(f"\n[4/{total_steps}] Running performance validation...")
    if run_performance_validation():
        success_steps.append("Performance Validation")
    
    # Step 5: Create maintenance script
    logger.info(f"\n[5/{total_steps}] Creating maintenance script...")
    try:
        create_maintenance_script()
        success_steps.append("Maintenance Script")
    except Exception as e:
        logger.error(f"Failed to create maintenance script: {e}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("OPTIMIZATION SUMMARY")
    logger.info("=" * 60)
    
    logger.info(f"Successfully completed: {len(success_steps)}/{total_steps} steps")
    
    for step in success_steps:
        logger.info(f"✓ {step}")
    
    failed_steps = total_steps - len(success_steps)
    if failed_steps > 0:
        logger.warning(f"! {failed_steps} steps failed or had issues")
    
    if len(success_steps) == total_steps:
        logger.info("\n🎉 All database optimizations applied successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Monitor application performance")
        logger.info("2. Run scripts/db_maintenance.py daily")
        logger.info("3. Check db_reports/ for performance metrics")
    else:
        logger.warning("\n⚠️  Some optimizations failed. Check logs above.")
        logger.info("You may need to:")
        logger.info("1. Fix any database connection issues")
        logger.info("2. Ensure sufficient permissions")
        logger.info("3. Check PostgreSQL configuration")
    
    logger.info("\n" + "=" * 60)


if __name__ == "__main__":
    main()