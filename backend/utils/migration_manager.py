"""
Database migration management and rollback utilities
"""

import subprocess
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class MigrationManager:
    """Manage database migrations with safety checks and rollback capabilities"""
    
    def __init__(self, db: Session, alembic_config_path: str = "alembic.ini"):
        self.db = db
        self.alembic_config = alembic_config_path
        self.backup_dir = Path("db_backups")
        self.backup_dir.mkdir(exist_ok=True)
    
    def create_backup(self, backup_name: Optional[str] = None) -> str:
        """Create database backup before migration"""
        
        if not backup_name:
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_file = self.backup_dir / f"{backup_name}.sql"
        
        try:
            # Get database connection details from current session
            db_url = str(self.db.bind.url)
            
            # Parse database URL
            from urllib.parse import urlparse
            parsed = urlparse(db_url.replace('+asyncpg', ''))
            
            # Use pg_dump to create backup
            cmd = [
                'pg_dump',
                '--verbose',
                '--no-password',
                '--format=custom',
                '--compress=9',
                f'--host={parsed.hostname}',
                f'--port={parsed.port or 5432}',
                f'--username={parsed.username}',
                f'--dbname={parsed.path.lstrip("/")}',
                f'--file={backup_file}'
            ]
            
            # Set password via environment if available
            import os
            env = os.environ.copy()
            if parsed.password:
                env['PGPASSWORD'] = parsed.password
            
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Database backup created: {backup_file}")
                return str(backup_file)
            else:
                logger.error(f"Backup failed: {result.stderr}")
                raise Exception(f"Backup failed: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise
    
    def restore_backup(self, backup_file: str) -> bool:
        """Restore database from backup"""
        
        backup_path = Path(backup_file)
        if not backup_path.exists():
            logger.error(f"Backup file not found: {backup_file}")
            return False
        
        try:
            # Get database connection details
            db_url = str(self.db.bind.url)
            from urllib.parse import urlparse
            parsed = urlparse(db_url.replace('+asyncpg', ''))
            
            # Drop existing connections to database
            self._drop_database_connections(parsed.path.lstrip("/"))
            
            # Use pg_restore to restore backup
            cmd = [
                'pg_restore',
                '--verbose',
                '--clean',
                '--no-owner',
                '--no-privileges',
                '--no-password',
                f'--host={parsed.hostname}',
                f'--port={parsed.port or 5432}',
                f'--username={parsed.username}',
                f'--dbname={parsed.path.lstrip("/")}',
                str(backup_path)
            ]
            
            import os
            env = os.environ.copy()
            if parsed.password:
                env['PGPASSWORD'] = parsed.password
            
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Database restored from: {backup_file}")
                return True
            else:
                logger.error(f"Restore failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to restore backup: {e}")
            return False
    
    def _drop_database_connections(self, database_name: str):
        """Drop all connections to database before restore"""
        
        try:
            drop_connections_query = text("""
                SELECT pg_terminate_backend(pid)
                FROM pg_stat_activity 
                WHERE datname = :db_name 
                  AND pid <> pg_backend_pid()
            """)
            
            self.db.execute(drop_connections_query, {'db_name': database_name})
            self.db.commit()
            
        except Exception as e:
            logger.warning(f"Could not drop connections: {e}")
    
    def get_current_migration_head(self) -> Optional[str]:
        """Get current migration head"""
        
        try:
            result = subprocess.run(
                ['alembic', '-c', self.alembic_config, 'current'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                output = result.stdout.strip()
                if output and 'current' in output.lower():
                    # Extract revision ID from output
                    lines = output.split('\n')
                    for line in lines:
                        if 'current' in line.lower() and '(' in line:
                            revision = line.split('(')[0].strip()
                            return revision
                return None
            else:
                logger.error(f"Failed to get current migration: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting current migration: {e}")
            return None
    
    def get_migration_history(self) -> List[Dict[str, Any]]:
        """Get migration history"""
        
        try:
            result = subprocess.run(
                ['alembic', '-c', self.alembic_config, 'history', '--verbose'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                history = []
                lines = result.stdout.strip().split('\n')
                
                for line in lines:
                    if '->' in line and 'Rev:' in line:
                        # Parse revision info
                        parts = line.split(',')
                        if len(parts) >= 2:
                            rev_part = parts[0].strip()
                            desc_part = parts[1].strip() if len(parts) > 1 else ""
                            
                            if 'Rev:' in rev_part:
                                revision = rev_part.split('Rev:')[1].strip().split('(')[0].strip()
                                history.append({
                                    'revision': revision,
                                    'description': desc_part,
                                    'line': line.strip()
                                })
                
                return history
            else:
                logger.error(f"Failed to get migration history: {result.stderr}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting migration history: {e}")
            return []
    
    def validate_migration_safety(self, target_revision: Optional[str] = None) -> Dict[str, Any]:
        """Validate migration safety before execution"""
        
        safety_checks = {
            'safe_to_migrate': True,
            'warnings': [],
            'critical_issues': [],
            'recommendations': []
        }
        
        try:
            # Check database size
            size_info = self._get_database_size()
            if size_info.get('size_mb', 0) > 10000:  # 10GB
                safety_checks['warnings'].append(
                    f"Large database ({size_info.get('size_pretty', 'unknown')}) - migration may take significant time"
                )
            
            # Check active connections
            active_connections = self._get_active_connections()
            if active_connections > 10:
                safety_checks['warnings'].append(
                    f"High number of active connections ({active_connections}) - consider migrating during low usage"
                )
            
            # Check table sizes for tables being modified
            large_tables = self._check_large_tables(['price_history', 'technical_indicators'])
            for table, size in large_tables.items():
                if size > 1000000:  # 1M rows
                    safety_checks['warnings'].append(
                        f"Large table {table} ({size:,} rows) - index creation may be slow"
                    )
            
            # Check available disk space
            free_space = self._check_disk_space()
            if free_space and free_space < 5000:  # 5GB
                safety_checks['critical_issues'].append(
                    f"Low disk space ({free_space:.1f} MB) - ensure sufficient space for migration"
                )
                safety_checks['safe_to_migrate'] = False
            
            # Check for long-running queries
            long_queries = self._check_long_running_queries()
            if long_queries:
                safety_checks['warnings'].append(
                    f"Found {len(long_queries)} long-running queries - may interfere with migration"
                )
            
            # Add recommendations
            if safety_checks['warnings'] or safety_checks['critical_issues']:
                safety_checks['recommendations'].extend([
                    "Create a backup before migration",
                    "Schedule migration during low-traffic period",
                    "Monitor migration progress",
                    "Have rollback plan ready"
                ])
            
            return safety_checks
            
        except Exception as e:
            logger.error(f"Error validating migration safety: {e}")
            safety_checks['critical_issues'].append(f"Could not validate safety: {e}")
            safety_checks['safe_to_migrate'] = False
            return safety_checks
    
    def _get_database_size(self) -> Dict[str, Any]:
        """Get database size information"""
        
        try:
            query = text("""
                SELECT 
                    pg_size_pretty(pg_database_size(current_database())) as size_pretty,
                    pg_database_size(current_database()) / 1024 / 1024 as size_mb
            """)
            
            result = self.db.execute(query).fetchone()
            return {
                'size_pretty': result.size_pretty,
                'size_mb': float(result.size_mb)
            }
            
        except Exception:
            return {}
    
    def _get_active_connections(self) -> int:
        """Get number of active database connections"""
        
        try:
            query = text("""
                SELECT count(*) as active_connections
                FROM pg_stat_activity 
                WHERE state = 'active' AND pid != pg_backend_pid()
            """)
            
            result = self.db.execute(query).fetchone()
            return int(result.active_connections or 0)
            
        except Exception:
            return 0
    
    def _check_large_tables(self, table_names: List[str]) -> Dict[str, int]:
        """Check size of specific tables"""
        
        try:
            table_sizes = {}
            
            for table in table_names:
                query = text(f"""
                    SELECT count(*) as row_count 
                    FROM {table}
                """)
                
                try:
                    result = self.db.execute(query).fetchone()
                    table_sizes[table] = int(result.row_count or 0)
                except Exception:
                    # Table might not exist yet
                    table_sizes[table] = 0
            
            return table_sizes
            
        except Exception:
            return {}
    
    def _check_disk_space(self) -> Optional[float]:
        """Check available disk space"""
        
        try:
            import shutil
            free_bytes = shutil.disk_usage('.').free
            return free_bytes / 1024 / 1024  # Convert to MB
            
        except Exception:
            return None
    
    def _check_long_running_queries(self) -> List[Dict[str, Any]]:
        """Check for long-running queries"""
        
        try:
            query = text("""
                SELECT 
                    pid,
                    query,
                    state,
                    extract(epoch from (now() - query_start)) as duration_seconds
                FROM pg_stat_activity 
                WHERE state = 'active' 
                  AND pid != pg_backend_pid()
                  AND extract(epoch from (now() - query_start)) > 300
                ORDER BY duration_seconds DESC
            """)
            
            result = self.db.execute(query).fetchall()
            
            return [
                {
                    'pid': row.pid,
                    'query': row.query[:100] + "..." if len(row.query) > 100 else row.query,
                    'duration_seconds': float(row.duration_seconds)
                }
                for row in result
            ]
            
        except Exception:
            return []
    
    def run_migration_with_safety_checks(
        self, 
        target_revision: Optional[str] = "head",
        create_backup: bool = True
    ) -> Dict[str, Any]:
        """Run migration with comprehensive safety checks"""
        
        migration_log = {
            'started_at': datetime.now().isoformat(),
            'target_revision': target_revision,
            'success': False,
            'backup_created': None,
            'current_revision_before': None,
            'current_revision_after': None,
            'safety_checks': {},
            'error': None
        }
        
        try:
            # Get current revision
            migration_log['current_revision_before'] = self.get_current_migration_head()
            
            # Run safety checks
            safety_checks = self.validate_migration_safety(target_revision)
            migration_log['safety_checks'] = safety_checks
            
            if not safety_checks['safe_to_migrate']:
                migration_log['error'] = "Migration aborted due to safety concerns"
                return migration_log
            
            # Create backup if requested
            if create_backup:
                backup_file = self.create_backup(f"pre_migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                migration_log['backup_created'] = backup_file
            
            # Run migration
            cmd = ['alembic', '-c', self.alembic_config, 'upgrade', target_revision or 'head']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                migration_log['success'] = True
                migration_log['current_revision_after'] = self.get_current_migration_head()
                logger.info(f"Migration successful: {result.stdout}")
            else:
                migration_log['error'] = result.stderr
                logger.error(f"Migration failed: {result.stderr}")
            
        except Exception as e:
            migration_log['error'] = str(e)
            logger.error(f"Migration error: {e}")
        
        finally:
            migration_log['completed_at'] = datetime.now().isoformat()
        
        return migration_log
    
    def rollback_migration(self, target_revision: str) -> Dict[str, Any]:
        """Rollback to specific migration revision"""
        
        rollback_log = {
            'started_at': datetime.now().isoformat(),
            'target_revision': target_revision,
            'success': False,
            'current_revision_before': None,
            'current_revision_after': None,
            'error': None
        }
        
        try:
            # Get current revision
            rollback_log['current_revision_before'] = self.get_current_migration_head()
            
            # Confirm rollback target exists
            history = self.get_migration_history()
            valid_revisions = [h['revision'] for h in history]
            
            if target_revision not in valid_revisions:
                rollback_log['error'] = f"Invalid revision: {target_revision}"
                return rollback_log
            
            # Run rollback
            cmd = ['alembic', '-c', self.alembic_config, 'downgrade', target_revision]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                rollback_log['success'] = True
                rollback_log['current_revision_after'] = self.get_current_migration_head()
                logger.info(f"Rollback successful: {result.stdout}")
            else:
                rollback_log['error'] = result.stderr
                logger.error(f"Rollback failed: {result.stderr}")
            
        except Exception as e:
            rollback_log['error'] = str(e)
            logger.error(f"Rollback error: {e}")
        
        finally:
            rollback_log['completed_at'] = datetime.now().isoformat()
        
        return rollback_log
    
    def cleanup_old_backups(self, keep_days: int = 30):
        """Clean up old backup files"""
        
        try:
            from pathlib import Path
            import time
            
            current_time = time.time()
            cutoff_time = current_time - (keep_days * 24 * 60 * 60)
            
            removed_count = 0
            for backup_file in self.backup_dir.glob("*.sql"):
                if backup_file.stat().st_mtime < cutoff_time:
                    backup_file.unlink()
                    removed_count += 1
            
            logger.info(f"Cleaned up {removed_count} old backup files")
            return removed_count
            
        except Exception as e:
            logger.error(f"Error cleaning up backups: {e}")
            return 0