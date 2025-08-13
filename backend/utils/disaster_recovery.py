"""
Disaster Recovery Procedures and Automation
Comprehensive disaster recovery with automated backup, validation, and restoration
"""

import asyncio
import time
import json
import shutil
import gzip
import hashlib
import subprocess
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
from collections import deque, defaultdict
import threading
import logging
from pathlib import Path
import aiofiles
import tarfile
import boto3
from botocore.exceptions import ClientError
import redis
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from .enhanced_error_handling import with_error_handling, error_handler
from .service_health_manager import ServiceHealthManager, HealthStatus
from .exceptions import *

logger = logging.getLogger(__name__)


class DisasterType(Enum):
    """Types of disasters requiring recovery"""
    DATA_CORRUPTION = "data_corruption"
    DATABASE_FAILURE = "database_failure"
    CACHE_FAILURE = "cache_failure"
    SERVICE_FAILURE = "service_failure"
    NETWORK_PARTITION = "network_partition"
    STORAGE_FAILURE = "storage_failure"
    SECURITY_BREACH = "security_breach"
    CONFIGURATION_ERROR = "configuration_error"
    DEPENDENCY_FAILURE = "dependency_failure"
    CASCADING_FAILURE = "cascading_failure"


class RecoveryMode(Enum):
    """Recovery operation modes"""
    AUTOMATIC = "automatic"        # Fully automated recovery
    SEMI_AUTOMATIC = "semi_automatic"  # Automated with human approval
    MANUAL = "manual"             # Human-initiated recovery
    EMERGENCY = "emergency"       # Emergency recovery procedures


class BackupType(Enum):
    """Types of backups"""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    TRANSACTION_LOG = "transaction_log"
    CONFIGURATION = "configuration"
    APPLICATION_STATE = "application_state"


@dataclass
class BackupMetadata:
    """Metadata for backup operations"""
    backup_id: str
    backup_type: BackupType
    timestamp: datetime
    size_bytes: int
    checksum: str
    encryption_enabled: bool
    compression_enabled: bool
    retention_days: int
    source_location: str
    backup_location: str
    verification_status: str
    related_backups: List[str] = None
    
    def __post_init__(self):
        if self.related_backups is None:
            self.related_backups = []


@dataclass
class RecoveryPlan:
    """Disaster recovery plan configuration"""
    disaster_type: DisasterType
    recovery_mode: RecoveryMode
    priority: int  # 1=highest, 5=lowest
    estimated_rto_minutes: int  # Recovery Time Objective
    estimated_rpo_minutes: int  # Recovery Point Objective
    prerequisites: List[str]
    recovery_steps: List[Dict[str, Any]]
    validation_steps: List[Dict[str, Any]]
    rollback_steps: List[Dict[str, Any]]
    notification_contacts: List[str]
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class BackupManager:
    """Manages automated backup operations for disaster recovery"""
    
    def __init__(
        self,
        backup_root: str = "data/backups",
        s3_bucket: Optional[str] = None,
        retention_days: int = 30,
        compression_enabled: bool = True,
        encryption_enabled: bool = True
    ):
        self.backup_root = Path(backup_root)
        self.backup_root.mkdir(parents=True, exist_ok=True)
        
        self.s3_bucket = s3_bucket
        self.retention_days = retention_days
        self.compression_enabled = compression_enabled
        self.encryption_enabled = encryption_enabled
        
        # S3 client for cloud backups
        self.s3_client = boto3.client('s3') if s3_bucket else None
        
        # Backup registry
        self.backup_registry: Dict[str, BackupMetadata] = {}
        self.backup_schedule: Dict[str, Dict] = {}
        
        # Background tasks
        self._backup_tasks: List[asyncio.Task] = []
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.backup_stats = {
            'total_backups': 0,
            'successful_backups': 0,
            'failed_backups': 0,
            'total_size_gb': 0.0,
            'last_backup': None
        }
        
        # Thread safety
        self._lock = asyncio.Lock()
    
    async def register_backup_source(
        self,
        source_name: str,
        source_path: str,
        backup_type: BackupType,
        schedule_cron: str,
        retention_days: int = None,
        backup_func: Optional[Callable] = None
    ):
        """Register a data source for automated backup"""
        schedule_config = {
            'source_path': source_path,
            'backup_type': backup_type,
            'schedule_cron': schedule_cron,
            'retention_days': retention_days or self.retention_days,
            'backup_func': backup_func,
            'last_backup': None,
            'next_backup': self._calculate_next_backup(schedule_cron)
        }
        
        async with self._lock:
            self.backup_schedule[source_name] = schedule_config
        
        logger.info(f"Registered backup source: {source_name}")
    
    def _calculate_next_backup(self, cron_expr: str) -> datetime:
        """Calculate next backup time from cron expression"""
        # Simplified cron parsing - in production, use croniter library
        now = datetime.now()
        if cron_expr == "hourly":
            return now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        elif cron_expr == "daily":
            return now.replace(hour=2, minute=0, second=0, microsecond=0) + timedelta(days=1)
        elif cron_expr == "weekly":
            return now.replace(hour=2, minute=0, second=0, microsecond=0) + timedelta(weeks=1)
        else:
            # Default to daily
            return now.replace(hour=2, minute=0, second=0, microsecond=0) + timedelta(days=1)
    
    async def start_backup_scheduler(self):
        """Start automated backup scheduler"""
        if self._backup_tasks:
            logger.warning("Backup scheduler already running")
            return
        
        # Start backup monitoring task
        self._backup_tasks.append(
            asyncio.create_task(self._backup_scheduler_loop())
        )
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_old_backups())
        
        logger.info("Backup scheduler started")
    
    async def stop_backup_scheduler(self):
        """Stop automated backup scheduler"""
        # Cancel all tasks
        for task in self._backup_tasks:
            task.cancel()
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        # Wait for tasks to complete
        all_tasks = self._backup_tasks + ([self._cleanup_task] if self._cleanup_task else [])
        await asyncio.gather(*all_tasks, return_exceptions=True)
        
        self._backup_tasks.clear()
        self._cleanup_task = None
        
        logger.info("Backup scheduler stopped")
    
    async def _backup_scheduler_loop(self):
        """Main backup scheduling loop"""
        while True:
            try:
                current_time = datetime.now()
                
                # Check which sources need backup
                sources_to_backup = []
                async with self._lock:
                    for source_name, config in self.backup_schedule.items():
                        if current_time >= config['next_backup']:
                            sources_to_backup.append((source_name, config))
                
                # Execute backups
                for source_name, config in sources_to_backup:
                    try:
                        await self._execute_scheduled_backup(source_name, config)
                    except Exception as e:
                        logger.error(f"Scheduled backup failed for {source_name}: {e}")
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Backup scheduler error: {e}")
                await asyncio.sleep(60)
    
    async def _execute_scheduled_backup(self, source_name: str, config: Dict):
        """Execute a scheduled backup"""
        logger.info(f"Starting scheduled backup for: {source_name}")
        
        try:
            if config['backup_func']:
                # Use custom backup function
                backup_metadata = await config['backup_func'](source_name, config)
            else:
                # Use default backup method
                backup_metadata = await self._default_backup(source_name, config)
            
            # Update schedule
            async with self._lock:
                config['last_backup'] = datetime.now()
                config['next_backup'] = self._calculate_next_backup(config['schedule_cron'])
            
            # Update stats
            self.backup_stats['total_backups'] += 1
            self.backup_stats['successful_backups'] += 1
            self.backup_stats['last_backup'] = datetime.now()
            
            logger.info(f"Scheduled backup completed for: {source_name}")
            
        except Exception as e:
            self.backup_stats['failed_backups'] += 1
            logger.error(f"Scheduled backup failed for {source_name}: {e}")
            raise
    
    @with_error_handling(service="backup_manager", operation="default_backup")
    async def _default_backup(self, source_name: str, config: Dict) -> BackupMetadata:
        """Default backup implementation"""
        source_path = Path(config['source_path'])
        backup_type = config['backup_type']
        
        if not source_path.exists():
            raise FileNotFoundError(f"Backup source not found: {source_path}")
        
        # Generate backup ID and paths
        backup_id = f"{source_name}_{backup_type.value}_{int(time.time())}"
        backup_filename = f"{backup_id}.tar"
        
        if self.compression_enabled:
            backup_filename += ".gz"
        
        backup_path = self.backup_root / backup_filename
        
        # Create backup archive
        start_time = time.time()
        
        if source_path.is_file():
            # Single file backup
            if self.compression_enabled:
                with gzip.open(backup_path, 'wb') as gz_file:
                    with open(source_path, 'rb') as src_file:
                        shutil.copyfileobj(src_file, gz_file)
            else:
                shutil.copy2(source_path, backup_path)
        else:
            # Directory backup
            mode = 'w:gz' if self.compression_enabled else 'w'
            with tarfile.open(backup_path, mode) as tar:
                tar.add(source_path, arcname=source_path.name)
        
        backup_time = time.time() - start_time
        backup_size = backup_path.stat().st_size
        
        # Calculate checksum
        checksum = await self._calculate_checksum(backup_path)
        
        # Create metadata
        backup_metadata = BackupMetadata(
            backup_id=backup_id,
            backup_type=backup_type,
            timestamp=datetime.now(),
            size_bytes=backup_size,
            checksum=checksum,
            encryption_enabled=self.encryption_enabled,
            compression_enabled=self.compression_enabled,
            retention_days=config['retention_days'],
            source_location=str(source_path),
            backup_location=str(backup_path),
            verification_status="pending"
        )
        
        # Verify backup
        verification_result = await self._verify_backup(backup_metadata)
        backup_metadata.verification_status = "verified" if verification_result else "failed"
        
        # Upload to cloud if configured
        if self.s3_client and self.s3_bucket:
            cloud_key = f"backups/{backup_filename}"
            await self._upload_to_s3(backup_path, cloud_key)
            backup_metadata.backup_location = f"s3://{self.s3_bucket}/{cloud_key}"
        
        # Register backup
        async with self._lock:
            self.backup_registry[backup_id] = backup_metadata
        
        # Save backup registry
        await self._save_backup_registry()
        
        logger.info(
            f"Backup completed: {backup_id}, size: {backup_size / 1024**2:.1f}MB, "
            f"time: {backup_time:.1f}s"
        )
        
        return backup_metadata
    
    async def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file"""
        sha256_hash = hashlib.sha256()
        
        async with aiofiles.open(file_path, 'rb') as f:
            async for chunk in f:
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    async def _verify_backup(self, backup_metadata: BackupMetadata) -> bool:
        """Verify backup integrity"""
        try:
            backup_path = Path(backup_metadata.backup_location)
            
            if not backup_path.exists():
                logger.error(f"Backup file not found: {backup_path}")
                return False
            
            # Verify file size
            actual_size = backup_path.stat().st_size
            if actual_size != backup_metadata.size_bytes:
                logger.error(f"Backup size mismatch: expected {backup_metadata.size_bytes}, got {actual_size}")
                return False
            
            # Verify checksum
            actual_checksum = await self._calculate_checksum(backup_path)
            if actual_checksum != backup_metadata.checksum:
                logger.error(f"Backup checksum mismatch")
                return False
            
            # Verify archive integrity (for compressed backups)
            if backup_metadata.compression_enabled:
                try:
                    if backup_path.suffix == '.gz' and backup_path.stem.endswith('.tar'):
                        with tarfile.open(backup_path, 'r:gz') as tar:
                            # Just try to list contents to verify integrity
                            list(tar.getnames())
                    elif backup_path.suffix == '.gz':
                        with gzip.open(backup_path, 'rb') as gz_file:
                            # Read small amount to verify it's not corrupted
                            gz_file.read(1024)
                except Exception as e:
                    logger.error(f"Backup archive integrity check failed: {e}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Backup verification failed: {e}")
            return False
    
    async def _upload_to_s3(self, local_path: Path, s3_key: str):
        """Upload backup to S3"""
        try:
            with open(local_path, 'rb') as f:
                self.s3_client.upload_fileobj(f, self.s3_bucket, s3_key)
            
            logger.info(f"Backup uploaded to S3: {s3_key}")
            
        except ClientError as e:
            logger.error(f"S3 upload failed: {e}")
            raise
    
    async def _cleanup_old_backups(self):
        """Cleanup old backups based on retention policy"""
        while True:
            try:
                await asyncio.sleep(24 * 3600)  # Run daily
                
                current_time = datetime.now()
                expired_backups = []
                
                async with self._lock:
                    for backup_id, metadata in self.backup_registry.items():
                        age_days = (current_time - metadata.timestamp).days
                        if age_days > metadata.retention_days:
                            expired_backups.append((backup_id, metadata))
                
                # Delete expired backups
                for backup_id, metadata in expired_backups:
                    try:
                        await self._delete_backup(backup_id)
                        logger.info(f"Deleted expired backup: {backup_id}")
                    except Exception as e:
                        logger.error(f"Failed to delete expired backup {backup_id}: {e}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Backup cleanup error: {e}")
    
    async def _delete_backup(self, backup_id: str):
        """Delete a backup and its metadata"""
        async with self._lock:
            if backup_id not in self.backup_registry:
                return
            
            metadata = self.backup_registry[backup_id]
            
            # Delete local file
            local_path = Path(metadata.backup_location)
            if local_path.exists():
                local_path.unlink()
            
            # Delete from S3 if applicable
            if metadata.backup_location.startswith('s3://'):
                s3_key = metadata.backup_location.replace(f's3://{self.s3_bucket}/', '')
                try:
                    self.s3_client.delete_object(Bucket=self.s3_bucket, Key=s3_key)
                except ClientError as e:
                    logger.warning(f"Failed to delete S3 backup: {e}")
            
            # Remove from registry
            del self.backup_registry[backup_id]
        
        await self._save_backup_registry()
    
    async def _save_backup_registry(self):
        """Save backup registry to disk"""
        try:
            registry_file = self.backup_root / "backup_registry.json"
            registry_data = {
                backup_id: asdict(metadata)
                for backup_id, metadata in self.backup_registry.items()
            }
            
            async with aiofiles.open(registry_file, 'w') as f:
                await f.write(json.dumps(registry_data, indent=2, default=str))
                
        except Exception as e:
            logger.error(f"Failed to save backup registry: {e}")
    
    async def load_backup_registry(self):
        """Load backup registry from disk"""
        try:
            registry_file = self.backup_root / "backup_registry.json"
            if not registry_file.exists():
                return
            
            async with aiofiles.open(registry_file, 'r') as f:
                content = await f.read()
                registry_data = json.loads(content)
            
            async with self._lock:
                for backup_id, metadata_dict in registry_data.items():
                    metadata_dict['timestamp'] = datetime.fromisoformat(metadata_dict['timestamp'])
                    metadata_dict['backup_type'] = BackupType(metadata_dict['backup_type'])
                    self.backup_registry[backup_id] = BackupMetadata(**metadata_dict)
            
            logger.info(f"Loaded {len(self.backup_registry)} backup records")
            
        except Exception as e:
            logger.error(f"Failed to load backup registry: {e}")
    
    async def create_manual_backup(
        self,
        source_name: str,
        source_path: str,
        backup_type: BackupType = BackupType.FULL,
        retention_days: int = None
    ) -> BackupMetadata:
        """Create backup manually"""
        config = {
            'source_path': source_path,
            'backup_type': backup_type,
            'retention_days': retention_days or self.retention_days
        }
        
        return await self._default_backup(source_name, config)
    
    def get_backup_status(self) -> Dict[str, Any]:
        """Get backup system status"""
        return {
            'backup_sources': len(self.backup_schedule),
            'total_backups': len(self.backup_registry),
            'backup_stats': self.backup_stats,
            'storage_location': str(self.backup_root),
            'cloud_storage': self.s3_bucket,
            'retention_days': self.retention_days,
            'compression_enabled': self.compression_enabled,
            'encryption_enabled': self.encryption_enabled
        }


class DisasterRecoveryOrchestrator:
    """
    Orchestrates disaster recovery procedures with automated decision making
    """
    
    def __init__(
        self,
        backup_manager: BackupManager,
        health_manager: ServiceHealthManager
    ):
        self.backup_manager = backup_manager
        self.health_manager = health_manager
        
        # Recovery plans registry
        self.recovery_plans: Dict[DisasterType, RecoveryPlan] = {}
        self.custom_recovery_handlers: Dict[str, Callable] = {}
        
        # Active recovery operations
        self.active_recoveries: Dict[str, Dict] = {}
        self.recovery_history: deque = deque(maxlen=1000)
        
        # Configuration
        self.auto_recovery_enabled = True
        self.emergency_contacts = []
        
        # Thread safety
        self._lock = asyncio.Lock()
        
        # Load default recovery plans
        self._load_default_recovery_plans()
    
    def _load_default_recovery_plans(self):
        """Load default disaster recovery plans"""
        
        # Database failure recovery plan
        db_recovery = RecoveryPlan(
            disaster_type=DisasterType.DATABASE_FAILURE,
            recovery_mode=RecoveryMode.AUTOMATIC,
            priority=1,
            estimated_rto_minutes=15,
            estimated_rpo_minutes=5,
            prerequisites=["backup_available", "database_service_stopped"],
            recovery_steps=[
                {"step": "stop_database_connections", "timeout": 30},
                {"step": "restore_from_backup", "timeout": 300},
                {"step": "verify_data_integrity", "timeout": 60},
                {"step": "restart_database_service", "timeout": 60},
                {"step": "run_health_checks", "timeout": 30}
            ],
            validation_steps=[
                {"step": "verify_database_connectivity", "timeout": 30},
                {"step": "verify_data_consistency", "timeout": 60},
                {"step": "run_application_tests", "timeout": 120}
            ],
            rollback_steps=[
                {"step": "stop_database_service", "timeout": 30},
                {"step": "restore_previous_backup", "timeout": 300},
                {"step": "restart_database_service", "timeout": 60}
            ],
            notification_contacts=["ops-team@company.com", "dba@company.com"]
        )
        
        # Cache failure recovery plan
        cache_recovery = RecoveryPlan(
            disaster_type=DisasterType.CACHE_FAILURE,
            recovery_mode=RecoveryMode.AUTOMATIC,
            priority=2,
            estimated_rto_minutes=5,
            estimated_rpo_minutes=0,
            prerequisites=["cache_service_accessible"],
            recovery_steps=[
                {"step": "clear_corrupted_cache", "timeout": 30},
                {"step": "restart_cache_service", "timeout": 60},
                {"step": "warm_cache_with_critical_data", "timeout": 120}
            ],
            validation_steps=[
                {"step": "verify_cache_connectivity", "timeout": 30},
                {"step": "verify_cache_functionality", "timeout": 30}
            ],
            rollback_steps=[
                {"step": "disable_cache", "timeout": 10},
                {"step": "run_without_cache", "timeout": 0}
            ],
            notification_contacts=["ops-team@company.com"]
        )
        
        # Service failure recovery plan
        service_recovery = RecoveryPlan(
            disaster_type=DisasterType.SERVICE_FAILURE,
            recovery_mode=RecoveryMode.SEMI_AUTOMATIC,
            priority=1,
            estimated_rto_minutes=10,
            estimated_rpo_minutes=1,
            prerequisites=["service_logs_available", "health_checks_failing"],
            recovery_steps=[
                {"step": "analyze_failure_cause", "timeout": 60},
                {"step": "restart_service", "timeout": 120},
                {"step": "verify_service_health", "timeout": 60},
                {"step": "restore_from_checkpoint", "timeout": 180, "condition": "restart_failed"}
            ],
            validation_steps=[
                {"step": "run_health_checks", "timeout": 60},
                {"step": "verify_api_endpoints", "timeout": 30},
                {"step": "check_dependent_services", "timeout": 60}
            ],
            rollback_steps=[
                {"step": "stop_service", "timeout": 30},
                {"step": "restore_previous_version", "timeout": 180},
                {"step": "restart_service", "timeout": 120}
            ],
            notification_contacts=["ops-team@company.com", "dev-team@company.com"]
        )
        
        # Data corruption recovery plan
        data_corruption_recovery = RecoveryPlan(
            disaster_type=DisasterType.DATA_CORRUPTION,
            recovery_mode=RecoveryMode.MANUAL,
            priority=1,
            estimated_rto_minutes=60,
            estimated_rpo_minutes=15,
            prerequisites=["backup_verified", "corruption_scope_identified"],
            recovery_steps=[
                {"step": "stop_write_operations", "timeout": 30},
                {"step": "create_corruption_snapshot", "timeout": 60},
                {"step": "identify_last_good_backup", "timeout": 30},
                {"step": "restore_from_backup", "timeout": 600},
                {"step": "verify_data_integrity", "timeout": 180},
                {"step": "replay_transactions", "timeout": 300}
            ],
            validation_steps=[
                {"step": "run_data_consistency_checks", "timeout": 300},
                {"step": "verify_business_logic", "timeout": 180},
                {"step": "run_integration_tests", "timeout": 300}
            ],
            rollback_steps=[
                {"step": "restore_corruption_snapshot", "timeout": 600},
                {"step": "analyze_alternative_recovery", "timeout": 0}
            ],
            notification_contacts=["ops-team@company.com", "data-team@company.com", "management@company.com"]
        )
        
        self.recovery_plans.update({
            DisasterType.DATABASE_FAILURE: db_recovery,
            DisasterType.CACHE_FAILURE: cache_recovery,
            DisasterType.SERVICE_FAILURE: service_recovery,
            DisasterType.DATA_CORRUPTION: data_corruption_recovery
        })
    
    async def detect_disaster(self) -> Optional[DisasterType]:
        """Detect potential disaster scenarios based on system health"""
        
        health_status = self.health_manager.get_health_status()
        
        # Database failure detection
        db_dependencies = [
            dep for dep_name, dep_metrics in health_status['dependency_health'].items()
            if 'database' in dep_name.lower()
        ]
        
        if any(dep['status'] in ['critical', 'unhealthy'] for dep in db_dependencies):
            return DisasterType.DATABASE_FAILURE
        
        # Cache failure detection  
        cache_dependencies = [
            dep for dep_name, dep_metrics in health_status['dependency_health'].items()
            if 'cache' in dep_name.lower() or 'redis' in dep_name.lower()
        ]
        
        if any(dep['status'] in ['critical', 'unhealthy'] for dep in cache_dependencies):
            return DisasterType.CACHE_FAILURE
        
        # Service failure detection
        if health_status['overall_status'] == 'critical':
            return DisasterType.SERVICE_FAILURE
        
        # Data corruption detection (would need additional logic)
        # This would integrate with data quality monitoring
        
        return None
    
    async def initiate_recovery(
        self,
        disaster_type: DisasterType,
        recovery_mode: RecoveryMode = None,
        manual_approval: bool = False
    ) -> str:
        """Initiate disaster recovery procedure"""
        
        if disaster_type not in self.recovery_plans:
            raise ValueError(f"No recovery plan found for disaster type: {disaster_type}")
        
        recovery_plan = self.recovery_plans[disaster_type]
        recovery_id = f"recovery_{disaster_type.value}_{int(time.time())}"
        
        # Override recovery mode if specified
        if recovery_mode:
            recovery_plan.recovery_mode = recovery_mode
        
        # Check if manual approval is required
        if recovery_plan.recovery_mode == RecoveryMode.SEMI_AUTOMATIC and not manual_approval:
            logger.warning(f"Recovery {recovery_id} requires manual approval")
            await self._send_approval_request(recovery_id, recovery_plan)
            return recovery_id
        
        # Start recovery operation
        recovery_operation = {
            'recovery_id': recovery_id,
            'disaster_type': disaster_type,
            'recovery_plan': recovery_plan,
            'start_time': datetime.now(),
            'status': 'in_progress',
            'current_step': 0,
            'steps_completed': 0,
            'steps_failed': 0,
            'error_messages': [],
            'estimated_completion': datetime.now() + timedelta(minutes=recovery_plan.estimated_rto_minutes)
        }
        
        async with self._lock:
            self.active_recoveries[recovery_id] = recovery_operation
        
        # Execute recovery in background
        asyncio.create_task(self._execute_recovery(recovery_id))
        
        # Send notifications
        await self._notify_recovery_start(recovery_id, recovery_plan)
        
        logger.critical(f"Initiated disaster recovery: {recovery_id} for {disaster_type.value}")
        return recovery_id
    
    async def _execute_recovery(self, recovery_id: str):
        """Execute disaster recovery steps"""
        try:
            async with self._lock:
                operation = self.active_recoveries[recovery_id]
            
            recovery_plan = operation['recovery_plan']
            
            logger.info(f"Starting recovery execution: {recovery_id}")
            
            # Check prerequisites
            prerequisites_met = await self._check_prerequisites(recovery_plan.prerequisites)
            if not prerequisites_met:
                await self._fail_recovery(recovery_id, "Prerequisites not met")
                return
            
            # Execute recovery steps
            for i, step_config in enumerate(recovery_plan.recovery_steps):
                operation['current_step'] = i
                
                logger.info(f"Recovery {recovery_id}: Executing step {i+1}/{len(recovery_plan.recovery_steps)}: {step_config['step']}")
                
                try:
                    success = await self._execute_recovery_step(step_config)
                    
                    if success:
                        operation['steps_completed'] += 1
                    else:
                        operation['steps_failed'] += 1
                        
                        # Check if step is optional
                        if not step_config.get('optional', False):
                            await self._fail_recovery(recovery_id, f"Critical step failed: {step_config['step']}")
                            return
                
                except Exception as e:
                    operation['steps_failed'] += 1
                    operation['error_messages'].append(f"Step {step_config['step']}: {str(e)}")
                    
                    if not step_config.get('optional', False):
                        await self._fail_recovery(recovery_id, f"Step execution error: {str(e)}")
                        return
            
            # Execute validation steps
            logger.info(f"Recovery {recovery_id}: Starting validation phase")
            
            validation_passed = True
            for step_config in recovery_plan.validation_steps:
                try:
                    success = await self._execute_validation_step(step_config)
                    if not success:
                        validation_passed = False
                        operation['error_messages'].append(f"Validation failed: {step_config['step']}")
                        break
                        
                except Exception as e:
                    validation_passed = False
                    operation['error_messages'].append(f"Validation error: {str(e)}")
                    break
            
            if validation_passed:
                await self._complete_recovery(recovery_id)
            else:
                await self._rollback_recovery(recovery_id)
                
        except Exception as e:
            logger.error(f"Recovery execution error: {e}")
            await self._fail_recovery(recovery_id, f"Execution error: {str(e)}")
    
    async def _execute_recovery_step(self, step_config: Dict) -> bool:
        """Execute a single recovery step"""
        step_name = step_config['step']
        timeout = step_config.get('timeout', 60)
        condition = step_config.get('condition')
        
        # Skip step if condition not met
        if condition and not await self._evaluate_condition(condition):
            logger.info(f"Skipping step {step_name} due to condition: {condition}")
            return True
        
        # Execute step with timeout
        try:
            success = await asyncio.wait_for(
                self._execute_step_handler(step_name),
                timeout=timeout
            )
            return success
            
        except asyncio.TimeoutError:
            logger.error(f"Recovery step timeout: {step_name}")
            return False
        except Exception as e:
            logger.error(f"Recovery step error {step_name}: {e}")
            return False
    
    async def _execute_step_handler(self, step_name: str) -> bool:
        """Execute specific recovery step handler"""
        
        if step_name in self.custom_recovery_handlers:
            return await self.custom_recovery_handlers[step_name]()
        
        # Built-in step handlers
        if step_name == "stop_database_connections":
            return await self._stop_database_connections()
        elif step_name == "restore_from_backup":
            return await self._restore_from_backup()
        elif step_name == "verify_data_integrity":
            return await self._verify_data_integrity()
        elif step_name == "restart_database_service":
            return await self._restart_database_service()
        elif step_name == "clear_corrupted_cache":
            return await self._clear_corrupted_cache()
        elif step_name == "restart_cache_service":
            return await self._restart_cache_service()
        elif step_name == "warm_cache_with_critical_data":
            return await self._warm_cache()
        elif step_name == "restart_service":
            return await self._restart_service()
        elif step_name == "run_health_checks":
            return await self._run_health_checks()
        else:
            logger.warning(f"Unknown recovery step: {step_name}")
            return False
    
    async def _execute_validation_step(self, step_config: Dict) -> bool:
        """Execute validation step"""
        step_name = step_config['step']
        timeout = step_config.get('timeout', 30)
        
        try:
            return await asyncio.wait_for(
                self._execute_validation_handler(step_name),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"Validation step timeout: {step_name}")
            return False
        except Exception as e:
            logger.error(f"Validation step error {step_name}: {e}")
            return False
    
    async def _execute_validation_handler(self, step_name: str) -> bool:
        """Execute specific validation step handler"""
        
        if step_name == "verify_database_connectivity":
            return await self._verify_database_connectivity()
        elif step_name == "verify_cache_connectivity":
            return await self._verify_cache_connectivity()
        elif step_name == "verify_service_health":
            return await self._verify_service_health()
        elif step_name == "run_integration_tests":
            return await self._run_integration_tests()
        else:
            logger.warning(f"Unknown validation step: {step_name}")
            return False
    
    # Recovery step implementations
    async def _stop_database_connections(self) -> bool:
        """Stop all database connections"""
        logger.info("Stopping database connections")
        # Implementation would terminate active connections
        await asyncio.sleep(1)  # Simulate operation
        return True
    
    async def _restore_from_backup(self) -> bool:
        """Restore database from latest backup"""
        logger.info("Restoring from backup")
        
        # Find latest database backup
        db_backups = [
            backup for backup in self.backup_manager.backup_registry.values()
            if 'database' in backup.source_location.lower()
        ]
        
        if not db_backups:
            logger.error("No database backups found")
            return False
        
        latest_backup = max(db_backups, key=lambda b: b.timestamp)
        logger.info(f"Using backup: {latest_backup.backup_id}")
        
        # Simulate restore operation
        await asyncio.sleep(5)
        return True
    
    async def _verify_data_integrity(self) -> bool:
        """Verify database integrity after restore"""
        logger.info("Verifying data integrity")
        await asyncio.sleep(2)  # Simulate verification
        return True
    
    async def _restart_database_service(self) -> bool:
        """Restart database service"""
        logger.info("Restarting database service")
        await asyncio.sleep(3)  # Simulate restart
        return True
    
    async def _clear_corrupted_cache(self) -> bool:
        """Clear corrupted cache data"""
        logger.info("Clearing corrupted cache")
        # Implementation would clear cache
        await asyncio.sleep(1)
        return True
    
    async def _restart_cache_service(self) -> bool:
        """Restart cache service"""
        logger.info("Restarting cache service")
        await asyncio.sleep(2)  # Simulate restart
        return True
    
    async def _warm_cache(self) -> bool:
        """Warm cache with critical data"""
        logger.info("Warming cache with critical data")
        await asyncio.sleep(3)  # Simulate cache warming
        return True
    
    async def _restart_service(self) -> bool:
        """Restart main service"""
        logger.info("Restarting main service")
        await asyncio.sleep(5)  # Simulate service restart
        return True
    
    async def _run_health_checks(self) -> bool:
        """Run system health checks"""
        logger.info("Running health checks")
        health_status = self.health_manager.get_health_status()
        return health_status['overall_status'] in ['healthy', 'degraded']
    
    # Validation step implementations
    async def _verify_database_connectivity(self) -> bool:
        """Verify database is accessible"""
        logger.info("Verifying database connectivity")
        await asyncio.sleep(1)
        return True  # Simulate successful verification
    
    async def _verify_cache_connectivity(self) -> bool:
        """Verify cache is accessible"""
        logger.info("Verifying cache connectivity")
        await asyncio.sleep(1)
        return True
    
    async def _verify_service_health(self) -> bool:
        """Verify service health"""
        return await self._run_health_checks()
    
    async def _run_integration_tests(self) -> bool:
        """Run integration tests"""
        logger.info("Running integration tests")
        await asyncio.sleep(10)  # Simulate test execution
        return True  # Simulate successful tests
    
    async def _check_prerequisites(self, prerequisites: List[str]) -> bool:
        """Check if recovery prerequisites are met"""
        for prerequisite in prerequisites:
            if not await self._check_prerequisite(prerequisite):
                logger.error(f"Prerequisite not met: {prerequisite}")
                return False
        return True
    
    async def _check_prerequisite(self, prerequisite: str) -> bool:
        """Check individual prerequisite"""
        if prerequisite == "backup_available":
            return len(self.backup_manager.backup_registry) > 0
        elif prerequisite == "database_service_stopped":
            return True  # Simulate check
        elif prerequisite == "cache_service_accessible":
            return True  # Simulate check
        else:
            logger.warning(f"Unknown prerequisite: {prerequisite}")
            return True
    
    async def _evaluate_condition(self, condition: str) -> bool:
        """Evaluate recovery step condition"""
        if condition == "restart_failed":
            return False  # Simulate condition evaluation
        else:
            return True
    
    async def _complete_recovery(self, recovery_id: str):
        """Complete successful recovery"""
        async with self._lock:
            operation = self.active_recoveries[recovery_id]
            operation['status'] = 'completed'
            operation['completion_time'] = datetime.now()
        
        # Add to history
        self.recovery_history.append(operation)
        
        # Clean up active recoveries
        async with self._lock:
            del self.active_recoveries[recovery_id]
        
        # Send notifications
        await self._notify_recovery_completion(recovery_id, True)
        
        logger.info(f"Recovery completed successfully: {recovery_id}")
    
    async def _fail_recovery(self, recovery_id: str, reason: str):
        """Mark recovery as failed"""
        async with self._lock:
            operation = self.active_recoveries[recovery_id]
            operation['status'] = 'failed'
            operation['failure_reason'] = reason
            operation['completion_time'] = datetime.now()
        
        # Add to history
        self.recovery_history.append(operation)
        
        # Clean up active recoveries
        async with self._lock:
            del self.active_recoveries[recovery_id]
        
        # Send notifications
        await self._notify_recovery_completion(recovery_id, False, reason)
        
        logger.error(f"Recovery failed: {recovery_id}, reason: {reason}")
    
    async def _rollback_recovery(self, recovery_id: str):
        """Rollback failed recovery"""
        logger.warning(f"Rolling back recovery: {recovery_id}")
        
        async with self._lock:
            operation = self.active_recoveries[recovery_id]
        
        recovery_plan = operation['recovery_plan']
        
        # Execute rollback steps
        for step_config in recovery_plan.rollback_steps:
            try:
                await self._execute_recovery_step(step_config)
            except Exception as e:
                logger.error(f"Rollback step failed: {step_config['step']}: {e}")
        
        await self._fail_recovery(recovery_id, "Recovery validation failed, rolled back")
    
    async def _send_approval_request(self, recovery_id: str, recovery_plan: RecoveryPlan):
        """Send approval request for semi-automatic recovery"""
        approval_data = {
            'recovery_id': recovery_id,
            'disaster_type': recovery_plan.disaster_type.value,
            'estimated_rto_minutes': recovery_plan.estimated_rto_minutes,
            'estimated_rpo_minutes': recovery_plan.estimated_rpo_minutes,
            'recovery_steps': len(recovery_plan.recovery_steps),
            'contacts': recovery_plan.notification_contacts
        }
        
        logger.critical(f"APPROVAL REQUIRED: Disaster recovery {recovery_id}")
        logger.critical(f"Recovery details: {json.dumps(approval_data, indent=2)}")
        
        # In production, this would integrate with approval system
    
    async def _notify_recovery_start(self, recovery_id: str, recovery_plan: RecoveryPlan):
        """Send notifications about recovery start"""
        message = f"Disaster recovery initiated: {recovery_id} for {recovery_plan.disaster_type.value}"
        logger.critical(message)
        
        # In production, this would send notifications to the contacts
        for contact in recovery_plan.notification_contacts:
            logger.info(f"Notifying {contact}: {message}")
    
    async def _notify_recovery_completion(self, recovery_id: str, success: bool, reason: str = None):
        """Send notifications about recovery completion"""
        status = "SUCCESS" if success else "FAILED"
        message = f"Disaster recovery {status}: {recovery_id}"
        
        if not success and reason:
            message += f" - {reason}"
        
        logger.critical(message)
    
    def register_recovery_handler(self, step_name: str, handler_func: Callable):
        """Register custom recovery step handler"""
        self.custom_recovery_handlers[step_name] = handler_func
        logger.info(f"Registered custom recovery handler: {step_name}")
    
    def get_recovery_status(self, recovery_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific recovery operation"""
        return self.active_recoveries.get(recovery_id)
    
    def get_all_recovery_status(self) -> Dict[str, Any]:
        """Get status of all recovery operations"""
        return {
            'active_recoveries': len(self.active_recoveries),
            'recovery_history': len(self.recovery_history),
            'auto_recovery_enabled': self.auto_recovery_enabled,
            'available_plans': list(self.recovery_plans.keys()),
            'active_operations': {
                recovery_id: {
                    'disaster_type': op['disaster_type'].value,
                    'status': op['status'],
                    'progress': f"{op['steps_completed']}/{op['steps_completed'] + op['steps_failed']}",
                    'start_time': op['start_time'].isoformat(),
                    'estimated_completion': op['estimated_completion'].isoformat()
                }
                for recovery_id, op in self.active_recoveries.items()
            }
        }


# Integration class for complete disaster recovery system
class DisasterRecoverySystem:
    """Complete disaster recovery system integration"""
    
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        
        # Initialize components
        self.backup_manager = BackupManager(
            backup_root=config.get('backup_root', 'data/backups'),
            s3_bucket=config.get('s3_bucket'),
            retention_days=config.get('retention_days', 30)
        )
        
        # Health manager would be passed from main application
        self.health_manager = None  # To be set by application
        
        self.orchestrator = DisasterRecoveryOrchestrator(
            self.backup_manager,
            self.health_manager
        )
        
        # Monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        self._disaster_monitoring_enabled = config.get('disaster_monitoring', True)
    
    async def initialize(self):
        """Initialize disaster recovery system"""
        # Load backup registry
        await self.backup_manager.load_backup_registry()
        
        # Start backup scheduler
        await self.backup_manager.start_backup_scheduler()
        
        # Start disaster monitoring if enabled
        if self._disaster_monitoring_enabled:
            self._monitoring_task = asyncio.create_task(self._disaster_monitoring_loop())
        
        logger.info("Disaster recovery system initialized")
    
    async def shutdown(self):
        """Shutdown disaster recovery system"""
        # Stop backup scheduler
        await self.backup_manager.stop_backup_scheduler()
        
        # Stop disaster monitoring
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Disaster recovery system shutdown complete")
    
    async def _disaster_monitoring_loop(self):
        """Monitor for disaster scenarios and auto-recover"""
        while True:
            try:
                if self.health_manager:
                    disaster_type = await self.orchestrator.detect_disaster()
                    
                    if disaster_type:
                        logger.warning(f"Disaster detected: {disaster_type.value}")
                        
                        if self.orchestrator.auto_recovery_enabled:
                            recovery_id = await self.orchestrator.initiate_recovery(disaster_type)
                            logger.info(f"Auto-recovery initiated: {recovery_id}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Disaster monitoring error: {e}")
                await asyncio.sleep(30)
    
    def set_health_manager(self, health_manager: ServiceHealthManager):
        """Set health manager for disaster detection"""
        self.health_manager = health_manager
        self.orchestrator.health_manager = health_manager
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get complete disaster recovery system status"""
        return {
            'backup_system': self.backup_manager.get_backup_status(),
            'recovery_system': self.orchestrator.get_all_recovery_status(),
            'disaster_monitoring_enabled': self._disaster_monitoring_enabled,
            'system_initialized': self.health_manager is not None
        }


# Global disaster recovery system instance
disaster_recovery = DisasterRecoverySystem()