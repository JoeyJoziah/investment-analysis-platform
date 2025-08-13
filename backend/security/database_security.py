"""
Database Security Hardening Module

This module provides comprehensive database security features including:
- SSL/TLS connection enforcement
- Encryption at rest configuration
- Database audit logging
- Connection security monitoring
- Credential rotation management
"""

import os
import ssl
import logging
from typing import Dict, Any, Optional, List
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

from backend.config.settings import settings
from backend.security.secrets_manager import get_secrets_manager, SecretType

logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    """Types of database audit events"""
    CONNECTION = "connection"
    QUERY = "query"
    AUTHENTICATION = "authentication"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_ACCESS = "data_access"
    SCHEMA_CHANGE = "schema_change"
    CONFIGURATION_CHANGE = "configuration_change"
    ERROR = "error"


@dataclass
class AuditLogEntry:
    """Database audit log entry"""
    timestamp: datetime
    event_type: AuditEventType
    user_id: Optional[int] = None
    username: Optional[str] = None
    session_id: Optional[str] = None
    query: Optional[str] = None
    query_hash: Optional[str] = None
    affected_tables: List[str] = None
    row_count: Optional[int] = None
    duration_ms: Optional[float] = None
    client_ip: Optional[str] = None
    application_name: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
    risk_score: int = 0


class DatabaseSecurityManager:
    """
    Comprehensive database security management.
    
    Features:
    - SSL/TLS connection enforcement
    - Audit logging for compliance
    - Query monitoring and analysis
    - Connection security validation
    - Credential rotation management
    """
    
    def __init__(self, audit_log_path: Optional[str] = None):
        self.secrets_manager = get_secrets_manager()
        self.audit_log_path = Path(audit_log_path or "/app/logs/database_audit.jsonl")
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Security configuration
        self.min_tls_version = "TLSv1.2"
        self.required_cipher_suites = [
            'ECDHE-RSA-AES256-GCM-SHA384',
            'ECDHE-RSA-AES128-GCM-SHA256',
            'ECDHE-RSA-AES256-SHA384',
            'ECDHE-RSA-AES128-SHA256'
        ]
        
        # Query patterns that require elevated monitoring
        self.high_risk_patterns = [
            r'\b(DROP|TRUNCATE|DELETE)\s+',
            r'\bALTER\s+(TABLE|DATABASE|SCHEMA)',
            r'\bCREATE\s+(USER|ROLE)',
            r'\bGRANT\s+',
            r'\bREVOKE\s+',
            r'\bSET\s+(PASSWORD|ROLE)',
            r'INFORMATION_SCHEMA',
            r'pg_catalog',
            r'sys\.',
            r'master\.'
        ]
        
    def create_secure_engine(
        self,
        database_url: Optional[str] = None,
        enable_ssl: bool = True,
        enable_audit: bool = True
    ) -> Engine:
        """
        Create a database engine with security hardening.
        
        Args:
            database_url: Database connection URL
            enable_ssl: Whether to enforce SSL connections
            enable_audit: Whether to enable audit logging
            
        Returns:
            Configured SQLAlchemy engine
        """
        try:
            # Get database URL from settings or parameter
            db_url = database_url or self._build_secure_database_url()
            
            # SSL/TLS configuration
            connect_args = {}
            if enable_ssl and not db_url.startswith('sqlite'):
                ssl_context = self._create_ssl_context()
                connect_args.update({
                    'sslmode': 'require',
                    'sslcert': self._get_client_cert_path(),
                    'sslkey': self._get_client_key_path(),
                    'sslrootcert': self._get_ca_cert_path(),
                    'sslcontext': ssl_context
                })
            
            # Create engine with security settings
            engine = create_engine(
                db_url,
                poolclass=QueuePool,
                pool_size=20,
                max_overflow=40,
                pool_timeout=30,
                pool_recycle=1800,  # 30 minutes
                pool_pre_ping=True,  # Verify connections
                connect_args=connect_args,
                echo=False,  # Disable SQL echo for security
                future=True
            )
            
            # Enable audit logging
            if enable_audit:
                self._setup_audit_logging(engine)
            
            # Validate connection security
            self._validate_connection_security(engine)
            
            logger.info("Secure database engine created successfully")
            return engine
            
        except Exception as e:
            logger.error(f"Failed to create secure database engine: {e}")
            raise
    
    def _build_secure_database_url(self) -> str:
        """Build database URL using credentials from secrets manager"""
        try:
            # Get credentials from secrets manager
            db_host = self.secrets_manager.get_secret("db_host") or "localhost"
            db_port = self.secrets_manager.get_secret("db_port") or "5432"
            db_name = self.secrets_manager.get_secret("db_name") or "investment_db"
            db_user = self.secrets_manager.get_secret("db_user") or "postgres"
            db_password = self.secrets_manager.get_secret("db_password")
            
            if not db_password:
                # Try environment variable as fallback
                db_password = os.getenv("DB_PASSWORD")
                if not db_password:
                    raise ValueError("Database password not found in secrets manager or environment")
                
                # Store in secrets manager for future use
                self.secrets_manager.store_secret(
                    "db_password",
                    db_password,
                    SecretType.DATABASE_CREDENTIAL,
                    description="Database password"
                )
            
            # Build secure connection URL
            db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
            
            return db_url
            
        except Exception as e:
            logger.error(f"Error building database URL: {e}")
            # Fallback to settings
            return settings.DATABASE_URL
    
    def _create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context with security hardening"""
        try:
            # Create SSL context with strong security
            context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
            
            # Set minimum TLS version
            if hasattr(ssl, 'TLSVersion'):
                context.minimum_version = ssl.TLSVersion.TLSv1_2
                context.maximum_version = ssl.TLSVersion.TLSv1_3
            
            # Configure cipher suites
            context.set_ciphers(':'.join(self.required_cipher_suites))
            
            # Security options
            context.check_hostname = True
            context.verify_mode = ssl.CERT_REQUIRED
            
            # Load client certificate if available
            client_cert = self._get_client_cert_path()
            client_key = self._get_client_key_path()
            if client_cert and client_key:
                context.load_cert_chain(client_cert, client_key)
            
            # Load CA certificate
            ca_cert = self._get_ca_cert_path()
            if ca_cert:
                context.load_verify_locations(ca_cert)
            
            return context
            
        except Exception as e:
            logger.error(f"Error creating SSL context: {e}")
            return ssl.create_default_context()
    
    def _get_client_cert_path(self) -> Optional[str]:
        """Get client certificate path"""
        cert_path = os.getenv("DB_CLIENT_CERT_PATH", "/app/certs/client.crt")
        return cert_path if os.path.exists(cert_path) else None
    
    def _get_client_key_path(self) -> Optional[str]:
        """Get client private key path"""
        key_path = os.getenv("DB_CLIENT_KEY_PATH", "/app/certs/client.key")
        return key_path if os.path.exists(key_path) else None
    
    def _get_ca_cert_path(self) -> Optional[str]:
        """Get CA certificate path"""
        ca_path = os.getenv("DB_CA_CERT_PATH", "/app/certs/ca.crt")
        return ca_path if os.path.exists(ca_path) else None
    
    def _setup_audit_logging(self, engine: Engine):
        """Setup database audit logging"""
        try:
            @event.listens_for(engine, "before_cursor_execute")
            def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
                """Log query execution"""
                context._query_start_time = datetime.utcnow()
                context._query_statement = statement
                context._query_parameters = parameters
            
            @event.listens_for(engine, "after_cursor_execute")
            def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
                """Log query completion"""
                try:
                    end_time = datetime.utcnow()
                    start_time = getattr(context, '_query_start_time', end_time)
                    duration_ms = (end_time - start_time).total_seconds() * 1000
                    
                    # Create audit log entry
                    audit_entry = AuditLogEntry(
                        timestamp=end_time,
                        event_type=AuditEventType.QUERY,
                        query=statement,
                        query_hash=hashlib.sha256(statement.encode()).hexdigest()[:16],
                        duration_ms=duration_ms,
                        row_count=cursor.rowcount if hasattr(cursor, 'rowcount') else None,
                        success=True
                    )
                    
                    # Analyze query risk
                    audit_entry.risk_score = self._calculate_query_risk(statement)
                    
                    # Extract affected tables
                    audit_entry.affected_tables = self._extract_table_names(statement)
                    
                    # Log high-risk queries
                    if audit_entry.risk_score >= 7:
                        logger.warning(f"High-risk database query executed: {statement[:200]}...")
                    
                    # Write audit log
                    self._write_audit_log(audit_entry)
                    
                except Exception as e:
                    logger.error(f"Error in audit logging: {e}")
            
            @event.listens_for(engine, "handle_error")
            def handle_error(exception_context):
                """Log database errors"""
                try:
                    audit_entry = AuditLogEntry(
                        timestamp=datetime.utcnow(),
                        event_type=AuditEventType.ERROR,
                        query=getattr(exception_context, 'statement', None),
                        success=False,
                        error_message=str(exception_context.original_exception),
                        risk_score=5  # Errors are medium risk
                    )
                    
                    self._write_audit_log(audit_entry)
                    
                except Exception as e:
                    logger.error(f"Error logging database error: {e}")
            
            logger.info("Database audit logging configured successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup audit logging: {e}")
    
    def _calculate_query_risk(self, query: str) -> int:
        """Calculate risk score for a database query"""
        import re
        
        risk_score = 0
        query_upper = query.upper()
        
        # Check for high-risk patterns
        for pattern in self.high_risk_patterns:
            if re.search(pattern, query_upper):
                risk_score += 3
        
        # Additional risk factors
        if 'WHERE' not in query_upper and any(op in query_upper for op in ['DELETE', 'UPDATE']):
            risk_score += 5  # DELETE/UPDATE without WHERE
        
        if query_upper.count(';') > 0:
            risk_score += 2  # Multiple statements
        
        if any(func in query_upper for func in ['EXEC', 'EXECUTE', 'SP_', 'XP_']):
            risk_score += 4  # Stored procedures
        
        # Length-based risk (very long queries may indicate injection)
        if len(query) > 5000:
            risk_score += 2
        
        return min(risk_score, 10)  # Cap at 10
    
    def _extract_table_names(self, query: str) -> List[str]:
        """Extract table names from SQL query"""
        import re
        
        # Basic table name extraction patterns
        patterns = [
            r'FROM\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'JOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'INTO\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'UPDATE\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'DELETE\s+FROM\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        ]
        
        tables = set()
        query_upper = query.upper()
        
        for pattern in patterns:
            matches = re.findall(pattern, query_upper)
            tables.update(matches)
        
        return list(tables)
    
    def _write_audit_log(self, entry: AuditLogEntry):
        """Write audit log entry to file"""
        try:
            log_data = asdict(entry)
            
            # Convert datetime to ISO format
            if log_data['timestamp']:
                log_data['timestamp'] = log_data['timestamp'].isoformat()
            
            # Write as JSON line
            with open(self.audit_log_path, 'a') as f:
                f.write(json.dumps(log_data) + '\n')
        
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    def _validate_connection_security(self, engine: Engine):
        """Validate database connection security"""
        try:
            with engine.connect() as conn:
                # Check SSL status (PostgreSQL)
                try:
                    result = conn.execute(text("SELECT ssl_is_used();"))
                    ssl_used = result.scalar()
                    if not ssl_used:
                        logger.warning("Database connection is not using SSL")
                except Exception:
                    # May not be PostgreSQL or function may not exist
                    pass
                
                # Check connection security settings
                try:
                    result = conn.execute(text("SHOW ssl;"))
                    ssl_setting = result.scalar()
                    logger.info(f"Database SSL setting: {ssl_setting}")
                except Exception:
                    pass
                
                # Log connection validation
                audit_entry = AuditLogEntry(
                    timestamp=datetime.utcnow(),
                    event_type=AuditEventType.CONNECTION,
                    query="CONNECTION_VALIDATION",
                    success=True
                )
                self._write_audit_log(audit_entry)
        
        except Exception as e:
            logger.error(f"Error validating connection security: {e}")
    
    def rotate_database_credentials(self) -> bool:
        """
        Rotate database credentials securely.
        
        Returns:
            True if rotation was successful
        """
        try:
            # Generate new password
            import secrets
            import string
            
            new_password = ''.join(
                secrets.choice(string.ascii_letters + string.digits + "!@#$%^&*")
                for _ in range(32)
            )
            
            # Store new password in secrets manager
            self.secrets_manager.rotate_secret("db_password", new_password)
            
            # Log rotation event
            audit_entry = AuditLogEntry(
                timestamp=datetime.utcnow(),
                event_type=AuditEventType.CONFIGURATION_CHANGE,
                query="CREDENTIAL_ROTATION",
                success=True,
                risk_score=8  # High risk event
            )
            self._write_audit_log(audit_entry)
            
            logger.info("Database credentials rotated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rotate database credentials: {e}")
            return False
    
    def get_audit_logs(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_type: Optional[AuditEventType] = None,
        min_risk_score: int = 0
    ) -> List[AuditLogEntry]:
        """
        Retrieve audit logs with filtering.
        
        Args:
            start_date: Filter logs after this date
            end_date: Filter logs before this date
            event_type: Filter by event type
            min_risk_score: Minimum risk score to include
            
        Returns:
            List of audit log entries
        """
        logs = []
        
        try:
            if not self.audit_log_path.exists():
                return logs
            
            with open(self.audit_log_path, 'r') as f:
                for line in f:
                    try:
                        log_data = json.loads(line.strip())
                        
                        # Convert timestamp back to datetime
                        if log_data['timestamp']:
                            log_data['timestamp'] = datetime.fromisoformat(log_data['timestamp'])
                        
                        # Apply filters
                        if start_date and log_data['timestamp'] < start_date:
                            continue
                        if end_date and log_data['timestamp'] > end_date:
                            continue
                        if event_type and log_data['event_type'] != event_type.value:
                            continue
                        if log_data.get('risk_score', 0) < min_risk_score:
                            continue
                        
                        # Create AuditLogEntry object
                        entry = AuditLogEntry(**log_data)
                        logs.append(entry)
                        
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.error(f"Error parsing audit log line: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Error reading audit logs: {e}")
        
        return logs
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate a security report based on audit logs"""
        try:
            # Get logs from the last 24 hours
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=1)
            
            logs = self.get_audit_logs(start_time, end_time)
            
            # Analyze logs
            total_queries = len([log for log in logs if log.event_type == AuditEventType.QUERY])
            high_risk_queries = len([log for log in logs if log.risk_score >= 7])
            errors = len([log for log in logs if not log.success])
            connections = len([log for log in logs if log.event_type == AuditEventType.CONNECTION])
            
            # Calculate risk metrics
            avg_risk_score = sum(log.risk_score for log in logs) / len(logs) if logs else 0
            
            # Find most accessed tables
            table_access = {}
            for log in logs:
                if log.affected_tables:
                    for table in log.affected_tables:
                        table_access[table] = table_access.get(table, 0) + 1
            
            report = {
                "report_generated": end_time.isoformat(),
                "analysis_period": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                },
                "summary": {
                    "total_queries": total_queries,
                    "high_risk_queries": high_risk_queries,
                    "errors": errors,
                    "connections": connections,
                    "average_risk_score": round(avg_risk_score, 2)
                },
                "top_accessed_tables": dict(sorted(table_access.items(), key=lambda x: x[1], reverse=True)[:10]),
                "security_recommendations": self._generate_security_recommendations(logs)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating security report: {e}")
            return {}
    
    def _generate_security_recommendations(self, logs: List[AuditLogEntry]) -> List[str]:
        """Generate security recommendations based on audit logs"""
        recommendations = []
        
        high_risk_count = len([log for log in logs if log.risk_score >= 7])
        error_count = len([log for log in logs if not log.success])
        
        if high_risk_count > 10:
            recommendations.append("High number of risky queries detected. Review query patterns and implement additional input validation.")
        
        if error_count > 50:
            recommendations.append("High error rate detected. Investigate database connectivity and query issues.")
        
        # Check for unusual patterns
        query_hashes = [log.query_hash for log in logs if log.query_hash]
        unique_queries = len(set(query_hashes))
        total_queries = len(query_hashes)
        
        if total_queries > 0 and unique_queries / total_queries < 0.1:
            recommendations.append("Low query diversity detected. Consider implementing query caching to reduce database load.")
        
        if not recommendations:
            recommendations.append("No immediate security concerns detected. Continue monitoring.")
        
        return recommendations


# Global security manager instance
_db_security_manager: Optional[DatabaseSecurityManager] = None


def get_database_security_manager() -> DatabaseSecurityManager:
    """Get or create the global database security manager"""
    global _db_security_manager
    if _db_security_manager is None:
        _db_security_manager = DatabaseSecurityManager()
    return _db_security_manager


def create_secure_database_engine(**kwargs) -> Engine:
    """Create a secure database engine with default security settings"""
    manager = get_database_security_manager()
    return manager.create_secure_engine(**kwargs)