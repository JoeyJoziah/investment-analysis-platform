"""
Enterprise-Grade Secrets Management Vault
Provides secure storage, encryption, and retrieval of sensitive credentials
"""

import os
import json
import hashlib
import secrets
from typing import Dict, Optional, Any, Union
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
from datetime import datetime, timedelta
import base64
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
import asyncio
import aiofiles
from pydantic import BaseModel, validator
from enum import Enum

logger = logging.getLogger(__name__)


class SecretType(str, Enum):
    """Types of secrets managed by the vault"""
    API_KEY = "api_key"
    DATABASE_CREDENTIAL = "database_credential"
    ENCRYPTION_KEY = "encryption_key"
    JWT_SECRET = "jwt_secret"
    OAUTH_SECRET = "oauth_secret"
    TLS_CERTIFICATE = "tls_certificate"
    WEBHOOK_SECRET = "webhook_secret"
    SERVICE_ACCOUNT = "service_account"


class RotationPolicy(str, Enum):
    """Secret rotation policies"""
    NEVER = "never"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"


@dataclass
class SecretMetadata:
    """Metadata for secrets in the vault"""
    secret_id: str
    secret_type: SecretType
    created_at: datetime
    updated_at: datetime
    rotation_policy: RotationPolicy
    expires_at: Optional[datetime] = None
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    tags: Dict[str, str] = None
    encrypted_checksum: str = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "secret_id": self.secret_id,
            "secret_type": self.secret_type.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "rotation_policy": self.rotation_policy.value,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "access_count": self.access_count,
            "tags": self.tags or {},
            "encrypted_checksum": self.encrypted_checksum
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SecretMetadata":
        """Create instance from dictionary"""
        return cls(
            secret_id=data["secret_id"],
            secret_type=SecretType(data["secret_type"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            rotation_policy=RotationPolicy(data["rotation_policy"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else None,
            access_count=data.get("access_count", 0),
            tags=data.get("tags", {}),
            encrypted_checksum=data.get("encrypted_checksum")
        )


class SecretAccessControl:
    """Role-based access control for secrets"""
    
    def __init__(self):
        self.permissions: Dict[str, Dict[str, bool]] = {
            "admin": {
                "read": True,
                "write": True,
                "delete": True,
                "rotate": True,
                "audit": True
            },
            "service": {
                "read": True,
                "write": False,
                "delete": False,
                "rotate": False,
                "audit": False
            },
            "read_only": {
                "read": True,
                "write": False,
                "delete": False,
                "rotate": False,
                "audit": True
            }
        }
    
    def check_permission(self, role: str, action: str) -> bool:
        """Check if role has permission for action"""
        return self.permissions.get(role, {}).get(action, False)


class SecretsVault:
    """
    Enterprise-grade secrets management vault with encryption,
    access control, audit logging, and rotation capabilities
    """
    
    def __init__(self, vault_path: Optional[str] = None, master_key: Optional[str] = None):
        self.vault_path = Path(vault_path or os.getenv("SECRETS_VAULT_PATH", "/app/secrets/vault"))
        self.vault_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize encryption
        self.master_key = master_key or os.getenv("MASTER_SECRET_KEY")
        if not self.master_key:
            raise ValueError("Master key required for secrets vault")
        
        self.fernet = self._initialize_encryption()
        self.access_control = SecretAccessControl()
        
        # Vault files
        self.secrets_file = self.vault_path / "secrets.vault"
        self.metadata_file = self.vault_path / "metadata.json"
        self.audit_file = self.vault_path / "audit.log"
        
        # In-memory cache for performance
        self._cache: Dict[str, Any] = {}
        self._cache_ttl: Dict[str, datetime] = {}
        
        # Initialize vault structure
        self._initialize_vault()
    
    def _initialize_encryption(self) -> Fernet:
        """Initialize Fernet encryption with PBKDF2 key derivation"""
        # Use PBKDF2 to derive encryption key from master key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"investment_platform_salt_2025",  # Use a unique salt
            iterations=100000,
            backend=default_backend()
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_key.encode()))
        return Fernet(key)
    
    def _initialize_vault(self):
        """Initialize vault structure if it doesn't exist"""
        if not self.secrets_file.exists():
            with open(self.secrets_file, 'w') as f:
                f.write("{}")
        
        if not self.metadata_file.exists():
            with open(self.metadata_file, 'w') as f:
                json.dump({}, f)
        
        if not self.audit_file.exists():
            self.audit_file.touch()
    
    def _log_audit_event(self, event_type: str, secret_id: str, user: str = "system", details: Optional[Dict] = None):
        """Log security audit event"""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "secret_id": secret_id,
            "user": user,
            "details": details or {}
        }
        
        with open(self.audit_file, 'a') as f:
            f.write(json.dumps(event) + "\n")
        
        logger.info(f"Audit: {event_type} for secret {secret_id} by {user}")
    
    def _generate_secret_checksum(self, secret_value: str) -> str:
        """Generate encrypted checksum for integrity verification"""
        checksum = hashlib.sha256(secret_value.encode()).hexdigest()
        return self.fernet.encrypt(checksum.encode()).decode()
    
    def _verify_secret_integrity(self, secret_value: str, encrypted_checksum: str) -> bool:
        """Verify secret integrity using checksum"""
        try:
            expected_checksum = hashlib.sha256(secret_value.encode()).hexdigest()
            actual_checksum = self.fernet.decrypt(encrypted_checksum.encode()).decode()
            return expected_checksum == actual_checksum
        except Exception:
            return False
    
    async def store_secret(
        self,
        secret_id: str,
        secret_value: Union[str, Dict],
        secret_type: SecretType,
        rotation_policy: RotationPolicy = RotationPolicy.MONTHLY,
        expires_at: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None,
        user: str = "system"
    ) -> bool:
        """
        Store a secret in the vault with metadata
        """
        try:
            # Convert dict to JSON string if needed
            if isinstance(secret_value, dict):
                secret_value = json.dumps(secret_value)
            
            # Encrypt the secret
            encrypted_secret = self.fernet.encrypt(secret_value.encode()).decode()
            
            # Generate checksum for integrity
            checksum = self._generate_secret_checksum(secret_value)
            
            # Create metadata
            metadata = SecretMetadata(
                secret_id=secret_id,
                secret_type=secret_type,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                rotation_policy=rotation_policy,
                expires_at=expires_at,
                tags=tags or {},
                encrypted_checksum=checksum
            )
            
            # Load existing vault data
            async with aiofiles.open(self.secrets_file, 'r') as f:
                vault_data = json.loads(await f.read())
            
            async with aiofiles.open(self.metadata_file, 'r') as f:
                metadata_data = json.loads(await f.read())
            
            # Store encrypted secret and metadata
            vault_data[secret_id] = encrypted_secret
            metadata_data[secret_id] = metadata.to_dict()
            
            # Write back to files
            async with aiofiles.open(self.secrets_file, 'w') as f:
                await f.write(json.dumps(vault_data, indent=2))
            
            async with aiofiles.open(self.metadata_file, 'w') as f:
                await f.write(json.dumps(metadata_data, indent=2))
            
            # Clear cache
            if secret_id in self._cache:
                del self._cache[secret_id]
                del self._cache_ttl[secret_id]
            
            # Log audit event
            self._log_audit_event("SECRET_STORED", secret_id, user, {
                "secret_type": secret_type.value,
                "rotation_policy": rotation_policy.value
            })
            
            logger.info(f"Secret {secret_id} stored successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store secret {secret_id}: {e}")
            self._log_audit_event("SECRET_STORE_FAILED", secret_id, user, {"error": str(e)})
            return False
    
    async def get_secret(self, secret_id: str, user: str = "system") -> Optional[str]:
        """
        Retrieve a secret from the vault
        """
        try:
            # Check cache first
            if secret_id in self._cache and self._cache_ttl.get(secret_id, datetime.min) > datetime.utcnow():
                self._update_access_metadata(secret_id)
                return self._cache[secret_id]
            
            # Load from vault
            async with aiofiles.open(self.secrets_file, 'r') as f:
                vault_data = json.loads(await f.read())
            
            async with aiofiles.open(self.metadata_file, 'r') as f:
                metadata_data = json.loads(await f.read())
            
            if secret_id not in vault_data:
                return None
            
            # Decrypt secret
            encrypted_secret = vault_data[secret_id]
            secret_value = self.fernet.decrypt(encrypted_secret.encode()).decode()
            
            # Verify integrity if checksum exists
            metadata = SecretMetadata.from_dict(metadata_data[secret_id])
            if metadata.encrypted_checksum and not self._verify_secret_integrity(secret_value, metadata.encrypted_checksum):
                logger.error(f"Secret {secret_id} failed integrity check")
                self._log_audit_event("SECRET_INTEGRITY_FAILURE", secret_id, user)
                return None
            
            # Check expiration
            if metadata.expires_at and datetime.utcnow() > metadata.expires_at:
                logger.warning(f"Secret {secret_id} has expired")
                self._log_audit_event("SECRET_EXPIRED_ACCESS", secret_id, user)
                return None
            
            # Cache the secret
            self._cache[secret_id] = secret_value
            self._cache_ttl[secret_id] = datetime.utcnow() + timedelta(minutes=5)
            
            # Update access metadata
            await self._update_access_metadata(secret_id)
            
            # Log audit event
            self._log_audit_event("SECRET_ACCESSED", secret_id, user)
            
            return secret_value
            
        except Exception as e:
            logger.error(f"Failed to retrieve secret {secret_id}: {e}")
            self._log_audit_event("SECRET_ACCESS_FAILED", secret_id, user, {"error": str(e)})
            return None
    
    async def _update_access_metadata(self, secret_id: str):
        """Update last accessed timestamp and access count"""
        try:
            async with aiofiles.open(self.metadata_file, 'r') as f:
                metadata_data = json.loads(await f.read())
            
            if secret_id in metadata_data:
                metadata_data[secret_id]["last_accessed"] = datetime.utcnow().isoformat()
                metadata_data[secret_id]["access_count"] += 1
                
                async with aiofiles.open(self.metadata_file, 'w') as f:
                    await f.write(json.dumps(metadata_data, indent=2))
        except Exception as e:
            logger.error(f"Failed to update access metadata for {secret_id}: {e}")
    
    async def rotate_secret(self, secret_id: str, new_secret_value: str, user: str = "system") -> bool:
        """
        Rotate a secret to a new value
        """
        try:
            # Get existing metadata
            async with aiofiles.open(self.metadata_file, 'r') as f:
                metadata_data = json.loads(await f.read())
            
            if secret_id not in metadata_data:
                return False
            
            metadata = SecretMetadata.from_dict(metadata_data[secret_id])
            
            # Update the secret with new value
            success = await self.store_secret(
                secret_id=secret_id,
                secret_value=new_secret_value,
                secret_type=metadata.secret_type,
                rotation_policy=metadata.rotation_policy,
                expires_at=metadata.expires_at,
                tags=metadata.tags,
                user=user
            )
            
            if success:
                self._log_audit_event("SECRET_ROTATED", secret_id, user)
                logger.info(f"Secret {secret_id} rotated successfully")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to rotate secret {secret_id}: {e}")
            self._log_audit_event("SECRET_ROTATION_FAILED", secret_id, user, {"error": str(e)})
            return False
    
    async def delete_secret(self, secret_id: str, user: str = "system") -> bool:
        """
        Securely delete a secret from the vault
        """
        try:
            # Load vault data
            async with aiofiles.open(self.secrets_file, 'r') as f:
                vault_data = json.loads(await f.read())
            
            async with aiofiles.open(self.metadata_file, 'r') as f:
                metadata_data = json.loads(await f.read())
            
            # Remove secret and metadata
            vault_data.pop(secret_id, None)
            metadata_data.pop(secret_id, None)
            
            # Write back to files
            async with aiofiles.open(self.secrets_file, 'w') as f:
                await f.write(json.dumps(vault_data, indent=2))
            
            async with aiofiles.open(self.metadata_file, 'w') as f:
                await f.write(json.dumps(metadata_data, indent=2))
            
            # Clear from cache
            self._cache.pop(secret_id, None)
            self._cache_ttl.pop(secret_id, None)
            
            # Log audit event
            self._log_audit_event("SECRET_DELETED", secret_id, user)
            
            logger.info(f"Secret {secret_id} deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete secret {secret_id}: {e}")
            self._log_audit_event("SECRET_DELETION_FAILED", secret_id, user, {"error": str(e)})
            return False
    
    async def list_secrets(self, secret_type: Optional[SecretType] = None) -> Dict[str, SecretMetadata]:
        """
        List all secrets (metadata only) with optional filtering
        """
        try:
            async with aiofiles.open(self.metadata_file, 'r') as f:
                metadata_data = json.loads(await f.read())
            
            result = {}
            for secret_id, data in metadata_data.items():
                metadata = SecretMetadata.from_dict(data)
                if secret_type is None or metadata.secret_type == secret_type:
                    result[secret_id] = metadata
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to list secrets: {e}")
            return {}
    
    async def get_secrets_requiring_rotation(self) -> List[str]:
        """
        Get list of secrets that need rotation based on their policy
        """
        secrets = await self.list_secrets()
        requiring_rotation = []
        
        for secret_id, metadata in secrets.items():
            if self._should_rotate_secret(metadata):
                requiring_rotation.append(secret_id)
        
        return requiring_rotation
    
    def _should_rotate_secret(self, metadata: SecretMetadata) -> bool:
        """
        Determine if a secret should be rotated based on policy
        """
        if metadata.rotation_policy == RotationPolicy.NEVER:
            return False
        
        now = datetime.utcnow()
        age = now - metadata.updated_at
        
        rotation_intervals = {
            RotationPolicy.DAILY: timedelta(days=1),
            RotationPolicy.WEEKLY: timedelta(weeks=1),
            RotationPolicy.MONTHLY: timedelta(days=30),
            RotationPolicy.QUARTERLY: timedelta(days=90),
            RotationPolicy.ANNUALLY: timedelta(days=365)
        }
        
        interval = rotation_intervals.get(metadata.rotation_policy)
        return interval and age >= interval
    
    def generate_api_key(self, length: int = 32) -> str:
        """Generate a secure API key"""
        return secrets.token_urlsafe(length)
    
    def generate_jwt_secret(self, length: int = 64) -> str:
        """Generate a secure JWT secret"""
        return secrets.token_urlsafe(length)
    
    async def backup_vault(self, backup_path: str) -> bool:
        """
        Create an encrypted backup of the vault
        """
        try:
            import shutil
            backup_dir = Path(backup_path)
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy vault files to backup location
            shutil.copy2(self.secrets_file, backup_dir / f"secrets_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.vault")
            shutil.copy2(self.metadata_file, backup_dir / f"metadata_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
            shutil.copy2(self.audit_file, backup_dir / f"audit_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log")
            
            logger.info(f"Vault backed up to {backup_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to backup vault: {e}")
            return False


# Global vault instance
_vault_instance: Optional[SecretsVault] = None


def get_secrets_vault() -> SecretsVault:
    """Get or create the global secrets vault instance"""
    global _vault_instance
    if _vault_instance is None:
        _vault_instance = SecretsVault()
    return _vault_instance


async def migrate_env_to_vault():
    """
    Migrate secrets from .env file to the secure vault
    """
    vault = get_secrets_vault()
    
    # Define secrets to migrate from .env
    secrets_to_migrate = {
        "SECRET_KEY": SecretType.ENCRYPTION_KEY,
        "JWT_SECRET_KEY": SecretType.JWT_SECRET,
        "MASTER_SECRET_KEY": SecretType.ENCRYPTION_KEY,
        "AIRFLOW_FERNET_KEY": SecretType.ENCRYPTION_KEY,
        "DB_PASSWORD": SecretType.DATABASE_CREDENTIAL,
        "POSTGRES_PASSWORD": SecretType.DATABASE_CREDENTIAL,
        "REDIS_PASSWORD": SecretType.DATABASE_CREDENTIAL,
        "ELASTICSEARCH_PASSWORD": SecretType.DATABASE_CREDENTIAL,
        "AIRFLOW_DB_PASSWORD": SecretType.DATABASE_CREDENTIAL,
        "AIRFLOW_ADMIN_PASSWORD": SecretType.DATABASE_CREDENTIAL,
        "GRAFANA_ADMIN_PASSWORD": SecretType.DATABASE_CREDENTIAL,
        "DOCKER_PASSWORD": SecretType.SERVICE_ACCOUNT,
        "ALPHA_VANTAGE_API_KEY": SecretType.API_KEY,
        "FINNHUB_API_KEY": SecretType.API_KEY,
        "POLYGON_API_KEY": SecretType.API_KEY,
        "NEWS_API_KEY": SecretType.API_KEY,
        "FMP_API_KEY": SecretType.API_KEY,
        "MARKETAUX_API_KEY": SecretType.API_KEY,
        "FRED_API_KEY": SecretType.API_KEY,
        "OPENWEATHER_API_KEY": SecretType.API_KEY
    }
    
    migrated_count = 0
    for env_var, secret_type in secrets_to_migrate.items():
        value = os.getenv(env_var)
        if value and value not in ["", "CHANGE_THIS_PASSWORD", "secure_password_123"]:
            success = await vault.store_secret(
                secret_id=env_var.lower(),
                secret_value=value,
                secret_type=secret_type,
                rotation_policy=RotationPolicy.MONTHLY if secret_type == SecretType.API_KEY else RotationPolicy.QUARTERLY,
                user="migration"
            )
            if success:
                migrated_count += 1
                logger.info(f"Migrated {env_var} to vault")
    
    logger.info(f"Migration complete: {migrated_count} secrets migrated to vault")
    return migrated_count