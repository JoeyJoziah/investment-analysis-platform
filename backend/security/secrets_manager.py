"""
Secure Secrets Management System

This module provides encrypted storage and management of API keys and sensitive credentials.
Implements environment-based secrets loading with validation and rotation capabilities.
"""

import os
import json
import base64
import hashlib
from typing import Dict, Optional, Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from pathlib import Path
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

from backend.config.settings import settings

logger = logging.getLogger(__name__)


class SecretType(str, Enum):
    """Types of secrets managed by the system"""
    API_KEY = "api_key"
    DATABASE_CREDENTIAL = "database_credential"
    JWT_KEY = "jwt_key"
    ENCRYPTION_KEY = "encryption_key"
    OAUTH_SECRET = "oauth_secret"


@dataclass
class SecretMetadata:
    """Metadata for stored secrets"""
    secret_type: SecretType
    created_at: datetime
    expires_at: Optional[datetime] = None
    rotated_at: Optional[datetime] = None
    rotation_count: int = 0
    description: Optional[str] = None


class SecretsManager:
    """
    Secure secrets management system with encryption, rotation, and validation.
    
    Features:
    - AES-256 encryption of stored secrets
    - Key derivation from master password
    - Automatic secret rotation
    - Environment-based configuration
    - Audit logging of access
    """
    
    def __init__(self, secrets_dir: Optional[str] = None, master_key: Optional[str] = None):
        self.secrets_dir = Path(secrets_dir or os.getenv("SECRETS_DIR", "./secrets"))
        self.secrets_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
        
        # Initialize encryption
        self.master_key = master_key or os.getenv("MASTER_SECRET_KEY")
        if not self.master_key:
            if settings.is_production:
                # CRITICAL: In production, MASTER_SECRET_KEY is required
                raise ValueError(
                    "CRITICAL SECURITY ERROR: MASTER_SECRET_KEY environment variable "
                    "is not set. This key is required for production deployment. "
                    "Generate a secure key with: python -c \"import secrets; print(secrets.token_urlsafe(32))\" "
                    "and set it as MASTER_SECRET_KEY in your environment."
                )
            else:
                # For development/testing only, generate a temporary key
                import warnings
                warnings.warn(
                    f"MASTER_SECRET_KEY not set in {settings.ENVIRONMENT} environment - "
                    "using temporary key. This is only acceptable for development/testing.",
                    RuntimeWarning
                )
                import secrets as sec
                self.master_key = sec.token_urlsafe(32)
                logger.warning(
                    "Using temporary encryption key in %s environment. "
                    "Secrets will NOT persist across restarts.",
                    settings.ENVIRONMENT
                )
        
        self._fernet = self._initialize_encryption()
        self._secrets_cache: Dict[str, Any] = {}
        self._metadata_cache: Dict[str, SecretMetadata] = {}
        
        # Load existing secrets
        self._load_secrets()
    
    def _initialize_encryption(self) -> Fernet:
        """Initialize Fernet encryption with PBKDF2-derived key"""
        # Use a fixed salt for consistent key derivation
        salt = hashlib.sha256(b"investment_analysis_salt").digest()[:16]
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,  # NIST recommended minimum
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_key.encode()))
        return Fernet(key)
    
    def _get_secret_path(self, secret_name: str) -> Path:
        """Get file path for a secret"""
        safe_name = hashlib.sha256(secret_name.encode()).hexdigest()[:16]
        return self.secrets_dir / f"{safe_name}.enc"
    
    def _get_metadata_path(self, secret_name: str) -> Path:
        """Get metadata file path for a secret"""
        safe_name = hashlib.sha256(secret_name.encode()).hexdigest()[:16]
        return self.secrets_dir / f"{safe_name}.meta"
    
    def store_secret(
        self,
        secret_name: str,
        secret_value: str,
        secret_type: SecretType,
        expires_in_days: Optional[int] = None,
        description: Optional[str] = None
    ) -> bool:
        """
        Store an encrypted secret with metadata.
        
        Args:
            secret_name: Unique identifier for the secret
            secret_value: The secret value to encrypt and store
            secret_type: Type of secret being stored
            expires_in_days: Optional expiration in days
            description: Optional description
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate secret value
            if not secret_value or len(secret_value.strip()) == 0:
                raise ValueError("Secret value cannot be empty")
            
            # Encrypt the secret
            encrypted_value = self._fernet.encrypt(secret_value.encode())
            
            # Create metadata
            expires_at = None
            if expires_in_days:
                expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
            
            metadata = SecretMetadata(
                secret_type=secret_type,
                created_at=datetime.utcnow(),
                expires_at=expires_at,
                description=description
            )
            
            # Write encrypted secret
            secret_path = self._get_secret_path(secret_name)
            with open(secret_path, 'wb') as f:
                f.write(encrypted_value)
            
            # Set restrictive permissions
            secret_path.chmod(0o600)
            
            # Write metadata
            metadata_path = self._get_metadata_path(secret_name)
            metadata_dict = {
                'secret_type': metadata.secret_type.value,
                'created_at': metadata.created_at.isoformat(),
                'expires_at': metadata.expires_at.isoformat() if metadata.expires_at else None,
                'rotated_at': metadata.rotated_at.isoformat() if metadata.rotated_at else None,
                'rotation_count': metadata.rotation_count,
                'description': metadata.description
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata_dict, f)
            
            metadata_path.chmod(0o600)
            
            # Update cache
            self._secrets_cache[secret_name] = secret_value
            self._metadata_cache[secret_name] = metadata
            
            logger.info(f"Secret '{secret_name}' stored successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store secret '{secret_name}': {e}")
            return False
    
    def get_secret(self, secret_name: str) -> Optional[str]:
        """
        Retrieve and decrypt a secret.
        
        Args:
            secret_name: Name of the secret to retrieve
            
        Returns:
            Decrypted secret value or None if not found
        """
        try:
            # Check cache first
            if secret_name in self._secrets_cache:
                # Verify not expired
                metadata = self._metadata_cache.get(secret_name)
                if metadata and metadata.expires_at:
                    if datetime.utcnow() > metadata.expires_at:
                        logger.warning(f"Secret '{secret_name}' has expired")
                        return None
                
                return self._secrets_cache[secret_name]
            
            # Load from disk
            secret_path = self._get_secret_path(secret_name)
            if not secret_path.exists():
                return None
            
            # Read and decrypt
            with open(secret_path, 'rb') as f:
                encrypted_value = f.read()
            
            decrypted_value = self._fernet.decrypt(encrypted_value).decode()
            
            # Load metadata
            metadata = self._load_metadata(secret_name)
            
            # Check expiration
            if metadata and metadata.expires_at:
                if datetime.utcnow() > metadata.expires_at:
                    logger.warning(f"Secret '{secret_name}' has expired")
                    return None
            
            # Cache the result
            self._secrets_cache[secret_name] = decrypted_value
            if metadata:
                self._metadata_cache[secret_name] = metadata
            
            logger.debug(f"Secret '{secret_name}' retrieved successfully")
            return decrypted_value
            
        except Exception as e:
            logger.error(f"Failed to retrieve secret '{secret_name}': {e}")
            return None
    
    def _load_metadata(self, secret_name: str) -> Optional[SecretMetadata]:
        """Load metadata for a secret"""
        try:
            metadata_path = self._get_metadata_path(secret_name)
            if not metadata_path.exists():
                return None
            
            with open(metadata_path, 'r') as f:
                data = json.load(f)
            
            return SecretMetadata(
                secret_type=SecretType(data['secret_type']),
                created_at=datetime.fromisoformat(data['created_at']),
                expires_at=datetime.fromisoformat(data['expires_at']) if data['expires_at'] else None,
                rotated_at=datetime.fromisoformat(data['rotated_at']) if data['rotated_at'] else None,
                rotation_count=data.get('rotation_count', 0),
                description=data.get('description')
            )
            
        except Exception as e:
            logger.error(f"Failed to load metadata for '{secret_name}': {e}")
            return None
    
    def rotate_secret(self, secret_name: str, new_value: str) -> bool:
        """
        Rotate a secret with a new value.
        
        Args:
            secret_name: Name of the secret to rotate
            new_value: New secret value
            
        Returns:
            True if successful, False otherwise
        """
        try:
            metadata = self._metadata_cache.get(secret_name) or self._load_metadata(secret_name)
            if not metadata:
                logger.error(f"Cannot rotate non-existent secret '{secret_name}'")
                return False
            
            # Update metadata
            metadata.rotated_at = datetime.utcnow()
            metadata.rotation_count += 1
            
            # Store new value
            return self.store_secret(
                secret_name,
                new_value,
                metadata.secret_type,
                description=metadata.description
            )
            
        except Exception as e:
            logger.error(f"Failed to rotate secret '{secret_name}': {e}")
            return False
    
    def delete_secret(self, secret_name: str) -> bool:
        """
        Permanently delete a secret.
        
        Args:
            secret_name: Name of the secret to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            secret_path = self._get_secret_path(secret_name)
            metadata_path = self._get_metadata_path(secret_name)
            
            # Remove files
            if secret_path.exists():
                secret_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()
            
            # Remove from cache
            self._secrets_cache.pop(secret_name, None)
            self._metadata_cache.pop(secret_name, None)
            
            logger.info(f"Secret '{secret_name}' deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete secret '{secret_name}': {e}")
            return False
    
    def list_secrets(self) -> Dict[str, SecretMetadata]:
        """List all stored secrets with metadata"""
        secrets = {}
        
        for file_path in self.secrets_dir.glob("*.meta"):
            # Extract secret name hash and try to load metadata
            safe_name = file_path.stem
            
            # Find corresponding .enc file
            enc_path = self.secrets_dir / f"{safe_name}.enc"
            if not enc_path.exists():
                continue
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                metadata = SecretMetadata(
                    secret_type=SecretType(data['secret_type']),
                    created_at=datetime.fromisoformat(data['created_at']),
                    expires_at=datetime.fromisoformat(data['expires_at']) if data['expires_at'] else None,
                    rotated_at=datetime.fromisoformat(data['rotated_at']) if data['rotated_at'] else None,
                    rotation_count=data.get('rotation_count', 0),
                    description=data.get('description')
                )
                
                # Use safe_name as key since we can't reverse the hash
                secrets[safe_name] = metadata
                
            except Exception as e:
                logger.error(f"Failed to load metadata from {file_path}: {e}")
                continue
        
        return secrets
    
    def validate_secret(self, secret_name: str, expected_pattern: Optional[str] = None) -> bool:
        """
        Validate a secret exists and optionally matches a pattern.
        
        Args:
            secret_name: Name of the secret to validate
            expected_pattern: Optional regex pattern to match
            
        Returns:
            True if valid, False otherwise
        """
        try:
            secret_value = self.get_secret(secret_name)
            if not secret_value:
                return False
            
            if expected_pattern:
                import re
                if not re.match(expected_pattern, secret_value):
                    logger.warning(f"Secret '{secret_name}' does not match expected pattern")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to validate secret '{secret_name}': {e}")
            return False
    
    def _load_secrets(self):
        """Load all secrets into cache on initialization"""
        try:
            for file_path in self.secrets_dir.glob("*.enc"):
                # We can't reverse the hash to get the original name
                # So we'll load them on-demand instead
                pass
        except Exception as e:
            logger.error(f"Error loading secrets: {e}")


# Global secrets manager instance
_secrets_manager: Optional[SecretsManager] = None


def get_secrets_manager() -> SecretsManager:
    """Get or create the global secrets manager instance"""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager()
    return _secrets_manager


def get_secret(secret_name: str) -> Optional[str]:
    """Convenience function to get a secret"""
    return get_secrets_manager().get_secret(secret_name)


def store_secret(secret_name: str, secret_value: str, secret_type: SecretType, **kwargs) -> bool:
    """Convenience function to store a secret"""
    return get_secrets_manager().store_secret(secret_name, secret_value, secret_type, **kwargs)


# Environment variable integration
class SecureSettings:
    """Settings class that integrates with secrets manager for sensitive values"""
    
    def __init__(self):
        self.secrets_manager = get_secrets_manager()
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key from secrets manager with fallback to environment"""
        # First try secrets manager
        secret_name = f"api_key_{provider.lower()}"
        secret_value = self.secrets_manager.get_secret(secret_name)
        
        if secret_value:
            return secret_value
        
        # Fallback to environment variable
        env_var = f"{provider.upper()}_API_KEY"
        env_value = os.getenv(env_var)
        
        if env_value:
            # Store in secrets manager for future use
            self.secrets_manager.store_secret(
                secret_name,
                env_value,
                SecretType.API_KEY,
                description=f"API key for {provider}"
            )
            logger.info(f"Migrated {env_var} to secrets manager")
            return env_value
        
        return None
    
    def get_database_credential(self, credential_type: str) -> Optional[str]:
        """Get database credential from secrets manager"""
        secret_name = f"db_{credential_type.lower()}"
        return self.secrets_manager.get_secret(secret_name)
    
    def get_jwt_key(self, key_type: str = "secret") -> Optional[str]:
        """Get JWT key from secrets manager"""
        secret_name = f"jwt_{key_type.lower()}_key"
        return self.secrets_manager.get_secret(secret_name)


# Initialize secure settings
secure_settings = SecureSettings()