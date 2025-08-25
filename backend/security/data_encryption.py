"""
Comprehensive Data Encryption System
Provides encryption at rest, in transit, and field-level encryption
"""

import os
import json
import base64
import hashlib
import secrets
from typing import Any, Dict, List, Optional, Union, Tuple, BinaryIO
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
import asyncio
import aiofiles
import logging
from pathlib import Path

# Cryptography imports
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization, padding
from cryptography.hazmat.primitives.asymmetric import rsa, padding as asym_padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.backends import default_backend
from cryptography.x509.oid import NameOID
from cryptography import x509

# Database encryption
import sqlalchemy
from sqlalchemy import TypeDecorator, String, LargeBinary
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncSession

# FastAPI imports
from fastapi import HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse

from .secrets_vault import get_secrets_vault

logger = logging.getLogger(__name__)


class EncryptionAlgorithm(str, Enum):
    """Supported encryption algorithms"""
    AES_256_GCM = "aes_256_gcm"
    AES_256_CBC = "aes_256_cbc"
    FERNET = "fernet"
    RSA_2048 = "rsa_2048"
    RSA_4096 = "rsa_4096"
    CHACHA20_POLY1305 = "chacha20_poly1305"


class KeyType(str, Enum):
    """Types of encryption keys"""
    MASTER_KEY = "master_key"
    DATA_ENCRYPTION_KEY = "dek"
    KEY_ENCRYPTION_KEY = "kek"
    FIELD_ENCRYPTION_KEY = "field_key"
    TRANSPORT_KEY = "transport_key"


@dataclass
class EncryptionMetadata:
    """Metadata for encrypted data"""
    algorithm: EncryptionAlgorithm
    key_id: str
    iv: Optional[str] = None
    tag: Optional[str] = None
    encrypted_at: Optional[datetime] = None
    version: int = 1


@dataclass
class KeyMetadata:
    """Metadata for encryption keys"""
    key_id: str
    key_type: KeyType
    algorithm: EncryptionAlgorithm
    created_at: datetime
    expires_at: Optional[datetime] = None
    is_active: bool = True
    purpose: str = ""


class KeyManager:
    """Advanced encryption key management system"""
    
    def __init__(self, vault_path: str = None):
        self.vault_path = Path(vault_path or os.getenv("ENCRYPTION_VAULT_PATH", "/app/keys"))
        self.vault_path.mkdir(parents=True, exist_ok=True)
        
        self.secrets_vault = get_secrets_vault()
        self.keys_cache = {}
        self.metadata_cache = {}
        
        # Initialize master key
        self.master_key = self._get_or_create_master_key()
    
    def _get_or_create_master_key(self) -> bytes:
        """Get or create the master encryption key"""
        try:
            # Try to get from secrets vault
            master_key_b64 = asyncio.run(
                self.secrets_vault.get_secret("master_encryption_key")
            )
            
            if master_key_b64:
                return base64.b64decode(master_key_b64)
            else:
                # Generate new master key
                master_key = os.urandom(32)  # 256-bit key
                master_key_b64 = base64.b64encode(master_key).decode()
                
                asyncio.run(
                    self.secrets_vault.store_secret(
                        "master_encryption_key",
                        master_key_b64,
                        secret_type="encryption_key"
                    )
                )
                
                logger.info("Generated new master encryption key")
                return master_key
                
        except Exception as e:
            logger.error(f"Failed to get master key: {e}")
            # Fallback to environment variable
            env_key = os.getenv("MASTER_SECRET_KEY")
            if env_key:
                return hashlib.sha256(env_key.encode()).digest()
            else:
                raise RuntimeError("No master key available")
    
    def generate_key(
        self,
        key_type: KeyType,
        algorithm: EncryptionAlgorithm,
        purpose: str = "",
        expires_in_days: Optional[int] = None
    ) -> Tuple[str, KeyMetadata]:
        """Generate a new encryption key"""
        
        key_id = f"{key_type.value}_{algorithm.value}_{secrets.token_hex(8)}"
        
        # Generate key based on algorithm
        if algorithm == EncryptionAlgorithm.AES_256_GCM or algorithm == EncryptionAlgorithm.AES_256_CBC:
            key_data = os.urandom(32)  # 256-bit key
        elif algorithm == EncryptionAlgorithm.FERNET:
            key_data = Fernet.generate_key()
        elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
            key_data = os.urandom(32)  # 256-bit key
        elif algorithm in [EncryptionAlgorithm.RSA_2048, EncryptionAlgorithm.RSA_4096]:
            key_size = 2048 if algorithm == EncryptionAlgorithm.RSA_2048 else 4096
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_size,
                backend=default_backend()
            )
            key_data = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # Create metadata
        metadata = KeyMetadata(
            key_id=key_id,
            key_type=key_type,
            algorithm=algorithm,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(days=expires_in_days) if expires_in_days else None,
            purpose=purpose
        )
        
        # Store encrypted key
        self._store_key(key_id, key_data, metadata)
        
        logger.info(f"Generated new {algorithm.value} key: {key_id}")
        return key_id, metadata
    
    def _store_key(self, key_id: str, key_data: bytes, metadata: KeyMetadata):
        """Store key encrypted with master key"""
        # Encrypt key with master key using AES-GCM
        iv = os.urandom(12)
        cipher = Cipher(
            algorithms.AES(self.master_key),
            modes.GCM(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        encrypted_key = encryptor.update(key_data) + encryptor.finalize()
        
        # Store key file
        key_file = self.vault_path / f"{key_id}.key"
        with open(key_file, "wb") as f:
            f.write(iv + encryptor.tag + encrypted_key)
        
        # Store metadata
        metadata_file = self.vault_path / f"{key_id}.meta"
        with open(metadata_file, "w") as f:
            json.dump(asdict(metadata), f, default=str)
        
        # Cache
        self.keys_cache[key_id] = key_data
        self.metadata_cache[key_id] = metadata
    
    def get_key(self, key_id: str) -> Optional[bytes]:
        """Get decrypted key by ID"""
        # Check cache first
        if key_id in self.keys_cache:
            return self.keys_cache[key_id]
        
        # Load from file
        key_file = self.vault_path / f"{key_id}.key"
        if not key_file.exists():
            return None
        
        try:
            with open(key_file, "rb") as f:
                data = f.read()
            
            # Extract components
            iv = data[:12]
            tag = data[12:28]
            encrypted_key = data[28:]
            
            # Decrypt
            cipher = Cipher(
                algorithms.AES(self.master_key),
                modes.GCM(iv, tag),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            key_data = decryptor.update(encrypted_key) + decryptor.finalize()
            
            # Cache
            self.keys_cache[key_id] = key_data
            return key_data
            
        except Exception as e:
            logger.error(f"Failed to decrypt key {key_id}: {e}")
            return None
    
    def get_key_metadata(self, key_id: str) -> Optional[KeyMetadata]:
        """Get key metadata"""
        # Check cache first
        if key_id in self.metadata_cache:
            return self.metadata_cache[key_id]
        
        # Load from file
        metadata_file = self.vault_path / f"{key_id}.meta"
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, "r") as f:
                data = json.load(f)
            
            metadata = KeyMetadata(
                key_id=data["key_id"],
                key_type=KeyType(data["key_type"]),
                algorithm=EncryptionAlgorithm(data["algorithm"]),
                created_at=datetime.fromisoformat(data["created_at"]),
                expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
                is_active=data.get("is_active", True),
                purpose=data.get("purpose", "")
            )
            
            # Cache
            self.metadata_cache[key_id] = metadata
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to load metadata for key {key_id}: {e}")
            return None
    
    def rotate_key(self, key_id: str) -> str:
        """Rotate an existing key"""
        metadata = self.get_key_metadata(key_id)
        if not metadata:
            raise ValueError(f"Key {key_id} not found")
        
        # Generate new key with same parameters
        new_key_id, new_metadata = self.generate_key(
            key_type=metadata.key_type,
            algorithm=metadata.algorithm,
            purpose=metadata.purpose
        )
        
        # Deactivate old key
        metadata.is_active = False
        self._update_key_metadata(key_id, metadata)
        
        logger.info(f"Rotated key {key_id} -> {new_key_id}")
        return new_key_id
    
    def _update_key_metadata(self, key_id: str, metadata: KeyMetadata):
        """Update key metadata"""
        metadata_file = self.vault_path / f"{key_id}.meta"
        with open(metadata_file, "w") as f:
            json.dump(asdict(metadata), f, default=str)
        
        # Update cache
        self.metadata_cache[key_id] = metadata
    
    def list_keys(self, key_type: Optional[KeyType] = None, active_only: bool = True) -> List[KeyMetadata]:
        """List all keys with optional filtering"""
        keys = []
        
        for meta_file in self.vault_path.glob("*.meta"):
            key_id = meta_file.stem
            metadata = self.get_key_metadata(key_id)
            
            if metadata:
                if key_type and metadata.key_type != key_type:
                    continue
                if active_only and not metadata.is_active:
                    continue
                
                keys.append(metadata)
        
        return sorted(keys, key=lambda k: k.created_at, reverse=True)


class DataEncryption:
    """High-level data encryption service"""
    
    def __init__(self, key_manager: KeyManager = None):
        self.key_manager = key_manager or KeyManager()
        
        # Get or create default data encryption key
        self.default_dek_id = self._get_or_create_default_dek()
    
    def _get_or_create_default_dek(self) -> str:
        """Get or create default data encryption key"""
        # Look for existing active DEK
        dek_keys = self.key_manager.list_keys(KeyType.DATA_ENCRYPTION_KEY)
        if dek_keys:
            return dek_keys[0].key_id
        
        # Create new DEK
        key_id, _ = self.key_manager.generate_key(
            KeyType.DATA_ENCRYPTION_KEY,
            EncryptionAlgorithm.AES_256_GCM,
            "Default data encryption key"
        )
        return key_id
    
    def encrypt_data(
        self,
        data: Union[str, bytes, dict],
        algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM,
        key_id: Optional[str] = None
    ) -> Tuple[bytes, EncryptionMetadata]:
        """Encrypt data using specified algorithm"""
        
        # Use default key if not specified
        if not key_id:
            key_id = self.default_dek_id
        
        # Get encryption key
        key_data = self.key_manager.get_key(key_id)
        if not key_data:
            raise ValueError(f"Encryption key {key_id} not found")
        
        # Convert data to bytes
        if isinstance(data, dict):
            data_bytes = json.dumps(data, separators=(',', ':')).encode('utf-8')
        elif isinstance(data, str):
            data_bytes = data.encode('utf-8')
        else:
            data_bytes = data
        
        # Encrypt based on algorithm
        if algorithm == EncryptionAlgorithm.AES_256_GCM:
            encrypted_data, metadata = self._encrypt_aes_gcm(data_bytes, key_data, key_id)
        elif algorithm == EncryptionAlgorithm.AES_256_CBC:
            encrypted_data, metadata = self._encrypt_aes_cbc(data_bytes, key_data, key_id)
        elif algorithm == EncryptionAlgorithm.FERNET:
            encrypted_data, metadata = self._encrypt_fernet(data_bytes, key_data, key_id)
        elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
            encrypted_data, metadata = self._encrypt_chacha20_poly1305(data_bytes, key_data, key_id)
        else:
            raise ValueError(f"Unsupported encryption algorithm: {algorithm}")
        
        metadata.encrypted_at = datetime.utcnow()
        return encrypted_data, metadata
    
    def decrypt_data(self, encrypted_data: bytes, metadata: EncryptionMetadata) -> bytes:
        """Decrypt data using metadata"""
        
        # Get decryption key
        key_data = self.key_manager.get_key(metadata.key_id)
        if not key_data:
            raise ValueError(f"Decryption key {metadata.key_id} not found")
        
        # Decrypt based on algorithm
        if metadata.algorithm == EncryptionAlgorithm.AES_256_GCM:
            return self._decrypt_aes_gcm(encrypted_data, key_data, metadata)
        elif metadata.algorithm == EncryptionAlgorithm.AES_256_CBC:
            return self._decrypt_aes_cbc(encrypted_data, key_data, metadata)
        elif metadata.algorithm == EncryptionAlgorithm.FERNET:
            return self._decrypt_fernet(encrypted_data, key_data, metadata)
        elif metadata.algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
            return self._decrypt_chacha20_poly1305(encrypted_data, key_data, metadata)
        else:
            raise ValueError(f"Unsupported decryption algorithm: {metadata.algorithm}")
    
    def _encrypt_aes_gcm(self, data: bytes, key: bytes, key_id: str) -> Tuple[bytes, EncryptionMetadata]:
        """Encrypt using AES-256-GCM"""
        iv = os.urandom(12)  # 96-bit IV for GCM
        
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        encrypted_data = encryptor.update(data) + encryptor.finalize()
        
        metadata = EncryptionMetadata(
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            key_id=key_id,
            iv=base64.b64encode(iv).decode(),
            tag=base64.b64encode(encryptor.tag).decode()
        )
        
        return encrypted_data, metadata
    
    def _decrypt_aes_gcm(self, encrypted_data: bytes, key: bytes, metadata: EncryptionMetadata) -> bytes:
        """Decrypt using AES-256-GCM"""
        iv = base64.b64decode(metadata.iv)
        tag = base64.b64decode(metadata.tag)
        
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(iv, tag),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        
        return decryptor.update(encrypted_data) + decryptor.finalize()
    
    def _encrypt_aes_cbc(self, data: bytes, key: bytes, key_id: str) -> Tuple[bytes, EncryptionMetadata]:
        """Encrypt using AES-256-CBC"""
        iv = os.urandom(16)  # 128-bit IV for CBC
        
        # Pad data to block size
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(data) + padder.finalize()
        
        cipher = Cipher(
            algorithms.AES(key),
            modes.CBC(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        
        metadata = EncryptionMetadata(
            algorithm=EncryptionAlgorithm.AES_256_CBC,
            key_id=key_id,
            iv=base64.b64encode(iv).decode()
        )
        
        return encrypted_data, metadata
    
    def _decrypt_aes_cbc(self, encrypted_data: bytes, key: bytes, metadata: EncryptionMetadata) -> bytes:
        """Decrypt using AES-256-CBC"""
        iv = base64.b64decode(metadata.iv)
        
        cipher = Cipher(
            algorithms.AES(key),
            modes.CBC(iv),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        padded_data = decryptor.update(encrypted_data) + decryptor.finalize()
        
        # Unpad data
        unpadder = padding.PKCS7(128).unpadder()
        return unpadder.update(padded_data) + unpadder.finalize()
    
    def _encrypt_fernet(self, data: bytes, key: bytes, key_id: str) -> Tuple[bytes, EncryptionMetadata]:
        """Encrypt using Fernet (AES-128-CBC + HMAC)"""
        fernet = Fernet(key)
        encrypted_data = fernet.encrypt(data)
        
        metadata = EncryptionMetadata(
            algorithm=EncryptionAlgorithm.FERNET,
            key_id=key_id
        )
        
        return encrypted_data, metadata
    
    def _decrypt_fernet(self, encrypted_data: bytes, key: bytes, metadata: EncryptionMetadata) -> bytes:
        """Decrypt using Fernet"""
        fernet = Fernet(key)
        return fernet.decrypt(encrypted_data)
    
    def _encrypt_chacha20_poly1305(self, data: bytes, key: bytes, key_id: str) -> Tuple[bytes, EncryptionMetadata]:
        """Encrypt using ChaCha20-Poly1305"""
        iv = os.urandom(12)  # 96-bit nonce
        
        cipher = Cipher(
            algorithms.ChaCha20(key, iv),
            None,
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        encrypted_data = encryptor.update(data) + encryptor.finalize()
        
        metadata = EncryptionMetadata(
            algorithm=EncryptionAlgorithm.CHACHA20_POLY1305,
            key_id=key_id,
            iv=base64.b64encode(iv).decode()
        )
        
        return encrypted_data, metadata
    
    def _decrypt_chacha20_poly1305(self, encrypted_data: bytes, key: bytes, metadata: EncryptionMetadata) -> bytes:
        """Decrypt using ChaCha20-Poly1305"""
        iv = base64.b64decode(metadata.iv)
        
        cipher = Cipher(
            algorithms.ChaCha20(key, iv),
            None,
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        return decryptor.update(encrypted_data) + decryptor.finalize()


# SQLAlchemy encrypted types
class EncryptedType(TypeDecorator):
    """SQLAlchemy type for encrypted fields"""
    
    impl = String
    cache_ok = True
    
    def __init__(self, algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM):
        self.algorithm = algorithm
        self.encryption_service = DataEncryption()
        super().__init__()
    
    def process_bind_param(self, value, dialect):
        """Encrypt value before storing in database"""
        if value is None:
            return None
        
        try:
            encrypted_data, metadata = self.encryption_service.encrypt_data(
                value, self.algorithm
            )
            
            # Store as base64 with metadata
            result = {
                "data": base64.b64encode(encrypted_data).decode(),
                "metadata": asdict(metadata, default=str)
            }
            return json.dumps(result)
            
        except Exception as e:
            logger.error(f"Encryption error: {e}")
            raise
    
    def process_result_value(self, value, dialect):
        """Decrypt value after retrieving from database"""
        if value is None:
            return None
        
        try:
            stored_data = json.loads(value)
            encrypted_data = base64.b64decode(stored_data["data"])
            
            # Reconstruct metadata
            meta_dict = stored_data["metadata"]
            metadata = EncryptionMetadata(
                algorithm=EncryptionAlgorithm(meta_dict["algorithm"]),
                key_id=meta_dict["key_id"],
                iv=meta_dict.get("iv"),
                tag=meta_dict.get("tag"),
                encrypted_at=datetime.fromisoformat(meta_dict["encrypted_at"]) if meta_dict.get("encrypted_at") else None,
                version=meta_dict.get("version", 1)
            )
            
            decrypted_data = self.encryption_service.decrypt_data(encrypted_data, metadata)
            return decrypted_data.decode('utf-8')
            
        except Exception as e:
            logger.error(f"Decryption error: {e}")
            raise


class EncryptedLargeBinaryType(TypeDecorator):
    """SQLAlchemy type for encrypted binary fields"""
    
    impl = LargeBinary
    cache_ok = True
    
    def __init__(self, algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM):
        self.algorithm = algorithm
        self.encryption_service = DataEncryption()
        super().__init__()
    
    def process_bind_param(self, value, dialect):
        """Encrypt binary value before storing"""
        if value is None:
            return None
        
        try:
            encrypted_data, metadata = self.encryption_service.encrypt_data(
                value, self.algorithm
            )
            
            # Create header with metadata
            metadata_json = json.dumps(asdict(metadata, default=str))
            header = len(metadata_json).to_bytes(4, 'big') + metadata_json.encode('utf-8')
            
            return header + encrypted_data
            
        except Exception as e:
            logger.error(f"Binary encryption error: {e}")
            raise
    
    def process_result_value(self, value, dialect):
        """Decrypt binary value after retrieving"""
        if value is None:
            return None
        
        try:
            # Extract metadata from header
            metadata_length = int.from_bytes(value[:4], 'big')
            metadata_json = value[4:4+metadata_length].decode('utf-8')
            encrypted_data = value[4+metadata_length:]
            
            # Reconstruct metadata
            meta_dict = json.loads(metadata_json)
            metadata = EncryptionMetadata(
                algorithm=EncryptionAlgorithm(meta_dict["algorithm"]),
                key_id=meta_dict["key_id"],
                iv=meta_dict.get("iv"),
                tag=meta_dict.get("tag"),
                encrypted_at=datetime.fromisoformat(meta_dict["encrypted_at"]) if meta_dict.get("encrypted_at") else None,
                version=meta_dict.get("version", 1)
            )
            
            return self.encryption_service.decrypt_data(encrypted_data, metadata)
            
        except Exception as e:
            logger.error(f"Binary decryption error: {e}")
            raise


class TLSManager:
    """TLS/SSL certificate and connection management"""
    
    def __init__(self, cert_path: str = "/etc/ssl/certs", key_path: str = "/etc/ssl/private"):
        self.cert_path = Path(cert_path)
        self.key_path = Path(key_path)
        self.cert_path.mkdir(parents=True, exist_ok=True)
        self.key_path.mkdir(parents=True, exist_ok=True)
    
    def generate_self_signed_cert(
        self,
        hostname: str,
        organization: str = "Investment Platform",
        validity_days: int = 365
    ) -> Tuple[str, str]:
        """Generate self-signed certificate for development"""
        
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        
        # Generate certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, organization),
            x509.NameAttribute(NameOID.COMMON_NAME, hostname),
        ])
        
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.utcnow()
        ).not_valid_after(
            datetime.utcnow() + timedelta(days=validity_days)
        ).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName(hostname),
                x509.DNSName(f"*.{hostname}"),
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
            ]),
            critical=False,
        ).sign(private_key, hashes.SHA256(), default_backend())
        
        # Save certificate and key
        cert_file = self.cert_path / f"{hostname}.crt"
        key_file = self.key_path / f"{hostname}.key"
        
        with open(cert_file, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        
        with open(key_file, "wb") as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
        
        # Set restrictive permissions
        os.chmod(key_file, 0o600)
        
        logger.info(f"Generated self-signed certificate for {hostname}")
        return str(cert_file), str(key_file)


class TransportEncryptionMiddleware(BaseHTTPMiddleware):
    """Middleware for transport-level encryption and security headers"""
    
    def __init__(self, app, force_https: bool = True, hsts_max_age: int = 31536000):
        super().__init__(app)
        self.force_https = force_https
        self.hsts_max_age = hsts_max_age
    
    async def dispatch(self, request: Request, call_next) -> StarletteResponse:
        """Add security headers and enforce HTTPS"""
        
        # Redirect to HTTPS if not already
        if self.force_https and request.url.scheme == "http":
            https_url = request.url.replace(scheme="https")
            return Response(
                status_code=301,
                headers={"Location": str(https_url)}
            )
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        security_headers = {
            # HTTPS Strict Transport Security
            "Strict-Transport-Security": f"max-age={self.hsts_max_age}; includeSubDomains; preload",
            
            # Prevent MIME type sniffing
            "X-Content-Type-Options": "nosniff",
            
            # XSS Protection
            "X-XSS-Protection": "1; mode=block",
            
            # Frame options
            "X-Frame-Options": "DENY",
            
            # Content Security Policy
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "font-src 'self' data:; "
                "connect-src 'self' wss: ws:; "
                "frame-ancestors 'none'; "
                "base-uri 'self'; "
                "form-action 'self'"
            ),
            
            # Referrer Policy
            "Referrer-Policy": "strict-origin-when-cross-origin",
            
            # Permissions Policy
            "Permissions-Policy": (
                "geolocation=(), "
                "microphone=(), "
                "camera=(), "
                "payment=(), "
                "usb=(), "
                "magnetometer=(), "
                "accelerometer=(), "
                "gyroscope=()"
            )
        }
        
        for header, value in security_headers.items():
            response.headers[header] = value
        
        return response


# Global instances
_key_manager: Optional[KeyManager] = None
_data_encryption: Optional[DataEncryption] = None


def get_key_manager() -> KeyManager:
    """Get global key manager instance"""
    global _key_manager
    if _key_manager is None:
        _key_manager = KeyManager()
    return _key_manager


def get_data_encryption() -> DataEncryption:
    """Get global data encryption instance"""
    global _data_encryption
    if _data_encryption is None:
        _data_encryption = DataEncryption()
    return _data_encryption


# Utility functions
def encrypt_sensitive_field(value: str, algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM) -> str:
    """Utility function to encrypt sensitive field data"""
    if not value:
        return value
    
    encryption = get_data_encryption()
    encrypted_data, metadata = encryption.encrypt_data(value, algorithm)
    
    # Return as base64 JSON
    result = {
        "data": base64.b64encode(encrypted_data).decode(),
        "metadata": asdict(metadata, default=str)
    }
    return json.dumps(result)


def decrypt_sensitive_field(encrypted_value: str) -> str:
    """Utility function to decrypt sensitive field data"""
    if not encrypted_value:
        return encrypted_value
    
    try:
        stored_data = json.loads(encrypted_value)
        encrypted_data = base64.b64decode(stored_data["data"])
        
        # Reconstruct metadata
        meta_dict = stored_data["metadata"]
        metadata = EncryptionMetadata(
            algorithm=EncryptionAlgorithm(meta_dict["algorithm"]),
            key_id=meta_dict["key_id"],
            iv=meta_dict.get("iv"),
            tag=meta_dict.get("tag"),
            encrypted_at=datetime.fromisoformat(meta_dict["encrypted_at"]) if meta_dict.get("encrypted_at") else None,
            version=meta_dict.get("version", 1)
        )
        
        encryption = get_data_encryption()
        decrypted_data = encryption.decrypt_data(encrypted_data, metadata)
        return decrypted_data.decode('utf-8')
        
    except Exception as e:
        logger.error(f"Field decryption error: {e}")
        return encrypted_value  # Return original if decryption fails