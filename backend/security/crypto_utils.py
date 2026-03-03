"""
Cryptographic Utilities Module

Provides symmetric encryption (Fernet/AES-128-CBC), asymmetric key generation
(RSA-2048), digital signatures (RSA-PSS + SHA-256), and secure random generation.
All cryptographic operations use the ``cryptography`` library.
"""

import secrets
import hashlib
from typing import Optional, Tuple

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization


class SecureRandom:
    """Cryptographically secure random number generation"""

    @staticmethod
    def generate_token(length: int = 32) -> str:
        """Generate secure random token"""
        return secrets.token_urlsafe(length)

    @staticmethod
    def generate_hex(length: int = 32) -> str:
        """Generate secure random hex string"""
        return secrets.token_hex(length)

    @staticmethod
    def generate_bytes(length: int = 32) -> bytes:
        """Generate secure random bytes"""
        return secrets.token_bytes(length)

    @staticmethod
    def generate_int(min_value: int, max_value: int) -> int:
        """Generate secure random integer in range"""
        return secrets.randbelow(max_value - min_value + 1) + min_value

    @staticmethod
    def generate_uuid() -> str:
        """Generate random UUID"""
        return secrets.token_hex(16)


class CryptoUtils:
    """Cryptographic utility functions using the ``cryptography`` library."""

    @staticmethod
    def hash_data(data: bytes, algorithm: str = "sha256") -> str:
        """Hash data with specified algorithm"""
        if algorithm == "sha256":
            return hashlib.sha256(data).hexdigest()
        elif algorithm == "sha512":
            return hashlib.sha512(data).hexdigest()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    @staticmethod
    def encrypt_data(data: bytes, key: bytes) -> bytes:
        """Encrypt data using Fernet symmetric encryption (AES-128-CBC + HMAC).

        ``key`` must be a 32-byte URL-safe base64-encoded Fernet key.
        Generate one with ``Fernet.generate_key()``.
        """
        f = Fernet(key)
        return f.encrypt(data)

    @staticmethod
    def decrypt_data(encrypted: bytes, key: bytes) -> bytes:
        """Decrypt Fernet-encrypted data."""
        f = Fernet(key)
        return f.decrypt(encrypted)

    @staticmethod
    def generate_key_pair() -> Tuple[bytes, bytes]:
        """Generate an RSA-2048 key pair.

        Returns (private_key_pem, public_key_pem) as PEM-encoded bytes.
        """
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        public_pem = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        return private_pem, public_pem

    @staticmethod
    def sign_data(data: bytes, private_key: bytes) -> bytes:
        """Sign data with an RSA private key using PSS padding + SHA-256.

        ``private_key`` must be PEM-encoded (as returned by generate_key_pair).
        """
        key = serialization.load_pem_private_key(private_key, password=None)
        return key.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )

    @staticmethod
    def verify_signature(data: bytes, signature: bytes, public_key: bytes) -> bool:
        """Verify an RSA-PSS signature.

        ``public_key`` must be PEM-encoded (as returned by generate_key_pair).
        Returns True if valid, False otherwise.
        """
        key = serialization.load_pem_public_key(public_key)
        try:
            key.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )
            return True
        except Exception:
            return False
