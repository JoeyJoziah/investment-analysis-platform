"""
Enhanced JWT Authentication Manager with RS256 and Token Blacklisting

This module provides secure JWT token management with:
- RS256 asymmetric encryption instead of HS256
- Token blacklisting and revocation
- Multi-factor authentication support
- Secure token validation and refresh mechanisms
"""

import os
import json
import jwt
import redis
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Set, Any
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import hashlib
import secrets
import pyotp
from pathlib import Path

from backend.security.secrets_manager import get_secrets_manager, SecretType
from backend.config.settings import settings

logger = logging.getLogger(__name__)


class TokenType(str, Enum):
    """Types of JWT tokens"""
    ACCESS = "access"
    REFRESH = "refresh"
    RESET = "reset"
    MFA = "mfa"


@dataclass
class TokenClaims:
    """Standard JWT claims plus custom claims"""
    user_id: int
    username: str
    email: str
    roles: List[str]
    scopes: List[str]
    is_admin: bool = False
    is_mfa_verified: bool = False
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class JWTManager:
    """
    Enhanced JWT Manager with RS256 encryption and token blacklisting.
    
    Features:
    - RS256 asymmetric encryption for better security
    - Token blacklisting with Redis backend
    - MFA integration
    - Session management
    - Token introspection and validation
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.secrets_manager = get_secrets_manager()
        self.redis_client = redis_client or self._get_redis_client()
        
        # Initialize RSA key pair
        self.private_key, self.public_key = self._initialize_rsa_keys()
        
        # Token settings
        self.access_token_expire_minutes = settings.ACCESS_TOKEN_EXPIRE_MINUTES
        self.refresh_token_expire_days = settings.REFRESH_TOKEN_EXPIRE_DAYS
        
        # Blacklist key prefix
        self.blacklist_prefix = "jwt_blacklist"
        self.session_prefix = "user_session"
    
    def _get_redis_client(self) -> redis.Redis:
        """Get Redis client for token blacklisting"""
        try:
            redis_url = settings.REDIS_URL
            return redis.from_url(redis_url, decode_responses=True)
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            # Fallback to in-memory storage (not recommended for production)
            return None
    
    def _initialize_rsa_keys(self) -> tuple:
        """Initialize or load RSA key pair for JWT signing"""
        try:
            # Try to load existing keys from secrets manager
            private_key_pem = self.secrets_manager.get_secret("jwt_private_key")
            public_key_pem = self.secrets_manager.get_secret("jwt_public_key")
            
            if private_key_pem and public_key_pem:
                private_key = serialization.load_pem_private_key(
                    private_key_pem.encode(),
                    password=None,
                    backend=default_backend()
                )
                public_key = serialization.load_pem_public_key(
                    public_key_pem.encode(),
                    backend=default_backend()
                )
                logger.info("Loaded existing RSA keys for JWT")
                return private_key, public_key
            
            # Generate new RSA key pair
            logger.info("Generating new RSA key pair for JWT")
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            public_key = private_key.public_key()
            
            # Serialize keys
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ).decode()
            
            public_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ).decode()
            
            # Store keys in secrets manager
            self.secrets_manager.store_secret(
                "jwt_private_key",
                private_pem,
                SecretType.JWT_KEY,
                description="RSA private key for JWT signing"
            )
            self.secrets_manager.store_secret(
                "jwt_public_key",
                public_pem,
                SecretType.JWT_KEY,
                description="RSA public key for JWT verification"
            )
            
            logger.info("Generated and stored new RSA keys for JWT")
            return private_key, public_key
            
        except Exception as e:
            logger.error(f"Failed to initialize RSA keys: {e}")
            raise
    
    def create_access_token(
        self,
        claims: TokenClaims,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create a new access token with RS256 signing.
        
        Args:
            claims: Token claims containing user information
            expires_delta: Optional custom expiration time
            
        Returns:
            Encoded JWT token
        """
        try:
            # Set expiration
            if expires_delta:
                expire = datetime.utcnow() + expires_delta
            else:
                expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
            
            # Create session ID if not provided
            if not claims.session_id:
                claims.session_id = secrets.token_urlsafe(32)
            
            # Build JWT payload
            payload = {
                "sub": claims.username,
                "user_id": claims.user_id,
                "email": claims.email,
                "roles": claims.roles,
                "scopes": claims.scopes,
                "is_admin": claims.is_admin,
                "is_mfa_verified": claims.is_mfa_verified,
                "session_id": claims.session_id,
                "ip_address": claims.ip_address,
                "user_agent": claims.user_agent,
                "type": TokenType.ACCESS.value,
                "iat": datetime.utcnow(),
                "exp": expire,
                "iss": "investment-analysis-app",
                "aud": "investment-analysis-users"
            }
            
            # Sign with RS256
            token = jwt.encode(
                payload,
                self.private_key,
                algorithm="RS256"
            )
            
            # Store session information
            if self.redis_client:
                session_key = f"{self.session_prefix}:{claims.user_id}:{claims.session_id}"
                session_data = {
                    "user_id": claims.user_id,
                    "username": claims.username,
                    "ip_address": claims.ip_address,
                    "user_agent": claims.user_agent,
                    "created_at": datetime.utcnow().isoformat(),
                    "expires_at": expire.isoformat()
                }
                self.redis_client.hset(session_key, mapping=session_data)
                self.redis_client.expire(session_key, int(expires_delta.total_seconds()) if expires_delta else self.access_token_expire_minutes * 60)
            
            logger.debug(f"Created access token for user {claims.username}")
            return token
            
        except Exception as e:
            logger.error(f"Failed to create access token: {e}")
            raise
    
    def create_refresh_token(
        self,
        claims: TokenClaims,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create a refresh token for token renewal.
        
        Args:
            claims: Token claims containing user information
            expires_delta: Optional custom expiration time
            
        Returns:
            Encoded refresh token
        """
        try:
            # Set expiration
            if expires_delta:
                expire = datetime.utcnow() + expires_delta
            else:
                expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
            
            # Build minimal payload for refresh token
            payload = {
                "sub": claims.username,
                "user_id": claims.user_id,
                "session_id": claims.session_id,
                "type": TokenType.REFRESH.value,
                "iat": datetime.utcnow(),
                "exp": expire,
                "iss": "investment-analysis-app",
                "aud": "investment-analysis-users"
            }
            
            # Sign with RS256
            token = jwt.encode(
                payload,
                self.private_key,
                algorithm="RS256"
            )
            
            logger.debug(f"Created refresh token for user {claims.username}")
            return token
            
        except Exception as e:
            logger.error(f"Failed to create refresh token: {e}")
            raise
    
    def verify_token(self, token: str, token_type: TokenType = TokenType.ACCESS) -> Optional[Dict[str, Any]]:
        """
        Verify and decode a JWT token.
        
        Args:
            token: JWT token to verify
            token_type: Expected token type
            
        Returns:
            Decoded token payload or None if invalid
        """
        try:
            # Check if token is blacklisted
            if self._is_token_blacklisted(token):
                logger.warning("Attempted to use blacklisted token")
                return None
            
            # Verify and decode token
            payload = jwt.decode(
                token,
                self.public_key,
                algorithms=["RS256"],
                issuer="investment-analysis-app",
                audience="investment-analysis-users"
            )
            
            # Verify token type
            if payload.get("type") != token_type.value:
                logger.warning(f"Token type mismatch: expected {token_type.value}, got {payload.get('type')}")
                return None
            
            # Verify session is still active (for access tokens)
            if token_type == TokenType.ACCESS and self.redis_client:
                session_id = payload.get("session_id")
                user_id = payload.get("user_id")
                if session_id and user_id:
                    session_key = f"{self.session_prefix}:{user_id}:{session_id}"
                    if not self.redis_client.exists(session_key):
                        logger.warning("Token session no longer exists")
                        return None
            
            logger.debug(f"Successfully verified {token_type.value} token for user {payload.get('sub')}")
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
        except Exception as e:
            logger.error(f"Error verifying token: {e}")
            return None
    
    def revoke_token(self, token: str) -> bool:
        """
        Revoke (blacklist) a token.
        
        Args:
            token: JWT token to revoke
            
        Returns:
            True if successfully revoked
        """
        try:
            # Decode token to get expiration (don't verify signature for revocation)
            unverified_payload = jwt.decode(token, options={"verify_signature": False})
            
            # Calculate TTL based on token expiration
            exp = unverified_payload.get("exp")
            if exp:
                exp_time = datetime.fromtimestamp(exp)
                ttl = int((exp_time - datetime.utcnow()).total_seconds())
                if ttl <= 0:
                    # Token already expired, no need to blacklist
                    return True
            else:
                # Default TTL if no expiration
                ttl = 86400  # 24 hours
            
            # Add to blacklist
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            blacklist_key = f"{self.blacklist_prefix}:{token_hash}"
            
            if self.redis_client:
                self.redis_client.setex(blacklist_key, ttl, "1")
                logger.info(f"Token revoked and added to blacklist")
            else:
                # Fallback to in-memory storage (not persistent)
                logger.warning("Redis not available, token revocation not persistent")
            
            # Revoke session if applicable
            session_id = unverified_payload.get("session_id")
            user_id = unverified_payload.get("user_id")
            if session_id and user_id and self.redis_client:
                session_key = f"{self.session_prefix}:{user_id}:{session_id}"
                self.redis_client.delete(session_key)
            
            return True
            
        except Exception as e:
            logger.error(f"Error revoking token: {e}")
            return False
    
    def _is_token_blacklisted(self, token: str) -> bool:
        """Check if a token is in the blacklist"""
        try:
            if not self.redis_client:
                return False
            
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            blacklist_key = f"{self.blacklist_prefix}:{token_hash}"
            return self.redis_client.exists(blacklist_key)
            
        except Exception as e:
            logger.error(f"Error checking token blacklist: {e}")
            return False
    
    def revoke_all_user_tokens(self, user_id: int) -> bool:
        """
        Revoke all active tokens for a user.
        
        Args:
            user_id: ID of the user whose tokens to revoke
            
        Returns:
            True if successful
        """
        try:
            if not self.redis_client:
                logger.warning("Redis not available, cannot revoke user sessions")
                return False
            
            # Find all user sessions
            session_pattern = f"{self.session_prefix}:{user_id}:*"
            session_keys = self.redis_client.keys(session_pattern)
            
            # Delete all sessions
            if session_keys:
                self.redis_client.delete(*session_keys)
                logger.info(f"Revoked {len(session_keys)} sessions for user {user_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error revoking user tokens: {e}")
            return False
    
    def refresh_access_token(self, refresh_token: str, new_claims: Optional[TokenClaims] = None) -> Optional[str]:
        """
        Create a new access token using a refresh token.
        
        Args:
            refresh_token: Valid refresh token
            new_claims: Optional updated claims
            
        Returns:
            New access token or None if refresh token is invalid
        """
        try:
            # Verify refresh token
            payload = self.verify_token(refresh_token, TokenType.REFRESH)
            if not payload:
                return None
            
            # Create new claims from refresh token
            if not new_claims:
                new_claims = TokenClaims(
                    user_id=payload["user_id"],
                    username=payload["sub"],
                    email="",  # Will need to be fetched from database
                    roles=[],  # Will need to be fetched from database
                    scopes=[],  # Will need to be fetched from database
                    session_id=payload.get("session_id")
                )
            
            # Create new access token
            new_access_token = self.create_access_token(new_claims)
            
            logger.debug(f"Refreshed access token for user {new_claims.username}")
            return new_access_token
            
        except Exception as e:
            logger.error(f"Error refreshing access token: {e}")
            return None
    
    def get_user_sessions(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Get all active sessions for a user.
        
        Args:
            user_id: ID of the user
            
        Returns:
            List of active session information
        """
        try:
            if not self.redis_client:
                return []
            
            sessions = []
            session_pattern = f"{self.session_prefix}:{user_id}:*"
            session_keys = self.redis_client.keys(session_pattern)
            
            for session_key in session_keys:
                session_data = self.redis_client.hgetall(session_key)
                if session_data:
                    sessions.append(session_data)
            
            return sessions
            
        except Exception as e:
            logger.error(f"Error getting user sessions: {e}")
            return []
    
    def generate_mfa_secret(self, username: str) -> str:
        """Generate a TOTP secret for MFA"""
        secret = pyotp.random_base32()
        
        # Store MFA secret
        self.secrets_manager.store_secret(
            f"mfa_secret_{username}",
            secret,
            SecretType.ENCRYPTION_KEY,
            description=f"MFA TOTP secret for {username}"
        )
        
        return secret
    
    def verify_mfa_token(self, username: str, token: str, window: int = 1) -> bool:
        """Verify MFA TOTP token"""
        try:
            secret = self.secrets_manager.get_secret(f"mfa_secret_{username}")
            if not secret:
                return False
            
            totp = pyotp.TOTP(secret)
            return totp.verify(token, valid_window=window)
            
        except Exception as e:
            logger.error(f"Error verifying MFA token: {e}")
            return False
    
    def get_public_key_jwks(self) -> Dict[str, Any]:
        """
        Get public key in JWKS format for token verification by other services.
        
        Returns:
            JWKS formatted public key
        """
        try:
            # Convert public key to numbers for JWKS
            public_numbers = self.public_key.public_numbers()
            
            # Convert to base64url encoding
            def int_to_base64url(value):
                byte_length = (value.bit_length() + 7) // 8
                value_bytes = value.to_bytes(byte_length, byteorder='big')
                import base64
                return base64.urlsafe_b64encode(value_bytes).decode('ascii').rstrip('=')
            
            jwks = {
                "keys": [
                    {
                        "kty": "RSA",
                        "use": "sig",
                        "alg": "RS256",
                        "n": int_to_base64url(public_numbers.n),
                        "e": int_to_base64url(public_numbers.e),
                        "kid": "investment-app-key-1"
                    }
                ]
            }
            
            return jwks
            
        except Exception as e:
            logger.error(f"Error generating JWKS: {e}")
            return {"keys": []}


# Global JWT manager instance
_jwt_manager: Optional[JWTManager] = None


def get_jwt_manager() -> JWTManager:
    """Get or create the global JWT manager instance"""
    global _jwt_manager
    if _jwt_manager is None:
        _jwt_manager = JWTManager()
    return _jwt_manager