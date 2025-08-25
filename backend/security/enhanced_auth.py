"""
Enhanced Authentication and Authorization System
Implements OAuth2, JWT, RBAC, and multi-factor authentication
"""

import os
import jwt
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Union
from dataclasses import dataclass
from enum import Enum
import hashlib
import pyotp
import qrcode
from io import BytesIO
import base64
from fastapi import HTTPException, Depends, Request, status
from fastapi.security import OAuth2PasswordBearer, HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
from passlib.hash import argon2
import redis
import logging
from pydantic import BaseModel, EmailStr, validator
import asyncio
import aioredis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

from .secrets_vault import get_secrets_vault

logger = logging.getLogger(__name__)


class UserRole(str, Enum):
    """User roles with hierarchical permissions"""
    SUPER_ADMIN = "super_admin"
    ADMIN = "admin"
    ANALYST = "analyst"
    TRADER = "trader"
    VIEWER = "viewer"
    SERVICE_ACCOUNT = "service_account"


class AuthProvider(str, Enum):
    """Authentication providers"""
    LOCAL = "local"
    GOOGLE = "google"
    MICROSOFT = "microsoft"
    GITHUB = "github"
    OKTA = "okta"


class SessionStatus(str, Enum):
    """Session status types"""
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    SUSPICIOUS = "suspicious"


@dataclass
class Permission:
    """Permission definition"""
    resource: str
    action: str
    conditions: Optional[Dict[str, Any]] = None


class TokenType(str, Enum):
    """Token types for different purposes"""
    ACCESS = "access"
    REFRESH = "refresh"
    MFA = "mfa"
    RESET = "reset"
    API = "api"


class AuthenticationError(HTTPException):
    """Custom authentication error"""
    def __init__(self, detail: str = "Authentication failed"):
        super().__init__(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)


class AuthorizationError(HTTPException):
    """Custom authorization error"""
    def __init__(self, detail: str = "Access denied"):
        super().__init__(status_code=status.HTTP_403_FORBIDDEN, detail=detail)


class UserRegistrationRequest(BaseModel):
    """User registration request model"""
    username: str
    email: EmailStr
    password: str
    full_name: str
    role: UserRole = UserRole.VIEWER
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 12:
            raise ValueError('Password must be at least 12 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        if not any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in v):
            raise ValueError('Password must contain at least one special character')
        return v


class LoginRequest(BaseModel):
    """Login request model"""
    username: str
    password: str
    mfa_token: Optional[str] = None
    remember_me: bool = False


class TokenResponse(BaseModel):
    """Token response model"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    scope: str
    user_id: str
    role: str


@dataclass
class UserSession:
    """User session information"""
    session_id: str
    user_id: str
    username: str
    role: UserRole
    created_at: datetime
    last_activity: datetime
    ip_address: str
    user_agent: str
    status: SessionStatus
    permissions: List[Permission]
    mfa_verified: bool = False
    api_key_id: Optional[str] = None


class PasswordHasher:
    """Secure password hashing with Argon2"""
    
    def __init__(self):
        self.context = CryptContext(
            schemes=["argon2"],
            deprecated="auto",
            argon2__time_cost=3,
            argon2__memory_cost=65536,
            argon2__parallelism=1
        )
    
    def hash_password(self, password: str) -> str:
        """Hash password using Argon2"""
        return self.context.hash(password)
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return self.context.verify(password, hashed)
    
    def needs_update(self, hashed: str) -> bool:
        """Check if password hash needs updating"""
        return self.context.needs_update(hashed)


class MFAManager:
    """Multi-Factor Authentication manager"""
    
    def __init__(self):
        self.app_name = "Investment Platform"
    
    def generate_secret(self) -> str:
        """Generate TOTP secret for user"""
        return pyotp.random_base32()
    
    def generate_qr_code(self, user_email: str, secret: str) -> str:
        """Generate QR code for TOTP setup"""
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=user_email,
            issuer_name=self.app_name
        )
        
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(totp_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        
        return base64.b64encode(buffer.getvalue()).decode()
    
    def verify_token(self, secret: str, token: str) -> bool:
        """Verify TOTP token"""
        totp = pyotp.TOTP(secret)
        return totp.verify(token, valid_window=1)
    
    def generate_backup_codes(self, count: int = 8) -> List[str]:
        """Generate backup codes for MFA"""
        return [secrets.token_hex(4).upper() for _ in range(count)]


class JWTManager:
    """JWT token management with enhanced security"""
    
    def __init__(self):
        self.vault = get_secrets_vault()
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 15
        self.refresh_token_expire_days = 7
        self.mfa_token_expire_minutes = 5
    
    async def _get_secret_key(self) -> str:
        """Get JWT secret from vault"""
        secret = await self.vault.get_secret("jwt_secret_key")
        if not secret:
            # Generate new secret if not exists
            new_secret = secrets.token_urlsafe(64)
            await self.vault.store_secret(
                "jwt_secret_key", 
                new_secret, 
                secret_type="jwt_secret"
            )
            return new_secret
        return secret
    
    async def create_token(
        self,
        user_id: str,
        username: str,
        role: str,
        token_type: TokenType = TokenType.ACCESS,
        expires_delta: Optional[timedelta] = None,
        additional_claims: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create JWT token with custom claims"""
        secret_key = await self._get_secret_key()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            if token_type == TokenType.ACCESS:
                expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
            elif token_type == TokenType.REFRESH:
                expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
            elif token_type == TokenType.MFA:
                expire = datetime.utcnow() + timedelta(minutes=self.mfa_token_expire_minutes)
            else:
                expire = datetime.utcnow() + timedelta(hours=1)
        
        # Standard claims
        claims = {
            "sub": user_id,
            "username": username,
            "role": role,
            "token_type": token_type.value,
            "exp": expire,
            "iat": datetime.utcnow(),
            "jti": str(uuid.uuid4()),  # JWT ID for token tracking
            "iss": "investment-platform",
            "aud": "investment-platform-users"
        }
        
        # Add additional claims
        if additional_claims:
            claims.update(additional_claims)
        
        return jwt.encode(claims, secret_key, algorithm=self.algorithm)
    
    async def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            secret_key = await self._get_secret_key()
            payload = jwt.decode(
                token, 
                secret_key, 
                algorithms=[self.algorithm],
                audience="investment-platform-users",
                issuer="investment-platform"
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid token: {str(e)}")
    
    async def refresh_access_token(self, refresh_token: str) -> Dict[str, str]:
        """Refresh access token using refresh token"""
        payload = await self.verify_token(refresh_token)
        
        if payload.get("token_type") != TokenType.REFRESH.value:
            raise AuthenticationError("Invalid refresh token")
        
        # Create new access token
        access_token = await self.create_token(
            user_id=payload["sub"],
            username=payload["username"],
            role=payload["role"],
            token_type=TokenType.ACCESS
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer"
        }


class SessionManager:
    """Session management with Redis backend"""
    
    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.session_timeout = 3600  # 1 hour default
        self.max_sessions_per_user = 5
        self._redis_pool = None
    
    async def _get_redis(self) -> aioredis.Redis:
        """Get Redis connection"""
        if not self._redis_pool:
            self._redis_pool = aioredis.from_url(self.redis_url)
        return self._redis_pool
    
    async def create_session(self, user_session: UserSession) -> str:
        """Create new user session"""
        redis = await self._get_redis()
        
        # Check for existing sessions and limit
        user_sessions = await self.get_user_sessions(user_session.user_id)
        if len(user_sessions) >= self.max_sessions_per_user:
            # Remove oldest session
            oldest_session = min(user_sessions, key=lambda s: s.created_at)
            await self.revoke_session(oldest_session.session_id)
        
        # Store session
        session_key = f"session:{user_session.session_id}"
        session_data = {
            "user_id": user_session.user_id,
            "username": user_session.username,
            "role": user_session.role.value,
            "created_at": user_session.created_at.isoformat(),
            "last_activity": user_session.last_activity.isoformat(),
            "ip_address": user_session.ip_address,
            "user_agent": user_session.user_agent,
            "status": user_session.status.value,
            "mfa_verified": user_session.mfa_verified,
            "api_key_id": user_session.api_key_id
        }
        
        await redis.hset(session_key, mapping=session_data)
        await redis.expire(session_key, self.session_timeout)
        
        # Track user sessions
        user_sessions_key = f"user_sessions:{user_session.user_id}"
        await redis.sadd(user_sessions_key, user_session.session_id)
        await redis.expire(user_sessions_key, self.session_timeout)
        
        logger.info(f"Created session {user_session.session_id} for user {user_session.user_id}")
        return user_session.session_id
    
    async def get_session(self, session_id: str) -> Optional[UserSession]:
        """Get session by ID"""
        redis = await self._get_redis()
        session_key = f"session:{session_id}"
        
        session_data = await redis.hgetall(session_key)
        if not session_data:
            return None
        
        # Convert bytes to strings
        session_data = {k.decode(): v.decode() for k, v in session_data.items()}
        
        return UserSession(
            session_id=session_id,
            user_id=session_data["user_id"],
            username=session_data["username"],
            role=UserRole(session_data["role"]),
            created_at=datetime.fromisoformat(session_data["created_at"]),
            last_activity=datetime.fromisoformat(session_data["last_activity"]),
            ip_address=session_data["ip_address"],
            user_agent=session_data["user_agent"],
            status=SessionStatus(session_data["status"]),
            mfa_verified=session_data.get("mfa_verified", "False") == "True",
            api_key_id=session_data.get("api_key_id"),
            permissions=[]  # Load permissions separately if needed
        )
    
    async def update_session_activity(self, session_id: str) -> bool:
        """Update session last activity"""
        redis = await self._get_redis()
        session_key = f"session:{session_id}"
        
        result = await redis.hset(
            session_key, 
            "last_activity", 
            datetime.utcnow().isoformat()
        )
        await redis.expire(session_key, self.session_timeout)
        
        return bool(result)
    
    async def revoke_session(self, session_id: str) -> bool:
        """Revoke a session"""
        redis = await self._get_redis()
        session_key = f"session:{session_id}"
        
        # Get session to remove from user sessions
        session = await self.get_session(session_id)
        if session:
            user_sessions_key = f"user_sessions:{session.user_id}"
            await redis.srem(user_sessions_key, session_id)
        
        # Delete session
        result = await redis.delete(session_key)
        
        logger.info(f"Revoked session {session_id}")
        return bool(result)
    
    async def get_user_sessions(self, user_id: str) -> List[UserSession]:
        """Get all sessions for a user"""
        redis = await self._get_redis()
        user_sessions_key = f"user_sessions:{user_id}"
        
        session_ids = await redis.smembers(user_sessions_key)
        sessions = []
        
        for session_id in session_ids:
            session = await self.get_session(session_id.decode())
            if session:
                sessions.append(session)
        
        return sessions
    
    async def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        redis = await self._get_redis()
        
        # This is handled automatically by Redis TTL
        # But we can also implement custom cleanup logic here
        pass


class RBACManager:
    """Role-Based Access Control manager"""
    
    def __init__(self):
        self.role_hierarchy = {
            UserRole.SUPER_ADMIN: [UserRole.ADMIN, UserRole.ANALYST, UserRole.TRADER, UserRole.VIEWER],
            UserRole.ADMIN: [UserRole.ANALYST, UserRole.TRADER, UserRole.VIEWER],
            UserRole.ANALYST: [UserRole.VIEWER],
            UserRole.TRADER: [UserRole.VIEWER],
            UserRole.VIEWER: [],
            UserRole.SERVICE_ACCOUNT: []
        }
        
        self.permissions = self._define_permissions()
    
    def _define_permissions(self) -> Dict[UserRole, List[Permission]]:
        """Define permissions for each role"""
        return {
            UserRole.SUPER_ADMIN: [
                Permission("*", "*"),  # All permissions
            ],
            UserRole.ADMIN: [
                Permission("users", "read"),
                Permission("users", "create"),
                Permission("users", "update"),
                Permission("users", "delete"),
                Permission("portfolios", "*"),
                Permission("recommendations", "*"),
                Permission("analysis", "*"),
                Permission("system", "read"),
                Permission("system", "update"),
                Permission("cache", "*"),
                Permission("audit", "*"),
            ],
            UserRole.ANALYST: [
                Permission("portfolios", "read"),
                Permission("portfolios", "create"),
                Permission("portfolios", "update"),
                Permission("recommendations", "read"),
                Permission("recommendations", "create"),
                Permission("analysis", "*"),
                Permission("stocks", "*"),
            ],
            UserRole.TRADER: [
                Permission("portfolios", "read"),
                Permission("portfolios", "update"),
                Permission("recommendations", "read"),
                Permission("analysis", "read"),
                Permission("stocks", "read"),
                Permission("trades", "*"),
            ],
            UserRole.VIEWER: [
                Permission("portfolios", "read"),
                Permission("recommendations", "read"),
                Permission("analysis", "read"),
                Permission("stocks", "read"),
            ],
            UserRole.SERVICE_ACCOUNT: [
                Permission("api", "*"),
                Permission("system", "read"),
            ],
        }
    
    def has_permission(self, user_role: UserRole, resource: str, action: str) -> bool:
        """Check if user role has permission for resource/action"""
        user_permissions = self.permissions.get(user_role, [])
        
        for permission in user_permissions:
            if (permission.resource == "*" or permission.resource == resource) and \
               (permission.action == "*" or permission.action == action):
                return True
        
        # Check inherited permissions from hierarchy
        inherited_roles = self.role_hierarchy.get(user_role, [])
        for inherited_role in inherited_roles:
            if self.has_permission(inherited_role, resource, action):
                return True
        
        return False
    
    def get_user_permissions(self, user_role: UserRole) -> List[Permission]:
        """Get all permissions for a user role"""
        permissions = self.permissions.get(user_role, []).copy()
        
        # Add inherited permissions
        inherited_roles = self.role_hierarchy.get(user_role, [])
        for inherited_role in inherited_roles:
            permissions.extend(self.permissions.get(inherited_role, []))
        
        return permissions


class EnhancedAuthManager:
    """Enhanced authentication manager with all security features"""
    
    def __init__(self):
        self.password_hasher = PasswordHasher()
        self.mfa_manager = MFAManager()
        self.jwt_manager = JWTManager()
        self.session_manager = SessionManager()
        self.rbac_manager = RBACManager()
        self.vault = get_secrets_vault()
    
    async def register_user(self, registration: UserRegistrationRequest) -> Dict[str, Any]:
        """Register a new user with security validation"""
        # Hash password
        hashed_password = self.password_hasher.hash_password(registration.password)
        
        # Generate MFA secret
        mfa_secret = self.mfa_manager.generate_secret()
        
        # Store user data securely (implementation depends on your database)
        user_data = {
            "username": registration.username,
            "email": registration.email,
            "hashed_password": hashed_password,
            "full_name": registration.full_name,
            "role": registration.role.value,
            "mfa_secret": mfa_secret,
            "created_at": datetime.utcnow().isoformat(),
            "is_active": True,
            "email_verified": False,
            "mfa_enabled": False
        }
        
        # Generate QR code for MFA setup
        qr_code = self.mfa_manager.generate_qr_code(registration.email, mfa_secret)
        backup_codes = self.mfa_manager.generate_backup_codes()
        
        logger.info(f"User {registration.username} registered successfully")
        
        return {
            "user_id": str(uuid.uuid4()),  # Generate user ID
            "username": registration.username,
            "email": registration.email,
            "role": registration.role.value,
            "mfa_qr_code": qr_code,
            "backup_codes": backup_codes
        }
    
    async def authenticate_user(
        self, 
        login_request: LoginRequest, 
        request: Request
    ) -> TokenResponse:
        """Authenticate user and create session"""
        # This would typically query your user database
        # For demo, we'll assume user validation is handled elsewhere
        
        user_id = str(uuid.uuid4())  # Would come from database
        role = UserRole.VIEWER  # Would come from database
        
        # Create session
        session = UserSession(
            session_id=str(uuid.uuid4()),
            user_id=user_id,
            username=login_request.username,
            role=role,
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            ip_address=request.client.host,
            user_agent=request.headers.get("User-Agent", ""),
            status=SessionStatus.ACTIVE,
            permissions=self.rbac_manager.get_user_permissions(role),
            mfa_verified=login_request.mfa_token is not None
        )
        
        session_id = await self.session_manager.create_session(session)
        
        # Create tokens
        access_token = await self.jwt_manager.create_token(
            user_id=user_id,
            username=login_request.username,
            role=role.value,
            token_type=TokenType.ACCESS,
            additional_claims={"session_id": session_id}
        )
        
        refresh_token = await self.jwt_manager.create_token(
            user_id=user_id,
            username=login_request.username,
            role=role.value,
            token_type=TokenType.REFRESH,
            additional_claims={"session_id": session_id}
        )
        
        logger.info(f"User {login_request.username} authenticated successfully")
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=900,  # 15 minutes
            scope="read write",
            user_id=user_id,
            role=role.value
        )
    
    async def verify_permission(
        self, 
        token: str, 
        resource: str, 
        action: str
    ) -> UserSession:
        """Verify token and check permissions"""
        # Verify JWT token
        payload = await self.jwt_manager.verify_token(token)
        
        # Get session
        session_id = payload.get("session_id")
        if not session_id:
            raise AuthenticationError("Invalid session")
        
        session = await self.session_manager.get_session(session_id)
        if not session or session.status != SessionStatus.ACTIVE:
            raise AuthenticationError("Session expired or invalid")
        
        # Check permissions
        if not self.rbac_manager.has_permission(session.role, resource, action):
            raise AuthorizationError(f"Access denied to {resource}:{action}")
        
        # Update session activity
        await self.session_manager.update_session_activity(session_id)
        
        return session


# Global auth manager instance
_auth_manager: Optional[EnhancedAuthManager] = None


def get_auth_manager() -> EnhancedAuthManager:
    """Get or create global auth manager instance"""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = EnhancedAuthManager()
    return _auth_manager


# FastAPI dependencies
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")
security = HTTPBearer()


async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserSession:
    """FastAPI dependency to get current authenticated user"""
    auth_manager = get_auth_manager()
    payload = await auth_manager.jwt_manager.verify_token(token)
    
    session_id = payload.get("session_id")
    if not session_id:
        raise AuthenticationError("Invalid session")
    
    session = await auth_manager.session_manager.get_session(session_id)
    if not session or session.status != SessionStatus.ACTIVE:
        raise AuthenticationError("Session expired or invalid")
    
    return session


def require_permission(resource: str, action: str):
    """FastAPI dependency factory for permission checking"""
    async def permission_checker(
        current_user: UserSession = Depends(get_current_user)
    ) -> UserSession:
        auth_manager = get_auth_manager()
        if not auth_manager.rbac_manager.has_permission(current_user.role, resource, action):
            raise AuthorizationError(f"Access denied to {resource}:{action}")
        return current_user
    
    return permission_checker


def require_role(*allowed_roles: UserRole):
    """FastAPI dependency factory for role checking"""
    async def role_checker(
        current_user: UserSession = Depends(get_current_user)
    ) -> UserSession:
        if current_user.role not in allowed_roles:
            raise AuthorizationError(f"Role {current_user.role} not authorized")
        return current_user
    
    return role_checker