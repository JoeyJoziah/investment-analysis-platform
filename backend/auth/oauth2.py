"""OAuth2 Authentication Implementation with Enhanced Security"""

from datetime import datetime, timedelta
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jwt.exceptions import InvalidTokenError
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy.orm import Session
import logging

from backend.config.settings import settings
from backend.models.unified_models import User
from backend.utils.database import get_db
from backend.security.jwt_manager import get_jwt_manager, TokenClaims, TokenType
from backend.security.secrets_manager import get_secrets_manager

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")

# Logger
logger = logging.getLogger(__name__)


class Token(BaseModel):
    """Token response model"""
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int


class TokenData(BaseModel):
    """Token data model"""
    username: Optional[str] = None
    user_id: Optional[int] = None
    scopes: list[str] = []


class UserInDB(BaseModel):
    """User model with hashed password"""
    id: int
    username: str
    email: str
    hashed_password: str
    is_active: bool
    is_admin: bool
    created_at: datetime


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash password"""
    return pwd_context.hash(password)


def create_tokens(user: User, request: Optional[Request] = None) -> dict:
    """Create both access and refresh tokens using enhanced JWT manager"""
    try:
        jwt_manager = get_jwt_manager()
        
        # Extract client information
        ip_address = None
        user_agent = None
        if request:
            ip_address = request.client.host if request.client else None
            user_agent = request.headers.get("user-agent")
        
        # Create token claims
        claims = TokenClaims(
            user_id=user.id,
            username=user.username,
            email=user.email,
            roles=["admin"] if user.is_admin else ["user"],
            scopes=["read", "write"] if user.is_active else ["read"],
            is_admin=user.is_admin,
            is_mfa_verified=getattr(user, 'mfa_enabled', False) and getattr(user, 'mfa_verified', False),
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        # Create tokens
        access_token = jwt_manager.create_access_token(claims)
        refresh_token = jwt_manager.create_refresh_token(claims)
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        }
        
    except Exception as e:
        logger.error(f"Error creating tokens: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not create tokens"
        )


def decode_access_token(token: str) -> TokenData:
    """Decode and validate JWT token using enhanced JWT manager"""
    try:
        jwt_manager = get_jwt_manager()
        payload = jwt_manager.verify_token(token, TokenType.ACCESS)
        
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        username: str = payload.get("sub")
        user_id: int = payload.get("user_id")
        scopes: list = payload.get("scopes", [])
        
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        token_data = TokenData(
            username=username,
            user_id=user_id,
            scopes=scopes
        )
        return token_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error decoding token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user"""
    token_data = decode_access_token(token)
    
    user = db.query(User).filter(User.username == token_data.username).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current active user"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    return current_user


async def get_current_admin_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current admin user"""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user


def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    """Authenticate user with username and password"""
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


class RateLimiter:
    """Rate limiter for API endpoints"""
    
    def __init__(self, calls: int = 10, period: int = 60):
        self.calls = calls
        self.period = period
        self.clock = {}
        
    async def __call__(self, user: User = Depends(get_current_user)):
        """Check rate limit for user"""
        now = datetime.now()
        key = f"user:{user.id}"
        
        if key not in self.clock:
            self.clock[key] = []
        
        # Remove old entries
        self.clock[key] = [
            timestamp for timestamp in self.clock[key]
            if (now - timestamp).total_seconds() < self.period
        ]
        
        if len(self.clock[key]) >= self.calls:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )
        
        self.clock[key].append(now)
        return True


# Create rate limiters for different tiers
rate_limit_basic = RateLimiter(calls=100, period=3600)  # 100 calls per hour
rate_limit_premium = RateLimiter(calls=1000, period=3600)  # 1000 calls per hour
rate_limit_admin = RateLimiter(calls=10000, period=3600)  # 10000 calls per hour