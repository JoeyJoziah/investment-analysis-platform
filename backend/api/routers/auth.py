from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
from passlib.context import CryptContext
from typing import Optional, Dict, Any
import os
import logging
from backend.config.database import get_async_db_session
from backend.models.unified_models import User
from backend.config.settings import settings
from backend.security.rate_limiter import get_rate_limiter, RateLimitCategory, rate_limit
from backend.security.jwt_manager import get_jwt_manager, TokenClaims
from backend.security.secrets_manager import get_secrets_manager
from backend.security.security_config import SecurityConfig
from backend.models.api_response import ApiResponse, success_response

router = APIRouter(tags=["authentication"])

# Security configurations
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

# JWT settings - use centralized SecurityConfig as single source of truth
SECRET_KEY = SecurityConfig.JWT_SECRET_KEY
ALGORITHM = SecurityConfig.JWT_ALGORITHM_FALLBACK  # HS256 for legacy compatibility
ACCESS_TOKEN_EXPIRE_MINUTES = SecurityConfig.JWT_ACCESS_TOKEN_EXPIRE_MINUTES

logger = logging.getLogger(__name__)

# Pydantic models
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

# Utility functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def authenticate_user(db: AsyncSession, email: str, password: str):
    """Authenticate user against database"""
    result = await db.execute(select(User).filter(User.email == email))
    user = result.scalars().first()
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_async_db_session)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    result = await db.execute(select(User).filter(User.email == email))
    user = result.scalars().first()
    if user is None:
        raise credentials_exception
    return user

# Authentication rate limiting dependency
async def auth_rate_limit(request: Request):
    """Rate limiting for authentication endpoints"""
    rate_limiter = get_rate_limiter()
    rate_status = await rate_limiter.check_rate_limit(request, RateLimitCategory.AUTHENTICATION)
    if not rate_status.allowed:
        headers = {
            "X-RateLimit-Remaining": str(rate_status.remaining),
            "X-RateLimit-Reset": str(int(rate_status.reset_time.timestamp()))
        }
        if rate_status.retry_after_seconds:
            headers["Retry-After"] = str(rate_status.retry_after_seconds)

        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many authentication attempts. Please try again later.",
            headers=headers
        )
    return rate_status


async def registration_rate_limit(request: Request):
    """Rate limiting for registration endpoints"""
    rate_limiter = get_rate_limiter()
    rate_status = await rate_limiter.check_rate_limit(request, RateLimitCategory.REGISTRATION)
    if not rate_status.allowed:
        headers = {
            "X-RateLimit-Remaining": str(rate_status.remaining),
            "X-RateLimit-Reset": str(int(rate_status.reset_time.timestamp()))
        }
        if rate_status.retry_after_seconds:
            headers["Retry-After"] = str(rate_status.retry_after_seconds)

        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many registration attempts. Please try again later.",
            headers=headers
        )
    return rate_status

# Endpoints
@router.post("/register")
async def register(
    user: UserCreate,
    request: Request,
    db: AsyncSession = Depends(get_async_db_session),
    _rate_status = Depends(registration_rate_limit)
) -> ApiResponse[Token]:
    """Register a new user"""
    # Check if user exists
    result = await db.execute(select(User).filter(User.email == user.email))
    db_user = result.scalars().first()
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Create new user
    hashed_password = get_password_hash(user.password)
    db_user = User(
        email=user.email,
        full_name=user.full_name,
        hashed_password=hashed_password,
        is_active=True,
        role="free_user"
    )

    try:
        db.add(db_user)
        await db.commit()
        await db.refresh(db_user)

        # Create access token
        access_token = create_access_token(data={"sub": user.email})

        logger.info(f"New user registered: {user.email}")
        return success_response(data=Token(
            access_token=access_token,
            token_type="bearer"
        ))

    except Exception as e:
        await db.rollback()
        logger.error(f"Error registering user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error creating user"
        )

@router.post("/token")
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_async_db_session),
    _auth_limit = Depends(auth_rate_limit)
) -> ApiResponse[Token]:
    """Login endpoint for OAuth2"""
    user = await authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Update last login
    user.last_login = datetime.now(timezone.utc)
    await db.commit()

    # Create token - use email as sub (consistent with get_current_user lookup)
    access_token = create_access_token(data={
        "sub": user.email,
        "user_id": user.id,
        "email": user.email,
        "role": "user"
    })
    return success_response(data=Token(
        access_token=access_token,
        token_type="bearer"
    ))

@router.post("/login")
async def login_alt(
    user: UserLogin,
    request: Request,
    db: AsyncSession = Depends(get_async_db_session),
    _auth_limit = Depends(auth_rate_limit)
) -> ApiResponse[Token]:
    """Alternative login endpoint"""
    db_user = await authenticate_user(db, user.email, user.password)
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )

    # Update last login
    db_user.last_login = datetime.now(timezone.utc)
    await db.commit()

    # Create token - use email as sub (consistent with get_current_user lookup)
    access_token = create_access_token(data={
        "sub": db_user.email,
        "user_id": db_user.id,
        "email": db_user.email,
        "role": "user"
    })
    return success_response(data=Token(
        access_token=access_token,
        token_type="bearer"
    ))

@router.get("/me")
async def read_users_me(current_user: User = Depends(get_current_user)) -> ApiResponse[Dict[str, Any]]:
    """Get current user information"""
    return success_response(data={
        "id": current_user.id,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "role": current_user.role,
        "is_active": current_user.is_active,
        "created_at": current_user.created_at.isoformat()
    })

@router.post("/logout")
async def logout(current_user: User = Depends(get_current_user)) -> ApiResponse[Dict[str, Any]]:
    """Logout endpoint (client should discard token)"""
    logger.info(f"User logged out: {current_user.email}")
    return success_response(data={"message": "Successfully logged out"})

@router.post("/refresh")
async def refresh_token(
    request: Request,
    current_user: User = Depends(get_current_user),
    _auth_limit = Depends(auth_rate_limit)
) -> ApiResponse[Token]:
    """Refresh access token"""
    access_token = create_access_token(data={"sub": current_user.email})
    return success_response(data=Token(
        access_token=access_token,
        token_type="bearer"
    ))