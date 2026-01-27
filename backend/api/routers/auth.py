from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from typing import Optional
import os
import logging
from backend.utils.database import get_db_sync
from backend.models.tables import User
from backend.config.settings import settings
from backend.security.rate_limiter import get_rate_limiter, RateLimitCategory, rate_limit
from backend.security.jwt_manager import get_jwt_manager, TokenClaims
from backend.security.secrets_manager import get_secrets_manager
from backend.security.security_config import SecurityConfig

router = APIRouter(prefix="/auth", tags=["authentication"])

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
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def authenticate_user(db: Session, email: str, password: str):
    """Authenticate user against database"""
    user = db.query(User).filter(User.email == email).first()
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db_sync)):
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
    
    user = db.query(User).filter(User.email == email).first()
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
@router.post("/register", response_model=Token)
async def register(
    user: UserCreate,
    request: Request,
    db: Session = Depends(get_db_sync),
    _rate_status = Depends(registration_rate_limit)
):
    """Register a new user"""
    # Check if user exists
    db_user = db.query(User).filter(User.email == user.email).first()
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
        db.commit()
        db.refresh(db_user)
        
        # Create access token
        access_token = create_access_token(data={"sub": user.email})
        
        logger.info(f"New user registered: {user.email}")
        return {"access_token": access_token, "token_type": "bearer"}
    
    except Exception as e:
        db.rollback()
        logger.error(f"Error registering user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error creating user"
        )

@router.post("/token", response_model=Token)
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db_sync),
    _auth_limit = Depends(auth_rate_limit)
):
    """Login endpoint for OAuth2"""
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()
    
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/login", response_model=Token)
async def login_alt(
    user: UserLogin,
    request: Request,
    db: Session = Depends(get_db_sync),
    _auth_limit = Depends(auth_rate_limit)
):
    """Alternative login endpoint"""
    db_user = authenticate_user(db, user.email, user.password)
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    
    # Update last login
    db_user.last_login = datetime.utcnow()
    db.commit()
    
    access_token = create_access_token(data={"sub": db_user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/me")
async def read_users_me(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return {
        "id": current_user.id,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "role": current_user.role,
        "is_active": current_user.is_active,
        "created_at": current_user.created_at.isoformat()
    }

@router.post("/logout")
async def logout(current_user: User = Depends(get_current_user)):
    """Logout endpoint (client should discard token)"""
    logger.info(f"User logged out: {current_user.email}")
    return {"message": "Successfully logged out"}

@router.post("/refresh")
async def refresh_token(current_user: User = Depends(get_current_user)):
    """Refresh access token"""
    access_token = create_access_token(data={"sub": current_user.email})
    return {"access_token": access_token, "token_type": "bearer"}