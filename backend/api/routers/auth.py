from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, EmailStr
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import logging
from backend.config.database import get_async_db_session
from backend.models.unified_models import User
from backend.security.rate_limiter import get_rate_limiter, RateLimitCategory, rate_limit
from backend.models.api_response import ApiResponse, success_response
# Canonical authentication primitives.
# Finding #201: All JWT issuance/verification MUST flow through
# backend.security.jwt_manager (RS256 + RSA keys + blacklist + session +
# issuer/audience checks). The canonical dependencies and token helpers live in
# backend.auth.oauth2, which delegates to the JWT manager. We re-export them here
# (see shim below) so existing imports of `get_current_user`/`get_current_admin_user`
# from this router continue to resolve to the single verification entry point.
from backend.auth.oauth2 import (
    verify_password,
    get_password_hash,
    create_tokens,
    get_current_user,
    get_current_admin_user,
)
from backend.security.jwt_manager import get_jwt_manager, TokenType

router = APIRouter(tags=["authentication"])

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
    refresh_token: Optional[str] = None


class RefreshRequest(BaseModel):
    refresh_token: str

class TokenData(BaseModel):
    email: Optional[str] = None

async def authenticate_user(db: AsyncSession, email: str, password: str):
    """Authenticate user against database"""
    result = await db.execute(select(User).filter(User.email == email))
    user = result.scalars().first()
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

# Finding #201: The previously-defined local `get_current_user` performed a
# direct `jwt.decode` against a STRING secret while declaring RS256, bypassing
# jwt_manager's RS256 verification, blacklist, session, issuer, and audience
# checks. It has been removed. `get_current_user` and `get_current_admin_user`
# are now re-exported (imported above) from backend.auth.oauth2, which is the
# single verification entry point delegating to jwt_manager. The names remain
# importable from this module as a thin re-export shim so dependents do not break.

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
def _issue_access_token(user: User, request: Optional[Request] = None) -> str:
    """Mint an access token for a DB user via the canonical jwt_manager path.

    Routes through backend.auth.oauth2.create_tokens, which uses RS256 RSA
    signing, session tracking, and issuer/audience claims. Role and admin claims
    are derived from the persisted DB user, never hardcoded (Finding #201).
    """
    tokens = create_tokens(user, request)
    return tokens["access_token"]


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

    # Create new user.
    #
    # #208 item 3: `username` MUST be populated. Under the #201 token contract
    # `create_tokens` sets `sub = user.username` and `get_current_user` looks the
    # user up by `username`. `UserCreate` has no username field, so we derive it
    # from the (unique, non-null) email -- otherwise registered users get a token
    # whose `sub` is null and every authenticated request from them 401s.
    hashed_password = get_password_hash(user.password)
    db_user = User(
        email=user.email,
        username=user.email,
        full_name=user.full_name,
        hashed_password=hashed_password,
        is_active=True,
        role="free_user"
    )

    try:
        db.add(db_user)
        await db.commit()
        await db.refresh(db_user)

        tokens = create_tokens(db_user, request)

        logger.info(f"New user registered: {user.email}")
        return success_response(data=Token(
            access_token=tokens["access_token"],
            refresh_token=tokens.get("refresh_token"),
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
    user.last_login = datetime.now(timezone.utc).replace(tzinfo=None)
    await db.commit()

    tokens = create_tokens(user, request)
    return success_response(data=Token(
        access_token=tokens["access_token"],
        refresh_token=tokens.get("refresh_token"),
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
    db_user.last_login = datetime.now(timezone.utc).replace(tzinfo=None)
    await db.commit()

    tokens = create_tokens(db_user, request)
    return success_response(data=Token(
        access_token=tokens["access_token"],
        refresh_token=tokens.get("refresh_token"),
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
async def logout(
    request: Request,
    current_user: User = Depends(get_current_user),
) -> ApiResponse[Dict[str, Any]]:
    """Logout: blacklist the presented access token and drop its session."""
    auth = request.headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1].strip()
        if token:
            get_jwt_manager().revoke_token(token)
    logger.info(f"User logged out: {current_user.email}")
    return success_response(data={"message": "Successfully logged out"})

@router.post("/refresh")
async def refresh_token(
    request: Request,
    body: RefreshRequest,
    db: AsyncSession = Depends(get_async_db_session),
    _auth_limit = Depends(auth_rate_limit)
) -> ApiResponse[Token]:
    """Mint a new access token from a still-valid refresh token."""
    payload = get_jwt_manager().verify_token(body.refresh_token, TokenType.REFRESH)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("user_id")
    username = payload.get("sub")
    result = await db.execute(
        select(User).filter(
            (User.id == user_id)
            | (User.username == username)
            | (User.email == username)
        )
    )
    user = result.scalars().first()
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    tokens = create_tokens(user, request)
    return success_response(data=Token(
        access_token=tokens["access_token"],
        refresh_token=tokens.get("refresh_token"),
        token_type="bearer"
    ))
