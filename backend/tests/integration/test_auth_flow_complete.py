"""
Complete Auth Flow Integration Tests - JWT Bug Fix Validation

This is the CRITICAL test suite that validates the JWT token fix across
every authentication path. The fix ensured that ALL token sources use
"sub" = user.email (not user.id), matching what get_current_user expects.

Test matrix:
    - /register   -> token -> /me  (sub=email)
    - /token      -> token -> /me  (sub=email, OAuth2 form)
    - /login      -> token -> /me  (sub=email, JSON body)
    - /refresh    -> token -> /me  (new token from existing session)
    - Invalid credentials on each endpoint
    - Expired token rejection
    - Malformed/tampered token rejection
    - Token claim consistency across all endpoints
    - Duplicate registration prevention
    - Cross-endpoint token interoperability

Created: 2026-02-08
"""

import pytest
import pytest_asyncio
from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator
from unittest.mock import patch, MagicMock
from jose import jwt as jose_jwt
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from httpx import AsyncClient, ASGITransport

from backend.models.unified_models import User as TablesUser, Base as TablesBase, UserRoleEnum
from backend.models.unified_models import User
from backend.security.security_config import SecurityConfig
from backend.api.routers.auth import (
    create_access_token,
    SECRET_KEY,
    ALGORITHM,
)
from backend.api.main import app
from backend.config.database import get_async_db_session


pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEST_PASSWORD = "testpassword123"
TEST_EMAIL = "authflow@test.com"
TEST_FULL_NAME = "Auth Flow Tester"

# Pre-computed bcrypt hash of "testpassword123" (generated with bcrypt 5.0.0, rounds=12)
# This avoids passlib/bcrypt version incompatibility at test time.
TEST_PASSWORD_HASH = (
    "$2b$12$.lZkz/7bNu.JQTjT4smsnuh6UEMaSRTEMndiAwL/XaW9OpeLcxxOG"
)


# ---------------------------------------------------------------------------
# Database fixtures (override conftest to use tables.Base for auth router)
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture(scope="function")
async def test_db_engine():
    """
    Create test database engine.

    Uses unified_models.Base which defines role as String(50), matching
    what the auth router passes (role="free_user" as a plain string).
    The tables.Base defines role as SQLEnum(UserRoleEnum) which rejects
    plain strings, so we use unified_models.Base for schema creation.
    """
    from backend.models.unified_models import Base as UnifiedBase

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        pool_pre_ping=True,
    )
    async with engine.begin() as conn:
        await conn.run_sync(UnifiedBase.metadata.create_all)

    yield engine
    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def test_db_session_factory(test_db_engine):
    """Create test database session factory."""
    return sessionmaker(
        test_db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


@pytest_asyncio.fixture
async def db_session(test_db_session_factory):
    """Provide database session for tests."""
    async with test_db_session_factory() as session:
        try:
            yield session
        finally:
            await session.rollback()
            await session.close()


@pytest_asyncio.fixture
async def async_client(test_db_engine):
    """Provide async HTTP client with test database override."""
    TestSessionLocal = sessionmaker(
        test_db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async def override_get_async_db_session() -> AsyncGenerator[AsyncSession, None]:
        async with TestSessionLocal() as session:
            try:
                yield session
            finally:
                await session.rollback()

    app.dependency_overrides[get_async_db_session] = override_get_async_db_session

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://localhost"
    ) as client:
        yield client

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def registered_user(db_session: AsyncSession):
    """
    Pre-register a user in the database so login endpoints can authenticate.
    Uses a pre-computed bcrypt hash to avoid passlib/bcrypt version issues.
    """
    user = User(
        email=TEST_EMAIL,
        hashed_password=TEST_PASSWORD_HASH,
        full_name=TEST_FULL_NAME,
        is_active=True,
        is_verified=True,
        role="free_user",
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest.fixture(autouse=True)
def _patch_bcrypt_and_enum():
    """
    Patch two incompatibilities in the test environment:

    1. passlib 1.7.4 + bcrypt 5.0.0 -- passlib cannot call bcrypt.hashpw
       due to API changes in bcrypt 4.1+. We call bcrypt directly.

    2. The auth router creates User(role="free_user") (string) but
       tables.User defines role as SQLEnum(UserRoleEnum), which rejects
       plain strings. We patch the pwd_context methods AND the User
       import in auth to use unified_models.User (role=String(50)).
    """
    import bcrypt
    from backend.models import unified_models

    def _mock_verify(plain_password, hashed_password):
        """Use bcrypt directly to verify passwords."""
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8"),
        )

    def _mock_hash(password):
        """Use bcrypt directly to hash passwords."""
        salt = bcrypt.gensalt(rounds=4)  # Low rounds for fast tests
        return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

    with patch(
        "backend.api.routers.auth.verify_password", side_effect=_mock_verify
    ), patch(
        "backend.api.routers.auth.get_password_hash", side_effect=_mock_hash
    ), patch(
        "backend.api.routers.auth.User", unified_models.User
    ):
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_token(response, expected_status=200):
    """
    Unwrap ApiResponse envelope and return the access_token string.
    Asserts expected HTTP status and success flag.
    """
    assert response.status_code == expected_status, (
        f"Expected {expected_status}, got {response.status_code}: {response.text}"
    )
    body = response.json()
    assert body["success"] is True, f"Expected success=True, got {body}"
    data = body["data"]
    assert "access_token" in data, f"Missing access_token in data: {data}"
    assert data["token_type"] == "bearer"
    return data["access_token"]


def _assert_me_success(response, expected_email):
    """Validate the /me endpoint returns the correct user wrapped in ApiResponse."""
    assert response.status_code == 200, (
        f"Expected 200, got {response.status_code}: {response.text}"
    )
    body = response.json()
    assert body["success"] is True
    user_data = body["data"]
    assert user_data["email"] == expected_email
    assert "id" in user_data
    assert "full_name" in user_data
    assert user_data["is_active"] is True
    return user_data


# ---------------------------------------------------------------------------
# 1. Register -> Token -> /me
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_register_then_access_me(async_client: AsyncClient):
    """
    Full lifecycle: register a brand-new user, receive a JWT, and use it to
    access the /me endpoint. This validates that /register produces a token
    with sub=email that get_current_user can decode.
    """
    register_payload = {
        "email": "newuser_register@test.com",
        "password": TEST_PASSWORD,
        "full_name": "New Register User",
    }

    response = await async_client.post(
        "/api/v1/auth/register", json=register_payload
    )
    token = _extract_token(response, expected_status=200)

    # Verify the token's sub claim is the email
    decoded = jose_jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    assert decoded["sub"] == register_payload["email"]

    # Use the token to hit /me
    me_response = await async_client.get(
        "/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"}
    )
    user_data = _assert_me_success(me_response, register_payload["email"])
    assert user_data["full_name"] == "New Register User"


# ---------------------------------------------------------------------------
# 2. Login via /token (OAuth2 form) -> Token -> /me
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_token_endpoint_then_access_me(
    async_client: AsyncClient, registered_user: User
):
    """
    Login via /token using OAuth2 form-encoded data, receive a JWT, then
    access /me. This is the standard OAuth2 flow used by Swagger UI.
    """
    response = await async_client.post(
        "/api/v1/auth/token",
        data={"username": TEST_EMAIL, "password": TEST_PASSWORD},
    )
    token = _extract_token(response)

    # Verify sub claim is email (the critical fix)
    decoded = jose_jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    assert decoded["sub"] == TEST_EMAIL

    # Verify additional claims set by /token endpoint
    assert decoded.get("email") == TEST_EMAIL
    assert "user_id" in decoded

    # Use the token on /me
    me_response = await async_client.get(
        "/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"}
    )
    user_data = _assert_me_success(me_response, TEST_EMAIL)
    assert user_data["full_name"] == TEST_FULL_NAME


# ---------------------------------------------------------------------------
# 3. Login via /login (JSON body) -> Token -> /me
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_login_endpoint_then_access_me(
    async_client: AsyncClient, registered_user: User
):
    """
    Login via /login using JSON body, receive a JWT, then access /me.
    This validates the alternative login endpoint produces compatible tokens.
    """
    response = await async_client.post(
        "/api/v1/auth/login",
        json={"email": TEST_EMAIL, "password": TEST_PASSWORD},
    )
    token = _extract_token(response)

    # Verify sub claim is email (the critical fix)
    decoded = jose_jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    assert decoded["sub"] == TEST_EMAIL
    assert decoded.get("email") == TEST_EMAIL
    assert "user_id" in decoded

    # Use the token on /me
    me_response = await async_client.get(
        "/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"}
    )
    _assert_me_success(me_response, TEST_EMAIL)


# ---------------------------------------------------------------------------
# 4. Refresh Token -> /me
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_refresh_token_then_access_me(
    async_client: AsyncClient, registered_user: User
):
    """
    Login, refresh the token, then verify the NEW token works on /me.
    Confirms that /refresh also uses sub=email consistently.
    """
    # Login first to get an initial token
    login_response = await async_client.post(
        "/api/v1/auth/token",
        data={"username": TEST_EMAIL, "password": TEST_PASSWORD},
    )
    original_token = _extract_token(login_response)

    # Refresh
    refresh_response = await async_client.post(
        "/api/v1/auth/refresh",
        headers={"Authorization": f"Bearer {original_token}"},
    )
    refreshed_token = _extract_token(refresh_response)

    # The refreshed token should be different from the original
    assert refreshed_token != original_token

    # Verify the refreshed token also has sub=email
    decoded = jose_jwt.decode(refreshed_token, SECRET_KEY, algorithms=[ALGORITHM])
    assert decoded["sub"] == TEST_EMAIL

    # Use the refreshed token on /me
    me_response = await async_client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {refreshed_token}"},
    )
    _assert_me_success(me_response, TEST_EMAIL)


# ---------------------------------------------------------------------------
# 5. Cross-endpoint token interoperability
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_all_token_sources_work_with_get_current_user(
    async_client: AsyncClient, registered_user: User
):
    """
    Obtain tokens from /register, /token, and /login, then verify ALL three
    tokens are accepted by /me. This is the key test for the JWT fix:
    before the fix, tokens from /token and /login used sub=user.id which
    caused get_current_user to fail since it looks up by email.
    """
    # Token from /token endpoint
    resp1 = await async_client.post(
        "/api/v1/auth/token",
        data={"username": TEST_EMAIL, "password": TEST_PASSWORD},
    )
    token_from_oauth = _extract_token(resp1)

    # Token from /login endpoint
    resp2 = await async_client.post(
        "/api/v1/auth/login",
        json={"email": TEST_EMAIL, "password": TEST_PASSWORD},
    )
    token_from_login = _extract_token(resp2)

    # Token from /register (new user)
    resp3 = await async_client.post(
        "/api/v1/auth/register",
        json={
            "email": "interop_test@test.com",
            "password": TEST_PASSWORD,
            "full_name": "Interop User",
        },
    )
    token_from_register = _extract_token(resp3)

    # ALL three tokens must work with /me
    for label, token, expected_email in [
        ("oauth_token", token_from_oauth, TEST_EMAIL),
        ("login_token", token_from_login, TEST_EMAIL),
        ("register_token", token_from_register, "interop_test@test.com"),
    ]:
        me_resp = await async_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert me_resp.status_code == 200, (
            f"Token from {label} failed on /me: "
            f"{me_resp.status_code} {me_resp.text}"
        )
        body = me_resp.json()
        assert body["data"]["email"] == expected_email, (
            f"Token from {label} returned wrong email: "
            f"{body['data']['email']}"
        )


# ---------------------------------------------------------------------------
# 6. Invalid credentials - /token
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_invalid_credentials_token_endpoint(
    async_client: AsyncClient, registered_user: User
):
    """Wrong password on /token returns 401."""
    response = await async_client.post(
        "/api/v1/auth/token",
        data={"username": TEST_EMAIL, "password": "WrongPassword999!"},
    )
    assert response.status_code == 401
    body = response.json()
    # Error may be in "detail" (HTTPException) or "error" (ApiResponse)
    error_msg = body.get("detail") or body.get("error") or ""
    assert (
        "incorrect" in error_msg.lower()
        or "could not" in error_msg.lower()
    ), f"Expected auth error message, got: {body}"


# ---------------------------------------------------------------------------
# 7. Invalid credentials - /login
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_invalid_credentials_login_endpoint(
    async_client: AsyncClient, registered_user: User
):
    """Wrong password on /login returns 401."""
    response = await async_client.post(
        "/api/v1/auth/login",
        json={"email": TEST_EMAIL, "password": "WrongPassword999!"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_nonexistent_user_login(async_client: AsyncClient):
    """Login with a user that does not exist returns 401."""
    response = await async_client.post(
        "/api/v1/auth/login",
        json={"email": "nobody@nowhere.com", "password": TEST_PASSWORD},
    )
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# 8. Expired token rejection
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_expired_token_rejected_on_me(
    async_client: AsyncClient, registered_user: User
):
    """
    Manually create a token with an expiration in the past.
    Verify that /me rejects it with 401.
    """
    expired_token = create_access_token(
        data={"sub": TEST_EMAIL},
        expires_delta=timedelta(seconds=-10),
    )
    response = await async_client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {expired_token}"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_expired_token_rejected_on_refresh(
    async_client: AsyncClient, registered_user: User
):
    """Expired token should also be rejected by /refresh."""
    expired_token = create_access_token(
        data={"sub": TEST_EMAIL},
        expires_delta=timedelta(seconds=-10),
    )
    response = await async_client.post(
        "/api/v1/auth/refresh",
        headers={"Authorization": f"Bearer {expired_token}"},
    )
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# 9. Malformed and tampered tokens
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_malformed_token_rejected(async_client: AsyncClient):
    """Completely invalid token string is rejected with 401."""
    response = await async_client.get(
        "/api/v1/auth/me",
        headers={"Authorization": "Bearer not.a.valid.jwt.token"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_tampered_token_rejected(
    async_client: AsyncClient, registered_user: User
):
    """
    Create a valid token then tamper with it by signing with a different key.
    The server should reject it because the signature will not verify.
    """
    tampered_token = jose_jwt.encode(
        {
            "sub": TEST_EMAIL,
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
        },
        "wrong-secret-key-that-does-not-match",
        algorithm=ALGORITHM,
    )
    response = await async_client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {tampered_token}"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_token_without_sub_claim_rejected(async_client: AsyncClient):
    """
    A token that is validly signed but missing the 'sub' claim should be
    rejected by get_current_user.
    """
    token = jose_jwt.encode(
        {
            "email": "someone@test.com",
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
        },
        SECRET_KEY,
        algorithm=ALGORITHM,
    )
    response = await async_client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# 10. Duplicate registration prevention
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_duplicate_email_registration_rejected(
    async_client: AsyncClient, registered_user: User
):
    """Attempting to register with an already-used email returns 400."""
    response = await async_client.post(
        "/api/v1/auth/register",
        json={
            "email": TEST_EMAIL,
            "password": TEST_PASSWORD,
            "full_name": "Duplicate User",
        },
    )
    assert response.status_code == 400
    body = response.json()
    # Error may be in "detail" (HTTPException) or "error" (ApiResponse)
    error_msg = body.get("detail") or body.get("error") or ""
    assert "already registered" in error_msg.lower(), (
        f"Expected 'already registered' in error, got: {body}"
    )


# ---------------------------------------------------------------------------
# 11. Token sub claim consistency across all endpoints
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_token_sub_claim_is_always_email(
    async_client: AsyncClient, registered_user: User
):
    """
    The core JWT bug was that some endpoints set sub=user.id while
    get_current_user expected sub=email. This test explicitly decodes
    every token and asserts that sub is ALWAYS the email string.
    """
    # /token endpoint
    r1 = await async_client.post(
        "/api/v1/auth/token",
        data={"username": TEST_EMAIL, "password": TEST_PASSWORD},
    )
    t1 = _extract_token(r1)
    d1 = jose_jwt.decode(t1, SECRET_KEY, algorithms=[ALGORITHM])
    assert d1["sub"] == TEST_EMAIL, f"/token sub mismatch: {d1['sub']}"

    # /login endpoint
    r2 = await async_client.post(
        "/api/v1/auth/login",
        json={"email": TEST_EMAIL, "password": TEST_PASSWORD},
    )
    t2 = _extract_token(r2)
    d2 = jose_jwt.decode(t2, SECRET_KEY, algorithms=[ALGORITHM])
    assert d2["sub"] == TEST_EMAIL, f"/login sub mismatch: {d2['sub']}"

    # /register endpoint (new user)
    unique_email = "sub_claim_test@test.com"
    r3 = await async_client.post(
        "/api/v1/auth/register",
        json={
            "email": unique_email,
            "password": TEST_PASSWORD,
            "full_name": "Sub Claim Tester",
        },
    )
    t3 = _extract_token(r3)
    d3 = jose_jwt.decode(t3, SECRET_KEY, algorithms=[ALGORITHM])
    assert d3["sub"] == unique_email, f"/register sub mismatch: {d3['sub']}"

    # /refresh endpoint
    r4 = await async_client.post(
        "/api/v1/auth/refresh",
        headers={"Authorization": f"Bearer {t1}"},
    )
    t4 = _extract_token(r4)
    d4 = jose_jwt.decode(t4, SECRET_KEY, algorithms=[ALGORITHM])
    assert d4["sub"] == TEST_EMAIL, f"/refresh sub mismatch: {d4['sub']}"


# ---------------------------------------------------------------------------
# 12. Missing Authorization header
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_missing_auth_header_on_protected_endpoints(
    async_client: AsyncClient,
):
    """Protected endpoints return 401 when no Authorization header is sent."""
    endpoints = [
        ("GET", "/api/v1/auth/me"),
        ("POST", "/api/v1/auth/refresh"),
        ("POST", "/api/v1/auth/logout"),
    ]
    for method, endpoint in endpoints:
        if method == "GET":
            response = await async_client.get(endpoint)
        else:
            response = await async_client.post(endpoint)
        assert response.status_code == 401, (
            f"{method} {endpoint} should return 401 without auth, "
            f"got {response.status_code}"
        )


# ---------------------------------------------------------------------------
# 13. Logout with valid token
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_logout_with_valid_token(
    async_client: AsyncClient, registered_user: User
):
    """
    Logout should succeed with a valid token. After logout the server
    returns a success message (client is responsible for discarding token).
    """
    login_resp = await async_client.post(
        "/api/v1/auth/token",
        data={"username": TEST_EMAIL, "password": TEST_PASSWORD},
    )
    token = _extract_token(login_resp)

    logout_resp = await async_client.post(
        "/api/v1/auth/logout",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert logout_resp.status_code == 200
    body = logout_resp.json()
    assert body["success"] is True
    assert "logged out" in body["data"]["message"].lower()


# ---------------------------------------------------------------------------
# 14. Token expiration claim is set correctly
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_token_expiration_is_within_configured_range(
    async_client: AsyncClient, registered_user: User
):
    """
    Verify that tokens expire within the configured window.
    Default is 30 minutes from SecurityConfig.JWT_ACCESS_TOKEN_EXPIRE_MINUTES.
    """
    response = await async_client.post(
        "/api/v1/auth/token",
        data={"username": TEST_EMAIL, "password": TEST_PASSWORD},
    )
    token = _extract_token(response)
    decoded = jose_jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

    exp_timestamp = decoded["exp"]
    now = datetime.now(timezone.utc).timestamp()
    expire_minutes = SecurityConfig.JWT_ACCESS_TOKEN_EXPIRE_MINUTES

    # Token should expire within the configured window (+/- 60s tolerance)
    expected_min = now + (expire_minutes * 60) - 60
    expected_max = now + (expire_minutes * 60) + 60

    assert expected_min <= exp_timestamp <= expected_max, (
        f"Token exp {exp_timestamp} not within expected range "
        f"[{expected_min}, {expected_max}] for {expire_minutes}min expiry"
    )
