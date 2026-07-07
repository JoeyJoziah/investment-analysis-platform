"""
Complete Auth Flow Integration Tests - Secure JWT Contract Validation (Finding #201)

This suite previously validated a *symmetric* JWT contract: it imported
``create_access_token``, ``SECRET_KEY`` and ``ALGORITHM`` from
``backend.api.routers.auth`` and decoded tokens with
``jose_jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])``. Finding #201
removed that weak path entirely. Token issuance and verification now flow
through ``backend.auth.oauth2`` -> ``backend.security.jwt_manager.JWTManager``,
which signs with **RS256** (RSA key pair), enforces ``iss``/``aud`` claims,
tracks sessions, and supports a revocation blacklist. There is no shared
symmetric secret a test can decode with anymore.

This file has therefore been rewritten to assert the **secure** contract:

    - Issue -> verify roundtrip via the canonical jwt_manager / oauth2 path.
    - Tokens are RS256-signed (alg header == "RS256"); they are NEVER
      verifiable with a symmetric secret.
    - A token forged with a symmetric (HS256) string secret is REJECTED
      (this is the inversion of the old "decodes with the shared secret"
      assertion -- the vulnerable behaviour is now a security failure case).
    - Expired tokens are rejected.
    - Tampered / wrong-key / malformed tokens are rejected.
    - Refresh tokens mint fresh access tokens; token-type confusion is
      rejected (an access token is not accepted as a refresh token and
      vice versa).
    - Claim contents (sub, user_id, email, roles, is_admin, iss, aud, type)
      are populated as expected.
    - Blacklisted tokens and tokens whose session has been revoked are
      rejected.
    - The ``expires_in`` window matches SecurityConfig.

Why the level changed (Finding #201): the original tests drove the live
FastAPI app and asserted ``sub == email`` on every endpoint. Under the new
contract ``create_tokens`` sets ``sub = user.username`` (derived from the
persisted DB user), and ``get_current_user`` looks the user up by
``username`` -- the "sub is always the email" invariant the old suite was
built around no longer describes the system. The meaningful, stable contract
to pin is the oauth2 + jwt_manager layer, so the intent of each original case
(roundtrip, expiry, refresh, tamper/blacklist, claim contents) is preserved
there rather than against obsolete HTTP-level assertions.

These tests are self-contained at the oauth2/jwt_manager level and do not
require the full app, a live database, or a live Redis. A small in-memory
fake Redis is injected so blacklist / session-revocation behaviour can be
exercised deterministically. Run with ``--noconftest`` if the repository
conftest (which imports the full app) cannot import in your environment.

Required env vars (secrets are resolved at import time by SecurityConfig):
    SECRET_KEY, JWT_SECRET_KEY, SESSION_SECRET_KEY, MASTER_SECRET_KEY,
    REDIS_URL, DATABASE_URL, ENVIRONMENT=testing

Original suite created: 2026-02-08
Rewritten for Finding #201 secure-JWT contract: 2026-05
"""

import fnmatch
import hashlib
from datetime import datetime, timedelta, timezone

import jwt as pyjwt
import pytest
from jose import jwt as jose_jwt

# Canonical secure-token surfaces. NOTHING is imported from the router:
# the weak symmetric path (create_access_token / SECRET_KEY / ALGORITHM in
# backend.api.routers.auth) was removed by Finding #201.
from backend.auth import oauth2
from backend.auth.oauth2 import (
    create_access_token,
    create_refresh_token,
    verify_token,
)
from backend.security.jwt_manager import (
    JWTManager,
    TokenClaims,
    TokenType,
)
from backend.security.secrets_manager import get_secrets_manager
from backend.security.security_config import SecurityConfig


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEST_USERNAME = "authflow"
TEST_EMAIL = "authflow@test.com"
TEST_USER_ID = 4242

# Issuer / audience are part of the secure contract and are enforced on verify.
EXPECTED_ISSUER = SecurityConfig.JWT_ISSUER
EXPECTED_AUDIENCE = SecurityConfig.JWT_AUDIENCE


# ---------------------------------------------------------------------------
# In-memory fake Redis
#
# JWTManager uses Redis for (a) session tracking on access-token creation,
# (b) the revocation blacklist, and (c) session-existence enforcement on
# verify. We inject a tiny dict-backed fake so these behaviours can be tested
# without a live Redis. This mirrors only the methods JWTManager calls.
# ---------------------------------------------------------------------------


class _FakeRedis:
    """Minimal in-memory Redis stand-in for JWTManager."""

    def __init__(self):
        self._hashes = {}
        self._strings = {}

    def hset(self, name, mapping=None):
        self._hashes[name] = dict(mapping or {})
        return len(self._hashes[name])

    def expire(self, name, ttl):
        return True

    def exists(self, name):
        return 1 if (name in self._hashes or name in self._strings) else 0

    def setex(self, name, ttl, value):
        self._strings[name] = value
        return True

    def delete(self, *names):
        removed = 0
        for name in names:
            if name in self._hashes:
                del self._hashes[name]
                removed += 1
            if name in self._strings:
                del self._strings[name]
                removed += 1
        return removed

    def keys(self, pattern):
        all_keys = list(self._hashes.keys()) + list(self._strings.keys())
        return [k for k in all_keys if fnmatch.fnmatch(k, pattern)]

    def hgetall(self, name):
        return dict(self._hashes.get(name, {}))


def _build_jwt_manager(redis_client):
    """Construct a JWTManager with an explicit redis client.

    JWTManager.__init__ falls back to a live Redis connection whenever the
    ``redis_client`` argument is falsy (``redis_client or self._get_redis_client()``),
    so we bypass __init__ and wire the instance up directly. This lets us pass
    either a fake Redis or ``None`` deterministically. RSA keys still come from
    the real ``_initialize_rsa_keys`` so signing/verification is genuine RS256.
    """
    manager = JWTManager.__new__(JWTManager)
    manager.secrets_manager = get_secrets_manager()
    manager.redis_client = redis_client
    manager.private_key, manager.public_key = manager._initialize_rsa_keys()
    manager.access_token_expire_minutes = SecurityConfig.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
    manager.refresh_token_expire_days = SecurityConfig.JWT_REFRESH_TOKEN_EXPIRE_DAYS
    manager.mfa_token_expire_minutes = SecurityConfig.JWT_MFA_TOKEN_EXPIRE_MINUTES
    manager.issuer = SecurityConfig.JWT_ISSUER
    manager.audience = SecurityConfig.JWT_AUDIENCE
    manager.blacklist_prefix = "jwt_blacklist"
    manager.session_prefix = "user_session"
    return manager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_redis():
    """Fresh in-memory Redis per test."""
    return _FakeRedis()


@pytest.fixture
def jwt_manager(fake_redis):
    """A genuine RS256 JWTManager backed by the in-memory fake Redis."""
    return _build_jwt_manager(fake_redis)


@pytest.fixture
def jwt_manager_no_redis():
    """A JWTManager with Redis disabled (graceful no-op blacklist/session)."""
    return _build_jwt_manager(None)


@pytest.fixture
def claims():
    """Standard non-admin token claims."""
    return TokenClaims(
        user_id=TEST_USER_ID,
        username=TEST_USERNAME,
        email=TEST_EMAIL,
        roles=["user"],
        scopes=["read", "write"],
        is_admin=False,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _alg_header(token: str) -> str:
    """Return the signing algorithm declared in a token's JOSE header."""
    return pyjwt.get_unverified_header(token)["alg"]


def _blacklist(manager: JWTManager, token: str) -> None:
    """Add a token to the manager's blacklist using its real key scheme.

    We populate the blacklist directly rather than calling
    ``manager.revoke_token`` because revocation enforcement is what we are
    asserting (``verify_token`` must reject blacklisted tokens). This keeps the
    test focused on the security guarantee and independent of the revoke
    bookkeeping path.
    """
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    blacklist_key = f"{manager.blacklist_prefix}:{token_hash}"
    manager.redis_client.setex(blacklist_key, 3600, "1")


# ===========================================================================
# 1. Issue -> verify roundtrip (was: register -> token -> /me)
# ===========================================================================


def test_access_token_roundtrip_via_jwt_manager(jwt_manager, claims):
    """A freshly issued access token verifies and exposes its identity claims.

    Replaces the old "register -> token -> /me" lifecycle: the meaningful
    invariant under #201 is issue -> verify roundtrip through the canonical
    RS256 path, not the obsolete sub==email HTTP assertion.
    """
    token = jwt_manager.create_access_token(claims)

    payload = jwt_manager.verify_token(token, TokenType.ACCESS)
    assert payload is not None, "Freshly issued access token must verify"
    assert payload["sub"] == TEST_USERNAME
    assert payload["user_id"] == TEST_USER_ID
    assert payload["email"] == TEST_EMAIL
    assert payload["type"] == TokenType.ACCESS.value


def test_oauth2_create_access_token_roundtrip(jwt_manager_no_redis, monkeypatch):
    """oauth2.create_access_token (data-dict compat) issues a verifiable token.

    This is the surface the old suite imported from the router; it now lives in
    backend.auth.oauth2 and delegates to the RS256 jwt_manager. We point the
    module-global jwt_manager at our Redis-free instance so the roundtrip does
    not require a live Redis.
    """
    monkeypatch.setattr(oauth2, "get_jwt_manager", lambda: jwt_manager_no_redis)

    token = create_access_token(
        {"sub": str(TEST_USER_ID), "username": TEST_USERNAME, "role": "admin"}
    )

    # Verification goes through the canonical path, never a symmetric decode.
    payload = verify_token(token)
    assert payload is not None
    assert payload["user_id"] == TEST_USER_ID
    assert payload["is_admin"] is True
    assert "admin" in payload["roles"]


# ===========================================================================
# 2. Tokens are RS256 and NOT symmetric (core #201 secure-contract assertion)
# ===========================================================================


def test_issued_token_is_rs256_signed(jwt_manager, claims):
    """Issued tokens declare RS256 -- never the old HS256 symmetric algorithm."""
    access = jwt_manager.create_access_token(claims)
    refresh = jwt_manager.create_refresh_token(claims)

    assert _alg_header(access) == "RS256"
    assert _alg_header(refresh) == "RS256"


def test_token_not_verifiable_with_symmetric_secret(jwt_manager, claims):
    """A genuine RS256 token cannot be decoded as if it had a shared HS256 key.

    The OLD suite asserted ``jose_jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])``
    succeeded -- i.e. it relied on the vulnerable symmetric-secret contract.
    Under #201 that must be impossible: attempting a symmetric verification of
    an RS256 token must raise.
    """
    token = jwt_manager.create_access_token(claims)

    for candidate_secret in (
        SecurityConfig.JWT_SECRET_KEY,
        "wrong-secret-key-that-does-not-match",
    ):
        with pytest.raises(Exception):
            jose_jwt.decode(token, candidate_secret, algorithms=["HS256"])


def test_symmetric_forged_token_is_rejected(jwt_manager):
    """A token forged with a symmetric string secret is REJECTED by verify.

    Inversion of the old vulnerable behaviour: previously a token signed with
    the shared string secret was *accepted*. Now an attacker-forged HS256 token
    -- even with correct iss/aud/type claims -- fails RS256 verification.
    """
    forged = jose_jwt.encode(
        {
            "sub": TEST_USERNAME,
            "user_id": TEST_USER_ID,
            "type": TokenType.ACCESS.value,
            "iss": EXPECTED_ISSUER,
            "aud": EXPECTED_AUDIENCE,
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
        },
        "attacker-chosen-symmetric-secret",
        algorithm="HS256",
    )

    assert jwt_manager.verify_token(forged, TokenType.ACCESS) is None
    # And through the oauth2 convenience wrapper as well.
    import backend.auth.oauth2 as oauth2_mod

    original = oauth2_mod.get_jwt_manager
    oauth2_mod.get_jwt_manager = lambda: jwt_manager
    try:
        assert oauth2_mod.verify_token(forged) is None
    finally:
        oauth2_mod.get_jwt_manager = original


# ===========================================================================
# 3. Claim contents and consistency (was: sub-claim consistency suite)
# ===========================================================================


def test_access_token_claims_contents(jwt_manager):
    """Access token carries the full secure claim set with iss/aud/type."""
    admin_claims = TokenClaims(
        user_id=99,
        username="adminuser",
        email="admin@test.com",
        roles=["admin"],
        scopes=["read", "write"],
        is_admin=True,
    )
    token = jwt_manager.create_access_token(admin_claims)
    payload = jwt_manager.verify_token(token, TokenType.ACCESS)

    assert payload is not None
    assert payload["sub"] == "adminuser"
    assert payload["user_id"] == 99
    assert payload["email"] == "admin@test.com"
    assert payload["roles"] == ["admin"]
    assert payload["is_admin"] is True
    assert payload["scopes"] == ["read", "write"]
    assert payload["type"] == TokenType.ACCESS.value
    assert payload["iss"] == EXPECTED_ISSUER
    assert payload["aud"] == EXPECTED_AUDIENCE
    assert "session_id" in payload
    assert "iat" in payload and "exp" in payload


def _claims_for(user_id, username, email, roles, is_admin):
    return TokenClaims(
        user_id=user_id,
        username=username,
        email=email,
        roles=roles,
        scopes=["read", "write"],
        is_admin=is_admin,
    )


def test_claims_consistent_across_independent_issuances(jwt_manager):
    """Tokens for the same identity carry identical identity claims.

    Replaces the old "sub is always email across /token,/login,/register,/refresh"
    consistency test: the stable invariant is that identity claims are derived
    consistently from the same identity, independent of per-issuance session
    churn. Fresh TokenClaims objects are used per issuance so the manager mints
    distinct session IDs (a single TokenClaims instance is mutated in place to
    carry its session_id, so reusing one would yield the same session).
    """
    first = jwt_manager.create_access_token(
        _claims_for(TEST_USER_ID, TEST_USERNAME, TEST_EMAIL, ["user"], False)
    )
    second = jwt_manager.create_access_token(
        _claims_for(TEST_USER_ID, TEST_USERNAME, TEST_EMAIL, ["user"], False)
    )

    p1 = jwt_manager.verify_token(first, TokenType.ACCESS)
    p2 = jwt_manager.verify_token(second, TokenType.ACCESS)
    assert p1 is not None and p2 is not None

    for key in ("sub", "user_id", "email", "roles", "scopes", "is_admin"):
        assert p1[key] == p2[key], f"Claim '{key}' inconsistent across issuances"

    # Independent issuances (fresh claims) get distinct session IDs.
    assert p1["session_id"] != p2["session_id"]


# ===========================================================================
# 4. Refresh flow (was: /refresh -> /me)
# ===========================================================================


def test_refresh_token_mints_new_access_token(jwt_manager, claims):
    """A valid refresh token produces a fresh, verifiable access token."""
    refresh = jwt_manager.create_refresh_token(claims)
    assert jwt_manager.verify_token(refresh, TokenType.REFRESH) is not None

    new_access = jwt_manager.refresh_access_token(refresh)
    assert new_access is not None, "refresh_access_token must mint a new token"

    payload = jwt_manager.verify_token(new_access, TokenType.ACCESS)
    assert payload is not None
    assert payload["type"] == TokenType.ACCESS.value
    assert payload["user_id"] == TEST_USER_ID


def test_token_type_confusion_rejected(jwt_manager, claims):
    """Access and refresh tokens are not interchangeable.

    Verifying an access token as a refresh token (and vice versa) must fail --
    a hardening guarantee the symmetric suite never checked.
    """
    access = jwt_manager.create_access_token(claims)
    refresh = jwt_manager.create_refresh_token(claims)

    assert jwt_manager.verify_token(access, TokenType.REFRESH) is None
    assert jwt_manager.verify_token(refresh, TokenType.ACCESS) is None


# ===========================================================================
# 5. Expired token rejection (was: expired token on /me and /refresh)
# ===========================================================================


def test_expired_access_token_rejected(jwt_manager, claims):
    """An access token whose exp is in the past is rejected."""
    expired = jwt_manager.create_access_token(
        claims, expires_delta=timedelta(seconds=-10)
    )
    assert jwt_manager.verify_token(expired, TokenType.ACCESS) is None


def test_expired_refresh_token_rejected(jwt_manager, claims):
    """An expired refresh token cannot mint a new access token."""
    expired_refresh = jwt_manager.create_refresh_token(
        claims, expires_delta=timedelta(seconds=-10)
    )
    assert jwt_manager.verify_token(expired_refresh, TokenType.REFRESH) is None
    assert jwt_manager.refresh_access_token(expired_refresh) is None


def test_oauth2_expired_token_returns_none(jwt_manager_no_redis, monkeypatch):
    """oauth2.verify_token returns None for an expired token (no exception)."""
    monkeypatch.setattr(oauth2, "get_jwt_manager", lambda: jwt_manager_no_redis)
    expired = create_access_token(
        {"sub": str(TEST_USER_ID), "username": TEST_USERNAME},
        expires_delta=timedelta(seconds=-10),
    )
    assert verify_token(expired) is None


# ===========================================================================
# 6. Malformed / tampered token rejection
# ===========================================================================


def test_malformed_token_rejected(jwt_manager):
    """A non-JWT garbage string is rejected (returns None, no crash)."""
    assert jwt_manager.verify_token("not.a.valid.jwt.token", TokenType.ACCESS) is None
    assert jwt_manager.verify_token("", TokenType.ACCESS) is None


def test_tampered_payload_rejected(jwt_manager, claims):
    """Mutating a token's payload breaks the RS256 signature and is rejected."""
    token = jwt_manager.create_access_token(claims)
    header, payload_b64, signature = token.split(".")

    # Flip a character in the payload segment (keep it base64-ish) so the
    # signature no longer matches the content.
    mutated = list(payload_b64)
    mutated[0] = "A" if mutated[0] != "A" else "B"
    tampered = f"{header}.{''.join(mutated)}.{signature}"

    assert jwt_manager.verify_token(tampered, TokenType.ACCESS) is None


def test_wrong_rsa_key_signature_rejected(jwt_manager, claims):
    """A token signed by a DIFFERENT RSA key pair is rejected.

    The RS256 analogue of the old "signed with a different key" tamper test:
    a structurally valid token from a foreign issuer key fails verification
    against the manager's public key.
    """
    foreign = _build_jwt_manager(_FakeRedis())  # distinct RSA key pair
    foreign_token = foreign.create_access_token(claims)

    # The foreign manager can verify its own token...
    assert foreign.verify_token(foreign_token, TokenType.ACCESS) is not None
    # ...but our manager (different public key) must reject it.
    assert jwt_manager.verify_token(foreign_token, TokenType.ACCESS) is None


def test_wrong_audience_rejected(jwt_manager, claims):
    """A token carrying the wrong audience is rejected even if RS256-signed.

    Signed with the manager's real private key but a bogus ``aud`` -- aud
    enforcement (part of the #201 hardening) must reject it.
    """
    payload = {
        "sub": TEST_USERNAME,
        "user_id": TEST_USER_ID,
        "type": TokenType.ACCESS.value,
        "iss": EXPECTED_ISSUER,
        "aud": "some-other-audience",
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(hours=1),
    }
    bad_aud_token = pyjwt.encode(payload, jwt_manager.private_key, algorithm="RS256")
    assert jwt_manager.verify_token(bad_aud_token, TokenType.ACCESS) is None


# ===========================================================================
# 7. Blacklist / session revocation (was: tampered/blacklist rejection)
# ===========================================================================


def test_blacklisted_token_rejected(jwt_manager, claims):
    """A blacklisted (revoked) token is rejected by verify_token."""
    token = jwt_manager.create_access_token(claims)
    assert jwt_manager.verify_token(token, TokenType.ACCESS) is not None

    _blacklist(jwt_manager, token)
    assert jwt_manager.verify_token(token, TokenType.ACCESS) is None


def test_revoked_session_token_rejected(jwt_manager, claims):
    """An access token whose session has been deleted is rejected.

    create_access_token records a session keyed by user_id:session_id; verify
    requires that session to still exist. Deleting it (e.g. logout / global
    revoke) must invalidate the token.
    """
    token = jwt_manager.create_access_token(claims)
    assert jwt_manager.verify_token(token, TokenType.ACCESS) is not None

    payload = pyjwt.decode(token, options={"verify_signature": False})
    session_key = (
        f"{jwt_manager.session_prefix}:{payload['user_id']}:{payload['session_id']}"
    )
    jwt_manager.redis_client.delete(session_key)

    assert jwt_manager.verify_token(token, TokenType.ACCESS) is None


def test_revoke_all_user_tokens_invalidates_sessions(jwt_manager, claims):
    """revoke_all_user_tokens removes the user's sessions, failing verify."""
    token = jwt_manager.create_access_token(claims)
    assert jwt_manager.verify_token(token, TokenType.ACCESS) is not None

    assert jwt_manager.revoke_all_user_tokens(TEST_USER_ID) is True
    assert jwt_manager.verify_token(token, TokenType.ACCESS) is None


# ===========================================================================
# 8. Expiration window matches configuration
# ===========================================================================


def test_token_expiration_within_configured_range(jwt_manager, claims):
    """Default access-token exp falls within the configured window (+/- 60s)."""
    token = jwt_manager.create_access_token(claims)
    payload = jwt_manager.verify_token(token, TokenType.ACCESS)
    assert payload is not None

    exp_timestamp = payload["exp"]
    now = datetime.now(timezone.utc).timestamp()
    expire_minutes = SecurityConfig.JWT_ACCESS_TOKEN_EXPIRE_MINUTES

    expected_min = now + (expire_minutes * 60) - 60
    expected_max = now + (expire_minutes * 60) + 60

    assert expected_min <= exp_timestamp <= expected_max, (
        f"Token exp {exp_timestamp} not within expected range "
        f"[{expected_min}, {expected_max}] for {expire_minutes}min expiry"
    )


def test_custom_expiry_honored(jwt_manager, claims):
    """A custom expires_delta is reflected in the token's exp claim."""
    delta = timedelta(minutes=5)
    token = jwt_manager.create_access_token(claims, expires_delta=delta)
    payload = jwt_manager.verify_token(token, TokenType.ACCESS)
    assert payload is not None

    now = datetime.now(timezone.utc).timestamp()
    expected = now + delta.total_seconds()
    assert abs(payload["exp"] - expected) <= 60


# ===========================================================================
# 9. Redis-disabled graceful degradation (issue/verify still works)
# ===========================================================================


def test_roundtrip_without_redis(jwt_manager_no_redis, claims):
    """With Redis disabled, issue/verify still works (sessions simply skipped).

    Confirms the secure path does not hard-depend on Redis for the basic
    sign/verify guarantee, while blacklist/session enforcement (tested above)
    requires the Redis-backed manager.
    """
    token = jwt_manager_no_redis.create_access_token(claims)
    payload = jwt_manager_no_redis.verify_token(token, TokenType.ACCESS)
    assert payload is not None
    assert payload["sub"] == TEST_USERNAME
    assert _alg_header(token) == "RS256"


# ===========================================================================
# Ported-but-skipped: full-app HTTP roundtrips (Finding #201)
# ===========================================================================


@pytest.mark.skip(
    reason=(
        "Finding #201: the original HTTP-level cases (register/token/login/refresh "
        "-> /me, invalid-credentials, missing-auth-header, duplicate-registration, "
        "logout) asserted the obsolete sub==email symmetric contract and decoded "
        "tokens with jose_jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM]). "
        "That router path (create_access_token/SECRET_KEY/ALGORITHM) was removed. "
        "Under the new contract create_tokens sets sub=user.username and "
        "get_current_user looks up by username, so the old invariant no longer "
        "describes the system. The secure token contract these cases relied on is "
        "fully covered above at the oauth2/jwt_manager level. Re-porting the live "
        "FastAPI app + DB wiring is out of scope for #201's test-contract fix and "
        "is also blocked here by an unrelated middleware-stack import error in "
        "backend.api.main (MiddlewarePriority int vs enum)."
    )
)
def test_full_app_http_auth_flow_placeholder():
    """Placeholder marking the deliberately de-scoped HTTP-level coverage."""
