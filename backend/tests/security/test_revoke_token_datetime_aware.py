"""Regression test: JWTManager.revoke_token must not raise on tz mismatch.

``datetime.fromtimestamp(exp)`` returns a NAIVE datetime; subtracting the
timezone-AWARE ``datetime.now(timezone.utc)`` raised ``TypeError: can't subtract
offset-naive and offset-aware datetimes`` inside the try/except, so revoke_token
silently returned False and NOTHING was ever blacklisted (logout never revoked).
"""
import hashlib
from datetime import datetime, timedelta, timezone

import jwt as pyjwt

from backend.security.jwt_manager import JWTManager


class _FakeRedis:
    def __init__(self):
        self.store = {}

    def setex(self, key, ttl, value):
        self.store[key] = (ttl, value)


def _make_manager():
    mgr = JWTManager.__new__(JWTManager)  # bypass __init__ (would need live Redis)
    mgr.blacklist_prefix = "blacklist"
    mgr.redis_client = _FakeRedis()
    return mgr


def _token_expiring_in(hours: int) -> str:
    exp = int((datetime.now(timezone.utc) + timedelta(hours=hours)).timestamp())
    return pyjwt.encode({"exp": exp, "sub": "u1"}, "k" * 32, algorithm="HS256")


def test_revoke_token_blacklists_unexpired_token():
    mgr = _make_manager()
    token = _token_expiring_in(1)

    result = mgr.revoke_token(token)

    assert result is True
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    assert f"blacklist:{token_hash}" in mgr.redis_client.store
    ttl, _ = mgr.redis_client.store[f"blacklist:{token_hash}"]
    assert ttl > 0  # positive TTL proves the aware/naive subtraction succeeded


def test_revoke_already_expired_token_is_noop_not_error():
    mgr = _make_manager()
    token = _token_expiring_in(-1)  # already expired

    result = mgr.revoke_token(token)

    assert result is True
    assert mgr.redis_client.store == {}  # nothing to blacklist, but no exception


def test_revoke_without_redis_blacklists_in_memory():
    mgr = JWTManager.__new__(JWTManager)
    mgr.blacklist_prefix = "jwt_blacklist"
    mgr.session_prefix = "user_session"
    mgr.redis_client = None
    mgr._memory_blacklist = {}
    token = _token_expiring_in(1)

    assert mgr.revoke_token(token) is True
    assert mgr._is_token_blacklisted(token) is True


def test_memory_blacklist_ignores_expired_entry():
    mgr = JWTManager.__new__(JWTManager)
    mgr.blacklist_prefix = "jwt_blacklist"
    mgr.redis_client = None
    mgr._memory_blacklist = {}
    token = _token_expiring_in(-1)

    assert mgr.revoke_token(token) is True
    assert mgr._is_token_blacklisted(token) is False
