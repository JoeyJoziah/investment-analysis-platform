"""
Regression test for #208 item 3.

Under the #201 token contract, ``create_tokens`` sets ``sub = user.username`` and
``get_current_user`` looks the user up by ``username``.  The async ``register``
endpoint created ``User`` rows without a ``username`` (it is nullable), so every
registered user received a token whose ``sub`` was null and all of their
authenticated requests 401'd.

This test exercises the real ``register`` coroutine with a mocked async DB
session and asserts the persisted user has ``username`` populated.

Run with::

    JWT_SECRET_KEY=x SECRET_KEY=y \
      pytest backend/tests/test_register_populates_username.py --noconftest
"""
import asyncio
import importlib
from unittest.mock import AsyncMock, MagicMock

auth_mod = importlib.import_module("backend.api.routers.auth")


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_register_populates_username(monkeypatch):
    captured = {}

    # Mocked AsyncSession: no existing user, capture the User passed to add().
    exec_result = MagicMock()
    exec_result.scalars.return_value.first.return_value = None

    db = MagicMock()
    db.execute = AsyncMock(return_value=exec_result)
    db.add = lambda obj: captured.__setitem__("user", obj)
    db.commit = AsyncMock()
    db.refresh = AsyncMock()

    # Keep the test off the RSA/redis token path and bcrypt.
    monkeypatch.setattr(auth_mod, "_issue_access_token", lambda u, r=None: "tok")
    monkeypatch.setattr(auth_mod, "get_password_hash", lambda p: "hashed")

    payload = auth_mod.UserCreate(
        email="newuser@example.com",
        full_name="New User",
        password="Str0ng!passw0rd",
    )

    _run(auth_mod.register(payload, MagicMock(), db, _rate_status=None))

    user = captured.get("user")
    assert user is not None, "register did not add a User to the session"
    # The core regression assertion: username must be populated (sub=username).
    assert user.username, (
        "register() must populate User.username; a null username makes the "
        "token sub null and breaks get_current_user lookups"
    )
    assert user.username == "newuser@example.com"
    assert user.email == "newuser@example.com"
