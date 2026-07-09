"""Phase 2 Lane B regression tests for ``generate_cache_key``.

``backend.utils.api_cache_decorators.generate_cache_key`` previously folded
EVERY positional arg via ``str(arg)`` and EVERY kwarg via ``f"{k}={v}"`` into
the cache key. When a decorated route received a FastAPI ``Request``, a
SQLAlchemy ``AsyncSession``, or the ``async_generator`` produced by the
DB-session dependency, that object stringified via ``object.__repr__`` -- which
embeds a per-instance ``id()``. The key therefore changed on every call
(non-deterministic), made those routes effectively uncacheable / occasionally
collision-prone, and surfaced the runtime warning "'async_generator' object
does not support the asynchronous context manager protocol".

Post-fix: only JSON-serializable scalar values (``str``/``int``/``float``/
``bool`` -- see ``_SERIALIZABLE``) are included. Non-scalar args are skipped, so
two calls with identical scalar args but DIFFERENT non-serializable args yield
identical keys.
"""

# CRITICAL: set TESTING/DATABASE_URL before importing the target module, which
# transitively imports backend.config.settings (matches test_utils_cache.py).
import os

os.environ.setdefault("TESTING", "True")
os.environ.setdefault("DEBUG", "True")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

import hashlib

import pytest

from backend.utils.api_cache_decorators import (
    _SERIALIZABLE,
    generate_cache_key,
)


# ---------------------------------------------------------------------------
# Test doubles that mimic the non-serializable arguments seen in production.
# ---------------------------------------------------------------------------


class _FakeAsyncSession:
    """Stand-in for sqlalchemy.ext.asyncio.AsyncSession.

    No __str__/__repr__ override, so it stringifies via object.__repr__ which
    embeds the per-instance id() -- exactly the failure mode under test.
    """


class _FakeRequestLike:
    """Stand-in for a FastAPI Request passed positionally (no query params)."""


async def _async_gen():
    """The DB-session dependency yields an async_generator like this."""
    yield object()


class _FakeModel:
    """Stand-in for an ORM/model object that is not JSON-serializable."""

    def __init__(self, value: int) -> None:
        self.value = value


class _MockRequest:
    """Minimal Request mock exposing query_params + optional state.user_id."""

    def __init__(self, query_params: dict | None = None, user_id=None) -> None:
        self.query_params = query_params or {}
        if user_id is not None:
            self.state = type("State", (), {"user_id": user_id})()


# ---------------------------------------------------------------------------
# (a) Scalar args/kwargs ARE included.
# ---------------------------------------------------------------------------


class TestScalarArgsIncluded:
    def test_serializable_tuple_is_scalar_only(self):
        assert _SERIALIZABLE == (str, int, float, bool)

    def test_positional_scalar_args_included(self):
        key = generate_cache_key("get_quote", args=("AAPL", 30, 1.5, True))
        assert key.startswith("get_quote:")
        assert "AAPL" in key
        assert "30" in key
        assert "1.5" in key

    def test_scalar_kwargs_included_and_sorted(self):
        key = generate_cache_key(
            "get_quote",
            kwargs={"symbol": "MSFT", "days": 7},
        )
        # sorted(kwargs) -> days before symbol, both present as k=v.
        assert "days=7" in key
        assert "symbol=MSFT" in key
        assert key.index("days=7") < key.index("symbol=MSFT")

    def test_distinct_scalar_values_yield_distinct_keys(self):
        k_aapl = generate_cache_key("get_quote", args=("AAPL",))
        k_msft = generate_cache_key("get_quote", args=("MSFT",))
        assert k_aapl != k_msft


# ---------------------------------------------------------------------------
# (b) Non-serializable objects are EXCLUDED -> identical keys regardless of
#     which Request / AsyncSession / async_generator / ORM instance is passed.
# ---------------------------------------------------------------------------


class TestNonSerializableExcluded:
    def test_async_session_excluded_from_args(self):
        session_a = _FakeAsyncSession()
        session_b = _FakeAsyncSession()
        # Same scalar arg, different (non-serializable) session instances.
        key_a = generate_cache_key("get_quote", args=("AAPL", session_a))
        key_b = generate_cache_key("get_quote", args=("AAPL", session_b))
        assert key_a == key_b, (
            "Phase 2 Lane B: differing AsyncSession instances must NOT change "
            "the cache key (they are not JSON-serializable scalars)"
        )
        # The id()-bearing repr must not leak into the key.
        assert "_FakeAsyncSession object at" not in key_a

    def test_async_session_excluded_from_kwargs(self):
        key_a = generate_cache_key(
            "get_quote", args=("AAPL",), kwargs={"db": _FakeAsyncSession()}
        )
        key_b = generate_cache_key(
            "get_quote", args=("AAPL",), kwargs={"db": _FakeAsyncSession()}
        )
        assert key_a == key_b
        assert "db=" not in key_a

    def test_async_generator_excluded(self):
        gen_a = _async_gen()
        gen_b = _async_gen()
        try:
            key_a = generate_cache_key("get_quote", args=("AAPL", gen_a))
            key_b = generate_cache_key("get_quote", args=("AAPL", gen_b))
            assert key_a == key_b, (
                "Phase 2 Lane B: async_generator dependency must be excluded "
                "from the cache key"
            )
            assert "async_generator" not in key_a
        finally:
            gen_a.aclose()
            gen_b.aclose()

    def test_request_like_positional_arg_excluded(self):
        key_a = generate_cache_key("get_quote", args=(_FakeRequestLike(), "AAPL"))
        key_b = generate_cache_key("get_quote", args=(_FakeRequestLike(), "AAPL"))
        assert key_a == key_b
        assert key_a == "get_quote:AAPL"

    def test_orm_model_object_excluded(self):
        key_a = generate_cache_key("get_quote", args=("AAPL", _FakeModel(1)))
        key_b = generate_cache_key("get_quote", args=("AAPL", _FakeModel(2)))
        assert key_a == key_b
        assert "_FakeModel object at" not in key_a

    def test_mixed_scalar_and_nonscalar_keeps_only_scalar(self):
        key = generate_cache_key(
            "get_quote",
            args=("AAPL", _FakeAsyncSession(), 30),
            kwargs={"db": _FakeAsyncSession(), "limit": 5},
        )
        assert key == "get_quote:AAPL:30:limit=5"


# ---------------------------------------------------------------------------
# (c) The long-key md5 hashing path still works.
# ---------------------------------------------------------------------------


class TestLongKeyHashing:
    def test_long_key_is_md5_hashed(self):
        # >200 chars of scalar payload forces the hash branch.
        long_value = "X" * 300
        key = generate_cache_key("get_quote", args=(long_value,))
        assert key.startswith("get_quote:hash_")

        expected_inner = f"get_quote:{long_value}"
        expected_hash = hashlib.md5(expected_inner.encode()).hexdigest()
        assert key == f"get_quote:hash_{expected_hash}"

    def test_short_key_is_not_hashed(self):
        key = generate_cache_key("get_quote", args=("AAPL",))
        assert "hash_" not in key
        assert key == "get_quote:AAPL"


# ---------------------------------------------------------------------------
# (d) Query params from a mock request are still included; user-id path intact.
# ---------------------------------------------------------------------------


class TestRequestQueryParams:
    def test_query_params_included_and_sorted(self):
        request = _MockRequest(query_params={"interval": "1d", "symbol": "AAPL"})
        key = generate_cache_key("get_quote", request=request)
        assert "q_interval=1d" in key
        assert "q_symbol=AAPL" in key
        # sorted by param name: interval before symbol.
        assert key.index("q_interval=1d") < key.index("q_symbol=AAPL")

    def test_query_params_combine_with_scalar_args(self):
        request = _MockRequest(query_params={"interval": "1d"})
        key = generate_cache_key("get_quote", args=("AAPL",), request=request)
        assert "AAPL" in key
        assert "q_interval=1d" in key

    def test_user_id_included_when_requested(self):
        request = _MockRequest(query_params={"interval": "1d"}, user_id="user-42")
        key = generate_cache_key("get_quote", request=request, include_user=True)
        assert "user=user-42" in key

    def test_user_id_omitted_when_not_requested(self):
        request = _MockRequest(query_params={"interval": "1d"}, user_id="user-42")
        key = generate_cache_key("get_quote", request=request, include_user=False)
        assert "user=user-42" not in key
