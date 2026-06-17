"""Regression test: MiddlewareStack must accept raw int priorities.

Arithmetic on an IntEnum (e.g. ``MiddlewarePriority.CACHING - 100`` used in
backend/api/main.py for the etag middleware) yields a plain ``int``, not an
enum member. The stack previously accessed ``priority.name`` / ``priority.value``
unconditionally, raising ``'int' object has no attribute 'name'`` at app import
time and blocking startup entirely.
"""
from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware

from backend.middleware.stack import (
    MiddlewareStack,
    MiddlewarePriority,
    _priority_label,
    _priority_value,
)


class _NoopMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):  # pragma: no cover - trivial
        return await call_next(request)


def test_register_and_apply_with_int_priority_does_not_raise():
    """A raw int priority (CACHING - 100) registers, applies and summarizes."""
    app = FastAPI()
    stack = MiddlewareStack(app)

    int_priority = MiddlewarePriority.CACHING - 100  # -> plain int 1900
    assert not isinstance(int_priority, MiddlewarePriority)

    stack.register("etag_like", _NoopMiddleware, int_priority)
    stack.register("cors_like", _NoopMiddleware, MiddlewarePriority.CORS)

    # Must not raise on apply() (sorting) or summary (.name/.value formatting).
    stack.apply(is_testing=False)
    summary = stack.get_stack_summary()
    assert "etag_like" in summary
    assert "cors_like" in summary


def test_priority_helpers_handle_both_enum_and_int():
    assert _priority_value(MiddlewarePriority.CORS) == 9000
    assert _priority_value(1900) == 1900
    assert _priority_label(MiddlewarePriority.CORS) == "CORS"
    assert _priority_label(1900) == "CUSTOM(1900)"


def test_int_priority_sorts_correctly_between_enum_members():
    """1900 must sort between CACHING (2000) and COMPRESSION (1000)."""
    app = FastAPI()
    stack = MiddlewareStack(app)
    stack.register("compression", _NoopMiddleware, MiddlewarePriority.COMPRESSION)
    stack.register("etag_like", _NoopMiddleware, MiddlewarePriority.CACHING - 100)
    stack.register("caching", _NoopMiddleware, MiddlewarePriority.CACHING)

    order = [m.name for m in sorted(
        stack.middlewares, key=lambda m: _priority_value(m.priority), reverse=True
    )]
    assert order == ["caching", "etag_like", "compression"]
