"""Regression guard for F-09-002: scan_market_streaming must exist.

``recommendation_optimized._scan_market_optimized`` consumes
``scanner.scan_market_streaming(...)`` as an async generator. That method was
never defined on :class:`MarketScanner`, which is a latent ``AttributeError``
land-mine on the recommendation hot path. This test asserts the interim
chunked wrapper exists, is an async generator, and yields fixed-size chunks.

Run source-level (the repo conftest eagerly imports the full app)::

    python3 -m pytest backend/tests/test_market_scanner_streaming_guard.py \
        --noconftest -q

The chunking behaviour is verified without live infra by extracting the
``scan_market_streaming`` coroutine function from source and binding it to a
fake ``self`` whose ``scan_market`` returns a small in-memory list. This keeps
the test free of the heavy ``backend.*`` import tree.
"""

import ast
import asyncio
import types
from pathlib import Path

import pytest

_SCANNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "data_ingestion"
    / "market_scanner.py"
)


def _scanner_class_node() -> ast.ClassDef:
    tree = ast.parse(_SCANNER_PATH.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "MarketScanner":
            return node
    raise AssertionError("MarketScanner class not found in market_scanner.py")


def _streaming_method_node() -> ast.AsyncFunctionDef:
    for item in _scanner_class_node().body:
        if (
            isinstance(item, ast.AsyncFunctionDef)
            and item.name == "scan_market_streaming"
        ):
            return item
    raise AssertionError(
        "MarketScanner.scan_market_streaming is not defined "
        "(F-09-002 AttributeError land-mine)"
    )


def test_scan_market_streaming_defined_as_async_generator():
    """The wrapper must exist and be an async generator (has a yield)."""
    node = _streaming_method_node()

    has_yield = any(
        isinstance(inner, (ast.Yield, ast.YieldFrom))
        for inner in ast.walk(node)
    )
    assert has_yield, "scan_market_streaming must yield (be an async generator)"

    arg_names = {a.arg for a in node.args.args}
    # Must accept the call contract used at recommendation_optimized.py:295.
    assert {"sectors", "market_cap_range", "chunk_size"} <= arg_names


def _load_streaming_func():
    """Compile only the wrapper function, isolated from heavy module imports."""
    node = _streaming_method_node()
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict = {
        "AsyncIterator": __import__("typing").AsyncIterator,
        "List": __import__("typing").List,
        "Dict": __import__("typing").Dict,
        "Any": __import__("typing").Any,
        "Optional": __import__("typing").Optional,
        "Tuple": __import__("typing").Tuple,
    }
    code = compile(module, str(_SCANNER_PATH), "exec")
    exec(code, namespace)  # noqa: S102 - controlled, single-function source
    return namespace["scan_market_streaming"]


def test_scan_market_streaming_yields_fixed_size_chunks():
    """Bound to a fake self, the wrapper chunks scan_market output."""
    streaming_func = _load_streaming_func()

    fake_stocks = [{"ticker": f"T{i}"} for i in range(25)]

    class FakeScanner:
        async def scan_market(self, **kwargs):
            return list(fake_stocks)

    fake = FakeScanner()
    bound = types.MethodType(streaming_func, fake)

    async def _collect():
        chunks = []
        async for chunk in bound(chunk_size=10):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(_collect())

    assert [len(c) for c in chunks] == [10, 10, 5]
    assert [s for c in chunks for s in c] == fake_stocks


def test_scan_market_streaming_rejects_nonpositive_chunk_size():
    streaming_func = _load_streaming_func()

    class FakeScanner:
        async def scan_market(self, **kwargs):
            return []

    bound = types.MethodType(streaming_func, FakeScanner())

    async def _drain():
        async for _ in bound(chunk_size=0):
            pass

    with pytest.raises(ValueError):
        asyncio.run(_drain())
