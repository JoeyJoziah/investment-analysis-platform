"""
PRD audit 2026-04 / Workstream F regression tests.

These tests assert that the dead V1-deprecation machinery has been removed from
backend.api.versioning and that no caller imports the deleted symbols. The
historical V1DeprecationMiddleware was incorrectly treating the current
`/api/v1/` prefix as a sunset legacy prefix and emitting Sunset/Deprecation/
Warning headers + log spam on every production request.

Findings covered:
- F-01-003 (V1DeprecationMiddleware removed)
- F-01-007 (V1_TO_V2_ENDPOINT_MAP removed)
- F-01-017 (create_versioned_router / register_router removed)
"""

import importlib
from pathlib import Path

import pytest

from backend.api import versioning


REMOVED_SYMBOLS = (
    "V1DeprecationMiddleware",
    "V1_TO_V2_ENDPOINT_MAP",
    "V1_TO_V2_PARAM_MAP",
    "map_v1_endpoint_to_v2",
    "transform_v1_params_to_v2",
    "create_versioned_router",
)


@pytest.mark.parametrize("symbol", REMOVED_SYMBOLS)
def test_dead_v1_symbols_removed_from_versioning(symbol: str) -> None:
    """The legacy V1 deprecation symbols must not exist on backend.api.versioning."""
    assert not hasattr(versioning, symbol), (
        f"{symbol} should have been removed in PRD audit 2026-04 / Workstream F"
    )


def test_register_router_method_removed_from_version_manager() -> None:
    """APIVersionManager.register_router was unused dead code; it must be gone."""
    manager = versioning.version_manager
    assert not hasattr(manager, "register_router"), (
        "APIVersionManager.register_router should have been removed in "
        "PRD audit 2026-04 / Workstream F"
    )


def test_no_backend_module_imports_v1_deprecation_middleware() -> None:
    """No backend module may import the deleted V1DeprecationMiddleware.

    Scans actual import statements (single-line and parenthesised multi-line)
    rather than file-wide substrings, so explanatory comments mentioning the
    removed symbol do not register as offenders.
    """
    import re

    backend_root = Path(__file__).resolve().parents[2]
    # Matches `import V1DeprecationMiddleware` and `from ... import (... V1DeprecationMiddleware ...)`
    single_line = re.compile(r"^\s*(?:from\s+\S+\s+)?import\s+[^\n#]*\bV1DeprecationMiddleware\b", re.MULTILINE)
    multi_line = re.compile(r"^\s*from\s+\S+\s+import\s*\([^)]*\bV1DeprecationMiddleware\b[^)]*\)", re.MULTILINE | re.DOTALL)

    offenders: list[str] = []
    for path in backend_root.rglob("*.py"):
        if path == Path(__file__) or "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if single_line.search(text) or multi_line.search(text):
            offenders.append(str(path))
    assert offenders == [], (
        "These files still import the deleted V1DeprecationMiddleware: "
        f"{offenders}"
    )


def test_versioning_module_still_imports_cleanly() -> None:
    """Sanity: the trimmed module still imports without errors."""
    module = importlib.reload(versioning)
    assert hasattr(module, "v1_migration_router")
    assert hasattr(module, "v1_migration_metrics")
    assert hasattr(module, "APIVersion")
