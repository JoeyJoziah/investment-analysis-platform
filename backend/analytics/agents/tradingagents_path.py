"""
TradingAgents path resolution (fixes F-01-005).

The previous default was a hardcoded Windows workstation path with a
fallback to backend/TradingAgents/, which does not exist in this repo —
so every non-Windows environment silently fell back to stubs. Resolution
order is now portable:

1. TRADINGAGENTS_PATH env var (when it points at an existing directory)
2. The internal archived copy (backend/_archive_TradingAgents_fork_pre_2026-05-12)
3. A sibling stockanalysistool checkout (<repo>/../stockanalysistool/TradingAgents)
4. The legacy backend/TradingAgents location
"""

import os
from pathlib import Path
from typing import Optional, Union

ARCHIVE_DIRNAME = "_archive_TradingAgents_fork_pre_2026-05-12"


def default_backend_dir() -> Path:
    """The backend/ directory of this repository."""
    return Path(__file__).resolve().parents[2]


def resolve_tradingagents_path(
    backend_dir: Union[str, Path, None] = None,
) -> Optional[str]:
    """Return the first existing TradingAgents directory, or None.

    None means no candidate exists and callers should use their stub
    classes rather than inserting a dead path into sys.path.
    """
    backend = Path(backend_dir) if backend_dir is not None else default_backend_dir()

    env_override = os.environ.get("TRADINGAGENTS_PATH")
    candidates = []
    if env_override:
        candidates.append(Path(env_override))
    candidates.extend(
        [
            backend / ARCHIVE_DIRNAME,
            backend.parent.parent / "stockanalysistool" / "TradingAgents",
            backend / "TradingAgents",
        ]
    )

    for candidate in candidates:
        if candidate.is_dir():
            return str(candidate)
    return None
