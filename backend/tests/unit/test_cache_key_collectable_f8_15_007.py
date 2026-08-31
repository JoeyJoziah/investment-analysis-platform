"""F8-15-007: test_cache_key.py must be collectable in isolation.

The module previously imported the backend settings chain with an incomplete
env preamble (missing SECRET_KEY / JWT_SECRET_KEY / REDIS_URL), so its import
outcome depended on what earlier test modules had put in ``os.environ`` —
pydantic ``ValidationError`` alone, ``TypeError`` in-suite. This meta-test
runs the acceptance criterion directly: a clean-env, isolated
``--collect-only --noconftest`` of the file must succeed.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TARGET = "backend/tests/unit/test_cache_key.py"

# Env vars whose presence (leaked from other test modules) previously masked
# the incomplete preamble. The subprocess must succeed WITHOUT them.
_LEAKABLE = (
    "TESTING", "DEBUG", "ENVIRONMENT", "DATABASE_URL", "REDIS_URL",
    "SECRET_KEY", "JWT_SECRET_KEY", "SESSION_SECRET_KEY", "MASTER_SECRET_KEY",
)


def test_cache_key_collects_in_isolation_with_clean_env():
    env = {k: v for k, v in os.environ.items() if k not in _LEAKABLE}
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "--noconftest",
         "-q", TARGET],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, (
        f"collect-only failed (rc={result.returncode}):\n"
        f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
    )
    assert " error" not in result.stdout.lower().split("\n")[-2], result.stdout[-500:]
