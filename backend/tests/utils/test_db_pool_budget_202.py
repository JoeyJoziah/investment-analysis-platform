"""
Regression tests for finding #202 (connection-pool slice):
  backend/utils/db_read_replicas.py pool budget guard.

These tests are purely unit-level: no live database required, no full
application startup needed.  The module under test imports
`backend.config.settings.settings` at module level; we patch that import
out so the tests run without a populated .env file.

Run with:
    pytest backend/tests/utils/test_db_pool_budget_202.py -v --noconftest
"""

import os
import sys
import types
import importlib
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Patch backend.config.settings before importing the module under test.
# backend/utils/db_read_replicas.py has `from backend.config.settings import settings`
# at the top level.  We inject a stub so the module can be imported without
# a real .env file.
# ---------------------------------------------------------------------------

def _ensure_module_importable():
    """
    Insert a minimal stub for backend.config.settings into sys.modules so that
    importing db_read_replicas does not trigger pydantic Settings validation.
    Idempotent — safe to call multiple times.
    """
    stub_key = "backend.config.settings"
    if stub_key not in sys.modules or not isinstance(
        getattr(sys.modules[stub_key], "settings", None), MagicMock
    ):
        stub = types.ModuleType(stub_key)
        stub.settings = MagicMock()
        stub.settings.DATABASE_URL = "postgresql://user:pass@localhost:5432/testdb"
        sys.modules[stub_key] = stub

        # Also ensure the parent package is present
        if "backend.config" not in sys.modules:
            pkg = types.ModuleType("backend.config")
            sys.modules["backend.config"] = pkg

    # Force a clean re-import of the target module so it picks up our stub
    sys.modules.pop("backend.utils.db_read_replicas", None)


_ensure_module_importable()


# Now the import is safe
from backend.utils.db_read_replicas import (  # noqa: E402
    _DEFAULT_MAX_CONNECTIONS,
    _DEFAULT_PRIMARY_POOL_SIZE,
    _DEFAULT_PRIMARY_MAX_OVERFLOW,
    _DEFAULT_REPLICA_POOL_SIZE,
    _DEFAULT_REPLICA_MAX_OVERFLOW,
    _get_pool_config,
    _assert_pool_budget,
    PoolBudgetExceededError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pool_cfg_from_env(env: dict) -> dict:
    """
    Call _get_pool_config() with specific env vars set, then restore originals.
    """
    original = {k: os.environ.get(k) for k in env}
    for k, v in env.items():
        os.environ[k] = str(v)
    try:
        return _get_pool_config()
    finally:
        for k, orig in original.items():
            if orig is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = orig


def _call_assert_with_env(env: dict, replica_count: int) -> None:
    """Set env vars, call _assert_pool_budget, then restore env."""
    original = {k: os.environ.get(k) for k in env}
    for k, v in env.items():
        os.environ[k] = str(v)
    try:
        _assert_pool_budget(_get_pool_config(), replica_count)
    finally:
        for k, orig in original.items():
            if orig is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = orig


def _clear_pool_env_vars():
    """Remove all pool-related env vars so defaults are used cleanly."""
    keys = [
        "DB_MAX_CONNECTIONS",
        "DB_PRIMARY_POOL_SIZE", "DB_PRIMARY_MAX_OVERFLOW",
        "DB_REPLICA_POOL_SIZE", "DB_REPLICA_MAX_OVERFLOW",
    ]
    return {k: os.environ.pop(k, None) for k in keys}


def _restore_env(saved: dict):
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v


# ---------------------------------------------------------------------------
# Tests: grand total with default values stays within budget
# ---------------------------------------------------------------------------

class TestDefaultPoolBudget:
    """Default configuration must stay within the 50-connection ceiling."""

    def test_default_grand_total_within_ceiling(self):
        """
        With defaults: primary(8+4) + 2 replicas×(5+2) = 12+14 = 26 ≤ 50.
        """
        primary_max = _DEFAULT_PRIMARY_POOL_SIZE + _DEFAULT_PRIMARY_MAX_OVERFLOW
        per_replica_max = _DEFAULT_REPLICA_POOL_SIZE + _DEFAULT_REPLICA_MAX_OVERFLOW
        replica_count = 2  # standard deployment
        grand_total = primary_max + replica_count * per_replica_max

        assert grand_total <= _DEFAULT_MAX_CONNECTIONS, (
            f"Default grand total {grand_total} exceeds ceiling {_DEFAULT_MAX_CONNECTIONS} — "
            "finding #202 re-introduced"
        )

    def test_default_grand_total_exact_arithmetic(self):
        """Pin the exact arithmetic so any accidental change fails fast."""
        # primary 8+4=12, each replica 5+2=7, two replicas → 26
        assert _DEFAULT_PRIMARY_POOL_SIZE + _DEFAULT_PRIMARY_MAX_OVERFLOW == 12
        assert _DEFAULT_REPLICA_POOL_SIZE + _DEFAULT_REPLICA_MAX_OVERFLOW == 7

    def test_assert_pool_budget_passes_with_defaults_and_2_replicas(self):
        """Budget guard must not raise with the safe defaults and 2 replicas."""
        saved = _clear_pool_env_vars()
        try:
            _assert_pool_budget(_get_pool_config(), replica_count=2)
            # No exception == pass
        finally:
            _restore_env(saved)

    def test_assert_pool_budget_passes_with_zero_replicas(self):
        """Primary-only deployment (no replicas) must also pass."""
        saved = _clear_pool_env_vars()
        try:
            _assert_pool_budget(_get_pool_config(), replica_count=0)
        finally:
            _restore_env(saved)


# ---------------------------------------------------------------------------
# Tests: construction-time guard raises when misconfigured
# ---------------------------------------------------------------------------

class TestPoolBudgetGuardRaises:
    """PoolBudgetExceededError must be raised for over-budget configurations."""

    def test_original_buggy_values_exceed_ceiling(self):
        """
        The pre-fix values (primary 20+40, replica 15+30 each) with 2 replicas
        produce 150 connections — well above 50.  The guard must raise.
        """
        _env = {
            "DB_MAX_CONNECTIONS": "50",
            "DB_PRIMARY_POOL_SIZE": "20",
            "DB_PRIMARY_MAX_OVERFLOW": "40",
            "DB_REPLICA_POOL_SIZE": "15",
            "DB_REPLICA_MAX_OVERFLOW": "30",
        }
        with pytest.raises(PoolBudgetExceededError) as exc_info:
            _call_assert_with_env(_env, replica_count=2)

        # Confirm message contains useful context
        msg = str(exc_info.value)
        assert "150" in msg or "exceeded" in msg.lower()

    def test_primary_alone_exceeds_ceiling(self):
        """Even a single over-sized primary must be caught."""
        _env = {
            "DB_MAX_CONNECTIONS": "10",
            "DB_PRIMARY_POOL_SIZE": "8",
            "DB_PRIMARY_MAX_OVERFLOW": "4",   # primary max = 12 > ceiling 10
            "DB_REPLICA_POOL_SIZE": "1",
            "DB_REPLICA_MAX_OVERFLOW": "0",
        }
        with pytest.raises(PoolBudgetExceededError):
            _call_assert_with_env(_env, replica_count=0)

    def test_replicas_push_total_over_ceiling(self):
        """Primary within budget, but two replicas tip it over."""
        _env = {
            "DB_MAX_CONNECTIONS": "20",
            "DB_PRIMARY_POOL_SIZE": "8",
            "DB_PRIMARY_MAX_OVERFLOW": "4",   # primary = 12
            "DB_REPLICA_POOL_SIZE": "5",
            "DB_REPLICA_MAX_OVERFLOW": "3",   # per replica = 8; 2×8=16 → total 28 > 20
        }
        with pytest.raises(PoolBudgetExceededError):
            _call_assert_with_env(_env, replica_count=2)

    def test_error_message_contains_grand_total_and_ceiling(self):
        """Error message must expose grand total and ceiling for operator debugging."""
        _env = {
            "DB_MAX_CONNECTIONS": "50",
            "DB_PRIMARY_POOL_SIZE": "20",
            "DB_PRIMARY_MAX_OVERFLOW": "40",
            "DB_REPLICA_POOL_SIZE": "15",
            "DB_REPLICA_MAX_OVERFLOW": "30",
        }
        with pytest.raises(PoolBudgetExceededError) as exc_info:
            _call_assert_with_env(_env, replica_count=2)

        msg = str(exc_info.value)
        # Must mention the ceiling value
        assert "50" in msg
        # Must mention the computed total (150)
        assert "150" in msg


# ---------------------------------------------------------------------------
# Tests: env-var overrides work correctly
# ---------------------------------------------------------------------------

class TestEnvVarOverrides:
    """Pool config must be fully driven by environment variables."""

    def test_custom_max_connections_from_env(self):
        cfg = _pool_cfg_from_env({"DB_MAX_CONNECTIONS": "100"})
        assert cfg["max_connections"] == 100

    def test_custom_primary_pool_size_from_env(self):
        cfg = _pool_cfg_from_env({"DB_PRIMARY_POOL_SIZE": "3"})
        assert cfg["primary_pool_size"] == 3

    def test_custom_primary_max_overflow_from_env(self):
        cfg = _pool_cfg_from_env({"DB_PRIMARY_MAX_OVERFLOW": "2"})
        assert cfg["primary_max_overflow"] == 2

    def test_custom_replica_pool_size_from_env(self):
        cfg = _pool_cfg_from_env({"DB_REPLICA_POOL_SIZE": "2"})
        assert cfg["replica_pool_size"] == 2

    def test_custom_replica_max_overflow_from_env(self):
        cfg = _pool_cfg_from_env({"DB_REPLICA_MAX_OVERFLOW": "1"})
        assert cfg["replica_max_overflow"] == 1

    def test_larger_ceiling_allows_larger_pools(self):
        """Raising DB_MAX_CONNECTIONS lets operators configure bigger pools."""
        _env = {
            "DB_MAX_CONNECTIONS": "200",
            "DB_PRIMARY_POOL_SIZE": "20",
            "DB_PRIMARY_MAX_OVERFLOW": "40",
            "DB_REPLICA_POOL_SIZE": "15",
            "DB_REPLICA_MAX_OVERFLOW": "30",
        }
        # Should NOT raise with a 200-connection ceiling
        _call_assert_with_env(_env, replica_count=2)

    def test_budget_passes_at_exact_ceiling(self):
        """A grand total equal to the ceiling (not exceeding) must pass."""
        _env = {
            "DB_MAX_CONNECTIONS": "26",
            "DB_PRIMARY_POOL_SIZE": "8",
            "DB_PRIMARY_MAX_OVERFLOW": "4",   # 12
            "DB_REPLICA_POOL_SIZE": "5",
            "DB_REPLICA_MAX_OVERFLOW": "2",   # 7 each; 2×7=14; total=26 == ceiling
        }
        _call_assert_with_env(_env, replica_count=2)  # must not raise

    def test_budget_fails_one_over_ceiling(self):
        """Grand total == ceiling+1 must raise."""
        _env = {
            "DB_MAX_CONNECTIONS": "25",       # one under 26
            "DB_PRIMARY_POOL_SIZE": "8",
            "DB_PRIMARY_MAX_OVERFLOW": "4",   # 12
            "DB_REPLICA_POOL_SIZE": "5",
            "DB_REPLICA_MAX_OVERFLOW": "2",   # 7 each; 2×7=14; total=26 > 25
        }
        with pytest.raises(PoolBudgetExceededError):
            _call_assert_with_env(_env, replica_count=2)
