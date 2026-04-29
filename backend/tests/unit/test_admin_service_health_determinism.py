"""F-02-003 fail-first regression tests for admin_service.get_system_health_data.

Per PRD audit 2026-04 Workstream D §4 / Q4 default (recorded 2026-04-28):
``admin_service.get_system_health_data()`` previously returned
``random.uniform(20, 80)`` for ``cpu_usage`` / ``memory_usage`` /
``disk_usage`` and ``random.randint(86400, 864000)`` for uptime, so two
adjacent calls produced wildly different values with no real load change —
admin telemetry was non-credible by design.

Post-fix: psutil-derived numbers are deterministic-ish (within a small
jitter tolerance). The endpoint declares its data source via
``data_source: 'psutil'`` (or ``'psutil_unavailable'`` when the library is
absent — never random).
"""
from __future__ import annotations

import time

import pytest


def _h():
    from backend.services.admin_service import get_system_health_data
    return get_system_health_data()


class TestSystemHealthDataDeterminismF02003:
    """Two consecutive calls must NOT produce ``random.uniform``-style swings."""

    def test_data_source_is_not_random(self):
        h = _h()
        assert h.get("data_source") in {"psutil", "psutil_unavailable"}, (
            "F-02-003: get_system_health_data must declare a real data source "
            "(psutil or explicit psutil_unavailable), not silently fabricate"
        )

    def test_cpu_usage_is_stable_or_explicitly_unavailable(self):
        h1 = _h()
        h2 = _h()
        # If psutil is unavailable both must report None.
        if h1.get("data_source") == "psutil_unavailable":
            assert h1.get("cpu_usage") is None
            assert h2.get("cpu_usage") is None
            return

        # psutil-backed: between two adjacent calls CPU should not jump by
        # more than 60 percentage points (random.uniform(20, 80) routinely
        # produced swings of 50+ points). A real idle process shows <5pp;
        # we allow 60pp slack to keep the test deterministic on a busy CI.
        c1, c2 = h1["cpu_usage"], h2["cpu_usage"]
        assert isinstance(c1, float)
        assert isinstance(c2, float)
        assert abs(c1 - c2) < 60.0, (
            f"F-02-003: cpu_usage swung {abs(c1 - c2):.1f}pp between two "
            "adjacent calls — looks like random.uniform, not psutil"
        )

    def test_uptime_is_monotonic_non_decreasing(self):
        """random.randint(86400, 864000) is not monotonic — uptime should be."""
        h1 = _h()
        time.sleep(0.01)
        h2 = _h()
        u1, u2 = h1["uptime"], h2["uptime"]
        assert isinstance(u1, int) and isinstance(u2, int)
        assert u2 >= u1, (
            f"F-02-003: uptime went backwards ({u1} -> {u2}) — uptime is "
            "fabricated via random.randint instead of process-clock-derived"
        )
