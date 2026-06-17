"""Regression test for HealthChecker.error_details population (B3 / F821).

HealthChecker.execute() previously computed ``error_details`` with
``str(e) if 'e' in locals() else None``. In Python 3 the ``except Exception
as e:`` target is deleted at the end of the except block, so after the retry
loop ``'e' in locals()`` is always False and ``error_details`` was always
None for failed checks (and ruff flagged the bare ``e`` as F821).

This test drives the retry path to exhaustion with a raising check function
and asserts the resulting HealthCheckResult carries the real exception text.
"""

import asyncio

import pytest

from backend.monitoring.health_checks import (
    HealthChecker,
    HealthStatus,
    ServiceType,
)


def test_execute_populates_error_details_on_failure():
    """A check that always raises must surface the exception text."""

    async def always_failing_check():
        raise RuntimeError("boom-db-down")

    checker = HealthChecker(
        name="db_ping",
        service="database",
        check_func=always_failing_check,
        service_type=ServiceType.DATABASE,
        timeout=1,
        retries=2,
        critical=True,
    )

    result = asyncio.run(checker.execute())

    # The check exhausted all retries and failed.
    assert result.status in (HealthStatus.CRITICAL, HealthStatus.UNHEALTHY)

    # The real regression guard: error_details must be a non-None string
    # containing the raised exception's message.
    assert result.error_details is not None
    assert isinstance(result.error_details, str)
    assert "boom-db-down" in result.error_details


def test_execute_error_details_none_on_success():
    """A passing check leaves error_details unset (None)."""

    async def healthy_check():
        return True

    checker = HealthChecker(
        name="db_ping",
        service="database",
        check_func=healthy_check,
        service_type=ServiceType.DATABASE,
        timeout=1,
        retries=0,
    )

    result = asyncio.run(checker.execute())

    assert result.status == HealthStatus.HEALTHY
    assert result.error_details is None


if __name__ == "__main__":  # pragma: no cover - manual run convenience
    raise SystemExit(pytest.main([__file__, "-q"]))
