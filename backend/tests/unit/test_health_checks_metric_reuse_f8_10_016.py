"""F8-10-016: _get_or_create_metric must not reuse a name-clashing collector
whose labelnames differ from what the caller asked for.

Without the guard, a duplicate-registration ValueError is resolved by handing
back whatever collector already owns the name — e.g. metrics_collector's
1-label ``health_check_status`` gauge would be returned to health_checks
callers that label with (service, check_type), and every ``.labels(...)`` call
would then blow up (or silently mislabel) far from the registration site.

Runs source-level under ``pytest --noconftest``.
"""

# Required env must exist before importing backend modules — settings
# instantiates at import (same preamble as the other --noconftest tests).
import os

os.environ.setdefault("TESTING", "True")
os.environ.setdefault("ENVIRONMENT", "testing")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-testing-only")
os.environ.setdefault("JWT_SECRET_KEY", "test-jwt-secret-key-for-testing-only")
os.environ.setdefault("SESSION_SECRET_KEY", "test-session-secret-for-testing-only")
os.environ.setdefault("MASTER_SECRET_KEY", "m" * 130)

import pytest
from prometheus_client import REGISTRY, Gauge

from backend.monitoring.health_checks import _get_or_create_metric


@pytest.fixture
def probe_name():
    name = "f8_10_016_probe_metric"
    yield name
    collector = REGISTRY._names_to_collectors.get(name)
    if collector is not None:
        REGISTRY.unregister(collector)


def test_labelname_mismatch_raises_instead_of_reusing(probe_name):
    """A name clash with DIFFERENT labelnames must raise, not hand back the
    other module's collector (F8-10-016)."""
    Gauge(probe_name, "probe with one label", ["service"])
    with pytest.raises(ValueError):
        _get_or_create_metric(
            Gauge, probe_name, "same name, three labels",
            ["service", "check_type", "extra"],
        )


def test_matching_labelnames_reuse_existing_collector(probe_name):
    """Idempotent re-registration (the T2.7 collect-only fix) must keep
    working when the labelnames DO match."""
    first = Gauge(probe_name, "probe", ["service", "check_type"])
    again = _get_or_create_metric(
        Gauge, probe_name, "probe", ["service", "check_type"]
    )
    assert again is first


def test_type_mismatch_raises_instead_of_reusing(probe_name):
    """A clash where the existing collector is a different metric type must
    also raise — a Gauge caller cannot operate a reused non-Gauge."""
    from prometheus_client import Counter

    Gauge(probe_name, "gauge occupying the name", ["service"])
    with pytest.raises(ValueError):
        _get_or_create_metric(Counter, probe_name, "counter wants the name", ["service"])
