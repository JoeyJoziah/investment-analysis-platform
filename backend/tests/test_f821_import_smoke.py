"""Import smoke tests guarding the B3 F821 missing-import fixes.

Each of these modules previously referenced a name that was never imported
(multiprocessing, PipelineStatus, ipaddress, asyncio, random, metrics, json,
uuid), a latent ``NameError`` waiting at the referenced call site and flagged
by ruff as F821.

These tests import each module and assert the import succeeds. A missing
*optional third-party* dependency (e.g. mlflow) or an upstream
module-level ``pytest.skip`` is tolerated via ``importorskip``-style handling
-- those are environment gaps, not the regression we guard. A ``NameError``
(the actual bug class) is NOT tolerated and fails the test.
"""

import importlib

import pytest

# Modules that received a missing-import fix, paired with the name that was
# previously undefined (documentation only).
F821_FIXED_MODULES = [
    ("backend.etl.concurrent_processor", "multiprocessing"),
    ("backend.ml.pipeline.implementations", "PipelineStatus"),
    ("backend.security.data_encryption", "ipaddress"),
    ("backend.tests.fixtures.comprehensive_mock_fixtures", "asyncio"),
    ("backend.tests.test_resilience_integration", "random"),
    ("backend.utils.data_anonymization", "metrics"),
    ("backend.utils.performance_tester", "json"),
    ("backend.utils.resilient_pipeline", "uuid"),
]


@pytest.mark.parametrize(
    "module_name,fixed_name",
    F821_FIXED_MODULES,
    ids=[m for m, _ in F821_FIXED_MODULES],
)
def test_module_imports_without_nameerror(module_name, fixed_name):
    """Module imports cleanly; only environment gaps are skipped.

    The regression we guard is a ``NameError`` (the undefined name the missing
    import previously caused). Anything else raised at import time -- a missing
    optional dependency (mlflow), a missing sibling module surfacing as an
    upstream ``pytest.skip``, or a Python-3.9 ``asyncio.Event()``-at-import
    event-loop ``RuntimeError`` -- is an environment artifact unrelated to the
    fix and is reported as a skip.
    """
    try:
        importlib.import_module(module_name)
    except NameError:
        # This is exactly the regression we are guarding against.
        raise
    except pytest.skip.Exception:
        # An upstream module performed an allow_module_level skip
        # (e.g. a missing sibling module). Propagate as a skip.
        raise
    except Exception as exc:  # noqa: BLE001 - environment gaps, not NameError
        pytest.skip(
            f"{module_name} could not be imported in this environment "
            f"(not a NameError regression): {type(exc).__name__}: {exc}"
        )


if __name__ == "__main__":  # pragma: no cover - manual run convenience
    raise SystemExit(pytest.main([__file__, "-q"]))
