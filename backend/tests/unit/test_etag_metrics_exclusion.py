"""
F-01-008 regression test (PRD audit 2026-04 / Workstream F).

The ETagMiddleware excluded_paths list must include `/api/v1/metrics` so the
Prometheus scrape endpoint does not get an ETag header on every response. The
metrics body changes on every request and an ETag adds zero benefit while
forcing scrapers to manage cache validation state.
"""

from pathlib import Path


def test_etag_excluded_paths_includes_v1_metrics() -> None:
    main_py = Path(__file__).resolve().parents[2] / "api" / "main.py"
    text = main_py.read_text(encoding="utf-8")

    # Find the ETag block and assert /api/v1/metrics appears inside its
    # excluded_paths list. The block lives after the comment "10. ETag".
    etag_marker = "ETagMiddleware"
    idx = text.find(etag_marker)
    assert idx != -1, "ETagMiddleware registration not found in main.py"

    # Take a generous window after the marker to capture the kwargs dict.
    window = text[idx : idx + 1500]
    assert '"/api/v1/metrics"' in window, (
        "/api/v1/metrics must be present in ETagMiddleware excluded_paths "
        "(see backend/api/main.py and PRD audit 2026-04 finding F-01-008)"
    )


def test_etag_excluded_paths_keeps_existing_entries() -> None:
    """The fix must not regress existing exclusions."""
    main_py = Path(__file__).resolve().parents[2] / "api" / "main.py"
    text = main_py.read_text(encoding="utf-8")
    idx = text.find("ETagMiddleware")
    window = text[idx : idx + 1500]
    for required in (
        '"/api/v1/auth/"',
        '"/api/v1/admin/"',
        '"/api/v1/ws/"',
        '"/api/health"',
    ):
        assert required in window, f"existing exclusion {required} was lost"
