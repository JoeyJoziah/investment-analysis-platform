"""
F-01-008 regression test (PRD audit 2026-04 / Workstream F).

The ETagMiddleware excluded_paths list must include `/api/v1/metrics` so the
Prometheus scrape endpoint does not get an ETag header on every response. The
metrics body changes on every request and an ETag adds zero benefit while
forcing scrapers to manage cache validation state.
"""

from pathlib import Path


def _etag_registration_window(text: str) -> str:
    """Return the source window covering the ETagMiddleware registration block.

    The first textual occurrence of "ETagMiddleware" is the import statement;
    the registration sits hundreds of lines later. Anchor on the registration
    tuple ("etag",\n        ETagMiddleware) so the window captures the kwargs
    dict that contains excluded_paths.
    """
    anchor = '"etag",'
    idx = text.find(anchor)
    assert idx != -1, "ETagMiddleware registration anchor not found in main.py"
    return text[idx : idx + 1500]


def test_etag_excluded_paths_includes_v1_metrics() -> None:
    main_py = Path(__file__).resolve().parents[2] / "api" / "main.py"
    window = _etag_registration_window(main_py.read_text(encoding="utf-8"))
    assert '"/api/v1/metrics"' in window, (
        "/api/v1/metrics must be present in ETagMiddleware excluded_paths "
        "(see backend/api/main.py and PRD audit 2026-04 finding F-01-008)"
    )


def test_etag_excluded_paths_keeps_existing_entries() -> None:
    """The fix must not regress existing exclusions."""
    main_py = Path(__file__).resolve().parents[2] / "api" / "main.py"
    window = _etag_registration_window(main_py.read_text(encoding="utf-8"))
    for required in (
        '"/api/v1/auth/"',
        '"/api/v1/admin/"',
        '"/api/v1/ws/"',
        '"/api/health"',
    ):
        assert required in window, f"existing exclusion {required} was lost"
