"""F8-08-004: .secrets.baseline must be an audited allowlist, not 1,925
blanket suppressions.

The old baseline (2026-01-28) carried is_secret: null on every entry —
never triaged — while .pre-commit-config.yaml fed it to detect-secrets,
permanently suppressing every location including the #219 Postgres anchor.
1,386 entries were datasets/ noise and 346 were .claude/ noise.

Source-level JSON checks, runs under ``pytest --noconftest``.
"""

import json
from pathlib import Path

BASELINE = Path(__file__).resolve().parents[1] / ".secrets.baseline"


def _load():
    return json.loads(BASELINE.read_text())


class TestF8_08_004_Baseline:
    def test_every_entry_is_adjudicated(self):
        d = _load()
        nulls = [
            (f, e.get("line_number"))
            for f, entries in d["results"].items()
            for e in entries
            if e.get("is_secret") is None
        ]
        assert nulls == [], f"{len(nulls)} unaudited suppressions remain"

    def test_no_noise_directories(self):
        d = _load()
        noisy = [f for f in d["results"] if f.startswith(("datasets/", ".claude/"))]
        assert noisy == [], noisy

    def test_true_entries_exist_as_rotation_inventory(self):
        """The audit-confirmed live exposures must be marked is_secret: true —
        they are inputs to the U2/A1 rotation window, not false positives."""
        d = _load()
        trues = {
            f for f, entries in d["results"].items()
            if any(e.get("is_secret") is True for e in entries)
        }
        for expected in (
            ".env.secure",
            ".env.airflow",
            "scripts/deployment/start_data_loading.sh",
            "docs/reports/security-audit-report.md",
            ".context/refresh_analysis.sh",
        ):
            assert expected in trues, f"{expected} not marked as live exposure"

    def test_filters_exclude_noise_dirs_for_regeneration(self):
        d = _load()
        patterns = str(d.get("filters_used", []))
        assert "datasets" in patterns and "claude" in patterns, (
            "regeneration filters must keep excluding the noise directories"
        )
