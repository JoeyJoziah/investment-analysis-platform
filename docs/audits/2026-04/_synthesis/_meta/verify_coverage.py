#!/usr/bin/env python3
"""Verification gate: every finding ID from master appears in at least one workpaper."""
import json
import re
from pathlib import Path

ROOT = Path("docs/audits/2026-04/_synthesis")
META = ROOT / "_meta"
WP = ROOT / "workpaper"

findings = [json.loads(l) for l in (META / "findings-master.jsonl").read_text().splitlines() if l.strip()]
all_ids = sorted(f["finding_id"] for f in findings)
print(f"Total findings: {len(all_ids)}")

# Workpaper inventory
expected_workpapers = ["A", "B", "C", "D", "E", "F",
                        "G1_backend", "G2_ml_data_a_crit_high", "G2_ml_data_b_med_low",
                        "G3_frontend_infra", "G4_storage_security_residual",
                        "G5_tests_config_scripts", "G6_docs"]
missing_wp = [w for w in expected_workpapers if not (WP / f"{w}.md").exists()]
if missing_wp:
    print(f"MISSING workpapers: {missing_wp}")
    exit(1)
print(f"All {len(expected_workpapers)} workpapers present")

# Status inventory
missing_status = [w for w in expected_workpapers if not (META / "status" / f"{w}.json").exists()]
if missing_status:
    print(f"MISSING status JSONs: {missing_status}")
print(f"Status JSONs: {len(expected_workpapers) - len(missing_status)}/{len(expected_workpapers)}")

# Coverage: every finding ID must appear in at least one workpaper
fid_re = re.compile(r"F-\d{2}-\d{3}")
appearances = {fid: [] for fid in all_ids}
for w in expected_workpapers:
    text = (WP / f"{w}.md").read_text()
    found = set(fid_re.findall(text))
    for fid in found:
        if fid in appearances:
            appearances[fid].append(w)

missing_coverage = sorted([fid for fid, wps in appearances.items() if not wps])
multi_coverage = sorted([(fid, wps) for fid, wps in appearances.items() if len(wps) > 1])

print(f"\nCoverage: {len(all_ids) - len(missing_coverage)}/{len(all_ids)} findings appear in ≥1 workpaper")
if missing_coverage:
    print(f"MISSING ({len(missing_coverage)}): {missing_coverage[:20]}")
print(f"Multi-workpaper appearances: {len(multi_coverage)} (expected — cross-cluster references)")

# Acceptance tests non-empty check (heuristic: workpaper contains pytest|curl|grep|npm|docker|alembic|ruff|mypy)
test_keywords = re.compile(r"\b(pytest|curl|grep|npm|docker|alembic|ruff|mypy|markdown-link-check|gh\s+workflow|ls\s)")
print("\nAcceptance test presence (regex match for test commands):")
for w in expected_workpapers:
    text = (WP / f"{w}.md").read_text()
    matches = len(test_keywords.findall(text))
    print(f"  {w}: {matches} test-command-like patterns")

# Word-count sanity
print("\nWorkpaper sizes:")
for w in expected_workpapers:
    text = (WP / f"{w}.md").read_text()
    print(f"  {w}: {len(text)} chars / {len(text.split())} words")

# Final
exit(0 if not missing_coverage else 1)
