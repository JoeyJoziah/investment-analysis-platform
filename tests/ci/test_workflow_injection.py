"""
CI/CD workflow security guards (audit 2026-04, workstream G3 phase 1).

Asserts that GitHub Actions workflows under .github/workflows/ do NOT:

  1. Interpolate untrusted `github.event.{issue,pull_request,comment,review}.
     {title,body,message,name,text}` into shell `run:` blocks
     (findings F-14-001, F-14-002 — script injection / RCE-in-CI).

  2. Download TA-Lib over plaintext HTTP without a checksum
     (finding F-14-003 — MITM → root-on-runner).

These tests are the "fail-first" half of the audit's commit-pair: they
existed as failing assertions against the unfixed workflows; the same
assertions must pass against the fixed workflows.

The tests parse YAML and walk into each step's `run:` string so that
attribute references appearing only in safe positions (e.g. `env:`,
`if:`, GitHub Script JS contexts) are not falsely flagged.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

# Matches ${{ github.event.<obj>.<...>.{title|body|message|name|text} }} —
# the attacker-controlled scalar fields that GitHub renders verbatim into
# the YAML before the shell sees the script.
UNSAFE_EVENT_FIELD_RE = re.compile(
    r"\$\{\{\s*github\.event\.(?:issue|pull_request|comment|review)"
    r"(?:\.[A-Za-z_]+)*"
    r"\.(?:title|body|message|name|text)\s*\}\}"
)

TA_LIB_TARBALL = "ta-lib-0.4.0-src.tar.gz"


def _iter_workflow_files() -> list[Path]:
    return sorted(p for p in WORKFLOWS_DIR.glob("*.yml") if p.is_file())


def _iter_run_blocks(workflow_path: Path):
    """Yield (step_name, run_string) for every step that has a `run:` field."""
    with workflow_path.open() as fh:
        doc = yaml.safe_load(fh)
    if not isinstance(doc, dict):
        return
    jobs = doc.get("jobs") or {}
    if not isinstance(jobs, dict):
        return
    for job_name, job in jobs.items():
        if not isinstance(job, dict):
            continue
        steps = job.get("steps") or []
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            run = step.get("run")
            if isinstance(run, str):
                yield (f"{job_name}::{step.get('name', step.get('id', '?'))}", run)


@pytest.mark.parametrize("workflow", _iter_workflow_files(), ids=lambda p: p.name)
def test_no_unsafe_event_interpolation_in_run_blocks(workflow: Path) -> None:
    """F-14-001 / F-14-002: shell injection via untrusted issue/PR fields."""
    offenses = []
    for step_id, run_body in _iter_run_blocks(workflow):
        for match in UNSAFE_EVENT_FIELD_RE.finditer(run_body):
            offenses.append(f"  step={step_id} -> {match.group(0)}")
    assert not offenses, (
        f"{workflow.name}: untrusted github.event.*.{{title,body,message,name,text}} "
        f"is interpolated directly into a shell run: block. "
        f"This is a script-injection / RCE-in-CI vector (F-14-001 / F-14-002). "
        f"Pass the value via `env:` and reference it as a quoted shell variable.\n"
        + "\n".join(offenses)
    )


@pytest.mark.parametrize("workflow", _iter_workflow_files(), ids=lambda p: p.name)
def test_talib_download_is_https_and_checksum_verified(workflow: Path) -> None:
    """F-14-003: TA-Lib must be fetched over HTTPS with a sha256 verification."""
    for step_id, run_body in _iter_run_blocks(workflow):
        if TA_LIB_TARBALL not in run_body:
            continue

        assert "http://prdownloads.sourceforge.net" not in run_body, (
            f"{workflow.name}::{step_id}: TA-Lib is downloaded from "
            f"http://prdownloads.sourceforge.net. Plaintext HTTP allows "
            f"MITM substitution → root code execution on the runner "
            f"(F-14-003). Switch to https:// from the GitHub release mirror."
        )

        assert "https://" in run_body, (
            f"{workflow.name}::{step_id}: TA-Lib download step must use "
            f"https:// (F-14-003)."
        )

        assert "sha256sum -c" in run_body or "sha256sum --check" in run_body, (
            f"{workflow.name}::{step_id}: TA-Lib download step must verify "
            f"the tarball with `sha256sum -c` against a pinned digest "
            f"(F-14-003). HTTPS alone is insufficient — a compromised "
            f"release would still pass."
        )
