"""CI-gate regression tests for scope-14 findings (2026-08 audit, cluster C3).

Source-level (pure file reads over .github/**), runs under
``pytest --noconftest``.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"


def _workflow_files():
    return sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml"))


class TestF8_14_001_MutableActionRefs:
    """Security actions must not float on a mutable branch ref: any push to
    the action's default branch would execute immediately inside jobs holding
    GITHUB_TOKEN (and, in production-deploy.yml, the production context)."""

    def test_no_uses_pinned_to_master_or_main(self):
        offenders = []
        for wf in _workflow_files():
            for i, line in enumerate(wf.read_text().split("\n"), 1):
                if re.search(r"uses:\s*[^@\s]+@(master|main)\s*$", line):
                    offenders.append(f"{wf.name}:{i}")
        assert offenders == [], offenders


class TestF8_14_003_ProductionTrivyGate:
    """The production security gate must fail when the Trivy reports are
    missing — otherwise jq emits nothing, wc -l says 0, and the step prints
    'deployment approved' after a failed scan."""

    def _gate(self):
        t = (WORKFLOWS / "production-deploy.yml").read_text()
        start = t.index("- name: Security gate check")
        end = t.index("- name:", start + 10)
        return t[start:end]

    def test_gate_uses_strict_shell_mode(self):
        assert "set -euo pipefail" in self._gate()

    def test_gate_checks_report_existence_before_jq(self):
        gate = self._gate()
        assert re.search(r'if \[ ! -f "\$f" \]', gate), "no existence preflight"
        assert gate.index("! -f") < gate.index("jq "), (
            "existence check must run before any jq parse"
        )


class TestF8_14_002_PermissionsBlocks:
    """Every workflow must declare a top-level permissions: block so jobs
    stop inheriting the repo-default GITHUB_TOKEN scope (F-14-006 regressed
    12 -> 22 missing)."""

    def test_every_workflow_declares_top_level_permissions(self):
        missing = [
            wf.name for wf in _workflow_files()
            if not re.search(r"^permissions:", wf.read_text(), re.M)
        ]
        assert missing == [], missing
