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
