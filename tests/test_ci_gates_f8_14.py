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


class TestF8_14_005_CanaryInterpolation:
    """workflow_dispatch inputs must not be YAML-interpolated into run:
    shell/heredoc text — the exact ${{ }}-into-shell pattern G3 closed."""

    def test_no_event_or_input_interpolation_inside_run_blocks(self):
        text = (WORKFLOWS / "canary-deploy.yml").read_text()
        offenders = []
        in_run = False
        run_indent = 0
        for i, line in enumerate(text.split("\n"), 1):
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            if stripped.startswith("run:"):
                in_run, run_indent = True, indent
                continue
            if in_run and stripped and indent <= run_indent:
                in_run = False
            if in_run and re.search(r"\$\{\{\s*(github\.event\.|inputs\.)", line):
                offenders.append(i)
        assert offenders == [], f"input interpolation inside run blocks at lines {offenders}"


class TestF8_14_011_CanaryHealthUrl:
    """The canary health gate must not default to a reserved example domain
    (every default-input dispatch probed https://staging.example.com five
    times, failed, and rolled back)."""

    def test_health_url_is_required_with_no_default(self):
        text = (WORKFLOWS / "canary-deploy.yml").read_text()
        block = re.search(r"health_url:\n(?:\s{8}.+\n)+", text).group(0)
        assert "required: true" in block, block
        assert "example.com" not in block, "placeholder default still present"

    def test_early_validation_rejects_example_domain(self):
        lines = (WORKFLOWS / "canary-deploy.yml").read_text().split("\n")
        for i, line in enumerate(lines):
            window = "\n".join(lines[i:i + 10])
            if ("HEALTH_URL" in line and "example.com" in window
                    and "exit 1" in window):
                return
        raise AssertionError("no fail-fast validation step rejecting an "
                             "example.com HEALTH_URL")


class TestF8_14_013_DryRunVar:
    """dry_run must be consumed or deleted — a GITHUB_ENV write nothing
    reads (and which cannot cross jobs anyway) is dead signalling."""

    def test_dry_run_written_only_if_read(self):
        text = (WORKFLOWS / "canary-deploy.yml").read_text()
        writes = len(re.findall(r'"?dry_run=', text))
        reads = len(re.findall(r"env\.dry_run|\$dry_run|\$\{dry_run\}", text))
        assert writes == 0 or reads > 0, f"{writes} writes, {reads} reads"


class TestF8_14_009_AutoSyncInterpolation:
    """workflow_run.name is repo-influenced text; it must reach the shell
    via env, not YAML interpolation inside a quoted assignment."""

    def test_workflow_run_name_not_interpolated_in_run(self):
        text = (WORKFLOWS / "auto-sync.yml").read_text()
        offenders = [
            i for i, line in enumerate(text.split("\n"), 1)
            if "${{ github.event.workflow_run.name }}" in line
            and "env:" not in line and not line.strip().startswith(("TRIGGER_WORKFLOW:",))
        ]
        assert offenders == [], offenders


class TestF8_14_010_MigrationDetection:
    """Migration detection must work on pull_request events (github.event.
    before is unset there) and must not swallow failures with || echo ''."""

    def test_no_silent_fallback_swallowing_diff_failures(self):
        text = (WORKFLOWS / "migration-check.yml").read_text()
        offenders = [
            i for i, line in enumerate(text.split("\n"), 1)
            if "git diff" in line and '|| echo ""' in line
        ]
        assert offenders == [], f"git diff failures swallowed at {offenders}"

    def test_pull_request_aware_diff_range(self):
        text = (WORKFLOWS / "migration-check.yml").read_text()
        assert re.search(r"pull_request\.base\.sha|base_ref", text), (
            "no PR-aware diff base"
        )


class TestF8_14_006_DenyByDefaultGuard:
    """The injection guard must deny by default: ANY event/inputs
    interpolation in a run/script block is flagged, regardless of variable
    name — a name-allowlist cannot catch code written after the allowlist."""

    @staticmethod
    def _guard():
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "injection_guard",
            REPO_ROOT / ".github" / "scripts" / "injection_guard.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_flags_names_outside_the_old_allowlist(self, tmp_path):
        wf = tmp_path / "x.yml"
        wf.write_text(
            "jobs:\n  j:\n    steps:\n      - run: |\n"
            '          TAG="${{ github.event.inputs.image_tag }}"\n'
        )
        assert len(self._guard().scan_file(wf)) == 1

    def test_opt_out_comment_is_honoured(self, tmp_path):
        wf = tmp_path / "x.yml"
        wf.write_text(
            "jobs:\n  j:\n    steps:\n      - run: |\n"
            "          # guard: allow-interpolation (reviewed: sha only)\n"
            '          BASE="${{ github.event.before }}"\n'
        )
        assert self._guard().scan_file(wf) == []

    def test_env_blocks_are_not_flagged(self, tmp_path):
        wf = tmp_path / "x.yml"
        wf.write_text(
            "jobs:\n  j:\n    steps:\n      - env:\n"
            "          TAG: ${{ github.event.inputs.image_tag }}\n"
            "        run: |\n"
            '          echo "$TAG"\n'
        )
        assert self._guard().scan_file(wf) == []

    def test_guard_workflow_invokes_the_scanner(self):
        text = (WORKFLOWS / "workflow-injection-guard.yml").read_text()
        assert "injection_guard.py" in text
        assert "AUTHOR|TITLE|BODY" not in text, "old name-allowlist regex remains"

    def test_repo_tree_has_no_unbaselined_findings(self):
        import subprocess, sys
        r = subprocess.run(
            [sys.executable, ".github/scripts/injection_guard.py",
             "--baseline", ".github/scripts/injection_guard_baseline.txt"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stdout + r.stderr
