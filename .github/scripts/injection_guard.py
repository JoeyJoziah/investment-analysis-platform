#!/usr/bin/env python3
"""Deny-by-default workflow shell-injection guard (F8-14-006, AT-G3-1.1).

Flags ANY ``${{ github.event.* }}`` or ``${{ inputs.* }}`` /
``${{ github.event.inputs.* }}`` expression appearing inside a ``run:`` or
``script:`` block of a workflow. The prior guard allowlisted nine variable
names, so any new name (TAG=, PCT=, REASON=) sailed through — a
name-allowlist cannot catch code written after the allowlist.

Opt-out for a reviewed line: put ``guard: allow-interpolation`` in a comment
on the same line or the line directly above.

Exit 0 = clean; exit 1 = findings (one per line on stdout).
"""

import re
import sys
from pathlib import Path

UNTRUSTED = re.compile(r"\$\{\{\s*(github\.event\.|inputs\.)")
BLOCK_KEY = re.compile(r"^(\s*)(?:-\s+)?(run|script):")
OPT_OUT = "guard: allow-interpolation"


def scan_file(path: Path):
    findings = []
    lines = path.read_text(errors="ignore").split("\n")
    in_block = False
    block_indent = 0
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        m = BLOCK_KEY.match(line)
        if m:
            in_block, block_indent = True, len(m.group(1))
            continue
        if in_block and stripped and indent <= block_indent:
            in_block = False
        if not in_block or not UNTRUSTED.search(line):
            continue
        if OPT_OUT in line or (i > 0 and OPT_OUT in lines[i - 1]):
            continue
        findings.append(f"{path}:{i + 1}: {stripped[:120]}")
    return findings


def _key(finding: str) -> str:
    """Baseline key: path + normalized line content (line numbers drift)."""
    path, _line, content = finding.split(":", 2)
    return f"{path}::{content.strip()}"


def main(argv):
    args = list(argv[1:])
    baseline_path = None
    if "--baseline" in args:
        baseline_path = Path(args[args.index("--baseline") + 1])
        del args[args.index("--baseline"):args.index("--baseline") + 2]

    roots = [Path(a) for a in args] or [Path(".github/workflows")]
    findings = []
    for root in roots:
        files = [root] if root.is_file() else sorted(root.rglob("*.yml")) + sorted(root.rglob("*.yaml"))
        for f in files:
            findings.extend(scan_file(f))

    # Differential gate (repo convention, cf. T2.4): legacy findings listed
    # in the baseline warn; anything NEW fails. Burn the baseline down to
    # empty — never add to it without review.
    baseline = set()
    if baseline_path and baseline_path.exists():
        baseline = {
            line.strip() for line in baseline_path.read_text().split("\n")
            if line.strip() and not line.startswith("#")
        }
    new = [f for f in findings if _key(f) not in baseline]
    legacy = len(findings) - len(new)
    if legacy:
        print(f"::notice::{legacy} baselined legacy interpolation(s) pending burn-down")
    for f in new:
        print(f)
    return 1 if new else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
