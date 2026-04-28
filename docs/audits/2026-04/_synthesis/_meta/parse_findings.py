#!/usr/bin/env python3
"""
Parse all 18 audit reports' finding tables → findings-master.jsonl.
3-way reconciliation: row count vs frontmatter findings_summary vs aggregate.json.
"""
import json
import re
import sys
from pathlib import Path

REPORTS_DIR = Path("docs/audits/2026-04/reports")
META_DIR = Path("docs/audits/2026-04/_meta")
OUT_DIR = Path("docs/audits/2026-04/_synthesis/_meta")
OUT_JSONL = OUT_DIR / "findings-master.jsonl"
ANOMALIES = OUT_DIR / "parse-anomalies.md"

EXPECTED_COLUMNS = [
    "id", "severity", "category", "file_line",
    "title", "description", "recommendation",
    "acceptance_test_hint", "effort_hours",
    "loki_actionable", "cross_scope"
]

FINDING_ID_RE = re.compile(r"^\|\s*F-(\d{2})-\d{3}\s*\|")


def parse_frontmatter(text: str) -> dict:
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    fm = text[3:end]
    out = {}
    # crude YAML extract for findings_summary
    m = re.search(r"findings_summary:\s*\n((?:\s+\w+:\s*\d+\n)+)", fm)
    if m:
        for line in m.group(1).splitlines():
            mm = re.match(r"\s+(\w+):\s*(\d+)", line)
            if mm:
                out[mm.group(1)] = int(mm.group(2))
    sid = re.search(r'scope_id:\s*"([^"]+)"', fm)
    if sid:
        out["scope_id"] = sid.group(1)
    return out


def split_pipe_row(line: str) -> list:
    # strip leading/trailing pipes, split on | not inside backticks
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    # naive split — pipes inside code spans are rare in these tables; verify post-split
    parts = [c.strip() for c in line.split("|")]
    return parts


def parse_cross_scope(s: str) -> list:
    s = s.strip()
    if not s or s in ("[]", "-"):
        return []
    # JSON-ish: ["a", "b"]
    try:
        return json.loads(s.replace("'", '"'))
    except Exception:
        # fallback: split on comma
        cleaned = s.strip("[]").replace('"', "").replace("'", "")
        return [x.strip() for x in cleaned.split(",") if x.strip()]


def parse_loki_actionable(s: str) -> bool:
    return s.strip().lower() in ("true", "yes", "✓", "y")


def parse_effort(s: str) -> float:
    s = s.strip()
    # could be "3", "0.5", "2-4", "~2"
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if m:
        return float(m.group(1))
    return 0.0


def parse_report(path: Path) -> tuple[list, dict, list]:
    """Return (findings, frontmatter_summary, anomalies)."""
    text = path.read_text()
    fm = parse_frontmatter(text)
    findings = []
    anomalies = []
    for i, line in enumerate(text.splitlines(), 1):
        if FINDING_ID_RE.match(line):
            parts = split_pipe_row(line)
            # Schema: id | sev | cat | file_line | title | desc | rec | acc | effort | loki | cross  (11 cols)
            # Some rows have 10 cols (missing file_line) or >11 cols (escaped pipes in text fields).
            # First 3 (id/sev/cat) and last 3 (effort/loki/cross) are reliable; reconstruct middle.
            if len(parts) == 10:
                # Heuristic: missing file_line — insert empty at idx 3
                parts = parts[:3] + [""] + parts[3:]
                anomalies.append(
                    f"{path.name}:{i} 10 cols → assumed missing file_line at idx 3"
                )
            if len(parts) < 11:
                anomalies.append(
                    f"{path.name}:{i} unrecoverable: {len(parts)} cols: {line[:120]}"
                )
                continue
            if len(parts) > 11:
                # Pipes inside title/desc/rec/acc fields (e.g. escaped \| in regex). Collapse middle.
                head = parts[:4]                 # id, sev, cat, file_line
                tail = parts[-3:]                # effort, loki, cross_scope
                middle = parts[4:-3]             # 4-N cols that should be 4 (title/desc/rec/acc)
                if len(middle) == 4:
                    parts = head + middle + tail
                else:
                    # Too many middle cols — preserve as 4 fields by collapsing last middle cols into acceptance
                    title = middle[0]
                    desc = middle[1] if len(middle) > 1 else ""
                    rec = middle[2] if len(middle) > 2 else ""
                    acc = " | ".join(middle[3:]) if len(middle) > 3 else ""
                    parts = head + [title, desc, rec, acc] + tail
                anomalies.append(
                    f"{path.name}:{i} {len(parts) if len(parts)!=11 else 'recovered to 11'} cols (escaped pipes in middle fields)"
                )
            f_id, severity, category, file_line, title, desc, rec, acc, effort, loki, cross = parts[:11]
            findings.append({
                "finding_id": f_id,
                "scope_id": fm.get("scope_id", path.stem),
                "severity": severity.lower(),
                "category": category.lower(),
                "file_line": file_line,
                "title": title,
                "description": desc,
                "recommendation": rec,
                "acceptance_test_hint": acc,
                "effort_hours": parse_effort(effort),
                "loki_actionable": parse_loki_actionable(loki),
                "cross_scope": parse_cross_scope(cross),
                "source_report": path.name,
                "source_line": i,
            })
    return findings, fm, anomalies


def main():
    aggregate = json.loads((META_DIR / "aggregate.json").read_text())
    by_scope_aggregate = aggregate["by_scope"]
    expected_total = aggregate["totals"]["total"]

    all_findings = []
    all_anomalies = []
    per_scope_counts = {}
    fm_totals = {}

    reports = sorted(REPORTS_DIR.glob("*.md"))
    for r in reports:
        findings, fm, anomalies = parse_report(r)
        scope_id = fm.get("scope_id", r.stem)
        per_scope_counts[scope_id] = len(findings)
        fm_totals[scope_id] = fm.get("total", -1)
        all_findings.extend(findings)
        all_anomalies.extend(anomalies)

    # Write JSONL
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w") as f:
        for fn in all_findings:
            f.write(json.dumps(fn, ensure_ascii=False) + "\n")

    # 3-way reconcile
    print(f"Total findings parsed: {len(all_findings)} (expected {expected_total})")
    mismatches = []
    for scope_id, parsed_n in per_scope_counts.items():
        fm_n = fm_totals.get(scope_id, -1)
        agg_n = by_scope_aggregate.get(scope_id, {}).get("total", -1)
        ok = parsed_n == fm_n == agg_n
        flag = "OK" if ok else "MISMATCH"
        if not ok:
            mismatches.append((scope_id, parsed_n, fm_n, agg_n))
        print(f"  {scope_id}: parsed={parsed_n} frontmatter={fm_n} aggregate={agg_n} {flag}")

    # Anomalies report
    if all_anomalies or mismatches:
        with ANOMALIES.open("w") as f:
            f.write("# Parse Anomalies\n\n")
            if mismatches:
                f.write("## Count Mismatches\n\n")
                for s, p, fm_n, agg_n in mismatches:
                    f.write(f"- {s}: parsed={p} frontmatter={fm_n} aggregate={agg_n}\n")
                f.write("\n")
            if all_anomalies:
                f.write("## Row-Parse Warnings\n\n")
                for a in all_anomalies:
                    f.write(f"- {a}\n")
        print(f"\nAnomalies written: {ANOMALIES}")

    if len(all_findings) != expected_total or mismatches:
        print("\nFAILED 3-way reconciliation")
        sys.exit(1)
    print("\n3-way reconciliation PASSED")


if __name__ == "__main__":
    main()
