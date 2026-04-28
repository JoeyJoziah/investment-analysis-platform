#!/usr/bin/env python3
"""
Build dedupe-graph.json (mechanical pass) and cluster-map.yaml (initial assignment).
- Dedupe: collide on (file_line, category) + Jaccard ≥0.4 OR symbolic cross_scope.
- Clustering: A-F per EXECUTIVE_SUMMARY §4 + G residual; explicit ID anchors + keyword rules.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

OUT_DIR = Path("docs/audits/2026-04/_synthesis/_meta")
JSONL = OUT_DIR / "findings-master.jsonl"
DEDUPE = OUT_DIR / "dedupe-graph.json"
CMAP = OUT_DIR / "cluster-map.yaml"

# --- Cluster anchors (per EXECUTIVE_SUMMARY §4 + handoff §6) ---

# Findings explicitly named in EXECUTIVE_SUMMARY top-10 / clusters
ANCHOR_A = {  # secret-rotation
    "F-08-009", "F-07-001", "F-17-001", "F-05-003",
}
ANCHOR_B = {  # jwt-auth
    "F-01-001", "F-08-002", "F-08-004", "F-08-005", "F-01-009",
}
ANCHOR_C = {  # csp
    "F-08-003", "F-12-003",
}
ANCHOR_D = {  # random-data
    "F-02-003", "F-03-003",
}
ANCHOR_E = {  # test-exclusion
    "F-15-003",
}
ANCHOR_F = {  # frontend-backend-contract
    "F-12-001", "F-12-002", "F-01-003", "F-01-007",
}

# Keyword rules — apply in order; first match wins. Lowercased substring tests against
# (title + " " + description + " " + category + " " + recommendation).
KEYWORD_RULES = [
    # A: secret-rotation
    ("A", ["hardcoded password", "secret rotation", "credentials in", "credentials committed",
           "plaintext password", "redis_password", "es_password", "grafana", "fernet key",
           "jwt_secret_key=", ".env", "alembic.ini", "secret_key =", "admin@admin"]),
    # B: jwt-auth (all jwt/auth/token related, except CSP)
    ("B", ["jwt", "rs256", "hs256", "token revocation", "token blacklist", "refresh token",
           "auth.py", "login", "register endpoint", "current_user", "@secure_websocket",
           "authentication", "unauthenticated"]),
    # C: csp
    ("C", ["csp", "content-security-policy", "unsafe-inline", "unsafe-eval", "nonce"]),
    # D: random-data
    ("D", ["random.uniform", "dummylstm", "dummyxgboost", "dummyprophet", "dummy fallback",
           "random value", "fake data", "stub data", "random_data"]),
    # E: test-exclusion (scope 15 + tests-disabled findings everywhere)
    ("E", ["test exclusion", "test_exclusion", "tests/security/", "skip = ", "pytest.ini",
           "ignore = tests", "tests excluded", "mocked over", "broken test runner"]),
    # F: frontend-backend-contract
    ("F", ["/api/v1/", "/api/v2/", "v1deprecation", "deprecation middleware",
           "api versioning", "response field", "wrong field", "response_field",
           "api client", "api_client.ts", "frontend api"]),
]

CRITICAL_KEYWORDS_FOR_F = ["api contract", "frontend.*backend", "field name mismatch"]


def jaccard(a: str, b: str) -> float:
    ta = set(re.findall(r"\w+", a.lower()))
    tb = set(re.findall(r"\w+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def normalize_file_line(s: str) -> str:
    """Drop trailing line ranges so backend/auth.py:42-50 == backend/auth.py:42."""
    if not s:
        return ""
    m = re.match(r"^([^:]+):(\d+)", s)
    if m:
        return f"{m.group(1)}:{m.group(2)}"
    return s.strip()


def assign_cluster(f: dict, debug=False) -> tuple[str, str]:
    """Return (cluster_id, reason)."""
    fid = f["finding_id"]
    if fid in ANCHOR_A:
        return ("A", "anchor")
    if fid in ANCHOR_B:
        return ("B", "anchor")
    if fid in ANCHOR_C:
        return ("C", "anchor")
    if fid in ANCHOR_D:
        return ("D", "anchor")
    if fid in ANCHOR_E:
        return ("E", "anchor")
    if fid in ANCHOR_F:
        return ("F", "anchor")
    blob = " ".join([
        f.get("title", ""), f.get("description", ""),
        f.get("category", ""), f.get("recommendation", ""),
        f.get("file_line", ""),
    ]).lower()
    for cluster_id, kws in KEYWORD_RULES:
        for kw in kws:
            if kw in blob:
                return (cluster_id, f"keyword: {kw}")
    # Scope 15 catch-all → E if it talks about tests
    if f["scope_id"] == "15-test-suite":
        if "exclu" in blob or "skip" in blob or "ignore" in blob or "mocked" in blob:
            return ("E", "scope-15 + test-disable phrasing")
    return ("G", "residual")


def main():
    findings = [json.loads(l) for l in JSONL.read_text().splitlines() if l.strip()]
    assert len(findings) == 374, f"Expected 374, got {len(findings)}"

    # ------- Dedupe graph -------
    dedupe_edges = []
    by_key = defaultdict(list)  # (norm_file_line, category) -> [finding]
    for f in findings:
        key = (normalize_file_line(f["file_line"]), f["category"])
        if key[0] and key[1]:
            by_key[key].append(f)

    seen_pairs = set()
    for key, group in by_key.items():
        if len(group) < 2:
            continue
        # primary = lowest scope_id then lowest finding_id
        group_sorted = sorted(group, key=lambda x: (x["scope_id"], x["finding_id"]))
        primary = group_sorted[0]
        for dup in group_sorted[1:]:
            j = jaccard(primary["description"], dup["description"])
            cross_link = (dup["scope_id"] in primary["cross_scope"] or
                          primary["scope_id"] in dup["cross_scope"])
            if j >= 0.4 or cross_link:
                pair = tuple(sorted([primary["finding_id"], dup["finding_id"]]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                dedupe_edges.append({
                    "primary_finding_id": primary["finding_id"],
                    "duplicate_id": dup["finding_id"],
                    "key": f"{key[0]}|{key[1]}",
                    "jaccard": round(j, 3),
                    "cross_scope_link": cross_link,
                    "merge_action": "primary_owns_fix; duplicate referenced in workpaper §2",
                })

    # Cross-scope only collisions (different file_lines but explicit cross_scope linkage)
    by_id = {f["finding_id"]: f for f in findings}
    for f in findings:
        for tgt_scope in f.get("cross_scope", []):
            # find findings in target scope that mention same file_line root
            fl = normalize_file_line(f["file_line"])
            if not fl:
                continue
            for g in findings:
                if g["scope_id"] != tgt_scope:
                    continue
                if normalize_file_line(g["file_line"]) != fl:
                    continue
                if g["category"] != f["category"]:
                    continue
                pair = tuple(sorted([f["finding_id"], g["finding_id"]]))
                if pair in seen_pairs:
                    continue
                j = jaccard(f["description"], g["description"])
                if j < 0.3:
                    continue
                seen_pairs.add(pair)
                primary = min(f, g, key=lambda x: (x["scope_id"], x["finding_id"]))
                dup = g if primary is f else f
                dedupe_edges.append({
                    "primary_finding_id": primary["finding_id"],
                    "duplicate_id": dup["finding_id"],
                    "key": f"{fl}|{f['category']} (cross_scope)",
                    "jaccard": round(j, 3),
                    "cross_scope_link": True,
                    "merge_action": "primary_owns_fix; duplicate referenced in workpaper §2",
                })

    DEDUPE.write_text(json.dumps({
        "edges": dedupe_edges,
        "total_findings": len(findings),
        "edge_count": len(dedupe_edges),
        "primaries": sorted(set(e["primary_finding_id"] for e in dedupe_edges)),
        "duplicates_absorbed": sorted(set(e["duplicate_id"] for e in dedupe_edges)),
    }, indent=2))
    print(f"Dedupe: {len(dedupe_edges)} edges; {len(set(e['duplicate_id'] for e in dedupe_edges))} duplicates absorbed")

    # ------- Cluster assignment -------
    by_cluster = defaultdict(list)
    reasons = {}
    for f in findings:
        cid, reason = assign_cluster(f)
        by_cluster[cid].append(f["finding_id"])
        reasons[f["finding_id"]] = (cid, reason)

    counts = {k: len(v) for k, v in sorted(by_cluster.items())}
    print(f"Cluster counts: {counts}")
    print(f"Total assigned: {sum(counts.values())} (expected 374)")

    # Shard G into ≤50-finding worker slices by severity × scope-group.
    # Scope groups:
    SCOPE_GROUPS = {
        "backend": ("01-backend-api", "02-backend-services-domain", "11-backend-utils-shared"),
        "ml_data": ("03-ml-engine", "04-trading-agents", "05-data-ingestion-etl",
                    "06-airflow-pipelines", "09-analytics"),
        "storage_ops": ("07-database-persistence", "10-monitoring-observability"),
        "security_residual": ("08-auth-security-compliance",),
        "frontend_infra": ("12-frontend", "13-infra-deployment", "14-ci-cd-workflows"),
        "tests_config_scripts": ("15-test-suite", "16-config-secrets", "17-scripts-tooling"),
        "docs": ("18-docs-health",),
    }

    def group_of(scope_id):
        for grp, scopes in SCOPE_GROUPS.items():
            if scope_id in scopes:
                return grp
        return "other"

    g_findings = sorted(by_cluster["G"])
    g_shard = {}
    if len(g_findings) > 50:
        # Topic-balanced sharding (mixed severity within topic, split if >50)
        by_group = defaultdict(list)
        for fid in g_findings:
            by_group[group_of(by_id[fid]["scope_id"])].append(fid)

        SHARD_SPECS = [
            ("G1_backend", ["backend"]),
            ("G2_ml_data", ["ml_data"]),
            ("G3_frontend_infra", ["frontend_infra"]),
            ("G4_storage_security_residual", ["storage_ops", "security_residual"]),
            ("G5_tests_config_scripts", ["tests_config_scripts"]),
            ("G6_docs", ["docs"]),
        ]
        for shard_name, groups in SHARD_SPECS:
            ids = sorted([fid for grp in groups for fid in by_group.get(grp, [])])
            if len(ids) > 50:
                # Split: crit/high vs med/low to keep coherent
                ch = [fid for fid in ids if by_id[fid]["severity"] in ("critical", "high")]
                ml = [fid for fid in ids if by_id[fid]["severity"] in ("medium", "low")]
                if ch:
                    g_shard[f"{shard_name}_a_crit_high"] = sorted(ch)
                if ml:
                    g_shard[f"{shard_name}_b_med_low"] = sorted(ml)
            else:
                g_shard[shard_name] = ids
        print(f"G sharded into {len(g_shard)} workers: {[(k, len(v)) for k,v in g_shard.items()]}")
        # Sanity: total preserved
        total_g = sum(len(v) for v in g_shard.values())
        assert total_g == len(g_findings), f"G shard total {total_g} != G findings {len(g_findings)}"

    # Write cluster-map.yaml (manually formatted; small enough)
    lines = []
    lines.append("# Cluster Map — initial assignment for synthesis swarm")
    lines.append("# A-F: pre-identified clusters from EXECUTIVE_SUMMARY §4")
    lines.append("# G/G1/G2/G3: residual, sharded if >50 findings")
    lines.append(f"total_findings: {len(findings)}")
    lines.append("")
    lines.append("clusters:")
    descs = {
        "A": "secret-rotation",
        "B": "jwt-auth",
        "C": "csp",
        "D": "random-data",
        "E": "test-exclusion",
        "F": "frontend-backend-contract",
        "G": "residual-findings",
    }
    for cid in sorted(by_cluster.keys()):
        ids = sorted(by_cluster[cid])
        lines.append(f"  {cid}:")
        lines.append(f"    name: {descs.get(cid, 'unknown')}")
        lines.append(f"    count: {len(ids)}")
        lines.append(f"    findings:")
        for fid in ids:
            lines.append(f"      - {fid}  # {reasons[fid][1]}")
    if g_shard:
        lines.append("")
        lines.append("g_shards:")
        for shard_name, shard_ids in g_shard.items():
            lines.append(f"  {shard_name}:")
            lines.append(f"    count: {len(shard_ids)}")
            lines.append(f"    findings:")
            for fid in shard_ids:
                lines.append(f"      - {fid}  # severity={by_id[fid]['severity']} scope={by_id[fid]['scope_id']}")
    lines.append("")
    lines.append("# Cross-cluster dependencies (initial — workers may add more):")
    lines.append("dependencies:")
    lines.append("  B: [{depends_on: A, type: blocks, reason: 'JWT secret rotation must complete before JWT algo fix to avoid masked failure'}]")
    lines.append("  E: [{depends_on: null, type: independent, reason: 'un-excluding tests early surfaces signal for B/C/D'}]")
    lines.append("  C: [{depends_on: B, type: soft, reason: 'CSP rollout easier after auth stabilized'}]")
    CMAP.write_text("\n".join(lines))

    print(f"\nWrote: {DEDUPE}")
    print(f"Wrote: {CMAP}")
    if g_shard:
        print(f"\nG-shard summary: {dict((k, len(v)) for k,v in g_shard.items())}")


if __name__ == "__main__":
    main()
