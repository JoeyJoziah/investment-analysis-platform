#!/usr/bin/env python3
"""Split findings-master.jsonl into per-cluster slice files."""
import json
import re
from collections import defaultdict
from pathlib import Path

OUT = Path("docs/audits/2026-04/_synthesis/_meta")
SLICES = OUT / "slices"
SLICES.mkdir(exist_ok=True)

findings = [json.loads(l) for l in (OUT / "findings-master.jsonl").read_text().splitlines() if l.strip()]
by_id = {f["finding_id"]: f for f in findings}

cmap = (OUT / "cluster-map.yaml").read_text()

# Parse clusters and g_shards out of YAML (cheap, format-controlled)
clusters = defaultdict(list)
current = None
in_findings = False
in_g = False
for line in cmap.splitlines():
    s = line.rstrip()
    if re.match(r"^clusters:", s):
        in_findings = False
        in_g = False
        continue
    if re.match(r"^g_shards:", s):
        in_g = True
        continue
    if re.match(r"^dependencies:", s):
        in_g = False
        in_findings = False
        current = None
        continue
    m = re.match(r"^  ([A-Z]\d*[a-z_]*\d*[a-z_]*):$", s)
    if m and (not in_g):
        current = m.group(1)
        continue
    m = re.match(r"^  (G\d+\w*):$", s)
    if m and in_g:
        current = m.group(1)
        continue
    m = re.match(r"^      - (F-\d{2}-\d{3})", s)
    if m and current:
        clusters[current].append(m.group(1))

# Build slice files. For G we use the shard names (G1_..., G2_...).
# For A-F we drop the cluster G itself (it's been sharded).
out_clusters = {}
for cid, ids in clusters.items():
    if cid == "G":
        continue  # superseded by shards
    out_clusters[cid] = sorted(set(ids))

for cid, ids in out_clusters.items():
    slice_path = SLICES / f"{cid}.jsonl"
    with slice_path.open("w") as f:
        for fid in ids:
            if fid in by_id:
                f.write(json.dumps(by_id[fid], ensure_ascii=False) + "\n")
    # Report unique source scopes
    scopes = sorted(set(by_id[fid]["scope_id"] for fid in ids if fid in by_id))
    print(f"{cid}: {len(ids)} findings | scopes: {', '.join(scopes)}")

# Coverage assertion
all_assigned = sorted(set(fid for ids in out_clusters.values() for fid in ids))
all_findings = sorted(by_id.keys())
missing = set(all_findings) - set(all_assigned)
extra = set(all_assigned) - set(all_findings)
print(f"\nTotal assigned: {len(all_assigned)} / {len(all_findings)}")
if missing:
    print(f"MISSING from cluster slices: {sorted(missing)}")
if extra:
    print(f"EXTRA: {sorted(extra)}")
assert not missing and not extra
print("Coverage: ALL FINDINGS ASSIGNED")
