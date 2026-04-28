# G6_docs — Documentation Health Cluster Workpaper

**Cluster:** G6_docs
**Scope source:** 18-docs-health
**Findings:** 37 (all from scope 18-docs-health)
**Synthesis date:** 2026-04-27
**Program position:** **DEFERRED — runs LAST in the rollout**

---

## 1. Cluster overview

This cluster aggregates all 37 documentation-health findings produced by the
audit. None of them touch source code, runtime behavior, security boundaries, or
data correctness. They describe drift between docs and reality — stale "Last
Updated" dates, un-archived phase reports, broken intra-doc references, missing
READMEs in `docs/<subdir>/`, an un-wired `.github/workflows/doc-health.yml`,
plan/checklist files whose recommendations were never executed, and the absence
of a documentation policy / changelog / per-role quick-start.

Per the synthesis-handoff guidance (§5: "Doc-health findings are mostly
low-risk cleanup; defer to the end of any rollout"), this entire cluster is
sequenced **after** clusters B/C/D (security, data, infra) so that:

1. The APIs, schemas, and deployment surfaces being documented have stabilized.
2. Doc rewrites do not have to be re-done after substantive code changes.
3. The cleanup happens once, against the post-remediation reality, not the
   pre-remediation one.

Severity distribution: **3 high** (F-18-001 stale architecture codemap, F-18-002
root-level clutter / 28 deletion-pending files, F-18-003 wave reports needing
archive, F-18-004 missing CI doc-health workflow), **6 medium**, **27 low**.
Even the "high" items are reputational/hygienic, not functional. No security
or data risk; no production impact.

The work is bulk-cleanup: file moves, header refreshes, README scaffolding,
one CI workflow file, and a few policy docs. It is largely Loki-actionable in
a single sweep and has near-zero blast radius.

---

## 2. Member findings (all 37 IDs)

| ID | Severity | Title | Sub-theme |
|---|---|---|---|
| F-18-001 | high | Stale architecture codemap (`docs/CODEMAPS/README.md`) | stale dates / status |
| F-18-002 | high | Root-level doc clutter; 28 files in DELETION_LOG never deleted | archive misses |
| F-18-003 | high | Wave/phase reports clutter `docs/` root | archive misses |
| F-18-004 | high | `.github/workflows/doc-health.yml` missing | CI integration |
| F-18-006 | medium | `check-doc-health.sh` / `validate_doc_health.py` never run in CI | CI integration |
| F-18-007 | medium | Linting rules reference outdated tools (no markdownlint) | CI integration |
| F-18-008 | medium | `DOCUMENTATION_INDEX.md` missing 50+ recent additions | missing READMEs / index |
| F-18-009 | medium | No deprecated/archive section in index | archive misses |
| F-18-010 | medium | Doc-health metrics schema not aligned with automation | CI integration |
| F-18-011 | medium | Monitoring tools (markdown-link-check, doctoc) not installed | CI integration |
| F-18-012 | low  | `DOCUMENTATION_SYNC_COMPLETE.md` claims cleanup done (it isn't) | stale dates / status |
| F-18-013 | low  | `DOC_HEALTH_QUICKSTART.md` references non-existent CI | broken links |
| F-18-014 | low  | `PROJECT_ROADMAP.md` phases outdated, no date | stale dates / status |
| F-18-015 | low  | `REFACTOR_AUDIT_SUMMARY.md` missing date | stale dates / status |
| F-18-016 | low  | `INSTALLATION_GUIDE.md` references `setup-dev.sh` (wrong) | broken links |
| F-18-017 | low  | `IMPLEMENTATION_ROADMAP.md` no live update log | stale dates / status |
| F-18-018 | low  | No automated TOC generation (doctoc) | CI integration |
| F-18-019 | low  | `docs/testing/TEST_DOCUMENTATION_INDEX.md` incomplete | missing READMEs / index |
| F-18-020 | low  | No `docs/security/README.md` (25 files, no index) | missing READMEs / index |
| F-18-021 | low  | No per-role quick-start guide | missing READMEs / index |
| F-18-022 | low  | `DOCUMENTATION_CHECKLIST_2026_01_29.md` stale (88 days) | stale dates / status |
| F-18-023 | low  | `CLAUDE_FOLDER_COMPREHENSIVE_REVIEW.md` outdated (88 days) | stale dates / status |
| F-18-024 | low  | `CLAUDE_FLOW_V3_VERSION_ALIGNMENT.md` version may be stale | stale dates / status |
| F-18-025 | low  | `NOTION_TASK_MAPPING.md` references unimplemented sync | broken links |
| F-18-026 | low  | No `docs/architecture/README.md` (9 files, no guide) | missing READMEs / index |
| F-18-027 | low  | `docs/reports/` 24 files, inconsistent naming, no manifest | missing READMEs / index |
| F-18-028 | low  | No documentation deprecation/archive policy | policy/governance |
| F-18-029 | low  | Code-example version specs (Python/Node) not tracked | CI integration |
| F-18-030 | low  | `IMPLEMENTATION_STATUS.md` no Last Updated date | stale dates / status |
| F-18-031 | low  | `IMPLEMENTATION_TRACKER.md` claims 97% (90 days old) | stale dates / status |
| F-18-032 | low  | `REFACTOR_AUDIT_SUMMARY.md` cleanup recs not executed | stale dates / status |
| F-18-033 | low  | `reports/SYNCHRONIZATION_QUALITY_REPORT.md` metrics outdated | stale dates / status |
| F-18-034 | low  | `CONTRIB.md` doc-contribution process incomplete | policy/governance |
| F-18-035 | low  | `architecture/DOCUMENTATION_REORGANIZATION_PLAN.md` not executed | stale dates / status |
| F-18-036 | low  | `GITHUB_ISSUES_BACKLOG.md` stale vs. real GitHub state | stale dates / status |
| F-18-037 | low  | `TROUBLESHOOTING.md` lacks update procedure / template | policy/governance |
| F-18-038 | low  | No `docs/CHANGELOG.md` for documentation updates | policy/governance |

Total: **37** findings, all referenced.

---

## 3. Sequenced fix steps (grouped by sub-theme)

The work is bulk-batched into 7 ordered steps. Each step groups similar
findings so a single Loki sweep can knock several out at once.

**Step 1 — Archive bulk cleanup (high-leverage)**
Group: F-18-002, F-18-003, F-18-022, F-18-023, F-18-035.
Create `docs/archive/`. Execute the deletion list from `docs/DELETION_LOG.md`
(28 root files). Move all `wave*`, `PHASE*`, `*SUMMARY.md`, the
2026-01-29-dated checklist, the CLAUDE_FOLDER review, and the unimplemented
reorganization plan into `docs/archive/`. One commit, atomic move.

**Step 2 — Stale-date / "Last Updated" header refresh sweep**
Group: F-18-001, F-18-014, F-18-015, F-18-017, F-18-024, F-18-030, F-18-031,
F-18-033, F-18-036.
Add or update `**Last Updated:**` and `**Status:**` headers across the listed
files. For F-18-031 and F-18-001, reconcile conflicting "% complete" /
remaining-work claims and pick `docs/CODEMAPS/README.md` as source of truth;
update or archive the others.

**Step 3 — Broken intra-doc references**
Group: F-18-013, F-18-016, F-18-025, F-18-012, F-18-032.
Fix `INSTALLATION_GUIDE.md` (`setup-dev.sh` → `setup.sh`), strip CI section
from `DOC_HEALTH_QUICKSTART.md` until the workflow exists, correct
`DOCUMENTATION_SYNC_COMPLETE.md` and `REFACTOR_AUDIT_SUMMARY.md` claims of
"cleanup complete," and reconcile `NOTION_TASK_MAPPING.md` against the actual
Notion sync workflow status.

**Step 4 — Missing per-directory READMEs and index expansion**
Group: F-18-008, F-18-019, F-18-020, F-18-026, F-18-027, F-18-021.
Generate `docs/security/README.md` (25 files), `docs/architecture/README.md`
(9 files), `docs/reports/README.md` (24 files, with consistent renaming),
expand `docs/testing/TEST_DOCUMENTATION_INDEX.md` to all 13 files, rebuild
the top-level `DOCUMENTATION_INDEX.md` to enumerate all 147 files by
category, and add `docs/QUICKSTART_BY_ROLE.md`.

**Step 5 — Deprecation / archive surfaces in index**
Group: F-18-009.
Add a "Deprecated" section to `DOCUMENTATION_INDEX.md` listing archived
items moved in Step 1; add a deprecation banner snippet to each archived doc.

**Step 6 — Doc-health CI workflow + tooling**
Group: F-18-004, F-18-006, F-18-007, F-18-010, F-18-011, F-18-018, F-18-029.
Create `.github/workflows/doc-health.yml` per
`docs/DOCUMENTATION_HEALTH.md §2.3`. Wire `markdownlint`, `markdown-link-check`,
`doctoc`, and the existing `scripts/check-doc-health.sh` /
`scripts/validate_doc_health.py`. Define `.reports/doc-health/schema.json`
and validate per-run report output. Replace the Flesch-Kincaid rule with
markdownlint config. Add Python/Node version specs to code blocks and a CI
job that runs the examples against minimum versions.
**Cross-scope dependency:** scope 14-ci-cd-workflows.

**Step 7 — Documentation governance**
Group: F-18-028, F-18-034, F-18-037, F-18-038.
Add `docs/DOCUMENTATION_POLICY.md` (deletion criteria, deprecation format,
archive structure, review cadence). Extend `CONTRIB.md` with a "Documentation
Contribution Process" section linking to the policy. Add an "Updating This
Guide" header + `docs/TROUBLESHOOTING_TEMPLATE.md`. Create
`docs/CHANGELOG.md`, optionally auto-populated from `git log --name-status
docs/**`.

---

## 4. Files touched

Created:
- `docs/archive/` (new directory)
- `docs/architecture/README.md`
- `docs/security/README.md`
- `docs/reports/README.md`
- `docs/QUICKSTART_BY_ROLE.md`
- `docs/DOCUMENTATION_POLICY.md`
- `docs/TROUBLESHOOTING_TEMPLATE.md`
- `docs/CHANGELOG.md`
- `.github/workflows/doc-health.yml`
- `.reports/doc-health/schema.json`
- `.example-versions.json`

Modified:
- `docs/CODEMAPS/README.md`
- `docs/DOCUMENTATION_INDEX.md`
- `docs/DOCUMENTATION_HEALTH.md`
- `docs/DOCUMENTATION_SYNC_COMPLETE.md`
- `docs/DOC_HEALTH_QUICKSTART.md`
- `docs/PROJECT_ROADMAP.md`
- `docs/REFACTOR_AUDIT_SUMMARY.md`
- `docs/INSTALLATION_GUIDE.md`
- `docs/IMPLEMENTATION_ROADMAP.md`
- `docs/IMPLEMENTATION_STATUS.md`
- `docs/IMPLEMENTATION_TRACKER.md`
- `docs/CLAUDE_FLOW_V3_VERSION_ALIGNMENT.md`
- `docs/NOTION_TASK_MAPPING.md`
- `docs/CONTRIB.md`
- `docs/TROUBLESHOOTING.md`
- `docs/testing/TEST_DOCUMENTATION_INDEX.md`
- `docs/reports/SYNCHRONIZATION_QUALITY_REPORT.md`
- `docs/GITHUB_ISSUES_BACKLOG.md`

Moved into `docs/archive/`:
- `docs/wave4_completion_report.md`
- `docs/wave4.5_completion_report.md`
- `docs/wave5_progress_report.md`
- `docs/wave5_session_complete.md`
- `docs/WAVE5_FINAL_SUMMARY.md`
- `docs/PHASE_0_3_SUMMARY.md`, `docs/PHASE_3_2_IMPLEMENTATION_SUMMARY.md`,
  `docs/PHASE3_IMPLEMENTATION_COMPLETE.md`, plus all other `PHASE*` and
  `*SUMMARY.md` files in `docs/` root
- `docs/DOCUMENTATION_CHECKLIST_2026_01_29.md`
- `docs/CLAUDE_FOLDER_COMPREHENSIVE_REVIEW.md`
- `docs/architecture/DOCUMENTATION_REORGANIZATION_PLAN.md` (if not implemented)
- 28 files enumerated in `docs/DELETION_LOG.md` Category 1 (deletions)

Touched scripts (test-only, no edits unless bitrot found):
- `scripts/check-doc-health.sh`
- `scripts/validate_doc_health.py`

---

## 5. Acceptance tests

Run after each step; all must pass before the cluster is signed off.

```bash
# Step 1 (archive cleanup)
test "$(ls docs/*.md | wc -l)" -lt 8                     # root reduced
test -d docs/archive                                     # archive exists
find docs -maxdepth 1 -name 'wave*' -o -name 'PHASE*' -o -name '*SUMMARY.md' | wc -l | grep -q '^0$'

# Step 2 (date refresh)
! grep -rl 'Last [Uu]pdated: 2026-01' docs/ --include='*.md'
! grep -rl 'Last [Uu]pdated: 2026-02' docs/ --include='*.md'
grep -q 'Last Updated' docs/CODEMAPS/README.md

# Step 3 (broken link sweep)
npx markdown-link-check docs/INSTALLATION_GUIDE.md       # 0 dead
npx markdown-link-check docs/DOC_HEALTH_QUICKSTART.md
npx markdown-link-check docs/NOTION_TASK_MAPPING.md
test -f setup.sh && grep -q 'setup.sh' docs/INSTALLATION_GUIDE.md

# Step 4 (per-dir READMEs)
test -f docs/security/README.md
test -f docs/architecture/README.md
test -f docs/reports/README.md
test -f docs/QUICKSTART_BY_ROLE.md
test "$(grep -c '^#' docs/DOCUMENTATION_INDEX.md)" -ge 50

# Step 5 (deprecated section)
grep -q -i '## .*Deprecated' docs/DOCUMENTATION_INDEX.md

# Step 6 (CI workflow)
test -f .github/workflows/doc-health.yml
test -f .reports/doc-health/schema.json
yamllint .github/workflows/doc-health.yml
bash scripts/check-doc-health.sh --report
python scripts/validate_doc_health.py --output summary

# Step 7 (governance)
test -f docs/DOCUMENTATION_POLICY.md
test -f docs/CHANGELOG.md
test -f docs/TROUBLESHOOTING_TEMPLATE.md
grep -q -i 'documentation contribution' docs/CONTRIB.md

# Whole-cluster sweep
npx markdown-link-check 'docs/**/*.md'                   # 0 dead links overall
```

---

## 6. Rollback plan

Trivial. Every change is doc-only; no runtime, schema, or build effect.

- All step commits are pure-doc commits (plus one workflow file in Step 6).
- Rollback for any step = `git revert <commit-sha>`.
- The archive moves are reversible (`git mv` history preserved); revert
  restores prior locations.
- The new CI workflow can be disabled by deleting
  `.github/workflows/doc-health.yml` (no shared infra coupling).
- No data, no users, no API consumers affected.

---

## 7. Dependencies

**This cluster runs LAST in the program. It is deferred to the end of the
rollout.** Rationale (per synthesis handoff §5):

- Doc updates cite API surfaces, schemas, deployment topology, and security
  controls owned by clusters B, C, and D. Documenting against in-flux APIs
  produces churn and a second cleanup pass.
- Stable inputs required from upstream clusters before G6 begins:
  - **B (security):** final list of security docs / `docs/security/`
    contents, post-remediation auth/perms behavior.
  - **C (data / schemas):** finalized DB schemas, migration log, any
    deprecated tables — fed into `docs/architecture/README.md`.
  - **D (infra / deployment / CI):** stabilized deployment runbook, finalized
    workflow filenames so `docs/DEPLOYMENT.md` and the new
    `.github/workflows/doc-health.yml` reference real siblings (cross-scope
    `14-ci-cd-workflows` for F-18-004, F-18-006, F-18-007, F-18-010,
    F-18-018, F-18-029, F-18-038).
- No internal cluster ordering risk: Steps 1→7 above are independent enough
  that they can run in parallel by sub-theme if extra hands are available,
  but the listed order minimizes merge conflicts (archive moves first so the
  index rebuild in Step 4 sees the final file set).

**Scheduling note:** Begin G6 only after B/C/D acceptance gates have passed
on `main`.

---

## 8. Effort & cost

Bulk doc work, mechanical and parallelizable.

| Sub-theme | Findings | Hours (sum of `effort_hours`) |
|---|---|---|
| Archive bulk cleanup (Step 1) | 5 | ~10h (3+3+1+1+2) |
| Stale-date sweep (Step 2) | 9 | ~16h |
| Broken refs (Step 3) | 5 | ~6h |
| Per-dir READMEs / index (Step 4) | 6 | ~18h |
| Deprecated section (Step 5) | 1 | ~4h |
| CI workflow + tooling (Step 6) | 7 | ~28h |
| Governance (Step 7) | 4 | ~10h |
| **Total** | **37** | **~92h raw, ~25–30h after batching** |

Raw sum across the slice's `effort_hours` is ~92 hours, but most of these are
1–4-hour items that share boilerplate (header inserts, README scaffolds);
batching brings the realistic effort to **~20–30 engineering hours** plus a
half-day of CI iteration.

**Loki cost estimate:** ~$2–3 in model time. The work is overwhelmingly
mechanical edits (header inserts, file moves, index regeneration), with one
small CI workflow authored from a spec that already exists in
`DOCUMENTATION_HEALTH.md §2.3`. No deep reasoning passes, no large
multi-file refactors.

---

## 9. Loki-actionable

**Entirely yes** — every finding in the slice carries `loki_actionable: true`,
and nothing in the recommended sequence is destructive in any irreversible
sense.

- Step 1's deletions are pre-vetted by `docs/DELETION_LOG.md` (audit dated
  2026-02-08, "SAFE TO DELETE"); we additionally archive instead of
  hard-delete the wave/phase reports so no information is lost.
- Steps 2–5 and 7 are pure file edits / new-file creation under `docs/`.
- Step 6 adds one new GitHub Actions workflow (additive; failure of the
  workflow does not block PRs unless we explicitly set the gate, which is
  itself reversible by editing the workflow).
- No code paths touched. No tests need to change. No production migration.
- No secrets, no PII, no data movement.

A single Loki agent with Edit/Write/Bash tools can execute Steps 1–5 and 7
end-to-end. Step 6 benefits from a human reviewing the workflow once before
merge.

---

## 10. Risks

Minimal. Doc-only cluster. Enumerated for completeness:

1. **Inbound link breakage** — moving files into `docs/archive/` may break
   external links (other repos, Notion, blog posts). Mitigation: leave
   redirect stubs at original paths for 30 days, listing the new archive
   location.
2. **Index regeneration drift** — re-listing all 147 files in
   `DOCUMENTATION_INDEX.md` may miss future additions. Mitigation: Step 6's
   CI job runs `find docs -name '*.md'` and warns if any file is missing
   from the index.
3. **CI workflow false-positives** — `markdown-link-check` against external
   URLs flakes on rate limits. Mitigation: configure the workflow with
   retries and a `--config` allow-list for known-flaky domains; set the job
   to soft-fail until trust is established.
4. **Conflicting source-of-truth** — F-18-031 vs. F-18-001 disagree on
   completion %. Mitigation: pick `docs/CODEMAPS/README.md` as canonical;
   archive `IMPLEMENTATION_TRACKER.md` per Step 2.
5. **Premature execution** — running G6 before B/C/D stabilize would force a
   second pass. Mitigation: hard gate on B/C/D acceptance per §7.

No security risk. No data risk. No availability risk. No regression risk.

---

**Signed off:** all 37 findings (F-18-001, F-18-002, F-18-003, F-18-004,
F-18-006, F-18-007, F-18-008, F-18-009, F-18-010, F-18-011, F-18-012,
F-18-013, F-18-014, F-18-015, F-18-016, F-18-017, F-18-018, F-18-019,
F-18-020, F-18-021, F-18-022, F-18-023, F-18-024, F-18-025, F-18-026,
F-18-027, F-18-028, F-18-029, F-18-030, F-18-031, F-18-032, F-18-033,
F-18-034, F-18-035, F-18-036, F-18-037, F-18-038) — covered. Total: 37/37.
