# Synthesis Handoff

> **Read this first.** Then read `EXECUTIVE_SUMMARY.md`. Only after both, drill into per-scope reports.
> The synthesis swarm in a future session uses this as its single entrypoint.

## 1. Audit Phase Summary

- **Date completed:** 2026-04-27
- **Total scopes audited:** 18 / 18 (0 failed, 0 stub)
- **Total findings:** 374 (48 critical · 114 high · 143 medium · 69 low)
- **Total prior reports validated and archived:** 88
- **Source files modified during audit:** 0 (read-only contract held — verified post-run)
- **Plan reviewed by:** Skeptic, Constraint Guardian, User Advocate, Arbiter (APPROVED)

## 2. Pointer List (all reports)

See `../README.md` table. Frontmatter is YAML-parseable; finding tables are markdown-pipe parseable.

## 3. Known Gaps

None — every scope produced a complete report with `state: complete`. However:

- **Scope 08-auth-security** had a sub-agent tool limitation (no Write access); content was returned inline and persisted by the orchestrator. Report is complete and parseable.
- **Scope 18-docs-health** reported validating 23 priors but archived only 1; remaining 22 prior reports under that scope's mapping were not copied to the archive folder. Synthesis should treat any docs-scope prior outside the archive as `unverified`. (Other scopes' archives do exist for some of those same priors via cross-cutting mapping.)
- Several scopes sampled rather than fully read where files-in-scope exceeded their tier budget: scope 08 (12 of 20 backend/security files read), scope 02 (partial reads on largest service files), scope 17 (high-signal selection across ~110 scripts).

## 4. Agent-Failure Log

No agents failed. One agent (scope 08) had a tool-permission limitation; orchestrator persisted its output. No retries needed.

## 5. Sequencing Hints for Synthesis

**Order findings by:**

1. **Severity first** — all 48 criticals before any high.
2. **Then by cluster** (see EXECUTIVE_SUMMARY §4) — secret-rotation cluster, JWT/auth cluster, CSP cluster, random-data cluster, test-exclusion cluster, frontend↔backend contract cluster all must be planned as coordinated change-sets.
3. **Within a cluster, root-cause finding before downstream findings** — e.g. fix `JWT_SECRET_KEY` ephemeral fallback (F-08-002) before fixing the RS256 string-key issue (F-01-001), because the fallback masks the latter.
4. **Test-suite findings** (scope 15) often unblock work in other scopes — un-excluding `tests/security/` will produce immediate signal on the auth fixes; plan the un-exclusion early.
5. **Doc-health findings** (scope 18) are mostly low-risk cleanup; defer to the end of any rollout.

## 6. Consolidated Open Questions

Pulled from each report's §6 — synthesis or human owner must answer:

**Product/policy decisions (Loki cannot execute):**
- Should the platform return "no recommendation available" instead of random values from `DummyLSTM`/`random.uniform()`? (F-02-003, F-03-003)
- Is the platform actually offering "investment advice" under the SEC Investment Advisers Act? Advisor registration status remains "Not Addressed". (F-08-Q3)
- Is the user roles model single-role or multi-role? Affects DB migration scope. (F-08-Q2)

**Architecture decisions:**
- DDD orchestrator wire-or-delete: scope 02 found significant DDD scaffolding that's never invoked. Wire it or remove it? (F-02-005)
- 1234-LOC `recommendation_service` vs the two unused mixin files: keep the service, restore the mixins, or finish the in-progress split? (F-02-001)
- Single bundled `recommendation_engine` vs the broken `OptimizedRecommendationEngine` split: revert or fix forward? (F-09-002)

**Operational/legal:**
- After F-08-009 (credentials in repo) is fixed, who owns the `git filter-repo` purge of history? Coordinate with team's git remote and shared clones.
- Should redacted security archives at `_meta/prior-reports-archive/` move to an access-controlled location before the audit folder is committed/pushed? (F-08-Q4)

## 7. Known Limitations

- **Cross-scope linkage relies on per-agent self-reporting.** Each agent only saw its own scope; cross-links were tagged in each report's §4. Synthesis should run a graph dedupe pass on file:line collisions across reports as a safety net — at least one of the JWT findings appears in 3 different reports with overlapping evidence.
- **Sanitized/redacted content in archives is NOT in the audit reports.** Synthesis cannot recover redacted credential strings; rely on synthesis-handoff or the original priors (with caution) for rotation work.
- **Scope coverage is ~1,000 files.** Excluded: generated artifacts, dotfiles outside explicit scopes, `.git/`/`.swarm/`/`.claude-flow/` working directories, `node_modules`, `dist`, `build`, `.venv`, `__pycache__`.
- **No git history audit.** Findings are based on current HEAD only.
- **Agent token budgets were tier-bounded.** The 4 large-tier scopes (frontend, tests, scripts, docs) had finding caps of 600 each; if any scope produced exactly its cap, additional findings may exist beyond what's reported.

## 8. Re-Run Recipe

To redo a single scope (e.g., scope 11):

```bash
# Remove the existing artifacts
rm docs/audits/2026-04/reports/11-backend-utils-shared.md
rm docs/audits/2026-04/_meta/status/11-backend-utils-shared.json
# (optionally) rm relevant prior archives if you want to re-validate priors

# Then re-spawn one agent with the canonical prompt
# (Use the same prompt template that was used originally — see _meta/agent-prompt-skeleton.md)
```

## 9. Inputs the Synthesis Swarm Needs

Required:
- **This file** (`synthesis-handoff.md`)
- **`EXECUTIVE_SUMMARY.md`**
- **`README.md`** (completion table)
- **`_meta/aggregate.json`** (machine-readable totals + cross-link graph)
- **All 18 reports' YAML frontmatter + TL;DR + findings tables**

Optional (for drill-down):
- Full per-scope reports (drill into specific findings)
- Prior-report archives in `_meta/prior-reports-archive/` (validation evidence)
- The plan: `~/.claude/plans/why-can-t-we-chunk-eager-whale.md`

**Aggregate token estimate:** All 18 reports ≈ 480 KB / ~120K tokens. Fits in one context window. No compressed-pack fallback needed.

## 10. Output the Synthesis Swarm Produces

`docs/audits/2026-04/PRD-for-loki.md` — deduped, ordered, effort-estimated work breakdown ready for `/loki-mode`. Should:

- Group findings by cluster (see §5).
- For each cluster: problem statement, root cause, sequenced fix steps, files touched, acceptance test hint, rollback plan, dependencies on other clusters.
- Include explicit `loki_actionable: false` items (product decisions) at the top, awaiting human input.
- Estimate per-cluster effort and total program length.
- Identify the minimum-viable "production-stable cut" — which subset of fixes brings the platform out of red and yellow severity bands.
