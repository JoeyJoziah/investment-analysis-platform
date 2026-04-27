---
scope_id: "NN-scope-name"
scope_name: "Human Readable Scope"
agent_type: "specialist-agent-type-or-general-purpose"
date: "2026-04-27"
files_in_scope: 0
files_reviewed: 0
files_skipped: []
prior_reports_validated:
  - path: "docs/EXAMPLE.md"
    status: "current | partially_stale | fully_stale | unverifiable"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/EXAMPLE.archived.md"
    claims_validated: 0
    claims_still_valid: 0
    claims_stale: 0
findings_summary:
  critical: 0
  high: 0
  medium: 0
  low: 0
  total: 0
estimated_remediation_effort_days: 0
agent_status: "complete | failed | partial"
agent_token_usage: 0
---

# [Scope Name] — Audit Report

## TL;DR (REQUIRED — exactly 5 bullets, max)

- Most important finding
- Second
- Third
- Fourth
- Fifth

> Read these 5 before anything else in this report.

## 1. Scope & Files Reviewed

- Path globs covered (list)
- Files explicitly excluded with reason

## 2. Prior Report Reconciliation

For EACH prior report mapped to this scope:

### `path/to/PRIOR.md` — status: `partially_stale`

**Validation method:** how each claim was verified (cite the grep command, file:line read, test run output).

**Archived to:** `docs/audits/2026-04/_meta/prior-reports-archive/PRIOR.archived.md`

**Per-claim validation table** (every row REQUIRED to have evidence > 20 chars):

| # | Claim (verbatim quote or paraphrase) | Original Location | Status | Evidence |
|---|---|---|---|---|
| 1 | "X is missing from Y" | PRIOR.md §3 | fully_stale | `grep -rn "X" backend/Y` returns 8 hits at backend/Y/foo.py:42, etc. |
| 2 | ... | ... | ... | ... |

## 3. Findings (every row REQUIRED to be Loki-actionable or flagged false)

| ID | Severity | Category | File:Line | Title | Description | Recommendation | Acceptance Test Hint | Effort (h) | Loki Actionable | Cross Scope |
|---|---|---|---|---|---|---|---|---|---|---|
| F-NN-001 | critical | security | backend/api/auth.py:42 | Title | What's wrong | What to do | `pytest tests/auth/test_x.py::test_y` passes | 4 | true | [] |

**Categories:** `bug`, `incomplete_code`, `stale_code`, `dead_code`, `broken_dependency`, `broken_import`, `schema_mismatch`, `code_quality`, `performance`, `security`, `architecture`, `testing_gap`, `doc_drift`, `better_pattern`

**Severities:** `critical` (data loss / security breach / outage risk), `high` (degrades correctness or perf measurably), `medium` (quality / maintainability), `low` (cosmetic / nit)

**Loki Actionable:** `false` only when the finding requires human judgement (product decision, third-party action). Still document but the synthesis swarm will triage.

**Cross Scope:** list of scope IDs this finding touches, e.g. `["11-backend-utils", "08-auth-security"]`. Empty list `[]` if scope-local.

## 4. Cross-Scope Linkages

For each finding tagged `cross_scope`, briefly explain the linkage:

- `F-NN-007` → scope 11 (backend/utils/http.py:88) — shared HTTP client used by 4 scopes
- ...

## 5. Risk-Prioritized Punch List (top 10)

Ordered by severity × cross-scope impact × effort:

1. **F-NN-001** — short title — why this is #1
2. ...

## 6. Open Questions

Items the agent could not resolve. Synthesis swarm or human owner must answer.

- Q1: ...
- Q2: ...
