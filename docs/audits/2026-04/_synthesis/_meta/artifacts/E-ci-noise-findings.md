# E-ci-noise-findings.md

CI failures observed on PR #146 (Workstream E) that are **NOT** caused by
Workstream E and are explicitly out of scope for this PR. Captured here as
hand-off signal for the appropriate downstream workstream.

**PR:** https://github.com/JoeyJoziah/investment-analysis-platform/pull/146
**Branch:** `remediation/audit-2026-04`
**Captured:** 2026-04-28

---

## 1. mypy / Type Check with mypy — workflow misconfig (→ G3 phase-1)

- **Symptom:** failed step "Comment PR with results" POSTs to a malformed
  URL: `api.github.com/repos/JoeyJoziah//issues/146/comments` (double slash,
  missing repo name).
- **Root cause:** the comment-action step is interpolating an empty value in
  place of `${{ github.repository }}` (or has the `JoeyJoziah/<repo>`
  segment hard-coded with a trailing-slash typo).
- **Substantive mypy result is HEALTHY:** 2,649 errors vs. 3,636 baseline =
  **−987 regression-free**. The workflow only failed at the comment-post
  step, not on the type-check itself.
- **Owner:** **Workstream G3 phase-1 (CI security / workflow hardening)**
- **Suggested fix:** patch `.github/workflows/{type-check,mypy}.yml`
  comment step to use `${{ github.repository }}` (or the issue URL provided
  by `${{ github.event.pull_request.comments_url }}`).
- **Severity:** medium (blocks PR check status, not real type signal).

## 2. Validate Links in Changed Files — anchor links in PRD reverse index (→ G6)

- **Symptom:** link-validator flags one or more URLs in the changed PRD /
  workpaper docs.
- **Likely cause:** anchor links in the audit reverse-index files; not
  introduced by Workstream E (E only added artifact files and one
  redirect README).
- **Owner:** **Workstream G6 (docs cleanup)**
- **Severity:** low (documentation hygiene only).

## 3. Vercel — frontend preview deploy failure (→ G3 / infra)

- **Symptom:** Vercel deploy preview fails on PR #146.
- **Why it's not E:** Workstream E touched **zero** frontend code. Diff
  contains only `pytest.ini`, Python tests, shell runner, audit artifacts,
  and `tests/README.md`. There is no path through E that would change a
  Vercel build artifact.
- **Owner:** **G3 / infra config** — pre-existing Vercel project /
  ignored-build-step configuration problem.
- **Severity:** low (preview only; production deploy unaffected).

## 4. PR Review Analysis / pr-health-check / stale-pr-check / Check Documentation Index

- **Symptom:** all four jobs fail in **< 12 seconds** — too fast to have
  performed substantive checks. Classic configuration / token / setup
  failure pattern in the GitHub Swarm Automation suite.
- **Why it's not E:** none of these jobs run domain logic against the diff;
  they're orchestration / metadata jobs.
- **Owner:** **G3 phase-1 (CI security / workflow hardening)** — likely the
  same ${{ github.repository }} interpolation class of bug, or missing
  permissions on the GITHUB_TOKEN scope.
- **Severity:** low (noise on every PR, not E-specific).

---

## What still matters for PR #146 mergeability

The **substantive** signal for Workstream E is the matrix of:

- backend-test / 3.10 × {unit, integration, security}
- backend-test / 3.11 × {unit, integration, security}
- backend-test / 3.12 × {unit, integration, security}

If those go green (or red with cascade-expected failures already documented
in `E-step1-failures.txt` / `E-step2-failures.txt`), the PR is mergeable on
its own merits. The 8 failures listed above are pre-existing CI plumbing
noise and must NOT be fixed inside this PR.
