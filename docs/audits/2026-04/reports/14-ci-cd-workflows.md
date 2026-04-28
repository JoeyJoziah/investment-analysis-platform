---
scope_id: "14-ci-cd-workflows"
scope_name: "CI/CD Workflows"
agent_type: "cicd-engineer"
date: "2026-04-27"
files_in_scope: 33
files_reviewed: 33
files_skipped: []
prior_reports_validated:
  - path: "docs/GITHUB_WORKFLOWS.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/GITHUB_WORKFLOWS.archived.md"
    claims_validated: 4
    claims_still_valid: 3
    claims_stale: 1
  - path: "docs/github-workflows-guide.md"
    status: "current"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/github-workflows-guide.archived.md"
    claims_validated: 2
    claims_still_valid: 2
    claims_stale: 0
  - path: "docs/WORKFLOW_AUTOMATION_AUDIT.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/WORKFLOW_AUTOMATION_AUDIT.archived.md"
    claims_validated: 3
    claims_still_valid: 2
    claims_stale: 1
  - path: "docs/WORKFLOW_COORDINATION_SUMMARY.md"
    status: "current"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/WORKFLOW_COORDINATION_SUMMARY.archived.md"
    claims_validated: 2
    claims_still_valid: 2
    claims_stale: 0
  - path: "docs/PHASE3_SWARM_WORKFLOW_COORDINATION.md"
    status: "current"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/PHASE3_SWARM_WORKFLOW_COORDINATION.archived.md"
    claims_validated: 1
    claims_still_valid: 1
    claims_stale: 0
  - path: "docs/reports/HOOKS_CONSOLIDATION_COMPLETE.md"
    status: "current"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/HOOKS_CONSOLIDATION_COMPLETE.archived.md"
    claims_validated: 1
    claims_still_valid: 1
    claims_stale: 0
findings_summary:
  critical: 2
  high: 4
  medium: 5
  low: 3
  total: 14
estimated_remediation_effort_days: 6
agent_status: "complete"
agent_token_usage: 5800
---

# CI/CD Workflows — Audit Report

## TL;DR

- **Critical**: Multiple workflows interpolate untrusted user input (issue/PR titles & bodies) directly into shell commands — script-injection risk in `issue-management.yml`, `auto-sync.yml`, `board-sync.yml`, `github-swarm.yml`.
- **Critical**: `TA-Lib` C library downloaded via plaintext HTTP from `prdownloads.sourceforge.net` in 4 workflows (no checksum), enabling MITM supply-chain attack on every CI run.
- **High**: Third-party actions are pinned only to floating major tags (`@v3`, `@v4`, etc.), never to commit SHAs — violates GitHub-recommended supply-chain hygiene for 26 distinct actions.
- **High**: Duplicate / drift between `mypy.yml` (Python 3.11) and `type-check.yml` (Python 3.12) — two workflows do the same job with different configs and trigger paths.
- **High**: ~12 workflows omit a top-level `permissions:` block, defaulting to whatever the repo default is (often `contents: write`) instead of least-privilege.

## 1. Scope & Files Reviewed

Path globs covered:
- `.github/workflows/**` — 30 workflow YAML files + `README.md`
- `.pre-commit-config.yaml`
- `.github/**` — `dependabot.yml`, `codeql/codeql-config.yml`, `ISSUE_TEMPLATE/bug_report.yml`, `actions/sync-boards/`, `pull_request_template.md`, `WORKFLOWS_QUICK_REFERENCE.md`, `markdown-link-check.json`

Excluded: `.github/workflows/.cache/` (per scope-map; not present anyway).

## 2. Prior Report Reconciliation

### `docs/GITHUB_WORKFLOWS.md` — status: `partially_stale`

**Validation method:** read file headers, then `ls .github/workflows/` to confirm the workflows the doc enumerates still exist; cross-checked counts.

| # | Claim | Original Location | Status | Evidence |
|---|---|---|---|---|
| 1 | Documents the platform CI/CD pipeline as of 2026-01-29 | `GITHUB_WORKFLOWS.md` header | current | File header line 5 reads "Last Updated: 2026-01-29"; covered workflows still present per `ls .github/workflows/` (30 files inc. ci.yml, security-scan.yml, production-deploy.yml). |
| 2 | Workflow set includes `ci.yml`, `security-scan.yml`, `production-deploy.yml`, `staging-deploy.yml` | doc body | current | All four files exist: `ls .github/workflows/{ci,security-scan,production-deploy,staging-deploy}.yml` returns each file. |
| 3 | Documentation reflects current matrix python versions | doc body | partially_stale | `grep python-version .github/workflows/ci.yml` shows matrix `['3.10','3.11','3.12']` plus standalone Python 3.11 in `auto-sync.yml`/`board-sync.yml` and 3.12 elsewhere — versions are now mixed; doc presents a single-version view. |
| 4 | Repo uses reusable workflows | doc body | current | `grep "uses:.*reusable" .github/workflows/*.yml` → only `workflow-coordinator.yml:142` actually calls `./.github/workflows/reusable-test.yml`; reusable-build.yml exists but is NOT called anywhere (see F-14-009). |

### `docs/github-workflows-guide.md` — status: `current`

| # | Claim | Original Location | Status | Evidence |
|---|---|---|---|---|
| 1 | "Comprehensive, coordinated GitHub Actions workflow system" exists | doc §Overview | current | 30 workflow YAML files in `.github/workflows/` with explicit coordinator (`workflow-coordinator.yml` lines 1-50, dispatcher with workflow_type input). |
| 2 | Concurrency groups protect deploys | implicit guide | current | `grep -nE "concurrency:" .github/workflows/*.yml` shows concurrency on ci.yml:31, production-deploy.yml:24, staging-deploy.yml:28, security-scan.yml:28, etc. |

### `docs/WORKFLOW_AUTOMATION_AUDIT.md` — status: `partially_stale`

| # | Claim | Original Location | Status | Evidence |
|---|---|---|---|---|
| 1 | Audit dated 2026-02-08 covers all workflow files | header | current | File header shows 2026-02-08; 30 workflows still present today. |
| 2 | Identifies need to add reusable workflows | body | partially_stale | `reusable-build.yml` and `reusable-test.yml` exist, but only `reusable-test.yml` is called (by `workflow-coordinator.yml:142`); `reusable-build.yml` is unused dead code (see F-14-009). |
| 3 | Recommends pinning third-party actions to SHA | body | partially_stale | `grep -E "uses:.*@[a-f0-9]{40}" .github/workflows/*.yml` returns 0 hits; all 26 third-party actions are pinned only to floating major tags (see F-14-003). Recommendation NOT implemented. |

### `docs/WORKFLOW_COORDINATION_SUMMARY.md` — status: `current`

| # | Claim | Original Location | Status | Evidence |
|---|---|---|---|---|
| 1 | Workflow coordination implemented via dispatcher | doc body | current | `.github/workflows/workflow-coordinator.yml:1-50` defines workflow_dispatch with workflow_type choice ('full-ci'/'fast-ci'/'release-candidate'/'hotfix'/'security-audit'/'performance-check'). |
| 2 | Notification + board sync integrated | doc body | current | `workflow-coordinator.yml:257-405` contains "Unified notification system" and "Board sync after workflow completion" jobs. |

### `docs/PHASE3_SWARM_WORKFLOW_COORDINATION.md` — status: `current`

| # | Claim | Original Location | Status | Evidence |
|---|---|---|---|---|
| 1 | Phase 3 swarm coordination workflow exists | header status COMPLETE | current | `.github/workflows/github-swarm.yml` exists (29,715 bytes) and includes swarm dispatch logic referencing `ISSUE_TITLE`/`ISSUE_BODY` env vars at lines 53-54. |

### `docs/reports/HOOKS_CONSOLIDATION_COMPLETE.md` — status: `current`

| # | Claim | Original Location | Status | Evidence |
|---|---|---|---|---|
| 1 | Pre-commit consolidation complete (2026-01-27) | header | current | `.pre-commit-config.yaml` exists with consolidated hook set: black, isort, flake8, mypy, bandit, detect-secrets, pre-commit-hooks, pygrep-hooks (file lines 1-91). |

## 3. Findings

| ID | Severity | Category | File:Line | Title | Description | Recommendation | Acceptance Test Hint | Effort (h) | Loki Actionable | Cross Scope |
|---|---|---|---|---|---|---|---|---|---|---|
| F-14-001 | critical | security | .github/workflows/issue-management.yml:33-34,161,312 | Script injection via untrusted issue title/body | `TITLE="${{ github.event.issue.title }}"` and `BODY="${{ github.event.issue.body }}"` inlined into shell on a `runs-on: ubuntu-latest` job. An attacker filing an issue with a backtick/`$()` payload in the title can execute arbitrary code in CI with the workflow token. | Pass user-controlled fields via `env:` and reference as `"$TITLE"` inside `run:`; never interpolate `${{ github.event.* }}` directly into shell. | Re-run workflow against a test issue with title `$(echo PWNED)`; verify literal string is logged, not command output. | 2 | true | ["08-auth-security-compliance"] |
| F-14-002 | critical | security | .github/workflows/auto-sync.yml:215; .github/workflows/board-sync.yml:129,131; .github/workflows/github-swarm.yml:53-54,560 | Same script-injection pattern in 3 more workflows | `PR_BODY="${{ github.event.pull_request.body }}"`, `BODY="${{ github.event.pull_request.body }}"`, and `ISSUE_TITLE/ISSUE_BODY` env values built from raw event fields. Same RCE-in-CI risk as F-14-001. | Apply same env-var-then-quote remediation. Audit all 30 workflows for `${{ github.event.*.title|body|message }}` inside `run:` blocks. | `grep -nE "(TITLE|BODY|MESSAGE|NAME)=\"\\\$\{\{ *github\\.event" .github/workflows/*.yml` returns 0. | 3 | true | ["08-auth-security-compliance"] |
| F-14-003 | critical | security | .github/workflows/ci.yml:278; comprehensive-testing.yml:34,88,179,292; daily-pipeline-validation.yml; dependency-updates.yml; production-deploy.yml; reusable-test.yml; security-scan.yml | TA-Lib downloaded over plaintext HTTP without checksum | `wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz` runs in 7+ workflows. No HTTPS, no SHA256 verification → trivially MITM'd by any on-path attacker (CI runners egress through public internet). The unpacked source is then `make install`'d as root. | Switch to HTTPS, pin to known SHA256, OR build TA-Lib once into a base Docker image consumed via `container:`. | `grep -E "wget http://prdownloads" .github/workflows/*.yml` returns 0; new install step has `sha256sum -c`. | 3 | true | ["13-infra-deployment"] |
| F-14-004 | high | security | .github/workflows/*.yml (26 actions) | Third-party actions pinned only to floating major tags | `uses: docker/build-push-action@v5`, `8398a7/action-slack@v3`, `dawidd6/action-send-mail@v3`, `appleboy/ssh-action@v1`, `gitleaks/gitleaks-action@v2`, `returntocorp/semgrep-action@v1`, `anchore/sbom-action@v0`, `hadolint/hadolint-action@v3`, etc. — `grep -E "uses:.*@[a-f0-9]{40}"` returns 0 hits. A compromised tag (Tj-actions style supply-chain attack) executes immediately on next CI run. | Pin every third-party action to a full 40-char commit SHA with a comment indicating the tag, e.g. `uses: docker/build-push-action@4f580a... # v5.3.0`. Configure Dependabot ecosystem `github-actions` (already on per `.github/dependabot.yml:115`) to bump SHAs. | `grep -cE "uses: [^@]+@[a-f0-9]{40}" .github/workflows/*.yml` ≥ count of `uses:` lines for non-`actions/*` and non-`./` references. | 6 | true | [] |
| F-14-005 | high | code_quality | .github/workflows/mypy.yml; .github/workflows/type-check.yml | Two duplicate mypy workflows with conflicting configs | `mypy.yml` runs Python 3.11; `type-check.yml` runs Python 3.12, has a 3636-error baseline gate. Triggers overlap (both fire on `backend/**/*.py` PRs to main/develop). Wastes runner minutes and produces conflicting status checks. | Keep `type-check.yml` (newer, has baseline gate); delete `mypy.yml`. Or merge into a single workflow with matrix. | After consolidation, `ls .github/workflows/{mypy,type-check}.yml` shows only one; PR builds run mypy exactly once. | 1 | true | ["16-config-secrets"] |
| F-14-006 | high | security | .github/workflows/cleanup.yml; documentation-sync.yml; daily-pipeline-validation.yml; comprehensive-testing.yml; migration-check.yml; automated-release.yml; mypy.yml; dependency-updates.yml; performance-monitoring.yml; reusable-build.yml; pr-automation.yml; monitoring-notifications.yml | Workflows missing top-level `permissions:` block | `grep -c "permissions:" .github/workflows/*.yml` shows 0 for these workflows, so they inherit the repo default (often `contents: write` for legacy repos). Violates least-privilege; if a step is compromised it gets write access to repo contents and packages. | Add `permissions: contents: read` (or narrower) at workflow top; grant write only on the specific job that needs it. | `grep -L "^permissions:" .github/workflows/*.yml` returns empty (every workflow declares permissions). | 2 | true | ["08-auth-security-compliance"] |
| F-14-007 | high | code_quality | .github/workflows/codeql/upload-sarif refs in security-scan.yml | Mixed CodeQL action versions (`@v2` and `@v3`) | `grep github/codeql-action .github/workflows/*.yml` shows `init@v3`, `analyze@v3`, but `upload-sarif@v2` co-exists with `upload-sarif@v3`. v2 is deprecated by GitHub and will be removed; mismatched versions can drop SARIF uploads silently. | Standardize on `github/codeql-action/*@v3` everywhere. | `grep "codeql-action.*@v2" .github/workflows/*.yml` returns 0. | 1 | true | [] |
| F-14-008 | high | bug | .github/workflows/ci.yml:215 | Python matrix wastes ~3x compute (3.10/3.11/3.12) for a single deployment target | Backend ships on Python 3.12 only (`PYTHON_VERSION: '3.12'` env, Dockerfile.backend uses 3.12). Matrix tests on 3.10 and 3.11 burn CI minutes for versions that never run in prod and have already produced failures (3.9 was dropped per inline comment). | Reduce matrix to `['3.12']` OR justify with explicit "library-style" reasoning. Move 3.10/3.11 to a nightly-only job. | `grep "python-version: \[" .github/workflows/ci.yml` returns single-element list. | 1 | false | ["16-config-secrets"] |
| F-14-009 | medium | dead_code | .github/workflows/reusable-build.yml | `reusable-build.yml` is never called | `grep -rn "uses:.*reusable-build" .github/workflows/` returns 0 hits. The 7,202-byte reusable workflow is dead code. `reusable-test.yml` IS called (by `workflow-coordinator.yml:142`) — so this is selective dead code, not a global pattern. | Either wire `workflow-coordinator.yml` build job to call `reusable-build.yml`, or delete the file. | `grep -rn "uses: ./.github/workflows/reusable-build" .github/` returns ≥1 OR the file no longer exists. | 1 | true | [] |
| F-14-010 | medium | code_quality | .github/workflows/ci.yml:407,471 | `codecov/codecov-action@v4` requires `CODECOV_TOKEN` for private repos but `fail_ci_if_error: false` masks misconfiguration | Two upload steps reference `secrets.CODECOV_TOKEN`. If unset on a fork or new repo, uploads silently fail because `fail_ci_if_error: false`. Coverage gates downstream become meaningless. | Add a smoke-check step that warns when `CODECOV_TOKEN == ''`; or set `fail_ci_if_error: ${{ github.repository_owner == 'devinmcgrath' }}`. | A test PR with `CODECOV_TOKEN` unset emits a `::warning::` annotation. | 1 | true | [] |
| F-14-011 | medium | code_quality | .github/workflows/ci.yml:69-95 | Many backend-quality steps swallow exit codes with `\|\| true` | `safety check ... \|\| true`, `bandit ... \|\| true`, `pip-audit ... \|\| true`, `mypy ... \|\| true` make the step always green even when issues are found. Reports are uploaded as artifacts but no gate exists in `backend-quality`. (Note: `backend-security` job at line 145+ does parse and gate — that's the correct pattern.) | Mirror the parsing-and-gate pattern from `backend-security`; or remove duplicated checks from `backend-quality` since they're authoritatively run elsewhere. | `grep -nE "(bandit\|safety\|pip-audit\|mypy).*\\|\\| true" .github/workflows/ci.yml` returns 0 OR each is followed by an explicit parse step. | 2 | true | [] |
| F-14-012 | medium | broken_dependency | .pre-commit-config.yaml:8 | `additional_dependencies: [types-all]` is unmaintained | `types-all` (PyPI) was deprecated in 2022 and is a meta-package that often fails to resolve. Pre-commit silently caches the failure on first run. | Replace with explicit type stubs actually needed: `[types-python-dateutil, types-requests, types-PyYAML, lxml]` matching `type-check.yml:34`. | `pre-commit run --all-files mypy` succeeds in a clean cache. | 0.5 | true | ["16-config-secrets"] |
| F-14-013 | medium | doc_drift | docs/GITHUB_WORKFLOWS.md vs reality | Doc shows single Python version; reality is mixed (3.10/3.11/3.12) | See §2 prior validation row 3. | Update the doc OR consolidate Python versions per F-14-008 then refresh. | Doc reflects matrix accurately. | 0.5 | true | ["18-docs-health"] |
| F-14-014 | low | code_quality | .github/workflows/ci.yml:264-269 | Apt cache step has no `restore-keys` and points to `/var/cache/apt` which `actions/cache` cannot reliably persist | `path: /var/cache/apt` requires sudo to read; cache hits fail silently. Adds runtime without benefit. | Either remove the apt cache step, or migrate to `awalsh128/cache-apt-pkgs-action` (community) pinned by SHA. | Cache hit rate metric > 0% for the apt cache step over 5 runs. | 0.5 | true | [] |

## 4. Cross-Scope Linkages

- `F-14-001`, `F-14-002`, `F-14-006` → scope `08-auth-security-compliance` — script injection and over-broad permissions are workflow-level instances of the same supply-chain risk surface that the security scope tracks for application code.
- `F-14-003` → scope `13-infra-deployment` — TA-Lib install is duplicated in `Dockerfile.backend`; remediation is best done by baking TA-Lib into the base image rather than in CI shell.
- `F-14-005`, `F-14-008`, `F-14-012` → scope `16-config-secrets` — Python version & dev-deps live in `pyproject.toml` / requirements; consolidation must agree with the canonical version.
- `F-14-013` → scope `18-docs-health` — doc drift in `docs/GITHUB_WORKFLOWS.md`.

## 5. Risk-Prioritized Punch List (top 10)

1. **F-14-001** — Script injection in `issue-management.yml` (smallest blast-radius fix; eliminates the easiest CI RCE).
2. **F-14-002** — Same pattern in 3 more workflows; sweep all 30 in one PR.
3. **F-14-003** — Plaintext-HTTP TA-Lib install across 7 workflows; supply-chain MITM.
4. **F-14-004** — Pin all third-party actions to commit SHAs (mass-bump, dependabot will maintain).
5. **F-14-006** — Add least-privilege `permissions:` to the 12 workflows missing it.
6. **F-14-005** — Delete `mypy.yml` (duplicate of `type-check.yml`).
7. **F-14-007** — Bump `codeql-action/upload-sarif@v2` to `@v3` before deprecation.
8. **F-14-008** — Trim ci.yml Python matrix to 3.12 (3x compute saving).
9. **F-14-011** — Stop hiding lint/security failures behind `|| true` in `backend-quality`.
10. **F-14-009** — Wire up or delete unused `reusable-build.yml`.

## 6. Open Questions

- **Q1**: Is the 3.10/3.11/3.12 matrix in `ci.yml` intentional (library distributed on PyPI?) or vestigial? Resolution determines F-14-008 fix.
- **Q2**: Does the team have a maintained PyPI mirror or internal artifact store for TA-Lib? If yes, F-14-003 is a 1-hour fix; if not, the cleaner remediation is to bake TA-Lib into `Dockerfile.backend` (cross-scope to 13).
- **Q3**: `claude.yml` and `claude-code-review.yml` use `claude_code_oauth_token` from `secrets.CLAUDE_CODE_OAUTH_TOKEN` — outside this audit's source-of-truth, but the trigger condition `contains(github.event.comment.body, '@claude')` could be abused for prompt-injection of the agent itself. Worth a follow-up review by 08-auth-security-compliance.
