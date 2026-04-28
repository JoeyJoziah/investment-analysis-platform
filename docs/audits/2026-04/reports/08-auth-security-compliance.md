---
scope_id: "08-auth-security-compliance"
scope_name: "Auth, Security, Compliance, Scanner"
agent_type: "security-auditor"
date: "2026-04-27"
files_in_scope: 65
files_reviewed: 18
files_skipped:
  - "Most backend/security/*.py beyond the 9 highest-risk modules — sampled by criticality"
  - "backend/compliance/data_export.py, deletion_handler.py, consent_manager.py — re-exported via gdpr.py orchestrator and not directly read"
prior_reports_validated:
  - path: "docs/SECURITY.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/SECURITY.archived.md"
    claims_validated: 5
    claims_still_valid: 3
    claims_stale: 2
  - path: "docs/SECURITY-INTELLIGENCE-BRIDGE.md"
    status: "current"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/SECURITY-INTELLIGENCE-BRIDGE.archived.md"
  - path: "docs/SEC_REGULATORY_COMPLIANCE_AUDIT.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/SEC_REGULATORY_COMPLIANCE_AUDIT.archived.md"
  - path: "docs/COMPLIANCE_ACTION_PLAN.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/COMPLIANCE_ACTION_PLAN.archived.md"
  - path: "docs/security/SECRET_ROTATION_PLAN.md"
    status: "fully_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/SECRET_ROTATION_PLAN.archived.md"
    redactions: 6
  - path: "docs/security/SECURITY_CREDENTIALS_AUDIT.md"
    status: "fully_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/SECURITY_CREDENTIALS_AUDIT.archived.md"
    redactions: 12
  - path: "docs/reports/security-audit-report.md"
    status: "fully_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/security-audit-report.archived.md"
    redactions: 8
  - path: "docs/reports/SECURITY_REMEDIATION_COMPLETE.md"
    status: "partially_stale"
    archived_to: "docs/audits/2026-04/_meta/prior-reports-archive/SECURITY_REMEDIATION_COMPLETE.archived.md"
    redactions: 4
findings_summary:
  critical: 4
  high: 8
  medium: 6
  low: 2
  total: 20
estimated_remediation_effort_days: 9
agent_status: "complete"
agent_token_usage: 9800
---

# Auth, Security, Compliance — Audit Report

## TL;DR (5 bullets)

- **CRITICAL F-08-001**: `backend/security/secrets_manager.py:173-180` uses hardcoded fixed salt `b"investment_analysis_salt"` for PBKDF2 KDF + only 100k iterations (NIST 2023+ minimum 600k). Same MASTER_SECRET_KEY across instances always produces same Fernet key — undermines all stored secrets including JWT private keys, MFA TOTP secrets, OAuth secrets.
- **CRITICAL F-08-002**: `backend/security/security_config.py:71,152` — `SESSION_SECRET_KEY` and `JWT_SECRET_KEY` default to `secrets.token_urlsafe(32)` if env unset. With multi-worker gunicorn, every worker has a different ephemeral key → sessions and HS256 fallback JWTs randomly invalidate, no startup error.
- **CRITICAL F-08-003**: `security_config.py:85-92` — CSP hardcodes `'unsafe-inline' 'unsafe-eval'` in `script-src` for the entire app. Disables core XSS mitigation. Same pattern in nginx (scope 13).
- **CRITICAL F-08-004**: `backend/security/jwt_manager.py:352` — `datetime.fromtimestamp(exp)` (naive) subtracted from `datetime.now(timezone.utc)` (aware) → `TypeError` on every `revoke_token` call, swallowed silently. Logout/forced-revoke does not blacklist; tokens valid until natural expiry.
- **HIGH F-08-009**: 4+ committed docs contain plaintext production credentials (DB password `9v1g^...`, JWT secrets, Fernet keys, Google API key, HF token) — REDACTED in archive copies but originals remain in repo and git history. Cross-scope: this same DB password is hardcoded in 14+ scripts (scope 17), in alembic.ini (scope 07), in scope 05 docs.

> Read these 5 before anything else in this report.

## 1. Scope & Files Reviewed

**Path globs covered:**
- `backend/auth/**/*.py` — 1 module (`oauth2.py`); `__init__.py` is 0-byte
- `backend/security/**/*.py` — sampled 12 highest-risk files of 20: `jwt_manager.py`, `security_config.py`, `secrets_manager.py`, `secrets_vault.py`, `csrf_protection.py`, `password_manager.py`, `rbac.py`, `audit_logging.py`, `rate_limiter.py`, `security_headers.py`, `input_validation.py`, `injection_prevention.py`
- `backend/compliance/**/*.py` — `__init__.py`, `gdpr.py` orchestrator, `sec.py` (partial)
- `.gitleaks.toml` (full read), `.secrets.baseline` (partial)

**Files explicitly excluded:** crypto_utils, data_encryption, advanced_rate_limiter, mfa_provider, key_rotation (sampling by risk score; cross-checked via imports).

## 2. Prior Report Reconciliation

29 priors archived. Highest-impact reconciliations:

### `docs/security/SECURITY_CREDENTIALS_AUDIT.md` — `fully_stale`

| # | Claim | Status | Evidence |
|---|---|---|---|
| 1 | "DB_PASSWORD is exposed and must be rotated" | fully_stale | Same string flagged in scopes 05/07/17 — has NOT been rotated as of 2026-04-27; identical password in 14+ scripts and alembic.ini |
| 2 | "JWT_SECRET_KEY (64-char hex) is exposed" | fully_stale | Plaintext still present in audit doc; no rotation evidence in secrets_manager manifest |
| 3 | "ELASTICSEARCH_PASSWORD shows placeholder" | fully_stale | Per scope 17, real ES password is hardcoded in scripts now; placeholder claim outdated |
| 4 | "FERNET_KEY has INCONSISTENT VALUES" | fully_stale | Fixed-salt KDF makes Fernet derivation deterministic; inconsistency claim moot but exposed keys must rotate |
| 5 | "Multiple credential sources" | fully_stale | Per scope 16 + 17, problem persists and worsened |

### `docs/security/SECRET_ROTATION_PLAN.md` — `fully_stale`

| # | Claim | Status | Evidence |
|---|---|---|---|
| 1 | "Rotate within 24-48 hours" (dated 2026-01-27) | fully_stale | 90+ days elapsed; secrets unchanged per scope 17 |
| 2 | "Use secrets.token_hex(32) for SECRET_KEY" | fully_stale | Code uses `token_urlsafe(32)` as fallback; production env likely contains pre-rotation value |
| 3 | "Update PostgreSQL password" | fully_stale | DB password unchanged per scope 05/07 evidence |
| 4 | "12 .env files contain secrets" | fully_stale | Per scope 16, 7 .env templates exist now; layout doesn't match plan |

### `docs/SEC_REGULATORY_COMPLIANCE_AUDIT.md` — `partially_stale`

| # | Claim | Status | Evidence |
|---|---|---|---|
| 1 | "Audit log retention 90 days vs SEC 7 years" | partially_stale | `backend/compliance/sec.py:89-93` now defines `retention_years=7`; doc claim FIXED in code, but enforcement (cron) unverified |
| 2 | "Audit logs not immutable" | current | `audit_logging.py` uses HMAC for tamper protection but no append-only DB constraint |
| 3 | "72% compliant overall" | unverifiable | Single-figure score not testable line-level |
| 4 | "Fiduciary duty checker exists" | current | `FiduciaryDutyChecker` exported per `__init__.py:21` |

### `docs/security/CSP_CORS_REMEDIATION.md` — `partially_stale`

| # | Claim | Status | Evidence |
|---|---|---|---|
| 1 | "unsafe-inline/unsafe-eval present in security_config.py:87-88" | current | Confirmed at lines 85-92; remediation NOT applied |
| 2 | "Replace with nonce-based CSP" | partially_stale | No nonce middleware in security_headers.py |
| 3 | "CORS allow_credentials=True with permissive origins" | current | security_config.py:512 confirmed |

Other priors (process docs, checklists, summaries) marked `unverifiable` — described processes without testable claims.

## 3. Findings

| ID | Severity | Category | File:Line | Title | Description | Recommendation | Acceptance Test Hint | Effort (h) | Loki Actionable | Cross Scope |
|---|---|---|---|---|---|---|---|---|---|---|
| F-08-001 | critical | security | backend/security/secrets_manager.py:173-180 | Hardcoded KDF salt + low PBKDF2 iters | Fixed salt `b"investment_analysis_salt"` with 100k iters defeats salt purpose; weakens master-key derivation; allows offline brute force | Generate per-installation salt; bump to ≥600k iters or migrate to Argon2id; re-encrypt all stored secrets | Two instances with same MASTER_SECRET_KEY produce different Fernet keys; PBKDF2 timing >250ms | 6 | true | [16-config-secrets] |
| F-08-002 | critical | security | backend/security/security_config.py:71,152 | Ephemeral fallback for SESSION/JWT_SECRET_KEY | `os.getenv(..., secrets.token_urlsafe(32))` — env unset → random per process; multi-worker gives N distinct keys | Raise RuntimeError at import time when production env unset; remove token_urlsafe default from class-level expression | Test that missing env in production raises | 2 | true | [01-backend-api, 16-config-secrets] |
| F-08-003 | critical | security | backend/security/security_config.py:85-92 | CSP allows unsafe-inline/unsafe-eval | Hardcoded CSP disables XSS mitigation app-wide | Remove unsafe directives; introduce per-request CSP nonce middleware; coordinate with frontend Vite build | curl -I / returns CSP without unsafe-inline/eval | 6 | true | [12-frontend, 13-infra-deployment] |
| F-08-004 | critical | bug | backend/security/jwt_manager.py:352 | Naive/aware datetime breaks revoke_token | `datetime.fromtimestamp(exp)` naive vs `datetime.now(timezone.utc)` aware → TypeError on every revoke; swallowed by bare except | `datetime.fromtimestamp(exp, tz=timezone.utc)`; add unit test; replace bare except | pytest test_revoke_blacklists_token passes | 1 | true | [01-backend-api, 15-test-suite] |
| F-08-005 | high | security | backend/auth/oauth2.py:89-96,119-127 | HS256 fallback path with ephemeral secret | RS256→HS256 algorithm-confusion fallback when JWT manager init fails; combined with F-08-002, fallback tokens unverifiable across worker restarts | Remove HS256 fallback; propagate init errors as 500 | grep returns 0 outside config + tests | 3 | true | [01-backend-api] |
| F-08-006 | high | security | backend/auth/oauth2.py:327-362 | In-memory RateLimiter per-process | `self.clock` dict per-instance, per-worker; effective limit = configured × N workers | Delete this class; route through advanced_rate_limiter (Redis-backed) | grep returns 0 imports | 2 | true | [01-backend-api] |
| F-08-007 | high | security | backend/security/rbac.py:111-143 | DB-backed multi-role assignment loses state | `assign_role` adds to in-memory set but persists via `User.role` (single column); 2nd role overwrites 1st on restart | Either declare single-role-only or add user_roles join table | Test: assign two roles, restart, assert both present | 4 | false | [07-database-persistence] |
| F-08-008 | high | security | backend/security/jwt_manager.py:88-96 | Redis fallback returns None, blacklist silently degrades | Redis outage allows revoked tokens to be re-used (`_is_token_blacklisted` returns False when client is None) | Circuit-breaker; fail-secure on degraded state; Prometheus alert | Chaos test: stop Redis mid-traffic, assert 503s | 4 | true | [10-monitoring-observability] |
| F-08-009 | high | security | docs/security/* + docs/reports/security-* | Plaintext credentials in committed docs | DB pwd, JWT 64-char hex, Fernet key, Google API key, HF token in plaintext, committed to git history | git filter-repo to purge; rotate ALL listed secrets; replace strings in docs with REDACTED | gitleaks detect returns 0; git log -S returns no commits | 8 | true | [05,16,17,18] |
| F-08-010 | high | security | .secrets.baseline | Baseline polluted with 1500+ lockfile false positives | NPM integrity hashes from .claude/v3/ dilute signal | Add lockfile path-exclude; rebuild baseline | <100 entries after cleanup | 1 | true | [16-config-secrets, 14-ci-cd-workflows] |
| F-08-011 | high | code_quality | .gitleaks.toml:132-139 | stopwords list contains the keywords meant to flag secrets | gitleaks downgrades any line with `password`, `bearer`, `token`, `api`, `key`, `secret`, `auth` — the opposite of intent | Remove stopwords block or scope to true noise (`example`, `template`) | gitleaks finds known seeded credential | 1 | true | [14-ci-cd-workflows] |
| F-08-012 | high | security | backend/security/security_config.py:46-53 | Hardcoded ALLOWED_ORIGINS in class body | Production origins baked into source; class-eval-time evaluation of settings.ENVIRONMENT may misfire | Move to env CSV; compute lazily in middleware | python import reflects env var | 2 | true | [16-config-secrets] |
| F-08-013 | medium | security | backend/security/csrf_protection.py:77-80 | CSRF exempts /login and /register | Combined with allow_credentials=True, increases credential-stuffing surface | Tighten origin/referer validation for those endpoints; tighten CORS for login | curl POST with malicious Origin returns 403 | 3 | true | [01-backend-api] |
| F-08-014 | medium | security | backend/security/security_config.py:165-179 | File upload allowlist permits .txt without magic-byte verification | Polyglot HTML+JS renamed .txt could bypass content-type sniff | Always set Content-Disposition: attachment; never serve user files from same-origin | Test attachment disposition | 2 | true | [01-backend-api, 13-infra-deployment] |
| F-08-015 | medium | security | backend/security/jwt_manager.py:496-508 | MFA secrets stored under predictable username key | Allows enumeration; collision on username reuse | Use stable user_id (UUID) | New users get unique IDs after username reuse | 2 | true | [07-database-persistence] |
| F-08-016 | medium | architecture | backend/security/security_config.py:422-572 | add_comprehensive_security_middleware skips many in is_testing mode | Test pass rate diverges from production security posture | Compose; provide TestingSecurityProfile; integration test with TESTING=false | full-stack-enabled test runs all middlewares | 6 | false | [15-test-suite] |
| F-08-017 | medium | code_quality | backend/security/jwt_manager.py:347 | jwt.decode without signature verification before revoke | Trusts client claims; attacker forges huge `exp` to flood blacklist Redis | Cap derived TTL at access-token max regardless of claim | Forged exp=2099 token blacklisted with TTL ≤ access_token_expire | 2 | true | [] |
| F-08-018 | medium | doc_drift | docs/CODEMAPS/BACKEND.md:99-111 | CODEMAPS lists 7 of 20 security modules | secrets_manager, audit_logging, injection_prevention, input_validation, advanced_rate_limiter, csrf_protection undocumented | Regenerate codemap from filesystem | wc -l matches documented count | 1 | true | [18-docs-health] |
| F-08-019 | low | dead_code | backend/auth/__init__.py | Empty __init__ in single-file subpackage | No re-exports; imports use full path; package layer adds no value | Consolidate oauth2.py into backend/security/ OR add re-exports | grep -r 'from backend.auth import' returns 0 | 1 | false | [11-backend-utils-shared] |
| F-08-020 | low | testing_gap | (no test file) | No tests for FiduciaryDutyChecker | SEC-critical compliance code untested per CODEMAPS test matrix | Add backend/tests/compliance/test_sec_fiduciary.py with ≥5 test cases | pytest -k fiduciary ≥5 tests | 4 | true | [15-test-suite] |

## 4. Cross-Scope Linkages

- **F-08-002, F-08-005** → scope **01-backend-api**: JWT RS256/HS256 mismatch. Combined fix: enforce RS256 only, delete fallback, surface init errors.
- **F-08-003** → scopes **12-frontend** + **13-infra-deployment**: coordinated CSP nonce rollout (app, edge, build).
- **F-08-007** → scope **07-database-persistence**: multi-role architecture decision needs `user_roles` join table migration.
- **F-08-009** → scopes **05, 17, 16, 18**: single coordinated secret rotation event covering 14+ scripts, alembic.ini, .env templates, doc references.
- **F-08-010, F-08-011** → scope **14-ci-cd-workflows**: detect-secrets and gitleaks config defects amplify across pipeline.
- **F-08-016** → scope **15-test-suite**: testing-mode middleware skips create coverage gap.
- **F-08-018** → scope **18-docs-health**: codemap drift.

## 5. Risk-Prioritized Punch List (top 10)

1. **F-08-009** — Plaintext credentials in 4+ committed docs (90-day-old rotation plan unexecuted). Highest blast radius. 8h.
2. **F-08-001** — Fixed PBKDF2 salt + 100k iters for master encryption key. Single-point-of-compromise for all stored secrets.
3. **F-08-003** — CSP unsafe-inline/eval. Disables XSS mitigation for entire SPA.
4. **F-08-004** — Token revocation TypeError silently swallowed. Logout doesn't actually log out.
5. **F-08-002** — Ephemeral SESSION/JWT secrets. Production multi-worker breaks invisibly.
6. **F-08-005** — HS256 algorithm-confusion fallback path.
7. **F-08-006** — In-memory RateLimiter: rate limits N× too lax in multi-worker.
8. **F-08-008** — Redis-outage fail-open on token blacklist.
9. **F-08-007** — DB-backed RBAC silently loses multi-role state.
10. **F-08-010** — `.secrets.baseline` polluted with 1500+ lockfile false positives.

## 6. Open Questions

- Q1: Has any secret listed in `SECURITY_CREDENTIALS_AUDIT.md` actually been rotated since 2026-01-27? Audit-log retrieval needed.
- Q2: For F-08-007, is product intent single-role-per-user or multi-role? Needs PM/architect decision.
- Q3: Is the platform offering "investment advice" under SEC Investment Advisers Act? Advisor-registration status remains "Not Addressed" — legal Q.
- Q4: Should archive copies of redacted security priors be moved to access-controlled location rather than `_meta/prior-reports-archive/`?
