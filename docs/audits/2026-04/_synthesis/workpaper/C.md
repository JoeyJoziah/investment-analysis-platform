# Cluster C — CSP `unsafe-inline` / `unsafe-eval` Removal

**Worker:** C (CSP)
**Cluster size:** 3 findings
**Status:** `partial` — see §9
**Severity floor:** medium · **Severity ceiling:** critical

---

## 1. Cluster overview

The application has Content Security Policy (CSP) headers set in three places, all of which include `'unsafe-inline'` and/or `'unsafe-eval'` directives — defeating the primary purpose of CSP as an XSS mitigation. The audit findings span:

- **Backend FastAPI/Flask layer** (`backend/security/security_config.py`) — emits CSP via response middleware.
- **Frontend SPA HTML layer** (`frontend/web/index.html`) — has *no* CSP `<meta>` tag (no compensating control).
- **Edge nginx layer** (`infrastructure/docker/nginx/conf.d/security-headers.conf`) — overrides backend CSP at the reverse-proxy.

EXECUTIVE_SUMMARY §4 explicitly calls out that the three layers must move "in lockstep with a nonce strategy" — partial removal in only one layer creates header conflicts (browsers obey the most-restrictive received CSP, so a strict edge CSP combined with inline `<script>` blocks left unfixed will break the SPA).

A second nginx variant exists at `infrastructure/docker/frontend/security-headers.conf` which (per F-13-014 description) does **not** include `'unsafe-inline'`/`'unsafe-eval'` — so the divergence itself is a code-quality smell to resolve in this cluster.

---

## 2. Member findings

| Finding ID  | Scope                       | Severity | File                                                                |
| ----------- | --------------------------- | -------- | ------------------------------------------------------------------- |
| F-08-003    | 08-auth-security-compliance | critical | `backend/security/security_config.py:85-92`                         |
| F-12-003    | 12-frontend                 | critical | `frontend/web/index.html` + `frontend/web/vite.config.ts`           |
| F-13-014    | 13-infra-deployment         | medium   | `infrastructure/docker/nginx/conf.d/security-headers.conf:6`        |

All three IDs are addressed in §3 below.

---

## 3. Sequenced fix steps

> **Critical sequencing rule:** roll out in **CSP-Report-Only** mode first (browsers evaluate but don't block), collect `report-uri` violations from real traffic, fix what's surfaced, **then** flip to enforcing. Skipping this is destructive — a strict CSP that's missed even one Vite/React/inline-handler edge case will white-screen the SPA for end users.

### Phase 0 — Decide nonce vs. hash strategy (BLOCKING DECISION)

The audit did not specify nonce-vs-hash. **Default recommendation: per-request nonce** (better for SPAs that may legitimately have inline JSON state script tags or analytics snippets). This is flagged for human ack in §9.

### Phase 1 — Backend nonce middleware (F-08-003 prep)

1. Add a request-scoped nonce generator in `backend/security/security_config.py`:
   - 16-byte URL-safe random per request, attached to `request.state.csp_nonce`.
2. Convert `SECURITY_HEADERS["Content-Security-Policy"]` from a static string into a callable that interpolates the nonce.
3. Expose nonce to template context (Jinja2 / response interceptor) so the frontend bootstrap script tag can carry `nonce="..."`.
4. **Switch directive name to `Content-Security-Policy-Report-Only`** for this phase. Add `report-uri /csp-report` (or `report-to` group) and a backend endpoint that logs violations.

### Phase 2 — Frontend Vite + index.html (F-12-003)

1. **Path correction:** the slice references `frontend/web/src/index.html` but the actual file is `frontend/web/index.html` (Vite root). Verified above.
2. In `frontend/web/index.html`, remove any inline `<script>` / `<style>` bodies. If any remain (e.g., bootstrap config, analytics), either:
   - move to a separate `.js`/`.css` served from `'self'`, or
   - keep inline and add `nonce="%CSP_NONCE%"` placeholder for backend interpolation.
3. In `frontend/web/vite.config.ts`:
   - For **dev**: Vite HMR uses inline scripts and (historically) `eval`-style transforms. Acceptable to keep `unsafe-eval` in dev CSP — but the dev/prod CSP must be split. Use Vite's `mode === 'production'` branch to emit a strict CSP via the `transformIndexHtml` hook.
   - For **prod**: ensure no plugin emits inline script bodies without a nonce placeholder. Verify with `vite build && grep -E '<script[^>]*>[^<]' dist/index.html`.
4. Add `<meta http-equiv="Content-Security-Policy-Report-Only" content="...">` as a defense-in-depth layer in `index.html` (browsers enforce the most-restrictive of header + meta; only meta-set CSP is ignored for `frame-ancestors`/`report-uri` — those must come from the header).

### Phase 3 — Nginx edge alignment (F-13-014)

1. Update `infrastructure/docker/nginx/conf.d/security-headers.conf:6`:
   - Remove `'unsafe-inline'` and `'unsafe-eval'` from `script-src`.
   - Replace with `'nonce-$csp_nonce'` where `$csp_nonce` is set via `set_secure_random_alphanum` (ngx_http_set_misc_module) **OR** trust the backend to set the header and have nginx pass it through with `proxy_pass_header Content-Security-Policy;` and *not* re-emit its own.
   - **Recommended:** delete the duplicate CSP from nginx entirely; let the backend own CSP since only the backend can synchronize the nonce with the rendered HTML. Keep nginx's other headers (HSTS, X-Frame-Options, etc.).
2. Reconcile divergence with `infrastructure/docker/frontend/security-headers.conf` and `config/services/nginx/security-headers.conf` — apply the same fix to all three; ideally collapse to a single included file.
3. Keep `Content-Security-Policy-Report-Only` for ≥7 days of production traffic.

### Phase 4 — Flip to enforcing

1. Review CSP violation reports from Phase 1–3 monitoring; fix any legitimate sources flagged.
2. Rename header `Content-Security-Policy-Report-Only` → `Content-Security-Policy` in backend, nginx, and HTML meta tag.
3. Keep `report-uri` active for ongoing visibility.

### Fail-first / TDD applicability

Limited — CSP is a runtime header, not a code path easily unit-tested. The closest fail-first equivalent is a **Playwright E2E test** that:
1. Loads the production build.
2. Listens for `securitypolicyviolation` console events.
3. Fails if any violation fires.

This test should be written in Phase 1 (will fail because backend still emits `unsafe-inline`), then pass after Phase 4. See acceptance tests §5.

---

## 4. Files touched

| Path                                                                     | Change                                                       |
| ------------------------------------------------------------------------ | ------------------------------------------------------------ |
| `backend/security/security_config.py`                                    | Add nonce middleware; remove `unsafe-inline`/`unsafe-eval`   |
| `backend/security/csp_report.py` (NEW)                                   | `/csp-report` endpoint logging violations                    |
| `frontend/web/index.html`                                                | Add nonce placeholder; remove inline script bodies if any    |
| `frontend/web/vite.config.ts`                                            | Split dev/prod CSP via `transformIndexHtml`                  |
| `infrastructure/docker/nginx/conf.d/security-headers.conf`               | Remove duplicate CSP OR add nonce passthrough                |
| `infrastructure/docker/frontend/security-headers.conf`                   | Same change for parity                                       |
| `config/services/nginx/security-headers.conf`                            | Same change for parity                                       |
| `tests/e2e/csp-violations.spec.ts` (NEW)                                 | Playwright test asserting zero CSP violations                |

All paths verified to exist (or new file paths use existing parent dirs).

---

## 5. Acceptance tests

### 5a. Header inspection (F-08-003, F-13-014)

```bash
# Should print CSP header WITHOUT 'unsafe-inline' or 'unsafe-eval' in script-src
curl -sI https://prod.example.com/ | grep -i 'content-security-policy' \
  | grep -vE "'unsafe-inline'|'unsafe-eval'" \
  && echo "PASS" || echo "FAIL: unsafe directives still present"
```

### 5b. Nonce presence

```bash
# Backend must inject a unique nonce per response
curl -sI https://prod.example.com/ | grep -oE "'nonce-[A-Za-z0-9+/=_-]{16,}'" \
  | wc -l   # expect >=1
```

### 5c. Browser-side (F-12-003)

- Open production app in Chrome DevTools → Console.
- Hard reload, navigate through all main routes (login, dashboard, portfolio, settings).
- **PASS condition:** zero `Refused to execute inline script because it violates the following Content Security Policy directive` messages.
- **PASS condition:** Lighthouse "Best Practices" audit reports CSP present and not using `unsafe-*`.

### 5d. E2E regression

```bash
pnpm --filter web test:e2e tests/e2e/csp-violations.spec.ts
# Should pass after Phase 4; fails before.
```

### 5e. Header de-duplication

```bash
# After Phase 3, only ONE Content-Security-Policy header should be sent
curl -sI https://prod.example.com/ | grep -ci 'content-security-policy'
# expect: 1 (not 2)
```

---

## 6. Rollback plan

CSP rollout is staged precisely so rollback is granular:

1. **During Phase 1–3 (Report-Only):** No user-visible impact possible; "rollback" = revert the report-only header config. Zero risk.
2. **At Phase 4 flip (enforcing):** If violations spike or users report broken UI:
   - **Immediate:** revert nginx config to re-add `'unsafe-inline' 'unsafe-eval'` in `script-src` (single-file edit, `nginx -s reload`, ~30s recovery).
   - **Short-term:** flip header name back to `Content-Security-Policy-Report-Only` while diagnosing.
   - Backend rollback: `git revert` the nonce middleware commit; previous static-string CSP returns.
3. **Database/state impact:** none — CSP is stateless header configuration.
4. **Feature-flag option:** wrap the nonce middleware in `if settings.CSP_ENFORCING:` so flip can be done via env var without redeploy.

---

## 7. Dependencies

- **Soft-depends on Cluster B (auth stabilization):** CSP changes are safer to ship after auth flows are stable, because login/signup are the most CSP-violation-prone pages (third-party SDKs, OAuth redirects, captcha widgets). Doing CSP first risks attributing CSP-induced login failures to auth bugs.
- **Hard-depends on:** none. This cluster can ship independently if Cluster B is delayed, but recommend sequence A → B → C.
- **Coordinates with:** any active work touching `backend/security/security_config.py`, `frontend/web/vite.config.ts`, or any nginx security headers conf — merge conflicts likely.
- **External:** if Sentry, analytics, or any third-party JS is loaded, their domains must be enumerated in `script-src`/`connect-src` *before* enforcing — easy to miss.

---

## 8. Effort & cost

| Phase                                  | Hours  |
| -------------------------------------- | ------ |
| Phase 0 — nonce strategy decision      | 0.5    |
| Phase 1 — backend nonce middleware     | 4.0    |
| Phase 2 — Vite + index.html            | 4.0    |
| Phase 3 — nginx alignment              | 2.0    |
| Phase 4 — flip + monitoring window     | 1.5 (active) + 7 days (passive) |
| E2E test authoring                     | 2.0    |
| **Total active engineering**           | **~14 hours** |

Audit-quoted total was 16h (6+6+4). Synthesis estimate is slightly lower because the three findings share the nonce infrastructure once built — cluster economy saves ~2h vs. fixing in isolation.

**Cost (engineer @ $150/hr loaded):** ~$2,100 active. Plus 7-day report-only soak (negligible engineering time, just monitoring).

---

## 9. Loki-actionable status

**`partial` — `requires_human_ack: true`**

Reasons this is not fully Loki-actionable:

1. **§2 Decision required:** Choose nonce vs. hash strategy.
   - **Default:** per-request nonce (recommended for SPAs).
   - **Alternative:** SHA-256 hashes of known inline script bodies (more brittle; breaks on every Vite build hash change). Rejected unless human overrides.
2. **§2 Decision required:** Should nginx own CSP or pass through from backend?
   - **Default:** backend owns CSP (only it can sync nonce with rendered HTML); nginx passes through.
   - **Alternative:** nginx owns CSP using `ngx_http_set_misc_module` for nonce. Requires confirming the module is loaded in the nginx image. Rejected unless ops confirms.
3. **§2 Decision required:** Acceptable rollout window for the 7-day report-only soak.
   - Loki cannot autonomously decide when to flip from report-only to enforcing — this needs a human-ack'd traffic window (e.g., "flip after Tuesday's market close").
4. **§2 Decision required:** Third-party domain allowlist completeness.
   - Loki can extract domains from current `script-src`/`connect-src` strings and the codebase, but cannot guarantee no runtime-loaded SDK is missed without production traffic observation.

**What IS Loki-actionable now (could ship today):**
- Backend nonce middleware scaffold (Phase 1 code).
- Vite config split for dev vs. prod CSP (Phase 2 code).
- Removal of duplicate CSP across the three nginx files (Phase 3 code, in report-only mode).
- E2E test scaffold (§5d).

**What is NOT Loki-actionable (gated on human ack):**
- The Phase 4 flip itself.
- Final third-party domain allowlist sign-off.

---

## 10. Risks

| Risk                                                                                                  | Likelihood | Impact   | Mitigation                                                                                              |
| ----------------------------------------------------------------------------------------------------- | ---------- | -------- | ------------------------------------------------------------------------------------------------------- |
| Strict CSP white-screens the SPA for users (missed inline script or third-party SDK)                  | High       | Critical | **Mandatory** report-only phase ≥7 days before enforcing                                                |
| Duplicate CSP from backend + nginx confuses browser (most-restrictive wins, surprising failures)      | Medium     | High     | Phase 3 collapses to single source of truth (backend)                                                   |
| Nonce desync between rendered HTML and response header (race condition in middleware)                 | Low        | High     | Generate nonce in single ASGI middleware, attach to `request.state` *before* template render           |
| Vite dev HMR breaks under strict CSP                                                                  | Medium     | Low      | Split dev/prod CSP; dev keeps `unsafe-eval` and `unsafe-inline` (acceptable on localhost only)          |
| Third-party analytics/Sentry blocked silently (no errors, just missing telemetry)                     | Medium     | Medium   | Monitor `report-uri` endpoint during soak; cross-check with vendor dashboards for delivery drop        |
| OAuth redirect flows break (provider injects inline scripts on callback page)                         | Medium     | High     | Test all auth flows in report-only mode; coordinates with Cluster B sequencing                         |
| `frame-ancestors` only enforced via header (not meta tag) — defense-in-depth gap if header is dropped | Low        | Medium   | Keep `frame-ancestors 'none'` in both nginx AND backend headers (redundant by design)                  |
| Browser cache serves old strict CSP after rollback                                                    | Low        | Low      | CSP is per-response; no cache concern for header itself. HTML files with cached `<meta>` tags may persist for short TTL — set `Cache-Control: no-cache` on `index.html`. |
| nginx config divergence (3 files) regresses after fix                                                 | Medium     | Medium   | Collapse to single included file; add lint check in CI                                                  |

---

**End of workpaper. All 3 findings (F-08-003, F-12-003, F-13-014) addressed.**
