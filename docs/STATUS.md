# Platform Status — Single Source of Truth

> **Last updated:** 2026-06-23
> **Branch at update:** `loki/state-remediation-2026-06`
> **Honest summary:** Advanced-beta. Not production-ready. A 2026-04 audit found 48 criticals (open items remain). A 2026-06 state analysis scored overall ~65% across 8 dimensions.

---

## Current Milestone: Loki State Remediation (active)

| Tier | Name | Status |
|------|------|--------|
| T0 — Deploy-blockers | Dev-Dockerfile in prod (T0.1 ✅ #247), crash-looping cost_monitor (T0.3 ✅ #248), alembic drift (#216) | T0.1 + T0.3 done; #216 queued |
| T1 — ML correctness | Random-init weights served silently at inference | T1.1–T1.4 done on branch |
| T2 — Docs truth | Lying "100% complete" docs archived; single status entry point | This file + T2.2/T2.3 in progress |
| T3 — Hygiene | Dockerfile/compose/ModelManager sprawl | Queued |
| T4 — Regulatory scaffolding | Form ADV, Privacy Policy, ToS drafts (human-gated) | Queued |
| SECRETS (#1) | Live secrets in committed `.env*` files | Owner: Devin (out-of-band; rotation required) |

**Outstanding blockers before production:**
1. Secrets rotation + git-history purge (#219)
2. Migration architecture gap (FINDING-T0.4; no zero-downtime envelope)
3. Regulatory documents (Form ADV / Privacy Policy / ToS)
4. ML model weights loaded at inference (T1 fixes guard the paths; real weights deployment TBD)

---

## Authoritative Inputs (read these, not the archived docs)

| Document | What it is |
|----------|------------|
| `README.md` | Quick-start, architecture, tech stack (Status line updated 2026-06) |
| `docs/CODEMAPS/README.md` | Index of per-layer codemaps |
| `docs/CODEMAPS/BACKEND.md` | Backend services/router map |
| `docs/CODEMAPS/FRONTEND.md` | Frontend components/pages map |
| `docs/CODEMAPS/DATA_FLOW.md` | ETL and data pipeline map |
| `docs/CODEMAPS/INFRASTRUCTURE.md` | Infra/docker/monitoring map |
| `docs/CODEMAPS/ARCHITECTURE.md` | High-level architecture overview |
| `docs/audits/2026-06/SESSION_HANDOFF_2026-06-23.md` | 8-dimension state analysis (honest scores per layer) |
| `docs/audits/2026-06/PRD-loki-state-remediation-2026-06.md` | Current work program (Tiers T0–T4) |
| `docs/audits/2026-06/FINDING-T0.4-migration-architecture-gap.md` | Open finding: migration strategy gap |
| `docs/audits/2026-04/` | Prior 48-critical audit (partially remediated; see 2026-06 analysis for delta) |

**Do not use** any doc under `docs/_superseded/` or `docs/_process/` for current state.
See `docs/SUPERSEDED.md` for the index of stale archived docs.

---

## Dimension Scores (2026-06-23 State Analysis)

| Dimension | Score | One-line read |
|-----------|-------|---------------|
| Backend architecture & API | **82%** | Real layered FastAPI, 18 routers, 19 services, 30-table ORM. README is honest. |
| Frontend | **82%** | Production-grade React/Vite SPA with real backend + websocket wiring. Alerts is local-state-only; ML/trading API clients have no UI. |
| Testing & quality | **72%** | ~5,298 real tests. Gates are soft: lint/mypy non-blocking, 8 big test files `--ignore`d, `xfail strict` off. |
| Security & compliance | **72%** | Real RS256 JWT + full GDPR + SEC retention. Live secrets committed; no Form ADV / Privacy / ToS; HS256 fallback survives. |
| Infra / DevOps | **68%** | Broad real CI/monitoring/deploy. Two concrete deploy bugs fixed (T0.1/T0.3). <$50/mo claim is aspirational. |
| Roadmap & open work | **55%** | Prior top-level docs claimed 97–100%. This file is the honest roadmap. |
| ML / Data / ETL | **45%** | Excellent data/ETL pipes; models served random-init weights (T1 guards added). "Great pipes, hollow endpoints." |
| Documentation | **45%** | Small accurate core (README + CODEMAPS) buried under stale sprawl (now archived). |

---

## What "advanced-beta" means here

- Core plumbing is real and substantial (API, ETL, frontend, auth, monitoring).
- Terminal deliverables are hollow or broken (ML weights, regulatory docs, migration envelope).
- CI gates are green but not stringent (mypy/lint non-blocking; coverage floor 60%).
- A clean production release requires completing at minimum: T0 remainder, secrets rotation, and regulatory scaffolding.

---

*This file is maintained by the loki remediation program. Update it when milestones close or new findings land — not from memory.*
