# Platform Status — Single Source of Truth

**Version: 1.0.0** · **Last Updated: 2026-08-11**

> **Last updated:** 2026-08-12 (originally authored 2026-06-23 on `loki/state-remediation-2026-06`; salvaged to main via the 2026-08 U1 branch-salvage)
> **Honest summary:** Advanced-beta. Not production-ready. A 2026-04 audit found 48 criticals (open items remain); a 2026-06 state analysis scored overall ~65% across 8 dimensions; a 2026-08 audit (19 scopes, 325 fresh findings) drives the current remediation cycle — see `docs/audits/2026-08/`.

---

## Current Milestone: 2026-08 Audit Remediation (active)

The 2026-06 tiers below are historical record; their salvageable work landed
via the U1 branch-salvage PRs. Current work is organized as clusters C1–C11
per `docs/audits/2026-08/EXECUTIVE_SUMMARY.md`.

| Cluster | Name | Status (2026-08-12) |
|---------|------|---------------------|
| C1 — Branch salvage (U1) | 2026-06 branch cherry-picked as PRs #260/#261/#262 | PRs open, CI green |
| C2 — Secret rotation | Scanner truth + U2 window runbook (PR #263) | Prep open; **rotation itself pending the operator's maintenance window** |
| C3 — CI gates that can fail | 16 findings (PR #264) | PR open, CI green |
| C4 — Deploy-path repair | nginx/compose/image-lineage (PR #266) | PR open, CI green |
| C5–C11 | Fork bridge (P1=real bridge), auth/realtime, admin truth, DB bootstrap (U3), observability, ML (P2=train real), docs/test tail | Not started |

*(2026-06 tiers, for the record: T0/T1 landed via #247/#248 + convergent
fixes; T2/T4 salvage landed in PRs #260–#262; T3 folded into C4/C6.)*

**Outstanding blockers before production:**
1. Secrets rotation + post-window hygiene (#219 + wider 2026-08 inventory; runbook §12) — scheduled, operator-executed
2. Migration architecture gap (D7/U3 — no initial-schema revision; C8)
3. Regulatory documents remain DRAFT pending counsel (Form ADV memo is HUMAN-BLOCKING)
4. ML models have no real trained weights (P2 decision 2026-08-11: train a real model; acceptance metrics TBD with owner)

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

## Dimension Scores (2026-06-23 State Analysis — historical snapshot, pre-2026-08 remediation)

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
