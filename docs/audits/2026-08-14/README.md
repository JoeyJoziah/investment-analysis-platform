# Six-project audit — 2026-08-14

Copied out of the Grok session directory so the work is not session-scoped.

## Scope

In: `investment-analysis-platform`, `portfolio-bridge`, `msos-options-monitor`, `wheel-analytics`, `market-intel`, `efinancialmodels-workshop`.

Out: `tax-prep-2025`, `thesis-monitor`. `.env` bodies and live DBs were not opened.

## Artifacts in this folder

| File | What it is |
|---|---|
| `AUDIT_CONTEXT.md` | Trail-of-Bits style context (purpose, inputs, 5 Whys, invariants) |
| `AUDIT_FINDINGS.md` | Scored findings IAP-001–012, MSOS-001, BRIDGE-001, plus remediations |
| `plan.md` | Approved hunt/fix plan |
| `SHIP.md` | What shipped, SHAs, residuals, how to resume |

## Product remediations (already on default branches)

| Repo | SHA | Remote |
|---|---|---|
| IAP | `06ef192` | https://github.com/JoeyJoziah/investment-analysis-platform |
| portfolio-bridge | `46403d4` (also pushed prior `f7789ac`) | https://github.com/JoeyJoziah/portfolio-bridge |
| msos-options-monitor | `358bec9` + `4e32038` | https://github.com/JoeyJoziah/msos-options-monitor (private; created this day) |

`wheel-analytics`, `market-intel`, and `efinancialmodels-workshop` had no scored findings and no product edits from this session.

## Breaking / deploy notes

- IAP `/refresh` now requires JSON `{ "refresh_token": "..." }`. Login/register return `refresh_token`.
- Set `ML_API_TOKEN` before starting the ML API. Compose publishes `127.0.0.1:8001` only.
- Set `METRICS_SCRAPE_TOKEN` if Prometheus should scrape without an admin JWT.
- Local `.env` with `ENVIRONMENT=development` still skips CSRF and HTTP rate-limit.
