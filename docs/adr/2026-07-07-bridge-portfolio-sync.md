# Bridge-backed portfolio sync and Neon divergence reporting

- **Date:** 2026-07-07
- **Status:** Accepted
- **Supersedes:** nothing
- **Related:** `tax-advisor/docs/adr/2026-06-11-cross-repo-finance-integration.md` (Phase 2, `sync_portfolio_from_neon.py`)

## Context

### The MCP visibility problem

Every brokerage and bank is now connected through MCP servers: Webull and
Robinhood (official, trading-capable), SnapTrade (Schwab / Fidelity / E*Trade /
eToro, read-only), and a Kubera connector for banks. That connectivity is real,
but it is **only reachable from inside a Claude session**. MCP tools are not
callable from cron, from CI, from Airflow, or from a plain `python script.py`
invocation. Any batch process that wants broker-stated positions therefore
cannot fetch them itself.

### Hub-and-spoke

`portfolio-bridge` resolves this with a hub-and-spoke arrangement:

- **Hub (write side):** a Claude session holding the MCP tools fetches each
  broker, normalizes the result, and writes it into a local SQLite database plus
  a denormalized `latest.json` snapshot under `FINANCE_DATA_DIR` (default
  `C:\Users\Devin McGrathj\finance-data`).
- **Spokes (read side):** ordinary scripts — this repo included, alongside
  `financial-skills`, `wheel-analytics`, and TradingAgents — read `latest.json`
  and never talk to a broker.

The snapshot carries a per-source status block, so a broker that failed to
refresh is *first-class data* rather than a crashed run:

```json
"sources": {
  "webull":    {"status": "ok",      "error": null, "fetched_at": "..."},
  "ibkr":      {"status": "missing", "error": null, "fetched_at": null},
  "kubera":    {"status": "error",   "error": "auth expired", "fetched_at": null}
}
```

`status` is one of `ok | partial | error | missing`. All numerics in the
snapshot are stored as **strings**, precisely so consumers are forced to choose
their own numeric type rather than inheriting SQLite's float coercion.

### The two-source problem

This repo already reconciles its portfolio against Neon
(`sync_portfolio_from_neon.py`), which derives open positions FIFO from
broker-direct fill transactions. Now a second source of position truth exists.
They will not agree, and it is important to be explicit about why that is fine.

## Decision

### 1. Two scripts, one diff engine

`backend/scripts/sync_portfolio_from_bridge.py` reconciles the IAP portfolio
against broker-stated positions. It **imports** `compute_sync_actions` from
`sync_portfolio_from_neon.py` rather than copying it. That function was already
written generically over `current`/`desired` dictionaries and returns only
add/remove deltas, which is what makes both syncs idempotent — re-running a
converged portfolio produces an empty plan.

`sync_portfolio_from_neon.py` already guarded its script body under
`if __name__ == "__main__":`, so the import is side-effect free and **no edit to
any existing file was required**. Both entrypoints are consequently guaranteed
to produce identical diffs for identical inputs, forever, by construction rather
than by discipline.

The one thing the bridge sync does *not* reuse is the REST serialization path:
bridge quantities can be fractional `Decimal`s (fractional shares), whereas the
Neon ledger deals in whole-share integer fills. Serialization differs;
the diff does not.

### 2. Neon FIFO is authoritative for tax

The precedence rule, stated verbatim in `compare_bridge_vs_neon.py`'s module
docstring:

> Neon FIFO is authoritative for tax. The bridge reflects what the broker states
> right now. Divergence is expected (unsettled trades, corporate actions,
> per-broker cost-basis conventions) and is NOT automatically an error.

This is the crux of the design. The two sources answer *different questions*:

| | Neon FIFO ledger | Bridge snapshot |
|---|---|---|
| Question answered | "What is my tax basis?" | "What does the broker say I hold?" |
| Derivation | FIFO lot consumption over the full fill history | Whatever the broker's API returned |
| Settlement | Trade date | Often settlement-lagged |
| Corporate actions | Reflected only if a fill row exists | Reflected immediately by the broker |
| Cost basis convention | Strict FIFO | Broker-specific (avg cost, HIFO, etc.) |

A T+1 unsettled trade legitimately shows in one and not the other. A stock split
shows in the broker's view before any fill row exists to describe it. Schwab may
report average cost where our ledger computes FIFO. **None of these are bugs.**

Therefore `compare_bridge_vs_neon.py` is a *report*, not a *check*. It exits 0
even when every row diverges. It flags rows (quantity differing at all; basis
differing by more than $0.01) for a human to interpret. Making it fail the build
on divergence would be actively wrong: it would train us to suppress a signal
that is usually benign and occasionally the only warning of a real problem, like
a missed corporate action.

`compare_bridge_vs_neon.py` never writes. It does not mutate the portfolio, it
does not write Neon, and it sets `conn.read_only = True` where psycopg supports
it — defense in depth, since the SQL is already a bare `SELECT`.

### 3. DSN from environment variables only

DSN resolution is `NEON_DSN or TAX_ADVISOR_NEON_DSN`, read from the process
environment. There is **deliberately no `.env.local` fallback, nor any other
file fallback.** If neither variable is set the script prints how to set them
and exits 3.

The sibling `wheel-analytics` project's `dsn.py` reads a *different* project's
secret file off disk. We explicitly did not import, copy, or imitate that
behavior. A script that silently harvests credentials from a sister repository
makes the blast radius of that credential invisible at the call site: nothing in
`compare_bridge_vs_neon.py`'s invocation would reveal that running it grants it
tax-advisor's database password. Environment variables keep the grant explicit
and auditable — the caller decides, per invocation, what the script may reach.

### 4. Safety posture of the write path

- **Dry-run is the default.** Writing requires an explicit `--apply`. The plan
  is always printed before it is applied; nothing is ever silently overwritten.
- **Freshness gate.** A snapshot whose `as_of` is older than `--max-age-hours`
  (default 24) exits 2. Stale broker data driving portfolio writes is worse than
  no data. `--force` bypasses it, loudly.
- **Per-source status gate.** Positions from any source not reporting `status:
  ok` are skipped, and each skip is printed with its reason. If *zero* sources
  are `ok`, the script exits 2 rather than concluding the portfolio should be
  emptied. This is the single most important gate: a naive implementation that
  read a snapshot in which every broker had failed would compute a `desired` set
  of `{}` and dutifully liquidate the entire recorded portfolio.
- **Account identifiers are masked** to their last four characters in all
  output.
- **Equity and ETF rows only.** `option`, `cash`, and `crypto` rows are skipped.
  An IAP portfolio position models share exposure; option contract counts are not
  comparable to share counts, and summing them would silently corrupt exposure
  math.

### 5. Money is `Decimal`

Every monetary and quantity value is parsed from the snapshot's strings straight
into `Decimal` and stays there until the JSON boundary. Never `float`. Basis is
summed in `Decimal` before any division, and average cost is quantized to cents
exactly once, with `ROUND_HALF_UP`.

This matters concretely for the divergence report. `derive_neon_fifo` computes
*exact total remaining basis* rather than reusing `derive_positions`' cent-
quantized *average* cost, because multiplying a rounded average back out by the
share count reintroduces rounding noise larger than the $0.01 threshold the
report flags on — the tool would manufacture the very divergence it exists to
detect. The two functions are held in quantity parity by a unit test.

## Consequences

### Positive

- Batch and CI processes can consume broker-stated positions without MCP.
- One diff engine, imported, means the two syncs cannot drift apart.
- A broker outage degrades to "that source is skipped," not a crash and not a
  spurious liquidation.
- Divergence between tax basis and broker state becomes visible and inspectable
  instead of being quietly reconciled away.
- Env-var-only DSN keeps credential scope explicit at every call site.

### Negative / accepted trade-offs

- **The snapshot can be stale.** Nothing refreshes it except a human running a
  Claude session. The freshness gate converts this from a silent correctness bug
  into a loud failure, but it does not solve it.
- **`compare_bridge_vs_neon.py` cannot be verified in CI**, because CI has no
  Neon credentials — by design, per the env-var-only decision. Every unit test
  mocks the database completely (no network, no psycopg connection, no real
  DSN). Running the comparison against the live `tax_advisor` ledger is a
  **separate, human-performed verification step**, and the module docstring says
  so.
- **Options are invisible to the IAP portfolio.** Wheel strategy exposure lives
  in `wheel-analytics`, which consumes a different bridge export. Anyone reading
  the IAP portfolio as a complete picture of risk will be wrong.
- **Two FIFO implementations exist** (`derive_positions` for average cost,
  `derive_neon_fifo` for exact total basis). This is a real duplication cost,
  accepted because collapsing them would require editing the blessed neon script.
  A parity test pins their quantities together; if they ever disagree, that test
  fails.
- Cross-repo coupling to `portfolio-bridge`'s `latest.json` shape is now load-
  bearing. That shape is a published contract (see `export.py`'s module
  docstring); changing it requires updating every consumer.

## Contract notes

Observed in `portfolio-bridge/src/bridge/{db.py,schema.sql,export.py}`, beyond
what was specified when this work began:

- `latest.json` additionally contains top-level `run_id` and `activities` keys.
- Position rows additionally carry `sync_run_id`, `as_of`, `unrealized_pnl`, and
  `currency`.
- Position rows **do not carry a `source` field.** Source must be resolved by
  joining `account_id` against the `accounts[]` array, whose rows do carry
  `source`. The namespaced `account_id` prefix (`"snaptrade:abc-123"`) is used as
  a fallback when no matching account row is present.
- `is_short` is an INTEGER (`0`/`1`), not a string, unlike every other numeric
  in the positions table.
