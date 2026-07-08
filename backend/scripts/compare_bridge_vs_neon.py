#!/usr/bin/env python3
"""Read-only divergence report: broker-stated positions vs the Neon FIFO ledger.

Precedence rule
---------------
Neon FIFO is authoritative for tax. The bridge reflects what the broker states
right now. Divergence is expected (unsettled trades, corporate actions,
per-broker cost-basis conventions) and is NOT automatically an error.

This script never writes. It does not mutate the IAP portfolio, and it opens
the Neon connection read-only where the driver supports it. Its only output is
a report.

DSN resolution
--------------
Environment variables ONLY: ``NEON_DSN`` or ``TAX_ADVISOR_NEON_DSN``, exactly
matching ``sync_portfolio_from_neon.py``. There is deliberately NO
``.env.local`` (or any other file) fallback: a script that silently harvests a
sister project's secret file makes the credential's blast radius invisible at
the call site. If neither variable is set, the script prints how to set them
and exits 3.

Verification note
-----------------
This is designed to be exercised without Neon credentials: every unit test
mocks the database entirely. Running it against the real Neon ``tax_advisor``
ledger is a separate, human-performed verification step.

Usage:
    NEON_DSN=postgresql://... FINANCE_DATA_DIR=... \
    python backend/scripts/compare_bridge_vs_neon.py [--json]

Exit codes:
    0  report produced (divergence, if any, is reported not failed on)
    2  bridge snapshot unreadable
    3  no DSN in the environment
"""

from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sync_portfolio_from_bridge import (  # noqa: E402
    EXIT_UNUSABLE_SNAPSHOT,
    aggregate_desired_positions,
    build_account_source_map,
    load_snapshot,
    ok_sources,
    resolve_data_dir,
)
from sync_portfolio_from_neon import FILL_SHAPED_SOURCES  # noqa: E402

__all__ = [
    "BASIS_TOLERANCE",
    "build_report",
    "derive_neon_fifo",
    "resolve_dsn",
]

BASIS_TOLERANCE = Decimal("0.01")
EXIT_OK = 0
EXIT_NO_DSN = 3

_DSN_ENV_VARS = ("NEON_DSN", "TAX_ADVISOR_NEON_DSN")
_DSN_HELP = """No Neon DSN found in the environment.

Set exactly one of:
    NEON_DSN=postgresql://user:pass@host/db
    TAX_ADVISOR_NEON_DSN=postgresql://user:pass@host/db

This script reads the DSN from environment variables only. It will not read
.env.local or any other secret file."""


def resolve_dsn(env: Dict[str, str]) -> Optional[str]:
    """Return the first DSN present in the environment, else ``None``."""
    for name in _DSN_ENV_VARS:
        value = env.get(name)
        if value:
            return value
    return None


def derive_neon_fifo(rows: List[Tuple[str, str, Any]]) -> Dict[str, Dict[str, Decimal]]:
    """FIFO-derive open quantity and EXACT total remaining basis per symbol.

    Kept in parity with ``sync_portfolio_from_neon.derive_positions`` -- the
    unit tests assert the quantities agree for the same rows. This function
    exists separately because it reports *total* basis, whereas
    ``derive_positions`` reports a cent-quantized *average* cost; multiplying
    that average back out would introduce rounding noise larger than the
    $0.01 divergence threshold used here.
    """
    lots: Dict[Tuple[str, str], List[List[Any]]] = {}
    for account_id, _txn_date, raw in rows:
        fill = raw if isinstance(raw, dict) else json.loads(raw)
        open_lots = lots.setdefault((account_id, fill["symbol"]), [])
        qty = int(fill["qty"])
        if fill["side"] == "buy":
            open_lots.append([qty, Decimal(str(fill["price"]))])
            continue
        remaining = qty
        while remaining > 0 and open_lots:
            lot = open_lots[0]
            take = min(lot[0], remaining)
            lot[0] -= take
            remaining -= take
            if lot[0] == 0:
                open_lots.pop(0)

    by_symbol: Dict[str, Dict[str, Decimal]] = {}
    for (_account, symbol), open_lots in lots.items():
        qty = sum((Decimal(lot_qty) for lot_qty, _ in open_lots), Decimal("0"))
        if qty <= 0:
            continue
        basis = sum(
            (Decimal(lot_qty) * price for lot_qty, price in open_lots), Decimal("0")
        )
        entry = by_symbol.setdefault(symbol, {"qty": Decimal("0"), "basis": Decimal("0")})
        entry["qty"] += qty
        entry["basis"] += basis
    return by_symbol


def _bridge_basis(position: Dict[str, Any]) -> Decimal:
    return Decimal(position["qty"]) * Decimal(position["avg_cost"])


def build_report(
    *,
    neon: Dict[str, Dict[str, Decimal]],
    bridge: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Per-symbol divergence rows, sorted by symbol.

    A row is flagged when quantity differs at all, or basis differs by more
    than ``BASIS_TOLERANCE``.
    """
    rows: List[Dict[str, Any]] = []
    for symbol in sorted(set(neon) | set(bridge)):
        neon_entry = neon.get(symbol)
        bridge_entry = bridge.get(symbol)

        neon_qty = neon_entry["qty"] if neon_entry else Decimal("0")
        neon_basis = neon_entry["basis"] if neon_entry else Decimal("0")
        bridge_qty = Decimal(bridge_entry["qty"]) if bridge_entry else Decimal("0")
        bridge_basis = _bridge_basis(bridge_entry) if bridge_entry else Decimal("0")

        qty_delta = bridge_qty - neon_qty
        basis_delta = bridge_basis - neon_basis
        rows.append(
            {
                "symbol": symbol,
                "neon_fifo_qty": neon_qty,
                "bridge_qty": bridge_qty,
                "delta": qty_delta,
                "neon_fifo_basis": neon_basis,
                "bridge_basis": bridge_basis,
                "basis_delta": basis_delta,
                "flagged": qty_delta != 0 or abs(basis_delta) > BASIS_TOLERANCE,
            }
        )
    return rows


def _fetch_neon_rows(dsn: str) -> List[Tuple[str, str, Any]]:
    """Read-only fetch of fill-shaped transactions."""
    import psycopg

    with psycopg.connect(dsn) as conn:
        # Best-effort: psycopg3 exposes a session-level read-only switch.
        try:
            conn.read_only = True
        except (AttributeError, TypeError):  # pragma: no cover - driver dependent
            pass
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT t.account_id::text, t.txn_date::text, t.raw
                  FROM tax_advisor.transactions t
                 WHERE t.source = ANY(%s)
                 ORDER BY t.account_id, t.txn_date, t.created_at
                """,
                (list(FILL_SHAPED_SOURCES),),
            )
            return cur.fetchall()


def _print_table(rows: List[Dict[str, Any]]) -> None:
    header = (
        f"{'symbol':<8} {'neon_fifo_qty':>14} {'bridge_qty':>12} {'delta':>12} "
        f"{'neon_fifo_basis':>17} {'bridge_basis':>14} {'basis_delta':>13}  flag"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        flag = "DIVERGENT" if row["flagged"] else ""
        print(
            f"{row['symbol']:<8} {row['neon_fifo_qty']:>14} {row['bridge_qty']:>12} "
            f"{row['delta']:>12} {row['neon_fifo_basis']:>17} "
            f"{row['bridge_basis']:>14} {row['basis_delta']:>13}  {flag}"
        )
    flagged = sum(1 for row in rows if row["flagged"])
    print(
        f"\n{len(rows)} symbol(s), {flagged} divergent. "
        "Neon FIFO is authoritative for tax; divergence is expected and is "
        "not automatically an error."
    )


def _jsonable(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            key: (str(value) if isinstance(value, Decimal) else value)
            for key, value in row.items()
        }
        for row in rows
    ]


def main(argv: Optional[List[str]] = None, env: Optional[Dict[str, str]] = None) -> int:
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Machine-readable output.")
    args = parser.parse_args(argv)

    dsn = resolve_dsn(dict(os.environ) if env is None else env)
    if not dsn:
        print(_DSN_HELP, file=sys.stderr)
        return EXIT_NO_DSN

    data_dir = resolve_data_dir()
    try:
        snapshot = load_snapshot(data_dir)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"cannot read {data_dir / 'latest.json'}: {exc}", file=sys.stderr)
        return EXIT_UNUSABLE_SNAPSHOT

    usable, _skipped = ok_sources(snapshot.get("sources", {}))
    bridge = aggregate_desired_positions(
        snapshot.get("positions", []),
        usable_sources=usable,
        account_source=build_account_source_map(snapshot.get("accounts", [])),
    )
    neon = derive_neon_fifo(_fetch_neon_rows(dsn))
    rows = build_report(neon=neon, bridge=bridge)

    if args.json:
        print(json.dumps({"rows": _jsonable(rows)}, indent=2))
    else:
        _print_table(rows)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
