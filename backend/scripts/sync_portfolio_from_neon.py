#!/usr/bin/env python3
"""Sync real holdings from the Neon tax_advisor ledger into an IAP portfolio.

Phase 2 of the cross-repo finance integration ADR
(tax-advisor/docs/adr/2026-06-11-cross-repo-finance-integration.md).

Reads broker-direct fill transactions (source='alpaca_phf', exported by
tax-advisor/scripts/export_phf_ledger.py) from Neon, derives open positions
FIFO, then reconciles the target IAP portfolio through the public REST API:
only deltas are applied (add/remove), so re-runs are no-ops when nothing
changed.

The FIFO derivation must stay in algorithmic parity with
financial-mcp-server/portfolio_queries.py (the MCP query plane); both are
covered by equivalent tests.

Usage:
    NEON_DSN=postgresql://... IAP_BASE_URL=http://localhost:8000 \
    IAP_USERNAME=... IAP_PASSWORD=... \
    python backend/scripts/sync_portfolio_from_neon.py --portfolio-id <uuid> [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Dict, List, Tuple

FILL_SHAPED_SOURCES = ("alpaca_phf",)
_CENT = Decimal("0.01")

Row = Tuple[str, str, Any]  # (account_id, txn_date, raw fill payload)


def derive_positions(rows: List[Row]) -> Dict[str, Dict[str, Any]]:
    """Aggregate open positions per symbol from chronological fill rows.

    FIFO lot consumption per (account, symbol); the result is aggregated
    across accounts because an IAP portfolio models one consolidated view.
    Returns {symbol: {qty, avg_cost}} for symbols with qty > 0.
    """
    lots: Dict[Tuple[str, str], List[List]] = {}
    for account_id, _txn_date, raw in rows:
        fill = raw if isinstance(raw, dict) else json.loads(raw)
        key = (account_id, fill["symbol"])
        open_lots = lots.setdefault(key, [])
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

    by_symbol: Dict[str, Dict[str, Any]] = {}
    for (_account, symbol), open_lots in lots.items():
        qty = sum(lot_qty for lot_qty, _ in open_lots)
        if qty <= 0:
            continue
        basis = sum(
            (Decimal(lot_qty) * price for lot_qty, price in open_lots),
            Decimal("0"),
        )
        agg = by_symbol.setdefault(symbol, {"qty": 0, "_basis": Decimal("0")})
        agg["qty"] += qty
        agg["_basis"] += basis

    for agg in by_symbol.values():
        agg["avg_cost"] = (agg.pop("_basis") / Decimal(agg["qty"])).quantize(
            _CENT, rounding=ROUND_HALF_UP
        )
    return by_symbol


def compute_sync_actions(
    *, current: Dict[str, int], desired: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Diff current IAP positions against desired ledger-derived positions.

    Only deltas are emitted, which makes the sync idempotent: equal
    positions produce no actions.
    """
    actions: List[Dict[str, Any]] = []
    for symbol in sorted(set(current) | set(desired)):
        have = current.get(symbol, 0)
        want = desired[symbol]["qty"] if symbol in desired else 0
        if want > have:
            actions.append(
                {
                    "action": "add",
                    "symbol": symbol,
                    "qty": want - have,
                    "price": desired[symbol]["avg_cost"],
                }
            )
        elif want < have:
            actions.append(
                {"action": "remove", "symbol": symbol, "qty": have - want}
            )
    return actions


def _fetch_neon_rows(dsn: str) -> List[Row]:
    import psycopg

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
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


def _sync_via_api(
    base_url: str,
    portfolio_id: str,
    actions: List[Dict[str, Any]],
    username: str,
    password: str,
) -> None:
    import httpx

    with httpx.Client(base_url=base_url, timeout=30) as client:
        login = client.post(
            "/api/v1/auth/login",
            data={"username": username, "password": password},
        )
        login.raise_for_status()
        token = login.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        for action in actions:
            if action["action"] == "add":
                resp = client.post(
                    f"/api/v1/portfolio/{portfolio_id}/positions",
                    headers=headers,
                    json={
                        "symbol": action["symbol"],
                        "quantity": action["qty"],
                        "price": float(action["price"]),
                        "transaction_type": "buy",
                        "notes": "sync_portfolio_from_neon (alpaca_phf ledger)",
                    },
                )
            else:
                resp = client.request(
                    "DELETE",
                    f"/api/v1/portfolio/{portfolio_id}/positions/{action['symbol']}",
                    headers=headers,
                    params={"quantity": action["qty"]},
                )
            resp.raise_for_status()
            print(f"applied: {action}")


def _current_positions(
    base_url: str, portfolio_id: str, username: str, password: str
) -> Dict[str, int]:
    import httpx

    with httpx.Client(base_url=base_url, timeout=30) as client:
        login = client.post(
            "/api/v1/auth/login",
            data={"username": username, "password": password},
        )
        login.raise_for_status()
        token = login.json()["access_token"]
        detail = client.get(
            f"/api/v1/portfolio/{portfolio_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        detail.raise_for_status()
        positions = detail.json().get("data", {}).get("positions", [])
        return {p["symbol"]: int(p["quantity"]) for p in positions}


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portfolio-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    dsn = os.environ.get("NEON_DSN") or os.environ.get("TAX_ADVISOR_NEON_DSN")
    if not dsn:
        print("NEON_DSN (or TAX_ADVISOR_NEON_DSN) not set", file=sys.stderr)
        return 2

    rows = _fetch_neon_rows(dsn)
    desired = derive_positions(rows)
    print(f"ledger: {len(rows)} fills -> {len(desired)} open symbols")

    if args.dry_run:
        for symbol, pos in sorted(desired.items()):
            print(f"  {symbol}: {pos['qty']} @ avg {pos['avg_cost']}")
        return 0

    base_url = os.environ["IAP_BASE_URL"]
    username = os.environ["IAP_USERNAME"]
    password = os.environ["IAP_PASSWORD"]

    current = _current_positions(base_url, args.portfolio_id, username, password)
    actions = compute_sync_actions(current=current, desired=desired)
    if not actions:
        print("portfolio already in sync; nothing to do")
        return 0
    _sync_via_api(base_url, args.portfolio_id, actions, username, password)
    print(f"sync complete: {len(actions)} actions applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
