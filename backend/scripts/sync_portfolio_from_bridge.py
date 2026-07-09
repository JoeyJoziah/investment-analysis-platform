#!/usr/bin/env python3
"""Reconcile an IAP portfolio against what the brokers currently state.

Reads the local portfolio-bridge snapshot hub (``latest.json``, written by
Claude sessions that hold the broker MCP tools) and drives the IAP portfolio
toward the broker-stated share positions through the public REST API.

Relationship to ``sync_portfolio_from_neon.py``
-----------------------------------------------
Both scripts reconcile the same IAP portfolio, from different sources of
truth, and both reuse :func:`compute_sync_actions` so the add/remove diff is
identical and idempotent. Neon FIFO is authoritative for tax. The bridge
reflects what the broker states *right now*. Divergence between them is
expected and is NOT automatically an error -- see
``compare_bridge_vs_neon.py`` for the read-only divergence report.

Scope
-----
Only ``equity`` and ``etf`` rows are synced. ``option``, ``cash`` and
``crypto`` rows are skipped: an IAP portfolio position models share exposure,
and option contract counts are not comparable to share counts.

Safety
------
* Dry-run is the default. Writing requires an explicit ``--apply``.
* A freshness gate rejects snapshots older than ``--max-age-hours``
  (default 24) unless ``--force`` is passed.
* Per-source status gate: positions from any source whose ``status`` is not
  ``ok`` are skipped, and the skip is reported.
* Account identifiers are never printed in full; they are masked to the last
  four characters.

Usage:
    FINANCE_DATA_DIR=... IAP_BASE_URL=http://localhost:8000 \
    IAP_USERNAME=... IAP_PASSWORD=... \
    python backend/scripts/sync_portfolio_from_bridge.py \
        --portfolio-id <uuid> [--apply]

Exit codes:
    0  success (in sync, dry-run printed, or actions applied)
    2  snapshot stale, unreadable, or no source reported status=ok
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ``compute_sync_actions`` is IMPORTED, never copied: the diff semantics must
# stay identical across both sync entrypoints. The neon script guards its
# script body under ``if __name__ == "__main__":`` so this import is side
# effect free.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sync_portfolio_from_neon import compute_sync_actions  # noqa: E402

__all__ = [
    "DEFAULT_FINANCE_DATA_DIR",
    "SYNCABLE_INSTRUMENT_TYPES",
    "aggregate_desired_positions",
    "build_account_source_map",
    "is_stale",
    "load_snapshot",
    "mask_account_id",
    "ok_sources",
    "parse_as_of",
    "resolve_data_dir",
]

DEFAULT_FINANCE_DATA_DIR = r"C:\Users\Devin McGrathj\finance-data"
SYNCABLE_INSTRUMENT_TYPES = frozenset({"equity", "etf"})
DEFAULT_MAX_AGE_HOURS = 24
_CENT = Decimal("0.01")

EXIT_OK = 0
EXIT_UNUSABLE_SNAPSHOT = 2


def resolve_data_dir() -> Path:
    """Locate the snapshot hub via ``FINANCE_DATA_DIR`` with a fallback."""
    return Path(os.environ.get("FINANCE_DATA_DIR") or DEFAULT_FINANCE_DATA_DIR)


def mask_account_id(account_id: str) -> str:
    """Redact an account identifier down to its last four characters."""
    if not account_id:
        return "****"
    return f"****{account_id[-4:]}"


def _dec(value: Any) -> Decimal:
    """Parse a bridge TEXT numeric into ``Decimal``. Never uses ``float``."""
    if value is None or value == "":
        return Decimal("0")
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return Decimal("0")


def parse_as_of(raw: str) -> datetime:
    """Parse the snapshot ``as_of`` timestamp into an aware UTC datetime.

    Naive timestamps are interpreted as UTC. A trailing ``Z`` is normalized
    because ``datetime.fromisoformat`` rejects it before Python 3.11.
    """
    normalized = raw.strip()
    if normalized.endswith(("Z", "z")):
        normalized = f"{normalized[:-1]}+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def is_stale(
    as_of: datetime, *, max_age_hours: int, now: Optional[datetime] = None
) -> bool:
    """True when the snapshot is older than the allowed age."""
    reference = now or datetime.now(timezone.utc)
    return (reference - as_of) > timedelta(hours=max_age_hours)


def load_snapshot(data_dir: Path) -> Dict[str, Any]:
    """Read ``latest.json`` from the snapshot hub."""
    path = data_dir / "latest.json"
    with path.open("r", encoding="utf-8") as handle:
        payload: Dict[str, Any] = json.load(handle)
    return payload


def ok_sources(
    sources: Dict[str, Dict[str, Any]],
) -> Tuple[frozenset, List[Tuple[str, str, Optional[str]]]]:
    """Split sources into the usable set and a reportable skip list.

    Returns ``(ok_source_names, [(source, status, error), ...])``.
    """
    usable = set()
    skipped: List[Tuple[str, str, Optional[str]]] = []
    for name, meta in sorted(sources.items()):
        status = (meta or {}).get("status")
        if status == "ok":
            usable.add(name)
        else:
            skipped.append((name, status or "unknown", (meta or {}).get("error")))
    return frozenset(usable), skipped


def build_account_source_map(accounts: List[Dict[str, Any]]) -> Dict[str, str]:
    """Map ``account_id`` -> owning source name.

    The bridge's position rows carry only ``account_id``; the source lives on
    the account row. Account ids are namespaced (``"snaptrade:abc-123"``), so
    the prefix is a safe fallback when an account row is absent.
    """
    return {
        account["account_id"]: account.get("source")
        or account["account_id"].split(":", 1)[0]
        for account in accounts
        if account.get("account_id")
    }


def _source_for(account_id: str, account_source: Dict[str, str]) -> str:
    if account_id in account_source:
        return account_source[account_id]
    return account_id.split(":", 1)[0]


def _position_basis(position: Dict[str, Any], quantity: Decimal) -> Decimal:
    """Total cost basis for a row, preferring the explicit total."""
    total = position.get("cost_basis_total")
    if total not in (None, ""):
        return _dec(total)
    return _dec(position.get("cost_basis_per_unit")) * quantity


def aggregate_desired_positions(
    positions: List[Dict[str, Any]],
    *,
    usable_sources: frozenset,
    account_source: Dict[str, str],
) -> Dict[str, Dict[str, Any]]:
    """Aggregate broker-stated share positions by symbol across accounts.

    Only ``equity``/``etf`` rows from sources whose status is ``ok`` are
    included. Quantities and basis stay in ``Decimal`` throughout; the
    resulting ``avg_cost`` is quantized to cents, matching the shape
    :func:`compute_sync_actions` expects (it reads only ``qty`` and
    ``avg_cost``).

    ``basis`` carries the EXACT unrounded total cost basis alongside the
    quantized ``avg_cost``. Consumers that compare basis against another
    ledger MUST use ``basis``: reconstructing it as ``qty * avg_cost``
    reintroduces up to half a cent of rounding error per share, which for
    any meaningful share count exceeds the $0.01 divergence threshold in
    ``compare_bridge_vs_neon.py`` and manufactures phantom divergences.
    """
    staged: Dict[str, Dict[str, Decimal]] = {}
    for position in positions:
        if position.get("instrument_type") not in SYNCABLE_INSTRUMENT_TYPES:
            continue
        account_id = position.get("account_id") or ""
        if _source_for(account_id, account_source) not in usable_sources:
            continue
        symbol = position.get("symbol")
        if not symbol:
            continue
        quantity = _dec(position.get("quantity"))
        if quantity <= 0:
            continue
        entry = staged.setdefault(
            symbol, {"qty": Decimal("0"), "basis": Decimal("0")}
        )
        entry["qty"] += quantity
        entry["basis"] += _position_basis(position, quantity)

    desired: Dict[str, Dict[str, Any]] = {}
    for symbol, entry in staged.items():
        avg_cost = (entry["basis"] / entry["qty"]).quantize(
            _CENT, rounding=ROUND_HALF_UP
        )
        desired[symbol] = {
            "qty": entry["qty"],
            "avg_cost": avg_cost,
            "basis": entry["basis"],  # exact, unrounded -- see docstring
        }
    return desired


def _json_qty(quantity: Decimal) -> Any:
    """Serialize a Decimal share count for the REST payload without float drift."""
    if quantity == quantity.to_integral_value():
        return int(quantity)
    return float(quantity)


def _apply_actions(
    base_url: str,
    portfolio_id: str,
    actions: List[Dict[str, Any]],
    username: str,
    password: str,
) -> None:
    """Push the add/remove plan through the IAP REST API.

    Serialization (not the diff) is the only thing that differs from the neon
    script: bridge quantities may be fractional Decimals.
    """
    import httpx

    with httpx.Client(base_url=base_url, timeout=30) as client:
        login = client.post(
            "/api/v1/auth/login",
            data={"username": username, "password": password},
        )
        login.raise_for_status()
        headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

        for action in actions:
            if action["action"] == "add":
                response = client.post(
                    f"/api/v1/portfolio/{portfolio_id}/positions",
                    headers=headers,
                    json={
                        "symbol": action["symbol"],
                        "quantity": _json_qty(action["qty"]),
                        "price": float(action["price"]),
                        "transaction_type": "buy",
                        "notes": "sync_portfolio_from_bridge (broker snapshot)",
                    },
                )
            else:
                response = client.request(
                    "DELETE",
                    f"/api/v1/portfolio/{portfolio_id}/positions/{action['symbol']}",
                    headers=headers,
                    params={"quantity": _json_qty(action["qty"])},
                )
            response.raise_for_status()
            print(f"applied: {action['action']} {action['symbol']} {action['qty']}")


def _print_plan(actions: List[Dict[str, Any]]) -> None:
    print(f"plan: {len(actions)} action(s)")
    for action in actions:
        if action["action"] == "add":
            print(f"  add    {action['symbol']:<8} {action['qty']} @ {action['price']}")
        else:
            print(f"  remove {action['symbol']:<8} {action['qty']}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portfolio-id", required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write to the portfolio. Without this, dry-run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Print the plan without writing (default).",
    )
    parser.add_argument("--max-age-hours", type=int, default=DEFAULT_MAX_AGE_HOURS)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass the snapshot freshness gate.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    data_dir = resolve_data_dir()
    try:
        snapshot = load_snapshot(data_dir)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"cannot read {data_dir / 'latest.json'}: {exc}", file=sys.stderr)
        return EXIT_UNUSABLE_SNAPSHOT

    as_of = parse_as_of(snapshot["as_of"])
    if is_stale(as_of, max_age_hours=args.max_age_hours):
        if not args.force:
            print(
                f"snapshot is stale: as_of={as_of.isoformat()} is older than "
                f"{args.max_age_hours}h. Re-run the bridge sync, or pass "
                f"--force to proceed anyway.",
                file=sys.stderr,
            )
            return EXIT_UNUSABLE_SNAPSHOT
        print(f"WARNING: --force set; using stale snapshot as_of={as_of.isoformat()}")

    usable, skipped = ok_sources(snapshot.get("sources", {}))
    for name, status, error in skipped:
        reason = f"{status}: {error}" if error else status
        print(f"skipping source {name!r} ({reason})")
    if not usable:
        print("no source reported status=ok; nothing trustworthy to sync",
              file=sys.stderr)
        return EXIT_UNUSABLE_SNAPSHOT
    print(f"using sources: {', '.join(sorted(usable))}")

    account_source = build_account_source_map(snapshot.get("accounts", []))
    for account_id, source in sorted(account_source.items()):
        if source in usable:
            print(f"  account {mask_account_id(account_id)} ({source})")

    desired = aggregate_desired_positions(
        snapshot.get("positions", []),
        usable_sources=usable,
        account_source=account_source,
    )
    print(f"broker-stated share positions: {len(desired)} symbol(s)")

    if not args.apply:
        for symbol, position in sorted(desired.items()):
            print(f"  {symbol}: {position['qty']} @ avg {position['avg_cost']}")
        print("dry-run: no changes written (pass --apply to write)")
        return EXIT_OK

    base_url = os.environ["IAP_BASE_URL"]
    username = os.environ["IAP_USERNAME"]
    password = os.environ["IAP_PASSWORD"]

    # Reuse the neon script's read path: same API, same shape.
    from sync_portfolio_from_neon import _current_positions

    current = _current_positions(base_url, args.portfolio_id, username, password)
    actions = compute_sync_actions(current=current, desired=desired)
    if not actions:
        print("portfolio already in sync; nothing to do")
        return EXIT_OK

    _print_plan(actions)
    _apply_actions(base_url, args.portfolio_id, actions, username, password)
    print(f"sync complete: {len(actions)} action(s) applied")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
