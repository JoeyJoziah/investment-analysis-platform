"""
Tests for backend/scripts/sync_portfolio_from_neon.py (Phase 2 of the
cross-repo finance integration ADR in tax-advisor).

Pure-function coverage: FIFO position derivation from Neon fill rows and
idempotent diff computation against current IAP positions. Network and DB
clients are imported lazily inside main(), so the module loads clean.
"""

import importlib.util
from decimal import Decimal
from pathlib import Path

# Load the script directly (repo pattern, see test_tradingagents_path.py)
_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "sync_portfolio_from_neon.py"
)
_spec = importlib.util.spec_from_file_location("sync_pfn", _MODULE_PATH)
sync_pfn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sync_pfn)


def _row(symbol, side, qty, price, account="a1", date="2026-06-01"):
    return (account, date, {"symbol": symbol, "side": side, "qty": qty,
                            "price": price})


class TestDerivePositions:
    def test_fifo_partial_sell(self):
        rows = [
            _row("AAPL", "buy", 10, "100.00", date="2026-06-01"),
            _row("AAPL", "buy", 10, "200.00", date="2026-06-02"),
            _row("AAPL", "sell", 15, "210.00", date="2026-06-03"),
        ]
        positions = sync_pfn.derive_positions(rows)
        assert positions == {"AAPL": {"qty": 5, "avg_cost": Decimal("200.00")}}

    def test_positions_aggregate_across_accounts(self):
        rows = [
            _row("AAPL", "buy", 1, "100.00", account="a1"),
            _row("AAPL", "buy", 2, "110.00", account="a2"),
        ]
        positions = sync_pfn.derive_positions(rows)
        assert positions["AAPL"]["qty"] == 3

    def test_closed_position_omitted(self):
        rows = [
            _row("MSFT", "buy", 5, "400.00", date="2026-06-01"),
            _row("MSFT", "sell", 5, "410.00", date="2026-06-02"),
        ]
        assert sync_pfn.derive_positions(rows) == {}


class TestComputeSyncActions:
    def test_new_symbol_is_added(self):
        actions = sync_pfn.compute_sync_actions(
            current={}, desired={"AAPL": {"qty": 10, "avg_cost": Decimal("150.00")}}
        )
        assert actions == [
            {"action": "add", "symbol": "AAPL", "qty": 10,
             "price": Decimal("150.00")}
        ]

    def test_increased_qty_adds_delta_only(self):
        actions = sync_pfn.compute_sync_actions(
            current={"AAPL": 4},
            desired={"AAPL": {"qty": 10, "avg_cost": Decimal("150.00")}},
        )
        assert actions[0]["qty"] == 6

    def test_decreased_qty_removes_delta(self):
        actions = sync_pfn.compute_sync_actions(
            current={"AAPL": 10},
            desired={"AAPL": {"qty": 4, "avg_cost": Decimal("150.00")}},
        )
        assert actions == [{"action": "remove", "symbol": "AAPL", "qty": 6}]

    def test_symbol_gone_from_desired_is_fully_removed(self):
        actions = sync_pfn.compute_sync_actions(
            current={"TSLA": 3}, desired={}
        )
        assert actions == [{"action": "remove", "symbol": "TSLA", "qty": 3}]

    def test_matching_positions_produce_no_actions(self):
        actions = sync_pfn.compute_sync_actions(
            current={"AAPL": 10},
            desired={"AAPL": {"qty": 10, "avg_cost": Decimal("1.00")}},
        )
        assert actions == []
