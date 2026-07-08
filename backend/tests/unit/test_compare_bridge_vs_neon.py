"""Tests for backend/scripts/compare_bridge_vs_neon.py.

Neon is mocked entirely: no network, no psycopg connection, no real DSN.
Bridge snapshots are synthetic and written into pytest's ``tmp_path``.
"""

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sync_pfn = _load("sync_portfolio_from_neon")
sync_pfb = _load("sync_portfolio_from_bridge")
compare = _load("compare_bridge_vs_neon")


# ---------------------------------------------------------------- fixtures


def _fill(symbol, side, qty, price, account="acct-1111", date="2026-06-01"):
    return (account, date, {"symbol": symbol, "side": side, "qty": qty,
                            "price": price})


def _equity(account_id, symbol, qty, per_unit):
    quantity = Decimal(str(qty))
    return {
        "account_id": account_id,
        "symbol": symbol,
        "instrument_type": "equity",
        "quantity": str(quantity),
        "cost_basis_total": str(quantity * Decimal(str(per_unit))),
        "cost_basis_per_unit": str(per_unit),
    }


def _write_hub(tmp_path, monkeypatch, positions):
    snapshot = {
        "schema_version": "1",
        "as_of": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
        "sources": {"webull": {"status": "ok", "error": None}},
        "accounts": [{"account_id": "webull:aaaa1111", "source": "webull"}],
        "balances": [],
        "positions": positions,
    }
    (tmp_path / "latest.json").write_text(json.dumps(snapshot), encoding="utf-8")
    monkeypatch.setenv("FINANCE_DATA_DIR", str(tmp_path))


# --------------------------------------------------- basis rounding regression


class TestBridgeBasisIsExact:
    """The bridge side of the report must not reconstruct basis from avg_cost.

    300 shares with a $10,000.00 total basis have an average cost of
    $33.3333... which quantizes to $33.33. Reconstructing the total as
    ``qty * avg_cost`` yields $9,999.00 -- a phantom $1.00 divergence, 100x
    the $0.01 threshold. Regression guard for that arithmetic.
    """

    def _bridge(self):
        # cost_basis_total is stated exactly by the broker; the per-share
        # average ($33.3333...) is what does not survive cent quantization.
        position = {
            "account_id": "webull:aaaa1111",
            "symbol": "ACME",
            "instrument_type": "equity",
            "quantity": "300",
            "cost_basis_total": "10000",
        }
        return sync_pfb.aggregate_desired_positions(
            [position],
            usable_sources=frozenset({"webull"}),
            account_source={"webull:aaaa1111": "webull"},
        )

    def test_aggregate_carries_exact_unrounded_basis(self):
        bridge = self._bridge()
        assert bridge["ACME"]["avg_cost"] == Decimal("33.33")
        assert bridge["ACME"]["basis"] == Decimal("10000")

    def test_matching_ledgers_are_not_flagged_divergent(self):
        bridge = self._bridge()
        neon = {"ACME": {"qty": Decimal("300"), "basis": Decimal("10000")}}

        (row,) = compare.build_report(neon=neon, bridge=bridge)

        assert row["bridge_basis"] == Decimal("10000")
        assert row["basis_delta"] == Decimal("0")
        assert row["flagged"] is False

    def test_reconstructing_from_avg_cost_would_have_lied(self):
        """Pin the magnitude of the bug this guards against."""
        bridge = self._bridge()
        naive = bridge["ACME"]["qty"] * bridge["ACME"]["avg_cost"]
        assert naive == Decimal("9999.00")
        assert abs(naive - Decimal("10000")) > compare.BASIS_TOLERANCE


# ------------------------------------------------------------ DSN resolution


class TestResolveDsn:
    def test_neon_dsn_preferred(self):
        env = {"NEON_DSN": "a", "TAX_ADVISOR_NEON_DSN": "b"}
        assert compare.resolve_dsn(env) == "a"

    def test_tax_advisor_dsn_fallback(self):
        assert compare.resolve_dsn({"TAX_ADVISOR_NEON_DSN": "b"}) == "b"

    def test_no_dsn_returns_none(self):
        assert compare.resolve_dsn({}) is None

    def test_empty_dsn_treated_as_missing(self):
        assert compare.resolve_dsn({"NEON_DSN": ""}) is None

    def test_missing_dsn_exits_3_with_guidance(self, capsys, monkeypatch):
        monkeypatch.delenv("NEON_DSN", raising=False)
        monkeypatch.delenv("TAX_ADVISOR_NEON_DSN", raising=False)
        assert compare.main([], env={}) == 3
        err = capsys.readouterr().err
        assert "NEON_DSN=postgresql://" in err
        assert "TAX_ADVISOR_NEON_DSN=postgresql://" in err

    def test_no_dotenv_file_fallback(self, tmp_path, monkeypatch, capsys):
        """A .env.local next to the script must NOT be consulted."""
        (tmp_path / ".env.local").write_text("NEON_DSN=postgresql://leaked/db")
        monkeypatch.chdir(tmp_path)
        assert compare.main([], env={}) == 3
        assert "leaked" not in capsys.readouterr().err

    def test_dsn_never_printed(self, tmp_path, monkeypatch, capsys):
        _write_hub(tmp_path, monkeypatch, [])
        monkeypatch.setattr(compare, "_fetch_neon_rows", lambda dsn: [])
        secret = "postgresql://user:hunter2@host/db"
        assert compare.main([], env={"NEON_DSN": secret}) == 0
        captured = capsys.readouterr()
        assert "hunter2" not in captured.out + captured.err


# ---------------------------------------------------------- FIFO derivation


class TestDeriveNeonFifo:
    def test_qty_matches_derive_positions_parity(self):
        rows = [
            _fill("AAPL", "buy", 10, "100.00", date="2026-06-01"),
            _fill("AAPL", "buy", 10, "200.00", date="2026-06-02"),
            _fill("AAPL", "sell", 15, "210.00", date="2026-06-03"),
        ]
        ours = compare.derive_neon_fifo(rows)
        theirs = sync_pfn.derive_positions(rows)
        assert ours["AAPL"]["qty"] == Decimal(theirs["AAPL"]["qty"])

    def test_exact_remaining_basis(self):
        rows = [
            _fill("AAPL", "buy", 10, "100.00", date="2026-06-01"),
            _fill("AAPL", "buy", 10, "200.00", date="2026-06-02"),
            _fill("AAPL", "sell", 15, "210.00", date="2026-06-03"),
        ]
        # FIFO consumes the 10 @100 lot and 5 of the @200 lot -> 5 @ 200.
        assert compare.derive_neon_fifo(rows)["AAPL"]["basis"] == Decimal("1000.00")

    def test_basis_is_decimal_exact_not_float(self):
        rows = [_fill("X", "buy", 3, "0.10")]
        basis = compare.derive_neon_fifo(rows)["X"]["basis"]
        assert isinstance(basis, Decimal)
        assert basis == Decimal("0.30")  # 3 * 0.1 drifts under float

    def test_closed_position_omitted(self):
        rows = [
            _fill("MSFT", "buy", 5, "400.00", date="2026-06-01"),
            _fill("MSFT", "sell", 5, "410.00", date="2026-06-02"),
        ]
        assert compare.derive_neon_fifo(rows) == {}

    def test_aggregates_across_accounts(self):
        rows = [
            _fill("AAPL", "buy", 1, "100.00", account="a1"),
            _fill("AAPL", "buy", 2, "110.00", account="a2"),
        ]
        entry = compare.derive_neon_fifo(rows)["AAPL"]
        assert entry["qty"] == Decimal("3")
        assert entry["basis"] == Decimal("320.00")

    def test_raw_json_string_payload_accepted(self):
        rows = [("a1", "2026-06-01", json.dumps(
            {"symbol": "AAPL", "side": "buy", "qty": 1, "price": "100.00"}))]
        assert compare.derive_neon_fifo(rows)["AAPL"]["qty"] == Decimal("1")


# ------------------------------------------------------------------- report


class TestBuildReport:
    def test_matching_positions_not_flagged(self):
        neon = {"AAPL": {"qty": Decimal("10"), "basis": Decimal("1000.00")}}
        bridge = {"AAPL": {"qty": Decimal("10"), "avg_cost": Decimal("100.00")}}
        row = compare.build_report(neon=neon, bridge=bridge)[0]
        assert row["flagged"] is False
        assert row["delta"] == Decimal("0")
        assert row["basis_delta"] == Decimal("0.00")

    def test_any_quantity_difference_is_flagged(self):
        neon = {"AAPL": {"qty": Decimal("10"), "basis": Decimal("1000.00")}}
        bridge = {"AAPL": {"qty": Decimal("10.5"), "avg_cost": Decimal("100.00")}}
        row = compare.build_report(neon=neon, bridge=bridge)[0]
        assert row["flagged"] is True
        assert row["delta"] == Decimal("0.5")

    def test_basis_delta_within_one_cent_not_flagged(self):
        neon = {"AAPL": {"qty": Decimal("10"), "basis": Decimal("1000.00")}}
        bridge = {"AAPL": {"qty": Decimal("10"), "avg_cost": Decimal("100.001")}}
        row = compare.build_report(neon=neon, bridge=bridge)[0]
        assert row["basis_delta"] == Decimal("0.010")
        assert row["flagged"] is False  # exactly at tolerance, not beyond

    def test_basis_delta_beyond_one_cent_is_flagged(self):
        neon = {"AAPL": {"qty": Decimal("10"), "basis": Decimal("1000.00")}}
        bridge = {"AAPL": {"qty": Decimal("10"), "avg_cost": Decimal("100.002")}}
        row = compare.build_report(neon=neon, bridge=bridge)[0]
        assert row["basis_delta"] == Decimal("0.020")
        assert row["flagged"] is True

    def test_symbol_only_in_neon(self):
        neon = {"TSLA": {"qty": Decimal("3"), "basis": Decimal("600.00")}}
        row = compare.build_report(neon=neon, bridge={})[0]
        assert row["bridge_qty"] == Decimal("0")
        assert row["delta"] == Decimal("-3")
        assert row["flagged"] is True

    def test_symbol_only_in_bridge(self):
        bridge = {"NVDA": {"qty": Decimal("2"), "avg_cost": Decimal("500.00")}}
        row = compare.build_report(neon={}, bridge=bridge)[0]
        assert row["neon_fifo_qty"] == Decimal("0")
        assert row["delta"] == Decimal("2")
        assert row["bridge_basis"] == Decimal("1000.00")
        assert row["flagged"] is True

    def test_rows_sorted_by_symbol(self):
        neon = {"ZZZ": {"qty": Decimal("1"), "basis": Decimal("1")}}
        bridge = {"AAA": {"qty": Decimal("1"), "avg_cost": Decimal("1")}}
        rows = compare.build_report(neon=neon, bridge=bridge)
        assert [r["symbol"] for r in rows] == ["AAA", "ZZZ"]

    def test_all_money_values_are_decimal(self):
        neon = {"AAPL": {"qty": Decimal("10"), "basis": Decimal("1000.00")}}
        bridge = {"AAPL": {"qty": Decimal("10"), "avg_cost": Decimal("100.00")}}
        row = compare.build_report(neon=neon, bridge=bridge)[0]
        for key in ("neon_fifo_basis", "bridge_basis", "basis_delta"):
            assert isinstance(row[key], Decimal)


# ------------------------------------------------------------ end-to-end main


class TestMainReadOnly:
    def test_json_output_is_machine_readable(self, tmp_path, monkeypatch, capsys):
        _write_hub(tmp_path, monkeypatch, [_equity("webull:aaaa1111", "AAPL", 10, "100.00")])
        monkeypatch.setattr(
            compare, "_fetch_neon_rows", lambda dsn: [_fill("AAPL", "buy", 8, "90.00")]
        )
        assert compare.main(["--json"], env={"NEON_DSN": "postgresql://x/y"}) == 0

        payload = json.loads(capsys.readouterr().out)
        row = payload["rows"][0]
        assert row["symbol"] == "AAPL"
        assert row["neon_fifo_qty"] == "8"
        assert row["bridge_qty"] == "10"
        assert row["delta"] == "2"
        assert row["flagged"] is True

    def test_table_output_states_precedence_rule(self, tmp_path, monkeypatch, capsys):
        _write_hub(tmp_path, monkeypatch, [])
        monkeypatch.setattr(compare, "_fetch_neon_rows", lambda dsn: [])
        assert compare.main([], env={"NEON_DSN": "postgresql://x/y"}) == 0
        out = capsys.readouterr().out
        assert "Neon FIFO is authoritative for tax" in out
        assert "not automatically an error" in out

    def test_unreadable_snapshot_exits_2(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("FINANCE_DATA_DIR", str(tmp_path))  # no latest.json
        monkeypatch.setattr(compare, "_fetch_neon_rows", lambda dsn: [])
        assert compare.main([], env={"NEON_DSN": "postgresql://x/y"}) == 2
        assert "cannot read" in capsys.readouterr().err

    def test_non_ok_bridge_sources_excluded_from_report(
        self, tmp_path, monkeypatch, capsys
    ):
        snapshot = {
            "schema_version": "1",
            "as_of": datetime.now(timezone.utc).isoformat(),
            "sources": {"kubera": {"status": "error", "error": "auth"}},
            "accounts": [{"account_id": "kubera:bbbb2222", "source": "kubera"}],
            "positions": [_equity("kubera:bbbb2222", "MSFT", 5, "400.00")],
        }
        (tmp_path / "latest.json").write_text(json.dumps(snapshot), encoding="utf-8")
        monkeypatch.setenv("FINANCE_DATA_DIR", str(tmp_path))
        monkeypatch.setattr(compare, "_fetch_neon_rows", lambda dsn: [])

        assert compare.main(["--json"], env={"NEON_DSN": "postgresql://x/y"}) == 0
        assert json.loads(capsys.readouterr().out)["rows"] == []

    def test_module_docstring_states_precedence_rule_verbatim(self):
        doc = compare.__doc__
        assert "Neon FIFO is authoritative for tax." in doc
        assert "The bridge reflects what the broker states" in doc
        assert "is NOT automatically an error." in doc

    def test_never_imports_psycopg_when_neon_is_mocked(self, tmp_path, monkeypatch):
        """The report must not open a real connection under test."""
        _write_hub(tmp_path, monkeypatch, [])
        called = {"n": 0}

        def _boom(dsn):
            called["n"] += 1
            return []

        monkeypatch.setattr(compare, "_fetch_neon_rows", _boom)
        assert compare.main([], env={"NEON_DSN": "postgresql://x/y"}) == 0
        assert called["n"] == 1
