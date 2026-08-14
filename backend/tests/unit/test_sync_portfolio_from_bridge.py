"""Tests for backend/scripts/sync_portfolio_from_bridge.py.

Synthetic fixtures only: no real account identifiers, no real DSN, no
network. The bridge snapshot is written into pytest's ``tmp_path`` and the
hub location is redirected through ``FINANCE_DATA_DIR``.
"""

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name):
    """Load a script module and register it, so the scripts' own
    ``from sync_portfolio_from_neon import ...`` statements resolve to this
    exact object (which is what makes monkeypatching it effective)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Neon first: the bridge script imports compute_sync_actions from it.
sync_pfn = _load("sync_portfolio_from_neon")
sync_pfb = _load("sync_portfolio_from_bridge")


# ---------------------------------------------------------------- fixtures


def _iso(dt):
    return dt.astimezone(timezone.utc).isoformat()


def _fresh():
    return _iso(datetime.now(timezone.utc) - timedelta(hours=1))


def _stale():
    return _iso(datetime.now(timezone.utc) - timedelta(hours=48))


def _account(account_id, source):
    return {"account_id": account_id, "source": source, "institution": "synthetic"}


def _equity(account_id, symbol, qty, per_unit, instrument_type="equity"):
    quantity = Decimal(str(qty))
    return {
        "account_id": account_id,
        "symbol": symbol,
        "instrument_type": instrument_type,
        "quantity": str(quantity),
        "cost_basis_total": str(quantity * Decimal(str(per_unit))),
        "cost_basis_per_unit": str(per_unit),
        "market_price": str(per_unit),
        "market_value": str(quantity * Decimal(str(per_unit))),
    }


def _option(account_id, underlying):
    return {
        "account_id": account_id,
        "symbol": f"{underlying}  260717C00100000",
        "instrument_type": "option",
        "quantity": "2",
        "cost_basis_total": "400.00",
        "cost_basis_per_unit": "2.00",
        "occ_symbol": f"{underlying}  260717C00100000",
        "underlying": underlying,
        "option_type": "call",
        "strike": "100",
        "expiration": "2026-07-17",
        "contract_multiplier": "100",
        "is_short": 0,
    }


def _cash(account_id):
    return {
        "account_id": account_id,
        "symbol": "USD",
        "instrument_type": "cash",
        "quantity": "5000.00",
        "cost_basis_total": "5000.00",
        "cost_basis_per_unit": "1.00",
    }


def _snapshot(as_of=None, sources=None, accounts=None, positions=None):
    return {
        "schema_version": "1",
        "as_of": as_of or _fresh(),
        "sources": sources or {"snaptrade": {"status": "ok", "error": None}},
        "accounts": accounts or [_account("snaptrade:acct-1234", "snaptrade")],
        "balances": [],
        "positions": positions or [],
    }


def _write_hub(tmp_path, monkeypatch, snapshot):
    (tmp_path / "latest.json").write_text(json.dumps(snapshot), encoding="utf-8")
    monkeypatch.setenv("FINANCE_DATA_DIR", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------- freshness


class TestFreshness:
    def test_recent_snapshot_is_not_stale(self):
        as_of = datetime.now(timezone.utc) - timedelta(hours=3)
        assert sync_pfb.is_stale(as_of, max_age_hours=24) is False

    def test_old_snapshot_is_stale(self):
        as_of = datetime.now(timezone.utc) - timedelta(hours=25)
        assert sync_pfb.is_stale(as_of, max_age_hours=24) is True

    def test_boundary_is_inclusive(self):
        now = datetime(2026, 7, 7, 12, 0, tzinfo=timezone.utc)
        exactly_24h = now - timedelta(hours=24)
        assert sync_pfb.is_stale(exactly_24h, max_age_hours=24, now=now) is False

    def test_naive_timestamp_treated_as_utc(self):
        parsed = sync_pfb.parse_as_of("2026-07-07T12:00:00")
        assert parsed.tzinfo is timezone.utc

    def test_z_suffix_parsed(self):
        parsed = sync_pfb.parse_as_of("2026-07-07T12:00:00Z")
        assert parsed == datetime(2026, 7, 7, 12, 0, tzinfo=timezone.utc)

    def test_stale_snapshot_exits_2(self, tmp_path, monkeypatch, capsys):
        _write_hub(tmp_path, monkeypatch, _snapshot(as_of=_stale()))
        code = sync_pfb.main(["--portfolio-id", "p1"])
        assert code == 2
        assert "stale" in capsys.readouterr().err

    def test_force_bypasses_stale_gate(self, tmp_path, monkeypatch, capsys):
        _write_hub(tmp_path, monkeypatch, _snapshot(as_of=_stale()))
        code = sync_pfb.main(["--portfolio-id", "p1", "--force"])
        assert code == 0
        assert "WARNING" in capsys.readouterr().out

    def test_max_age_hours_override(self, tmp_path, monkeypatch):
        _write_hub(tmp_path, monkeypatch, _snapshot(as_of=_stale()))
        assert sync_pfb.main(["--portfolio-id", "p1", "--max-age-hours", "72"]) == 0


# ------------------------------------------------------------ source status


class TestSourceStatusGate:
    @pytest.mark.parametrize("status", ["partial", "error", "missing", "unknown"])
    def test_non_ok_source_is_skipped(self, status):
        usable, skipped = sync_pfb.ok_sources(
            {"snaptrade": {"status": status, "error": "boom"}}
        )
        assert usable == frozenset()
        assert skipped == [("snaptrade", status, "boom")]

    def test_only_ok_sources_are_usable(self):
        usable, skipped = sync_pfb.ok_sources(
            {
                "webull": {"status": "ok", "error": None},
                "robinhood": {"status": "ok", "error": None},
                "ibkr": {"status": "missing", "error": None},
                "kubera": {"status": "error", "error": "auth expired"},
            }
        )
        assert usable == frozenset({"webull", "robinhood"})
        assert [s[0] for s in skipped] == ["ibkr", "kubera"]

    def test_positions_from_non_ok_source_excluded(self):
        accounts = [
            _account("webull:aaaa1111", "webull"),
            _account("kubera:bbbb2222", "kubera"),
        ]
        positions = [
            _equity("webull:aaaa1111", "AAPL", 10, "100.00"),
            _equity("kubera:bbbb2222", "MSFT", 5, "400.00"),
        ]
        desired = sync_pfb.aggregate_desired_positions(
            positions,
            usable_sources=frozenset({"webull"}),
            account_source=sync_pfb.build_account_source_map(accounts),
        )
        assert set(desired) == {"AAPL"}

    def test_zero_ok_sources_exits_2(self, tmp_path, monkeypatch, capsys):
        snapshot = _snapshot(
            sources={
                "ibkr": {"status": "missing", "error": None},
                "kubera": {"status": "error", "error": "auth expired"},
            }
        )
        _write_hub(tmp_path, monkeypatch, snapshot)
        code = sync_pfb.main(["--portfolio-id", "p1"])
        assert code == 2
        captured = capsys.readouterr()
        assert "no source reported status=ok" in captured.err
        assert "auth expired" in captured.out

    def test_source_falls_back_to_account_id_prefix(self):
        # Position references an account with no accounts[] row.
        desired = sync_pfb.aggregate_desired_positions(
            [_equity("webull:zzzz9999", "AAPL", 1, "10.00")],
            usable_sources=frozenset({"webull"}),
            account_source={},
        )
        assert desired["AAPL"]["qty"] == Decimal("1")


# ------------------------------------------------------------- desired set


class TestDesiredSet:
    def test_options_cash_and_crypto_excluded(self):
        positions = [
            _equity("snaptrade:acct-1234", "AAPL", 10, "100.00"),
            _equity("snaptrade:acct-1234", "VOO", 3, "500.00", "etf"),
            _option("snaptrade:acct-1234", "AAPL"),
            _cash("snaptrade:acct-1234"),
            _equity("snaptrade:acct-1234", "BTC", 1, "60000", "crypto"),
        ]
        desired = sync_pfb.aggregate_desired_positions(
            positions,
            usable_sources=frozenset({"snaptrade"}),
            account_source={"snaptrade:acct-1234": "snaptrade"},
        )
        assert set(desired) == {"AAPL", "VOO"}

    def test_multi_account_symbol_aggregation(self):
        accounts = [
            _account("webull:aaaa1111", "webull"),
            _account("robinhood:bbbb2222", "robinhood"),
        ]
        positions = [
            _equity("webull:aaaa1111", "AAPL", 10, "100.00"),
            _equity("robinhood:bbbb2222", "AAPL", 5, "220.00"),
        ]
        desired = sync_pfb.aggregate_desired_positions(
            positions,
            usable_sources=frozenset({"webull", "robinhood"}),
            account_source=sync_pfb.build_account_source_map(accounts),
        )
        assert desired["AAPL"]["qty"] == Decimal("15")
        # (10*100 + 5*220) / 15 = 2100/15 = 140.00
        assert desired["AAPL"]["avg_cost"] == Decimal("140.00")

    def test_avg_cost_is_decimal_not_float(self):
        desired = sync_pfb.aggregate_desired_positions(
            [_equity("webull:aaaa1111", "AAPL", 3, "0.10")],
            usable_sources=frozenset({"webull"}),
            account_source={"webull:aaaa1111": "webull"},
        )
        avg = desired["AAPL"]["avg_cost"]
        assert isinstance(avg, Decimal)
        assert isinstance(desired["AAPL"]["qty"], Decimal)
        # 0.30/3 == 0.10 exactly under Decimal; float would drift.
        assert avg == Decimal("0.10")

    def test_decimal_basis_sum_avoids_float_error(self):
        # 0.1 + 0.2 != 0.3 under binary float.
        positions = [
            _equity("webull:aaaa1111", "X", 1, "0.10"),
            _equity("webull:aaaa1111", "X", 1, "0.20"),
        ]
        desired = sync_pfb.aggregate_desired_positions(
            positions,
            usable_sources=frozenset({"webull"}),
            account_source={"webull:aaaa1111": "webull"},
        )
        assert desired["X"]["avg_cost"] == Decimal("0.15")

    def test_fractional_shares_preserved(self):
        desired = sync_pfb.aggregate_desired_positions(
            [_equity("webull:aaaa1111", "AAPL", "1.5", "100.00")],
            usable_sources=frozenset({"webull"}),
            account_source={"webull:aaaa1111": "webull"},
        )
        assert desired["AAPL"]["qty"] == Decimal("1.5")

    def test_cost_basis_total_preferred_over_per_unit(self):
        position = _equity("webull:aaaa1111", "AAPL", 10, "100.00")
        position["cost_basis_total"] = "999.00"  # authoritative
        desired = sync_pfb.aggregate_desired_positions(
            [position],
            usable_sources=frozenset({"webull"}),
            account_source={"webull:aaaa1111": "webull"},
        )
        assert desired["AAPL"]["avg_cost"] == Decimal("99.90")

    def test_missing_total_falls_back_to_per_unit(self):
        position = _equity("webull:aaaa1111", "AAPL", 4, "25.00")
        position["cost_basis_total"] = None
        desired = sync_pfb.aggregate_desired_positions(
            [position],
            usable_sources=frozenset({"webull"}),
            account_source={"webull:aaaa1111": "webull"},
        )
        assert desired["AAPL"]["avg_cost"] == Decimal("25.00")


# ----------------------------------------------------------- idempotency


class TestComputeSyncActionsIdempotency:
    """compute_sync_actions is IMPORTED from the neon script, not copied."""

    def test_function_is_the_same_object(self):
        assert sync_pfb.compute_sync_actions is sync_pfn.compute_sync_actions

    def test_running_the_plan_twice_yields_empty_second_plan(self):
        desired = sync_pfb.aggregate_desired_positions(
            [
                _equity("webull:aaaa1111", "AAPL", 10, "100.00"),
                _equity("webull:aaaa1111", "VOO", 3, "500.00", "etf"),
            ],
            usable_sources=frozenset({"webull"}),
            account_source={"webull:aaaa1111": "webull"},
        )
        current = {}

        first = sync_pfb.compute_sync_actions(current=current, desired=desired)
        assert len(first) == 2

        # Simulate the portfolio after applying the plan.
        for action in first:
            delta = action["qty"] if action["action"] == "add" else -action["qty"]
            current[action["symbol"]] = current.get(action["symbol"], 0) + delta

        second = sync_pfb.compute_sync_actions(current=current, desired=desired)
        assert second == []

    def test_decimal_desired_vs_int_current_compares_exactly(self):
        desired = {"AAPL": {"qty": Decimal("10"), "avg_cost": Decimal("1.00")}}
        assert sync_pfb.compute_sync_actions(current={"AAPL": 10}, desired=desired) == []

    def test_removal_when_broker_no_longer_reports_symbol(self):
        actions = sync_pfb.compute_sync_actions(current={"TSLA": 3}, desired={})
        assert actions == [{"action": "remove", "symbol": "TSLA", "qty": 3}]


# ---------------------------------------------------------------- masking


class TestMasking:
    def test_account_id_masked_to_last_four(self):
        assert sync_pfb.mask_account_id("snaptrade:abc-123456") == "****3456"

    def test_empty_account_id_masked(self):
        assert sync_pfb.mask_account_id("") == "****"

    def test_full_account_id_never_printed(self, tmp_path, monkeypatch, capsys):
        account_id = "snaptrade:supersecret-9876"
        snapshot = _snapshot(
            accounts=[_account(account_id, "snaptrade")],
            positions=[_equity(account_id, "AAPL", 10, "100.00")],
        )
        _write_hub(tmp_path, monkeypatch, snapshot)
        assert sync_pfb.main(["--portfolio-id", "p1"]) == 0
        out = capsys.readouterr().out
        assert account_id not in out
        assert "supersecret" not in out
        assert "****9876" in out


# ------------------------------------------------------------------ dry run


class TestDryRunDefault:
    def test_dry_run_is_default_and_writes_nothing(
        self, tmp_path, monkeypatch, capsys
    ):
        snapshot = _snapshot(
            positions=[_equity("snaptrade:acct-1234", "AAPL", 10, "100.00")]
        )
        _write_hub(tmp_path, monkeypatch, snapshot)
        # No IAP_* env vars set: if the script tried to write, it would KeyError.
        monkeypatch.delenv("IAP_BASE_URL", raising=False)

        assert sync_pfb.main(["--portfolio-id", "p1"]) == 0
        out = capsys.readouterr().out
        assert "dry-run: no changes written" in out
        assert "AAPL: 10 @ avg 100.00" in out

    def test_unreadable_snapshot_exits_2(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("FINANCE_DATA_DIR", str(tmp_path))  # no latest.json
        assert sync_pfb.main(["--portfolio-id", "p1"]) == 2
        assert "cannot read" in capsys.readouterr().err

    def test_apply_calls_the_write_path(self, tmp_path, monkeypatch, capsys):
        snapshot = _snapshot(
            positions=[_equity("snaptrade:acct-1234", "AAPL", 10, "100.00")]
        )
        _write_hub(tmp_path, monkeypatch, snapshot)
        for key in ("IAP_BASE_URL", "IAP_USERNAME", "IAP_PASSWORD"):
            monkeypatch.setenv(key, "x")

        applied = {}
        monkeypatch.setattr(
            sync_pfn, "_current_positions", lambda *a, **k: {}, raising=True
        )
        monkeypatch.setattr(
            sync_pfb,
            "_apply_actions",
            lambda *args: applied.setdefault("actions", args[2]),
            raising=True,
        )
        assert sync_pfb.main(["--portfolio-id", "p1", "--apply"]) == 0
        assert applied["actions"][0]["symbol"] == "AAPL"
        assert applied["actions"][0]["qty"] == Decimal("10")
        assert "sync complete" in capsys.readouterr().out

    def test_apply_refuses_empty_desired(self, tmp_path, monkeypatch, capsys):
        snapshot = _snapshot(positions=[])
        _write_hub(tmp_path, monkeypatch, snapshot)
        for key in ("IAP_BASE_URL", "IAP_USERNAME", "IAP_PASSWORD"):
            monkeypatch.setenv(key, "x")

        applied = {}
        monkeypatch.setattr(
            sync_pfb,
            "_apply_actions",
            lambda *args: applied.setdefault("hit", True),
            raising=True,
        )
        assert sync_pfb.main(["--portfolio-id", "p1", "--apply"]) == 2
        assert not applied
        assert "refusing --apply" in capsys.readouterr().err


# --------------------------------------------------------- json serialization


class TestJsonQty:
    def test_whole_share_count_serializes_as_int(self):
        assert sync_pfb._json_qty(Decimal("10")) == 10
        assert isinstance(sync_pfb._json_qty(Decimal("10")), int)

    def test_fractional_share_count_serializes_as_float(self):
        assert sync_pfb._json_qty(Decimal("1.5")) == 1.5
