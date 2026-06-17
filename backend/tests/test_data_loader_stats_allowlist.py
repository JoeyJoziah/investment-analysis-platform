"""
Regression guard for #208 item 2 / #197 follow-up.

`DataLoader.get_loading_stats()` builds a per-table ``COUNT(*)`` query using
SQLAlchemy Core constructs (``select(func.count()).select_from(sa_table(name))``).
The table identifiers must only ever come from the module-level
``_STATS_TABLE_ALLOWLIST`` — never from caller input — so a count can never be
issued against an arbitrary, caller-influenced table.

The compiled Core SQL reads ``SELECT count(*) AS count_1 FROM <table>`` (note the
lowercase ``count`` and the ``AS count_1`` alias), so this test parses the
``FROM`` target out of the compiled statement case-insensitively rather than
grepping for ``COUNT(*) FROM <table>`` verbatim.

This is a source-level test (import the package module directly) and mocks the
DB connection, so it runs without a live Postgres.  Run with::

    pytest backend/tests/test_data_loader_stats_allowlist.py --noconftest
"""
import contextlib
import re

import importlib

dl = importlib.import_module("backend.etl.data_loader")


class _FakeResult:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _FakeConn:
    """Records every SQL string executed; returns benign rows."""

    def __init__(self, sink):
        self.sink = sink

    def execute(self, statement):
        sql = str(statement)
        self.sink.append(sql)
        if "MIN(date)" in sql:
            return _FakeResult((None, None))
        return _FakeResult((0,))


def test_get_loading_stats_only_queries_allowlisted_tables(monkeypatch):
    loader = dl.DataLoader.__new__(dl.DataLoader)  # skip __init__ (no DB needed)
    sink: list[str] = []

    @contextlib.contextmanager
    def _fake_get_connection():
        yield _FakeConn(sink)

    monkeypatch.setattr(loader, "get_connection", _fake_get_connection)

    loader.get_loading_stats()

    # The compiled COUNT statements look like
    #   SELECT count(*) AS count_1 \n FROM <table>
    # Examine each recorded statement independently (so a count(*) query can
    # never be matched against a *different* statement's FROM clause) and
    # capture the FROM target of every count(*) query, case-insensitively.
    counted = set()
    for sql in sink:
        if not re.search(r"count\(\*\)", sql, flags=re.IGNORECASE):
            continue
        m = re.search(
            r"\bFROM\s+\"?(\w+)\"?", sql, flags=re.IGNORECASE | re.DOTALL
        )
        if m:
            counted.add(m.group(1))
    assert counted, "expected at least one COUNT(*) query"

    # Regression: nothing outside the allowlist may ever be COUNT-queried.
    assert counted <= dl._STATS_TABLE_ALLOWLIST, (
        f"non-allowlisted table(s) queried: {counted - dl._STATS_TABLE_ALLOWLIST}"
    )
    # Coverage: every allowlisted table is still reported.
    assert dl._STATS_TABLE_ALLOWLIST <= counted, (
        f"allowlisted table(s) no longer counted: {dl._STATS_TABLE_ALLOWLIST - counted}"
    )


def test_stats_allowlist_is_frozen_and_complete():
    assert isinstance(dl._STATS_TABLE_ALLOWLIST, frozenset)
    assert dl._STATS_TABLE_ALLOWLIST == {
        "stocks", "price_history", "technical_indicators",
        "news_sentiment", "ml_predictions", "recommendations",
    }
