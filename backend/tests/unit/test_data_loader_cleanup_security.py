"""Security regression tests for DataLoader.cleanup_old_data (SQL injection fix).

Covers issue #197: table names were f-string-interpolated and the retention
window was inlined, allowing SQL injection via caller-supplied retention_days.
The fix validates tables against an allowlist and binds days as a parameter.
"""
import contextlib
from unittest.mock import MagicMock, patch

import pytest

from backend.etl.data_loader import DataLoader, _RETENTION_TABLES


@pytest.fixture
def loader():
    # Engine creation is irrelevant here; get_connection is mocked per test.
    with patch.object(DataLoader, "_create_engine", return_value=MagicMock()):
        return DataLoader()


def _mock_connection(loader):
    """Wire loader.get_connection() to yield a MagicMock and return it."""
    conn = MagicMock()

    @contextlib.contextmanager
    def fake_get_connection():
        yield conn

    loader.get_connection = fake_get_connection
    return conn


def _executed_sql(conn):
    return " ".join(str(call.args[0]) for call in conn.execute.call_args_list)


def test_unknown_table_is_skipped_and_not_injected(loader):
    conn = _mock_connection(loader)
    malicious = "price_history; DROP TABLE users; --"

    assert loader.cleanup_old_data({malicious: 30, "price_history": 30}) is True

    sql = _executed_sql(conn)
    assert "DROP TABLE users" not in sql          # injection payload never reaches SQL
    assert malicious not in sql
    assert "DELETE FROM price_history" in sql      # the allowlisted table still runs


def test_days_is_bound_not_interpolated(loader):
    conn = _mock_connection(loader)

    loader.cleanup_old_data({"price_history": 30})

    delete_calls = [
        c for c in conn.execute.call_args_list
        if "DELETE FROM price_history" in str(c.args[0])
    ]
    assert delete_calls, "expected a DELETE for price_history"
    args = delete_calls[0].args
    assert len(args) == 2, "query must be executed with a bound-parameter dict"
    assert args[1] == {"days": 30}
    assert "INTERVAL '30 days'" not in str(args[0])   # days not f-string-interpolated


def test_non_integer_days_is_rejected(loader):
    conn = _mock_connection(loader)

    loader.cleanup_old_data({"price_history": "30 days'; DROP TABLE x; --"})

    sql = _executed_sql(conn)
    assert "DROP TABLE" not in sql
    assert "DELETE FROM price_history" not in sql     # skipped due to invalid days


def test_recommendations_archive_uses_bound_param(loader):
    conn = _mock_connection(loader)

    loader.cleanup_old_data({"recommendations": 30})

    archive_calls = [
        c for c in conn.execute.call_args_list
        if "UPDATE recommendations" in str(c.args[0])
    ]
    assert archive_calls, "expected an UPDATE for recommendations"
    assert archive_calls[0].args[1] == {"days": 30}
    assert "INTERVAL '30 days'" not in str(archive_calls[0].args[0])


def test_default_tables_are_all_allowlisted():
    # Guards against drift: the documented default retention tables must remain
    # members of the allowlist, or cleanup would silently skip them.
    for table in ("price_history", "technical_indicators", "news_sentiment",
                  "ml_predictions", "recommendations"):
        assert table in _RETENTION_TABLES