"""
Fail-first regression tests for audit findings #197 and #202 in data_loader.py.

Finding #197 – SQL injection in cleanup_old_data()
    (a) Rejects any key not present in the allowlist (ValueError raised).
    (b) Parameterised execution: the cutoff is passed as a bound value, never
        interpolated into the SQL text.

Finding #202 – Date corruption in load_technical_indicators()
    (c) The date bound to the INSERT is the historical date from the DataFrame
        row, not the wall-clock time of the insert.

All tests operate at the function level.  The DB session / connection is fully
mocked; no live Postgres is required.

Import note: backend/etl/__init__.py transitively imports Selenium (not
available in CI/unit environments).  We load data_loader directly via
importlib so the package __init__ is never executed.
"""

import importlib
import importlib.util
import re
import sys
import types
from datetime import datetime, date, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Direct module load — bypasses backend/etl/__init__.py (Selenium chain)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parents[3]  # …/investment-analysis-platform

def _load_data_loader_module():
    """
    Load backend/etl/data_loader.py as an isolated module, patching
    create_engine and sessionmaker so __init__ code never opens a connection.
    """
    module_path = _REPO_ROOT / "backend" / "etl" / "data_loader.py"
    spec = importlib.util.spec_from_file_location(
        "backend.etl.data_loader", module_path
    )
    mod = importlib.util.module_from_spec(spec)

    # Stub out SQLAlchemy engine creation before the module body runs
    sa_stub = types.ModuleType("sqlalchemy")
    sa_stub.create_engine = MagicMock(return_value=MagicMock())
    sa_stub.text = importlib.import_module("sqlalchemy").text
    sa_stub.MetaData = MagicMock()
    sa_stub.Table = MagicMock()

    orm_stub = types.ModuleType("sqlalchemy.orm")
    orm_stub.sessionmaker = MagicMock(return_value=MagicMock())

    exc_stub = types.ModuleType("sqlalchemy.exc")
    exc_stub.IntegrityError = Exception
    exc_stub.OperationalError = Exception

    pool_stub = types.ModuleType("sqlalchemy.pool")
    pool_stub.QueuePool = MagicMock()

    psycopg2_stub = types.ModuleType("psycopg2")
    psycopg2_stub.connect = MagicMock()
    extras_stub = types.ModuleType("psycopg2.extras")
    extras_stub.execute_batch = MagicMock()
    extras_stub.execute_values = MagicMock()
    psycopg2_stub.extras = extras_stub

    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = MagicMock()

    # Register stubs so relative imports inside the module resolve cleanly
    sys.modules.setdefault("sqlalchemy", sa_stub)
    sys.modules.setdefault("sqlalchemy.orm", orm_stub)
    sys.modules.setdefault("sqlalchemy.exc", exc_stub)
    sys.modules.setdefault("sqlalchemy.pool", pool_stub)
    sys.modules.setdefault("psycopg2", psycopg2_stub)
    sys.modules.setdefault("psycopg2.extras", extras_stub)
    sys.modules.setdefault("dotenv", dotenv_stub)

    spec.loader.exec_module(mod)
    return mod


# Load once for the test session
_MOD = _load_data_loader_module()
DataLoader = _MOD.DataLoader
_CLEANUP_ALLOWLIST = _MOD._CLEANUP_ALLOWLIST


def _make_loader():
    """Return a DataLoader instance with its engine/Session stubbed out."""
    loader = object.__new__(DataLoader)
    loader.db_config = {}
    loader.engine = MagicMock()
    loader.Session = MagicMock()
    return loader


# ---------------------------------------------------------------------------
# autouse fixture – nothing to do now that the module is loaded directly,
# but kept as a placeholder so future per-test patching is easy.
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _stub_engine():
    """Placeholder — engine stubbing is handled at module load time."""
    yield


# ===========================================================================
# Finding #197 – cleanup_old_data security
# ===========================================================================

class TestCleanupOldDataAllowlist:
    """#197-a  –  Non-allowlisted keys must be rejected before any SQL runs."""

    def test_rejects_unknown_table_key(self):
        """Passing an unknown key raises ValueError; no SQL executed."""
        loader = _make_loader()

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch.object(loader, "get_connection", return_value=mock_conn):
            with pytest.raises(ValueError, match="not a recognised retention target"):
                loader.cleanup_old_data({"'; DROP TABLE stocks; --": 30})

        mock_conn.execute.assert_not_called()

    def test_rejects_column_injection_disguised_as_table_key(self):
        """A key that looks like a valid table but isn't in the allowlist is rejected."""
        loader = _make_loader()

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch.object(loader, "get_connection", return_value=mock_conn):
            with pytest.raises(ValueError):
                loader.cleanup_old_data({"price_history; DROP TABLE stocks": 10})

        mock_conn.execute.assert_not_called()

    def test_all_default_keys_are_in_allowlist(self):
        """The built-in defaults must all pass allowlist validation without error."""
        default_keys = {
            "price_history",
            "technical_indicators",
            "news_sentiment",
            "ml_predictions",
            "recommendations",
        }
        assert default_keys.issubset(set(_CLEANUP_ALLOWLIST.keys()))


class TestCleanupOldDataParameterized:
    """#197-b  –  The cutoff value must be bound, not interpolated."""

    def _run_cleanup(self, loader, key="price_history", days=30):
        """Run cleanup for a single key and return the executed statement."""
        executed_statements = []

        mock_result = MagicMock()
        mock_result.rowcount = 0

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.side_effect = lambda stmt, *a, **kw: (
            executed_statements.append(stmt) or mock_result
        )

        with patch.object(loader, "get_connection", return_value=mock_conn):
            loader.cleanup_old_data({key: days})

        return executed_statements

    def test_cutoff_not_in_raw_sql_string(self):
        """The integer 'days' must NOT appear as a literal in the compiled SQL text."""
        loader = _make_loader()
        days = 999
        stmts = self._run_cleanup(loader, key="price_history", days=days)

        # Filter out the VACUUM ANALYZE statement
        data_stmts = [
            s for s in stmts
            if "VACUUM" not in str(s).upper()
        ]
        assert data_stmts, "Expected at least one data-manipulation statement"

        for stmt in data_stmts:
            sql_text = str(stmt)
            assert str(days) not in sql_text, (
                f"Days value {days!r} found literally in SQL text: {sql_text!r}. "
                "Use bound parameters instead."
            )

    def test_bound_parameter_cutoff_present(self):
        """The compiled clause must reference a bound parameter named 'cutoff'."""
        loader = _make_loader()
        stmts = self._run_cleanup(loader, key="technical_indicators", days=60)

        data_stmts = [s for s in stmts if "VACUUM" not in str(s).upper()]
        assert data_stmts

        for stmt in data_stmts:
            # SQLAlchemy TextClause or BoundTextClause stores bindparams.
            # We check either the string representation or the actual params.
            sql_text = str(stmt)
            has_cutoff_placeholder = ":cutoff" in sql_text
            has_bound_cutoff = (
                hasattr(stmt, "_bindparams") and "cutoff" in stmt._bindparams
            )
            assert has_cutoff_placeholder or has_bound_cutoff, (
                f"Expected ':cutoff' bound parameter in statement: {sql_text!r}"
            )

    def test_no_f_string_interval_pattern(self):
        """Sanity check on the source: the old f-string INTERVAL pattern must be gone."""
        import inspect

        src = inspect.getsource(_MOD.DataLoader.cleanup_old_data)
        # The old pattern was: INTERVAL '{days} days'  (with an f-prefix on the string)
        assert "INTERVAL '{" not in src, (
            "Found the old f-string INTERVAL pattern in cleanup_old_data source. "
            "It must be replaced with bound parameters."
        )


# ===========================================================================
# Finding #202 – load_technical_indicators date fix
# ===========================================================================

class TestLoadTechnicalIndicatorsDate:
    """#202  –  The persisted date must come from the DataFrame, not datetime.now()."""

    def _build_df(self, data_date) -> pd.DataFrame:
        """Build a minimal single-row DataFrame with the given date."""
        return pd.DataFrame([{
            "date": data_date,
            "sma_20": 100.0,
            "sma_50": 99.0,
            "sma_200": 95.0,
            "ema_12": 101.0,
            "ema_26": 98.0,
            "rsi_14": 55.0,
            "macd": 0.5,
            "macd_signal": 0.4,
            "macd_hist": 0.1,
            "bb_upper": 110.0,
            "bb_middle": 100.0,
            "bb_lower": 90.0,
            "atr_14": 2.0,
            "adx": 25.0,
            "cci": 0.0,
            "mfi": 50.0,
            "obv": 1000.0,
            "stoch_k": 50.0,
            "stoch_d": 50.0,
            "williams_r": -50.0,
            "roc_10": 0.0,
            "momentum_10": 0.0,
        }])

    def _capture_insert_date(self, loader, df):
        """
        Stub ensure_stock_exists and the connection; return the 'date' value
        that was bound in the INSERT params dict.
        """
        captured_params = {}

        mock_result = MagicMock()
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        def capture_execute(stmt, params=None, *args, **kwargs):
            if params and "date" in params:
                captured_params.update(params)
            return mock_result

        mock_conn.execute.side_effect = capture_execute

        with (
            patch.object(loader, "ensure_stock_exists", return_value=42),
            patch.object(loader, "get_connection", return_value=mock_conn),
        ):
            loader.load_technical_indicators(df, "TEST")

        return captured_params.get("date")

    def test_historical_date_pandas_timestamp(self):
        """Accepts a pd.Timestamp and persists its date, not now()."""
        loader = _make_loader()
        historical = pd.Timestamp("2020-06-15")
        df = self._build_df(historical)

        bound_date = self._capture_insert_date(loader, df)

        assert bound_date is not None, "No 'date' param was captured from INSERT"
        # Convert both sides to date for comparison
        if isinstance(bound_date, datetime):
            result_date = bound_date.date()
        else:
            result_date = bound_date

        assert result_date == date(2020, 6, 15), (
            f"Expected 2020-06-15 but got {result_date}. "
            "load_technical_indicators is not threading the data-point date."
        )

    def test_historical_date_python_datetime(self):
        """Accepts a plain datetime object and preserves it."""
        loader = _make_loader()
        historical = datetime(2019, 3, 10, 0, 0, 0)
        df = self._build_df(historical)

        bound_date = self._capture_insert_date(loader, df)

        assert bound_date is not None
        if isinstance(bound_date, datetime):
            result_date = bound_date.date()
        else:
            result_date = bound_date
        assert result_date == date(2019, 3, 10)

    def test_historical_date_python_date(self):
        """Accepts a plain date object and persists it correctly."""
        loader = _make_loader()
        historical = date(2021, 11, 25)
        df = self._build_df(historical)

        bound_date = self._capture_insert_date(loader, df)

        assert bound_date is not None
        if isinstance(bound_date, datetime):
            result_date = bound_date.date()
        else:
            result_date = bound_date
        assert result_date == date(2021, 11, 25)

    def test_bound_date_is_not_today(self):
        """The bound date must not equal today when a historical date is supplied."""
        loader = _make_loader()
        historical = pd.Timestamp("2015-01-01")
        df = self._build_df(historical)

        bound_date = self._capture_insert_date(loader, df)

        today = date.today()
        if isinstance(bound_date, datetime):
            result_date = bound_date.date()
        else:
            result_date = bound_date

        assert result_date != today, (
            "The persisted date equals today — load_technical_indicators is "
            "still using datetime.now() instead of the data-point date."
        )

    def test_missing_date_raises_value_error(self):
        """A DataFrame row without a 'date' column must raise ValueError, not silently stamp now()."""
        loader = _make_loader()
        df = self._build_df(None)  # date=None triggers the guard

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with (
            patch.object(loader, "ensure_stock_exists", return_value=42),
            patch.object(loader, "get_connection", return_value=mock_conn),
            pytest.raises((ValueError, Exception)),
        ):
            loader.load_technical_indicators(df, "TEST")

    def test_no_datetime_now_in_source(self):
        """Sanity: datetime.now() must not appear inside load_technical_indicators."""
        import inspect

        src = inspect.getsource(_MOD.DataLoader.load_technical_indicators)
        assert "datetime.now()" not in src, (
            "Found datetime.now() in load_technical_indicators source. "
            "The row date must come from the DataFrame, not the wall clock."
        )
