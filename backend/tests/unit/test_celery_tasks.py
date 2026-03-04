"""
Unit tests for Celery task modules.

Tests the underlying function logic of all Celery tasks by mocking
external dependencies (database, Redis, SMTP, Celery infrastructure).
"""
import sys
import types
from unittest.mock import patch, MagicMock, PropertyMock
from datetime import datetime, date, timedelta, timezone
from decimal import Decimal
import json
import pytest

# ---------------------------------------------------------------------------
# Mock Celery before importing any task modules.  The actual celery package
# may or may not be installed -- either way we want to test pure function
# logic, not Celery infrastructure.
# ---------------------------------------------------------------------------

_celery_mock = types.ModuleType("celery")
_celery_mock.shared_task = lambda *a, **kw: (lambda fn: fn)
_celery_mock.Celery = MagicMock
_celery_mock.Task = MagicMock
_celery_mock.group = MagicMock
_celery_mock.chain = MagicMock
_celery_mock.current_app = MagicMock

# celery.signals sub-module
_celery_signals = types.ModuleType("celery.signals")
_celery_signals.worker_ready = MagicMock()
_celery_signals.worker_ready.connect = MagicMock()
_celery_signals.worker_shutdown = MagicMock()
_celery_signals.worker_shutdown.connect = MagicMock()
_celery_signals.task_postrun = MagicMock()
_celery_signals.task_postrun.connect = MagicMock()
_celery_signals.task_failure = MagicMock()
_celery_signals.task_failure.connect = MagicMock()

_celery_schedules = types.ModuleType("celery.schedules")
_celery_schedules.crontab = MagicMock

_celery_beat = types.ModuleType("celery.beat")
_celery_beat.PersistentScheduler = MagicMock

_kombu = types.ModuleType("kombu")
_kombu.Exchange = MagicMock
_kombu.Queue = MagicMock

sys.modules.setdefault("celery", _celery_mock)
sys.modules.setdefault("celery.signals", _celery_signals)
sys.modules.setdefault("celery.schedules", _celery_schedules)
sys.modules.setdefault("celery.beat", _celery_beat)
sys.modules.setdefault("kombu", _kombu)

# Mock the celery_app module so task decorators become no-ops.
# We use a plain object (not MagicMock) so attribute assignment works normally.
class _FakeCeleryApp:
    """Minimal stand-in for the Celery app. .task() is a no-op decorator."""

    class Task:
        pass

    class conf:
        pass

    @staticmethod
    def task(*args, **kwargs):
        """Accept @celery_app.task or @celery_app.task(bind=True, ...) forms."""
        # @celery_app.task          -> args=(func,), return func
        # @celery_app.task(...)     -> args=(), return decorator
        if args and callable(args[0]):
            return args[0]
        return lambda fn: fn


_celery_app_mod = types.ModuleType("backend.tasks.celery_app")
_celery_app_mod.celery_app = _FakeCeleryApp()
sys.modules["backend.tasks.celery_app"] = _celery_app_mod

# Clear any previously-imported task modules so they re-import with the fake
# celery_app (when real celery is installed, earlier tests may have imported these).
# We must also remove them as attributes on the parent package, because Python
# caches submodules as attributes; `from backend.tasks.X import Y` would find
# the old (real-celery) module via the attribute rather than re-importing.
_tasks_pkg = sys.modules.get("backend.tasks")
for _task_mod in list(sys.modules):
    if _task_mod.startswith("backend.tasks.") and _task_mod != "backend.tasks.celery_app":
        _attr = _task_mod.rsplit(".", 1)[-1]
        if _tasks_pkg is not None and hasattr(_tasks_pkg, _attr):
            delattr(_tasks_pkg, _attr)
        del sys.modules[_task_mod]

# Mock heavy optional dependencies that may not be installed
for mod_name in ("psutil", "redis"):
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()


# =========================================================================
# Helper factories
# =========================================================================

def _make_stock(symbol="AAPL", stock_id=1, name="Apple Inc.", sector="Technology",
                is_active=True, is_tradable=True, market_cap=3_000_000_000_000):
    s = MagicMock()
    s.id = stock_id
    s.symbol = symbol
    s.name = name
    s.sector = sector
    s.is_active = is_active
    s.is_tradable = is_tradable
    s.market_cap = market_cap
    s.industry = "Consumer Electronics"
    return s


def _make_price_history(close=150.0, open_=149.0, high=152.0, low=148.0,
                        volume=75_000_000, price_date=None, stock_id=1):
    ph = MagicMock()
    ph.stock_id = stock_id
    ph.close = Decimal(str(close))
    ph.open = Decimal(str(open_))
    ph.high = Decimal(str(high))
    ph.low = Decimal(str(low))
    ph.volume = volume
    ph.date = price_date or date.today()
    return ph


def _make_portfolio(portfolio_id=1, name="Test Portfolio", cash_balance=10000.0, user_id=1):
    p = MagicMock()
    p.id = portfolio_id
    p.name = name
    p.cash_balance = Decimal(str(cash_balance))
    p.user_id = user_id
    p.updated_at = datetime.now(timezone.utc)
    p.target_allocation = None
    p.rebalance_frequency = None
    return p


def _make_position(position_id=1, portfolio_id=1, stock_id=1, quantity=10,
                   avg_cost_basis=145.0, symbol="AAPL"):
    pos = MagicMock()
    pos.id = position_id
    pos.portfolio_id = portfolio_id
    pos.stock_id = stock_id
    pos.quantity = Decimal(str(quantity))
    pos.avg_cost_basis = Decimal(str(avg_cost_basis))
    stock = _make_stock(symbol=symbol, stock_id=stock_id)
    pos.stock = stock
    return pos


def _make_user(user_id=1, email="test@example.com", is_active=True, is_verified=True):
    u = MagicMock()
    u.id = user_id
    u.email = email
    u.is_active = is_active
    u.is_verified = is_verified
    u.notification_settings = {"email_daily_summary": True}
    return u


def _make_alert(alert_id=1, user_id=1, stock_id=1, condition=None, is_active=True):
    a = MagicMock()
    a.id = alert_id
    a.user_id = user_id
    a.stock_id = stock_id
    a.is_active = is_active
    a.condition = condition or {"type": "price_above", "value": 155.0}
    a.alert_type = "price_alert"
    a.triggered_count = 0
    a.last_triggered = None
    return a


def _make_performance_record(portfolio_id=1, total_value=10000.0, perf_date=None):
    rec = MagicMock()
    rec.portfolio_id = portfolio_id
    rec.total_value = Decimal(str(total_value))
    rec.date = perf_date or date.today()
    rec.cash_value = Decimal("5000")
    rec.positions_value = Decimal(str(total_value - 5000))
    return rec


# =========================================================================
# Context manager helper for mocking get_db_sync
# =========================================================================

class FakeDBContext:
    """Fake context manager returned by get_db_sync()."""

    def __init__(self, session):
        self.session = session

    def __enter__(self):
        return self.session

    def __exit__(self, *args):
        pass


def make_db_context(session):
    return FakeDBContext(session)


# =========================================================================
# ANALYSIS TASKS
# =========================================================================

class TestAnalyzeStock:
    """Tests for analyze_stock task."""

    @patch("backend.tasks.analysis_tasks.get_redis_client")
    @patch("backend.tasks.analysis_tasks.get_db_sync")
    def test_stock_not_found_returns_error(self, mock_get_db, mock_redis):
        from backend.tasks.analysis_tasks import analyze_stock

        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = None
        mock_get_db.return_value = FakeDBContext(db)

        # analyze_stock is bound (self is first arg), pass a mock for self
        mock_self = MagicMock()
        result = analyze_stock(mock_self, "ZZZZ")
        assert "error" in result
        assert "ZZZZ" in result["error"]

    @patch("backend.tasks.analysis_tasks.create_recommendation")
    @patch("backend.tasks.analysis_tasks.get_redis_client")
    @patch("backend.tasks.analysis_tasks.get_db_sync")
    def test_no_price_history_returns_error(self, mock_get_db, mock_redis, mock_create_rec):
        from backend.tasks.analysis_tasks import analyze_stock

        db = MagicMock()
        stock = _make_stock()
        # First query returns stock, second returns empty price history
        query_chain = MagicMock()
        db.query.return_value = query_chain

        def filter_side_effect(*args, **kwargs):
            return query_chain

        query_chain.filter.side_effect = filter_side_effect
        query_chain.order_by.return_value = query_chain
        query_chain.limit.return_value = query_chain

        # first() calls: stock found, then no prices
        query_chain.first.side_effect = [stock, None]
        query_chain.all.return_value = []

        mock_get_db.return_value = FakeDBContext(db)

        mock_self = MagicMock()
        result = analyze_stock(mock_self, "AAPL")
        # After finding the stock, it queries price history - with empty history
        # the function should return error about no price history
        assert "error" in result

    @patch("backend.tasks.analysis_tasks.create_recommendation")
    @patch("backend.tasks.analysis_tasks.get_redis_client")
    @patch("backend.tasks.analysis_tasks.get_db_sync")
    @patch("backend.tasks.analysis_tasks.TechnicalAnalyzer")
    def test_valid_stock_returns_analysis(self, mock_tech, mock_get_db, mock_redis, mock_create_rec):
        from backend.tasks.analysis_tasks import analyze_stock

        db = MagicMock()
        stock = _make_stock()
        prices = [_make_price_history(close=150 + i, price_date=date.today() - timedelta(days=i))
                  for i in range(30)]

        # Set up the query chain to return stock then prices
        call_count = {"n": 0}

        def query_side_effect(model):
            chain = MagicMock()
            chain.filter.return_value = chain
            chain.order_by.return_value = chain
            chain.limit.return_value = chain
            if call_count["n"] == 0:
                # Stock query
                chain.first.return_value = stock
                call_count["n"] += 1
            elif call_count["n"] == 1:
                # Price history query
                chain.all.return_value = prices
                call_count["n"] += 1
            else:
                chain.first.return_value = None
                chain.all.return_value = []
                call_count["n"] += 1
            return chain

        db.query.side_effect = query_side_effect
        mock_get_db.return_value = FakeDBContext(db)

        mock_tech_instance = MagicMock()
        mock_tech_instance.analyze.return_value = {"score": 75, "trend": "bullish"}
        mock_tech.return_value = mock_tech_instance

        redis_client = MagicMock()
        mock_redis.return_value = redis_client

        mock_self = MagicMock()
        result = analyze_stock(mock_self, "AAPL", ["technical"])

        assert result["symbol"] == "AAPL"
        assert "timestamp" in result
        assert "overall_score" in result
        assert "recommendation" in result


class TestBatchAnalyzeStocks:
    """Tests for batch analysis via run_daily_analysis."""

    @patch("backend.tasks.analysis_tasks.get_redis_client")
    @patch("backend.tasks.analysis_tasks.get_db_sync")
    def test_no_active_stocks(self, mock_get_db, mock_redis):
        from backend.tasks.analysis_tasks import run_daily_analysis

        db = MagicMock()
        db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []
        mock_get_db.return_value = FakeDBContext(db)

        # group().apply_async().get() returns empty
        with patch("backend.tasks.analysis_tasks.group") as mock_group:
            mock_result = MagicMock()
            mock_result.get.return_value = []
            mock_group.return_value.apply_async.return_value = mock_result

            result = run_daily_analysis()
            assert result["stocks_analyzed"] == 0
            assert result["strong_buys"] == []


class TestGenerateRecommendation:
    """Tests for recommendation helper functions."""

    def test_generate_recommendation_strong_buy(self):
        from backend.tasks.analysis_tasks import generate_recommendation
        assert generate_recommendation(85) == "strong_buy"

    def test_generate_recommendation_buy(self):
        from backend.tasks.analysis_tasks import generate_recommendation
        assert generate_recommendation(70) == "buy"

    def test_generate_recommendation_hold(self):
        from backend.tasks.analysis_tasks import generate_recommendation
        assert generate_recommendation(50) == "hold"

    def test_generate_recommendation_sell(self):
        from backend.tasks.analysis_tasks import generate_recommendation
        assert generate_recommendation(25) == "sell"

    def test_generate_recommendation_strong_sell(self):
        from backend.tasks.analysis_tasks import generate_recommendation
        assert generate_recommendation(10) == "strong_sell"


class TestCalculateOverallScore:
    """Tests for calculate_overall_score helper."""

    def test_with_all_scores(self):
        from backend.tasks.analysis_tasks import calculate_overall_score
        analysis = {
            "technical": {"score": 80},
            "fundamental": {"score": 70},
            "sentiment": {"score": 60},
        }
        score = calculate_overall_score(analysis)
        # (80*0.4 + 70*0.3 + 60*0.3) / 1.0 = 71.0
        assert abs(score - 71.0) < 0.01

    def test_with_no_scores_returns_default(self):
        from backend.tasks.analysis_tasks import calculate_overall_score
        score = calculate_overall_score({})
        assert score == 50

    def test_with_partial_scores(self):
        from backend.tasks.analysis_tasks import calculate_overall_score
        analysis = {"technical": {"score": 90}}
        score = calculate_overall_score(analysis)
        # Only technical present: 90*0.4 / 1.0 = 36.0
        assert abs(score - 36.0) < 0.01


class TestExtractKeyFactors:
    """Tests for extract_key_factors helper."""

    def test_high_score_adds_technical_factor(self):
        from backend.tasks.analysis_tasks import extract_key_factors
        result = extract_key_factors({"overall_score": 80, "analysis": {}})
        assert "Strong technical indicators" in result

    def test_positive_sentiment_adds_factor(self):
        from backend.tasks.analysis_tasks import extract_key_factors
        result = extract_key_factors({
            "overall_score": 50,
            "analysis": {"sentiment": {"overall": 0.8}},
        })
        assert "Positive market sentiment" in result

    def test_negative_sentiment_adds_factor(self):
        from backend.tasks.analysis_tasks import extract_key_factors
        result = extract_key_factors({
            "overall_score": 50,
            "analysis": {"sentiment": {"overall": -0.7}},
        })
        assert "Negative market sentiment" in result

    def test_no_factors_returns_default(self):
        from backend.tasks.analysis_tasks import extract_key_factors
        result = extract_key_factors({"overall_score": 50, "analysis": {}})
        assert result == ["Market conditions"]


class TestCalculateRiskLevel:
    """Tests for calculate_risk_level helper."""

    def test_low_volatility(self):
        from backend.tasks.analysis_tasks import calculate_risk_level
        result = calculate_risk_level({"analysis": {"technical": {"volatility": 0.10}}})
        assert result == "low"

    def test_medium_volatility(self):
        from backend.tasks.analysis_tasks import calculate_risk_level
        result = calculate_risk_level({"analysis": {"technical": {"volatility": 0.20}}})
        assert result == "medium"

    def test_high_volatility(self):
        from backend.tasks.analysis_tasks import calculate_risk_level
        result = calculate_risk_level({"analysis": {"technical": {"volatility": 0.30}}})
        assert result == "high"

    def test_default_volatility(self):
        from backend.tasks.analysis_tasks import calculate_risk_level
        # No volatility data => default 0.2 => medium
        result = calculate_risk_level({"analysis": {}})
        assert result == "medium"


# =========================================================================
# PORTFOLIO TASKS
# =========================================================================

class TestUpdatePortfolioValue:
    """Tests for update_portfolio_value task."""

    @patch("backend.tasks.portfolio_tasks.get_redis_client")
    @patch("backend.tasks.portfolio_tasks.get_db_sync")
    def test_portfolio_not_found(self, mock_get_db, mock_redis):
        from backend.tasks.portfolio_tasks import update_portfolio_value

        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = None
        mock_get_db.return_value = FakeDBContext(db)

        result = update_portfolio_value(999)
        assert "error" in result
        assert "999" in result["error"]

    @patch("backend.tasks.portfolio_tasks.get_redis_client")
    @patch("backend.tasks.portfolio_tasks.get_db_sync")
    def test_correct_value_calculation(self, mock_get_db, mock_redis):
        from backend.tasks.portfolio_tasks import update_portfolio_value

        portfolio = _make_portfolio(cash_balance=5000.0)
        position = _make_position(quantity=10, avg_cost_basis=145.0)
        price = _make_price_history(close=150.0)

        db = MagicMock()
        call_count = {"n": 0}

        def query_side_effect(model):
            chain = MagicMock()
            chain.filter.return_value = chain
            chain.order_by.return_value = chain
            if call_count["n"] == 0:
                # Portfolio query
                chain.first.return_value = portfolio
                call_count["n"] += 1
            elif call_count["n"] == 1:
                # Positions query
                chain.all.return_value = [position]
                call_count["n"] += 1
            elif call_count["n"] == 2:
                # Price query for position
                chain.first.return_value = price
                call_count["n"] += 1
            else:
                # PortfolioPerformance existence check
                chain.first.return_value = None
                call_count["n"] += 1
            return chain

        db.query.side_effect = query_side_effect
        mock_get_db.return_value = FakeDBContext(db)

        redis_client = MagicMock()
        mock_redis.return_value = redis_client

        result = update_portfolio_value(1)

        assert "error" not in result
        assert result["portfolio_id"] == 1
        # 10 shares * 150 = 1500 position value + 5000 cash = 6500
        expected_position_value = float(Decimal("10") * Decimal("150.0"))
        assert result["positions_value"] == expected_position_value
        expected_total = expected_position_value + 5000.0
        assert result["total_value"] == expected_total

    @patch("backend.tasks.portfolio_tasks.get_redis_client")
    @patch("backend.tasks.portfolio_tasks.get_db_sync")
    def test_empty_portfolio(self, mock_get_db, mock_redis):
        from backend.tasks.portfolio_tasks import update_portfolio_value

        portfolio = _make_portfolio(cash_balance=10000.0)

        db = MagicMock()
        call_count = {"n": 0}

        def query_side_effect(model):
            chain = MagicMock()
            chain.filter.return_value = chain
            chain.order_by.return_value = chain
            if call_count["n"] == 0:
                chain.first.return_value = portfolio
                call_count["n"] += 1
            elif call_count["n"] == 1:
                chain.all.return_value = []  # No positions
                call_count["n"] += 1
            else:
                chain.first.return_value = None
                call_count["n"] += 1
            return chain

        db.query.side_effect = query_side_effect
        mock_get_db.return_value = FakeDBContext(db)
        mock_redis.return_value = MagicMock()

        result = update_portfolio_value(1)
        assert result["positions_value"] == 0.0
        assert result["total_value"] == 10000.0


class TestCalculatePortfolioPerformance:
    """Tests for calculate_portfolio_performance task."""

    @patch("backend.tasks.portfolio_tasks.get_redis_client")
    @patch("backend.tasks.portfolio_tasks.get_db_sync")
    def test_insufficient_data(self, mock_get_db, mock_redis):
        from backend.tasks.portfolio_tasks import calculate_portfolio_performance

        db = MagicMock()
        chain = MagicMock()
        db.query.return_value = chain
        chain.filter.return_value = chain
        chain.order_by.return_value = chain
        chain.all.return_value = [_make_performance_record()]  # Only 1 record

        mock_get_db.return_value = FakeDBContext(db)

        result = calculate_portfolio_performance(1, 30)
        assert "error" in result
        assert "Insufficient" in result["error"]

    @patch("backend.tasks.portfolio_tasks.get_redis_client")
    @patch("backend.tasks.portfolio_tasks.get_db_sync")
    def test_valid_performance_calculation(self, mock_get_db, mock_redis):
        from backend.tasks.portfolio_tasks import calculate_portfolio_performance

        records = [
            _make_performance_record(total_value=10000, perf_date=date.today() - timedelta(days=30)),
            _make_performance_record(total_value=10100, perf_date=date.today() - timedelta(days=20)),
            _make_performance_record(total_value=10050, perf_date=date.today() - timedelta(days=10)),
            _make_performance_record(total_value=10500, perf_date=date.today()),
        ]

        db = MagicMock()
        chain = MagicMock()
        db.query.return_value = chain
        chain.filter.return_value = chain
        chain.order_by.return_value = chain
        chain.all.return_value = records

        mock_get_db.return_value = FakeDBContext(db)
        mock_redis.return_value = MagicMock()

        result = calculate_portfolio_performance(1, 30)
        assert "error" not in result
        assert result["portfolio_id"] == 1
        # total_return = (10500 - 10000) / 10000 = 0.05
        assert abs(result["total_return"] - 0.05) < 0.001
        assert "volatility" in result
        assert "sharpe_ratio" in result
        assert "max_drawdown" in result
        assert "win_rate" in result


# =========================================================================
# NOTIFICATION TASKS
# =========================================================================

class TestSendDailySummaries:
    """Tests for send_daily_summaries task."""

    @patch("backend.tasks.notification_tasks.send_email")
    @patch("backend.tasks.notification_tasks.gather_daily_summary_data")
    @patch("backend.tasks.notification_tasks.get_db_sync")
    def test_user_with_disabled_summaries_skipped(self, mock_get_db, mock_gather, mock_send):
        from backend.tasks.notification_tasks import send_daily_summaries

        user = _make_user()
        user.notification_settings = {"email_daily_summary": False}

        db = MagicMock()
        db.query.return_value.filter.return_value.all.return_value = [user]
        mock_get_db.return_value = FakeDBContext(db)

        result = send_daily_summaries()
        assert result["sent"] == 0
        mock_gather.assert_not_called()

    @patch("backend.tasks.notification_tasks.send_email")
    @patch("backend.tasks.notification_tasks.gather_daily_summary_data")
    @patch("backend.tasks.notification_tasks.get_db_sync")
    def test_valid_user_gets_summary(self, mock_get_db, mock_gather, mock_send):
        from backend.tasks.notification_tasks import send_daily_summaries

        user = _make_user()
        user.notification_settings = {"email_daily_summary": True}

        db = MagicMock()
        db.query.return_value.filter.return_value.all.return_value = [user]
        mock_get_db.return_value = FakeDBContext(db)

        mock_gather.return_value = {
            "date": "January 01, 2026",
            "portfolios": [],
            "recommendations": [],
            "market_overview": [],
            "news": [],
        }

        # send_email is a Celery task, so it has .delay
        mock_send.delay = MagicMock()

        result = send_daily_summaries()
        assert result["sent"] == 1
        mock_send.delay.assert_called_once()


class TestCheckPriceAlerts:
    """Tests for check_price_alerts task."""

    @patch("backend.tasks.notification_tasks.send_alert_notification")
    @patch("backend.tasks.notification_tasks.get_db_sync")
    def test_alert_triggers_above_threshold(self, mock_get_db, mock_send_alert):
        from backend.tasks.notification_tasks import check_price_alerts

        alert = _make_alert(condition={"type": "price_above", "value": 145.0})
        stock = _make_stock()
        price = _make_price_history(close=150.0)
        user = _make_user()

        db = MagicMock()
        call_count = {"n": 0}

        def query_side_effect(model):
            chain = MagicMock()
            chain.filter.return_value = chain
            chain.order_by.return_value = chain
            if call_count["n"] == 0:
                # Alert query
                chain.all.return_value = [alert]
                call_count["n"] += 1
            elif call_count["n"] == 1:
                # Price history
                chain.first.return_value = price
                call_count["n"] += 1
            elif call_count["n"] == 2:
                # Stock query
                chain.first.return_value = stock
                call_count["n"] += 1
            elif call_count["n"] == 3:
                # User query
                chain.first.return_value = user
                call_count["n"] += 1
            else:
                chain.first.return_value = None
                call_count["n"] += 1
            return chain

        db.query.side_effect = query_side_effect
        mock_get_db.return_value = FakeDBContext(db)

        mock_send_alert.delay = MagicMock()

        result = check_price_alerts()
        assert result["alerts_checked"] == 1
        assert result["alerts_triggered"] == 1
        mock_send_alert.delay.assert_called_once()

    @patch("backend.tasks.notification_tasks.send_alert_notification")
    @patch("backend.tasks.notification_tasks.get_db_sync")
    def test_alert_below_threshold_triggers(self, mock_get_db, mock_send_alert):
        from backend.tasks.notification_tasks import check_price_alerts

        alert = _make_alert(condition={"type": "price_below", "value": 160.0})
        stock = _make_stock()
        price = _make_price_history(close=155.0)  # Below 160
        user = _make_user()

        db = MagicMock()
        call_count = {"n": 0}

        def query_side_effect(model):
            chain = MagicMock()
            chain.filter.return_value = chain
            chain.order_by.return_value = chain
            if call_count["n"] == 0:
                chain.all.return_value = [alert]
                call_count["n"] += 1
            elif call_count["n"] == 1:
                chain.first.return_value = price
                call_count["n"] += 1
            elif call_count["n"] == 2:
                chain.first.return_value = stock
                call_count["n"] += 1
            elif call_count["n"] == 3:
                chain.first.return_value = user
                call_count["n"] += 1
            else:
                chain.first.return_value = None
                call_count["n"] += 1
            return chain

        db.query.side_effect = query_side_effect
        mock_get_db.return_value = FakeDBContext(db)
        mock_send_alert.delay = MagicMock()

        result = check_price_alerts()
        assert result["alerts_triggered"] == 1

    @patch("backend.tasks.notification_tasks.send_alert_notification")
    @patch("backend.tasks.notification_tasks.get_db_sync")
    def test_alert_not_triggered_when_price_not_met(self, mock_get_db, mock_send_alert):
        from backend.tasks.notification_tasks import check_price_alerts

        # price_above 200 but current price is only 150
        alert = _make_alert(condition={"type": "price_above", "value": 200.0})
        price = _make_price_history(close=150.0)

        db = MagicMock()
        call_count = {"n": 0}

        def query_side_effect(model):
            chain = MagicMock()
            chain.filter.return_value = chain
            chain.order_by.return_value = chain
            if call_count["n"] == 0:
                chain.all.return_value = [alert]
                call_count["n"] += 1
            elif call_count["n"] == 1:
                chain.first.return_value = price
                call_count["n"] += 1
            else:
                chain.first.return_value = None
                call_count["n"] += 1
            return chain

        db.query.side_effect = query_side_effect
        mock_get_db.return_value = FakeDBContext(db)
        mock_send_alert.delay = MagicMock()

        result = check_price_alerts()
        assert result["alerts_triggered"] == 0
        mock_send_alert.delay.assert_not_called()

    @patch("backend.tasks.notification_tasks.get_db_sync")
    def test_no_active_alerts(self, mock_get_db):
        from backend.tasks.notification_tasks import check_price_alerts

        db = MagicMock()
        db.query.return_value.filter.return_value.all.return_value = []
        mock_get_db.return_value = FakeDBContext(db)

        result = check_price_alerts()
        assert result["alerts_checked"] == 0
        assert result["alerts_triggered"] == 0


class TestSendAlertNotification:
    """Tests for send_alert_notification task."""

    @patch("backend.tasks.notification_tasks.send_email")
    def test_sends_formatted_email(self, mock_send_email):
        from backend.tasks.notification_tasks import send_alert_notification

        mock_send_email.return_value = True
        result = send_alert_notification(
            "user@example.com", "price_above", "AAPL", "Price rose above $155", 160.0
        )
        assert result is True
        mock_send_email.assert_called_once()
        call_args = mock_send_email.call_args
        assert "AAPL" in call_args[0][1]  # subject


class TestSendEmail:
    """Tests for send_email task."""

    @patch("backend.tasks.notification_tasks.ENABLE_EMAIL", False)
    def test_email_disabled_returns_true(self):
        from backend.tasks.notification_tasks import send_email
        result = send_email("test@example.com", "Test", "<p>Hello</p>")
        assert result is True

    @patch("backend.tasks.notification_tasks.ENABLE_EMAIL", True)
    @patch("backend.tasks.notification_tasks.smtplib.SMTP")
    def test_email_enabled_sends(self, mock_smtp):
        from backend.tasks.notification_tasks import send_email

        mock_server = MagicMock()
        mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp.return_value.__exit__ = MagicMock(return_value=False)

        result = send_email("test@example.com", "Subject", "<p>Body</p>")
        assert result is True


# =========================================================================
# MAINTENANCE TASKS
# =========================================================================

class TestCleanupOldData:
    """Tests for cleanup_old_data task."""

    @patch("backend.tasks.maintenance_tasks.cleanup_old_backups")
    @patch("backend.tasks.maintenance_tasks.cleanup_redis_cache")
    @patch("backend.tasks.maintenance_tasks.get_redis_client")
    @patch("backend.tasks.maintenance_tasks.get_db_sync")
    def test_cleanup_deletes_old_records(self, mock_get_db, mock_redis, mock_cache_cleanup, mock_backup_cleanup):
        from backend.tasks.maintenance_tasks import cleanup_old_data

        db = MagicMock()
        # Each db.query().filter().delete() returns count of deleted rows
        db.query.return_value.filter.return_value.delete.return_value = 5
        mock_get_db.return_value = FakeDBContext(db)

        redis_client = MagicMock()
        mock_redis.return_value = redis_client
        mock_cache_cleanup.return_value = 3
        mock_backup_cleanup.return_value = 1

        result = cleanup_old_data()
        assert "error" not in result or result.get("errors") == []
        assert result["price_history"] == 5
        assert result["news"] == 5
        assert result["recommendations"] == 5
        assert result["cache_keys_removed"] == 3
        assert result["backups_removed"] == 1

    @patch("backend.tasks.maintenance_tasks.cleanup_old_backups")
    @patch("backend.tasks.maintenance_tasks.cleanup_redis_cache")
    @patch("backend.tasks.maintenance_tasks.get_redis_client")
    @patch("backend.tasks.maintenance_tasks.get_db_sync")
    def test_cleanup_handles_partial_failures(self, mock_get_db, mock_redis,
                                               mock_cache_cleanup, mock_backup_cleanup):
        from backend.tasks.maintenance_tasks import cleanup_old_data

        db = MagicMock()
        # First delete succeeds, rest raise exceptions
        delete_mock = MagicMock()
        delete_call_count = {"n": 0}

        def delete_side_effect():
            delete_call_count["n"] += 1
            if delete_call_count["n"] == 1:
                return 10
            raise Exception("DB error")

        db.query.return_value.filter.return_value.delete.side_effect = delete_side_effect
        mock_get_db.return_value = FakeDBContext(db)

        mock_redis.return_value = MagicMock()
        mock_cache_cleanup.return_value = 0
        mock_backup_cleanup.return_value = 0

        result = cleanup_old_data()
        # Should have captured errors but not crashed
        assert len(result["errors"]) > 0
        assert result["price_history"] == 10


class TestClearCache:
    """Tests for clear_cache task."""

    @patch("backend.tasks.maintenance_tasks.get_redis_client")
    def test_clear_all_keys(self, mock_redis):
        from backend.tasks.maintenance_tasks import clear_cache

        redis_client = MagicMock()
        redis_client.keys.return_value = [b"key1", b"key2", b"key3"]
        mock_redis.return_value = redis_client

        result = clear_cache()
        assert result["deleted_keys"] == 3
        assert result["pattern"] == "all"

    @patch("backend.tasks.maintenance_tasks.get_redis_client")
    def test_clear_with_pattern(self, mock_redis):
        from backend.tasks.maintenance_tasks import clear_cache

        redis_client = MagicMock()
        redis_client.keys.return_value = [b"analysis:AAPL", b"analysis:GOOG"]
        mock_redis.return_value = redis_client

        result = clear_cache("analysis:*")
        assert result["deleted_keys"] == 2
        assert result["pattern"] == "analysis:*"
        redis_client.keys.assert_called_with("analysis:*")

    @patch("backend.tasks.maintenance_tasks.get_redis_client")
    def test_clear_cache_empty(self, mock_redis):
        from backend.tasks.maintenance_tasks import clear_cache

        redis_client = MagicMock()
        redis_client.keys.return_value = []
        mock_redis.return_value = redis_client

        result = clear_cache()
        assert result["deleted_keys"] == 0


class TestGenerateSystemReports:
    """Tests for generate_system_reports task."""

    @patch("backend.tasks.maintenance_tasks.generate_error_report")
    @patch("backend.tasks.maintenance_tasks.generate_performance_report")
    @patch("backend.tasks.maintenance_tasks.generate_api_usage_report")
    @patch("backend.tasks.maintenance_tasks.generate_database_report")
    @patch("builtins.open", create=True)
    @patch("backend.tasks.maintenance_tasks.Path")
    def test_generates_all_report_types(self, mock_path, mock_open, mock_db_report,
                                         mock_api_report, mock_perf_report, mock_err_report):
        from backend.tasks.maintenance_tasks import generate_system_reports

        mock_db_report.return_value = {"total_stocks": 500}
        mock_api_report.return_value = {"total_api_calls": 1000}
        mock_perf_report.return_value = {"avg_response_time_ms": 150}
        mock_err_report.return_value = {"total_errors": 0}

        mock_path.return_value.mkdir.return_value = None

        result = generate_system_reports()
        assert "error" not in result
        assert len(result["reports"]) == 4

        report_types = [r["type"] for r in result["reports"]]
        assert "database" in report_types
        assert "api_usage" in report_types
        assert "performance" in report_types
        assert "errors" in report_types


class TestCleanupRedisCache:
    """Tests for cleanup_redis_cache helper."""

    def test_sets_expiry_on_persistent_keys(self):
        from backend.tasks.maintenance_tasks import cleanup_redis_cache

        redis_client = MagicMock()
        redis_client.keys.return_value = [b"key1", b"key2"]
        # ttl == -1 means no expiry
        redis_client.ttl.return_value = -1

        expired = cleanup_redis_cache(redis_client)
        assert expired == 2
        assert redis_client.expire.call_count == 2


class TestGenerateReasoning:
    """Tests for generate_reasoning helper."""

    def test_with_rsi(self):
        from backend.tasks.analysis_tasks import generate_reasoning
        result = generate_reasoning({
            "analysis": {"technical": {"rsi": 72.5}}
        })
        assert "RSI at 72.5" in result

    def test_with_pe_ratio(self):
        from backend.tasks.analysis_tasks import generate_reasoning
        result = generate_reasoning({
            "analysis": {"fundamental": {"pe_ratio": 25.3}}
        })
        assert "P/E ratio of 25.3" in result

    def test_with_sentiment(self):
        from backend.tasks.analysis_tasks import generate_reasoning
        result = generate_reasoning({
            "analysis": {"sentiment": {"overall": 0.85}}
        })
        assert "Sentiment score 0.85" in result

    def test_empty_analysis(self):
        from backend.tasks.analysis_tasks import generate_reasoning
        result = generate_reasoning({"analysis": {}})
        assert result == "Based on comprehensive analysis"


class TestCreateRecommendation:
    """Tests for create_recommendation task."""

    @patch("backend.tasks.analysis_tasks.get_db_sync")
    def test_stock_not_found_returns_false(self, mock_get_db):
        from backend.tasks.analysis_tasks import create_recommendation

        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = None
        mock_get_db.return_value = FakeDBContext(db)

        result = create_recommendation("ZZZZ", {"overall_score": 85})
        assert result is False

    @patch("backend.tasks.analysis_tasks.get_db_sync")
    def test_no_price_returns_false(self, mock_get_db):
        from backend.tasks.analysis_tasks import create_recommendation

        stock = _make_stock()
        db = MagicMock()
        call_count = {"n": 0}

        def query_side_effect(model):
            chain = MagicMock()
            chain.filter.return_value = chain
            chain.order_by.return_value = chain
            if call_count["n"] == 0:
                chain.first.return_value = stock
                call_count["n"] += 1
            else:
                chain.first.return_value = None
                call_count["n"] += 1
            return chain

        db.query.side_effect = query_side_effect
        mock_get_db.return_value = FakeDBContext(db)

        result = create_recommendation("AAPL", {"overall_score": 85})
        assert result is False

    @patch("backend.tasks.analysis_tasks.get_db_sync")
    def test_strong_buy_recommendation_created(self, mock_get_db):
        from backend.tasks.analysis_tasks import create_recommendation

        stock = _make_stock()
        price = _make_price_history(close=150.0)
        db = MagicMock()
        call_count = {"n": 0}

        def query_side_effect(model):
            chain = MagicMock()
            chain.filter.return_value = chain
            chain.order_by.return_value = chain
            if call_count["n"] == 0:
                chain.first.return_value = stock
                call_count["n"] += 1
            elif call_count["n"] == 1:
                chain.first.return_value = price
                call_count["n"] += 1
            else:
                chain.first.return_value = None
                call_count["n"] += 1
            return chain

        db.query.side_effect = query_side_effect
        mock_get_db.return_value = FakeDBContext(db)

        analysis = {
            "overall_score": 85,
            "analysis": {
                "technical": {"score": 80},
                "fundamental": {"score": 90},
                "sentiment": {"score": 85},
            },
        }
        result = create_recommendation("AAPL", analysis)
        assert result is True
        assert db.add.call_count == 2  # recommendation + performance
        assert db.commit.call_count == 2


class TestCheckStopLosses:
    """Tests for check_stop_losses task."""

    @patch("backend.tasks.portfolio_tasks.get_db_sync")
    def test_stop_loss_triggered(self, mock_get_db):
        from backend.tasks.portfolio_tasks import check_stop_losses

        position = _make_position(quantity=10, avg_cost_basis=100.0, symbol="FAIL")
        # Current price is 85, which is a -15% loss (exceeds -10% threshold)
        price = _make_price_history(close=85.0)

        db = MagicMock()
        call_count = {"n": 0}

        def query_side_effect(model):
            chain = MagicMock()
            chain.filter.return_value = chain
            chain.order_by.return_value = chain
            if call_count["n"] == 0:
                chain.all.return_value = [position]
                call_count["n"] += 1
            else:
                chain.first.return_value = price
                call_count["n"] += 1
            return chain

        db.query.side_effect = query_side_effect
        mock_get_db.return_value = FakeDBContext(db)

        result = check_stop_losses()
        assert result["stop_losses_triggered"] == 1
        assert result["triggered_positions"][0]["symbol"] == "FAIL"
        assert result["triggered_positions"][0]["loss_percent"] < -10

    @patch("backend.tasks.portfolio_tasks.get_db_sync")
    def test_no_stop_loss_when_price_stable(self, mock_get_db):
        from backend.tasks.portfolio_tasks import check_stop_losses

        position = _make_position(quantity=10, avg_cost_basis=100.0)
        price = _make_price_history(close=105.0)  # 5% gain, no stop loss

        db = MagicMock()
        call_count = {"n": 0}

        def query_side_effect(model):
            chain = MagicMock()
            chain.filter.return_value = chain
            chain.order_by.return_value = chain
            if call_count["n"] == 0:
                chain.all.return_value = [position]
                call_count["n"] += 1
            else:
                chain.first.return_value = price
                call_count["n"] += 1
            return chain

        db.query.side_effect = query_side_effect
        mock_get_db.return_value = FakeDBContext(db)

        result = check_stop_losses()
        assert result["stop_losses_triggered"] == 0


class TestUpdateMarketCalendars:
    """Tests for update_market_calendars task."""

    @patch("backend.tasks.maintenance_tasks.get_redis_client")
    def test_stores_calendars_in_redis(self, mock_redis):
        from backend.tasks.maintenance_tasks import update_market_calendars

        redis_client = MagicMock()
        mock_redis.return_value = redis_client

        result = update_market_calendars()
        assert result["status"] == "updated"
        assert "NYSE" in result["exchanges"]
        assert "NASDAQ" in result["exchanges"]
        redis_client.setex.assert_called_once()
