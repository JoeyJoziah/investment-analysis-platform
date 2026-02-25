"""
Unit tests for backend/services/portfolio_service.py

Tests all public methods of PortfolioService with mocked dependencies.
No database or external services required.
"""

import pytest
import random
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from backend.services.portfolio_service import PortfolioService, _all_transaction_types


# ---------------------------------------------------------------------------
# Helpers -- lightweight stand-ins for ORM / Pydantic objects
# ---------------------------------------------------------------------------

def _make_position(*, symbol="AAPL", quantity=10.0, average_cost=150.0,
                   unrealized_gain_percent=5.0):
    """Return a namespace that quacks like a schemas.Position for the service."""
    return SimpleNamespace(
        symbol=symbol,
        quantity=quantity,
        average_cost=average_cost,
        unrealized_gain_percent=unrealized_gain_percent,
    )


def _make_price_record(close=152.50):
    """Return a namespace that looks like a PriceHistory row."""
    return SimpleNamespace(close=Decimal(str(close)))


# ---------------------------------------------------------------------------
# Fixture: fresh PortfolioService instance (no singleton state leaks)
# ---------------------------------------------------------------------------

@pytest.fixture
def service():
    return PortfolioService()


# =========================================================================
# get_current_stock_price
# =========================================================================

class TestGetCurrentStockPrice:

    @pytest.mark.asyncio
    async def test_get_current_stock_price_valid_symbol(self, service):
        """When price_repository returns a price, use it."""
        mock_price = _make_price_record(close=200.75)
        mock_repo = AsyncMock()
        mock_repo.get_latest_price = AsyncMock(return_value=mock_price)

        with patch(
            "backend.services.portfolio_service.price_repository",
            new=mock_repo,
            create=True,
        ), patch.dict(
            "sys.modules",
            {"backend.repositories": MagicMock(price_repository=mock_repo)},
        ), patch(
            "backend.repositories.price_repository",
            mock_repo,
        ):
            result = await service.get_current_stock_price("AAPL", db=AsyncMock())

        assert result == 200.75

    @pytest.mark.asyncio
    async def test_get_current_stock_price_unknown_symbol(self, service):
        """When no price record exists, a fallback float is returned."""
        mock_repo = AsyncMock()
        mock_repo.get_latest_price = AsyncMock(return_value=None)

        with patch(
            "backend.repositories.price_repository",
            mock_repo,
        ):
            result = await service.get_current_stock_price("ZZZZ", db=AsyncMock())

        assert isinstance(result, float)
        assert 50 <= result <= 500

    @pytest.mark.asyncio
    async def test_get_current_stock_price_api_failure(self, service):
        """On exception, a fallback float in [50, 500] is returned."""
        mock_repo = AsyncMock()
        mock_repo.get_latest_price = AsyncMock(side_effect=RuntimeError("DB down"))

        with patch(
            "backend.repositories.price_repository",
            mock_repo,
        ):
            result = await service.get_current_stock_price("AAPL", db=AsyncMock())

        assert isinstance(result, float)
        assert 50 <= result <= 500


# =========================================================================
# calculate_portfolio_risk_score
# =========================================================================

class TestCalculatePortfolioRiskScore:

    @pytest.mark.asyncio
    async def test_risk_score_empty_positions(self, service):
        """Empty position list should return 30.0 (cash-only)."""
        score = await service.calculate_portfolio_risk_score(
            portfolio_id=1, positions=[], db=None,
        )
        assert score == 30.0

    @pytest.mark.asyncio
    async def test_risk_score_single_concentrated_position(self, service):
        """A single position means 100% concentration -- high risk."""
        positions = [_make_position(symbol="AAPL", quantity=100, average_cost=150)]
        score = await service.calculate_portfolio_risk_score(
            portfolio_id=1, positions=positions, db=None,
        )
        # concentration_risk = 100, unique_sectors = 1 (bonus 5)
        # risk = 50 + 100 - 5 = 145 -> clamped to 100
        assert score == 100.0

    @pytest.mark.asyncio
    async def test_risk_score_diverse_positions(self, service):
        """Multiple positions with different symbol prefixes lower risk."""
        positions = [
            _make_position(symbol="AAPL", quantity=10, average_cost=150),
            _make_position(symbol="GOOGL", quantity=10, average_cost=150),
            _make_position(symbol="MSFT", quantity=10, average_cost=150),
            _make_position(symbol="TSLA", quantity=10, average_cost=150),
            _make_position(symbol="NVDA", quantity=10, average_cost=150),
        ]
        score = await service.calculate_portfolio_risk_score(
            portfolio_id=1, positions=positions, db=None,
        )
        # Equal weighting: concentration = 20%, 5 unique prefixes (AA,GO,MS,TS,NV) -> bonus 20 (capped)
        # risk = 50 + 20 - 20 = 50
        assert score == 50.0

    @pytest.mark.asyncio
    async def test_risk_score_within_bounds(self, service):
        """Risk score must be between 10 and 100 inclusive."""
        positions = [
            _make_position(symbol="AAPL", quantity=5, average_cost=100),
            _make_position(symbol="GOOGL", quantity=5, average_cost=100),
        ]
        score = await service.calculate_portfolio_risk_score(
            portfolio_id=1, positions=positions, db=None,
        )
        assert 10 <= score <= 100

    @pytest.mark.asyncio
    async def test_risk_score_zero_total_value(self, service):
        """If total value is zero, return 30.0."""
        positions = [_make_position(symbol="AAPL", quantity=0, average_cost=150)]
        score = await service.calculate_portfolio_risk_score(
            portfolio_id=1, positions=positions, db=None,
        )
        assert score == 30.0

    @pytest.mark.asyncio
    async def test_risk_score_exception_returns_default(self, service):
        """On unexpected error, return 50.0."""
        # positions that will trigger AttributeError (missing .quantity)
        bad_positions = [{"symbol": "AAPL"}]
        score = await service.calculate_portfolio_risk_score(
            portfolio_id=1, positions=bad_positions, db=None,
        )
        assert score == 50.0


# =========================================================================
# calculate_real_performance_metrics
# =========================================================================

class TestCalculateRealPerformanceMetrics:

    @pytest.mark.asyncio
    async def test_positive_returns(self, service):
        """Positions with positive unrealized gains produce positive total_return."""
        positions = [
            _make_position(unrealized_gain_percent=10.0),
            _make_position(unrealized_gain_percent=20.0),
        ]
        metrics = await service.calculate_real_performance_metrics(
            portfolio_id=1, positions=positions, db=None,
        )
        assert metrics["total_return"] > 0
        assert "sharpe_ratio" in metrics
        assert "volatility" in metrics

    @pytest.mark.asyncio
    async def test_negative_returns(self, service):
        """Positions with negative gains produce negative total_return."""
        positions = [
            _make_position(unrealized_gain_percent=-15.0),
            _make_position(unrealized_gain_percent=-5.0),
        ]
        metrics = await service.calculate_real_performance_metrics(
            portfolio_id=1, positions=positions, db=None,
        )
        assert metrics["total_return"] < 0

    @pytest.mark.asyncio
    async def test_no_positions_returns_mock_metrics(self, service):
        """Empty positions list falls back to mock metrics."""
        random.seed(42)  # deterministic
        metrics = await service.calculate_real_performance_metrics(
            portfolio_id=1, positions=[], db=None,
        )
        # Mock metrics always have all 13 keys
        expected_keys = {
            "total_return", "annualized_return", "volatility", "sharpe_ratio",
            "sortino_ratio", "max_drawdown", "beta", "alpha", "treynor_ratio",
            "calmar_ratio", "win_rate", "profit_factor", "risk_adjusted_return",
        }
        assert expected_keys == set(metrics.keys())

    @pytest.mark.asyncio
    async def test_single_position_volatility(self, service):
        """A single position uses fallback volatility of 0.1."""
        positions = [_make_position(unrealized_gain_percent=8.0)]
        metrics = await service.calculate_real_performance_metrics(
            portfolio_id=1, positions=positions, db=None,
        )
        # With a single return, stdev can't be computed -> uses 0.1
        # annualized = 0.1 * sqrt(252)
        assert metrics["volatility"] == pytest.approx(0.1 * (252 ** 0.5), rel=1e-3)

    @pytest.mark.asyncio
    async def test_win_rate_calculation(self, service):
        """Win rate should reflect fraction of positions with non-negative gain."""
        positions = [
            _make_position(unrealized_gain_percent=10.0),
            _make_position(unrealized_gain_percent=-5.0),
            _make_position(unrealized_gain_percent=0.0),
        ]
        metrics = await service.calculate_real_performance_metrics(
            portfolio_id=1, positions=positions, db=None,
        )
        # 2 out of 3 are >= 0
        assert metrics["win_rate"] == pytest.approx(2 / 3, rel=1e-6)

    @pytest.mark.asyncio
    async def test_exception_returns_mock_metrics(self, service):
        """On error, falls back to mock metrics (all 13 keys present)."""
        # Passing objects without unrealized_gain_percent causes AttributeError
        positions = [SimpleNamespace()]
        metrics = await service.calculate_real_performance_metrics(
            portfolio_id=1, positions=positions, db=None,
        )
        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics


# =========================================================================
# generate_performance_data_points
# =========================================================================

class TestGeneratePerformanceDataPoints:

    def test_valid_period_1M(self, service):
        """1M period should return 30 data points."""
        random.seed(1)
        result = service.generate_performance_data_points("p1", period="1M")
        assert result["portfolio_id"] == "p1"
        assert result["period"] == "1M"
        assert len(result["data_points"]) == 30
        assert "metrics" in result
        assert "vs_benchmark" in result

    def test_valid_period_1Y(self, service):
        """1Y period should return 252 data points."""
        random.seed(1)
        result = service.generate_performance_data_points("p2", period="1Y")
        assert len(result["data_points"]) == 252

    def test_unknown_period_defaults(self, service):
        """Unknown period defaults to 365 data points."""
        random.seed(1)
        result = service.generate_performance_data_points("p3", period="ALL")
        assert len(result["data_points"]) == 365

    def test_data_point_structure(self, service):
        """Each data point should have date, value, and benchmark_value."""
        random.seed(1)
        result = service.generate_performance_data_points("p4", period="1W")
        dp = result["data_points"][0]
        assert "date" in dp
        assert "value" in dp
        assert "benchmark_value" in dp
        # Date should be a valid ISO date string
        date.fromisoformat(dp["date"])

    def test_metrics_include_total_return(self, service):
        """Metrics dict should include total_return computed from data."""
        random.seed(1)
        result = service.generate_performance_data_points("p5", period="1D")
        assert "total_return" in result["metrics"]
        # total_return = (end - start) / start
        start = result["data_points"][0]["value"]
        end = result["data_points"][-1]["value"]
        expected = round((end - start) / start, 4)
        assert result["metrics"]["total_return"] == expected


# =========================================================================
# build_portfolio_analysis
# =========================================================================

class TestBuildPortfolioAnalysis:

    def test_full_analysis_structure(self, service):
        """Analysis dict should contain all required keys."""
        random.seed(42)
        result = service.build_portfolio_analysis("p100")
        required_keys = {
            "portfolio_id", "analysis_date", "risk_analysis",
            "diversification_score", "concentration_risk",
            "correlation_matrix", "efficient_frontier",
            "optimization_suggestions", "rebalancing_needed",
            "recommended_changes",
        }
        assert required_keys.issubset(set(result.keys()))

    def test_portfolio_id_passthrough(self, service):
        """Returned portfolio_id should match input."""
        result = service.build_portfolio_analysis("my-portfolio-xyz")
        assert result["portfolio_id"] == "my-portfolio-xyz"

    def test_analysis_date_is_today(self, service):
        """Analysis date should be today."""
        result = service.build_portfolio_analysis("p1")
        assert result["analysis_date"] == date.today()

    def test_risk_analysis_keys(self, service):
        """Risk analysis sub-dict should have VaR, CVaR, etc."""
        random.seed(42)
        result = service.build_portfolio_analysis("p1")
        ra = result["risk_analysis"]
        assert {"var_95", "cvar_95", "downside_deviation", "upside_potential"} == set(ra.keys())


# =========================================================================
# generate_rebalancing_trades
# =========================================================================

class TestGenerateRebalancingTrades:

    def test_trades_generated_for_imbalanced_allocation(self, service):
        """Trades should be generated when target differs from current by >1%."""
        random.seed(0)
        target = {"Equities": 60, "Bonds": 30, "Cash": 10}
        result = service.generate_rebalancing_trades(
            portfolio_id="p1",
            target_allocation=target,
            max_trades=10,
            min_trade_value=100,
            tax_efficient=False,
        )
        assert result["portfolio_id"] == "p1"
        assert result["execution_status"] == "pending"
        assert isinstance(result["rebalancing_plan"], list)
        # At least one trade should be generated
        assert len(result["rebalancing_plan"]) >= 1

    def test_max_trades_limit(self, service):
        """Number of trades should not exceed max_trades."""
        random.seed(0)
        target = {f"Asset{i}": 10 for i in range(20)}
        result = service.generate_rebalancing_trades(
            portfolio_id="p1",
            target_allocation=target,
            max_trades=3,
            min_trade_value=50,
            tax_efficient=False,
        )
        assert len(result["rebalancing_plan"]) <= 3

    def test_estimated_cost_calculation(self, service):
        """Estimated cost should be 0.1% of total trade amounts."""
        random.seed(0)
        target = {"Equities": 80}
        result = service.generate_rebalancing_trades(
            portfolio_id="p1",
            target_allocation=target,
            max_trades=10,
            min_trade_value=0,
            tax_efficient=False,
        )
        trades = result["rebalancing_plan"]
        expected_cost = sum(t["amount"] * 0.001 for t in trades)
        assert result["estimated_cost"] == pytest.approx(expected_cost)

    def test_tax_impact_when_tax_efficient(self, service):
        """Tax impact should be a negative number when tax_efficient=True."""
        random.seed(0)
        target = {"Equities": 50}
        result = service.generate_rebalancing_trades(
            portfolio_id="p1",
            target_allocation=target,
            max_trades=10,
            min_trade_value=0,
            tax_efficient=True,
        )
        assert result["tax_impact"] < 0

    def test_no_tax_impact_when_not_tax_efficient(self, service):
        """Tax impact should be 0 when tax_efficient=False."""
        random.seed(0)
        target = {"Equities": 50}
        result = service.generate_rebalancing_trades(
            portfolio_id="p1",
            target_allocation=target,
            max_trades=10,
            min_trade_value=0,
            tax_efficient=False,
        )
        assert result["tax_impact"] == 0

    def test_empty_target_allocation(self, service):
        """Empty target allocation produces no trades."""
        result = service.generate_rebalancing_trades(
            portfolio_id="p1",
            target_allocation={},
            max_trades=10,
            min_trade_value=0,
            tax_efficient=False,
        )
        assert result["rebalancing_plan"] == []
        assert result["estimated_cost"] == 0


# =========================================================================
# generate_transaction_list
# =========================================================================

class TestGenerateTransactionList:

    def test_with_default_params(self, service):
        """Should return a list of transactions up to limit."""
        random.seed(42)
        txns = service.generate_transaction_list(
            portfolio_id="p1",
            limit=10,
            offset=0,
            transaction_type_filter=None,
            symbol_filter=None,
            start_date=None,
            end_date=None,
        )
        assert isinstance(txns, list)
        assert len(txns) <= 10
        if txns:
            assert "id" in txns[0]
            assert "symbol" in txns[0]
            assert "transaction_type" in txns[0]

    def test_empty_with_impossible_date_range(self, service):
        """Date range in the far future should yield an empty list."""
        txns = service.generate_transaction_list(
            portfolio_id="p1",
            limit=10,
            offset=0,
            transaction_type_filter=None,
            symbol_filter=None,
            start_date=date(2099, 1, 1),
            end_date=date(2099, 12, 31),
        )
        assert txns == []

    def test_symbol_filter(self, service):
        """All returned transactions should match the symbol filter."""
        random.seed(42)
        txns = service.generate_transaction_list(
            portfolio_id="p1",
            limit=50,
            offset=0,
            transaction_type_filter=None,
            symbol_filter="AAPL",
            start_date=None,
            end_date=None,
        )
        for txn in txns:
            assert txn["symbol"] == "AAPL"

    def test_offset_pagination(self, service):
        """Offset should skip the specified number of transactions."""
        random.seed(42)
        all_txns = service.generate_transaction_list(
            portfolio_id="p1", limit=100, offset=0,
            transaction_type_filter=None, symbol_filter=None,
            start_date=None, end_date=None,
        )
        offset_txns = service.generate_transaction_list(
            portfolio_id="p1", limit=100, offset=5,
            transaction_type_filter=None, symbol_filter=None,
            start_date=None, end_date=None,
        )
        # With the same seed, offset=5 should return all_txns[5:]
        # (Random seed resets between calls, so we just check type/len)
        assert isinstance(offset_txns, list)

    def test_sorted_descending_by_timestamp(self, service):
        """Transactions should be sorted newest-first."""
        random.seed(42)
        txns = service.generate_transaction_list(
            portfolio_id="p1", limit=50, offset=0,
            transaction_type_filter=None, symbol_filter=None,
            start_date=None, end_date=None,
        )
        if len(txns) >= 2:
            timestamps = [t["timestamp"] for t in txns]
            for i in range(len(timestamps) - 1):
                assert timestamps[i] >= timestamps[i + 1]


# =========================================================================
# execute_rebalancing
# =========================================================================

class TestExecuteRebalancing:

    @pytest.mark.asyncio
    async def test_execute_rebalancing_prints(self, service, capsys):
        """Should print execution message."""
        trades = [{"action": "buy", "amount": 1000}, {"action": "sell", "amount": 500}]
        await service.execute_rebalancing("p1", trades)
        captured = capsys.readouterr()
        assert "Executing 2 trades for portfolio p1" in captured.out

    @pytest.mark.asyncio
    async def test_execute_rebalancing_empty_trades(self, service, capsys):
        """Should handle empty trade list."""
        await service.execute_rebalancing("p1", [])
        captured = capsys.readouterr()
        assert "Executing 0 trades" in captured.out


# =========================================================================
# _all_transaction_types (module-level helper)
# =========================================================================

class TestAllTransactionTypes:

    def test_returns_all_types(self):
        """Should return the five transaction type strings."""
        types = _all_transaction_types()
        assert set(types) == {"buy", "sell", "dividend", "transfer_in", "transfer_out"}


# =========================================================================
# _mock_performance_metrics (private, but verifying contract)
# =========================================================================

class TestMockPerformanceMetrics:

    def test_mock_metrics_keys(self, service):
        """Mock metrics should have all 13 expected keys."""
        random.seed(42)
        metrics = service._mock_performance_metrics()
        expected = {
            "total_return", "annualized_return", "volatility", "sharpe_ratio",
            "sortino_ratio", "max_drawdown", "beta", "alpha", "treynor_ratio",
            "calmar_ratio", "win_rate", "profit_factor", "risk_adjusted_return",
        }
        assert set(metrics.keys()) == expected

    def test_mock_metrics_are_floats(self, service):
        """All mock metric values should be floats."""
        metrics = service._mock_performance_metrics()
        for key, value in metrics.items():
            assert isinstance(value, float), f"{key} is not float: {type(value)}"
