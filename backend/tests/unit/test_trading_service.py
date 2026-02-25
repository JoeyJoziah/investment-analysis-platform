"""
Unit tests for backend/services/trading_service.py

Tests all public methods of TradingService with mocked dependencies.
No database or external services required.
"""

import pytest
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from backend.services.trading_service import (
    TradingService,
    OrderType,
    OrderSide,
    trading_service,
)


# ---------------------------------------------------------------------------
# Helpers -- lightweight stand-ins for ORM objects
# ---------------------------------------------------------------------------

def _make_portfolio(*, cash_balance=10000.0, portfolio_id=1):
    """Return a namespace that quacks like a Portfolio ORM object."""
    return SimpleNamespace(
        id=portfolio_id,
        cash_balance=Decimal(str(cash_balance)),
    )


def _make_position(*, position_id=1):
    """Return a namespace that quacks like a Position ORM object."""
    return SimpleNamespace(id=position_id)


def _valid_buy_order():
    """Return a minimal valid buy order dict."""
    return {
        'portfolio_id': 1,
        'symbol': 'AAPL',
        'side': OrderSide.BUY,
        'order_type': OrderType.MARKET,
        'quantity': 10,
        'price': 150.0,
    }


def _valid_sell_order():
    """Return a minimal valid sell order dict."""
    return {
        'portfolio_id': 1,
        'symbol': 'AAPL',
        'side': OrderSide.SELL,
        'order_type': OrderType.MARKET,
        'quantity': 5,
        'price': 160.0,
    }


# ---------------------------------------------------------------------------
# Fixture: fresh TradingService instance (no singleton state leaks)
# ---------------------------------------------------------------------------

@pytest.fixture
def service():
    return TradingService()


@pytest.fixture
def mock_repo():
    """Return an AsyncMock standing in for portfolio_repository."""
    repo = AsyncMock()
    repo.get_by_id = AsyncMock(return_value=_make_portfolio())
    repo.add_position = AsyncMock(return_value=_make_position())
    repo.get_portfolio_allocation = AsyncMock(return_value={
        'sector_allocation': {'Technology': 60.0, 'Healthcare': 40.0}
    })
    repo.calculate_portfolio_value = AsyncMock(return_value={
        'total_value': 50000.0,
    })
    return repo


# =========================================================================
# OrderType and OrderSide enums
# =========================================================================

class TestEnums:

    def test_order_type_values(self):
        """OrderType enum should contain market, limit, stop, stop_limit."""
        assert OrderType.MARKET == "market"
        assert OrderType.LIMIT == "limit"
        assert OrderType.STOP == "stop"
        assert OrderType.STOP_LIMIT == "stop_limit"

    def test_order_side_values(self):
        """OrderSide enum should contain buy and sell."""
        assert OrderSide.BUY == "buy"
        assert OrderSide.SELL == "sell"

    def test_order_type_is_str_enum(self):
        """OrderType values should be usable as strings."""
        assert isinstance(OrderType.MARKET, str)
        assert OrderType.LIMIT.upper() == "LIMIT"

    def test_order_side_is_str_enum(self):
        """OrderSide values should be usable as strings."""
        assert isinstance(OrderSide.BUY, str)
        assert OrderSide.SELL.upper() == "SELL"


# =========================================================================
# validate_order
# =========================================================================

class TestValidateOrder:

    @pytest.mark.asyncio
    async def test_valid_market_buy_order(self, service, mock_repo):
        """A complete market buy order with sufficient cash should be valid."""
        with patch.object(service, 'repository', mock_repo):
            result = await service.validate_order(_valid_buy_order())
        assert result['valid'] is True
        assert 'message' in result

    @pytest.mark.asyncio
    async def test_valid_market_sell_order(self, service, mock_repo):
        """A complete market sell order with valid portfolio should pass."""
        with patch.object(service, 'repository', mock_repo):
            result = await service.validate_order(_valid_sell_order())
        assert result['valid'] is True

    @pytest.mark.asyncio
    async def test_missing_portfolio_id(self, service, mock_repo):
        """Missing portfolio_id field should produce a validation error."""
        order = _valid_buy_order()
        del order['portfolio_id']
        with patch.object(service, 'repository', mock_repo):
            result = await service.validate_order(order)
        assert result['valid'] is False
        assert any('portfolio_id' in e for e in result['errors'])

    @pytest.mark.asyncio
    async def test_missing_symbol(self, service, mock_repo):
        """Missing symbol field should produce a validation error."""
        order = _valid_buy_order()
        del order['symbol']
        with patch.object(service, 'repository', mock_repo):
            result = await service.validate_order(order)
        assert result['valid'] is False
        assert any('symbol' in e for e in result['errors'])

    @pytest.mark.asyncio
    async def test_missing_side(self, service, mock_repo):
        """Missing side field should produce a validation error."""
        order = _valid_buy_order()
        del order['side']
        with patch.object(service, 'repository', mock_repo):
            result = await service.validate_order(order)
        assert result['valid'] is False
        assert any('side' in e for e in result['errors'])

    @pytest.mark.asyncio
    async def test_missing_order_type(self, service, mock_repo):
        """Missing order_type should produce a validation error."""
        order = _valid_buy_order()
        del order['order_type']
        with patch.object(service, 'repository', mock_repo):
            result = await service.validate_order(order)
        assert result['valid'] is False
        assert any('order_type' in e for e in result['errors'])

    @pytest.mark.asyncio
    async def test_missing_quantity(self, service, mock_repo):
        """Missing quantity should produce a validation error."""
        order = _valid_buy_order()
        del order['quantity']
        with patch.object(service, 'repository', mock_repo):
            result = await service.validate_order(order)
        assert result['valid'] is False
        assert any('quantity' in e for e in result['errors'])

    @pytest.mark.asyncio
    async def test_missing_multiple_fields(self, service, mock_repo):
        """Missing multiple fields should produce one error per field."""
        order = {'symbol': 'AAPL'}
        with patch.object(service, 'repository', mock_repo):
            result = await service.validate_order(order)
        assert result['valid'] is False
        assert len(result['errors']) >= 4  # portfolio_id, side, order_type, quantity

    @pytest.mark.asyncio
    async def test_zero_quantity_rejected(self, service, mock_repo):
        """Quantity of 0 should produce a validation error."""
        order = _valid_buy_order()
        order['quantity'] = 0
        with patch.object(service, 'repository', mock_repo):
            result = await service.validate_order(order)
        assert result['valid'] is False
        assert any('Quantity' in e for e in result['errors'])

    @pytest.mark.asyncio
    async def test_negative_quantity_rejected(self, service, mock_repo):
        """Negative quantity should produce a validation error."""
        order = _valid_buy_order()
        order['quantity'] = -5
        with patch.object(service, 'repository', mock_repo):
            result = await service.validate_order(order)
        assert result['valid'] is False
        assert any('Quantity' in e for e in result['errors'])

    @pytest.mark.asyncio
    async def test_limit_order_requires_price(self, service, mock_repo):
        """Limit orders without a price should produce a validation error."""
        order = _valid_buy_order()
        order['order_type'] = OrderType.LIMIT
        del order['price']
        with patch.object(service, 'repository', mock_repo):
            result = await service.validate_order(order)
        assert result['valid'] is False
        assert any('price' in e.lower() for e in result['errors'])

    @pytest.mark.asyncio
    async def test_limit_order_with_zero_price_rejected(self, service, mock_repo):
        """Limit order with price=0 should produce a validation error."""
        order = _valid_buy_order()
        order['order_type'] = OrderType.LIMIT
        order['price'] = 0
        with patch.object(service, 'repository', mock_repo):
            result = await service.validate_order(order)
        assert result['valid'] is False

    @pytest.mark.asyncio
    async def test_limit_order_with_negative_price_rejected(self, service, mock_repo):
        """Limit order with negative price should be rejected."""
        order = _valid_buy_order()
        order['order_type'] = OrderType.LIMIT
        order['price'] = -10.0
        with patch.object(service, 'repository', mock_repo):
            result = await service.validate_order(order)
        assert result['valid'] is False
        assert any('Price must be greater than 0' in e for e in result['errors'])

    @pytest.mark.asyncio
    async def test_stop_order_requires_stop_price(self, service, mock_repo):
        """Stop orders without a stop_price should produce a validation error."""
        order = _valid_buy_order()
        order['order_type'] = OrderType.STOP
        with patch.object(service, 'repository', mock_repo):
            result = await service.validate_order(order)
        assert result['valid'] is False
        assert any('stop_price' in e.lower() for e in result['errors'])

    @pytest.mark.asyncio
    async def test_stop_order_with_negative_stop_price_rejected(self, service, mock_repo):
        """Stop order with negative stop_price should be rejected."""
        order = _valid_buy_order()
        order['order_type'] = OrderType.STOP
        order['stop_price'] = -5.0
        with patch.object(service, 'repository', mock_repo):
            result = await service.validate_order(order)
        assert result['valid'] is False
        assert any('Stop price must be greater than 0' in e for e in result['errors'])

    @pytest.mark.asyncio
    async def test_stop_limit_requires_both_prices(self, service, mock_repo):
        """Stop-limit orders require both price and stop_price."""
        order = _valid_buy_order()
        order['order_type'] = OrderType.STOP_LIMIT
        # Has price from _valid_buy_order, but no stop_price
        with patch.object(service, 'repository', mock_repo):
            result = await service.validate_order(order)
        assert result['valid'] is False
        assert any('stop_price' in e.lower() for e in result['errors'])

    @pytest.mark.asyncio
    async def test_stop_limit_valid_with_both_prices(self, service, mock_repo):
        """Stop-limit order with valid price and stop_price should pass."""
        order = _valid_buy_order()
        order['order_type'] = OrderType.STOP_LIMIT
        order['price'] = 145.0
        order['stop_price'] = 140.0
        with patch.object(service, 'repository', mock_repo):
            result = await service.validate_order(order)
        assert result['valid'] is True

    @pytest.mark.asyncio
    async def test_invalid_symbol_numeric(self, service, mock_repo):
        """Numeric symbol should be rejected."""
        order = _valid_buy_order()
        order['symbol'] = '12345'
        with patch.object(service, 'repository', mock_repo):
            result = await service.validate_order(order)
        assert result['valid'] is False
        assert any('symbol' in e.lower() for e in result['errors'])

    @pytest.mark.asyncio
    async def test_invalid_symbol_too_long(self, service, mock_repo):
        """Symbol longer than 5 characters should be rejected."""
        order = _valid_buy_order()
        order['symbol'] = 'ABCDEF'
        with patch.object(service, 'repository', mock_repo):
            result = await service.validate_order(order)
        assert result['valid'] is False
        assert any('symbol' in e.lower() for e in result['errors'])

    @pytest.mark.asyncio
    async def test_invalid_symbol_empty(self, service, mock_repo):
        """Empty symbol string should be rejected."""
        order = _valid_buy_order()
        order['symbol'] = ''
        with patch.object(service, 'repository', mock_repo):
            result = await service.validate_order(order)
        assert result['valid'] is False

    @pytest.mark.asyncio
    async def test_invalid_symbol_special_chars(self, service, mock_repo):
        """Symbol with special characters should be rejected."""
        order = _valid_buy_order()
        order['symbol'] = 'AA-L'
        with patch.object(service, 'repository', mock_repo):
            result = await service.validate_order(order)
        assert result['valid'] is False

    @pytest.mark.asyncio
    async def test_portfolio_not_found(self, service, mock_repo):
        """Non-existent portfolio should produce a validation error."""
        mock_repo.get_by_id = AsyncMock(return_value=None)
        with patch.object(service, 'repository', mock_repo):
            result = await service.validate_order(_valid_buy_order())
        assert result['valid'] is False
        assert any('not found' in e for e in result['errors'])

    @pytest.mark.asyncio
    async def test_insufficient_cash_for_buy(self, service, mock_repo):
        """Buy order exceeding cash balance should be rejected."""
        mock_repo.get_by_id = AsyncMock(
            return_value=_make_portfolio(cash_balance=100.0)
        )
        order = _valid_buy_order()
        order['quantity'] = 100  # 100 * 150.0 = 15000 > 100
        with patch.object(service, 'repository', mock_repo):
            result = await service.validate_order(order)
        assert result['valid'] is False
        assert any('Insufficient' in e for e in result['errors'])

    @pytest.mark.asyncio
    async def test_buy_with_exact_cash_succeeds(self, service, mock_repo):
        """Buy order with exactly enough cash should succeed."""
        mock_repo.get_by_id = AsyncMock(
            return_value=_make_portfolio(cash_balance=1500.0)
        )
        order = _valid_buy_order()
        order['quantity'] = 10
        order['price'] = 150.0  # 10 * 150 = 1500 == 1500
        with patch.object(service, 'repository', mock_repo):
            result = await service.validate_order(order)
        assert result['valid'] is True

    @pytest.mark.asyncio
    async def test_sell_order_passes_validation_with_portfolio(self, service, mock_repo):
        """Sell order with existing portfolio should pass basic validation."""
        with patch.object(service, 'repository', mock_repo):
            result = await service.validate_order(_valid_sell_order())
        assert result['valid'] is True

    @pytest.mark.asyncio
    async def test_exception_during_validation_returns_error(self, service, mock_repo):
        """Unexpected exceptions should be caught and returned as errors."""
        mock_repo.get_by_id = AsyncMock(side_effect=RuntimeError("DB connection lost"))
        with patch.object(service, 'repository', mock_repo):
            result = await service.validate_order(_valid_buy_order())
        assert result['valid'] is False
        assert any('DB connection lost' in e for e in result['errors'])

    @pytest.mark.asyncio
    async def test_empty_order_data(self, service, mock_repo):
        """Empty order dict should fail with multiple missing field errors."""
        with patch.object(service, 'repository', mock_repo):
            result = await service.validate_order({})
        assert result['valid'] is False
        assert len(result['errors']) == 5  # all 5 required fields

    @pytest.mark.asyncio
    async def test_market_buy_no_price_uses_zero_for_cost(self, service, mock_repo):
        """Market buy without price field uses 0 for estimated cost check."""
        order = _valid_buy_order()
        del order['price']
        # quantity=10, price defaults to 0 => cost=0, always passes cash check
        with patch.object(service, 'repository', mock_repo):
            result = await service.validate_order(order)
        assert result['valid'] is True


# =========================================================================
# execute_trade
# =========================================================================

class TestExecuteTrade:

    @pytest.mark.asyncio
    async def test_successful_buy_execution(self, service, mock_repo):
        """A valid buy order should execute and return success with trade details."""
        with patch.object(service, 'repository', mock_repo):
            result = await service.execute_trade(1, {
                'symbol': 'AAPL',
                'side': OrderSide.BUY,
                'order_type': OrderType.MARKET,
                'quantity': 10,
                'price': 150.0,
            })
        assert result['success'] is True
        assert result['symbol'] == 'AAPL'
        assert result['side'] == OrderSide.BUY
        assert result['quantity'] == 10.0
        assert result['price'] == 150.0
        assert result['total_cost'] == 1500.0
        assert result['portfolio_id'] == 1
        assert 'executed_at' in result
        assert 'trade_id' in result

    @pytest.mark.asyncio
    async def test_buy_trade_calls_add_position(self, service, mock_repo):
        """Buy execution should call repository.add_position with correct args."""
        with patch.object(service, 'repository', mock_repo):
            await service.execute_trade(1, {
                'symbol': 'AAPL',
                'side': OrderSide.BUY,
                'order_type': OrderType.MARKET,
                'quantity': 5,
                'price': 200.0,
            })
        mock_repo.add_position.assert_called_once_with(
            portfolio_id=1,
            stock_id=1,  # placeholder
            quantity=Decimal('5'),
            price=Decimal('200.0'),
            transaction_type='buy',
        )

    @pytest.mark.asyncio
    async def test_sell_trade_returns_success(self, service, mock_repo):
        """Sell execution should return success even though position is None."""
        with patch.object(service, 'repository', mock_repo):
            result = await service.execute_trade(1, {
                'symbol': 'AAPL',
                'side': OrderSide.SELL,
                'order_type': OrderType.MARKET,
                'quantity': 5,
                'price': 160.0,
            })
        assert result['success'] is True
        assert result['trade_id'] is None  # sell returns position=None
        assert result['total_cost'] == 800.0

    @pytest.mark.asyncio
    async def test_execute_trade_validation_failure(self, service, mock_repo):
        """Invalid order data should fail at validation step."""
        mock_repo.get_by_id = AsyncMock(return_value=None)
        with patch.object(service, 'repository', mock_repo):
            result = await service.execute_trade(1, {
                'symbol': 'AAPL',
                'side': OrderSide.BUY,
                'order_type': OrderType.MARKET,
                'quantity': 10,
                'price': 150.0,
            })
        assert result['success'] is False
        assert 'validation_errors' in result

    @pytest.mark.asyncio
    async def test_execute_trade_add_position_returns_none(self, service, mock_repo):
        """If add_position returns None for a buy, trade should fail."""
        mock_repo.add_position = AsyncMock(return_value=None)
        with patch.object(service, 'repository', mock_repo):
            result = await service.execute_trade(1, {
                'symbol': 'AAPL',
                'side': OrderSide.BUY,
                'order_type': OrderType.MARKET,
                'quantity': 10,
                'price': 150.0,
            })
        assert result['success'] is False
        assert 'Failed to execute trade' in result['error']

    @pytest.mark.asyncio
    async def test_execute_trade_exception(self, service, mock_repo):
        """Unexpected exceptions during execution should be caught gracefully."""
        mock_repo.add_position = AsyncMock(side_effect=RuntimeError("DB write failed"))
        with patch.object(service, 'repository', mock_repo):
            result = await service.execute_trade(1, {
                'symbol': 'AAPL',
                'side': OrderSide.BUY,
                'order_type': OrderType.MARKET,
                'quantity': 10,
                'price': 150.0,
            })
        assert result['success'] is False
        assert 'DB write failed' in result['error']

    @pytest.mark.asyncio
    async def test_executed_at_is_iso_format(self, service, mock_repo):
        """The executed_at timestamp should be a valid ISO 8601 string."""
        with patch.object(service, 'repository', mock_repo):
            result = await service.execute_trade(1, {
                'symbol': 'AAPL',
                'side': OrderSide.BUY,
                'order_type': OrderType.MARKET,
                'quantity': 1,
                'price': 100.0,
            })
        assert result['success'] is True
        # Should not raise
        datetime.fromisoformat(result['executed_at'])

    @pytest.mark.asyncio
    async def test_execute_trade_with_zero_price(self, service, mock_repo):
        """Trade with no price key should use 0 and compute total_cost=0."""
        with patch.object(service, 'repository', mock_repo):
            result = await service.execute_trade(1, {
                'symbol': 'AAPL',
                'side': OrderSide.BUY,
                'order_type': OrderType.MARKET,
                'quantity': 10,
            })
        assert result['success'] is True
        assert result['total_cost'] == 0.0


# =========================================================================
# calculate_portfolio_impact
# =========================================================================

class TestCalculatePortfolioImpact:

    @pytest.mark.asyncio
    async def test_buy_impact_increases_value(self, service, mock_repo):
        """A buy trade should increase total value and decrease cash."""
        with patch.object(service, 'repository', mock_repo):
            result = await service.calculate_portfolio_impact(1, {
                'symbol': 'AAPL',
                'side': OrderSide.BUY,
                'quantity': 10,
                'price': 150.0,
            })
        assert result['success'] is True
        assert result['trade_impact']['impact_type'] == 'increase'
        assert result['after']['total_value'] > result['before']['total_value']
        assert result['after']['cash_balance'] < result['before']['cash_balance']

    @pytest.mark.asyncio
    async def test_sell_impact_decreases_value(self, service, mock_repo):
        """A sell trade should decrease total value and increase cash."""
        with patch.object(service, 'repository', mock_repo):
            result = await service.calculate_portfolio_impact(1, {
                'symbol': 'AAPL',
                'side': OrderSide.SELL,
                'quantity': 5,
                'price': 160.0,
            })
        assert result['success'] is True
        assert result['trade_impact']['impact_type'] == 'decrease'
        assert result['after']['total_value'] < result['before']['total_value']
        assert result['after']['cash_balance'] > result['before']['cash_balance']

    @pytest.mark.asyncio
    async def test_portfolio_not_found_returns_error(self, service, mock_repo):
        """Non-existent portfolio should return an error."""
        mock_repo.get_by_id = AsyncMock(return_value=None)
        with patch.object(service, 'repository', mock_repo):
            result = await service.calculate_portfolio_impact(999, {
                'symbol': 'AAPL',
                'side': OrderSide.BUY,
                'quantity': 10,
                'price': 150.0,
            })
        assert result['success'] is False
        assert 'not found' in result['error']

    @pytest.mark.asyncio
    async def test_impact_allocation_change_calculated(self, service, mock_repo):
        """Allocation change percentage should be calculated correctly."""
        with patch.object(service, 'repository', mock_repo):
            result = await service.calculate_portfolio_impact(1, {
                'symbol': 'AAPL',
                'side': OrderSide.BUY,
                'quantity': 10,
                'price': 150.0,
            })
        # trade_value=1500, total_value=50000 => 3%
        assert result['after']['allocation_change'] == pytest.approx(3.0)

    @pytest.mark.asyncio
    async def test_impact_with_zero_total_value(self, service, mock_repo):
        """Zero total value should give 100% allocation change."""
        mock_repo.calculate_portfolio_value = AsyncMock(
            return_value={'total_value': 0}
        )
        with patch.object(service, 'repository', mock_repo):
            result = await service.calculate_portfolio_impact(1, {
                'symbol': 'AAPL',
                'side': OrderSide.BUY,
                'quantity': 10,
                'price': 150.0,
            })
        assert result['success'] is True
        assert result['after']['allocation_change'] == 100.0

    @pytest.mark.asyncio
    async def test_impact_when_portfolio_value_is_none(self, service, mock_repo):
        """When calculate_portfolio_value returns None, fallback to cash balance."""
        mock_repo.calculate_portfolio_value = AsyncMock(return_value=None)
        with patch.object(service, 'repository', mock_repo):
            result = await service.calculate_portfolio_impact(1, {
                'symbol': 'AAPL',
                'side': OrderSide.BUY,
                'quantity': 1,
                'price': 100.0,
            })
        assert result['success'] is True
        # total_value falls back to cash_balance=10000
        assert result['before']['total_value'] == 10000.0

    @pytest.mark.asyncio
    async def test_impact_metrics_value_change(self, service, mock_repo):
        """Metrics should include correct value change and percentage."""
        with patch.object(service, 'repository', mock_repo):
            result = await service.calculate_portfolio_impact(1, {
                'symbol': 'TSLA',
                'side': OrderSide.BUY,
                'quantity': 20,
                'price': 250.0,
            })
        # trade_value=5000, total=50000 => change=5000, pct=10%
        assert result['metrics']['value_change'] == pytest.approx(5000.0)
        assert result['metrics']['value_change_percent'] == pytest.approx(10.0)

    @pytest.mark.asyncio
    async def test_impact_metrics_cash_utilization(self, service, mock_repo):
        """Cash utilization should reflect percentage of cash used."""
        with patch.object(service, 'repository', mock_repo):
            result = await service.calculate_portfolio_impact(1, {
                'symbol': 'AAPL',
                'side': OrderSide.BUY,
                'quantity': 10,
                'price': 150.0,
            })
        # cash=10000, new_cash=8500, utilization=(10000-8500)/10000*100=15%
        assert result['metrics']['cash_utilization'] == pytest.approx(15.0)

    @pytest.mark.asyncio
    async def test_sell_impact_negative_cash_utilization(self, service, mock_repo):
        """Sell trade increases cash, so cash_utilization should be negative."""
        with patch.object(service, 'repository', mock_repo):
            result = await service.calculate_portfolio_impact(1, {
                'symbol': 'AAPL',
                'side': OrderSide.SELL,
                'quantity': 5,
                'price': 200.0,
            })
        # cash=10000, new_cash=11000, utilization=(10000-11000)/10000*100=-10%
        assert result['metrics']['cash_utilization'] < 0

    @pytest.mark.asyncio
    async def test_impact_exception_returns_error(self, service, mock_repo):
        """Unexpected exceptions should be caught and returned as errors."""
        mock_repo.get_by_id = AsyncMock(side_effect=RuntimeError("Network error"))
        with patch.object(service, 'repository', mock_repo):
            result = await service.calculate_portfolio_impact(1, {
                'symbol': 'AAPL',
                'side': OrderSide.BUY,
                'quantity': 10,
                'price': 150.0,
            })
        assert result['success'] is False
        assert 'Network error' in result['error']

    @pytest.mark.asyncio
    async def test_impact_includes_before_allocation(self, service, mock_repo):
        """Before section should include the current allocation from repository."""
        with patch.object(service, 'repository', mock_repo):
            result = await service.calculate_portfolio_impact(1, {
                'symbol': 'AAPL',
                'side': OrderSide.BUY,
                'quantity': 1,
                'price': 100.0,
            })
        assert 'allocation' in result['before']
        assert result['before']['allocation'] is not None

    @pytest.mark.asyncio
    async def test_impact_trade_details_in_result(self, service, mock_repo):
        """Trade impact section should echo symbol, side, and trade_value."""
        with patch.object(service, 'repository', mock_repo):
            result = await service.calculate_portfolio_impact(1, {
                'symbol': 'GOOG',
                'side': OrderSide.BUY,
                'quantity': 5,
                'price': 300.0,
            })
        impact = result['trade_impact']
        assert impact['symbol'] == 'GOOG'
        assert impact['side'] == OrderSide.BUY
        assert impact['trade_value'] == 1500.0

    @pytest.mark.asyncio
    async def test_impact_with_zero_cash_balance(self, service, mock_repo):
        """Portfolio with zero cash should handle cash_utilization without division error."""
        mock_repo.get_by_id = AsyncMock(
            return_value=_make_portfolio(cash_balance=0.0)
        )
        with patch.object(service, 'repository', mock_repo):
            result = await service.calculate_portfolio_impact(1, {
                'symbol': 'AAPL',
                'side': OrderSide.SELL,
                'quantity': 5,
                'price': 100.0,
            })
        assert result['success'] is True
        assert result['metrics']['cash_utilization'] == 0


# =========================================================================
# Singleton instance
# =========================================================================

class TestSingletonInstance:

    def test_trading_service_is_trading_service(self):
        """Module-level trading_service should be a TradingService instance."""
        assert isinstance(trading_service, TradingService)

    def test_singleton_has_repository(self):
        """Singleton instance should have a repository attribute."""
        assert hasattr(trading_service, 'repository')
