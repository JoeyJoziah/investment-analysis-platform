"""
Unit tests for the trading router.
Tests order validation, trade execution, and portfolio impact endpoints.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock


# ============================================================================
# OrderRequest / TradeRequest / ImpactRequest models
# ============================================================================

class TestOrderRequestModel:
    """Tests for Pydantic request models."""

    def test_order_request_valid(self):
        from backend.api.routers.trading import OrderRequest, OrderSide, OrderType

        req = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10,
        )
        assert req.symbol == "AAPL"
        assert req.quantity == 10

    def test_order_request_with_limit_price(self):
        from backend.api.routers.trading import OrderRequest, OrderSide, OrderType

        req = OrderRequest(
            symbol="MSFT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=5,
            price=350.0,
        )
        assert req.price == 350.0

    def test_trade_request_valid(self):
        from backend.api.routers.trading import TradeRequest, OrderSide, OrderType

        req = TradeRequest(
            symbol="GOOG",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=2,
            price=180.0,
        )
        assert req.price == 180.0

    def test_impact_request_valid(self):
        from backend.api.routers.trading import ImpactRequest, OrderSide

        req = ImpactRequest(
            symbol="TSLA",
            side=OrderSide.BUY,
            quantity=3,
            price=250.0,
        )
        assert req.quantity == 3


# ============================================================================
# Validate Order Endpoint
# ============================================================================

class TestValidateOrder:
    """Tests for POST /orders/validate."""

    @pytest.mark.asyncio
    async def test_validate_order_success(self):
        from backend.api.routers.trading import validate_order, OrderRequest, OrderSide, OrderType

        mock_svc = AsyncMock()
        mock_svc.validate_order.return_value = {"valid": True, "message": "OK"}

        order = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10,
        )
        mock_user = MagicMock()

        result = await validate_order(
            portfolio_id=1,
            order=order,
            current_user=mock_user,
            svc=mock_svc,
        )
        assert result.success is True
        assert result.data["valid"] is True
        mock_svc.validate_order.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_validate_order_failure_raises_422(self):
        from fastapi import HTTPException
        from backend.api.routers.trading import validate_order, OrderRequest, OrderSide, OrderType

        mock_svc = AsyncMock()
        mock_svc.validate_order.return_value = {
            "valid": False,
            "errors": ["Insufficient cash balance for order"],
        }

        order = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10,
        )

        with pytest.raises(HTTPException) as exc_info:
            await validate_order(
                portfolio_id=1,
                order=order,
                current_user=MagicMock(),
                svc=mock_svc,
            )
        assert exc_info.value.status_code == 422

    @pytest.mark.asyncio
    async def test_validate_order_passes_price_fields(self):
        from backend.api.routers.trading import validate_order, OrderRequest, OrderSide, OrderType

        mock_svc = AsyncMock()
        mock_svc.validate_order.return_value = {"valid": True, "message": "OK"}

        order = OrderRequest(
            symbol="MSFT",
            side=OrderSide.BUY,
            order_type=OrderType.STOP_LIMIT,
            quantity=5,
            price=400.0,
            stop_price=390.0,
        )

        await validate_order(
            portfolio_id=2,
            order=order,
            current_user=MagicMock(),
            svc=mock_svc,
        )

        call_args = mock_svc.validate_order.call_args[0][0]
        assert call_args["price"] == 400.0
        assert call_args["stop_price"] == 390.0
        assert call_args["portfolio_id"] == 2


# ============================================================================
# Execute Trade Endpoint
# ============================================================================

class TestExecuteTrade:
    """Tests for POST /orders/{portfolio_id}."""

    @pytest.mark.asyncio
    async def test_execute_trade_success(self):
        from backend.api.routers.trading import execute_trade, TradeRequest, OrderSide, OrderType

        mock_svc = AsyncMock()
        mock_svc.execute_trade.return_value = {
            "success": True,
            "trade_id": 42,
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 10,
            "price": 175.0,
            "total_cost": 1750.0,
        }

        trade = TradeRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10,
            price=175.0,
        )

        result = await execute_trade(
            portfolio_id=1,
            trade=trade,
            current_user=MagicMock(),
            svc=mock_svc,
        )
        assert result.success is True
        assert result.data["trade_id"] == 42

    @pytest.mark.asyncio
    async def test_execute_trade_failure_raises_400(self):
        from fastapi import HTTPException
        from backend.api.routers.trading import execute_trade, TradeRequest, OrderSide, OrderType

        mock_svc = AsyncMock()
        mock_svc.execute_trade.return_value = {
            "success": False,
            "error": "Order validation failed",
        }

        trade = TradeRequest(
            symbol="AAPL",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=100,
            price=175.0,
        )

        with pytest.raises(HTTPException) as exc_info:
            await execute_trade(
                portfolio_id=1,
                trade=trade,
                current_user=MagicMock(),
                svc=mock_svc,
            )
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_execute_trade_passes_order_dict(self):
        from backend.api.routers.trading import execute_trade, TradeRequest, OrderSide, OrderType

        mock_svc = AsyncMock()
        mock_svc.execute_trade.return_value = {"success": True, "trade_id": 1}

        trade = TradeRequest(
            symbol="GOOG",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=3,
            price=180.0,
        )

        await execute_trade(
            portfolio_id=5,
            trade=trade,
            current_user=MagicMock(),
            svc=mock_svc,
        )

        mock_svc.execute_trade.assert_awaited_once_with(
            5,
            {
                "symbol": "GOOG",
                "side": OrderSide.BUY,
                "order_type": OrderType.LIMIT,
                "quantity": 3,
                "price": 180.0,
            },
        )


# ============================================================================
# Calculate Impact Endpoint
# ============================================================================

class TestCalculateImpact:
    """Tests for POST /orders/{portfolio_id}/impact."""

    @pytest.mark.asyncio
    async def test_calculate_impact_success(self):
        from backend.api.routers.trading import calculate_impact, ImpactRequest, OrderSide

        mock_svc = AsyncMock()
        mock_svc.calculate_portfolio_impact.return_value = {
            "success": True,
            "portfolio_id": 1,
            "trade_impact": {"symbol": "AAPL", "side": "buy", "trade_value": 1750.0},
            "before": {"total_value": 50000.0, "cash_balance": 20000.0},
            "after": {"total_value": 51750.0, "cash_balance": 18250.0},
        }

        impact = ImpactRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            price=175.0,
        )

        result = await calculate_impact(
            portfolio_id=1,
            impact=impact,
            current_user=MagicMock(),
            svc=mock_svc,
        )
        assert result.success is True
        assert result.data["portfolio_id"] == 1

    @pytest.mark.asyncio
    async def test_calculate_impact_portfolio_not_found(self):
        from fastapi import HTTPException
        from backend.api.routers.trading import calculate_impact, ImpactRequest, OrderSide

        mock_svc = AsyncMock()
        mock_svc.calculate_portfolio_impact.return_value = {
            "success": False,
            "error": "Portfolio 999 not found",
        }

        impact = ImpactRequest(
            symbol="TSLA",
            side=OrderSide.SELL,
            quantity=5,
            price=250.0,
        )

        with pytest.raises(HTTPException) as exc_info:
            await calculate_impact(
                portfolio_id=999,
                impact=impact,
                current_user=MagicMock(),
                svc=mock_svc,
            )
        assert exc_info.value.status_code == 404


# ============================================================================
# Service Dependency
# ============================================================================

class TestTradingServiceDependency:
    """Tests for get_trading_service dependency."""

    @pytest.mark.asyncio
    async def test_get_trading_service_returns_singleton(self):
        from backend.api.routers.trading import get_trading_service
        from backend.services.trading_service import trading_service

        result = await get_trading_service()
        assert result is trading_service
