"""
Trading Router
Exposes order validation, trade execution, and portfolio impact endpoints.
"""

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import logging

from backend.auth.oauth2 import get_current_user
from backend.models.unified_models import User
from backend.models.api_response import ApiResponse, success_response
from backend.services.trading_service import (
    trading_service,
    TradingService,
    OrderType,
    OrderSide,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["trading"])


# ============================================================================
# Service Layer Dependencies
# ============================================================================

async def get_trading_service() -> TradingService:
    """Dependency to get trading service instance."""
    return trading_service


# ============================================================================
# Request / Response Models
# ============================================================================

class OrderRequest(BaseModel):
    """Request body for creating/validating an order."""
    symbol: str = Field(..., min_length=1, max_length=5, description="Stock ticker symbol")
    side: OrderSide = Field(..., description="buy or sell")
    order_type: OrderType = Field(..., description="market, limit, stop, or stop_limit")
    quantity: float = Field(..., gt=0, description="Number of shares")
    price: Optional[float] = Field(None, gt=0, description="Limit price (required for limit/stop_limit)")
    stop_price: Optional[float] = Field(None, gt=0, description="Stop price (required for stop/stop_limit)")


class TradeRequest(BaseModel):
    """Request body for executing a trade."""
    symbol: str = Field(..., min_length=1, max_length=5)
    side: OrderSide
    order_type: OrderType
    quantity: float = Field(..., gt=0)
    price: float = Field(..., gt=0, description="Execution price")


class ImpactRequest(BaseModel):
    """Request body for calculating portfolio impact."""
    symbol: str = Field(..., min_length=1, max_length=5)
    side: OrderSide
    quantity: float = Field(..., gt=0)
    price: float = Field(..., gt=0)


# ============================================================================
# Endpoints
# ============================================================================

@router.post(
    "/orders/validate",
    response_model=ApiResponse[Dict[str, Any]],
    summary="Validate a trading order",
)
async def validate_order(
    portfolio_id: int,
    order: OrderRequest,
    current_user: User = Depends(get_current_user),
    svc: TradingService = Depends(get_trading_service),
):
    """
    Validate a trading order without executing it.
    Checks required fields, order-type constraints, and portfolio balance.
    """
    order_data = {
        "portfolio_id": portfolio_id,
        "symbol": order.symbol,
        "side": order.side,
        "order_type": order.order_type,
        "quantity": order.quantity,
    }
    if order.price is not None:
        order_data["price"] = order.price
    if order.stop_price is not None:
        order_data["stop_price"] = order.stop_price

    result = await svc.validate_order(order_data)
    if not result.get("valid"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=result.get("errors", ["Order validation failed"]),
        )
    return success_response(result)


@router.post(
    "/orders/{portfolio_id}",
    response_model=ApiResponse[Dict[str, Any]],
    summary="Execute a trade",
)
async def execute_trade(
    portfolio_id: int,
    trade: TradeRequest,
    current_user: User = Depends(get_current_user),
    svc: TradingService = Depends(get_trading_service),
):
    """
    Execute a trade for a portfolio.
    Validates the order first, then executes it.
    """
    order_dict = {
        "symbol": trade.symbol,
        "side": trade.side,
        "order_type": trade.order_type,
        "quantity": trade.quantity,
        "price": trade.price,
    }
    result = await svc.execute_trade(portfolio_id, order_dict)
    if not result.get("success"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("error", "Trade execution failed"),
        )
    return success_response(result)


@router.post(
    "/orders/{portfolio_id}/impact",
    response_model=ApiResponse[Dict[str, Any]],
    summary="Calculate trade impact on portfolio",
)
async def calculate_impact(
    portfolio_id: int,
    impact: ImpactRequest,
    current_user: User = Depends(get_current_user),
    svc: TradingService = Depends(get_trading_service),
):
    """
    Calculate the impact of a proposed trade on portfolio metrics
    without actually executing it.
    """
    trade_data = {
        "symbol": impact.symbol,
        "side": impact.side,
        "quantity": impact.quantity,
        "price": impact.price,
    }
    result = await svc.calculate_portfolio_impact(portfolio_id, trade_data)
    if not result.get("success"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=result.get("error", "Portfolio not found"),
        )
    return success_response(result)
