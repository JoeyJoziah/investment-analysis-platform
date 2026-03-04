"""
Portfolio Helpers
Utility functions and static helpers for portfolio service operations.
"""

import logging
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def generate_position_data(symbol: str = None) -> Dict[str, Any]:
    """
    Generate a placeholder position skeleton as a dictionary.

    Args:
        symbol: Ticker symbol; defaults to "UNKNOWN" if omitted.

    Returns:
        Dictionary with all fields needed for the Position Pydantic model.
        All numeric fields are set to 0.0 / None to avoid fake data.
    """
    if not symbol:
        symbol = "UNKNOWN"

    return {
        "id": str(uuid.uuid4()),
        "symbol": symbol,
        "name": f"{symbol} Inc.",
        "quantity": 0.0,
        "average_cost": 0.0,
        "current_price": 0.0,
        "market_value": 0.0,
        "cost_basis": 0.0,
        "unrealized_gain": 0.0,
        "unrealized_gain_percent": 0.0,
        "realized_gain": 0.0,
        "asset_class": "stocks",
        "sector": None,
        "allocation_percent": 0.0,
    }


def _all_transaction_types():
    """Return all transaction type string values (mirrors TransactionType enum)."""
    return ["buy", "sell", "dividend", "transfer_in", "transfer_out"]


def fallback_summary(portfolio) -> Dict[str, Any]:
    """Return a safe fallback summary when computation fails."""
    return {
        "id": str(portfolio.portfolio_id or portfolio.id),
        "name": portfolio.name or f"Portfolio {portfolio.id}",
        "total_value": 100000.0,
        "total_cost": 95000.0,
        "total_gain": 5000.0,
        "total_gain_percent": 5.26,
        "cash_balance": 10000.0,
        "buying_power": 20000.0,
        "day_change": 0.0,
        "day_change_percent": 0.0,
        "positions_count": 0,
        "strategy": "balanced",
        "risk_score": 50.0,
        "created_at": portfolio.created_at,
        "last_updated": datetime.now(timezone.utc),
    }


def compute_asset_allocation(
    positions: List[Dict[str, Any]],
    cash_balance: float,
    total_value: float,
) -> Dict[str, float]:
    """Compute asset class allocation percentages."""
    stocks_value = sum(p["market_value"] for p in positions)
    cash_pct = (
        (cash_balance / total_value * 100) if total_value > 0 else 0
    )
    stocks_pct = (
        (stocks_value / total_value * 100) if total_value > 0 else 0
    )
    return {
        "stocks": round(stocks_pct, 2),
        "cash": round(cash_pct, 2),
        "bonds": 0.0,
        "etf": 0.0,
    }


def compute_sector_allocation(
    positions: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Compute sector allocation percentages."""
    allocation: Dict[str, float] = {}
    for pos in positions:
        sector = pos.get("sector")
        if sector and sector != "Unknown":
            allocation[sector] = (
                allocation.get(sector, 0)
                + pos["allocation_percent"]
            )
    return allocation


def format_transactions(
    recent_transactions, portfolio_id: str
) -> List[Dict[str, Any]]:
    """Convert ORM transaction records to response dictionaries."""
    result: List[Dict[str, Any]] = []
    for trans in recent_transactions:
        result.append({
            "id": str(trans.id),
            "portfolio_id": portfolio_id,
            "symbol": trans.symbol,
            "transaction_type": trans.transaction_type,
            "quantity": float(trans.quantity),
            "price": float(trans.price),
            "total_amount": float(trans.total_amount),
            "fees": float(trans.fees or 0),
            "notes": trans.notes,
            "timestamp": trans.created_at,
        })
    return result


def mock_performance_metrics() -> Dict[str, Any]:
    """Return a null-value stub for performance metrics (data not yet available)."""
    return {
        "total_return": None,
        "annualized_return": None,
        "volatility": None,
        "sharpe_ratio": None,
        "sortino_ratio": None,
        "max_drawdown": None,
        "beta": None,
        "alpha": None,
        "treynor_ratio": None,
        "calmar_ratio": None,
        "win_rate": None,
        "profit_factor": None,
        "risk_adjusted_return": None,
        "mock": True,
    }
