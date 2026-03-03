"""
Portfolio Helpers
Utility functions and static helpers for portfolio service operations.
"""

import logging
import random
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def generate_position_data(symbol: str = None) -> Dict[str, Any]:
    """
    Generate a sample position as a dictionary.

    Args:
        symbol: Optional ticker symbol; picks randomly if omitted.

    Returns:
        Dictionary with all fields needed for the Position Pydantic model.
    """
    if not symbol:
        symbols = [
            "AAPL", "GOOGL", "MSFT", "AMZN", "META",
            "NVDA", "TSLA", "JPM", "V", "JNJ",
        ]
        symbol = random.choice(symbols)

    quantity = random.uniform(10, 100)
    average_cost = random.uniform(50, 300)
    current_price = average_cost * random.uniform(0.7, 1.5)

    return {
        "id": str(uuid.uuid4()),
        "symbol": symbol,
        "name": f"{symbol} Inc.",
        "quantity": round(quantity, 2),
        "average_cost": round(average_cost, 2),
        "current_price": round(current_price, 2),
        "market_value": round(quantity * current_price, 2),
        "cost_basis": round(quantity * average_cost, 2),
        "unrealized_gain": round((current_price - average_cost) * quantity, 2),
        "unrealized_gain_percent": round(
            (current_price - average_cost) / average_cost * 100, 2
        ),
        "realized_gain": random.uniform(-1000, 5000),
        "asset_class": "stocks",
        "sector": random.choice(
            ["Technology", "Healthcare", "Finance", "Consumer"]
        ),
        "allocation_percent": random.uniform(5, 25),
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
    """Return randomly-generated placeholder performance metrics."""
    return {
        "total_return": random.uniform(-0.1, 0.3),
        "annualized_return": random.uniform(0.05, 0.15),
        "volatility": random.uniform(0.1, 0.3),
        "sharpe_ratio": random.uniform(0.5, 2.0),
        "sortino_ratio": random.uniform(0.7, 2.5),
        "max_drawdown": random.uniform(-0.3, -0.05),
        "beta": random.uniform(0.8, 1.2),
        "alpha": random.uniform(-0.02, 0.05),
        "treynor_ratio": random.uniform(0.1, 0.3),
        "calmar_ratio": random.uniform(0.5, 2.0),
        "win_rate": random.uniform(0.4, 0.7),
        "profit_factor": random.uniform(1.2, 2.5),
        "risk_adjusted_return": random.uniform(0.08, 0.20),
    }
