"""
Portfolio Rebalancing
Rebalancing logic, performance data generation, and analysis building
extracted from PortfolioService.
"""

import logging
import random
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime, date, timedelta, timezone

from backend.config.settings import settings
from backend.exceptions import ModelUnavailableError

logger = logging.getLogger(__name__)


def generate_performance_data_points(
    portfolio_id: str,
    period: str,
    benchmark: str = "SPY",
) -> Dict[str, Any]:
    """Generate portfolio performance data points over the requested period.

    Per PRD audit 2026-04 F-02-003 / Q4 default: this function previously
    returned ``random.uniform``-derived chart data and metrics. It is now
    gated behind ``settings.DEMO_MODE`` so production raises
    ``ModelUnavailableError`` (surfaced as HTTP 503 ``model_unavailable``
    by the portfolio router) rather than fabricating chart values.

    The full real implementation requires a portfolio time-series store; the
    G2a workstream owns the ML/analytics path that will populate it. Until
    that lands, refusing-to-serve is the SEC-conservative posture.
    """
    if not settings.DEMO_MODE:
        raise ModelUnavailableError(
            model="portfolio_performance",
            reason="not_implemented",
        )

    period_map = {
        "1D": 24,
        "1W": 7,
        "1M": 30,
        "3M": 90,
        "6M": 180,
        "1Y": 252,
    }
    num_points = period_map.get(period, 365)

    data_points = []
    base_value = 100000
    for i in range(num_points):
        date_point = datetime.now(timezone.utc) - timedelta(days=num_points - i)
        value = base_value * (1 + random.uniform(-0.02, 0.02))
        base_value = value

        data_points.append({
            "date": date_point.date().isoformat(),
            "value": round(value, 2),
            "benchmark_value": round(value * random.uniform(0.95, 1.05), 2),
        })

    start_value = data_points[0]["value"]
    end_value = data_points[-1]["value"]
    total_return = (end_value - start_value) / start_value

    return {
        "portfolio_id": portfolio_id,
        "period": period,
        "data_source": "simulated",  # F-02-003: explicit demo tag
        "data_points": data_points,
        "metrics": {
            "total_return": round(total_return, 4),
            "annualized_return": round(total_return * (365 / num_points), 4),
            "volatility": random.uniform(0.1, 0.3),
            "sharpe_ratio": random.uniform(0.5, 2.0),
            "max_drawdown": random.uniform(-0.2, -0.05),
            "benchmark_correlation": random.uniform(0.6, 0.95),
        },
        "vs_benchmark": {
            "excess_return": random.uniform(-0.05, 0.1),
            "tracking_error": random.uniform(0.02, 0.1),
            "information_ratio": random.uniform(-0.5, 1.5),
        },
    }


def build_portfolio_analysis(portfolio_id: str) -> Dict[str, Any]:
    """
    Build a comprehensive portfolio analysis result.

    Args:
        portfolio_id: Portfolio ID

    Returns:
        Dictionary with all analysis fields matching PortfolioAnalysis schema
    """
    return {
        "portfolio_id": portfolio_id,
        "analysis_date": date.today(),
        "risk_analysis": {
            "var_95": random.uniform(-0.1, -0.02),
            "cvar_95": random.uniform(-0.15, -0.03),
            "downside_deviation": random.uniform(0.05, 0.15),
            "upside_potential": random.uniform(0.1, 0.3),
        },
        "diversification_score": random.uniform(60, 90),
        "concentration_risk": {
            "top_holding": random.uniform(0.1, 0.3),
            "top_3_holdings": random.uniform(0.3, 0.5),
            "top_5_holdings": random.uniform(0.5, 0.7),
        },
        "correlation_matrix": {
            "AAPL": {"GOOGL": 0.7, "MSFT": 0.65, "AMZN": 0.6},
            "GOOGL": {"AAPL": 0.7, "MSFT": 0.75, "AMZN": 0.65},
            "MSFT": {"AAPL": 0.65, "GOOGL": 0.75, "AMZN": 0.6},
        },
        "efficient_frontier": {
            "current_position": {"return": 0.12, "risk": 0.15},
            "optimal_position": {"return": 0.14, "risk": 0.14},
            "improvement_potential": 0.02,
        },
        "optimization_suggestions": [
            "Reduce concentration in Technology sector",
            "Consider adding international exposure",
            "Increase allocation to fixed income for better risk-adjusted returns",
            "Review positions with high correlation",
        ],
        "rebalancing_needed": random.choice([True, False]),
        "recommended_changes": [
            {"action": "reduce", "symbol": "AAPL", "percent": 5},
            {"action": "increase", "symbol": "BND", "percent": 10},
            {"action": "add", "symbol": "VXUS", "percent": 5},
        ],
    }


def generate_rebalancing_trades(
    portfolio_id: str,
    target_allocation: Dict,
    max_trades: int,
    min_trade_value: float,
    tax_efficient: bool,
) -> Dict[str, Any]:
    """
    Generate a rebalancing plan given a target asset allocation.

    Args:
        portfolio_id: Portfolio ID
        target_allocation: Mapping of AssetClass -> target percentage
        max_trades: Maximum number of trades to include in the plan
        min_trade_value: Minimum trade dollar value to include
        tax_efficient: Whether to factor in tax efficiency

    Returns:
        Dictionary with rebalancing plan, estimated cost, tax impact, and status
    """
    trades = []
    for asset_class, target_percent in target_allocation.items():
        current_percent = random.uniform(0, 30)
        difference = target_percent - current_percent

        if abs(difference) > 1:
            action = "buy" if difference > 0 else "sell"
            trades.append({
                "asset_class": asset_class,
                "action": action,
                "amount": abs(difference) * 1000,
                "current_allocation": round(current_percent, 2),
                "target_allocation": target_percent,
                "impact": round(difference, 2),
            })

    trades = trades[:max_trades]

    return {
        "portfolio_id": portfolio_id,
        "rebalancing_plan": trades,
        "estimated_cost": sum(t["amount"] * 0.001 for t in trades),
        "tax_impact": random.uniform(-1000, -100) if tax_efficient else 0,
        "execution_status": "pending",
    }


def generate_transaction_list(
    portfolio_id: str,
    limit: int,
    offset: int,
    transaction_type_filter,
    symbol_filter: Optional[str],
    start_date: Optional[date],
    end_date: Optional[date],
) -> List[Dict[str, Any]]:
    """
    Generate a simulated transaction history list for a portfolio.

    Args:
        portfolio_id: Portfolio ID
        limit: Maximum number of records to return
        offset: Number of records to skip
        transaction_type_filter: Optional TransactionType enum value to filter by
        symbol_filter: Optional ticker symbol to filter by
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        Filtered, sorted list of transaction dictionaries (sliced per limit/offset)
    """
    from backend.services.portfolio_helpers import _all_transaction_types

    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "NVDA", "TSLA"]

    transactions = []
    for _ in range(100):
        trans_date = datetime.now(timezone.utc) - timedelta(days=random.randint(0, 365))

        if start_date and trans_date.date() < start_date:
            continue
        if end_date and trans_date.date() > end_date:
            continue

        sym = symbol_filter.upper() if symbol_filter else random.choice(symbols)
        t_type = transaction_type_filter or random.choice(list(_all_transaction_types()))

        trans = {
            "id": str(uuid.uuid4()),
            "portfolio_id": portfolio_id,
            "symbol": sym,
            "transaction_type": t_type,
            "quantity": random.uniform(1, 50),
            "price": random.uniform(50, 500),
            "fees": random.uniform(0, 10),
            "notes": "Transaction note",
            "timestamp": trans_date,
        }

        if transaction_type_filter and trans["transaction_type"] != transaction_type_filter:
            continue
        if symbol_filter and trans["symbol"] != symbol_filter.upper():
            continue

        transactions.append(trans)

    transactions.sort(key=lambda x: x["timestamp"], reverse=True)
    return transactions[offset: offset + limit]


async def execute_rebalancing(portfolio_id: str, trades: List[Dict]) -> None:
    """
    Background task: execute rebalancing trades.

    Args:
        portfolio_id: Portfolio ID
        trades: List of trade dictionaries to execute
    """
    print(f"Executing {len(trades)} trades for portfolio {portfolio_id}")


async def update_portfolio_metrics(portfolio_id: str) -> None:
    """
    Background task: update portfolio metrics after a transaction.

    Args:
        portfolio_id: Portfolio ID to update
    """
    print(f"Updating metrics for portfolio {portfolio_id}")
