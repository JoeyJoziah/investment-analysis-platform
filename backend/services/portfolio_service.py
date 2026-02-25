"""
Portfolio Service
Business logic for portfolio management operations.
"""

import logging
import random
import uuid
import statistics
from typing import Dict, List, Optional, Any
from decimal import Decimal
from datetime import datetime, date, timedelta, timezone

from backend.repositories.portfolio_repository import portfolio_repository

logger = logging.getLogger(__name__)


class PortfolioService:
    """
    Service for portfolio management operations.
    Provides business logic layer between API and repository.
    """

    def __init__(self):
        self.repository = portfolio_repository

    async def get_portfolio_summary(
        self,
        user_id: int,
        portfolio_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive portfolio summary with positions and current values.

        Args:
            user_id: User ID (for authorization)
            portfolio_id: Portfolio ID

        Returns:
            Dictionary containing portfolio summary or None if not found
        """
        try:
            # Get portfolio with positions
            portfolio = await self.repository.get_portfolio_with_positions(portfolio_id)

            if not portfolio:
                logger.warning(f"Portfolio {portfolio_id} not found")
                return None

            # Verify ownership
            if portfolio.user_id != user_id:
                logger.warning(f"User {user_id} not authorized for portfolio {portfolio_id}")
                return None

            # Calculate current portfolio value
            portfolio_value = await self.repository.calculate_portfolio_value(portfolio_id)

            if not portfolio_value:
                logger.error(f"Failed to calculate value for portfolio {portfolio_id}")
                return None

            # Get allocation breakdown
            allocation = await self.repository.get_portfolio_allocation(portfolio_id)

            return {
                'portfolio_id': portfolio.id,
                'name': portfolio.name,
                'description': portfolio.description,
                'cash_balance': float(portfolio.cash_balance),
                'is_default': portfolio.is_default,
                'created_at': portfolio.created_at.isoformat(),
                'updated_at': portfolio.updated_at.isoformat(),
                'value': portfolio_value,
                'allocation': allocation,
                'position_count': len(portfolio.positions)
            }

        except Exception as e:
            logger.error(f"Error getting portfolio summary for portfolio {portfolio_id}: {e}")
            return None

    async def add_position(
        self,
        portfolio_id: int,
        stock_symbol: str,
        quantity: float,
        cost: float,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Add a stock position to portfolio.

        Args:
            portfolio_id: Portfolio ID
            stock_symbol: Stock ticker symbol
            quantity: Number of shares to buy
            cost: Cost per share
            user_id: Optional user ID for authorization

        Returns:
            Dictionary with operation result
        """
        try:
            # Verify portfolio ownership if user_id provided
            if user_id:
                portfolio = await self.repository.get_by_id(portfolio_id)
                if not portfolio or portfolio.user_id != user_id:
                    return {
                        'success': False,
                        'error': 'Portfolio not found or access denied'
                    }

            # Convert to Decimal for precision
            quantity_decimal = Decimal(str(quantity))
            cost_decimal = Decimal(str(cost))

            # Note: Would need to lookup stock_id from stock_symbol
            # For now, this is a placeholder
            logger.warning(f"Stock symbol lookup not implemented: {stock_symbol}")
            stock_id = 1  # Placeholder

            # Add position using repository
            position = await self.repository.add_position(
                portfolio_id=portfolio_id,
                stock_id=stock_id,
                quantity=quantity_decimal,
                price=cost_decimal,
                transaction_type='buy'
            )

            if not position:
                return {
                    'success': False,
                    'error': 'Failed to add position'
                }

            return {
                'success': True,
                'position_id': position.id,
                'portfolio_id': portfolio_id,
                'stock_symbol': stock_symbol,
                'quantity': float(quantity_decimal),
                'average_cost': float(position.avg_cost_basis),
                'total_value': float(quantity_decimal * position.avg_cost_basis)
            }

        except Exception as e:
            logger.error(f"Error adding position to portfolio {portfolio_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def get_allocation(
        self,
        portfolio_id: int
    ) -> Dict[str, Any]:
        """
        Get current portfolio allocation percentages.

        Args:
            portfolio_id: Portfolio ID

        Returns:
            Dictionary containing allocation breakdown by sector and stock
        """
        try:
            allocation = await self.repository.get_portfolio_allocation(portfolio_id)

            if not allocation:
                return {
                    'portfolio_id': portfolio_id,
                    'total_value': 0,
                    'cash_allocation_pct': 100,
                    'sector_allocation': {},
                    'stock_allocation': []
                }

            return allocation

        except Exception as e:
            logger.error(f"Error getting allocation for portfolio {portfolio_id}: {e}")
            return {
                'error': str(e)
            }

    async def get_performance(
        self,
        portfolio_id: int,
        timeframe: str = '1y'
    ) -> Optional[Dict[str, Any]]:
        """
        Get portfolio performance metrics.

        Args:
            portfolio_id: Portfolio ID
            timeframe: Time period (1w, 1m, 3m, 6m, 1y, ytd, all)

        Returns:
            Dictionary containing performance metrics
        """
        try:
            # Map timeframe to date range
            end_date = date.today()

            if timeframe == '1w':
                start_date = end_date - timedelta(days=7)
            elif timeframe == '1m':
                start_date = end_date - timedelta(days=30)
            elif timeframe == '3m':
                start_date = end_date - timedelta(days=90)
            elif timeframe == '6m':
                start_date = end_date - timedelta(days=180)
            elif timeframe == 'ytd':
                start_date = date(end_date.year, 1, 1)
            elif timeframe == '1y':
                start_date = end_date - timedelta(days=365)
            else:  # 'all'
                start_date = None

            # Get performance from repository
            performance = await self.repository.calculate_portfolio_performance(
                portfolio_id=portfolio_id,
                start_date=start_date,
                end_date=end_date
            )

            return performance

        except Exception as e:
            logger.error(f"Error getting performance for portfolio {portfolio_id}: {e}")
            return None

    async def get_transactions(
        self,
        portfolio_id: int,
        limit: Optional[int] = 50
    ) -> List[Dict[str, Any]]:
        """
        Get transaction history for portfolio.

        Args:
            portfolio_id: Portfolio ID
            limit: Maximum number of transactions to return

        Returns:
            List of transaction records
        """
        try:
            transactions = await self.repository.get_portfolio_transactions(
                portfolio_id=portfolio_id,
                limit=limit
            )

            return [
                {
                    'id': txn.id,
                    'portfolio_id': txn.portfolio_id,
                    'stock_id': txn.stock_id,
                    'transaction_type': txn.transaction_type.value,
                    'quantity': float(txn.quantity),
                    'price': float(txn.price),
                    'total_amount': float(txn.total_amount),
                    'trade_date': txn.trade_date.isoformat(),
                    'executed_at': txn.executed_at.isoformat()
                }
                for txn in transactions
            ]

        except Exception as e:
            logger.error(f"Error getting transactions for portfolio {portfolio_id}: {e}")
            return []

    async def create_portfolio(
        self,
        user_id: int,
        name: str,
        description: Optional[str] = None,
        initial_cash: float = 10000.0
    ) -> Dict[str, Any]:
        """
        Create a new portfolio for a user.

        Args:
            user_id: User ID
            name: Portfolio name
            description: Optional portfolio description
            initial_cash: Initial cash balance

        Returns:
            Dictionary with created portfolio details
        """
        try:
            portfolio_data = {
                'user_id': user_id,
                'name': name,
                'description': description or f'{name} portfolio',
                'cash_balance': Decimal(str(initial_cash)),
                'is_default': False
            }

            portfolio = await self.repository.create(portfolio_data)

            return {
                'success': True,
                'portfolio_id': portfolio.id,
                'name': portfolio.name,
                'cash_balance': float(portfolio.cash_balance),
                'created_at': portfolio.created_at.isoformat()
            }

        except Exception as e:
            logger.error(f"Error creating portfolio for user {user_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    # =========================================================================
    # Business logic extracted from the portfolio router
    # =========================================================================

    async def get_current_stock_price(self, symbol: str, db) -> float:
        """
        Get current stock price from database or fall back to a mock price.

        Args:
            symbol: Stock ticker symbol
            db: Async database session

        Returns:
            Current price as a float
        """
        try:
            from backend.repositories import price_repository
            latest_price = await price_repository.get_latest_price(symbol, session=db)

            if latest_price:
                return float(latest_price.close)

            # Fallback to mock price
            return random.uniform(50, 500)

        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return random.uniform(50, 500)

    async def calculate_portfolio_risk_score(
        self,
        portfolio_id: int,
        positions: List,
        db
    ) -> float:
        """
        Calculate portfolio risk score based on positions and concentration.

        Args:
            portfolio_id: Portfolio ID (used for logging)
            positions: List of position ORM objects
            db: Async database session (unused but kept for API consistency)

        Returns:
            Risk score in the range [10, 100]
        """
        try:
            if not positions:
                return 30.0  # Low risk for cash-only portfolio

            # Simple risk calculation based on position concentration
            total_value = sum(pos.quantity * pos.average_cost for pos in positions)

            if total_value == 0:
                return 30.0

            # Calculate concentration risk
            max_position_value = max(pos.quantity * pos.average_cost for pos in positions)
            concentration_risk = (max_position_value / total_value) * 100

            # Calculate sector diversification (simplified)
            unique_sectors = len(set(pos.symbol[:2] for pos in positions))  # Approximation
            diversification_bonus = min(unique_sectors * 5, 20)

            # Base risk score
            risk_score = 50 + concentration_risk - diversification_bonus

            return min(100, max(10, risk_score))

        except Exception as e:
            logger.error(f"Error calculating risk score for portfolio {portfolio_id}: {e}")
            return 50.0

    async def calculate_real_performance_metrics(
        self,
        portfolio_id: int,
        positions: List,
        db
    ) -> Dict[str, Any]:
        """
        Calculate real performance metrics from portfolio position data.

        Falls back to mock metrics when there are no positions or on errors.

        Args:
            portfolio_id: Portfolio ID (used for logging)
            positions: List of Position Pydantic model instances
            db: Async database session (reserved for future use)

        Returns:
            Dictionary of performance metric values suitable for PerformanceMetrics
        """
        try:
            if not positions:
                return self._mock_performance_metrics()

            # Calculate portfolio returns
            total_return = sum(p.unrealized_gain_percent for p in positions) / len(positions) / 100

            # Calculate volatility (simplified)
            returns = [p.unrealized_gain_percent / 100 for p in positions]

            volatility = statistics.stdev(returns) if len(returns) > 1 else 0.1
            annualized_volatility = volatility * (252 ** 0.5)

            # Risk-free rate approximation
            risk_free_rate = 0.02

            # Calculate Sharpe ratio
            sharpe_ratio = (
                (total_return - risk_free_rate) / annualized_volatility
                if annualized_volatility > 0
                else 0
            )

            # Calculate other metrics
            positive_returns = [r for r in returns if r >= 0]
            win_rate = len(positive_returns) / len(returns) if returns else 0.5

            return {
                "total_return": total_return,
                "annualized_return": total_return,  # Simplified
                "volatility": annualized_volatility,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sharpe_ratio * 1.2,  # Approximation
                "max_drawdown": min(returns) if returns else 0,
                "beta": random.uniform(0.8, 1.2),  # Would calculate vs market
                "alpha": total_return - 0.08,  # vs benchmark approximation
                "treynor_ratio": total_return / 1.0,  # Simplified
                "calmar_ratio": total_return / abs(min(returns, default=0.1)),
                "win_rate": win_rate,
                "profit_factor": 2.0 if total_return > 0 else 0.8,
                "risk_adjusted_return": total_return / max(volatility, 0.01),
            }

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return self._mock_performance_metrics()

    def _mock_performance_metrics(self) -> Dict[str, Any]:
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

    def generate_performance_data_points(
        self,
        portfolio_id: str,
        period: str,
        benchmark: str = "SPY"
    ) -> Dict[str, Any]:
        """
        Generate portfolio performance data points over the requested period.

        Args:
            portfolio_id: Portfolio ID
            period: One of 1D, 1W, 1M, 3M, 6M, 1Y, 3Y, 5Y, ALL
            benchmark: Benchmark ticker symbol

        Returns:
            Dictionary with data_points list and aggregated metrics
        """
        # Determine number of points based on period
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

        # Calculate metrics
        start_value = data_points[0]["value"]
        end_value = data_points[-1]["value"]
        total_return = (end_value - start_value) / start_value

        return {
            "portfolio_id": portfolio_id,
            "period": period,
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

    def build_portfolio_analysis(self, portfolio_id: str) -> Dict[str, Any]:
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
        self,
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

            if abs(difference) > 1:  # Only rebalance if difference > 1%
                action = "buy" if difference > 0 else "sell"
                trades.append({
                    "asset_class": asset_class,
                    "action": action,
                    "amount": abs(difference) * 1000,  # Convert to dollar amount
                    "current_allocation": round(current_percent, 2),
                    "target_allocation": target_percent,
                    "impact": round(difference, 2),
                })

        # Limit number of trades
        trades = trades[:max_trades]

        return {
            "portfolio_id": portfolio_id,
            "rebalancing_plan": trades,
            "estimated_cost": sum(t["amount"] * 0.001 for t in trades),  # 0.1% transaction cost
            "tax_impact": random.uniform(-1000, -100) if tax_efficient else 0,
            "execution_status": "pending",
        }

    def generate_transaction_list(
        self,
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

        # Sort by timestamp descending
        transactions.sort(key=lambda x: x["timestamp"], reverse=True)
        return transactions[offset: offset + limit]

    async def update_portfolio_metrics(self, portfolio_id: str) -> None:
        """
        Background task: update portfolio metrics after a transaction.

        Args:
            portfolio_id: Portfolio ID to update
        """
        # In production, this would recalculate portfolio metrics
        print(f"Updating metrics for portfolio {portfolio_id}")

    async def execute_rebalancing(self, portfolio_id: str, trades: List[Dict]) -> None:
        """
        Background task: execute rebalancing trades.

        Args:
            portfolio_id: Portfolio ID
            trades: List of trade dictionaries to execute
        """
        # In production, this would execute the trades
        print(f"Executing {len(trades)} trades for portfolio {portfolio_id}")


def _all_transaction_types():
    """Return all transaction type string values (mirrors TransactionType enum)."""
    return ["buy", "sell", "dividend", "transfer_in", "transfer_out"]


# Create singleton instance
portfolio_service = PortfolioService()
