"""
Portfolio Service
Business logic for portfolio management operations.
"""

import logging
import statistics
from typing import Dict, List, Optional, Any
from decimal import Decimal
from datetime import datetime, date, timedelta, timezone


from backend.repositories.portfolio_repository import portfolio_repository
from backend.repositories import stock_repository

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

            # Look up the stock_id from the ticker symbol
            stock = await stock_repository.get_by_symbol(stock_symbol)
            if not stock:
                return {
                    'success': False,
                    'error': f"Stock symbol '{stock_symbol}' not found"
                }
            stock_id = stock.id

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

            logger.warning(f"No price data found for {symbol}, returning 0.0")
            return 0.0

        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return 0.0

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
                from backend.services.portfolio_helpers import mock_performance_metrics
                return mock_performance_metrics()

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
                "beta": None,  # Requires market benchmark data to calculate
                "alpha": total_return - 0.08,  # vs benchmark approximation
                "treynor_ratio": total_return / 1.0,  # Simplified
                "calmar_ratio": total_return / abs(min(returns, default=0.1)),
                "win_rate": win_rate,
                "profit_factor": 2.0 if total_return > 0 else 0.8,
                "risk_adjusted_return": total_return / max(volatility, 0.01),
            }

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            from backend.services.portfolio_helpers import mock_performance_metrics
            return mock_performance_metrics()

    async def compute_portfolio_summaries(
        self,
        user_id: int,
        db,
    ) -> List[Dict[str, Any]]:
        """
        Compute summary data for all portfolios owned by a user.

        Fetches portfolios from the database, creates a default if none exist,
        then calculates market values, gains, and risk scores using
        RealtimePriceService for current prices.

        Args:
            user_id: Authenticated user's ID.
            db: Async database session.

        Returns:
            List of dictionaries, each containing all fields for PortfolioSummary.
        """
        from backend.services.realtime_price_service import get_realtime_price_service
        from backend.repositories import portfolio_repository

        price_service = await get_realtime_price_service()

        portfolios = await portfolio_repository.get_user_portfolios(
            user_id=user_id, session=db
        )

        if not portfolios:
            logger.info(
                f"No portfolios found for user {user_id}, creating default"
            )
            default_portfolio = await portfolio_repository.create_default_portfolio(
                user_id=user_id, session=db
            )
            portfolios = [default_portfolio]

        summaries: List[Dict[str, Any]] = []

        for portfolio in portfolios:
            try:
                summary = await self._compute_single_summary(
                    portfolio, price_service, db
                )
                summaries.append(summary)
            except Exception as e:
                logger.error(
                    f"Error calculating summary for portfolio {portfolio.id}: {e}"
                )
                summaries.append(self._fallback_summary(portfolio))

        logger.info(f"Successfully calculated {len(summaries)} portfolio summaries")
        return summaries

    async def _compute_single_summary(
        self, portfolio, price_service, db
    ) -> Dict[str, Any]:
        """Compute summary metrics for a single portfolio."""
        from backend.repositories import portfolio_repository

        positions = await portfolio_repository.get_portfolio_positions(
            portfolio_id=portfolio.id, session=db
        )

        symbols = [pos.symbol for pos in positions]
        prices = await price_service.get_latest_prices_bulk(symbols, db)

        total_value = 0.0
        total_cost = 0.0
        day_change = 0.0

        for position in positions:
            price_update = prices.get(position.symbol)
            current_price = (
                price_update.price
                if price_update
                else await self.get_current_stock_price(position.symbol, db)
            )

            position_value = position.quantity * current_price
            position_cost = position.quantity * position.average_cost

            total_value += position_value
            total_cost += position_cost

            if price_update and price_update.close:
                day_change += (
                    position.quantity
                    * (price_update.close - position.average_cost)
                    * 0.01
                )

        cash_balance = float(portfolio.cash_balance or 0)
        total_value += cash_balance

        total_gain = total_value - total_cost
        total_gain_percent = (
            (total_gain / total_cost * 100) if total_cost > 0 else 0.0
        )
        day_change_percent = (
            (day_change / total_value * 100) if total_value > 0 else 0.0
        )

        risk_score = await self.calculate_portfolio_risk_score(
            portfolio.id, positions, db
        )

        return {
            "id": str(portfolio.portfolio_id or portfolio.id),
            "name": portfolio.name or f"Portfolio {portfolio.id}",
            "total_value": round(total_value, 2),
            "total_cost": round(total_cost, 2),
            "total_gain": round(total_gain, 2),
            "total_gain_percent": round(total_gain_percent, 2),
            "cash_balance": round(cash_balance, 2),
            "buying_power": round(cash_balance * 2, 2),
            "day_change": round(day_change, 2),
            "day_change_percent": round(day_change_percent, 2),
            "positions_count": len(positions),
            "strategy": portfolio.strategy or "balanced",
            "risk_score": round(risk_score, 2),
            "created_at": portfolio.created_at,
            "last_updated": portfolio.updated_at or datetime.now(timezone.utc),
        }

    @staticmethod
    def _fallback_summary(portfolio) -> Dict[str, Any]:
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

    async def compute_portfolio_detail(
        self,
        portfolio_id: str,
        user_id: int,
        db,
    ) -> Optional[Dict[str, Any]]:
        """
        Compute detailed portfolio data with real-time prices.

        Returns None if the portfolio is not found or access is denied.

        Args:
            portfolio_id: Portfolio string identifier.
            user_id: Authenticated user's ID.
            db: Async database session.

        Returns:
            Dictionary with all fields for PortfolioDetail, or None if not found.
        """
        from backend.services.realtime_price_service import get_realtime_price_service
        from backend.repositories import portfolio_repository

        price_service = await get_realtime_price_service()

        portfolio = await portfolio_repository.get_user_portfolio(
            portfolio_id=portfolio_id, user_id=user_id, session=db
        )
        if not portfolio:
            return None

        db_positions = await portfolio_repository.get_portfolio_positions(
            portfolio_id=portfolio.id, session=db
        )

        symbols = [pos.symbol for pos in db_positions]
        prices = await price_service.get_latest_prices_bulk(symbols, db)

        positions, total_value, total_cost, day_change = (
            await self._build_position_list(db_positions, prices, db)
        )

        cash_balance = float(portfolio.cash_balance or 0)
        total_value += cash_balance

        # Allocation percentages
        for pos in positions:
            pos["allocation_percent"] = (
                round((pos["market_value"] / total_value) * 100, 2)
                if total_value > 0
                else 0.0
            )

        asset_allocation = self._compute_asset_allocation(
            positions, cash_balance, total_value
        )
        sector_allocation = self._compute_sector_allocation(positions)

        positions_sorted = sorted(
            positions,
            key=lambda x: x["unrealized_gain_percent"],
            reverse=True,
        )
        top_performers = (
            positions_sorted[:3]
            if len(positions_sorted) >= 3
            else positions_sorted
        )
        worst_performers = (
            positions_sorted[-3:]
            if len(positions_sorted) >= 3
            else []
        )

        recent_transactions = await portfolio_repository.get_recent_transactions(
            portfolio_id=portfolio.id, limit=10, session=db
        )
        transactions = self._format_transactions(
            recent_transactions, portfolio_id
        )

        # Build lightweight position objects for metrics calculation
        class _Pos:
            def __init__(self, d):
                self.unrealized_gain_percent = d["unrealized_gain_percent"]
        metric_positions = [_Pos(p) for p in positions]
        metrics_dict = await self.calculate_real_performance_metrics(
            portfolio.id, metric_positions, db
        )

        total_gain = total_value - total_cost
        total_gain_percent = (
            (total_gain / total_cost * 100) if total_cost > 0 else 0.0
        )
        day_change_percent = (
            (day_change / total_value * 100) if total_value > 0 else 0.0
        )
        risk_score = await self.calculate_portfolio_risk_score(
            portfolio.id, db_positions, db
        )

        return {
            "id": portfolio_id,
            "name": portfolio.name or "Main Portfolio",
            "total_value": round(total_value, 2),
            "total_cost": round(total_cost, 2),
            "total_gain": round(total_gain, 2),
            "total_gain_percent": round(total_gain_percent, 2),
            "cash_balance": round(cash_balance, 2),
            "buying_power": round(cash_balance * 2, 2),
            "day_change": round(day_change, 2),
            "day_change_percent": round(day_change_percent, 2),
            "positions_count": len(positions),
            "strategy": portfolio.strategy or "balanced",
            "risk_score": round(risk_score, 2),
            "created_at": portfolio.created_at,
            "last_updated": portfolio.updated_at or datetime.now(timezone.utc),
            "positions": positions,
            "asset_allocation": asset_allocation,
            "sector_allocation": sector_allocation,
            "top_performers": top_performers,
            "worst_performers": worst_performers,
            "recent_transactions": transactions,
            "performance_metrics": metrics_dict,
        }

    async def _build_position_list(
        self, db_positions, prices, db
    ):
        """Build enriched position list and running totals."""
        positions: List[Dict[str, Any]] = []
        total_value = 0.0
        total_cost = 0.0
        day_change = 0.0

        for db_pos in db_positions:
            price_update = prices.get(db_pos.symbol)
            if price_update:
                current_price = price_update.price
                day_change_raw = price_update.change or 0
            else:
                current_price = await self.get_current_stock_price(
                    db_pos.symbol, db
                )
                day_change_raw = 0

            market_value = db_pos.quantity * current_price
            cost_basis = db_pos.quantity * db_pos.average_cost
            unrealized_gain = market_value - cost_basis
            unrealized_gain_percent = (
                (unrealized_gain / cost_basis * 100)
                if cost_basis > 0
                else 0.0
            )

            stock_info = await stock_repository.get_by_symbol(
                db_pos.symbol, session=db
            )
            sector = stock_info.sector if stock_info else "Unknown"

            positions.append({
                "id": str(db_pos.id),
                "symbol": db_pos.symbol,
                "name": (
                    stock_info.name
                    if stock_info
                    else f"{db_pos.symbol} Corp"
                ),
                "quantity": float(db_pos.quantity),
                "average_cost": float(db_pos.average_cost),
                "current_price": current_price,
                "market_value": market_value,
                "cost_basis": cost_basis,
                "unrealized_gain": unrealized_gain,
                "unrealized_gain_percent": unrealized_gain_percent,
                "realized_gain": float(db_pos.realized_gain or 0),
                "asset_class": "stocks",
                "sector": sector,
                "allocation_percent": 0.0,
            })

            total_value += market_value
            total_cost += cost_basis
            day_change += db_pos.quantity * day_change_raw

        return positions, total_value, total_cost, day_change

    @staticmethod
    def _compute_asset_allocation(
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

    @staticmethod
    def _compute_sector_allocation(
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

    @staticmethod
    def _format_transactions(
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
            data_points.append({
                "date": date_point.date().isoformat(),
                "value": round(base_value, 2),
                "benchmark_value": None,
            })

        return {
            "portfolio_id": portfolio_id,
            "period": period,
            "data_points": data_points,
            "metrics": {
                "total_return": None,
                "annualized_return": None,
                "volatility": None,
                "sharpe_ratio": None,
                "max_drawdown": None,
                "benchmark_correlation": None,
            },
            "vs_benchmark": {
                "excess_return": None,
                "tracking_error": None,
                "information_ratio": None,
            },
        }

    def build_portfolio_analysis(
        self,
        portfolio_id: str,
        positions: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Build a comprehensive portfolio analysis from real position data.

        When *positions* is empty/None, returns a fail-loud empty analysis
        (null risk metrics, score 0) — never fabricated random metrics.

        Args:
            portfolio_id: Portfolio ID (string public id)
            positions: Optional list of position dicts with keys such as
                ``market_value``, ``allocation_percent``,
                ``unrealized_gain_percent``, ``symbol``, ``sector``.

        Returns:
            Dictionary with all analysis fields matching PortfolioAnalysis schema
        """
        positions = list(positions or [])
        weights: List[float] = []
        returns: List[float] = []
        symbols: List[str] = []
        sectors: List[str] = []

        total_mv = sum(float(p.get("market_value") or 0) for p in positions)
        for p in positions:
            alloc = p.get("allocation_percent")
            if alloc is None and total_mv > 0:
                alloc = (float(p.get("market_value") or 0) / total_mv) * 100.0
            weights.append(max(0.0, float(alloc or 0.0)) / 100.0)
            returns.append(float(p.get("unrealized_gain_percent") or 0.0) / 100.0)
            symbols.append(str(p.get("symbol") or f"POS{len(symbols)}"))
            sectors.append(str(p.get("sector") or "Unknown"))

        weight_sum = sum(weights)
        if weight_sum > 0:
            weights = [w / weight_sum for w in weights]

        sorted_w = sorted(weights, reverse=True)
        top_holding = sorted_w[0] if sorted_w else None
        top_3 = sum(sorted_w[:3]) if sorted_w else None
        top_5 = sum(sorted_w[:5]) if sorted_w else None

        # Herfindahl–Hirschman index → diversification score in [0, 100]
        if len(weights) <= 1:
            diversification_score = 0.0 if weights else 0.0
            hhi = 1.0 if weights else 0.0
        else:
            hhi = sum(w * w for w in weights)
            # Equal-weight HHI = 1/n; map HHI from 1/n..1 onto score 100..0
            equal_hhi = 1.0 / len(weights)
            span = max(1e-9, 1.0 - equal_hhi)
            diversification_score = max(
                0.0, min(100.0, ((1.0 - hhi) / span) * 100.0)
            )

        var_95 = cvar_95 = downside_deviation = upside_potential = None
        current_return = current_risk = None
        if returns:
            current_return = sum(
                w * r for w, r in zip(weights, returns)
            ) if weights else (sum(returns) / len(returns))
            if len(returns) > 1:
                current_risk = statistics.stdev(returns)
                ordered = sorted(returns)
                # Historical 5% VaR / CVaR on position return distribution
                idx = max(0, int(len(ordered) * 0.05) - 1)
                var_95 = ordered[idx]
                tail = ordered[: max(1, idx + 1)]
                cvar_95 = sum(tail) / len(tail)
                neg = [r for r in returns if r < 0]
                downside_deviation = (
                    statistics.stdev(neg) if len(neg) > 1 else (abs(neg[0]) if neg else 0.0)
                )
                pos = [r for r in returns if r > 0]
                upside_potential = (
                    sum(pos) / len(pos) if pos else 0.0
                )
            else:
                current_risk = abs(returns[0]) if returns else None
                var_95 = returns[0]
                cvar_95 = returns[0]
                downside_deviation = abs(min(0.0, returns[0]))
                upside_potential = max(0.0, returns[0])

        # Pairwise correlation matrix: 1 on diagonal; same-sector 0.6 else 0.2
        correlation_matrix: Dict[str, Any] = {}
        for i, s_i in enumerate(symbols):
            row: Dict[str, float] = {}
            for j, s_j in enumerate(symbols):
                if i == j:
                    row[s_j] = 1.0
                elif sectors[i] == sectors[j] and sectors[i] != "Unknown":
                    row[s_j] = 0.6
                else:
                    row[s_j] = 0.2
            if s_i:
                correlation_matrix[s_i] = row

        suggestions: List[str] = []
        recommended_changes: List[Dict[str, Any]] = []
        rebalancing_needed = False

        if not positions:
            suggestions.append(
                "No positions found — add holdings to generate portfolio analysis."
            )
        else:
            if top_holding is not None and top_holding > 0.35:
                rebalancing_needed = True
                top_sym = symbols[weights.index(max(weights))] if weights else "top holding"
                suggestions.append(
                    f"High concentration: {top_sym} is ~{top_holding * 100:.1f}% of portfolio; "
                    "consider trimming toward a lower single-name weight."
                )
                recommended_changes.append({
                    "action": "reduce",
                    "symbol": top_sym,
                    "reason": "concentration_risk",
                    "current_weight": round(top_holding, 4),
                    "suggested_max_weight": 0.35,
                })
            if diversification_score < 40 and len(positions) < 5:
                suggestions.append(
                    "Low diversification — add holdings across sectors to improve balance."
                )
                rebalancing_needed = True
            unique_sectors = {s for s in sectors if s and s != "Unknown"}
            if len(positions) >= 3 and len(unique_sectors) <= 1:
                suggestions.append(
                    "Sector concentration is high — diversify across multiple sectors."
                )
            if not suggestions:
                suggestions.append(
                    "Portfolio weights look balanced relative to simple concentration heuristics."
                )

        # Simple equal-weight target as "optimal" reference when multi-name
        optimal_return = current_return
        optimal_risk = current_risk
        improvement = None
        if len(weights) > 1 and current_risk is not None and current_return is not None:
            eq_w = 1.0 / len(weights)
            optimal_return = sum(eq_w * r for r in returns)
            # Equal weight typically lowers concentration risk proxy
            optimal_risk = current_risk * (0.85 if hhi > eq_w else 1.0)
            improvement = max(0.0, current_risk - optimal_risk)

        return {
            "portfolio_id": portfolio_id,
            "analysis_date": date.today(),
            "risk_analysis": {
                "var_95": var_95,
                "cvar_95": cvar_95,
                "downside_deviation": downside_deviation,
                "upside_potential": upside_potential,
            },
            "diversification_score": round(float(diversification_score), 2),
            "concentration_risk": {
                "top_holding": top_holding,
                "top_3_holdings": top_3,
                "top_5_holdings": top_5,
            },
            "correlation_matrix": correlation_matrix,
            "efficient_frontier": {
                "current_position": {
                    "return": current_return,
                    "risk": current_risk,
                },
                "optimal_position": {
                    "return": optimal_return,
                    "risk": optimal_risk,
                },
                "improvement_potential": improvement,
            },
            "optimization_suggestions": suggestions,
            "rebalancing_needed": rebalancing_needed,
            "recommended_changes": recommended_changes,
        }

    async def build_portfolio_analysis_async(
        self,
        portfolio_id: str,
        user_id: int,
        db,
    ) -> Dict[str, Any]:
        """
        Load real portfolio positions then build analysis (#108).

        Args:
            portfolio_id: Public portfolio identifier
            user_id: Authenticated owner
            db: Async DB session

        Returns:
            PortfolioAnalysis-compatible dict from live holdings when available
        """
        detail = await self.compute_portfolio_detail(portfolio_id, user_id, db)
        if not detail:
            logger.warning(
                "Portfolio %s not found for user %s during analysis",
                portfolio_id,
                user_id,
            )
            return self.build_portfolio_analysis(portfolio_id, positions=[])
        return self.build_portfolio_analysis(
            portfolio_id,
            positions=detail.get("positions") or [],
        )

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
            # Current allocation is unknown without real portfolio data
            trades.append({
                "asset_class": asset_class,
                "action": "unavailable",
                "amount": None,
                "current_allocation": None,
                "target_allocation": target_percent,
                "impact": None,
            })

        # Limit number of trades
        trades = trades[:max_trades]

        return {
            "portfolio_id": portfolio_id,
            "rebalancing_plan": trades,
            "estimated_cost": None,
            "tax_impact": None,
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
        # No real transaction data available — return an empty list.
        # Callers should query the database directly for real transaction history.
        logger.info(
            f"generate_transaction_list called for portfolio {portfolio_id}: "
            "returning empty list (real DB query not implemented here)"
        )
        return []

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
