"""
Portfolio Management Domain Contract (Domain 2).

Defines the interface for portfolio and position management, including
user portfolios, positions, transactions, and watchlists.
"""

from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from .base import ContractResult, DomainContract


class OrderSide(Enum):
    """Order side for transactions."""
    BUY = "buy"
    SELL = "sell"


class TransactionStatus(Enum):
    """Status of a transaction."""
    PENDING = "pending"
    EXECUTED = "executed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class UserRole(Enum):
    """User subscription/role types."""
    FREE_USER = "free_user"
    PREMIUM_USER = "premium_user"
    ADMIN = "admin"


@dataclass(frozen=True)
class UserDTO:
    """Data transfer object for user information."""
    id: int
    email: str
    username: Optional[str] = None
    full_name: Optional[str] = None
    role: UserRole = UserRole.FREE_USER
    is_active: bool = True
    is_verified: bool = False
    subscription_tier: str = "free"
    created_at: Optional[datetime] = None


@dataclass(frozen=True)
class PortfolioDTO:
    """Data transfer object for portfolio information."""
    id: int
    user_id: int
    name: str
    description: Optional[str] = None
    cash_balance: Decimal = Decimal("0.00")
    total_value: Optional[Decimal] = None
    total_return: Optional[Decimal] = None
    total_return_pct: Optional[float] = None
    is_public: bool = False
    is_default: bool = False
    benchmark: Optional[str] = None
    created_at: Optional[datetime] = None


@dataclass(frozen=True)
class PositionDTO:
    """Data transfer object for position information."""
    id: int
    portfolio_id: int
    stock_id: int
    stock_symbol: Optional[str] = None
    stock_name: Optional[str] = None
    quantity: Decimal = Decimal("0")
    average_cost: Decimal = Decimal("0.00")
    current_price: Optional[Decimal] = None
    market_value: Optional[Decimal] = None
    unrealized_gain_loss: Optional[Decimal] = None
    unrealized_gain_loss_pct: Optional[float] = None
    realized_gain_loss: Optional[Decimal] = None
    first_purchase_date: Optional[datetime] = None
    last_transaction_date: Optional[datetime] = None


@dataclass(frozen=True)
class TransactionDTO:
    """Data transfer object for transaction information."""
    id: int
    portfolio_id: int
    stock_id: int
    stock_symbol: Optional[str] = None
    side: OrderSide = OrderSide.BUY
    quantity: Decimal = Decimal("0")
    price: Decimal = Decimal("0.00")
    total_amount: Optional[Decimal] = None
    commission: Decimal = Decimal("0.00")
    status: TransactionStatus = TransactionStatus.PENDING
    executed_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    notes: Optional[str] = None


@dataclass(frozen=True)
class WatchlistItemDTO:
    """Data transfer object for watchlist items."""
    id: int
    user_id: int
    stock_id: int
    stock_symbol: Optional[str] = None
    stock_name: Optional[str] = None
    added_date: Optional[datetime] = None
    notes: Optional[str] = None
    alert_rules: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class PortfolioSummaryDTO:
    """Summary statistics for a portfolio."""
    portfolio_id: int
    total_value: Decimal
    cash_balance: Decimal
    invested_value: Decimal
    total_return: Decimal
    total_return_pct: float
    day_change: Decimal
    day_change_pct: float
    position_count: int
    top_holdings: List[PositionDTO]
    sector_allocation: Dict[str, float]


@dataclass(frozen=True)
class AllocationDTO:
    """Target allocation for rebalancing."""
    symbol: str
    target_percentage: float
    current_percentage: Optional[float] = None
    difference: Optional[float] = None


class PortfolioContract(DomainContract):
    """
    Contract for Domain 2: Portfolio Management.

    Provides access to user portfolios, positions, transactions, and
    watchlists. Manages the user's investment holdings and tracks
    performance.
    """

    @property
    def domain_name(self) -> str:
        return "Portfolio"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def capabilities(self) -> List[str]:
        return [
            "get_user",
            "get_portfolio",
            "list_user_portfolios",
            "create_portfolio",
            "update_portfolio",
            "delete_portfolio",
            "get_position",
            "list_positions",
            "add_position",
            "update_position",
            "close_position",
            "get_transactions",
            "add_transaction",
            "get_portfolio_summary",
            "get_allocation",
            "rebalance_portfolio",
            "get_watchlist",
            "add_to_watchlist",
            "remove_from_watchlist",
        ]

    # User Operations

    @abstractmethod
    async def get_user(self, user_id: int) -> ContractResult[UserDTO]:
        """
        Get user by ID.

        Args:
            user_id: User ID

        Returns:
            ContractResult containing UserDTO if found
        """
        pass

    @abstractmethod
    async def get_user_by_email(self, email: str) -> ContractResult[UserDTO]:
        """
        Get user by email.

        Args:
            email: User email

        Returns:
            ContractResult containing UserDTO if found
        """
        pass

    # Portfolio Operations

    @abstractmethod
    async def get_portfolio(
        self,
        portfolio_id: int,
        user_id: int
    ) -> ContractResult[PortfolioDTO]:
        """
        Get portfolio by ID.

        Args:
            portfolio_id: Portfolio ID
            user_id: User ID (for authorization)

        Returns:
            ContractResult containing PortfolioDTO if found
        """
        pass

    @abstractmethod
    async def list_user_portfolios(
        self,
        user_id: int
    ) -> ContractResult[List[PortfolioDTO]]:
        """
        List all portfolios for a user.

        Args:
            user_id: User ID

        Returns:
            ContractResult containing list of portfolios
        """
        pass

    @abstractmethod
    async def create_portfolio(
        self,
        user_id: int,
        name: str,
        description: Optional[str] = None,
        cash_balance: Decimal = Decimal("0.00"),
        benchmark: Optional[str] = None
    ) -> ContractResult[PortfolioDTO]:
        """
        Create a new portfolio.

        Args:
            user_id: Owner user ID
            name: Portfolio name
            description: Optional description
            cash_balance: Initial cash balance
            benchmark: Benchmark symbol

        Returns:
            ContractResult containing created portfolio
        """
        pass

    @abstractmethod
    async def update_portfolio(
        self,
        portfolio_id: int,
        user_id: int,
        updates: Dict[str, Any]
    ) -> ContractResult[PortfolioDTO]:
        """
        Update portfolio fields.

        Args:
            portfolio_id: Portfolio ID
            user_id: User ID (for authorization)
            updates: Dictionary of fields to update

        Returns:
            ContractResult containing updated portfolio
        """
        pass

    @abstractmethod
    async def delete_portfolio(
        self,
        portfolio_id: int,
        user_id: int
    ) -> ContractResult[bool]:
        """
        Delete a portfolio.

        Args:
            portfolio_id: Portfolio ID
            user_id: User ID (for authorization)

        Returns:
            ContractResult indicating success
        """
        pass

    # Position Operations

    @abstractmethod
    async def get_position(
        self,
        portfolio_id: int,
        stock_id: int
    ) -> ContractResult[PositionDTO]:
        """
        Get a specific position.

        Args:
            portfolio_id: Portfolio ID
            stock_id: Stock ID

        Returns:
            ContractResult containing position if found
        """
        pass

    @abstractmethod
    async def list_positions(
        self,
        portfolio_id: int
    ) -> ContractResult[List[PositionDTO]]:
        """
        List all positions in a portfolio.

        Args:
            portfolio_id: Portfolio ID

        Returns:
            ContractResult containing list of positions
        """
        pass

    @abstractmethod
    async def add_position(
        self,
        portfolio_id: int,
        stock_id: int,
        quantity: Decimal,
        average_cost: Decimal
    ) -> ContractResult[PositionDTO]:
        """
        Add or update a position.

        Args:
            portfolio_id: Portfolio ID
            stock_id: Stock ID
            quantity: Number of shares
            average_cost: Average cost per share

        Returns:
            ContractResult containing position
        """
        pass

    @abstractmethod
    async def close_position(
        self,
        portfolio_id: int,
        stock_id: int,
        price: Decimal
    ) -> ContractResult[TransactionDTO]:
        """
        Close a position entirely.

        Args:
            portfolio_id: Portfolio ID
            stock_id: Stock ID
            price: Closing price

        Returns:
            ContractResult containing the closing transaction
        """
        pass

    # Transaction Operations

    @abstractmethod
    async def get_transactions(
        self,
        portfolio_id: int,
        stock_id: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> ContractResult[List[TransactionDTO]]:
        """
        Get transactions for a portfolio.

        Args:
            portfolio_id: Portfolio ID
            stock_id: Optional stock filter
            start_date: Optional start date filter
            end_date: Optional end date filter
            limit: Maximum results

        Returns:
            ContractResult containing list of transactions
        """
        pass

    @abstractmethod
    async def add_transaction(
        self,
        portfolio_id: int,
        stock_id: int,
        side: OrderSide,
        quantity: Decimal,
        price: Decimal,
        commission: Decimal = Decimal("0.00"),
        notes: Optional[str] = None
    ) -> ContractResult[TransactionDTO]:
        """
        Record a new transaction.

        Args:
            portfolio_id: Portfolio ID
            stock_id: Stock ID
            side: Buy or sell
            quantity: Number of shares
            price: Price per share
            commission: Trading commission
            notes: Optional notes

        Returns:
            ContractResult containing created transaction
        """
        pass

    # Portfolio Analysis

    @abstractmethod
    async def get_portfolio_summary(
        self,
        portfolio_id: int
    ) -> ContractResult[PortfolioSummaryDTO]:
        """
        Get portfolio summary with performance metrics.

        Args:
            portfolio_id: Portfolio ID

        Returns:
            ContractResult containing portfolio summary
        """
        pass

    @abstractmethod
    async def get_allocation(
        self,
        portfolio_id: int
    ) -> ContractResult[List[AllocationDTO]]:
        """
        Get current portfolio allocation.

        Args:
            portfolio_id: Portfolio ID

        Returns:
            ContractResult containing allocation breakdown
        """
        pass

    @abstractmethod
    async def calculate_rebalance(
        self,
        portfolio_id: int,
        target_allocation: Dict[str, float]
    ) -> ContractResult[List[TransactionDTO]]:
        """
        Calculate transactions needed to rebalance.

        Args:
            portfolio_id: Portfolio ID
            target_allocation: Target allocation by symbol

        Returns:
            ContractResult containing proposed transactions
        """
        pass

    # Watchlist Operations

    @abstractmethod
    async def get_watchlist(
        self,
        user_id: int
    ) -> ContractResult[List[WatchlistItemDTO]]:
        """
        Get user's watchlist.

        Args:
            user_id: User ID

        Returns:
            ContractResult containing watchlist items
        """
        pass

    @abstractmethod
    async def add_to_watchlist(
        self,
        user_id: int,
        stock_id: int,
        notes: Optional[str] = None
    ) -> ContractResult[WatchlistItemDTO]:
        """
        Add stock to watchlist.

        Args:
            user_id: User ID
            stock_id: Stock ID
            notes: Optional notes

        Returns:
            ContractResult containing watchlist item
        """
        pass

    @abstractmethod
    async def remove_from_watchlist(
        self,
        user_id: int,
        stock_id: int
    ) -> ContractResult[bool]:
        """
        Remove stock from watchlist.

        Args:
            user_id: User ID
            stock_id: Stock ID

        Returns:
            ContractResult indicating success
        """
        pass

    # Health Check

    async def health_check(self) -> ContractResult[Dict[str, Any]]:
        """Check portfolio service health."""
        return ContractResult.ok({
            "domain": self.domain_name,
            "version": self.version,
            "status": "healthy",
            "capabilities": self.capabilities,
        })
