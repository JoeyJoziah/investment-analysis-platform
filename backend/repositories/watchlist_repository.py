"""
Watchlist Repository
Specialized async repository for watchlist-related operations with user-scoped access.
"""

from typing import List, Optional, Dict, Any
from decimal import Decimal
from datetime import datetime
import logging

from sqlalchemy import select, func, and_, delete, update
from sqlalchemy.orm import selectinload, joinedload
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError

from backend.repositories.base import AsyncCRUDRepository, FilterCriteria, PaginationParams
from backend.models.tables import Watchlist, WatchlistItem, Stock, PriceHistory
from backend.config.database import get_db_session

logger = logging.getLogger(__name__)


class WatchlistRepository(AsyncCRUDRepository[Watchlist]):
    """
    Repository for Watchlist with user-scoped operations.

    Provides comprehensive watchlist management including:
    - User-scoped CRUD operations
    - Watchlist item management
    - Stock price integration
    - Alert-enabled item queries
    """

    # Default watchlist name
    DEFAULT_WATCHLIST_NAME = "My Watchlist"

    # Maximum items per user (from SystemSettings.max_watchlist_size)
    MAX_ITEMS_PER_USER = 50

    def __init__(self):
        super().__init__(Watchlist)

    async def get_user_watchlists(
        self,
        user_id: int,
        include_items: bool = False,
        session: Optional[AsyncSession] = None
    ) -> List[Watchlist]:
        """
        Get all watchlists for a user.

        Args:
            user_id: User ID
            include_items: Whether to load watchlist items
            session: Optional existing session

        Returns:
            List of user's watchlists
        """
        async def _get_watchlists(session: AsyncSession) -> List[Watchlist]:
            query = select(Watchlist).where(Watchlist.user_id == user_id)

            if include_items:
                query = query.options(
                    selectinload(Watchlist.items).selectinload(WatchlistItem.stock)
                )

            query = query.order_by(Watchlist.created_at.desc())

            result = await session.execute(query)
            return result.scalars().all()

        if session:
            return await _get_watchlists(session)
        else:
            async with get_db_session(readonly=True) as session:
                return await _get_watchlists(session)

    async def get_watchlist_by_name(
        self,
        user_id: int,
        name: str,
        session: Optional[AsyncSession] = None
    ) -> Optional[Watchlist]:
        """
        Get specific watchlist by name for a user.

        Args:
            user_id: User ID
            name: Watchlist name
            session: Optional existing session

        Returns:
            Watchlist or None if not found
        """
        async def _get_by_name(session: AsyncSession) -> Optional[Watchlist]:
            query = select(Watchlist).where(
                and_(
                    Watchlist.user_id == user_id,
                    Watchlist.name == name
                )
            ).options(
                selectinload(Watchlist.items).selectinload(WatchlistItem.stock)
            )

            result = await session.execute(query)
            return result.scalar_one_or_none()

        if session:
            return await _get_by_name(session)
        else:
            async with get_db_session(readonly=True) as session:
                return await _get_by_name(session)

    async def get_default_watchlist(
        self,
        user_id: int,
        session: Optional[AsyncSession] = None
    ) -> Watchlist:
        """
        Get or create the default 'My Watchlist' for a user.

        Args:
            user_id: User ID
            session: Optional existing session

        Returns:
            Default watchlist (creates if doesn't exist)
        """
        async def _get_or_create_default(session: AsyncSession) -> Watchlist:
            # Try to find existing default watchlist
            watchlist = await self.get_watchlist_by_name(
                user_id,
                self.DEFAULT_WATCHLIST_NAME,
                session=session
            )

            if watchlist:
                return watchlist

            # Create default watchlist
            return await self.create_watchlist(
                user_id=user_id,
                name=self.DEFAULT_WATCHLIST_NAME,
                description="Your default watchlist",
                is_public=False,
                session=session
            )

        if session:
            return await _get_or_create_default(session)
        else:
            async with get_db_session() as session:
                return await _get_or_create_default(session)

    async def create_watchlist(
        self,
        user_id: int,
        name: str,
        description: Optional[str] = None,
        is_public: bool = False,
        session: Optional[AsyncSession] = None
    ) -> Watchlist:
        """
        Create a new watchlist.

        Args:
            user_id: User ID
            name: Watchlist name
            description: Optional description
            is_public: Whether watchlist is public
            session: Optional existing session

        Returns:
            Created watchlist

        Raises:
            IntegrityError: If watchlist name already exists for user
        """
        async def _create(session: AsyncSession) -> Watchlist:
            watchlist_data = {
                "user_id": user_id,
                "name": name,
                "description": description,
                "is_public": is_public
            }

            watchlist = Watchlist(**watchlist_data)
            session.add(watchlist)
            await session.flush()
            await session.refresh(watchlist)

            logger.info(f"Created watchlist '{name}' for user {user_id}")
            return watchlist

        if session:
            return await _create(session)
        else:
            async with get_db_session() as session:
                return await _create(session)

    async def add_item_to_watchlist(
        self,
        watchlist_id: int,
        stock_id: int,
        target_price: Optional[Decimal] = None,
        notes: Optional[str] = None,
        alert_enabled: bool = False,
        session: Optional[AsyncSession] = None
    ) -> WatchlistItem:
        """
        Add stock to watchlist.

        Args:
            watchlist_id: Watchlist ID
            stock_id: Stock ID to add
            target_price: Optional target price for alerts
            notes: Optional notes
            alert_enabled: Whether to enable price alerts
            session: Optional existing session

        Returns:
            Created watchlist item

        Raises:
            IntegrityError: If stock already in watchlist
            ValueError: If user item limit exceeded
        """
        async def _add_item(session: AsyncSession) -> WatchlistItem:
            # Check if stock already in watchlist
            existing = await self.is_stock_in_watchlist(
                watchlist_id,
                stock_id,
                session=session
            )
            if existing:
                raise IntegrityError(
                    "Stock already in watchlist",
                    params={"watchlist_id": watchlist_id, "stock_id": stock_id},
                    orig=None
                )

            # Get watchlist to check user's total item count
            watchlist = await self.get_by_id(watchlist_id, session=session)
            if watchlist:
                total_items = await self.count_user_items(
                    watchlist.user_id,
                    session=session
                )
                if total_items >= self.MAX_ITEMS_PER_USER:
                    raise ValueError(
                        f"Maximum watchlist items ({self.MAX_ITEMS_PER_USER}) exceeded"
                    )

            item = WatchlistItem(
                watchlist_id=watchlist_id,
                stock_id=stock_id,
                target_price=target_price,
                notes=notes,
                alert_enabled=alert_enabled
            )

            session.add(item)
            await session.flush()
            await session.refresh(item)

            logger.info(f"Added stock {stock_id} to watchlist {watchlist_id}")
            return item

        if session:
            return await _add_item(session)
        else:
            async with get_db_session() as session:
                return await _add_item(session)

    async def remove_item_from_watchlist(
        self,
        watchlist_id: int,
        stock_id: int,
        session: Optional[AsyncSession] = None
    ) -> bool:
        """
        Remove stock from watchlist.

        Args:
            watchlist_id: Watchlist ID
            stock_id: Stock ID to remove
            session: Optional existing session

        Returns:
            True if removed, False if not found
        """
        async def _remove_item(session: AsyncSession) -> bool:
            stmt = delete(WatchlistItem).where(
                and_(
                    WatchlistItem.watchlist_id == watchlist_id,
                    WatchlistItem.stock_id == stock_id
                )
            )
            result = await session.execute(stmt)

            if result.rowcount > 0:
                logger.info(f"Removed stock {stock_id} from watchlist {watchlist_id}")
                return True
            return False

        if session:
            return await _remove_item(session)
        else:
            async with get_db_session() as session:
                return await _remove_item(session)

    async def update_item(
        self,
        item_id: int,
        target_price: Optional[Decimal] = None,
        notes: Optional[str] = None,
        alert_enabled: Optional[bool] = None,
        session: Optional[AsyncSession] = None
    ) -> Optional[WatchlistItem]:
        """
        Update watchlist item.

        Args:
            item_id: Watchlist item ID
            target_price: New target price (pass None to keep, pass 0 to clear)
            notes: New notes
            alert_enabled: New alert status
            session: Optional existing session

        Returns:
            Updated item or None if not found
        """
        async def _update_item(session: AsyncSession) -> Optional[WatchlistItem]:
            # Build update data
            update_data = {}

            # Only update fields that are explicitly provided
            if target_price is not None:
                update_data["target_price"] = target_price if target_price > 0 else None
            if notes is not None:
                update_data["notes"] = notes
            if alert_enabled is not None:
                update_data["alert_enabled"] = alert_enabled

            if not update_data:
                # No updates, return existing item
                query = select(WatchlistItem).where(WatchlistItem.id == item_id)
                result = await session.execute(query)
                return result.scalar_one_or_none()

            stmt = (
                update(WatchlistItem)
                .where(WatchlistItem.id == item_id)
                .values(**update_data)
            )
            result = await session.execute(stmt)

            if result.rowcount == 0:
                return None

            # Fetch updated item
            query = select(WatchlistItem).where(WatchlistItem.id == item_id)
            result = await session.execute(query)
            return result.scalar_one_or_none()

        if session:
            return await _update_item(session)
        else:
            async with get_db_session() as session:
                return await _update_item(session)

    async def get_watchlist_items_with_prices(
        self,
        watchlist_id: int,
        session: Optional[AsyncSession] = None
    ) -> List[Dict[str, Any]]:
        """
        Get watchlist items with current stock data and prices.

        Args:
            watchlist_id: Watchlist ID
            session: Optional existing session

        Returns:
            List of items with stock data and latest prices
        """
        async def _get_items_with_prices(session: AsyncSession) -> List[Dict[str, Any]]:
            # Subquery for latest price
            latest_price_subq = (
                select(
                    PriceHistory.stock_id,
                    PriceHistory.close.label('current_price'),
                    PriceHistory.date.label('price_date'),
                    PriceHistory.volume,
                    func.row_number().over(
                        partition_by=PriceHistory.stock_id,
                        order_by=PriceHistory.date.desc()
                    ).label('rn')
                ).subquery()
            )

            # Previous day price for change calculation
            prev_price_subq = (
                select(
                    PriceHistory.stock_id,
                    PriceHistory.close.label('prev_price'),
                    func.row_number().over(
                        partition_by=PriceHistory.stock_id,
                        order_by=PriceHistory.date.desc()
                    ).label('rn')
                ).subquery()
            )

            # Main query joining items with stock and price data
            query = (
                select(
                    WatchlistItem,
                    Stock,
                    latest_price_subq.c.current_price,
                    latest_price_subq.c.volume,
                    prev_price_subq.c.prev_price
                )
                .join(Stock, WatchlistItem.stock_id == Stock.id)
                .outerjoin(
                    latest_price_subq,
                    and_(
                        Stock.id == latest_price_subq.c.stock_id,
                        latest_price_subq.c.rn == 1
                    )
                )
                .outerjoin(
                    prev_price_subq,
                    and_(
                        Stock.id == prev_price_subq.c.stock_id,
                        prev_price_subq.c.rn == 2
                    )
                )
                .where(WatchlistItem.watchlist_id == watchlist_id)
                .order_by(WatchlistItem.added_at.desc())
            )

            result = await session.execute(query)
            rows = result.all()

            items_with_data = []
            for row in rows:
                item = row.WatchlistItem
                stock = row.Stock
                current_price = float(row.current_price) if row.current_price else None
                prev_price = float(row.prev_price) if row.prev_price else None
                volume = row.volume

                # Calculate price change
                price_change = None
                price_change_percent = None
                if current_price and prev_price:
                    price_change = current_price - prev_price
                    price_change_percent = (price_change / prev_price) * 100

                items_with_data.append({
                    "id": item.id,
                    "watchlist_id": item.watchlist_id,
                    "stock_id": item.stock_id,
                    "target_price": float(item.target_price) if item.target_price else None,
                    "notes": item.notes,
                    "alert_enabled": item.alert_enabled,
                    "added_at": item.added_at,
                    # Stock details
                    "symbol": stock.symbol,
                    "company_name": stock.name,
                    "sector": stock.sector,
                    "market_cap": stock.market_cap,
                    # Price data
                    "current_price": current_price,
                    "price_change": round(price_change, 4) if price_change else None,
                    "price_change_percent": round(price_change_percent, 2) if price_change_percent else None,
                    "volume": volume
                })

            return items_with_data

        if session:
            return await _get_items_with_prices(session)
        else:
            async with get_db_session(readonly=True) as session:
                return await _get_items_with_prices(session)

    async def count_user_items(
        self,
        user_id: int,
        session: Optional[AsyncSession] = None
    ) -> int:
        """
        Count total items across all user's watchlists.

        Args:
            user_id: User ID
            session: Optional existing session

        Returns:
            Total item count
        """
        async def _count_items(session: AsyncSession) -> int:
            query = (
                select(func.count(WatchlistItem.id))
                .join(Watchlist, WatchlistItem.watchlist_id == Watchlist.id)
                .where(Watchlist.user_id == user_id)
            )
            result = await session.execute(query)
            return result.scalar() or 0

        if session:
            return await _count_items(session)
        else:
            async with get_db_session(readonly=True) as session:
                return await _count_items(session)

    async def is_stock_in_watchlist(
        self,
        watchlist_id: int,
        stock_id: int,
        session: Optional[AsyncSession] = None
    ) -> bool:
        """
        Check if stock is already in watchlist.

        Args:
            watchlist_id: Watchlist ID
            stock_id: Stock ID
            session: Optional existing session

        Returns:
            True if stock exists in watchlist
        """
        async def _check_exists(session: AsyncSession) -> bool:
            query = (
                select(func.count())
                .select_from(WatchlistItem)
                .where(
                    and_(
                        WatchlistItem.watchlist_id == watchlist_id,
                        WatchlistItem.stock_id == stock_id
                    )
                )
            )
            result = await session.execute(query)
            return result.scalar() > 0

        if session:
            return await _check_exists(session)
        else:
            async with get_db_session(readonly=True) as session:
                return await _check_exists(session)

    async def get_items_with_alerts(
        self,
        session: Optional[AsyncSession] = None
    ) -> List[WatchlistItem]:
        """
        Get all watchlist items with alerts enabled.
        Used for background alert processing.

        Args:
            session: Optional existing session

        Returns:
            List of items with alerts enabled
        """
        async def _get_alert_items(session: AsyncSession) -> List[WatchlistItem]:
            query = (
                select(WatchlistItem)
                .options(
                    joinedload(WatchlistItem.stock),
                    joinedload(WatchlistItem.watchlist)
                )
                .where(
                    and_(
                        WatchlistItem.alert_enabled == True,
                        WatchlistItem.target_price.isnot(None)
                    )
                )
            )

            result = await session.execute(query)
            return result.scalars().unique().all()

        if session:
            return await _get_alert_items(session)
        else:
            async with get_db_session(readonly=True) as session:
                return await _get_alert_items(session)

    async def get_public_watchlists(
        self,
        pagination: Optional[PaginationParams] = None,
        session: Optional[AsyncSession] = None
    ) -> List[Watchlist]:
        """
        Get public watchlists for discovery.

        Args:
            pagination: Pagination parameters
            session: Optional existing session

        Returns:
            List of public watchlists
        """
        async def _get_public(session: AsyncSession) -> List[Watchlist]:
            query = (
                select(Watchlist)
                .options(selectinload(Watchlist.items))
                .where(Watchlist.is_public == True)
                .order_by(Watchlist.updated_at.desc())
            )

            if pagination:
                query = query.offset(pagination.offset).limit(pagination.limit)

            result = await session.execute(query)
            return result.scalars().all()

        if session:
            return await _get_public(session)
        else:
            async with get_db_session(readonly=True) as session:
                return await _get_public(session)

    async def get_watchlist_with_items(
        self,
        watchlist_id: int,
        user_id: Optional[int] = None,
        session: Optional[AsyncSession] = None
    ) -> Optional[Watchlist]:
        """
        Get watchlist with items, optionally verifying ownership.

        Args:
            watchlist_id: Watchlist ID
            user_id: Optional user ID to verify ownership
            session: Optional existing session

        Returns:
            Watchlist with items or None if not found/not authorized
        """
        async def _get_with_items(session: AsyncSession) -> Optional[Watchlist]:
            query = (
                select(Watchlist)
                .options(
                    selectinload(Watchlist.items).selectinload(WatchlistItem.stock)
                )
                .where(Watchlist.id == watchlist_id)
            )

            # Add ownership check if user_id provided
            if user_id is not None:
                query = query.where(
                    (Watchlist.user_id == user_id) | (Watchlist.is_public == True)
                )

            result = await session.execute(query)
            return result.scalar_one_or_none()

        if session:
            return await _get_with_items(session)
        else:
            async with get_db_session(readonly=True) as session:
                return await _get_with_items(session)

    async def delete_watchlist(
        self,
        watchlist_id: int,
        user_id: int,
        session: Optional[AsyncSession] = None
    ) -> bool:
        """
        Delete a watchlist (user-scoped).

        Args:
            watchlist_id: Watchlist ID
            user_id: User ID (for ownership verification)
            session: Optional existing session

        Returns:
            True if deleted, False if not found or not authorized
        """
        async def _delete(session: AsyncSession) -> bool:
            stmt = delete(Watchlist).where(
                and_(
                    Watchlist.id == watchlist_id,
                    Watchlist.user_id == user_id
                )
            )
            result = await session.execute(stmt)

            if result.rowcount > 0:
                logger.info(f"Deleted watchlist {watchlist_id} for user {user_id}")
                return True
            return False

        if session:
            return await _delete(session)
        else:
            async with get_db_session() as session:
                return await _delete(session)

    async def get_watchlist_summary(
        self,
        watchlist_id: int,
        session: Optional[AsyncSession] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get watchlist summary with aggregated data.

        Args:
            watchlist_id: Watchlist ID
            session: Optional existing session

        Returns:
            Summary dict with item count, total value, etc.
        """
        async def _get_summary(session: AsyncSession) -> Optional[Dict[str, Any]]:
            # Get watchlist
            watchlist = await self.get_by_id(watchlist_id, session=session)
            if not watchlist:
                return None

            # Get items with prices
            items = await self.get_watchlist_items_with_prices(
                watchlist_id,
                session=session
            )

            # Calculate totals
            total_value = 0.0
            total_change = 0.0

            for item in items:
                if item.get("current_price") and item.get("market_cap"):
                    # Using market cap as proxy for position value in watchlist
                    total_value += item["market_cap"]
                if item.get("price_change_percent"):
                    total_change += item["price_change_percent"]

            avg_change = total_change / len(items) if items else 0.0

            return {
                "id": watchlist.id,
                "name": watchlist.name,
                "description": watchlist.description,
                "item_count": len(items),
                "total_value": total_value,
                "daily_change_percent": round(avg_change, 2),
                "created_at": watchlist.created_at,
                "updated_at": watchlist.updated_at
            }

        if session:
            return await _get_summary(session)
        else:
            async with get_db_session(readonly=True) as session:
                return await _get_summary(session)


# Create singleton instance
watchlist_repository = WatchlistRepository()
