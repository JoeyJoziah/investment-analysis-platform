"""
Watchlist Service
Business logic for watchlist management operations.
Provides the layer between API routers and repositories.
"""

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from backend.repositories.watchlist_repository import watchlist_repository
from backend.repositories.stock_repository import stock_repository
from backend.models.schemas import (
    WatchlistItemResponse,
    WatchlistResponse,
    WatchlistSummary,
)

logger = logging.getLogger(__name__)


# =======================
# Conversion Helpers
# =======================

def convert_watchlist_to_response(
    watchlist: Any,
    items_data: List[Dict] = None,
) -> WatchlistResponse:
    """Convert a watchlist model to WatchlistResponse with items."""
    items = []

    if items_data:
        # Use pre-fetched items with price data
        for item in items_data:
            items.append(WatchlistItemResponse(
                id=item["id"],
                watchlist_id=item["watchlist_id"],
                stock_id=item["stock_id"],
                target_price=item.get("target_price"),
                notes=item.get("notes"),
                alert_enabled=item.get("alert_enabled", False),
                added_at=item["added_at"],
                symbol=item["symbol"],
                company_name=item.get("company_name"),
                current_price=item.get("current_price"),
                price_change=item.get("price_change"),
                price_change_percent=item.get("price_change_percent"),
                volume=item.get("volume"),
                market_cap=item.get("market_cap"),
                sector=item.get("sector"),
            ))
    elif hasattr(watchlist, 'items') and watchlist.items:
        # Use loaded relationship items
        for item in watchlist.items:
            stock = item.stock if hasattr(item, 'stock') else None
            items.append(WatchlistItemResponse(
                id=item.id,
                watchlist_id=item.watchlist_id,
                stock_id=item.stock_id,
                target_price=float(item.target_price) if item.target_price else None,
                notes=item.notes,
                alert_enabled=item.alert_enabled,
                added_at=item.added_at,
                symbol=stock.symbol if stock else "UNKNOWN",
                company_name=stock.name if stock else None,
                current_price=None,  # Would need price lookup
                price_change=None,
                price_change_percent=None,
                volume=None,
                market_cap=stock.market_cap if stock else None,
                sector=stock.sector if stock else None,
            ))

    return WatchlistResponse(
        id=watchlist.id,
        user_id=watchlist.user_id,
        name=watchlist.name,
        description=watchlist.description,
        is_public=watchlist.is_public,
        created_at=watchlist.created_at,
        updated_at=watchlist.updated_at,
        items=items,
        item_count=len(items),
    )


def convert_watchlist_to_summary(
    watchlist: Any,
    summary_data: Dict = None,
) -> WatchlistSummary:
    """Convert a watchlist to summary format."""
    item_count = 0
    if hasattr(watchlist, 'items') and watchlist.items:
        item_count = len(watchlist.items)
    elif summary_data and 'item_count' in summary_data:
        item_count = summary_data['item_count']

    return WatchlistSummary(
        id=watchlist.id,
        name=watchlist.name,
        description=watchlist.description,
        item_count=item_count,
        total_value=summary_data.get('total_value') if summary_data else None,
        daily_change_percent=(
            summary_data.get('daily_change_percent') if summary_data else None
        ),
        created_at=watchlist.created_at,
        updated_at=watchlist.updated_at,
    )


# =======================
# Watchlist Service Class
# =======================

class WatchlistService:
    """
    Service for watchlist management operations.
    Provides business logic layer between API and repositories.
    """

    # ------------------------------------------------------------------
    # Watchlist helpers
    # ------------------------------------------------------------------

    async def get_watchlist_with_access_check(
        self,
        watchlist_id: int,
        user_id: int,
        db: AsyncSession,
        require_ownership: bool = True,
    ) -> Any:
        """
        Fetch a watchlist by ID and verify access rights.

        Returns the watchlist if found and access is permitted.
        Returns None if not found.
        Raises PermissionError if the caller is not authorised.
        """
        watchlist = await watchlist_repository.get_watchlist_with_items(
            watchlist_id,
            user_id=user_id if require_ownership else None,
            session=db,
        )

        if not watchlist:
            return None

        if require_ownership and watchlist.user_id != user_id:
            raise PermissionError(
                f"Not authorized to access watchlist {watchlist_id}"
            )

        return watchlist

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    async def list_user_watchlists(
        self,
        user_id: int,
        include_items: bool,
        db: AsyncSession,
    ) -> List[WatchlistSummary]:
        """Return summary list of all watchlists owned by the user."""
        watchlists = await watchlist_repository.get_user_watchlists(
            user_id=user_id,
            include_items=include_items,
            session=db,
        )

        summaries = []
        for watchlist in watchlists:
            summary_data = None
            if include_items:
                summary_data = await watchlist_repository.get_watchlist_summary(
                    watchlist.id,
                    session=db,
                )
            summaries.append(convert_watchlist_to_summary(watchlist, summary_data))

        logger.debug(
            "Listed %d watchlists for user %d", len(summaries), user_id
        )
        return summaries

    async def get_default_watchlist_response(
        self,
        user_id: int,
        db: AsyncSession,
    ) -> WatchlistResponse:
        """Get (or auto-create) the user's default watchlist with price data."""
        watchlist = await watchlist_repository.get_default_watchlist(
            user_id=user_id,
            session=db,
        )
        items_data = await watchlist_repository.get_watchlist_items_with_prices(
            watchlist.id,
            session=db,
        )
        return convert_watchlist_to_response(watchlist, items_data)

    async def get_watchlist_response(
        self,
        watchlist_id: int,
        user_id: int,
        db: AsyncSession,
    ) -> Optional[WatchlistResponse]:
        """
        Fetch a watchlist with items and price data.

        Access is granted to the owner or any user when the watchlist is public.
        Returns None if the watchlist does not exist.
        Raises PermissionError for private watchlists owned by another user.
        """
        watchlist = await watchlist_repository.get_watchlist_with_items(
            watchlist_id,
            user_id=None,
            session=db,
        )

        if not watchlist:
            return None

        if watchlist.user_id != user_id and not watchlist.is_public:
            raise PermissionError(
                f"Not authorized to access watchlist {watchlist_id}"
            )

        items_data = await watchlist_repository.get_watchlist_items_with_prices(
            watchlist.id,
            session=db,
        )
        return convert_watchlist_to_response(watchlist, items_data)

    # ------------------------------------------------------------------
    # Watchlist mutations
    # ------------------------------------------------------------------

    async def create_watchlist(
        self,
        user_id: int,
        name: str,
        description: Optional[str],
        is_public: bool,
        db: AsyncSession,
    ) -> WatchlistResponse:
        """Create a new watchlist and return its response representation."""
        watchlist = await watchlist_repository.create_watchlist(
            user_id=user_id,
            name=name,
            description=description,
            is_public=is_public,
            session=db,
        )
        logger.info("Created watchlist '%s' for user %d", watchlist.name, user_id)
        return convert_watchlist_to_response(watchlist)

    async def update_watchlist(
        self,
        watchlist_id: int,
        user_id: int,
        name: Optional[str],
        description: Optional[str],
        is_public: Optional[bool],
        db: AsyncSession,
    ) -> Optional[WatchlistResponse]:
        """
        Update mutable fields on a watchlist.

        Returns None if the watchlist does not exist or is not owned by the user.
        """
        watchlist = await self.get_watchlist_with_access_check(
            watchlist_id, user_id, db, require_ownership=True
        )
        if not watchlist:
            return None

        update_data: Dict[str, Any] = {}
        if name is not None:
            update_data["name"] = name
        if description is not None:
            update_data["description"] = description
        if is_public is not None:
            update_data["is_public"] = is_public

        if update_data:
            for key, value in update_data.items():
                setattr(watchlist, key, value)
            watchlist.updated_at = datetime.now(timezone.utc)
            await db.flush()
            await db.refresh(watchlist)

        items_data = await watchlist_repository.get_watchlist_items_with_prices(
            watchlist.id,
            session=db,
        )
        logger.info("Updated watchlist %d for user %d", watchlist_id, user_id)
        return convert_watchlist_to_response(watchlist, items_data)

    async def delete_watchlist(
        self,
        watchlist_id: int,
        user_id: int,
        db: AsyncSession,
    ) -> bool:
        """
        Delete a watchlist and all its items.

        Returns True on success, False if the watchlist was not found.
        """
        deleted = await watchlist_repository.delete_watchlist(
            watchlist_id=watchlist_id,
            user_id=user_id,
            session=db,
        )
        if deleted:
            logger.info("Deleted watchlist %d for user %d", watchlist_id, user_id)
        return deleted

    # ------------------------------------------------------------------
    # Item mutations
    # ------------------------------------------------------------------

    async def add_item(
        self,
        watchlist_id: int,
        user_id: int,
        symbol: str,
        target_price: Optional[float],
        notes: Optional[str],
        alert_enabled: bool,
        db: AsyncSession,
    ) -> WatchlistItemResponse:
        """
        Add a stock to a watchlist by symbol.

        Raises LookupError if the stock is not found.
        The caller must first confirm the watchlist exists and is owned by user_id.
        """
        stock = await stock_repository.get_by_symbol(symbol, session=db)
        if not stock:
            raise LookupError(f"Stock with symbol '{symbol}' not found")

        decimal_price = Decimal(str(target_price)) if target_price else None
        item = await watchlist_repository.add_item_to_watchlist(
            watchlist_id=watchlist_id,
            stock_id=stock.id,
            target_price=decimal_price,
            notes=notes,
            alert_enabled=alert_enabled,
            session=db,
        )

        logger.info("Added %s to watchlist %d", symbol, watchlist_id)

        return WatchlistItemResponse(
            id=item.id,
            watchlist_id=item.watchlist_id,
            stock_id=item.stock_id,
            target_price=float(item.target_price) if item.target_price else None,
            notes=item.notes,
            alert_enabled=item.alert_enabled,
            added_at=item.added_at,
            symbol=stock.symbol,
            company_name=stock.name,
            current_price=None,
            price_change=None,
            price_change_percent=None,
            volume=None,
            market_cap=stock.market_cap,
            sector=stock.sector,
        )

    async def update_item(
        self,
        watchlist_id: int,
        item_id: int,
        user_id: int,
        target_price: Optional[float],
        notes: Optional[str],
        alert_enabled: Optional[bool],
        db: AsyncSession,
    ) -> Optional[WatchlistItemResponse]:
        """
        Update a watchlist item's fields.

        Returns None if the item does not exist or does not belong to the watchlist.
        """
        decimal_price = None
        if target_price is not None:
            decimal_price = Decimal(str(target_price))

        updated_item = await watchlist_repository.update_item(
            item_id=item_id,
            target_price=decimal_price,
            notes=notes,
            alert_enabled=alert_enabled,
            session=db,
        )

        if not updated_item:
            return None

        if updated_item.watchlist_id != watchlist_id:
            return None

        stock = await stock_repository.get_by_id(updated_item.stock_id, session=db)

        logger.info("Updated item %d in watchlist %d", item_id, watchlist_id)

        return WatchlistItemResponse(
            id=updated_item.id,
            watchlist_id=updated_item.watchlist_id,
            stock_id=updated_item.stock_id,
            target_price=(
                float(updated_item.target_price) if updated_item.target_price else None
            ),
            notes=updated_item.notes,
            alert_enabled=updated_item.alert_enabled,
            added_at=updated_item.added_at,
            symbol=stock.symbol if stock else "UNKNOWN",
            company_name=stock.name if stock else None,
            current_price=None,
            price_change=None,
            price_change_percent=None,
            volume=None,
            market_cap=stock.market_cap if stock else None,
            sector=stock.sector if stock else None,
        )

    async def remove_item(
        self,
        watchlist_id: int,
        item_id: int,
        user_id: int,
        db: AsyncSession,
    ) -> bool:
        """
        Remove an item from a watchlist by item ID.

        Returns True on success, False if the item was not found.
        """
        from sqlalchemy import select
        from backend.models.unified_models import WatchlistItem

        query = select(WatchlistItem).where(
            WatchlistItem.id == item_id,
            WatchlistItem.watchlist_id == watchlist_id,
        )
        result = await db.execute(query)
        item = result.scalar_one_or_none()

        if not item:
            return False

        removed = await watchlist_repository.remove_item_from_watchlist(
            watchlist_id=watchlist_id,
            stock_id=item.stock_id,
            session=db,
        )

        if removed:
            logger.info("Removed item %d from watchlist %d", item_id, watchlist_id)
        return bool(removed)

    # ------------------------------------------------------------------
    # Default-watchlist convenience operations
    # ------------------------------------------------------------------

    async def add_to_default_watchlist(
        self,
        user_id: int,
        symbol: str,
        db: AsyncSession,
    ) -> WatchlistItemResponse:
        """
        Add a stock to the user's default watchlist (creating it if absent).

        Raises LookupError if the stock symbol is not recognised.
        """
        watchlist = await watchlist_repository.get_default_watchlist(
            user_id=user_id,
            session=db,
        )

        stock = await stock_repository.get_by_symbol(symbol, session=db)
        if not stock:
            raise LookupError(f"Stock with symbol '{symbol}' not found")

        item = await watchlist_repository.add_item_to_watchlist(
            watchlist_id=watchlist.id,
            stock_id=stock.id,
            session=db,
        )

        logger.info("Added %s to default watchlist for user %d", symbol, user_id)

        return WatchlistItemResponse(
            id=item.id,
            watchlist_id=item.watchlist_id,
            stock_id=item.stock_id,
            target_price=float(item.target_price) if item.target_price else None,
            notes=item.notes,
            alert_enabled=item.alert_enabled,
            added_at=item.added_at,
            symbol=stock.symbol,
            company_name=stock.name,
            current_price=None,
            price_change=None,
            price_change_percent=None,
            volume=None,
            market_cap=stock.market_cap,
            sector=stock.sector,
        )

    async def remove_from_default_watchlist(
        self,
        user_id: int,
        symbol: str,
        db: AsyncSession,
    ) -> bool:
        """
        Remove a stock from the user's default watchlist.

        Raises LookupError if the stock is not found.
        Returns False if the stock was not in the watchlist.
        """
        watchlist = await watchlist_repository.get_default_watchlist(
            user_id=user_id,
            session=db,
        )

        stock = await stock_repository.get_by_symbol(symbol, session=db)
        if not stock:
            raise LookupError(f"Stock with symbol '{symbol}' not found")

        in_watchlist = await watchlist_repository.is_stock_in_watchlist(
            watchlist_id=watchlist.id,
            stock_id=stock.id,
            session=db,
        )

        if not in_watchlist:
            return False

        await watchlist_repository.remove_item_from_watchlist(
            watchlist_id=watchlist.id,
            stock_id=stock.id,
            session=db,
        )

        logger.info(
            "Removed %s from default watchlist for user %d", symbol, user_id
        )
        return True

    async def check_symbol_in_watchlists(
        self,
        user_id: int,
        symbol: str,
        db: AsyncSession,
    ) -> Dict[str, Any]:
        """
        Check whether a symbol appears in any of the user's watchlists.

        Raises LookupError if the stock symbol is not recognised.
        Returns a dict with keys: symbol, stock_id, in_watchlists, is_watched.
        """
        stock = await stock_repository.get_by_symbol(symbol, session=db)
        if not stock:
            raise LookupError(f"Stock with symbol '{symbol}' not found")

        watchlists = await watchlist_repository.get_user_watchlists(
            user_id=user_id,
            include_items=True,
            session=db,
        )

        in_watchlists = []
        for watchlist in watchlists:
            is_in = await watchlist_repository.is_stock_in_watchlist(
                watchlist_id=watchlist.id,
                stock_id=stock.id,
                session=db,
            )
            if is_in:
                in_watchlists.append({
                    "watchlist_id": watchlist.id,
                    "watchlist_name": watchlist.name,
                })

        return {
            "symbol": symbol,
            "stock_id": stock.id,
            "in_watchlists": in_watchlists,
            "is_watched": len(in_watchlists) > 0,
        }


# Module-level singleton
watchlist_service = WatchlistService()
