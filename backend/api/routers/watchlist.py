"""
Watchlist API Router - Production-Ready Implementation
Provides user-scoped watchlist management with stock tracking and price alerts.
"""

from fastapi import APIRouter, HTTPException, Depends, status, Path, Query
from typing import List, Optional, Dict, Any
from datetime import datetime
from decimal import Decimal
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError
import logging

from backend.config.database import get_async_db_session
from backend.auth.oauth2 import get_current_user
from backend.models.unified_models import User
from backend.repositories.watchlist_repository import watchlist_repository
from backend.repositories.stock_repository import stock_repository
from backend.models.schemas import (
    WatchlistCreate,
    WatchlistUpdate,
    WatchlistResponse,
    WatchlistSummary,
    WatchlistItemCreate,
    WatchlistItemUpdate,
    WatchlistItemResponse,
)
from backend.models.api_response import ApiResponse, success_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/watchlists", tags=["watchlists"])


# =======================
# Helper Functions
# =======================

async def get_watchlist_or_404(
    watchlist_id: int,
    user_id: int,
    db: AsyncSession,
    require_ownership: bool = True
) -> Any:
    """
    Get a watchlist by ID with ownership verification.

    Args:
        watchlist_id: The watchlist ID
        user_id: The current user's ID
        db: Database session
        require_ownership: If True, verify user owns the watchlist

    Returns:
        Watchlist object if found and authorized

    Raises:
        HTTPException: 404 if not found, 403 if not authorized
    """
    watchlist = await watchlist_repository.get_watchlist_with_items(
        watchlist_id,
        user_id=user_id if require_ownership else None,
        session=db
    )

    if not watchlist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Watchlist with ID {watchlist_id} not found"
        )

    # Additional ownership check for mutations
    if require_ownership and watchlist.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this watchlist"
        )

    return watchlist


def convert_watchlist_to_response(watchlist: Any, items_data: List[Dict] = None) -> WatchlistResponse:
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
        item_count=len(items)
    )


def convert_watchlist_to_summary(watchlist: Any, summary_data: Dict = None) -> WatchlistSummary:
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
        daily_change_percent=summary_data.get('daily_change_percent') if summary_data else None,
        created_at=watchlist.created_at,
        updated_at=watchlist.updated_at,
    )


# =======================
# Watchlist CRUD Endpoints
# =======================

@router.get(
    "",
    summary="Get user's watchlists",
    responses={
        200: {"description": "List of user's watchlists"},
        401: {"description": "Not authenticated"},
    }
)
async def get_user_watchlists(
    include_items: bool = Query(False, description="Include item count in response"),
    db: AsyncSession = Depends(get_async_db_session),
    current_user: User = Depends(get_current_user),
) -> ApiResponse[List[WatchlistSummary]]:
    """
    Get all watchlists for the authenticated user.

    Returns a summary list without full item details for performance.
    Use the individual watchlist endpoint to get full item details.
    """
    try:
        watchlists = await watchlist_repository.get_user_watchlists(
            user_id=current_user.id,
            include_items=include_items,
            session=db
        )

        summaries = []
        for watchlist in watchlists:
            summary_data = None
            if include_items:
                summary_data = await watchlist_repository.get_watchlist_summary(
                    watchlist.id,
                    session=db
                )
            summaries.append(convert_watchlist_to_summary(watchlist, summary_data))

        return success_response(data=summaries)

    except Exception as e:
        logger.error(f"Error fetching watchlists for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving watchlists"
        )


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    summary="Create a new watchlist",
    responses={
        201: {"description": "Watchlist created successfully"},
        400: {"description": "Invalid request or duplicate name"},
        401: {"description": "Not authenticated"},
    }
)
async def create_watchlist(
    watchlist_data: WatchlistCreate,
    db: AsyncSession = Depends(get_async_db_session),
    current_user: User = Depends(get_current_user),
) -> ApiResponse[WatchlistResponse]:
    """
    Create a new watchlist for the authenticated user.

    Each user can have multiple watchlists with unique names.
    """
    try:
        watchlist = await watchlist_repository.create_watchlist(
            user_id=current_user.id,
            name=watchlist_data.name,
            description=watchlist_data.description,
            is_public=watchlist_data.is_public,
            session=db
        )

        logger.info(f"Created watchlist '{watchlist.name}' for user {current_user.id}")
        return success_response(data=convert_watchlist_to_response(watchlist))

    except IntegrityError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Watchlist with name '{watchlist_data.name}' already exists"
        )
    except Exception as e:
        logger.error(f"Error creating watchlist: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error creating watchlist"
        )


@router.get(
    "/default",
    summary="Get default watchlist",
    responses={
        200: {"description": "Default watchlist with items"},
        401: {"description": "Not authenticated"},
    }
)
async def get_default_watchlist(
    db: AsyncSession = Depends(get_async_db_session),
    current_user: User = Depends(get_current_user),
) -> ApiResponse[WatchlistResponse]:
    """
    Get or create the user's default watchlist.

    The default watchlist is named "My Watchlist" and is automatically
    created if it doesn't exist.
    """
    try:
        watchlist = await watchlist_repository.get_default_watchlist(
            user_id=current_user.id,
            session=db
        )

        # Get items with price data
        items_data = await watchlist_repository.get_watchlist_items_with_prices(
            watchlist.id,
            session=db
        )

        return success_response(data=convert_watchlist_to_response(watchlist, items_data))

    except Exception as e:
        logger.error(f"Error getting default watchlist: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving default watchlist"
        )


@router.get(
    "/{watchlist_id}",
    summary="Get watchlist by ID",
    responses={
        200: {"description": "Watchlist with items"},
        401: {"description": "Not authenticated"},
        403: {"description": "Not authorized"},
        404: {"description": "Watchlist not found"},
    }
)
async def get_watchlist(
    watchlist_id: int = Path(..., description="Watchlist ID", gt=0),
    db: AsyncSession = Depends(get_async_db_session),
    current_user: User = Depends(get_current_user),
) -> ApiResponse[WatchlistResponse]:
    """
    Get a specific watchlist with all items and current price data.

    Users can only access their own watchlists or public watchlists.
    """
    try:
        watchlist = await get_watchlist_or_404(
            watchlist_id, current_user.id, db, require_ownership=False
        )

        # Verify access (owner or public)
        if watchlist.user_id != current_user.id and not watchlist.is_public:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this watchlist"
            )

        # Get items with price data
        items_data = await watchlist_repository.get_watchlist_items_with_prices(
            watchlist.id,
            session=db
        )

        return success_response(data=convert_watchlist_to_response(watchlist, items_data))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting watchlist {watchlist_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving watchlist"
        )


@router.put(
    "/{watchlist_id}",
    summary="Update watchlist",
    responses={
        200: {"description": "Watchlist updated successfully"},
        400: {"description": "Invalid request or duplicate name"},
        401: {"description": "Not authenticated"},
        403: {"description": "Not authorized"},
        404: {"description": "Watchlist not found"},
    }
)
async def update_watchlist(
    watchlist_id: int = Path(..., description="Watchlist ID", gt=0),
    watchlist_data: WatchlistUpdate = ...,
    db: AsyncSession = Depends(get_async_db_session),
    current_user: User = Depends(get_current_user),
) -> ApiResponse[WatchlistResponse]:
    """
    Update a watchlist's name, description, or visibility.

    Only the owner can update a watchlist.
    """
    try:
        watchlist = await get_watchlist_or_404(watchlist_id, current_user.id, db)

        # Build update dict with only provided fields
        update_data = {}
        if watchlist_data.name is not None:
            update_data["name"] = watchlist_data.name
        if watchlist_data.description is not None:
            update_data["description"] = watchlist_data.description
        if watchlist_data.is_public is not None:
            update_data["is_public"] = watchlist_data.is_public

        if update_data:
            # Use repository update method if available, otherwise update directly
            for key, value in update_data.items():
                setattr(watchlist, key, value)
            watchlist.updated_at = datetime.utcnow()
            await db.flush()
            await db.refresh(watchlist)

        # Get items with price data
        items_data = await watchlist_repository.get_watchlist_items_with_prices(
            watchlist.id,
            session=db
        )

        logger.info(f"Updated watchlist {watchlist_id} for user {current_user.id}")
        return success_response(data=convert_watchlist_to_response(watchlist, items_data))

    except HTTPException:
        raise
    except IntegrityError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Watchlist with name '{watchlist_data.name}' already exists"
        )
    except Exception as e:
        logger.error(f"Error updating watchlist {watchlist_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error updating watchlist"
        )


@router.delete(
    "/{watchlist_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete watchlist",
    responses={
        204: {"description": "Watchlist deleted successfully"},
        401: {"description": "Not authenticated"},
        403: {"description": "Not authorized"},
        404: {"description": "Watchlist not found"},
    }
)
async def delete_watchlist(
    watchlist_id: int = Path(..., description="Watchlist ID", gt=0),
    db: AsyncSession = Depends(get_async_db_session),
    current_user: User = Depends(get_current_user),
) -> None:
    """
    Delete a watchlist and all its items.

    Only the owner can delete a watchlist.
    This action is irreversible.
    """
    try:
        deleted = await watchlist_repository.delete_watchlist(
            watchlist_id=watchlist_id,
            user_id=current_user.id,
            session=db
        )

        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Watchlist with ID {watchlist_id} not found"
            )

        logger.info(f"Deleted watchlist {watchlist_id} for user {current_user.id}")
        return None

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting watchlist {watchlist_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error deleting watchlist"
        )


# =======================
# Watchlist Item Endpoints
# =======================

@router.post(
    "/{watchlist_id}/items",
    status_code=status.HTTP_201_CREATED,
    summary="Add item to watchlist",
    responses={
        201: {"description": "Item added successfully"},
        400: {"description": "Stock already in watchlist or limit exceeded"},
        401: {"description": "Not authenticated"},
        403: {"description": "Not authorized"},
        404: {"description": "Watchlist or stock not found"},
    }
)
async def add_watchlist_item(
    watchlist_id: int = Path(..., description="Watchlist ID", gt=0),
    item_data: WatchlistItemCreate = ...,
    db: AsyncSession = Depends(get_async_db_session),
    current_user: User = Depends(get_current_user),
) -> ApiResponse[WatchlistItemResponse]:
    """
    Add a stock to a watchlist.

    Users can add stocks with optional target price and notes.
    Enable alerts to get notified when the target price is reached.
    """
    try:
        # Verify watchlist ownership
        watchlist = await get_watchlist_or_404(watchlist_id, current_user.id, db)

        # Look up stock by symbol
        stock = await stock_repository.get_by_symbol(item_data.symbol, session=db)
        if not stock:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Stock with symbol '{item_data.symbol}' not found"
            )

        # Add item to watchlist
        target_price = Decimal(str(item_data.target_price)) if item_data.target_price else None
        item = await watchlist_repository.add_item_to_watchlist(
            watchlist_id=watchlist_id,
            stock_id=stock.id,
            target_price=target_price,
            notes=item_data.notes,
            alert_enabled=item_data.alert_enabled,
            session=db
        )

        logger.info(f"Added {item_data.symbol} to watchlist {watchlist_id}")

        return success_response(data=WatchlistItemResponse(
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
        ))

    except HTTPException:
        raise
    except IntegrityError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Stock '{item_data.symbol}' is already in this watchlist"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error adding item to watchlist {watchlist_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error adding item to watchlist"
        )


@router.put(
    "/{watchlist_id}/items/{item_id}",
    summary="Update watchlist item",
    responses={
        200: {"description": "Item updated successfully"},
        401: {"description": "Not authenticated"},
        403: {"description": "Not authorized"},
        404: {"description": "Watchlist or item not found"},
    }
)
async def update_watchlist_item(
    watchlist_id: int = Path(..., description="Watchlist ID", gt=0),
    item_id: int = Path(..., description="Item ID", gt=0),
    item_data: WatchlistItemUpdate = ...,
    db: AsyncSession = Depends(get_async_db_session),
    current_user: User = Depends(get_current_user),
) -> ApiResponse[WatchlistItemResponse]:
    """
    Update a watchlist item's target price, notes, or alert status.

    Set target_price to 0 to clear the target price.
    """
    try:
        # Verify watchlist ownership
        watchlist = await get_watchlist_or_404(watchlist_id, current_user.id, db)

        # Convert target_price to Decimal if provided
        target_price = None
        if item_data.target_price is not None:
            target_price = Decimal(str(item_data.target_price))

        # Update the item
        updated_item = await watchlist_repository.update_item(
            item_id=item_id,
            target_price=target_price,
            notes=item_data.notes,
            alert_enabled=item_data.alert_enabled,
            session=db
        )

        if not updated_item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Item with ID {item_id} not found"
            )

        # Verify item belongs to the watchlist
        if updated_item.watchlist_id != watchlist_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Item with ID {item_id} not found in this watchlist"
            )

        # Get stock details
        stock = await stock_repository.get_by_id(updated_item.stock_id, session=db)

        logger.info(f"Updated item {item_id} in watchlist {watchlist_id}")

        return success_response(data=WatchlistItemResponse(
            id=updated_item.id,
            watchlist_id=updated_item.watchlist_id,
            stock_id=updated_item.stock_id,
            target_price=float(updated_item.target_price) if updated_item.target_price else None,
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
        ))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating item {item_id} in watchlist {watchlist_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error updating watchlist item"
        )


@router.delete(
    "/{watchlist_id}/items/{item_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove item from watchlist",
    responses={
        204: {"description": "Item removed successfully"},
        401: {"description": "Not authenticated"},
        403: {"description": "Not authorized"},
        404: {"description": "Watchlist or item not found"},
    }
)
async def remove_watchlist_item(
    watchlist_id: int = Path(..., description="Watchlist ID", gt=0),
    item_id: int = Path(..., description="Item ID", gt=0),
    db: AsyncSession = Depends(get_async_db_session),
    current_user: User = Depends(get_current_user),
) -> None:
    """
    Remove an item from a watchlist.

    This action is irreversible.
    """
    try:
        # Verify watchlist ownership
        watchlist = await get_watchlist_or_404(watchlist_id, current_user.id, db)

        # Find the item first to get the stock_id
        from sqlalchemy import select
        from backend.models.tables import WatchlistItem

        query = select(WatchlistItem).where(
            WatchlistItem.id == item_id,
            WatchlistItem.watchlist_id == watchlist_id
        )
        result = await db.execute(query)
        item = result.scalar_one_or_none()

        if not item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Item with ID {item_id} not found in this watchlist"
            )

        # Remove the item
        removed = await watchlist_repository.remove_item_from_watchlist(
            watchlist_id=watchlist_id,
            stock_id=item.stock_id,
            session=db
        )

        if not removed:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Item with ID {item_id} not found"
            )

        logger.info(f"Removed item {item_id} from watchlist {watchlist_id}")
        return None

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing item {item_id} from watchlist {watchlist_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error removing item from watchlist"
        )


# =======================
# Convenience Endpoints
# =======================

@router.post(
    "/default/symbols/{symbol}",
    status_code=status.HTTP_201_CREATED,
    summary="Add symbol to default watchlist",
    responses={
        201: {"description": "Symbol added successfully"},
        400: {"description": "Stock already in watchlist or limit exceeded"},
        401: {"description": "Not authenticated"},
        404: {"description": "Stock not found"},
    }
)
async def add_to_default_watchlist(
    symbol: str = Path(..., description="Stock symbol", min_length=1, max_length=10),
    db: AsyncSession = Depends(get_async_db_session),
    current_user: User = Depends(get_current_user),
) -> ApiResponse[WatchlistItemResponse]:
    """
    Quick add a stock to the user's default watchlist.

    Creates the default watchlist if it doesn't exist.
    This is a convenience endpoint for quickly adding stocks from search results.
    """
    try:
        symbol = symbol.upper()

        # Get or create default watchlist
        watchlist = await watchlist_repository.get_default_watchlist(
            user_id=current_user.id,
            session=db
        )

        # Look up stock by symbol
        stock = await stock_repository.get_by_symbol(symbol, session=db)
        if not stock:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Stock with symbol '{symbol}' not found"
            )

        # Add item to watchlist
        item = await watchlist_repository.add_item_to_watchlist(
            watchlist_id=watchlist.id,
            stock_id=stock.id,
            session=db
        )

        logger.info(f"Added {symbol} to default watchlist for user {current_user.id}")

        return success_response(data=WatchlistItemResponse(
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
        ))

    except HTTPException:
        raise
    except IntegrityError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Stock '{symbol}' is already in your default watchlist"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error adding {symbol} to default watchlist: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error adding to default watchlist"
        )


@router.delete(
    "/default/symbols/{symbol}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove symbol from default watchlist",
    responses={
        204: {"description": "Symbol removed successfully"},
        401: {"description": "Not authenticated"},
        404: {"description": "Stock not found or not in watchlist"},
    }
)
async def remove_from_default_watchlist(
    symbol: str = Path(..., description="Stock symbol", min_length=1, max_length=10),
    db: AsyncSession = Depends(get_async_db_session),
    current_user: User = Depends(get_current_user),
) -> None:
    """
    Quick remove a stock from the user's default watchlist.

    This is a convenience endpoint for quickly removing stocks from the watchlist.
    """
    try:
        symbol = symbol.upper()

        # Get default watchlist
        watchlist = await watchlist_repository.get_default_watchlist(
            user_id=current_user.id,
            session=db
        )

        # Look up stock by symbol
        stock = await stock_repository.get_by_symbol(symbol, session=db)
        if not stock:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Stock with symbol '{symbol}' not found"
            )

        # Check if stock is in watchlist
        in_watchlist = await watchlist_repository.is_stock_in_watchlist(
            watchlist_id=watchlist.id,
            stock_id=stock.id,
            session=db
        )

        if not in_watchlist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Stock '{symbol}' is not in your default watchlist"
            )

        # Remove the stock
        await watchlist_repository.remove_item_from_watchlist(
            watchlist_id=watchlist.id,
            stock_id=stock.id,
            session=db
        )

        logger.info(f"Removed {symbol} from default watchlist for user {current_user.id}")
        return None

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing {symbol} from default watchlist: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error removing from default watchlist"
        )


@router.get(
    "/check/{symbol}",
    summary="Check if symbol is in any watchlist",
    responses={
        200: {"description": "Watchlist status for symbol"},
        401: {"description": "Not authenticated"},
        404: {"description": "Stock not found"},
    }
)
async def check_symbol_in_watchlists(
    symbol: str = Path(..., description="Stock symbol", min_length=1, max_length=10),
    db: AsyncSession = Depends(get_async_db_session),
    current_user: User = Depends(get_current_user),
) -> ApiResponse[Dict]:
    """
    Check if a stock symbol is in any of the user's watchlists.

    Returns information about which watchlists contain the symbol.
    """
    try:
        symbol = symbol.upper()

        # Look up stock by symbol
        stock = await stock_repository.get_by_symbol(symbol, session=db)
        if not stock:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Stock with symbol '{symbol}' not found"
            )

        # Get all user's watchlists
        watchlists = await watchlist_repository.get_user_watchlists(
            user_id=current_user.id,
            include_items=True,
            session=db
        )

        # Check each watchlist
        in_watchlists = []
        for watchlist in watchlists:
            is_in = await watchlist_repository.is_stock_in_watchlist(
                watchlist_id=watchlist.id,
                stock_id=stock.id,
                session=db
            )
            if is_in:
                in_watchlists.append({
                    "watchlist_id": watchlist.id,
                    "watchlist_name": watchlist.name
                })

        return success_response(data={
            "symbol": symbol,
            "stock_id": stock.id,
            "in_watchlists": in_watchlists,
            "is_watched": len(in_watchlists) > 0
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking watchlist status for {symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error checking watchlist status"
        )
