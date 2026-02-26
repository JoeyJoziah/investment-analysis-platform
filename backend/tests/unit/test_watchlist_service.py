"""
Unit tests for backend/services/watchlist_service.py

Tests all public methods of WatchlistService and conversion helpers with
mocked dependencies.  No database or external services required.
"""

import sys
import pytest
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from backend.services.watchlist_service import (
    WatchlistService,
    watchlist_service,
    convert_watchlist_to_response,
    convert_watchlist_to_summary,
)
from backend.models.schemas import (
    WatchlistItemResponse,
    WatchlistResponse,
    WatchlistSummary,
)

# Grab the real module so patch.object works correctly
_ws_mod = sys.modules["backend.services.watchlist_service"]


# ---------------------------------------------------------------------------
# Helpers -- lightweight stand-ins for ORM objects
# ---------------------------------------------------------------------------

def _ts():
    """Return a fixed UTC timestamp for deterministic tests."""
    return datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)


def _make_watchlist(
    *,
    id=1,
    user_id=100,
    name="Tech Stocks",
    description="My tech watchlist",
    is_public=False,
    items=None,
    created_at=None,
    updated_at=None,
):
    """Return a namespace that quacks like a Watchlist ORM object."""
    return SimpleNamespace(
        id=id,
        user_id=user_id,
        name=name,
        description=description,
        is_public=is_public,
        items=items or [],
        created_at=created_at or _ts(),
        updated_at=updated_at or _ts(),
    )


def _make_watchlist_item(
    *,
    id=10,
    watchlist_id=1,
    stock_id=200,
    target_price=None,
    notes=None,
    alert_enabled=False,
    added_at=None,
    stock=None,
):
    """Return a namespace that quacks like a WatchlistItem ORM object."""
    return SimpleNamespace(
        id=id,
        watchlist_id=watchlist_id,
        stock_id=stock_id,
        target_price=target_price,
        notes=notes,
        alert_enabled=alert_enabled,
        added_at=added_at or _ts(),
        stock=stock,
    )


def _make_stock(
    *,
    id=200,
    symbol="AAPL",
    name="Apple Inc.",
    sector="Technology",
    market_cap=3000000000000,
):
    """Return a namespace that quacks like a Stock ORM object."""
    return SimpleNamespace(
        id=id,
        symbol=symbol,
        name=name,
        sector=sector,
        market_cap=market_cap,
    )


def _make_items_data(count=2):
    """Return a list of dicts resembling get_watchlist_items_with_prices output."""
    items = []
    for i in range(count):
        items.append({
            "id": 10 + i,
            "watchlist_id": 1,
            "stock_id": 200 + i,
            "target_price": 150.0 + i,
            "notes": f"Note {i}",
            "alert_enabled": i % 2 == 0,
            "added_at": _ts(),
            "symbol": f"SYM{i}",
            "company_name": f"Company {i}",
            "current_price": 155.0 + i,
            "price_change": 1.5 + i,
            "price_change_percent": 1.0 + i,
            "volume": 1_000_000 * (i + 1),
            "market_cap": 100_000_000_000 * (i + 1),
            "sector": "Technology",
        })
    return items


# ---------------------------------------------------------------------------
# Fixture: fresh WatchlistService instance
# ---------------------------------------------------------------------------

@pytest.fixture
def service():
    return WatchlistService()


@pytest.fixture
def mock_db():
    """Return an AsyncMock standing in for AsyncSession."""
    db = AsyncMock()
    db.flush = AsyncMock()
    db.refresh = AsyncMock()
    db.execute = AsyncMock()
    return db


# =========================================================================
# convert_watchlist_to_response  (free function)
# =========================================================================

class TestConvertWatchlistToResponse:

    def test_with_items_data(self):
        """When items_data is provided, use pre-fetched item dicts."""
        watchlist = _make_watchlist()
        items_data = _make_items_data(2)

        result = convert_watchlist_to_response(watchlist, items_data)

        assert isinstance(result, WatchlistResponse)
        assert result.id == watchlist.id
        assert result.user_id == watchlist.user_id
        assert result.name == watchlist.name
        assert len(result.items) == 2
        assert result.item_count == 2

    def test_with_items_data_fields_populated(self):
        """Verify all fields from items_data dicts propagate correctly."""
        watchlist = _make_watchlist()
        items_data = _make_items_data(1)

        result = convert_watchlist_to_response(watchlist, items_data)
        item = result.items[0]

        assert item.symbol == "SYM0"
        assert item.company_name == "Company 0"
        assert item.current_price == 155.0
        assert item.price_change == 1.5
        assert item.price_change_percent == 1.0
        assert item.volume == 1_000_000
        assert item.sector == "Technology"

    def test_with_empty_items_data(self):
        """Empty items_data list produces no items."""
        watchlist = _make_watchlist()
        result = convert_watchlist_to_response(watchlist, [])
        assert result.items == []
        assert result.item_count == 0

    def test_with_relationship_items_and_stock(self):
        """When no items_data, fall back to watchlist.items relationship."""
        stock = _make_stock(symbol="TSLA", name="Tesla Inc.", sector="Automotive",
                            market_cap=800_000_000_000)
        item = _make_watchlist_item(
            stock=stock,
            target_price=Decimal("250.00"),
            notes="growth play",
            alert_enabled=True,
        )
        watchlist = _make_watchlist(items=[item])

        result = convert_watchlist_to_response(watchlist)

        assert len(result.items) == 1
        assert result.items[0].symbol == "TSLA"
        assert result.items[0].company_name == "Tesla Inc."
        assert result.items[0].target_price == 250.0
        assert result.items[0].notes == "growth play"
        assert result.items[0].alert_enabled is True
        assert result.items[0].market_cap == 800_000_000_000
        assert result.items[0].sector == "Automotive"

    def test_relationship_item_without_stock(self):
        """When item.stock is None, symbol falls back to 'UNKNOWN'."""
        item = _make_watchlist_item(stock=None)
        watchlist = _make_watchlist(items=[item])

        result = convert_watchlist_to_response(watchlist)

        assert result.items[0].symbol == "UNKNOWN"
        assert result.items[0].company_name is None

    def test_relationship_item_no_target_price(self):
        """When target_price is None, it stays None in the response."""
        item = _make_watchlist_item(target_price=None, stock=_make_stock())
        watchlist = _make_watchlist(items=[item])
        result = convert_watchlist_to_response(watchlist)
        assert result.items[0].target_price is None

    def test_no_items_data_and_no_relationship_items(self):
        """Watchlist with no items at all produces empty list."""
        watchlist = _make_watchlist(items=[])
        result = convert_watchlist_to_response(watchlist, items_data=None)
        assert result.items == []
        assert result.item_count == 0

    def test_watchlist_fields_passthrough(self):
        """All top-level watchlist fields are passed through."""
        watchlist = _make_watchlist(
            id=42, user_id=7, name="My List",
            description="desc", is_public=True,
        )
        result = convert_watchlist_to_response(watchlist)
        assert result.id == 42
        assert result.user_id == 7
        assert result.name == "My List"
        assert result.description == "desc"
        assert result.is_public is True

    def test_items_data_with_missing_optional_fields(self):
        """Optional fields missing from items_data dict default gracefully."""
        watchlist = _make_watchlist()
        items_data = [{
            "id": 1,
            "watchlist_id": 1,
            "stock_id": 200,
            "added_at": _ts(),
            "symbol": "XYZ",
        }]
        result = convert_watchlist_to_response(watchlist, items_data)
        item = result.items[0]
        assert item.target_price is None
        assert item.notes is None
        assert item.alert_enabled is False
        assert item.current_price is None
        assert item.volume is None


# =========================================================================
# convert_watchlist_to_summary  (free function)
# =========================================================================

class TestConvertWatchlistToSummary:

    def test_with_summary_data(self):
        """When summary_data is provided, use its values."""
        watchlist = _make_watchlist()
        summary_data = {
            "item_count": 5,
            "total_value": 1_000_000.0,
            "daily_change_percent": 2.5,
        }
        result = convert_watchlist_to_summary(watchlist, summary_data)

        assert isinstance(result, WatchlistSummary)
        assert result.item_count == 5
        assert result.total_value == 1_000_000.0
        assert result.daily_change_percent == 2.5

    def test_item_count_from_relationship(self):
        """When no summary_data, count from watchlist.items."""
        items = [_make_watchlist_item(id=i) for i in range(3)]
        watchlist = _make_watchlist(items=items)

        result = convert_watchlist_to_summary(watchlist)
        assert result.item_count == 3

    def test_no_items_and_no_summary_data(self):
        """No items and no summary_data yields item_count=0."""
        watchlist = _make_watchlist(items=[])
        result = convert_watchlist_to_summary(watchlist)
        assert result.item_count == 0
        assert result.total_value is None
        assert result.daily_change_percent is None

    def test_summary_fields_passthrough(self):
        """Name, description, timestamps pass through."""
        watchlist = _make_watchlist(name="Growth", description="high growth")
        result = convert_watchlist_to_summary(watchlist)
        assert result.name == "Growth"
        assert result.description == "high growth"
        assert result.created_at == _ts()

    def test_summary_data_missing_optional_keys(self):
        """summary_data with only item_count; total_value defaults to None."""
        watchlist = _make_watchlist()
        summary_data = {"item_count": 2}
        result = convert_watchlist_to_summary(watchlist, summary_data)
        assert result.item_count == 2
        assert result.total_value is None
        assert result.daily_change_percent is None


# =========================================================================
# get_watchlist_with_access_check
# =========================================================================

class TestGetWatchlistWithAccessCheck:

    @pytest.mark.asyncio
    async def test_returns_watchlist_when_owner(self, service, mock_db):
        """Owner should get the watchlist back."""
        watchlist = _make_watchlist(user_id=100)
        with patch.object(
            _ws_mod, "watchlist_repository"
        ) as mock_repo:
            mock_repo.get_watchlist_with_items = AsyncMock(return_value=watchlist)

            result = await service.get_watchlist_with_access_check(
                watchlist_id=1, user_id=100, db=mock_db, require_ownership=True,
            )
        assert result == watchlist

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self, service, mock_db):
        """Non-existent watchlist returns None."""
        with patch.object(
            _ws_mod, "watchlist_repository"
        ) as mock_repo:
            mock_repo.get_watchlist_with_items = AsyncMock(return_value=None)

            result = await service.get_watchlist_with_access_check(
                watchlist_id=999, user_id=100, db=mock_db,
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_raises_permission_error_for_non_owner(self, service, mock_db):
        """Non-owner with require_ownership=True raises PermissionError."""
        watchlist = _make_watchlist(user_id=999)
        with patch.object(
            _ws_mod, "watchlist_repository"
        ) as mock_repo:
            mock_repo.get_watchlist_with_items = AsyncMock(return_value=watchlist)

            with pytest.raises(PermissionError, match="Not authorized"):
                await service.get_watchlist_with_access_check(
                    watchlist_id=1, user_id=100, db=mock_db, require_ownership=True,
                )

    @pytest.mark.asyncio
    async def test_skips_ownership_check_when_not_required(self, service, mock_db):
        """require_ownership=False skips the ownership assertion."""
        watchlist = _make_watchlist(user_id=999)
        with patch.object(
            _ws_mod, "watchlist_repository"
        ) as mock_repo:
            mock_repo.get_watchlist_with_items = AsyncMock(return_value=watchlist)

            result = await service.get_watchlist_with_access_check(
                watchlist_id=1, user_id=100, db=mock_db, require_ownership=False,
            )
        assert result == watchlist

    @pytest.mark.asyncio
    async def test_passes_user_id_to_repo_when_require_ownership(self, service, mock_db):
        """When require_ownership=True, user_id is forwarded to the repository."""
        watchlist = _make_watchlist(user_id=100)
        with patch.object(
            _ws_mod, "watchlist_repository"
        ) as mock_repo:
            mock_repo.get_watchlist_with_items = AsyncMock(return_value=watchlist)

            await service.get_watchlist_with_access_check(
                watchlist_id=1, user_id=100, db=mock_db, require_ownership=True,
            )

        mock_repo.get_watchlist_with_items.assert_called_once_with(
            1, user_id=100, session=mock_db,
        )

    @pytest.mark.asyncio
    async def test_passes_none_user_id_when_not_require_ownership(self, service, mock_db):
        """When require_ownership=False, user_id=None is forwarded to repo."""
        watchlist = _make_watchlist(user_id=100)
        with patch.object(
            _ws_mod, "watchlist_repository"
        ) as mock_repo:
            mock_repo.get_watchlist_with_items = AsyncMock(return_value=watchlist)

            await service.get_watchlist_with_access_check(
                watchlist_id=1, user_id=100, db=mock_db, require_ownership=False,
            )

        mock_repo.get_watchlist_with_items.assert_called_once_with(
            1, user_id=None, session=mock_db,
        )


# =========================================================================
# list_user_watchlists
# =========================================================================

class TestListUserWatchlists:

    @pytest.mark.asyncio
    async def test_returns_summaries(self, service, mock_db):
        """Should return WatchlistSummary list for the user."""
        wl1 = _make_watchlist(id=1, name="A")
        wl2 = _make_watchlist(id=2, name="B")

        with patch.object(_ws_mod, "watchlist_repository") as mock_repo:
            mock_repo.get_user_watchlists = AsyncMock(return_value=[wl1, wl2])
            mock_repo.get_watchlist_summary = AsyncMock(return_value=None)

            result = await service.list_user_watchlists(
                user_id=100, include_items=False, db=mock_db,
            )

        assert len(result) == 2
        assert all(isinstance(s, WatchlistSummary) for s in result)

    @pytest.mark.asyncio
    async def test_include_items_fetches_summaries(self, service, mock_db):
        """When include_items=True, summary data is fetched per watchlist."""
        wl = _make_watchlist(id=1)
        summary = {"item_count": 3, "total_value": 500.0, "daily_change_percent": 1.0}

        with patch.object(_ws_mod, "watchlist_repository") as mock_repo:
            mock_repo.get_user_watchlists = AsyncMock(return_value=[wl])
            mock_repo.get_watchlist_summary = AsyncMock(return_value=summary)

            result = await service.list_user_watchlists(
                user_id=100, include_items=True, db=mock_db,
            )

        assert result[0].item_count == 3
        assert result[0].total_value == 500.0
        mock_repo.get_watchlist_summary.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_watchlists(self, service, mock_db):
        """User with no watchlists gets empty list."""
        with patch.object(_ws_mod, "watchlist_repository") as mock_repo:
            mock_repo.get_user_watchlists = AsyncMock(return_value=[])

            result = await service.list_user_watchlists(
                user_id=100, include_items=False, db=mock_db,
            )
        assert result == []

    @pytest.mark.asyncio
    async def test_include_items_false_skips_summary_fetch(self, service, mock_db):
        """When include_items=False, get_watchlist_summary is never called."""
        wl = _make_watchlist(id=1)
        with patch.object(_ws_mod, "watchlist_repository") as mock_repo:
            mock_repo.get_user_watchlists = AsyncMock(return_value=[wl])
            mock_repo.get_watchlist_summary = AsyncMock()

            await service.list_user_watchlists(
                user_id=100, include_items=False, db=mock_db,
            )

        mock_repo.get_watchlist_summary.assert_not_called()


# =========================================================================
# get_default_watchlist_response
# =========================================================================

class TestGetDefaultWatchlistResponse:

    @pytest.mark.asyncio
    async def test_returns_response_with_items(self, service, mock_db):
        """Auto-created or existing default watchlist is returned with items."""
        wl = _make_watchlist(id=5, name="My Watchlist")
        items_data = _make_items_data(2)

        with patch.object(_ws_mod, "watchlist_repository") as mock_repo:
            mock_repo.get_default_watchlist = AsyncMock(return_value=wl)
            mock_repo.get_watchlist_items_with_prices = AsyncMock(return_value=items_data)

            result = await service.get_default_watchlist_response(
                user_id=100, db=mock_db,
            )

        assert isinstance(result, WatchlistResponse)
        assert result.id == 5
        assert len(result.items) == 2

    @pytest.mark.asyncio
    async def test_empty_watchlist(self, service, mock_db):
        """Default watchlist with no items returns empty items list."""
        wl = _make_watchlist(id=5)

        with patch.object(_ws_mod, "watchlist_repository") as mock_repo:
            mock_repo.get_default_watchlist = AsyncMock(return_value=wl)
            mock_repo.get_watchlist_items_with_prices = AsyncMock(return_value=[])

            result = await service.get_default_watchlist_response(
                user_id=100, db=mock_db,
            )

        assert result.items == []
        assert result.item_count == 0


# =========================================================================
# get_watchlist_response
# =========================================================================

class TestGetWatchlistResponse:

    @pytest.mark.asyncio
    async def test_owner_gets_private_watchlist(self, service, mock_db):
        """Owner can access their own private watchlist."""
        wl = _make_watchlist(id=1, user_id=100, is_public=False)
        items_data = _make_items_data(1)

        with patch.object(_ws_mod, "watchlist_repository") as mock_repo:
            mock_repo.get_watchlist_with_items = AsyncMock(return_value=wl)
            mock_repo.get_watchlist_items_with_prices = AsyncMock(return_value=items_data)

            result = await service.get_watchlist_response(
                watchlist_id=1, user_id=100, db=mock_db,
            )

        assert isinstance(result, WatchlistResponse)
        assert result.id == 1

    @pytest.mark.asyncio
    async def test_other_user_gets_public_watchlist(self, service, mock_db):
        """Any user can access a public watchlist."""
        wl = _make_watchlist(id=1, user_id=999, is_public=True)
        items_data = _make_items_data(1)

        with patch.object(_ws_mod, "watchlist_repository") as mock_repo:
            mock_repo.get_watchlist_with_items = AsyncMock(return_value=wl)
            mock_repo.get_watchlist_items_with_prices = AsyncMock(return_value=items_data)

            result = await service.get_watchlist_response(
                watchlist_id=1, user_id=100, db=mock_db,
            )

        assert result is not None

    @pytest.mark.asyncio
    async def test_raises_for_private_watchlist_of_other_user(self, service, mock_db):
        """Non-owner accessing private watchlist raises PermissionError."""
        wl = _make_watchlist(id=1, user_id=999, is_public=False)

        with patch.object(_ws_mod, "watchlist_repository") as mock_repo:
            mock_repo.get_watchlist_with_items = AsyncMock(return_value=wl)

            with pytest.raises(PermissionError, match="Not authorized"):
                await service.get_watchlist_response(
                    watchlist_id=1, user_id=100, db=mock_db,
                )

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self, service, mock_db):
        """Non-existent watchlist returns None."""
        with patch.object(_ws_mod, "watchlist_repository") as mock_repo:
            mock_repo.get_watchlist_with_items = AsyncMock(return_value=None)

            result = await service.get_watchlist_response(
                watchlist_id=999, user_id=100, db=mock_db,
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_passes_user_id_none_to_repo(self, service, mock_db):
        """get_watchlist_response passes user_id=None to repo (access check is local)."""
        wl = _make_watchlist(id=1, user_id=100)
        with patch.object(_ws_mod, "watchlist_repository") as mock_repo:
            mock_repo.get_watchlist_with_items = AsyncMock(return_value=wl)
            mock_repo.get_watchlist_items_with_prices = AsyncMock(return_value=[])

            await service.get_watchlist_response(
                watchlist_id=1, user_id=100, db=mock_db,
            )

        mock_repo.get_watchlist_with_items.assert_called_once_with(
            1, user_id=None, session=mock_db,
        )


# =========================================================================
# create_watchlist
# =========================================================================

class TestCreateWatchlist:

    @pytest.mark.asyncio
    async def test_creates_and_returns_response(self, service, mock_db):
        """New watchlist is created via repository and returned as response."""
        wl = _make_watchlist(id=10, user_id=100, name="New List", is_public=True)

        with patch.object(_ws_mod, "watchlist_repository") as mock_repo:
            mock_repo.create_watchlist = AsyncMock(return_value=wl)

            result = await service.create_watchlist(
                user_id=100, name="New List",
                description="desc", is_public=True, db=mock_db,
            )

        assert isinstance(result, WatchlistResponse)
        assert result.id == 10
        assert result.name == "New List"
        assert result.is_public is True
        assert result.items == []

    @pytest.mark.asyncio
    async def test_passes_correct_args_to_repo(self, service, mock_db):
        """Arguments are correctly forwarded to repository."""
        wl = _make_watchlist()
        with patch.object(_ws_mod, "watchlist_repository") as mock_repo:
            mock_repo.create_watchlist = AsyncMock(return_value=wl)

            await service.create_watchlist(
                user_id=42, name="Test", description="hello", is_public=False,
                db=mock_db,
            )

        mock_repo.create_watchlist.assert_called_once_with(
            user_id=42, name="Test", description="hello",
            is_public=False, session=mock_db,
        )


# =========================================================================
# update_watchlist
# =========================================================================

class TestUpdateWatchlist:

    @pytest.mark.asyncio
    async def test_updates_name(self, service, mock_db):
        """Updating name sets the attribute and flushes."""
        wl = _make_watchlist(id=1, user_id=100, name="Old Name")

        with patch.object(_ws_mod, "watchlist_repository") as mock_repo:
            mock_repo.get_watchlist_with_items = AsyncMock(return_value=wl)
            mock_repo.get_watchlist_items_with_prices = AsyncMock(return_value=[])

            result = await service.update_watchlist(
                watchlist_id=1, user_id=100, name="New Name",
                description=None, is_public=None, db=mock_db,
            )

        assert result is not None
        assert wl.name == "New Name"
        mock_db.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_updates_description(self, service, mock_db):
        """Updating description sets the attribute."""
        wl = _make_watchlist(id=1, user_id=100)

        with patch.object(_ws_mod, "watchlist_repository") as mock_repo:
            mock_repo.get_watchlist_with_items = AsyncMock(return_value=wl)
            mock_repo.get_watchlist_items_with_prices = AsyncMock(return_value=[])

            await service.update_watchlist(
                watchlist_id=1, user_id=100, name=None,
                description="updated desc", is_public=None, db=mock_db,
            )

        assert wl.description == "updated desc"

    @pytest.mark.asyncio
    async def test_updates_is_public(self, service, mock_db):
        """Updating is_public toggles the flag."""
        wl = _make_watchlist(id=1, user_id=100, is_public=False)

        with patch.object(_ws_mod, "watchlist_repository") as mock_repo:
            mock_repo.get_watchlist_with_items = AsyncMock(return_value=wl)
            mock_repo.get_watchlist_items_with_prices = AsyncMock(return_value=[])

            await service.update_watchlist(
                watchlist_id=1, user_id=100, name=None,
                description=None, is_public=True, db=mock_db,
            )

        assert wl.is_public is True

    @pytest.mark.asyncio
    async def test_no_changes_skips_flush(self, service, mock_db):
        """When all update params are None, flush is not called."""
        wl = _make_watchlist(id=1, user_id=100)

        with patch.object(_ws_mod, "watchlist_repository") as mock_repo:
            mock_repo.get_watchlist_with_items = AsyncMock(return_value=wl)
            mock_repo.get_watchlist_items_with_prices = AsyncMock(return_value=[])

            result = await service.update_watchlist(
                watchlist_id=1, user_id=100, name=None,
                description=None, is_public=None, db=mock_db,
            )

        assert result is not None
        mock_db.flush.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self, service, mock_db):
        """Non-existent watchlist returns None."""
        with patch.object(_ws_mod, "watchlist_repository") as mock_repo:
            mock_repo.get_watchlist_with_items = AsyncMock(return_value=None)

            result = await service.update_watchlist(
                watchlist_id=999, user_id=100, name="X",
                description=None, is_public=None, db=mock_db,
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_sets_updated_at_on_change(self, service, mock_db):
        """updated_at should be refreshed when there are changes."""
        old_ts = _ts()
        wl = _make_watchlist(id=1, user_id=100, updated_at=old_ts)

        with patch.object(_ws_mod, "watchlist_repository") as mock_repo:
            mock_repo.get_watchlist_with_items = AsyncMock(return_value=wl)
            mock_repo.get_watchlist_items_with_prices = AsyncMock(return_value=[])

            await service.update_watchlist(
                watchlist_id=1, user_id=100, name="Changed",
                description=None, is_public=None, db=mock_db,
            )

        assert wl.updated_at != old_ts

    @pytest.mark.asyncio
    async def test_returns_response_with_items(self, service, mock_db):
        """Updated response includes items from price lookup."""
        wl = _make_watchlist(id=1, user_id=100)
        items_data = _make_items_data(3)

        with patch.object(_ws_mod, "watchlist_repository") as mock_repo:
            mock_repo.get_watchlist_with_items = AsyncMock(return_value=wl)
            mock_repo.get_watchlist_items_with_prices = AsyncMock(return_value=items_data)

            result = await service.update_watchlist(
                watchlist_id=1, user_id=100, name="Changed",
                description=None, is_public=None, db=mock_db,
            )

        assert len(result.items) == 3


# =========================================================================
# delete_watchlist
# =========================================================================

class TestDeleteWatchlist:

    @pytest.mark.asyncio
    async def test_delete_returns_true_on_success(self, service, mock_db):
        """Successful deletion returns True."""
        with patch.object(_ws_mod, "watchlist_repository") as mock_repo:
            mock_repo.delete_watchlist = AsyncMock(return_value=True)

            result = await service.delete_watchlist(
                watchlist_id=1, user_id=100, db=mock_db,
            )
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_returns_false_when_not_found(self, service, mock_db):
        """Not-found watchlist returns False."""
        with patch.object(_ws_mod, "watchlist_repository") as mock_repo:
            mock_repo.delete_watchlist = AsyncMock(return_value=False)

            result = await service.delete_watchlist(
                watchlist_id=999, user_id=100, db=mock_db,
            )
        assert result is False

    @pytest.mark.asyncio
    async def test_passes_correct_args_to_repo(self, service, mock_db):
        """Arguments are forwarded to repository.delete_watchlist."""
        with patch.object(_ws_mod, "watchlist_repository") as mock_repo:
            mock_repo.delete_watchlist = AsyncMock(return_value=True)

            await service.delete_watchlist(
                watchlist_id=7, user_id=42, db=mock_db,
            )

        mock_repo.delete_watchlist.assert_called_once_with(
            watchlist_id=7, user_id=42, session=mock_db,
        )


# =========================================================================
# add_item
# =========================================================================

class TestAddItem:

    @pytest.mark.asyncio
    async def test_adds_stock_by_symbol(self, service, mock_db):
        """Stock looked up by symbol, item added with correct fields."""
        stock = _make_stock(id=200, symbol="AAPL", name="Apple Inc.")
        item = _make_watchlist_item(
            id=50, watchlist_id=1, stock_id=200,
            target_price=Decimal("175.00"), notes="buy dip",
            alert_enabled=True,
        )

        with patch.object(_ws_mod, "stock_repository") as mock_sr, \
             patch.object(_ws_mod, "watchlist_repository") as mock_wr:
            mock_sr.get_by_symbol = AsyncMock(return_value=stock)
            mock_wr.add_item_to_watchlist = AsyncMock(return_value=item)

            result = await service.add_item(
                watchlist_id=1, user_id=100, symbol="AAPL",
                target_price=175.0, notes="buy dip",
                alert_enabled=True, db=mock_db,
            )

        assert isinstance(result, WatchlistItemResponse)
        assert result.id == 50
        assert result.symbol == "AAPL"
        assert result.company_name == "Apple Inc."
        assert result.target_price == 175.0
        assert result.notes == "buy dip"
        assert result.alert_enabled is True

    @pytest.mark.asyncio
    async def test_raises_lookup_error_for_unknown_symbol(self, service, mock_db):
        """Unknown stock symbol raises LookupError."""
        with patch.object(_ws_mod, "stock_repository") as mock_sr:
            mock_sr.get_by_symbol = AsyncMock(return_value=None)

            with pytest.raises(LookupError, match="not found"):
                await service.add_item(
                    watchlist_id=1, user_id=100, symbol="ZZZZ",
                    target_price=None, notes=None,
                    alert_enabled=False, db=mock_db,
                )

    @pytest.mark.asyncio
    async def test_converts_target_price_to_decimal(self, service, mock_db):
        """Float target_price is converted to Decimal before passing to repo."""
        stock = _make_stock()
        item = _make_watchlist_item(target_price=Decimal("123.45"))

        with patch.object(_ws_mod, "stock_repository") as mock_sr, \
             patch.object(_ws_mod, "watchlist_repository") as mock_wr:
            mock_sr.get_by_symbol = AsyncMock(return_value=stock)
            mock_wr.add_item_to_watchlist = AsyncMock(return_value=item)

            await service.add_item(
                watchlist_id=1, user_id=100, symbol="AAPL",
                target_price=123.45, notes=None,
                alert_enabled=False, db=mock_db,
            )

        call_kwargs = mock_wr.add_item_to_watchlist.call_args[1]
        assert isinstance(call_kwargs["target_price"], Decimal)
        assert call_kwargs["target_price"] == Decimal("123.45")

    @pytest.mark.asyncio
    async def test_none_target_price_passes_none(self, service, mock_db):
        """When target_price is None, None is passed to repo."""
        stock = _make_stock()
        item = _make_watchlist_item()

        with patch.object(_ws_mod, "stock_repository") as mock_sr, \
             patch.object(_ws_mod, "watchlist_repository") as mock_wr:
            mock_sr.get_by_symbol = AsyncMock(return_value=stock)
            mock_wr.add_item_to_watchlist = AsyncMock(return_value=item)

            await service.add_item(
                watchlist_id=1, user_id=100, symbol="AAPL",
                target_price=None, notes=None,
                alert_enabled=False, db=mock_db,
            )

        call_kwargs = mock_wr.add_item_to_watchlist.call_args[1]
        assert call_kwargs["target_price"] is None

    @pytest.mark.asyncio
    async def test_response_has_none_price_fields(self, service, mock_db):
        """current_price, price_change, price_change_percent, volume are None."""
        stock = _make_stock()
        item = _make_watchlist_item()

        with patch.object(_ws_mod, "stock_repository") as mock_sr, \
             patch.object(_ws_mod, "watchlist_repository") as mock_wr:
            mock_sr.get_by_symbol = AsyncMock(return_value=stock)
            mock_wr.add_item_to_watchlist = AsyncMock(return_value=item)

            result = await service.add_item(
                watchlist_id=1, user_id=100, symbol="AAPL",
                target_price=None, notes=None,
                alert_enabled=False, db=mock_db,
            )

        assert result.current_price is None
        assert result.price_change is None
        assert result.price_change_percent is None
        assert result.volume is None


# =========================================================================
# update_item
# =========================================================================

class TestUpdateItem:

    @pytest.mark.asyncio
    async def test_updates_item_successfully(self, service, mock_db):
        """Item is updated and returned with stock info."""
        updated = _make_watchlist_item(
            id=10, watchlist_id=1, stock_id=200,
            target_price=Decimal("180.00"), notes="updated",
            alert_enabled=True,
        )
        stock = _make_stock(id=200, symbol="AAPL")

        with patch.object(_ws_mod, "watchlist_repository") as mock_wr, \
             patch.object(_ws_mod, "stock_repository") as mock_sr:
            mock_wr.update_item = AsyncMock(return_value=updated)
            mock_sr.get_by_id = AsyncMock(return_value=stock)

            result = await service.update_item(
                watchlist_id=1, item_id=10, user_id=100,
                target_price=180.0, notes="updated",
                alert_enabled=True, db=mock_db,
            )

        assert isinstance(result, WatchlistItemResponse)
        assert result.target_price == 180.0
        assert result.notes == "updated"
        assert result.symbol == "AAPL"

    @pytest.mark.asyncio
    async def test_returns_none_when_item_not_found(self, service, mock_db):
        """Non-existent item returns None."""
        with patch.object(_ws_mod, "watchlist_repository") as mock_wr:
            mock_wr.update_item = AsyncMock(return_value=None)

            result = await service.update_item(
                watchlist_id=1, item_id=999, user_id=100,
                target_price=None, notes=None,
                alert_enabled=None, db=mock_db,
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_item_wrong_watchlist(self, service, mock_db):
        """Item belonging to a different watchlist returns None."""
        updated = _make_watchlist_item(id=10, watchlist_id=99)

        with patch.object(_ws_mod, "watchlist_repository") as mock_wr:
            mock_wr.update_item = AsyncMock(return_value=updated)

            result = await service.update_item(
                watchlist_id=1, item_id=10, user_id=100,
                target_price=None, notes=None,
                alert_enabled=None, db=mock_db,
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_converts_target_price_to_decimal(self, service, mock_db):
        """Float target_price is converted to Decimal."""
        updated = _make_watchlist_item(id=10, watchlist_id=1)
        stock = _make_stock()

        with patch.object(_ws_mod, "watchlist_repository") as mock_wr, \
             patch.object(_ws_mod, "stock_repository") as mock_sr:
            mock_wr.update_item = AsyncMock(return_value=updated)
            mock_sr.get_by_id = AsyncMock(return_value=stock)

            await service.update_item(
                watchlist_id=1, item_id=10, user_id=100,
                target_price=99.99, notes=None,
                alert_enabled=None, db=mock_db,
            )

        call_kwargs = mock_wr.update_item.call_args[1]
        assert isinstance(call_kwargs["target_price"], Decimal)
        assert call_kwargs["target_price"] == Decimal("99.99")

    @pytest.mark.asyncio
    async def test_none_target_price_stays_none(self, service, mock_db):
        """When target_price is None, decimal_price passed as None."""
        updated = _make_watchlist_item(id=10, watchlist_id=1)
        stock = _make_stock()

        with patch.object(_ws_mod, "watchlist_repository") as mock_wr, \
             patch.object(_ws_mod, "stock_repository") as mock_sr:
            mock_wr.update_item = AsyncMock(return_value=updated)
            mock_sr.get_by_id = AsyncMock(return_value=stock)

            await service.update_item(
                watchlist_id=1, item_id=10, user_id=100,
                target_price=None, notes=None,
                alert_enabled=None, db=mock_db,
            )

        call_kwargs = mock_wr.update_item.call_args[1]
        assert call_kwargs["target_price"] is None

    @pytest.mark.asyncio
    async def test_unknown_stock_shows_unknown_symbol(self, service, mock_db):
        """When stock lookup returns None, symbol is 'UNKNOWN'."""
        updated = _make_watchlist_item(id=10, watchlist_id=1, stock_id=200)

        with patch.object(_ws_mod, "watchlist_repository") as mock_wr, \
             patch.object(_ws_mod, "stock_repository") as mock_sr:
            mock_wr.update_item = AsyncMock(return_value=updated)
            mock_sr.get_by_id = AsyncMock(return_value=None)

            result = await service.update_item(
                watchlist_id=1, item_id=10, user_id=100,
                target_price=None, notes=None,
                alert_enabled=None, db=mock_db,
            )

        assert result.symbol == "UNKNOWN"
        assert result.company_name is None
        assert result.market_cap is None
        assert result.sector is None


# =========================================================================
# remove_item
# =========================================================================

class TestRemoveItem:

    @pytest.mark.asyncio
    async def test_removes_item_successfully(self, service, mock_db):
        """Item found and removed returns True."""
        item = _make_watchlist_item(id=10, watchlist_id=1, stock_id=200)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = item
        mock_db.execute = AsyncMock(return_value=mock_result)

        with patch.object(_ws_mod, "watchlist_repository") as mock_wr:
            mock_wr.remove_item_from_watchlist = AsyncMock(return_value=True)

            result = await service.remove_item(
                watchlist_id=1, item_id=10, user_id=100, db=mock_db,
            )
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_item_not_found(self, service, mock_db):
        """Item not found in DB returns False."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await service.remove_item(
            watchlist_id=1, item_id=999, user_id=100, db=mock_db,
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_when_repo_remove_fails(self, service, mock_db):
        """When repo returns False (e.g. already removed), result is False."""
        item = _make_watchlist_item(id=10, watchlist_id=1, stock_id=200)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = item
        mock_db.execute = AsyncMock(return_value=mock_result)

        with patch.object(_ws_mod, "watchlist_repository") as mock_wr:
            mock_wr.remove_item_from_watchlist = AsyncMock(return_value=False)

            result = await service.remove_item(
                watchlist_id=1, item_id=10, user_id=100, db=mock_db,
            )
        assert result is False

    @pytest.mark.asyncio
    async def test_uses_item_stock_id_for_removal(self, service, mock_db):
        """The stock_id from the found item is used to call remove_item_from_watchlist."""
        item = _make_watchlist_item(id=10, watchlist_id=1, stock_id=555)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = item
        mock_db.execute = AsyncMock(return_value=mock_result)

        with patch.object(_ws_mod, "watchlist_repository") as mock_wr:
            mock_wr.remove_item_from_watchlist = AsyncMock(return_value=True)

            await service.remove_item(
                watchlist_id=1, item_id=10, user_id=100, db=mock_db,
            )

        mock_wr.remove_item_from_watchlist.assert_called_once_with(
            watchlist_id=1, stock_id=555, session=mock_db,
        )


# =========================================================================
# add_to_default_watchlist
# =========================================================================

class TestAddToDefaultWatchlist:

    @pytest.mark.asyncio
    async def test_adds_stock_to_default_watchlist(self, service, mock_db):
        """Stock is added to the user's default watchlist."""
        wl = _make_watchlist(id=5)
        stock = _make_stock(id=200, symbol="MSFT", name="Microsoft")
        item = _make_watchlist_item(id=77, watchlist_id=5, stock_id=200)

        with patch.object(_ws_mod, "watchlist_repository") as mock_wr, \
             patch.object(_ws_mod, "stock_repository") as mock_sr:
            mock_wr.get_default_watchlist = AsyncMock(return_value=wl)
            mock_sr.get_by_symbol = AsyncMock(return_value=stock)
            mock_wr.add_item_to_watchlist = AsyncMock(return_value=item)

            result = await service.add_to_default_watchlist(
                user_id=100, symbol="MSFT", db=mock_db,
            )

        assert isinstance(result, WatchlistItemResponse)
        assert result.symbol == "MSFT"
        assert result.watchlist_id == 5

    @pytest.mark.asyncio
    async def test_raises_lookup_error_for_unknown_symbol(self, service, mock_db):
        """Unknown stock raises LookupError."""
        wl = _make_watchlist(id=5)

        with patch.object(_ws_mod, "watchlist_repository") as mock_wr, \
             patch.object(_ws_mod, "stock_repository") as mock_sr:
            mock_wr.get_default_watchlist = AsyncMock(return_value=wl)
            mock_sr.get_by_symbol = AsyncMock(return_value=None)

            with pytest.raises(LookupError, match="not found"):
                await service.add_to_default_watchlist(
                    user_id=100, symbol="ZZZZ", db=mock_db,
                )

    @pytest.mark.asyncio
    async def test_response_includes_stock_metadata(self, service, mock_db):
        """Response carries stock's market_cap, sector, company_name."""
        wl = _make_watchlist(id=5)
        stock = _make_stock(
            id=200, symbol="GOOG", name="Alphabet",
            sector="Communication", market_cap=2_000_000_000_000,
        )
        item = _make_watchlist_item(id=77, watchlist_id=5, stock_id=200)

        with patch.object(_ws_mod, "watchlist_repository") as mock_wr, \
             patch.object(_ws_mod, "stock_repository") as mock_sr:
            mock_wr.get_default_watchlist = AsyncMock(return_value=wl)
            mock_sr.get_by_symbol = AsyncMock(return_value=stock)
            mock_wr.add_item_to_watchlist = AsyncMock(return_value=item)

            result = await service.add_to_default_watchlist(
                user_id=100, symbol="GOOG", db=mock_db,
            )

        assert result.company_name == "Alphabet"
        assert result.sector == "Communication"
        assert result.market_cap == 2_000_000_000_000


# =========================================================================
# remove_from_default_watchlist
# =========================================================================

class TestRemoveFromDefaultWatchlist:

    @pytest.mark.asyncio
    async def test_removes_stock_successfully(self, service, mock_db):
        """Stock in default watchlist is removed and returns True."""
        wl = _make_watchlist(id=5)
        stock = _make_stock(id=200, symbol="AAPL")

        with patch.object(_ws_mod, "watchlist_repository") as mock_wr, \
             patch.object(_ws_mod, "stock_repository") as mock_sr:
            mock_wr.get_default_watchlist = AsyncMock(return_value=wl)
            mock_sr.get_by_symbol = AsyncMock(return_value=stock)
            mock_wr.is_stock_in_watchlist = AsyncMock(return_value=True)
            mock_wr.remove_item_from_watchlist = AsyncMock(return_value=True)

            result = await service.remove_from_default_watchlist(
                user_id=100, symbol="AAPL", db=mock_db,
            )
        assert result is True

    @pytest.mark.asyncio
    async def test_raises_lookup_error_for_unknown_symbol(self, service, mock_db):
        """Unknown stock raises LookupError."""
        wl = _make_watchlist(id=5)

        with patch.object(_ws_mod, "watchlist_repository") as mock_wr, \
             patch.object(_ws_mod, "stock_repository") as mock_sr:
            mock_wr.get_default_watchlist = AsyncMock(return_value=wl)
            mock_sr.get_by_symbol = AsyncMock(return_value=None)

            with pytest.raises(LookupError, match="not found"):
                await service.remove_from_default_watchlist(
                    user_id=100, symbol="ZZZZ", db=mock_db,
                )

    @pytest.mark.asyncio
    async def test_returns_false_when_stock_not_in_watchlist(self, service, mock_db):
        """Stock not in the default watchlist returns False."""
        wl = _make_watchlist(id=5)
        stock = _make_stock(id=200)

        with patch.object(_ws_mod, "watchlist_repository") as mock_wr, \
             patch.object(_ws_mod, "stock_repository") as mock_sr:
            mock_wr.get_default_watchlist = AsyncMock(return_value=wl)
            mock_sr.get_by_symbol = AsyncMock(return_value=stock)
            mock_wr.is_stock_in_watchlist = AsyncMock(return_value=False)

            result = await service.remove_from_default_watchlist(
                user_id=100, symbol="AAPL", db=mock_db,
            )
        assert result is False

    @pytest.mark.asyncio
    async def test_calls_remove_with_correct_ids(self, service, mock_db):
        """Correct watchlist_id and stock_id are passed to the repository."""
        wl = _make_watchlist(id=42)
        stock = _make_stock(id=300)

        with patch.object(_ws_mod, "watchlist_repository") as mock_wr, \
             patch.object(_ws_mod, "stock_repository") as mock_sr:
            mock_wr.get_default_watchlist = AsyncMock(return_value=wl)
            mock_sr.get_by_symbol = AsyncMock(return_value=stock)
            mock_wr.is_stock_in_watchlist = AsyncMock(return_value=True)
            mock_wr.remove_item_from_watchlist = AsyncMock(return_value=True)

            await service.remove_from_default_watchlist(
                user_id=100, symbol="AAPL", db=mock_db,
            )

        mock_wr.remove_item_from_watchlist.assert_called_once_with(
            watchlist_id=42, stock_id=300, session=mock_db,
        )


# =========================================================================
# check_symbol_in_watchlists
# =========================================================================

class TestCheckSymbolInWatchlists:

    @pytest.mark.asyncio
    async def test_symbol_in_one_watchlist(self, service, mock_db):
        """Stock found in one watchlist returns correct structure."""
        stock = _make_stock(id=200, symbol="AAPL")
        wl = _make_watchlist(id=1, name="Tech")

        with patch.object(_ws_mod, "stock_repository") as mock_sr, \
             patch.object(_ws_mod, "watchlist_repository") as mock_wr:
            mock_sr.get_by_symbol = AsyncMock(return_value=stock)
            mock_wr.get_user_watchlists = AsyncMock(return_value=[wl])
            mock_wr.is_stock_in_watchlist = AsyncMock(return_value=True)

            result = await service.check_symbol_in_watchlists(
                user_id=100, symbol="AAPL", db=mock_db,
            )

        assert result["symbol"] == "AAPL"
        assert result["stock_id"] == 200
        assert result["is_watched"] is True
        assert len(result["in_watchlists"]) == 1
        assert result["in_watchlists"][0]["watchlist_id"] == 1
        assert result["in_watchlists"][0]["watchlist_name"] == "Tech"

    @pytest.mark.asyncio
    async def test_symbol_in_multiple_watchlists(self, service, mock_db):
        """Stock in multiple watchlists returns all of them."""
        stock = _make_stock(id=200, symbol="AAPL")
        wl1 = _make_watchlist(id=1, name="Tech")
        wl2 = _make_watchlist(id=2, name="Growth")

        with patch.object(_ws_mod, "stock_repository") as mock_sr, \
             patch.object(_ws_mod, "watchlist_repository") as mock_wr:
            mock_sr.get_by_symbol = AsyncMock(return_value=stock)
            mock_wr.get_user_watchlists = AsyncMock(return_value=[wl1, wl2])
            mock_wr.is_stock_in_watchlist = AsyncMock(return_value=True)

            result = await service.check_symbol_in_watchlists(
                user_id=100, symbol="AAPL", db=mock_db,
            )

        assert result["is_watched"] is True
        assert len(result["in_watchlists"]) == 2

    @pytest.mark.asyncio
    async def test_symbol_not_in_any_watchlist(self, service, mock_db):
        """Stock not in any watchlist returns empty list."""
        stock = _make_stock(id=200, symbol="AAPL")
        wl = _make_watchlist(id=1, name="Tech")

        with patch.object(_ws_mod, "stock_repository") as mock_sr, \
             patch.object(_ws_mod, "watchlist_repository") as mock_wr:
            mock_sr.get_by_symbol = AsyncMock(return_value=stock)
            mock_wr.get_user_watchlists = AsyncMock(return_value=[wl])
            mock_wr.is_stock_in_watchlist = AsyncMock(return_value=False)

            result = await service.check_symbol_in_watchlists(
                user_id=100, symbol="AAPL", db=mock_db,
            )

        assert result["is_watched"] is False
        assert result["in_watchlists"] == []

    @pytest.mark.asyncio
    async def test_user_has_no_watchlists(self, service, mock_db):
        """User with no watchlists returns empty list."""
        stock = _make_stock(id=200, symbol="AAPL")

        with patch.object(_ws_mod, "stock_repository") as mock_sr, \
             patch.object(_ws_mod, "watchlist_repository") as mock_wr:
            mock_sr.get_by_symbol = AsyncMock(return_value=stock)
            mock_wr.get_user_watchlists = AsyncMock(return_value=[])

            result = await service.check_symbol_in_watchlists(
                user_id=100, symbol="AAPL", db=mock_db,
            )

        assert result["is_watched"] is False
        assert result["in_watchlists"] == []

    @pytest.mark.asyncio
    async def test_raises_lookup_error_for_unknown_symbol(self, service, mock_db):
        """Unknown stock symbol raises LookupError."""
        with patch.object(_ws_mod, "stock_repository") as mock_sr:
            mock_sr.get_by_symbol = AsyncMock(return_value=None)

            with pytest.raises(LookupError, match="not found"):
                await service.check_symbol_in_watchlists(
                    user_id=100, symbol="ZZZZ", db=mock_db,
                )

    @pytest.mark.asyncio
    async def test_mixed_membership(self, service, mock_db):
        """Stock in some watchlists but not others returns only matching ones."""
        stock = _make_stock(id=200, symbol="AAPL")
        wl1 = _make_watchlist(id=1, name="Tech")
        wl2 = _make_watchlist(id=2, name="Growth")
        wl3 = _make_watchlist(id=3, name="Income")

        with patch.object(_ws_mod, "stock_repository") as mock_sr, \
             patch.object(_ws_mod, "watchlist_repository") as mock_wr:
            mock_sr.get_by_symbol = AsyncMock(return_value=stock)
            mock_wr.get_user_watchlists = AsyncMock(return_value=[wl1, wl2, wl3])
            # Only in wl1 and wl3
            mock_wr.is_stock_in_watchlist = AsyncMock(
                side_effect=[True, False, True]
            )

            result = await service.check_symbol_in_watchlists(
                user_id=100, symbol="AAPL", db=mock_db,
            )

        assert result["is_watched"] is True
        assert len(result["in_watchlists"]) == 2
        ids = [w["watchlist_id"] for w in result["in_watchlists"]]
        assert 1 in ids
        assert 3 in ids
        assert 2 not in ids


# =========================================================================
# Singleton instance
# =========================================================================

class TestSingleton:

    def test_module_level_singleton_is_watchlist_service(self):
        """Module exports a pre-built WatchlistService singleton."""
        assert isinstance(watchlist_service, WatchlistService)
