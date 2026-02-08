"""
Alert Repository
Specialized async repository for alert-related operations.
"""

from typing import List, Optional
from datetime import datetime, timezone
import logging

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from backend.repositories.base import AsyncCRUDRepository, FilterCriteria, PaginationParams
from backend.models.unified_models import Alert
from backend.config.database import get_db_session

logger = logging.getLogger(__name__)


class AlertRepository(AsyncCRUDRepository[Alert]):
    """
    Specialized repository for Alert model with price-alert-specific operations.
    """

    def __init__(self):
        super().__init__(Alert)

    async def get_by_alert_id(
        self,
        alert_id: str,
        *,
        session: Optional[AsyncSession] = None
    ) -> Optional[Alert]:
        """
        Get alert by its public UUID.

        Args:
            alert_id: The public alert UUID string
            session: Optional existing session

        Returns:
            Alert instance or None
        """
        return await self.get_by_field('alert_id', alert_id, session=session)

    async def get_active_alerts_for_stock(
        self,
        stock_id: int,
        *,
        session: Optional[AsyncSession] = None
    ) -> List[Alert]:
        """
        Get all active alerts for a given stock.

        Args:
            stock_id: The stock's database ID
            session: Optional existing session

        Returns:
            List of active alerts for the stock
        """
        filters = [
            FilterCriteria(field='stock_id', operator='eq', value=stock_id),
            FilterCriteria(field='is_active', operator='eq', value=True),
        ]
        return await self.get_multi(filters=filters, session=session)

    async def get_active_alerts_for_user(
        self,
        user_id: int,
        *,
        limit: int = 100,
        session: Optional[AsyncSession] = None
    ) -> List[Alert]:
        """
        Get all active alerts for a given user.

        Args:
            user_id: The user's database ID
            limit: Maximum results to return
            session: Optional existing session

        Returns:
            List of active alerts for the user
        """
        filters = [
            FilterCriteria(field='user_id', operator='eq', value=user_id),
            FilterCriteria(field='is_active', operator='eq', value=True),
        ]
        pagination = PaginationParams(limit=limit)
        return await self.get_multi(
            filters=filters,
            pagination=pagination,
            session=session
        )

    async def deactivate_alert(
        self,
        alert_id: str,
        *,
        session: Optional[AsyncSession] = None
    ) -> Optional[Alert]:
        """
        Deactivate an alert by its public UUID.

        Args:
            alert_id: The public alert UUID string
            session: Optional existing session

        Returns:
            Updated alert or None if not found
        """
        async def _deactivate(session: AsyncSession) -> Optional[Alert]:
            alert = await self.get_by_alert_id(alert_id, session=session)
            if not alert:
                return None
            return await self.update(alert.id, {'is_active': False}, session=session)

        if session:
            return await _deactivate(session)
        else:
            async with get_db_session() as session:
                return await _deactivate(session)


# Create repository instance
alert_repository = AlertRepository()
