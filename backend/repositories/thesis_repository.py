"""
Investment Thesis Repository
Specialized async repository for investment thesis operations with user-scoped access.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from sqlalchemy import select, and_, desc, func
from sqlalchemy.orm import joinedload
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError

from backend.repositories.base import AsyncCRUDRepository
from backend.models.thesis import InvestmentThesis
from backend.models.unified_models import Stock
from backend.config.database import get_db_session

logger = logging.getLogger(__name__)


class InvestmentThesisRepository(AsyncCRUDRepository[InvestmentThesis]):
    """
    Repository for InvestmentThesis with user-scoped operations.

    Provides comprehensive thesis management including:
    - User-scoped CRUD operations
    - Stock-specific thesis retrieval
    - Version history tracking
    - Search and filtering
    """

    def __init__(self):
        super().__init__(InvestmentThesis)

    async def get_user_thesis_by_stock(
        self,
        user_id: int,
        stock_id: int,
        session: Optional[AsyncSession] = None
    ) -> Optional[InvestmentThesis]:
        """
        Get the latest thesis for a user and stock.

        Args:
            user_id: User ID
            stock_id: Stock ID
            session: Optional existing session

        Returns:
            Latest thesis or None if not found
        """
        async def _get_thesis(session: AsyncSession) -> Optional[InvestmentThesis]:
            query = (
                select(InvestmentThesis)
                .where(
                    and_(
                        InvestmentThesis.user_id == user_id,
                        InvestmentThesis.stock_id == stock_id
                    )
                )
                .order_by(desc(InvestmentThesis.updated_at))
                .limit(1)
            )

            result = await session.execute(query)
            return result.scalar_one_or_none()

        if session:
            return await _get_thesis(session)
        else:
            async with get_db_session(readonly=True) as session:
                return await _get_thesis(session)

    async def get_user_theses(
        self,
        user_id: int,
        limit: int = 50,
        offset: int = 0,
        session: Optional[AsyncSession] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all theses for a user with stock information.

        Args:
            user_id: User ID
            limit: Maximum number of results
            offset: Pagination offset
            session: Optional existing session

        Returns:
            List of theses with stock information
        """
        async def _get_theses(session: AsyncSession) -> List[Dict[str, Any]]:
            query = (
                select(InvestmentThesis, Stock.symbol, Stock.name)
                .join(Stock, InvestmentThesis.stock_id == Stock.id)
                .where(InvestmentThesis.user_id == user_id)
                .order_by(desc(InvestmentThesis.updated_at))
                .limit(limit)
                .offset(offset)
            )

            result = await session.execute(query)
            rows = result.all()

            theses = []
            for thesis, symbol, name in rows:
                thesis_dict = {
                    "id": thesis.id,
                    "user_id": thesis.user_id,
                    "stock_id": thesis.stock_id,
                    "investment_objective": thesis.investment_objective,
                    "time_horizon": thesis.time_horizon,
                    "target_price": thesis.target_price,
                    "business_model": thesis.business_model,
                    "competitive_advantages": thesis.competitive_advantages,
                    "financial_health": thesis.financial_health,
                    "growth_drivers": thesis.growth_drivers,
                    "risks": thesis.risks,
                    "valuation_rationale": thesis.valuation_rationale,
                    "exit_strategy": thesis.exit_strategy,
                    "content": thesis.content,
                    "version": thesis.version,
                    "created_at": thesis.created_at,
                    "updated_at": thesis.updated_at,
                    "stock_symbol": symbol,
                    "stock_name": name
                }
                theses.append(thesis_dict)

            return theses

        if session:
            return await _get_theses(session)
        else:
            async with get_db_session(readonly=True) as session:
                return await _get_theses(session)

    async def create_thesis(
        self,
        user_id: int,
        stock_id: int,
        data: Dict[str, Any],
        session: Optional[AsyncSession] = None
    ) -> InvestmentThesis:
        """
        Create a new investment thesis.

        Args:
            user_id: User ID
            stock_id: Stock ID
            data: Thesis data
            session: Optional existing session

        Returns:
            Created thesis

        Raises:
            IntegrityError: If foreign key constraints fail
        """
        async def _create(session: AsyncSession) -> InvestmentThesis:
            thesis = InvestmentThesis(
                user_id=user_id,
                stock_id=stock_id,
                **data
            )
            session.add(thesis)
            await session.commit()
            await session.refresh(thesis)
            return thesis

        if session:
            return await _create(session)
        else:
            async with get_db_session() as session:
                return await _create(session)

    async def update_thesis(
        self,
        thesis_id: int,
        user_id: int,
        data: Dict[str, Any],
        session: Optional[AsyncSession] = None
    ) -> Optional[InvestmentThesis]:
        """
        Update an existing thesis (user-scoped).

        Args:
            thesis_id: Thesis ID
            user_id: User ID (for ownership verification)
            data: Updated fields
            session: Optional existing session

        Returns:
            Updated thesis or None if not found/not owned

        Raises:
            ValueError: If user doesn't own the thesis
        """
        async def _update(session: AsyncSession) -> Optional[InvestmentThesis]:
            query = select(InvestmentThesis).where(
                and_(
                    InvestmentThesis.id == thesis_id,
                    InvestmentThesis.user_id == user_id
                )
            )
            result = await session.execute(query)
            thesis = result.scalar_one_or_none()

            if not thesis:
                return None

            # Increment version on update
            data['version'] = thesis.version + 1
            data['updated_at'] = datetime.utcnow()

            for key, value in data.items():
                if hasattr(thesis, key):
                    setattr(thesis, key, value)

            await session.commit()
            await session.refresh(thesis)
            return thesis

        if session:
            return await _update(session)
        else:
            async with get_db_session() as session:
                return await _update(session)

    async def delete_thesis(
        self,
        thesis_id: int,
        user_id: int,
        session: Optional[AsyncSession] = None
    ) -> bool:
        """
        Delete a thesis (user-scoped).

        Args:
            thesis_id: Thesis ID
            user_id: User ID (for ownership verification)
            session: Optional existing session

        Returns:
            True if deleted, False if not found/not owned
        """
        async def _delete(session: AsyncSession) -> bool:
            query = select(InvestmentThesis).where(
                and_(
                    InvestmentThesis.id == thesis_id,
                    InvestmentThesis.user_id == user_id
                )
            )
            result = await session.execute(query)
            thesis = result.scalar_one_or_none()

            if not thesis:
                return False

            await session.delete(thesis)
            await session.commit()
            return True

        if session:
            return await _delete(session)
        else:
            async with get_db_session() as session:
                return await _delete(session)

    async def count_user_theses(
        self,
        user_id: int,
        session: Optional[AsyncSession] = None
    ) -> int:
        """
        Count total theses for a user.

        Args:
            user_id: User ID
            session: Optional existing session

        Returns:
            Total thesis count
        """
        async def _count(session: AsyncSession) -> int:
            query = select(func.count(InvestmentThesis.id)).where(
                InvestmentThesis.user_id == user_id
            )
            result = await session.execute(query)
            return result.scalar() or 0

        if session:
            return await _count(session)
        else:
            async with get_db_session(readonly=True) as session:
                return await _count(session)


# Singleton instance
thesis_repository = InvestmentThesisRepository()
