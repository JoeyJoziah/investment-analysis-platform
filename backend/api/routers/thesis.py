"""
Investment Thesis API Router
Provides CRUD endpoints for investment thesis documentation.
"""

from fastapi import APIRouter, HTTPException, Depends, status, Path, Query
from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError
import logging

from backend.config.database import get_async_db_session
from backend.auth.oauth2 import get_current_user
from backend.models.unified_models import User
from backend.repositories.thesis_repository import thesis_repository
from backend.repositories.stock_repository import stock_repository
from backend.models.schemas import (
    InvestmentThesisCreate,
    InvestmentThesisUpdate,
    InvestmentThesisResponse,
)
from backend.models.api_response import ApiResponse, success_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/thesis", tags=["investment-thesis"])


# =======================
# Helper Functions
# =======================

async def get_thesis_or_404(
    thesis_id: int,
    user_id: int,
    db: AsyncSession,
) -> Any:
    """
    Get a thesis by ID with ownership verification.

    Args:
        thesis_id: The thesis ID
        user_id: The current user's ID
        db: Database session

    Returns:
        Thesis object if found and authorized

    Raises:
        HTTPException: 404 if not found, 403 if not authorized
    """
    thesis = await thesis_repository.get_user_thesis_by_stock(
        user_id=user_id,
        stock_id=thesis_id,
        session=db
    )

    if not thesis:
        # Try direct ID lookup
        from sqlalchemy import select, and_
        from backend.models.thesis import InvestmentThesis

        query = select(InvestmentThesis).where(InvestmentThesis.id == thesis_id)
        result = await db.execute(query)
        thesis = result.scalar_one_or_none()

        if not thesis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Investment thesis with ID {thesis_id} not found"
            )

        # Ownership check
        if thesis.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this thesis"
            )

    return thesis


def convert_thesis_to_response(thesis: Any, stock_symbol: str = None, stock_name: str = None) -> InvestmentThesisResponse:
    """Convert a thesis model to InvestmentThesisResponse."""
    return InvestmentThesisResponse(
        id=thesis.id,
        user_id=thesis.user_id,
        stock_id=thesis.stock_id,
        investment_objective=thesis.investment_objective,
        time_horizon=thesis.time_horizon,
        target_price=thesis.target_price,
        business_model=thesis.business_model,
        competitive_advantages=thesis.competitive_advantages,
        financial_health=thesis.financial_health,
        growth_drivers=thesis.growth_drivers,
        risks=thesis.risks,
        valuation_rationale=thesis.valuation_rationale,
        exit_strategy=thesis.exit_strategy,
        content=thesis.content,
        version=thesis.version,
        created_at=thesis.created_at,
        updated_at=thesis.updated_at,
        stock_symbol=stock_symbol,
        stock_name=stock_name
    )


# =======================
# Endpoints
# =======================

@router.post(
    "/",
    status_code=status.HTTP_201_CREATED,
    summary="Create Investment Thesis",
    description="Create a new investment thesis for a stock"
)
async def create_thesis(
    thesis_data: InvestmentThesisCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db_session)
) -> ApiResponse[InvestmentThesisResponse]:
    """
    Create a new investment thesis.

    - **stock_id**: ID of the stock this thesis is for
    - **investment_objective**: Primary investment goal
    - **time_horizon**: Expected holding period (short-term/medium-term/long-term)
    - Additional fields for comprehensive analysis
    """
    try:
        # Verify stock exists
        stock = await stock_repository.get_by_id(thesis_data.stock_id, session=db)
        if not stock:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Stock with ID {thesis_data.stock_id} not found"
            )

        # Check if thesis already exists for this user+stock
        existing = await thesis_repository.get_user_thesis_by_stock(
            user_id=current_user.id,
            stock_id=thesis_data.stock_id,
            session=db
        )
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Investment thesis already exists for this stock. Use PUT to update."
            )

        # Create thesis
        thesis_dict = thesis_data.model_dump(exclude={'stock_id'})
        thesis = await thesis_repository.create_thesis(
            user_id=current_user.id,
            stock_id=thesis_data.stock_id,
            data=thesis_dict,
            session=db
        )

        logger.info(f"Created thesis {thesis.id} for user {current_user.id}, stock {thesis_data.stock_id}")

        return success_response(data=convert_thesis_to_response(thesis, stock.symbol, stock.name))

    except IntegrityError as e:
        logger.error(f"Database integrity error creating thesis: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid stock_id or database constraint violation"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating thesis: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create investment thesis"
        )


@router.get(
    "/{thesis_id}",
    summary="Get Investment Thesis",
    description="Get a specific investment thesis by ID"
)
async def get_thesis(
    thesis_id: int = Path(..., gt=0, description="Thesis ID"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db_session)
) -> ApiResponse[InvestmentThesisResponse]:
    """
    Retrieve a specific investment thesis by ID.

    Returns the thesis with stock information if found and user has access.
    """
    thesis = await get_thesis_or_404(thesis_id, current_user.id, db)

    # Get stock info
    stock = await stock_repository.get_by_id(thesis.stock_id, session=db)

    return success_response(data=convert_thesis_to_response(
        thesis,
        stock.symbol if stock else None,
        stock.name if stock else None
    ))


@router.get(
    "/stock/{stock_id}",
    summary="Get Thesis by Stock",
    description="Get the investment thesis for a specific stock"
)
async def get_thesis_by_stock(
    stock_id: int = Path(..., gt=0, description="Stock ID"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db_session)
) -> ApiResponse[InvestmentThesisResponse]:
    """
    Retrieve the investment thesis for a specific stock.

    Returns 404 if no thesis exists for this user+stock combination.
    """
    thesis = await thesis_repository.get_user_thesis_by_stock(
        user_id=current_user.id,
        stock_id=stock_id,
        session=db
    )

    if not thesis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No investment thesis found for stock ID {stock_id}"
        )

    # Get stock info
    stock = await stock_repository.get_by_id(stock_id, session=db)

    return success_response(data=convert_thesis_to_response(
        thesis,
        stock.symbol if stock else None,
        stock.name if stock else None
    ))


@router.get(
    "/",
    summary="List Investment Theses",
    description="Get all investment theses for the current user"
)
async def list_theses(
    limit: int = Query(50, ge=1, le=100, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db_session)
) -> ApiResponse[List[InvestmentThesisResponse]]:
    """
    List all investment theses for the current user.

    Returns theses sorted by most recently updated.
    """
    theses_data = await thesis_repository.get_user_theses(
        user_id=current_user.id,
        limit=limit,
        offset=offset,
        session=db
    )

    return success_response(data=[
        InvestmentThesisResponse(**thesis_dict)
        for thesis_dict in theses_data
    ])


@router.put(
    "/{thesis_id}",
    summary="Update Investment Thesis",
    description="Update an existing investment thesis"
)
async def update_thesis(
    thesis_id: int = Path(..., gt=0, description="Thesis ID"),
    thesis_data: InvestmentThesisUpdate = ...,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db_session)
) -> ApiResponse[InvestmentThesisResponse]:
    """
    Update an existing investment thesis.

    Only the owner can update a thesis. Version is automatically incremented.
    """
    # Verify ownership
    await get_thesis_or_404(thesis_id, current_user.id, db)

    # Update thesis
    update_dict = thesis_data.model_dump(exclude_unset=True)

    if not update_dict:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No fields to update"
        )

    thesis = await thesis_repository.update_thesis(
        thesis_id=thesis_id,
        user_id=current_user.id,
        data=update_dict,
        session=db
    )

    if not thesis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thesis not found or update failed"
        )

    # Get stock info
    stock = await stock_repository.get_by_id(thesis.stock_id, session=db)

    logger.info(f"Updated thesis {thesis_id} for user {current_user.id}, now version {thesis.version}")

    return success_response(data=convert_thesis_to_response(
        thesis,
        stock.symbol if stock else None,
        stock.name if stock else None
    ))


@router.delete(
    "/{thesis_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Investment Thesis",
    description="Delete an investment thesis"
)
async def delete_thesis(
    thesis_id: int = Path(..., gt=0, description="Thesis ID"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db_session)
):
    """
    Delete an investment thesis.

    Only the owner can delete a thesis. This action is permanent.
    """
    # Verify ownership
    await get_thesis_or_404(thesis_id, current_user.id, db)

    # Delete thesis
    deleted = await thesis_repository.delete_thesis(
        thesis_id=thesis_id,
        user_id=current_user.id,
        session=db
    )

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thesis not found or deletion failed"
        )

    logger.info(f"Deleted thesis {thesis_id} for user {current_user.id}")
    return None
