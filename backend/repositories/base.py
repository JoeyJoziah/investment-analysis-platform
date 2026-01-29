"""
Async Repository Base Classes
Provides comprehensive async repository pattern with transaction handling and error management.
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional, List, Dict, Any, Union, Sequence
from contextlib import asynccontextmanager
import logging
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert, update, delete, text, func
from sqlalchemy.orm import selectinload, joinedload
from sqlalchemy.exc import IntegrityError, NoResultFound
import asyncpg

from backend.config.database import get_db_session, TransactionIsolationLevel, db_manager
from backend.models.unified_models import Base
from backend.exceptions import StaleDataError

logger = logging.getLogger(__name__)

# Type variables for generic repository
ModelType = TypeVar("ModelType", bound=Base)
CreateSchemaType = TypeVar("CreateSchemaType")
UpdateSchemaType = TypeVar("UpdateSchemaType")


class SortDirection(Enum):
    """Sort direction enum"""
    ASC = "asc"
    DESC = "desc"


@dataclass
class FilterCriteria:
    """Filter criteria for database queries"""
    field: str
    operator: str  # eq, ne, gt, gte, lt, lte, in, not_in, like, ilike
    value: Any
    
    def __post_init__(self):
        valid_operators = {
            'eq', 'ne', 'gt', 'gte', 'lt', 'lte', 
            'in', 'not_in', 'like', 'ilike', 'is_null', 'is_not_null'
        }
        if self.operator not in valid_operators:
            raise ValueError(f"Invalid operator: {self.operator}")


@dataclass
class PaginationParams:
    """Pagination parameters"""
    offset: int = 0
    limit: int = 100
    
    def __post_init__(self):
        if self.offset < 0:
            self.offset = 0
        if self.limit < 1:
            self.limit = 100
        if self.limit > 1000:
            self.limit = 1000


@dataclass
class SortParams:
    """Sort parameters"""
    field: str
    direction: SortDirection = SortDirection.ASC


class AsyncBaseRepository(Generic[ModelType], ABC):
    """
    Base async repository with comprehensive CRUD operations,
    transaction management, and error handling.
    """
    
    def __init__(self, model: type[ModelType]):
        self.model = model
        self.model_name = model.__name__
    
    async def create(
        self,
        data: Union[dict, CreateSchemaType],
        session: Optional[AsyncSession] = None
    ) -> ModelType:
        """
        Create a new record.
        
        Args:
            data: Data to create record with
            session: Optional existing session
        
        Returns:
            Created model instance
        """
        async def _create(session: AsyncSession) -> ModelType:
            # Convert Pydantic model to dict if necessary
            if hasattr(data, 'dict'):
                create_data = data.dict(exclude_unset=True)
            else:
                create_data = data
            
            # Create model instance
            instance = self.model(**create_data)
            session.add(instance)
            await session.flush()  # Get ID without committing
            await session.refresh(instance)
            
            logger.debug(f"Created {self.model_name} with ID: {instance.id}")
            return instance
        
        if session:
            return await _create(session)
        else:
            async with get_db_session() as session:
                return await _create(session)
    
    async def get_by_id(
        self,
        id: int,
        *,
        load_relationships: Optional[List[str]] = None,
        session: Optional[AsyncSession] = None
    ) -> Optional[ModelType]:
        """
        Get record by ID with optional relationship loading.
        
        Args:
            id: Record ID
            load_relationships: List of relationship names to eagerly load
            session: Optional existing session
        
        Returns:
            Model instance or None if not found
        """
        async def _get(session: AsyncSession) -> Optional[ModelType]:
            query = select(self.model).where(self.model.id == id)
            
            # Add relationship loading
            if load_relationships:
                for rel in load_relationships:
                    if hasattr(self.model, rel):
                        query = query.options(selectinload(getattr(self.model, rel)))
            
            result = await session.execute(query)
            return result.scalar_one_or_none()
        
        if session:
            return await _get(session)
        else:
            async with get_db_session(readonly=True) as session:
                return await _get(session)
    
    async def get_by_field(
        self,
        field: str,
        value: Any,
        *,
        load_relationships: Optional[List[str]] = None,
        session: Optional[AsyncSession] = None
    ) -> Optional[ModelType]:
        """
        Get record by specific field value.
        
        Args:
            field: Field name
            value: Field value
            load_relationships: List of relationship names to eagerly load
            session: Optional existing session
        
        Returns:
            Model instance or None if not found
        """
        async def _get(session: AsyncSession) -> Optional[ModelType]:
            if not hasattr(self.model, field):
                raise AttributeError(f"{self.model_name} has no field '{field}'")
            
            query = select(self.model).where(getattr(self.model, field) == value)
            
            # Add relationship loading
            if load_relationships:
                for rel in load_relationships:
                    if hasattr(self.model, rel):
                        query = query.options(selectinload(getattr(self.model, rel)))
            
            result = await session.execute(query)
            return result.scalar_one_or_none()
        
        if session:
            return await _get(session)
        else:
            async with get_db_session(readonly=True) as session:
                return await _get(session)
    
    async def get_multi(
        self,
        *,
        filters: Optional[List[FilterCriteria]] = None,
        sort_params: Optional[List[SortParams]] = None,
        pagination: Optional[PaginationParams] = None,
        load_relationships: Optional[List[str]] = None,
        session: Optional[AsyncSession] = None
    ) -> List[ModelType]:
        """
        Get multiple records with filtering, sorting, and pagination.
        
        Args:
            filters: List of filter criteria
            sort_params: List of sort parameters
            pagination: Pagination parameters
            load_relationships: List of relationship names to eagerly load
            session: Optional existing session
        
        Returns:
            List of model instances
        """
        async def _get_multi(session: AsyncSession) -> List[ModelType]:
            query = select(self.model)
            
            # Apply filters
            if filters:
                for filter_criteria in filters:
                    query = self._apply_filter(query, filter_criteria)
            
            # Apply sorting
            if sort_params:
                for sort_param in sort_params:
                    if hasattr(self.model, sort_param.field):
                        field = getattr(self.model, sort_param.field)
                        if sort_param.direction == SortDirection.DESC:
                            query = query.order_by(field.desc())
                        else:
                            query = query.order_by(field.asc())
            
            # Apply pagination
            if pagination:
                query = query.offset(pagination.offset).limit(pagination.limit)
            
            # Add relationship loading
            if load_relationships:
                for rel in load_relationships:
                    if hasattr(self.model, rel):
                        query = query.options(selectinload(getattr(self.model, rel)))
            
            result = await session.execute(query)
            return result.scalars().all()
        
        if session:
            return await _get_multi(session)
        else:
            async with get_db_session(readonly=True) as session:
                return await _get_multi(session)
    
    def _apply_filter(self, query, filter_criteria: FilterCriteria):
        """Apply filter criteria to query"""
        if not hasattr(self.model, filter_criteria.field):
            raise AttributeError(f"{self.model_name} has no field '{filter_criteria.field}'")
        
        field = getattr(self.model, filter_criteria.field)
        
        if filter_criteria.operator == 'eq':
            return query.where(field == filter_criteria.value)
        elif filter_criteria.operator == 'ne':
            return query.where(field != filter_criteria.value)
        elif filter_criteria.operator == 'gt':
            return query.where(field > filter_criteria.value)
        elif filter_criteria.operator == 'gte':
            return query.where(field >= filter_criteria.value)
        elif filter_criteria.operator == 'lt':
            return query.where(field < filter_criteria.value)
        elif filter_criteria.operator == 'lte':
            return query.where(field <= filter_criteria.value)
        elif filter_criteria.operator == 'in':
            return query.where(field.in_(filter_criteria.value))
        elif filter_criteria.operator == 'not_in':
            return query.where(~field.in_(filter_criteria.value))
        elif filter_criteria.operator == 'like':
            return query.where(field.like(filter_criteria.value))
        elif filter_criteria.operator == 'ilike':
            return query.where(field.ilike(filter_criteria.value))
        elif filter_criteria.operator == 'is_null':
            return query.where(field.is_(None))
        elif filter_criteria.operator == 'is_not_null':
            return query.where(field.is_not(None))
        else:
            raise ValueError(f"Unsupported operator: {filter_criteria.operator}")
    
    async def count(
        self,
        *,
        filters: Optional[List[FilterCriteria]] = None,
        session: Optional[AsyncSession] = None
    ) -> int:
        """
        Count records with optional filtering.
        
        Args:
            filters: List of filter criteria
            session: Optional existing session
        
        Returns:
            Record count
        """
        async def _count(session: AsyncSession) -> int:
            query = select(func.count()).select_from(self.model)
            
            # Apply filters
            if filters:
                for filter_criteria in filters:
                    query = self._apply_filter(query, filter_criteria)
            
            result = await session.execute(query)
            return result.scalar()
        
        if session:
            return await _count(session)
        else:
            async with get_db_session(readonly=True) as session:
                return await _count(session)
    
    async def update(
        self,
        id: int,
        data: Union[dict, UpdateSchemaType],
        session: Optional[AsyncSession] = None
    ) -> Optional[ModelType]:
        """
        Update record by ID.

        Args:
            id: Record ID
            data: Update data
            session: Optional existing session

        Returns:
            Updated model instance or None if not found
        """
        async def _update(session: AsyncSession) -> Optional[ModelType]:
            # Convert Pydantic model to dict if necessary
            if hasattr(data, 'dict'):
                update_data = data.dict(exclude_unset=True)
            else:
                update_data = data

            # Remove None values to avoid updating with None
            update_data = {k: v for k, v in update_data.items() if v is not None}

            if not update_data:
                # No data to update, just return existing record
                return await self.get_by_id(id, session=session)

            stmt = update(self.model).where(self.model.id == id).values(**update_data)
            result = await session.execute(stmt)

            if result.rowcount == 0:
                return None

            # Return updated record
            return await self.get_by_id(id, session=session)

        if session:
            return await _update(session)
        else:
            async with get_db_session() as session:
                return await _update(session)

    async def update_with_lock(
        self,
        id: int,
        data: Union[dict, UpdateSchemaType],
        *,
        expected_version: Optional[int] = None,
        for_update: bool = True,
        session: Optional[AsyncSession] = None
    ) -> Optional[ModelType]:
        """
        Update record with optimistic locking and optional pessimistic lock.

        Args:
            id: Record ID
            data: Update data
            expected_version: Expected version for optimistic locking
            for_update: Use SELECT FOR UPDATE (pessimistic lock)
            session: Optional existing session

        Returns:
            Updated model instance or None if not found

        Raises:
            StaleDataError: If version mismatch detected
        """
        async def _update_with_lock(session: AsyncSession) -> Optional[ModelType]:
            # Build query with optional FOR UPDATE lock
            query = select(self.model).where(self.model.id == id)

            if for_update:
                # Pessimistic lock - prevents other transactions from reading/writing
                query = query.with_for_update()

            result = await session.execute(query)
            instance = result.scalar_one_or_none()

            if not instance:
                return None

            # Optimistic locking - check version if model has version column
            if hasattr(instance, 'version') and expected_version is not None:
                if instance.version != expected_version:
                    raise StaleDataError(
                        entity_type=self.model_name,
                        entity_id=id,
                        expected_version=expected_version,
                        current_version=instance.version
                    )

            # Convert Pydantic model to dict if necessary
            if hasattr(data, 'dict'):
                update_data = data.dict(exclude_unset=True)
            else:
                update_data = dict(data) if not isinstance(data, dict) else data

            # Remove None values
            update_data = {k: v for k, v in update_data.items() if v is not None}

            if not update_data:
                return instance

            # Increment version if model supports it
            if hasattr(instance, 'version'):
                update_data['version'] = instance.version + 1

            # Apply updates
            for key, value in update_data.items():
                if hasattr(instance, key):
                    setattr(instance, key, value)

            await session.flush()
            await session.refresh(instance)

            logger.debug(
                f"Updated {self.model_name} ID {id} with version {getattr(instance, 'version', 'N/A')}"
            )

            return instance

        if session:
            return await _update_with_lock(session)
        else:
            async with get_db_session() as session:
                return await _update_with_lock(session)

    async def get_with_lock(
        self,
        id: int,
        *,
        for_update: bool = True,
        skip_locked: bool = False,
        nowait: bool = False,
        session: Optional[AsyncSession] = None
    ) -> Optional[ModelType]:
        """
        Get record with pessimistic lock (SELECT FOR UPDATE).

        Args:
            id: Record ID
            for_update: Use FOR UPDATE lock
            skip_locked: Skip locked rows (returns None if locked)
            nowait: Don't wait for lock (raises exception if locked)
            session: Optional existing session

        Returns:
            Model instance or None if not found/locked
        """
        async def _get_with_lock(session: AsyncSession) -> Optional[ModelType]:
            query = select(self.model).where(self.model.id == id)

            if for_update:
                if skip_locked:
                    query = query.with_for_update(skip_locked=True)
                elif nowait:
                    query = query.with_for_update(nowait=True)
                else:
                    query = query.with_for_update()

            result = await session.execute(query)
            return result.scalar_one_or_none()

        if session:
            return await _get_with_lock(session)
        else:
            async with get_db_session() as session:
                return await _get_with_lock(session)
    
    async def delete(
        self,
        id: int,
        session: Optional[AsyncSession] = None
    ) -> bool:
        """
        Delete record by ID.
        
        Args:
            id: Record ID
            session: Optional existing session
        
        Returns:
            True if deleted, False if not found
        """
        async def _delete(session: AsyncSession) -> bool:
            stmt = delete(self.model).where(self.model.id == id)
            result = await session.execute(stmt)
            return result.rowcount > 0
        
        if session:
            return await _delete(session)
        else:
            async with get_db_session() as session:
                return await _delete(session)
    
    async def bulk_create(
        self,
        data_list: List[Union[dict, CreateSchemaType]],
        *,
        batch_size: int = 1000,
        session: Optional[AsyncSession] = None
    ) -> List[ModelType]:
        """
        Bulk create records with batching.
        
        Args:
            data_list: List of data to create records with
            batch_size: Number of records per batch
            session: Optional existing session
        
        Returns:
            List of created model instances
        """
        if not data_list:
            return []
        
        async def _bulk_create(session: AsyncSession) -> List[ModelType]:
            created_instances = []
            
            for i in range(0, len(data_list), batch_size):
                batch = data_list[i:i + batch_size]
                batch_instances = []
                
                for item_data in batch:
                    if hasattr(item_data, 'dict'):
                        create_data = item_data.dict(exclude_unset=True)
                    else:
                        create_data = item_data
                    
                    instance = self.model(**create_data)
                    session.add(instance)
                    batch_instances.append(instance)
                
                await session.flush()
                
                # Refresh instances to get IDs
                for instance in batch_instances:
                    await session.refresh(instance)
                
                created_instances.extend(batch_instances)
                logger.debug(f"Created batch of {len(batch)} {self.model_name} records")
            
            logger.info(f"Bulk created {len(created_instances)} {self.model_name} records")
            return created_instances
        
        if session:
            return await _bulk_create(session)
        else:
            async with get_db_session() as session:
                return await _bulk_create(session)
    
    async def upsert(
        self,
        data: Union[dict, CreateSchemaType],
        *,
        conflict_fields: List[str],
        update_fields: Optional[List[str]] = None,
        session: Optional[AsyncSession] = None
    ) -> ModelType:
        """
        Upsert (insert or update) a record.
        
        Args:
            data: Data for upsert
            conflict_fields: Fields to check for conflicts
            update_fields: Fields to update on conflict (if None, updates all)
            session: Optional existing session
        
        Returns:
            Upserted model instance
        """
        async def _upsert(session: AsyncSession) -> ModelType:
            from sqlalchemy.dialects.postgresql import insert
            
            if hasattr(data, 'dict'):
                upsert_data = data.dict(exclude_unset=True)
            else:
                upsert_data = data
            
            stmt = insert(self.model).values(**upsert_data)
            
            # Determine update columns
            if update_fields:
                update_columns = {
                    field: stmt.excluded[field] 
                    for field in update_fields 
                    if field in upsert_data and hasattr(self.model, field)
                }
            else:
                # Update all non-conflict fields
                update_columns = {
                    key: stmt.excluded[key] 
                    for key in upsert_data.keys() 
                    if key not in conflict_fields and hasattr(self.model, key)
                }
            
            if update_columns:
                # Get conflict columns
                conflict_columns = [
                    getattr(self.model, field) 
                    for field in conflict_fields 
                    if hasattr(self.model, field)
                ]
                
                stmt = stmt.on_conflict_do_update(
                    index_elements=conflict_columns,
                    set_=update_columns
                )
            else:
                stmt = stmt.on_conflict_do_nothing()
            
            # Add returning clause to get the record
            stmt = stmt.returning(self.model)
            
            result = await session.execute(stmt)
            return result.scalar_one()
        
        if session:
            return await _upsert(session)
        else:
            async with get_db_session() as session:
                return await _upsert(session)
    
    @asynccontextmanager
    async def transaction(
        self,
        *,
        isolation_level: Optional[TransactionIsolationLevel] = None
    ):
        """
        Context manager for database transaction with retry logic.
        
        Args:
            isolation_level: Transaction isolation level
        
        Yields:
            AsyncSession: Database session within transaction
        """
        async def _execute_transaction():
            async with get_db_session(isolation_level=isolation_level) as session:
                yield session
        
        # Execute transaction with retry logic for deadlocks
        await db_manager.execute_with_retry(_execute_transaction)


class AsyncCRUDRepository(AsyncBaseRepository[ModelType]):
    """
    Enhanced CRUD repository with additional convenience methods.
    """
    
    async def exists(
        self,
        id: int,
        session: Optional[AsyncSession] = None
    ) -> bool:
        """Check if record exists by ID"""
        async def _exists(session: AsyncSession) -> bool:
            query = select(func.count()).select_from(self.model).where(self.model.id == id)
            result = await session.execute(query)
            return result.scalar() > 0
        
        if session:
            return await _exists(session)
        else:
            async with get_db_session(readonly=True) as session:
                return await _exists(session)
    
    async def get_or_create(
        self,
        defaults: Optional[dict] = None,
        session: Optional[AsyncSession] = None,
        **kwargs
    ) -> tuple[ModelType, bool]:
        """
        Get existing record or create new one.
        
        Args:
            defaults: Default values for creation
            session: Optional existing session
            **kwargs: Lookup parameters
        
        Returns:
            Tuple of (instance, created_flag)
        """
        async def _get_or_create(session: AsyncSession) -> tuple[ModelType, bool]:
            # Try to get existing record
            filters = [FilterCriteria(field=k, operator='eq', value=v) for k, v in kwargs.items()]
            existing_records = await self.get_multi(filters=filters, session=session)
            
            if existing_records:
                return existing_records[0], False
            
            # Create new record
            create_data = kwargs.copy()
            if defaults:
                create_data.update(defaults)
            
            instance = await self.create(create_data, session=session)
            return instance, True
        
        if session:
            return await _get_or_create(session)
        else:
            async with get_db_session() as session:
                return await _get_or_create(session)
    
    async def update_or_create(
        self,
        defaults: Optional[dict] = None,
        session: Optional[AsyncSession] = None,
        **kwargs
    ) -> tuple[ModelType, bool]:
        """
        Update existing record or create new one.
        
        Args:
            defaults: Default/update values
            session: Optional existing session
            **kwargs: Lookup parameters
        
        Returns:
            Tuple of (instance, created_flag)
        """
        async def _update_or_create(session: AsyncSession) -> tuple[ModelType, bool]:
            # Try to get existing record
            filters = [FilterCriteria(field=k, operator='eq', value=v) for k, v in kwargs.items()]
            existing_records = await self.get_multi(filters=filters, session=session)
            
            if existing_records:
                # Update existing record
                instance = existing_records[0]
                if defaults:
                    updated_instance = await self.update(instance.id, defaults, session=session)
                    return updated_instance or instance, False
                return instance, False
            
            # Create new record
            create_data = kwargs.copy()
            if defaults:
                create_data.update(defaults)
            
            instance = await self.create(create_data, session=session)
            return instance, True
        
        if session:
            return await _update_or_create(session)
        else:
            async with get_db_session() as session:
                return await _update_or_create(session)


# Repository registry for dependency injection
_repository_registry: Dict[type, AsyncCRUDRepository] = {}


def get_repository(model: type[ModelType]) -> AsyncCRUDRepository[ModelType]:
    """
    Get repository instance for a model (singleton pattern).
    
    Args:
        model: SQLAlchemy model class
    
    Returns:
        Repository instance for the model
    """
    if model not in _repository_registry:
        _repository_registry[model] = AsyncCRUDRepository(model)
    
    return _repository_registry[model]