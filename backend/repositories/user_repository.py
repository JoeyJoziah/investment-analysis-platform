"""
User Repository
Specialized async repository for user management operations.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from sqlalchemy import select, func, and_, or_, desc, asc, text
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from backend.repositories.base import AsyncCRUDRepository, FilterCriteria, SortParams, PaginationParams
from backend.models.unified_models import User, Portfolio, Watchlist, UserSession
from backend.config.database import get_db_session

logger = logging.getLogger(__name__)


class UserRepository(AsyncCRUDRepository[User]):
    """
    Specialized repository for User model with authentication and profile operations.
    """
    
    def __init__(self):
        super().__init__(User)
    
    async def get_by_email(
        self,
        email: str,
        session: Optional[AsyncSession] = None
    ) -> Optional[User]:
        """Get user by email address"""
        return await self.get_by_field('email', email.lower(), session=session)
    
    async def get_by_username(
        self,
        username: str,
        session: Optional[AsyncSession] = None
    ) -> Optional[User]:
        """Get user by username"""
        return await self.get_by_field('username', username, session=session)
    
    async def create_user(
        self,
        user_data: Dict[str, Any],
        session: Optional[AsyncSession] = None
    ) -> User:
        """Create new user with email normalization"""
        # Normalize email to lowercase
        if 'email' in user_data:
            user_data['email'] = user_data['email'].lower()
        
        return await self.create(user_data, session=session)
    
    async def authenticate_user(
        self,
        email: str,
        password_hash: str,
        session: Optional[AsyncSession] = None
    ) -> Optional[User]:
        """Authenticate user by email and password hash"""
        async def _authenticate(session: AsyncSession) -> Optional[User]:
            user = await self.get_by_email(email, session=session)
            
            if user and user.hashed_password == password_hash and user.is_active:
                # Update last login
                user.last_login = datetime.utcnow()
                user.failed_login_attempts = 0
                user.locked_until = None
                return user
            
            return None
        
        if session:
            return await _authenticate(session)
        else:
            async with get_db_session() as session:
                return await _authenticate(session)
    
    async def increment_failed_login(
        self,
        email: str,
        max_attempts: int = 5,
        lockout_minutes: int = 30,
        session: Optional[AsyncSession] = None
    ) -> Optional[User]:
        """Increment failed login attempts and lock account if necessary"""
        async def _increment_failed(session: AsyncSession) -> Optional[User]:
            user = await self.get_by_email(email, session=session)
            
            if not user:
                return None
            
            user.failed_login_attempts = (user.failed_login_attempts or 0) + 1
            
            if user.failed_login_attempts >= max_attempts:
                user.locked_until = datetime.utcnow() + timedelta(minutes=lockout_minutes)
                logger.warning(f"Account locked for user {email} due to failed login attempts")
            
            return user
        
        if session:
            return await _increment_failed(session)
        else:
            async with get_db_session() as session:
                return await _increment_failed(session)
    
    async def is_account_locked(
        self,
        email: str,
        session: Optional[AsyncSession] = None
    ) -> bool:
        """Check if user account is currently locked"""
        user = await self.get_by_email(email, session=session)
        
        if not user or not user.locked_until:
            return False
        
        return user.locked_until > datetime.utcnow()
    
    async def unlock_account(
        self,
        email: str,
        session: Optional[AsyncSession] = None
    ) -> Optional[User]:
        """Manually unlock user account"""
        async def _unlock(session: AsyncSession) -> Optional[User]:
            user = await self.get_by_email(email, session=session)
            
            if user:
                user.locked_until = None
                user.failed_login_attempts = 0
                logger.info(f"Account unlocked for user {email}")
                return user
            
            return None
        
        if session:
            return await _unlock(session)
        else:
            async with get_db_session() as session:
                return await _unlock(session)
    
    async def get_user_with_portfolios(
        self,
        user_id: int,
        session: Optional[AsyncSession] = None
    ) -> Optional[User]:
        """Get user with all portfolios loaded"""
        return await self.get_by_id(
            user_id,
            load_relationships=['portfolios'],
            session=session
        )
    
    async def get_active_users(
        self,
        days_back: int = 30,
        limit: Optional[int] = None,
        session: Optional[AsyncSession] = None
    ) -> List[User]:
        """Get users who were active within specified days"""
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        filters = [
            FilterCriteria(field='is_active', operator='eq', value=True),
            FilterCriteria(field='last_login', operator='gte', value=cutoff_date)
        ]
        
        pagination = PaginationParams(limit=limit) if limit else None
        
        return await self.get_multi(
            filters=filters,
            sort_params=[SortParams(field='last_login', direction='desc')],
            pagination=pagination,
            session=session
        )
    
    async def get_user_statistics(
        self,
        session: Optional[AsyncSession] = None
    ) -> Dict[str, Any]:
        """Get comprehensive user statistics"""
        async def _get_stats(session: AsyncSession) -> Dict[str, Any]:
            # Basic user counts
            stats_query = select(
                func.count(User.id).label('total_users'),
                func.count(User.id).filter(User.is_active == True).label('active_users'),
                func.count(User.id).filter(User.is_verified == True).label('verified_users'),
                func.count(User.id).filter(
                    User.last_login >= (datetime.utcnow() - timedelta(days=30))
                ).label('active_last_30_days'),
                func.count(User.id).filter(
                    User.created_at >= (datetime.utcnow() - timedelta(days=30))
                ).label('new_last_30_days')
            )
            
            result = await session.execute(stats_query)
            stats = result.first()
            
            # User role distribution
            role_query = select(
                User.role,
                func.count(User.id).label('count')
            ).where(User.is_active == True).group_by(User.role)
            
            result = await session.execute(role_query)
            role_distribution = {row.role.value: row.count for row in result}
            
            # Subscription tier distribution (if applicable)
            tier_query = select(
                User.subscription_tier,
                func.count(User.id).label('count')
            ).where(
                and_(
                    User.is_active == True,
                    User.subscription_tier.is_not(None)
                )
            ).group_by(User.subscription_tier)
            
            result = await session.execute(tier_query)
            tier_distribution = {row.subscription_tier: row.count for row in result}
            
            return {
                'total_users': stats.total_users or 0,
                'active_users': stats.active_users or 0,
                'verified_users': stats.verified_users or 0,
                'active_last_30_days': stats.active_last_30_days or 0,
                'new_last_30_days': stats.new_last_30_days or 0,
                'role_distribution': role_distribution,
                'subscription_distribution': tier_distribution
            }
        
        if session:
            return await _get_stats(session)
        else:
            async with get_db_session(readonly=True) as session:
                return await _get_stats(session)
    
    async def update_user_preferences(
        self,
        user_id: int,
        preferences: Dict[str, Any],
        session: Optional[AsyncSession] = None
    ) -> Optional[User]:
        """Update user preferences (merge with existing)"""
        async def _update_preferences(session: AsyncSession) -> Optional[User]:
            user = await self.get_by_id(user_id, session=session)
            
            if not user:
                return None
            
            # Merge with existing preferences
            current_prefs = user.preferences or {}
            current_prefs.update(preferences)
            
            return await self.update(user_id, {'preferences': current_prefs}, session=session)
        
        if session:
            return await _update_preferences(session)
        else:
            async with get_db_session() as session:
                return await _update_preferences(session)
    
    async def update_notification_settings(
        self,
        user_id: int,
        settings: Dict[str, Any],
        session: Optional[AsyncSession] = None
    ) -> Optional[User]:
        """Update user notification settings"""
        async def _update_notifications(session: AsyncSession) -> Optional[User]:
            user = await self.get_by_id(user_id, session=session)
            
            if not user:
                return None
            
            # Merge with existing settings
            current_settings = user.notification_settings or {}
            current_settings.update(settings)
            
            return await self.update(
                user_id, 
                {'notification_settings': current_settings}, 
                session=session
            )
        
        if session:
            return await _update_notifications(session)
        else:
            async with get_db_session() as session:
                return await _update_notifications(session)
    
    async def deactivate_user(
        self,
        user_id: int,
        reason: Optional[str] = None,
        session: Optional[AsyncSession] = None
    ) -> Optional[User]:
        """Deactivate user account"""
        async def _deactivate(session: AsyncSession) -> Optional[User]:
            user = await self.update(
                user_id,
                {
                    'is_active': False,
                    'updated_at': datetime.utcnow()
                },
                session=session
            )
            
            if user and reason:
                logger.info(f"User {user.email} deactivated. Reason: {reason}")
            
            return user
        
        if session:
            return await _deactivate(session)
        else:
            async with get_db_session() as session:
                return await _deactivate(session)
    
    async def search_users(
        self,
        query: str,
        *,
        active_only: bool = True,
        limit: int = 50,
        session: Optional[AsyncSession] = None
    ) -> List[User]:
        """Search users by email, username, or full name"""
        async def _search_users(session: AsyncSession) -> List[User]:
            search_filter = or_(
                User.email.ilike(f'%{query}%'),
                User.username.ilike(f'%{query}%'),
                User.full_name.ilike(f'%{query}%')
            )
            
            query_builder = select(User).where(search_filter)
            
            if active_only:
                query_builder = query_builder.where(User.is_active == True)
            
            query_builder = query_builder.order_by(User.created_at.desc()).limit(limit)
            
            result = await session.execute(query_builder)
            return result.scalars().all()
        
        if session:
            return await _search_users(session)
        else:
            async with get_db_session(readonly=True) as session:
                return await _search_users(session)


# Create repository instance
user_repository = UserRepository()