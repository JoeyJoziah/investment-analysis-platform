"""
Repository Package
Exports all repository instances and utilities.
"""

from backend.repositories.base import (
    AsyncBaseRepository,
    AsyncCRUDRepository,
    FilterCriteria,
    SortParams,
    PaginationParams,
    SortDirection,
    get_repository
)

# Import specific repositories - will be created as needed
try:
    from backend.repositories.stock_repository import StockRepository, stock_repository
except ImportError:
    StockRepository = None
    stock_repository = None

try:
    from backend.repositories.price_repository import PriceHistoryRepository, price_repository
except ImportError:
    PriceHistoryRepository = None
    price_repository = None

try:
    from backend.repositories.recommendation_repository import RecommendationRepository, recommendation_repository
except ImportError:
    RecommendationRepository = None
    recommendation_repository = None

try:
    from backend.repositories.user_repository import UserRepository, user_repository
except ImportError:
    UserRepository = None
    user_repository = None

try:
    from backend.repositories.portfolio_repository import PortfolioRepository, portfolio_repository
except ImportError:
    PortfolioRepository = None
    portfolio_repository = None

try:
    from backend.repositories.watchlist_repository import WatchlistRepository, watchlist_repository
except ImportError:
    WatchlistRepository = None
    watchlist_repository = None

__all__ = [
    # Base classes
    'AsyncBaseRepository',
    'AsyncCRUDRepository',
    'FilterCriteria',
    'SortParams', 
    'PaginationParams',
    'SortDirection',
    'get_repository',
    
    # Specific repositories
    'StockRepository',
    'stock_repository',
    'PriceHistoryRepository',
    'price_repository',
    'RecommendationRepository',
    'recommendation_repository',
    'UserRepository', 
    'user_repository',
    'PortfolioRepository',
    'portfolio_repository',
    'WatchlistRepository',
    'watchlist_repository'
]