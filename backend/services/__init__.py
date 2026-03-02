"""
Service Layer
Business logic layer that sits between API routers and repositories/analytics.
"""

from backend.services.recommendation_service import RecommendationService, recommendation_service
from backend.services.portfolio_service import PortfolioService, portfolio_service
from backend.services.analysis_service import AnalysisService, analysis_service
from backend.services.trading_service import TradingService, trading_service
from backend.services.watchlist_service import WatchlistService, watchlist_service
from backend.services.stocks_service import StocksService, stocks_service
import backend.services.gdpr_service as gdpr_service
import backend.services.agents_service as agents_service
import backend.services.websocket_service as websocket_service
import backend.services.settings_service as settings_service
import backend.services.news_service as news_service
import backend.services.market_data_service as market_data_service

__all__ = [
    'RecommendationService',
    'recommendation_service',
    'PortfolioService',
    'portfolio_service',
    'AnalysisService',
    'analysis_service',
    'TradingService',
    'trading_service',
    'WatchlistService',
    'watchlist_service',
    'StocksService',
    'stocks_service',
    'gdpr_service',
    'agents_service',
    'websocket_service',
    'settings_service',
    'news_service',
    'market_data_service',
]
