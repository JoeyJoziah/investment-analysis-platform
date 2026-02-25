"""
Service Layer
Business logic layer that sits between API routers and repositories/analytics.
"""

from backend.services.recommendation_service import RecommendationService, recommendation_service
from backend.services.portfolio_service import PortfolioService, portfolio_service
from backend.services.analysis_service import AnalysisService, analysis_service
from backend.services.trading_service import TradingService, trading_service
import backend.services.gdpr_service as gdpr_service

__all__ = [
    'RecommendationService',
    'recommendation_service',
    'PortfolioService',
    'portfolio_service',
    'AnalysisService',
    'analysis_service',
    'TradingService',
    'trading_service',
    'gdpr_service',
]
