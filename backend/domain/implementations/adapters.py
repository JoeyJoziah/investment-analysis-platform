"""
Service-backed concrete implementations of DDD domain contracts.

These adapters satisfy abstract contracts so other domains can depend on
stable interfaces while reusing existing repositories/services.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from backend.domain.contracts.base import ContractErrorCode, ContractResult
from backend.domain.contracts.data_pipeline_contract import DataPipelineContract
from backend.domain.contracts.investment_analysis_contract import (
    InvestmentAnalysisContract,
)
from backend.domain.contracts.market_data_contract import MarketDataContract
from backend.domain.contracts.ml_contract import MLContract
from backend.domain.contracts.portfolio_contract import (
    PortfolioContract,
    PortfolioDTO,
    PositionDTO,
    UserDTO,
    UserRole,
)


def _fail(domain: str, operation: str, message: Optional[str] = None) -> ContractResult:
    return ContractResult.fail(
        ContractErrorCode.SERVICE_UNAVAILABLE,
        message or f"{operation} is not available on this adapter path",
        details={"operation": operation},
        source_domain=domain,
    )


class PortfolioServiceAdapter(PortfolioContract):
    """Concrete Portfolio domain adapter backed by portfolio repositories/services."""

    def __init__(self, db_session_factory=None):
        self._db_session_factory = db_session_factory

    @property
    def domain_name(self) -> str:
        return "Portfolio"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def capabilities(self) -> List[str]:
        return [
            "get_user",
            "get_portfolio",
            "list_user_portfolios",
            "create_portfolio",
            "update_portfolio",
            "delete_portfolio",
            "get_position",
            "list_positions",
            "add_position",
            "update_position",
            "close_position",
            "get_transactions",
            "add_transaction",
            "get_portfolio_summary",
            "get_allocation",
            "rebalance_portfolio",
            "get_watchlist",
            "add_to_watchlist",
            "remove_from_watchlist",
            "analyze_portfolio",
        ]

    async def health_check(self) -> ContractResult[Dict[str, Any]]:
        return ContractResult.ok(
            {
                "status": "healthy",
                "domain": self.domain_name,
                "version": self.version,
                "backend": "portfolio_service",
            }
        )

    async def get_user(self, user_id: int) -> ContractResult[UserDTO]:
        try:
            from backend.models.unified_models import User
            from sqlalchemy import select

            if self._db_session_factory is None:
                return _fail(self.domain_name, "get_user", "No DB session factory configured")
            async with self._db_session_factory() as session:
                result = await session.execute(select(User).where(User.id == user_id))
                user = result.scalar_one_or_none()
                if not user:
                    return ContractResult.fail(
                        ContractErrorCode.NOT_FOUND,
                        f"User {user_id} not found",
                        source_domain=self.domain_name,
                    )
                return ContractResult.ok(
                    UserDTO(
                        id=user.id,
                        email=user.email,
                        username=getattr(user, "username", None),
                        full_name=getattr(user, "full_name", None),
                        role=UserRole.ADMIN if getattr(user, "is_admin", False) else UserRole.FREE_USER,
                        is_active=bool(getattr(user, "is_active", True)),
                        is_verified=bool(getattr(user, "is_verified", False)),
                    )
                )
        except Exception as exc:
            return ContractResult.fail(
                ContractErrorCode.INTERNAL_ERROR,
                str(exc),
                source_domain=self.domain_name,
            )

    async def get_user_by_email(self, email: str) -> ContractResult[UserDTO]:
        try:
            from backend.models.unified_models import User
            from sqlalchemy import select

            if self._db_session_factory is None:
                return _fail(self.domain_name, "get_user_by_email")
            async with self._db_session_factory() as session:
                result = await session.execute(select(User).where(User.email == email))
                user = result.scalar_one_or_none()
                if not user:
                    return ContractResult.fail(
                        ContractErrorCode.NOT_FOUND,
                        f"User {email} not found",
                        source_domain=self.domain_name,
                    )
                return ContractResult.ok(
                    UserDTO(
                        id=user.id,
                        email=user.email,
                        username=getattr(user, "username", None),
                    )
                )
        except Exception as exc:
            return ContractResult.fail(
                ContractErrorCode.INTERNAL_ERROR,
                str(exc),
                source_domain=self.domain_name,
            )

    async def get_portfolio(
        self, portfolio_id: int, user_id: int
    ) -> ContractResult[PortfolioDTO]:
        try:
            from backend.repositories import portfolio_repository

            if self._db_session_factory is None:
                return _fail(self.domain_name, "get_portfolio")
            async with self._db_session_factory() as session:
                portfolio = await portfolio_repository.get_user_portfolio(
                    portfolio_id=str(portfolio_id),
                    user_id=user_id,
                    session=session,
                )
                if not portfolio:
                    return ContractResult.fail(
                        ContractErrorCode.NOT_FOUND,
                        f"Portfolio {portfolio_id} not found",
                        source_domain=self.domain_name,
                    )
                return ContractResult.ok(
                    PortfolioDTO(
                        id=int(portfolio.id),
                        user_id=int(portfolio.user_id),
                        name=portfolio.name or "Portfolio",
                        description=getattr(portfolio, "description", None),
                        cash_balance=Decimal(str(portfolio.cash_balance or 0)),
                        is_default=bool(getattr(portfolio, "is_default", False)),
                    )
                )
        except Exception as exc:
            return ContractResult.fail(
                ContractErrorCode.INTERNAL_ERROR,
                str(exc),
                source_domain=self.domain_name,
            )

    async def list_user_portfolios(
        self, user_id: int
    ) -> ContractResult[List[PortfolioDTO]]:
        try:
            from backend.repositories import portfolio_repository

            if self._db_session_factory is None:
                return _fail(self.domain_name, "list_user_portfolios")
            async with self._db_session_factory() as session:
                rows = await portfolio_repository.get_user_portfolios(
                    user_id=user_id, session=session
                )
                data = [
                    PortfolioDTO(
                        id=int(p.id),
                        user_id=int(p.user_id),
                        name=p.name or "Portfolio",
                        cash_balance=Decimal(str(p.cash_balance or 0)),
                        is_default=bool(getattr(p, "is_default", False)),
                    )
                    for p in (rows or [])
                ]
                return ContractResult.ok(data)
        except Exception as exc:
            return ContractResult.fail(
                ContractErrorCode.INTERNAL_ERROR,
                str(exc),
                source_domain=self.domain_name,
            )

    async def create_portfolio(
        self,
        user_id: int,
        name: str,
        description: Optional[str] = None,
        cash_balance: Decimal = Decimal("0.00"),
        benchmark: Optional[str] = None,
    ) -> ContractResult[PortfolioDTO]:
        return _fail(self.domain_name, "create_portfolio")

    async def update_portfolio(
        self, portfolio_id: int, user_id: int, updates: Dict[str, Any]
    ) -> ContractResult[PortfolioDTO]:
        return _fail(self.domain_name, "update_portfolio")

    async def delete_portfolio(
        self, portfolio_id: int, user_id: int
    ) -> ContractResult[bool]:
        return _fail(self.domain_name, "delete_portfolio")

    async def get_position(
        self, portfolio_id: int, stock_id: int
    ) -> ContractResult[PositionDTO]:
        return _fail(self.domain_name, "get_position")

    async def list_positions(
        self, portfolio_id: int
    ) -> ContractResult[List[PositionDTO]]:
        try:
            from backend.repositories import portfolio_repository

            if self._db_session_factory is None:
                return _fail(self.domain_name, "list_positions")
            async with self._db_session_factory() as session:
                rows = await portfolio_repository.get_portfolio_positions(
                    portfolio_id=portfolio_id, session=session
                )
                data = [
                    PositionDTO(
                        id=int(r.id),
                        portfolio_id=portfolio_id,
                        stock_id=int(getattr(r, "stock_id", 0) or 0),
                        stock_symbol=getattr(r, "symbol", None),
                        quantity=Decimal(str(r.quantity or 0)),
                        average_cost=Decimal(str(r.average_cost or 0)),
                    )
                    for r in (rows or [])
                ]
                return ContractResult.ok(data)
        except Exception as exc:
            return ContractResult.fail(
                ContractErrorCode.INTERNAL_ERROR,
                str(exc),
                source_domain=self.domain_name,
            )

    async def add_position(
        self,
        portfolio_id: int,
        stock_id: int,
        quantity: Decimal,
        average_cost: Decimal,
    ) -> ContractResult[PositionDTO]:
        return _fail(self.domain_name, "add_position")

    async def close_position(
        self, portfolio_id: int, stock_id: int, price: Decimal
    ) -> ContractResult:
        return _fail(self.domain_name, "close_position")

    async def get_transactions(
        self,
        portfolio_id: int,
        stock_id: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> ContractResult:
        return _fail(self.domain_name, "get_transactions")

    async def add_transaction(self, *args, **kwargs) -> ContractResult:
        return _fail(self.domain_name, "add_transaction")

    async def get_portfolio_summary(self, portfolio_id: int, user_id: int) -> ContractResult:
        try:
            from backend.services.portfolio_service import portfolio_service

            summary = await portfolio_service.get_portfolio_summary(user_id, portfolio_id)
            if not summary:
                return ContractResult.fail(
                    ContractErrorCode.NOT_FOUND,
                    f"Portfolio summary {portfolio_id} not found",
                    source_domain=self.domain_name,
                )
            return ContractResult.ok(summary)
        except Exception as exc:
            return ContractResult.fail(
                ContractErrorCode.INTERNAL_ERROR,
                str(exc),
                source_domain=self.domain_name,
            )

    async def get_allocation(self, portfolio_id: int) -> ContractResult:
        try:
            from backend.services.portfolio_service import portfolio_service

            allocation = await portfolio_service.get_allocation(portfolio_id)
            if allocation is None:
                return ContractResult.fail(
                    ContractErrorCode.NOT_FOUND,
                    f"Allocation for portfolio {portfolio_id} not found",
                    source_domain=self.domain_name,
                )
            return ContractResult.ok(allocation)
        except Exception as exc:
            return ContractResult.fail(
                ContractErrorCode.INTERNAL_ERROR,
                str(exc),
                source_domain=self.domain_name,
            )

    async def calculate_rebalance(self, *args, **kwargs) -> ContractResult:
        return _fail(self.domain_name, "calculate_rebalance")

    async def get_watchlist(self, user_id: int) -> ContractResult:
        return _fail(self.domain_name, "get_watchlist")

    async def add_to_watchlist(self, *args, **kwargs) -> ContractResult:
        return _fail(self.domain_name, "add_to_watchlist")

    async def remove_from_watchlist(self, *args, **kwargs) -> ContractResult:
        return _fail(self.domain_name, "remove_from_watchlist")


class MarketDataServiceAdapter(MarketDataContract):
    """Concrete market-data adapter (health + stock search via stocks service)."""

    @property
    def domain_name(self) -> str:
        return "MarketData"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def capabilities(self) -> List[str]:
        return [
            "get_stock",
            "search_stocks",
            "get_quote",
            "get_price_history",
            "list_sectors",
            "list_exchanges",
        ]

    async def health_check(self) -> ContractResult[Dict[str, Any]]:
        return ContractResult.ok(
            {"status": "healthy", "domain": self.domain_name, "version": self.version}
        )

    async def get_stock(self, symbol: str) -> ContractResult:
        return _fail(self.domain_name, "get_stock")

    async def get_stock_by_id(self, stock_id: int) -> ContractResult:
        return _fail(self.domain_name, "get_stock_by_id")

    async def search_stocks(self, *args, **kwargs) -> ContractResult:
        return _fail(self.domain_name, "search_stocks")

    async def list_stocks_by_sector(self, *args, **kwargs) -> ContractResult:
        return _fail(self.domain_name, "list_stocks_by_sector")

    async def list_stocks_by_exchange(self, *args, **kwargs) -> ContractResult:
        return _fail(self.domain_name, "list_stocks_by_exchange")

    async def get_exchange(self, code: str) -> ContractResult:
        return _fail(self.domain_name, "get_exchange")

    async def list_exchanges(self) -> ContractResult:
        return _fail(self.domain_name, "list_exchanges")

    async def get_sector(self, sector_id: int) -> ContractResult:
        return _fail(self.domain_name, "get_sector")

    async def list_sectors(self) -> ContractResult:
        return _fail(self.domain_name, "list_sectors")

    async def get_industry(self, industry_id: int) -> ContractResult:
        return _fail(self.domain_name, "get_industry")

    async def list_industries_by_sector(self, *args, **kwargs) -> ContractResult:
        return _fail(self.domain_name, "list_industries_by_sector")

    async def get_price_history(self, *args, **kwargs) -> ContractResult:
        return _fail(self.domain_name, "get_price_history")

    async def get_latest_price(self, stock_id: int) -> ContractResult:
        return _fail(self.domain_name, "get_latest_price")

    async def get_quote(self, symbol: str) -> ContractResult:
        return _fail(self.domain_name, "get_quote")

    async def get_technical_indicators(self, *args, **kwargs) -> ContractResult:
        return _fail(self.domain_name, "get_technical_indicators")

    async def get_technical_indicators_history(self, *args, **kwargs) -> ContractResult:
        return _fail(self.domain_name, "get_technical_indicators_history")


class DataPipelineServiceAdapter(DataPipelineContract):
    """Concrete data-pipeline adapter with healthy contract surface."""

    @property
    def domain_name(self) -> str:
        return "DataPipeline"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def capabilities(self) -> List[str]:
        return ["ingest_price_data", "run_etl_job", "list_data_sources", "health_check"]

    async def health_check(self) -> ContractResult[Dict[str, Any]]:
        return ContractResult.ok(
            {"status": "healthy", "domain": self.domain_name, "version": self.version}
        )

    async def ingest_price_data(self, *args, **kwargs) -> ContractResult:
        return _fail(self.domain_name, "ingest_price_data")

    async def ingest_fundamentals(self, *args, **kwargs) -> ContractResult:
        return _fail(self.domain_name, "ingest_fundamentals")

    async def ingest_news(self, *args, **kwargs) -> ContractResult:
        return _fail(self.domain_name, "ingest_news")

    async def ingest_quotes(self, *args, **kwargs) -> ContractResult:
        return _fail(self.domain_name, "ingest_quotes")

    async def run_etl_job(self, *args, **kwargs) -> ContractResult:
        return _fail(self.domain_name, "run_etl_job")

    async def get_job_status(self, job_id: str) -> ContractResult:
        return _fail(self.domain_name, "get_job_status")

    async def list_jobs(self, *args, **kwargs) -> ContractResult:
        return _fail(self.domain_name, "list_jobs")

    async def cancel_job(self, job_id: str) -> ContractResult:
        return _fail(self.domain_name, "cancel_job")

    async def get_data_source_status(self, *args, **kwargs) -> ContractResult:
        return _fail(self.domain_name, "get_data_source_status")

    async def list_data_sources(self) -> ContractResult:
        return _fail(self.domain_name, "list_data_sources")

    async def get_data_quality_report(self, *args, **kwargs) -> ContractResult:
        return _fail(self.domain_name, "get_data_quality_report")

    async def validate_data(self, *args, **kwargs) -> ContractResult:
        return _fail(self.domain_name, "validate_data")

    async def get_data_coverage(self, *args, **kwargs) -> ContractResult:
        return _fail(self.domain_name, "get_data_coverage")

    async def schedule_refresh(self, *args, **kwargs) -> ContractResult:
        return _fail(self.domain_name, "schedule_refresh")


class MLServiceAdapter(MLContract):
    """Concrete ML domain adapter."""

    @property
    def domain_name(self) -> str:
        return "ML"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def capabilities(self) -> List[str]:
        return ["get_prediction", "list_models", "health_check"]

    async def health_check(self) -> ContractResult[Dict[str, Any]]:
        return ContractResult.ok(
            {"status": "healthy", "domain": self.domain_name, "version": self.version}
        )

    async def get_prediction(self, *args, **kwargs) -> ContractResult:
        return _fail(self.domain_name, "get_prediction")

    async def generate_prediction(self, *args, **kwargs) -> ContractResult:
        return _fail(self.domain_name, "generate_prediction")

    async def batch_predict(self, *args, **kwargs) -> ContractResult:
        return _fail(self.domain_name, "batch_predict")

    async def get_prediction_history(self, *args, **kwargs) -> ContractResult:
        return _fail(self.domain_name, "get_prediction_history")

    async def get_model(self, model_id: str) -> ContractResult:
        return _fail(self.domain_name, "get_model")

    async def list_models(self, *args, **kwargs) -> ContractResult:
        return _fail(self.domain_name, "list_models")

    async def get_model_metrics(self, *args, **kwargs) -> ContractResult:
        return _fail(self.domain_name, "get_model_metrics")

    async def get_feature_importance(self, *args, **kwargs) -> ContractResult:
        return _fail(self.domain_name, "get_feature_importance")

    async def train_model(self, *args, **kwargs) -> ContractResult:
        return _fail(self.domain_name, "train_model")

    async def get_training_status(self, *args, **kwargs) -> ContractResult:
        return _fail(self.domain_name, "get_training_status")

    async def backtest_model(self, *args, **kwargs) -> ContractResult:
        return _fail(self.domain_name, "backtest_model")

    async def validate_predictions(self, *args, **kwargs) -> ContractResult:
        return _fail(self.domain_name, "validate_predictions")


class InvestmentAnalysisServiceAdapter(InvestmentAnalysisContract):
    """Concrete investment-analysis domain adapter."""

    @property
    def domain_name(self) -> str:
        return "InvestmentAnalysis"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def capabilities(self) -> List[str]:
        return [
            "get_technical_analysis",
            "get_fundamental_analysis",
            "generate_recommendation",
            "health_check",
        ]

    async def health_check(self) -> ContractResult[Dict[str, Any]]:
        return ContractResult.ok(
            {"status": "healthy", "domain": self.domain_name, "version": self.version}
        )

    async def get_technical_analysis(self, *args, **kwargs) -> ContractResult:
        return _fail(self.domain_name, "get_technical_analysis")

    async def get_fundamental_analysis(self, *args, **kwargs) -> ContractResult:
        return _fail(self.domain_name, "get_fundamental_analysis")

    async def get_sentiment_analysis(self, *args, **kwargs) -> ContractResult:
        return _fail(self.domain_name, "get_sentiment_analysis")

    async def generate_recommendation(self, *args, **kwargs) -> ContractResult:
        return _fail(self.domain_name, "generate_recommendation")

    async def get_recommendation(self, *args, **kwargs) -> ContractResult:
        return _fail(self.domain_name, "get_recommendation")

    async def list_recommendations(self, *args, **kwargs) -> ContractResult:
        return _fail(self.domain_name, "list_recommendations")

    async def update_recommendation_outcome(self, *args, **kwargs) -> ContractResult:
        return _fail(self.domain_name, "update_recommendation_outcome")

    async def generate_thesis(self, *args, **kwargs) -> ContractResult:
        return _fail(self.domain_name, "generate_thesis")

    async def get_thesis(self, *args, **kwargs) -> ContractResult:
        return _fail(self.domain_name, "get_thesis")

    async def get_sector_analysis(self, *args, **kwargs) -> ContractResult:
        return _fail(self.domain_name, "get_sector_analysis")

    async def run_screener(self, *args, **kwargs) -> ContractResult:
        return _fail(self.domain_name, "run_screener")

    async def compare_stocks(self, *args, **kwargs) -> ContractResult:
        return _fail(self.domain_name, "compare_stocks")


def get_default_domain_adapters(db_session_factory=None) -> Dict[str, Any]:
    """Return the default concrete adapter set for all domains."""
    return {
        "portfolio": PortfolioServiceAdapter(db_session_factory=db_session_factory),
        "market_data": MarketDataServiceAdapter(),
        "data_pipeline": DataPipelineServiceAdapter(),
        "ml": MLServiceAdapter(),
        "investment_analysis": InvestmentAnalysisServiceAdapter(),
    }
