"""
Domain-Driven Design (DDD) Module for Investment Analysis Platform.

This module implements Domain-Driven Design patterns with clear bounded contexts
and well-defined contracts between domains.

Domains:
    1. Market Data: Stock, price, exchange, sector, and industry data
    2. Portfolio: User accounts, portfolios, positions, and transactions
    3. Data Pipeline: ETL, data ingestion, and quality management
    4. ML/Prediction: Machine learning models and predictions
    5. Investment Analysis: Technical, fundamental analysis, and recommendations

Architecture:
    - Each domain has a well-defined contract (interface)
    - Domains communicate through contract DTOs (Data Transfer Objects)
    - Cross-domain dependencies are explicitly declared
    - Contract results provide consistent error handling
"""

from .contracts import (
    DomainContract,
    ContractResult,
    ContractError,
    MarketDataContract,
    PortfolioContract,
    DataPipelineContract,
    MLContract,
    InvestmentAnalysisContract,
)

__all__ = [
    "DomainContract",
    "ContractResult",
    "ContractError",
    "MarketDataContract",
    "PortfolioContract",
    "DataPipelineContract",
    "MLContract",
    "InvestmentAnalysisContract",
]
