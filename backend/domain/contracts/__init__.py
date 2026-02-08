"""
Domain Contracts for Investment Analysis Platform.

This module defines the contracts (interfaces) between different DDD domains
to ensure loose coupling and proper boundary separation.

Domains:
    - Domain 1: Market Data (Stock/Price/Exchange)
    - Domain 2: Portfolio Management (User/Portfolio/Position)
    - Domain 3: Data Pipeline (ETL/Ingestion)
    - Domain 4: ML/Prediction
    - Domain 5: Investment Analysis (Technical/Fundamental/Recommendations)
"""

from .base import DomainContract, ContractResult, ContractError
from .market_data_contract import MarketDataContract
from .portfolio_contract import PortfolioContract
from .data_pipeline_contract import DataPipelineContract
from .ml_contract import MLContract
from .investment_analysis_contract import InvestmentAnalysisContract

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
