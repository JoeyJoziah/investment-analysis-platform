"""
Concrete domain contract implementations (Wave 9 / #109).

Adapters wrap existing platform services/repositories so DDD contracts
are no longer abstract-only stubs.
"""

from .adapters import (
    DataPipelineServiceAdapter,
    InvestmentAnalysisServiceAdapter,
    MarketDataServiceAdapter,
    MLServiceAdapter,
    PortfolioServiceAdapter,
    get_default_domain_adapters,
)

__all__ = [
    "PortfolioServiceAdapter",
    "MarketDataServiceAdapter",
    "DataPipelineServiceAdapter",
    "MLServiceAdapter",
    "InvestmentAnalysisServiceAdapter",
    "get_default_domain_adapters",
]
