"""
Compliance Module for SEC 2025 and GDPR Requirements

This module provides compliance services for:
- SEC 2025 Investment Adviser regulations
- GDPR data protection requirements
- Audit trail management
- Data retention policies
"""

from backend.compliance.gdpr import (
    GDPRDataPortability,
    GDPRDataDeletion,
    ConsentManager,
    DataBreachNotification,
)

from backend.compliance.sec import (
    DataRetentionManager,
    InvestmentAdviceDocumentation,
    FiduciaryDutyChecker,
)

__all__ = [
    "GDPRDataPortability",
    "GDPRDataDeletion",
    "ConsentManager",
    "DataBreachNotification",
    "DataRetentionManager",
    "InvestmentAdviceDocumentation",
    "FiduciaryDutyChecker",
]
