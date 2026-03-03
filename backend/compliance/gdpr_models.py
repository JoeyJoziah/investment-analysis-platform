"""
GDPR Compliance - Shared Types and Models

Enumerations, constants, and dataclasses used across the GDPR compliance
sub-modules. Extracted from gdpr.py to allow clean sub-module imports
without circular dependencies.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class ConsentType(str, Enum):
    """Types of consent required for GDPR compliance"""
    DATA_PROCESSING = "data_processing"
    MARKETING = "marketing"
    ANALYTICS = "analytics"
    THIRD_PARTY_SHARING = "third_party_sharing"
    PROFILING = "profiling"
    AUTOMATED_DECISIONS = "automated_decisions"


class DeletionStatus(str, Enum):
    """Status of deletion requests"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIALLY_COMPLETED = "partially_completed"


class RetentionCategory(str, Enum):
    """Data retention categories with different retention periods"""
    TRANSACTION_DATA = "transaction_data"  # 7 years for SEC compliance
    AUDIT_LOGS = "audit_logs"  # 7 years for compliance
    USER_PROFILE = "user_profile"  # Until deletion request
    CONSENT_RECORDS = "consent_records"  # Until deletion + 3 years
    SESSION_DATA = "session_data"  # 90 days
    ANALYTICS_DATA = "analytics_data"  # 2 years


# Retention periods in days
RETENTION_PERIODS = {
    RetentionCategory.TRANSACTION_DATA: 2555,  # 7 years
    RetentionCategory.AUDIT_LOGS: 2555,  # 7 years
    RetentionCategory.USER_PROFILE: None,  # Until deletion
    RetentionCategory.CONSENT_RECORDS: 3650,  # 10 years (deletion + 3)
    RetentionCategory.SESSION_DATA: 90,
    RetentionCategory.ANALYTICS_DATA: 730,  # 2 years
}


@dataclass
class ConsentRecord:
    """Record of user consent"""
    consent_id: str
    user_id: int
    consent_type: ConsentType
    consent_given: bool
    consent_date: datetime
    legal_basis: str
    version: str = "1.0"
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    withdrawn_date: Optional[datetime] = None


@dataclass
class DeletionRequest:
    """GDPR deletion request tracking"""
    request_id: str
    user_id: int
    status: DeletionStatus
    request_date: datetime
    completion_date: Optional[datetime] = None
    deleted_records: Dict[str, int] = field(default_factory=dict)
    retained_records: Dict[str, int] = field(default_factory=dict)
    anonymized_records: Dict[str, int] = field(default_factory=dict)
    error_message: Optional[str] = None


@dataclass
class DataExportResult:
    """Result of data export operation"""
    export_id: str
    user_id: int
    export_date: datetime
    categories: List[str]
    record_counts: Dict[str, int]
    data: Dict[str, Any]
    format: str = "json"
