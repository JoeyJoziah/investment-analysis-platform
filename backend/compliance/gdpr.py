"""
GDPR Compliance Services

Implements GDPR requirements including:
- Right to Data Portability (Article 20)
- Right to Erasure / Right to be Forgotten (Article 17)
- Consent Management (Article 7)
- Data Breach Notification (Articles 33-34)
- Data Subject Access Requests (DSAR)
- Data Retention Policy Enforcement

This module is a thin orchestrator that re-exports all public symbols from
the extracted sub-modules. All existing import paths continue to work
without modification.

Sub-modules:
- gdpr_models.py   : Enums, constants, and dataclasses
- data_export.py   : GDPRDataPortability (Article 20)
- deletion_handler.py : GDPRDataDeletion (Article 17)
- consent_manager.py  : ConsentManager, DataRetentionManager,
                        DataBreachNotification (Articles 7, 33-34)
"""

# ---------------------------------------------------------------------------
# Third-party helpers re-exported to preserve legacy patch paths
# Tests patch "backend.compliance.gdpr.get_audit_logger" which requires
# the symbol to be a direct attribute of this module namespace.
# ---------------------------------------------------------------------------
from backend.security.audit_logging import get_audit_logger  # noqa: F401

# ---------------------------------------------------------------------------
# Shared types and constants (re-exported for backward compatibility)
# ---------------------------------------------------------------------------
from backend.compliance.gdpr_models import (  # noqa: F401
    ConsentRecord,
    ConsentType,
    DataExportResult,
    DeletionRequest,
    DeletionStatus,
    RETENTION_PERIODS,
    RetentionCategory,
)

# ---------------------------------------------------------------------------
# Core service classes (re-exported for backward compatibility)
# ---------------------------------------------------------------------------
from backend.compliance.data_export import GDPRDataPortability  # noqa: F401
from backend.compliance.deletion_handler import GDPRDataDeletion  # noqa: F401
from backend.compliance.consent_manager import (  # noqa: F401
    ConsentManager,
    DataBreachNotification,
    DataRetentionManager,
)

# ---------------------------------------------------------------------------
# Module-level service singletons
# ---------------------------------------------------------------------------
data_portability = GDPRDataPortability()
data_deletion = GDPRDataDeletion()
consent_manager = ConsentManager()
retention_manager = DataRetentionManager()
breach_notification = DataBreachNotification()
