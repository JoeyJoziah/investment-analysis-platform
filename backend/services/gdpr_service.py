"""
GDPR Service
Business logic for GDPR compliance operations including data export,
deletion, consent management, data retention, anonymization, and audit trails.
"""

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.compliance.gdpr import (
    ConsentType,
    consent_manager,
    data_deletion,
    data_portability,
    retention_manager,
)
from backend.models.unified_models import (
    Alert,
    AuditLog,
    Portfolio,
    Transaction,
    User,
    UserSession,
    Watchlist,
)
from backend.security.audit_logging import get_audit_logger
from backend.utils.data_anonymization import data_anonymizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Consent type mapping (shared across consent operations)
# ---------------------------------------------------------------------------

CONSENT_TYPE_MAP: Dict[str, ConsentType] = {
    "data_processing": ConsentType.DATA_PROCESSING,
    "marketing": ConsentType.MARKETING,
    "analytics": ConsentType.ANALYTICS,
    "third_party_sharing": ConsentType.THIRD_PARTY_SHARING,
    "profiling": ConsentType.PROFILING,
    "automated_decisions": ConsentType.AUTOMATED_DECISIONS,
}


# ---------------------------------------------------------------------------
# Data Export (GDPR Articles 15 & 20)
# ---------------------------------------------------------------------------

async def export_user_data(
    user_id: int,
    session: AsyncSession,
    include_categories: Optional[List[str]] = None,
) -> Any:
    """
    Export all personal data for a user.

    Delegates to the data_portability service to produce a structured export
    covering all personal data categories held for the user.

    Args:
        user_id: ID of the user whose data is being exported.
        session: Async database session.
        include_categories: Optional list of data categories to include.

    Returns:
        Export result object with export_id, categories, record_counts, and data.
    """
    logger.info(f"Data export requested for user {user_id}")
    return await data_portability.export_user_data(
        user_id=user_id,
        session=session,
        include_categories=include_categories,
    )


# ---------------------------------------------------------------------------
# Data Deletion (GDPR Article 17)
# ---------------------------------------------------------------------------

async def request_deletion(
    user_id: int,
    session: AsyncSession,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Initiate an account deletion request for a user.

    Creates a pending deletion record. Financial transaction data is
    anonymised rather than deleted to satisfy SEC 7-year audit requirements.

    Args:
        user_id: ID of the user requesting deletion.
        session: Async database session.
        reason: Optional reason provided by the user.

    Returns:
        Dictionary containing request_id, status, message, and
        estimated_completion timestamp.
    """
    logger.info(f"Deletion request initiated for user {user_id}")
    return await data_deletion.request_deletion(
        user_id=user_id,
        reason=reason,
        session=session,
    )


async def process_deletion(request_id: str, session: AsyncSession) -> Dict[str, Any]:
    """
    Process a pending deletion request.

    Executes the anonymisation / deletion steps for the given request.

    Args:
        request_id: Identifier of the pending deletion request.
        session: Async database session.

    Returns:
        Dictionary with completion details, deleted/anonymised record counts,
        and retained compliance data keys.

    Raises:
        ValueError: If the request_id does not correspond to a known request.
    """
    return await data_deletion.process_deletion(
        request_id=request_id,
        session=session,
    )


def get_deletion_audit(request_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve the audit trail for a specific deletion request.

    Args:
        request_id: Identifier of the deletion request.

    Returns:
        Dictionary describing the deletion audit, or None if not found.
    """
    return data_deletion.get_deletion_audit(request_id)


# ---------------------------------------------------------------------------
# Consent Management (GDPR Article 7)
# ---------------------------------------------------------------------------

def resolve_consent_type(consent_type_str: str) -> Optional[ConsentType]:
    """
    Map a consent type string to the ConsentType enum value.

    Args:
        consent_type_str: String representation of the consent type.

    Returns:
        Matching ConsentType enum member, or None if the string is unrecognised.
    """
    return CONSENT_TYPE_MAP.get(consent_type_str)


async def get_consent_status(
    user_id: int,
    session: AsyncSession,
) -> Dict[str, Any]:
    """
    Retrieve the current consent status for all consent types for a user.

    Args:
        user_id: ID of the user.
        session: Async database session.

    Returns:
        Dictionary mapping each consent type to its grant status and metadata.
    """
    logger.info(f"Consent status requested for user {user_id}")
    return await consent_manager.get_consent_status(user_id=user_id, session=session)


def derive_last_updated(status_data: Dict[str, Any]) -> Optional[datetime]:
    """
    Determine the most recent consent update timestamp from consent status data.

    Args:
        status_data: Dictionary as returned by get_consent_status.

    Returns:
        Most recent consent datetime, or None if no timestamps are present.
    """
    last_updated: Optional[datetime] = None
    for info in status_data.values():
        if info.get("consent_date"):
            consent_date = datetime.fromisoformat(info["consent_date"])
            if last_updated is None or consent_date > last_updated:
                last_updated = consent_date
    return last_updated


async def get_consent_history(
    user_id: int,
    session: AsyncSession,
) -> List[Dict[str, Any]]:
    """
    Retrieve the complete consent history for a user.

    Args:
        user_id: ID of the user.
        session: Async database session.

    Returns:
        List of consent history entries sorted by date.
    """
    logger.info(f"Consent history requested for user {user_id}")
    return await consent_manager.get_consent_history(user_id=user_id, session=session)


async def record_consent(
    user_id: int,
    consent_type: ConsentType,
    consent_given: bool,
    legal_basis: str,
    ip_address: Optional[str],
    user_agent: Optional[str],
    session: AsyncSession,
) -> str:
    """
    Record a consent grant or denial for a specific purpose.

    Args:
        user_id: ID of the user.
        consent_type: Enum value for the consent category.
        consent_given: True if consent is granted, False if denied.
        legal_basis: Legal basis string (e.g. "explicit_consent").
        ip_address: Anonymised IP address of the request.
        user_agent: User-Agent header value.
        session: Async database session.

    Returns:
        String identifier for the created consent record.
    """
    logger.info(f"Recording consent for user {user_id}: {consent_type}")
    return await consent_manager.record_consent(
        user_id=user_id,
        consent_type=consent_type,
        consent_given=consent_given,
        legal_basis=legal_basis,
        ip_address=ip_address,
        user_agent=user_agent,
        session=session,
    )


async def withdraw_consent(
    user_id: int,
    consent_type: ConsentType,
    ip_address: Optional[str],
    session: AsyncSession,
) -> str:
    """
    Withdraw a previously granted consent.

    Records the withdrawal with the current IP address for audit purposes.

    Args:
        user_id: ID of the user.
        consent_type: Enum value for the consent category being withdrawn.
        ip_address: Raw (pre-anonymisation) IP address of the request.
        session: Async database session.

    Returns:
        String identifier for the withdrawal consent record.
    """
    logger.info(f"Withdrawing consent for user {user_id}: {consent_type}")
    return await consent_manager.withdraw_consent(
        user_id=user_id,
        consent_type=consent_type,
        ip_address=ip_address,
        session=session,
    )


async def check_consent(
    user_id: int,
    consent_type: Optional[ConsentType],
    session: AsyncSession,
) -> bool:
    """
    Check whether a user has active consent for a specific purpose.

    Args:
        user_id: ID of the user.
        consent_type: Enum value for the consent category to check.
        session: Async database session.

    Returns:
        True if consent is currently granted, False otherwise.
    """
    return await consent_manager.check_consent(
        user_id=user_id,
        consent_type=consent_type,
        session=session,
    )


# ---------------------------------------------------------------------------
# Data Retention
# ---------------------------------------------------------------------------

async def get_retention_report(
    user_id: int,
    session: AsyncSession,
) -> Dict[str, Any]:
    """
    Generate a data retention report for a user.

    Returns information about each data category and its applicable
    retention period under GDPR and financial regulatory requirements.

    Args:
        user_id: ID of the user.
        session: Async database session.

    Returns:
        Dictionary containing user_id, report_date, and categories.
    """
    return await retention_manager.get_retention_report(
        user_id=user_id,
        session=session,
    )


# ---------------------------------------------------------------------------
# Anonymization (GDPR Article 17 - Alternative to deletion)
# ---------------------------------------------------------------------------

async def anonymize_user_data(
    user_id: int,
    reason: Optional[str],
    session: AsyncSession,
) -> Dict[str, Any]:
    """
    Anonymize all personal data for a user while retaining records for compliance.

    Replaces personally identifiable information with anonymised values.
    Transaction and audit data are retained to satisfy SEC 7-year requirements.
    Non-critical records (sessions, watchlists, alerts) are deleted outright.

    Args:
        user_id: ID of the user whose data is to be anonymised.
        reason: Optional reason provided by the user.
        session: Async database session.

    Returns:
        Dictionary with:
            - request_id: Unique identifier for this anonymisation run.
            - anonymized_counts: Record counts per category.
    """
    anon_id = hashlib.sha256(str(user_id).encode()).hexdigest()[:12]
    anonymized_counts: Dict[str, int] = {}

    # Anonymize user profile
    await session.execute(
        update(User)
        .where(User.id == user_id)
        .values(
            email=f"deleted_{anon_id}@anonymized.local",
            username=f"deleted_{anon_id}",
            full_name=f"Anonymized User {anon_id}",
            phone_number=None,
            preferences={},
            notification_settings={},
        )
    )
    await session.commit()
    anonymized_counts["profile"] = 1

    # Anonymize portfolios and linked transactions
    result = await session.execute(
        select(Portfolio.id).where(Portfolio.user_id == user_id)
    )
    portfolio_ids = [row[0] for row in result.fetchall()]

    if portfolio_ids:
        await session.execute(
            update(Portfolio)
            .where(Portfolio.user_id == user_id)
            .values(name=f"Anonymized_{anon_id}", description=None, is_public=False)
        )
        anonymized_counts["portfolios"] = len(portfolio_ids)

        # Clear transaction notes (retain financial data for regulatory compliance)
        tx_result = await session.execute(
            update(Transaction)
            .where(Transaction.portfolio_id.in_(portfolio_ids))
            .values(notes=None)
        )
        anonymized_counts["transactions"] = tx_result.rowcount

    # Delete non-critical records
    session_result = await session.execute(
        delete(UserSession).where(UserSession.user_id == user_id)
    )
    anonymized_counts["sessions_deleted"] = session_result.rowcount

    watchlist_result = await session.execute(
        delete(Watchlist).where(Watchlist.user_id == user_id)
    )
    anonymized_counts["watchlists_deleted"] = watchlist_result.rowcount

    alert_result = await session.execute(
        delete(Alert).where(Alert.user_id == user_id)
    )
    anonymized_counts["alerts_deleted"] = alert_result.rowcount

    await session.commit()

    request_id = f"anon_{anon_id}_{int(datetime.now(timezone.utc).timestamp())}"

    # Log to audit trail
    audit_logger = get_audit_logger()
    await audit_logger.log_gdpr_request(
        request_type="data_anonymization",
        user_id=str(user_id),
        details={
            "request_id": request_id,
            "anonymized_records": anonymized_counts,
            "reason": reason,
        },
    )

    return {
        "request_id": request_id,
        "anonymized_counts": anonymized_counts,
    }


# ---------------------------------------------------------------------------
# Audit Trail (GDPR Article 30)
# ---------------------------------------------------------------------------

async def get_audit_trail(
    user_id: int,
    skip: int,
    limit: int,
    session: AsyncSession,
) -> Dict[str, Any]:
    """
    Retrieve a paginated audit trail of data processing activities for a user.

    Covers data access, modifications, consent changes, and account actions
    as required by GDPR Article 30 (Records of processing activities).

    Args:
        user_id: ID of the user.
        skip: Number of entries to skip (for pagination).
        limit: Maximum number of entries to return.
        session: Async database session.

    Returns:
        Dictionary with total_entries, entries list, page, and limit.
    """
    logger.info(
        f"Audit trail requested for user {user_id} (skip={skip}, limit={limit})"
    )

    count_result = await session.execute(
        select(func.count(AuditLog.id)).where(AuditLog.user_id == user_id)
    )
    total_entries: int = count_result.scalar() or 0

    result = await session.execute(
        select(AuditLog)
        .where(AuditLog.user_id == user_id)
        .order_by(AuditLog.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    audit_logs = result.scalars().all()

    entries = [
        {
            "id": log.id,
            "action": log.action,
            "resource_type": log.resource_type or "unknown",
            "resource_id": log.resource_id,
            "ip_address": log.ip_address,
            "user_agent": log.user_agent,
            "meta_data": log.meta_data,
            "created_at": log.created_at,
        }
        for log in audit_logs
    ]

    page = (skip // limit) + 1 if limit > 0 else 1

    return {
        "total_entries": total_entries,
        "entries": entries,
        "page": page,
        "limit": limit,
    }


# ---------------------------------------------------------------------------
# IP Anonymization helper
# ---------------------------------------------------------------------------

def anonymize_ip(raw_ip: Optional[str]) -> Optional[str]:
    """
    Anonymise an IP address for GDPR-compliant storage.

    Args:
        raw_ip: Raw IP address string, or None.

    Returns:
        Anonymised IP string, or None if input was None.
    """
    return data_anonymizer.anonymize_ip(raw_ip) if raw_ip else None
