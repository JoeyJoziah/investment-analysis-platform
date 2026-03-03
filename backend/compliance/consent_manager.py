"""
GDPR Consent and Retention Management

Implements:
- GDPR Article 7 (Conditions for Consent) via ConsentManager
- Data retention policy enforcement via DataRetentionManager
- GDPR Articles 33-34 (Data Breach Notification) via DataBreachNotification
"""

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.compliance.gdpr_models import ConsentType, RETENTION_PERIODS, RetentionCategory
from backend.config.database import get_db_session
from backend.models.unified_models import (
    AuditLog,
    Portfolio,
    Transaction,
    UserSession,
)
from backend.security.audit_logging import get_audit_logger

logger = logging.getLogger(__name__)


class ConsentManager:
    """
    Implements GDPR Article 7 - Conditions for Consent

    Manages user consent for data processing activities:
    - Recording consent given
    - Tracking consent withdrawal
    - Maintaining consent history
    - Verifying consent status
    """

    async def record_consent(
        self,
        user_id: int,
        consent_type: ConsentType,
        consent_given: bool,
        legal_basis: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        session: Optional[AsyncSession] = None
    ) -> str:
        """
        Record a user's consent decision in the database.

        Args:
            user_id: The user's ID
            consent_type: Type of consent
            consent_given: Whether consent was given
            legal_basis: Legal basis for processing
            ip_address: Optional IP address for audit
            user_agent: Optional user agent for audit
            session: Optional database session

        Returns:
            Consent record ID
        """
        consent_id = str(uuid.uuid4())

        async def _record(session: AsyncSession) -> str:
            # Create audit log entry for consent
            audit_entry = AuditLog(
                user_id=user_id,
                action=f"gdpr_consent_{consent_type.value}",
                resource_type="consent",
                resource_id=consent_id,
                ip_address=ip_address,
                user_agent=user_agent[:500] if user_agent else None,
                meta_data={
                    "consent_id": consent_id,
                    "consent_type": consent_type.value,
                    "consent_given": consent_given,
                    "legal_basis": legal_basis,
                    "version": "1.0",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
            session.add(audit_entry)

            logger.info(
                f"Consent recorded for user {user_id}: "
                f"{consent_type.value}={consent_given}"
            )

            return consent_id

        if session:
            return await _record(session)
        else:
            async with get_db_session() as session:
                return await _record(session)

    async def get_consent_status(
        self,
        user_id: int,
        session: Optional[AsyncSession] = None
    ) -> Dict[str, Any]:
        """
        Get current consent status for a user from the database.

        Returns:
            Dictionary of consent types and their status
        """
        async def _get_status(session: AsyncSession) -> Dict[str, Any]:
            # Get latest consent records for each type
            result = await session.execute(
                select(AuditLog)
                .where(
                    and_(
                        AuditLog.user_id == user_id,
                        AuditLog.action.like('gdpr_consent_%')
                    )
                )
                .order_by(AuditLog.created_at.desc())
            )
            consent_logs = result.scalars().all()

            # Build consent status from most recent records
            consent_status = {}
            seen_types = set()

            for log in consent_logs:
                if log.meta_data and "consent_type" in log.meta_data:
                    consent_type = log.meta_data["consent_type"]
                    if consent_type not in seen_types:
                        seen_types.add(consent_type)
                        consent_status[consent_type] = {
                            "granted": log.meta_data.get("consent_given", False),
                            "consent_date": log.created_at.isoformat() if log.created_at else None,
                            "consent_id": log.meta_data.get("consent_id"),
                            "legal_basis": log.meta_data.get("legal_basis")
                        }

            return consent_status

        if session:
            return await _get_status(session)
        else:
            async with get_db_session(readonly=True) as session:
                return await _get_status(session)

    async def withdraw_consent(
        self,
        user_id: int,
        consent_type: ConsentType,
        ip_address: Optional[str] = None,
        session: Optional[AsyncSession] = None
    ) -> str:
        """Withdraw consent for a specific purpose"""
        return await self.record_consent(
            user_id=user_id,
            consent_type=consent_type,
            consent_given=False,
            legal_basis="consent_withdrawal",
            ip_address=ip_address,
            session=session
        )

    async def get_consent_history(
        self,
        user_id: int,
        session: Optional[AsyncSession] = None
    ) -> List[Dict[str, Any]]:
        """Get complete consent history for a user from the database"""
        async def _get_history(session: AsyncSession) -> List[Dict[str, Any]]:
            result = await session.execute(
                select(AuditLog)
                .where(
                    and_(
                        AuditLog.user_id == user_id,
                        AuditLog.action.like('gdpr_consent_%')
                    )
                )
                .order_by(AuditLog.created_at.desc())
            )
            consent_logs = result.scalars().all()

            history = []
            for log in consent_logs:
                if log.meta_data:
                    history.append({
                        "consent_id": log.meta_data.get("consent_id"),
                        "consent_type": log.meta_data.get("consent_type"),
                        "consent_given": log.meta_data.get("consent_given"),
                        "legal_basis": log.meta_data.get("legal_basis"),
                        "timestamp": log.created_at.isoformat() if log.created_at else None,
                        "ip_address": log.ip_address
                    })

            return history

        if session:
            return await _get_history(session)
        else:
            async with get_db_session(readonly=True) as session:
                return await _get_history(session)

    async def check_consent(
        self,
        user_id: int,
        consent_type: ConsentType,
        session: Optional[AsyncSession] = None
    ) -> bool:
        """Check if user has valid consent for a specific purpose"""
        status = await self.get_consent_status(user_id, session)
        consent_info = status.get(consent_type.value, {})
        return consent_info.get("granted", False)


class DataRetentionManager:
    """
    Manages data retention policies for GDPR compliance.

    Enforces retention periods and automatic data cleanup.
    """

    async def enforce_retention_policies(
        self,
        session: Optional[AsyncSession] = None
    ) -> Dict[str, int]:
        """
        Enforce data retention policies by cleaning up expired data.

        Returns:
            Dictionary with counts of records cleaned up per category
        """
        cleanup_results = {}

        async def _enforce(session: AsyncSession) -> Dict[str, int]:
            # Clean up expired sessions
            session_cutoff = datetime.now(timezone.utc) - timedelta(
                days=RETENTION_PERIODS[RetentionCategory.SESSION_DATA]
            )
            result = await session.execute(
                delete(UserSession)
                .where(UserSession.created_at < session_cutoff)
            )
            cleanup_results["sessions"] = result.rowcount

            logger.info(f"Retention policy cleanup completed: {cleanup_results}")
            return cleanup_results

        if session:
            return await _enforce(session)
        else:
            async with get_db_session() as session:
                return await _enforce(session)

    async def get_retention_report(
        self,
        user_id: int,
        session: Optional[AsyncSession] = None
    ) -> Dict[str, Any]:
        """
        Generate a retention report for a user.

        Returns:
            Report showing data categories and their retention periods
        """
        async def _get_report(session: AsyncSession) -> Dict[str, Any]:
            report = {
                "user_id": user_id,
                "report_date": datetime.now(timezone.utc).isoformat(),
                "categories": {}
            }

            # Count records in each category
            # Transactions
            result = await session.execute(
                select(func.count(Transaction.id))
                .join(Portfolio)
                .where(Portfolio.user_id == user_id)
            )
            transaction_count = result.scalar() or 0
            report["categories"]["transactions"] = {
                "record_count": transaction_count,
                "retention_period_days": RETENTION_PERIODS[RetentionCategory.TRANSACTION_DATA],
                "reason": "SEC regulatory compliance"
            }

            # Audit logs
            result = await session.execute(
                select(func.count(AuditLog.id))
                .where(AuditLog.user_id == user_id)
            )
            audit_count = result.scalar() or 0
            report["categories"]["audit_logs"] = {
                "record_count": audit_count,
                "retention_period_days": RETENTION_PERIODS[RetentionCategory.AUDIT_LOGS],
                "reason": "Regulatory compliance and security"
            }

            # Sessions
            result = await session.execute(
                select(func.count(UserSession.id))
                .where(UserSession.user_id == user_id)
            )
            session_count = result.scalar() or 0
            report["categories"]["sessions"] = {
                "record_count": session_count,
                "retention_period_days": RETENTION_PERIODS[RetentionCategory.SESSION_DATA],
                "reason": "Security and access tracking"
            }

            return report

        if session:
            return await _get_report(session)
        else:
            async with get_db_session(readonly=True) as session:
                return await _get_report(session)


class DataBreachNotification:
    """
    Implements GDPR Articles 33-34 - Data Breach Notification

    Article 33: Notification to supervisory authority within 72 hours
    Article 34: Communication to data subjects when high risk
    """

    def __init__(self):
        self._breaches: Dict[str, Dict[str, Any]] = {}

    def report_breach(self, breach_details: Dict[str, Any]) -> str:
        """
        Report a data breach incident.

        Args:
            breach_details: Dictionary containing breach information

        Returns:
            Unique breach ID
        """
        breach_id = str(uuid.uuid4())

        breach_record = {
            "breach_id": breach_id,
            "reported_at": datetime.now(timezone.utc).isoformat(),
            "breach_type": breach_details.get("breach_type", "unknown"),
            "affected_records": breach_details.get("affected_records", 0),
            "data_categories": breach_details.get("data_categories", []),
            "discovery_date": (
                breach_details.get("discovery_date", datetime.now(timezone.utc)).isoformat()
                if isinstance(breach_details.get("discovery_date"), datetime)
                else breach_details.get("discovery_date")
            ),
            "containment_measures": breach_details.get("containment_measures", ""),
            "notification_deadline": (
                datetime.now(timezone.utc) + timedelta(hours=72)
            ).isoformat(),
            "status": "reported"
        }

        self._breaches[breach_id] = breach_record

        logger.critical(
            f"DATA BREACH REPORTED: {breach_id} - {breach_record['breach_type']}"
        )

        return breach_id

    def is_notification_required(self, breach_id: str) -> bool:
        """
        Determine if regulatory notification is required.

        Per GDPR Article 33, notification is required unless the breach
        is unlikely to result in a risk to the rights and freedoms of
        natural persons.
        """
        if breach_id not in self._breaches:
            raise ValueError(f"Breach {breach_id} not found")

        breach = self._breaches[breach_id]

        # High risk categories for financial platform
        high_risk_categories = {
            "financial", "portfolio_data", "personal_id",
            "authentication", "transaction_data"
        }

        affected_records = breach.get("affected_records", 0)
        data_categories = set(breach.get("data_categories", []))

        if affected_records >= 500:
            return True

        if data_categories & high_risk_categories:
            return True

        return False

    def generate_regulatory_notification(
        self,
        breach_id: str
    ) -> Dict[str, Any]:
        """
        Generate the regulatory notification document.

        Per GDPR Article 33(3), the notification must describe:
        - Nature of the breach
        - Categories and approximate number of data subjects
        - Likely consequences
        - Measures taken or proposed
        """
        if breach_id not in self._breaches:
            raise ValueError(f"Breach {breach_id} not found")

        breach = self._breaches[breach_id]

        return {
            "notification_type": (
                "GDPR Article 33 - Supervisory Authority Notification"
            ),
            "breach_reference": breach_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "breach_description": (
                f"Security incident of type '{breach['breach_type']}' "
                f"discovered on {breach['discovery_date']}"
            ),
            "affected_data_subjects": (
                f"Approximately {breach['affected_records']} data subjects"
            ),
            "data_categories_affected": breach["data_categories"],
            "likely_consequences": self._assess_consequences(breach),
            "measures_taken": breach["containment_measures"],
            "dpo_contact": {
                "name": "Data Protection Officer",
                "email": "dpo@investmentplatform.com",
                "phone": "+1-XXX-XXX-XXXX"
            },
            "notification_deadline": breach["notification_deadline"]
        }

    def _assess_consequences(self, breach: Dict[str, Any]) -> List[str]:
        """Assess likely consequences of the breach"""
        consequences = []

        data_categories = breach.get("data_categories", [])

        if "email" in data_categories:
            consequences.append(
                "Potential for phishing attacks targeting affected users"
            )

        if "portfolio_data" in data_categories or "financial" in data_categories:
            consequences.append("Exposure of sensitive financial information")
            consequences.append("Potential for financial fraud or identity theft")

        if "authentication" in data_categories:
            consequences.append("Risk of unauthorized account access")
            consequences.append("Users should reset passwords immediately")

        if "transaction_data" in data_categories:
            consequences.append("Exposure of trading activity and patterns")

        if not consequences:
            consequences.append("Low risk - No sensitive personal data exposed")

        return consequences
