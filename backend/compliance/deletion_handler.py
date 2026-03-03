"""
GDPR Data Deletion - Article 17

Implements GDPR Article 17 (Right to Erasure / Right to be Forgotten):
Users have the right to have their personal data erased when:
- Data is no longer necessary for original purpose
- User withdraws consent
- User objects to processing
- Data was unlawfully processed
"""

import hashlib
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.compliance.gdpr_models import DeletionRequest, DeletionStatus
from backend.config.database import get_db_session
from backend.models.unified_models import (
    Alert,
    AuditLog,
    Order,
    Portfolio,
    Transaction,
    User,
    UserSession,
    Watchlist,
)
from backend.security.audit_logging import get_audit_logger

logger = logging.getLogger(__name__)


class GDPRDataDeletion:
    """
    Implements GDPR Article 17 - Right to Erasure (Right to be Forgotten)

    Users have the right to have their personal data erased when:
    - Data is no longer necessary for original purpose
    - User withdraws consent
    - User objects to processing
    - Data was unlawfully processed
    """

    def __init__(self):
        self._pending_requests: Dict[str, DeletionRequest] = {}
        self._completed_requests: Dict[str, DeletionRequest] = {}

    async def request_deletion(
        self,
        user_id: int,
        reason: Optional[str] = None,
        session: Optional[AsyncSession] = None
    ) -> Dict[str, Any]:
        """
        Submit a data deletion request.

        Args:
            user_id: The user's ID
            reason: Optional reason for deletion
            session: Optional database session

        Returns:
            Dictionary with request_id and status
        """
        request_id = str(uuid.uuid4())

        request = DeletionRequest(
            request_id=request_id,
            user_id=user_id,
            status=DeletionStatus.PENDING,
            request_date=datetime.now(timezone.utc)
        )

        self._pending_requests[request_id] = request

        logger.info(f"Deletion request {request_id} created for user {user_id}")

        # Log the request
        audit_logger = get_audit_logger()
        await audit_logger.log_gdpr_request(
            request_type="data_deletion",
            user_id=str(user_id),
            details={
                "request_id": request_id,
                "reason": reason,
                "status": "pending"
            }
        )

        return {
            "request_id": request_id,
            "status": "pending",
            "message": (
                "Deletion request received. Processing will begin within "
                "30 days as per GDPR requirements."
            ),
            "estimated_completion": (
                datetime.now(timezone.utc) + timedelta(days=30)
            ).isoformat()
        }

    async def process_deletion(
        self,
        request_id: str,
        session: Optional[AsyncSession] = None
    ) -> Dict[str, Any]:
        """
        Process a pending deletion request.

        Args:
            request_id: The deletion request ID
            session: Optional database session

        Returns:
            Dictionary with completion status
        """
        if request_id not in self._pending_requests:
            raise ValueError(f"Deletion request {request_id} not found")

        request = self._pending_requests[request_id]
        request.status = DeletionStatus.PROCESSING

        async def _process(session: AsyncSession) -> Dict[str, Any]:
            try:
                user_id = request.user_id
                deleted_records = {}
                retained_records = {}
                anonymized_records = {}

                # 1. Delete non-critical data
                deleted_records["sessions"] = await self._delete_sessions(
                    session, user_id
                )
                deleted_records["alerts"] = await self._delete_alerts(
                    session, user_id
                )
                deleted_records["watchlists"] = await self._delete_watchlists(
                    session, user_id
                )

                # 2. Anonymize financial data (retained for SEC compliance)
                anonymized_records["transactions"] = await self._anonymize_transactions(
                    session, user_id
                )
                anonymized_records["orders"] = await self._anonymize_orders(
                    session, user_id
                )
                anonymized_records["portfolios"] = await self._anonymize_portfolios(
                    session, user_id
                )

                # 3. Retain audit logs (required for compliance)
                retained_records["audit_logs"] = await self._retain_audit_logs(
                    session, user_id
                )

                # 4. Anonymize user profile
                await self._anonymize_user_profile(session, user_id)
                anonymized_records["profile"] = 1

                # Update request status
                request.status = DeletionStatus.COMPLETED
                request.completion_date = datetime.now(timezone.utc)
                request.deleted_records = deleted_records
                request.retained_records = retained_records
                request.anonymized_records = anonymized_records

                # Move to completed
                self._completed_requests[request_id] = request
                del self._pending_requests[request_id]

                total_deleted = sum(deleted_records.values())
                total_anonymized = sum(anonymized_records.values())

                logger.info(
                    f"Deletion request {request_id} completed. "
                    f"Deleted: {total_deleted}, Anonymized: {total_anonymized}"
                )

                # Log completion
                audit_logger = get_audit_logger()
                await audit_logger.log_gdpr_request(
                    request_type="data_deletion",
                    user_id=str(user_id),
                    details={
                        "request_id": request_id,
                        "status": "completed",
                        "deleted_records": deleted_records,
                        "anonymized_records": anonymized_records,
                        "retained_records": retained_records
                    }
                )

                return {
                    "status": "completed",
                    "request_id": request_id,
                    "deleted_records": deleted_records,
                    "anonymized_records": anonymized_records,
                    "retained_for_compliance": retained_records,
                    "completion_date": request.completion_date.isoformat()
                }

            except Exception as e:
                request.status = DeletionStatus.FAILED
                request.error_message = str(e)
                logger.error(f"Deletion request {request_id} failed: {e}")
                raise

        if session:
            return await _process(session)
        else:
            async with get_db_session() as session:
                return await _process(session)

    async def _delete_sessions(
        self,
        session: AsyncSession,
        user_id: int
    ) -> int:
        """Delete user sessions"""
        result = await session.execute(
            delete(UserSession).where(UserSession.user_id == user_id)
        )
        return result.rowcount

    async def _delete_alerts(
        self,
        session: AsyncSession,
        user_id: int
    ) -> int:
        """Delete user alerts"""
        result = await session.execute(
            delete(Alert).where(Alert.user_id == user_id)
        )
        return result.rowcount

    async def _delete_watchlists(
        self,
        session: AsyncSession,
        user_id: int
    ) -> int:
        """Delete user watchlists"""
        result = await session.execute(
            delete(Watchlist).where(Watchlist.user_id == user_id)
        )
        return result.rowcount

    async def _anonymize_transactions(
        self,
        session: AsyncSession,
        user_id: int
    ) -> int:
        """Anonymize transaction data (required for SEC compliance)"""
        # Get portfolio IDs for the user
        result = await session.execute(
            select(Portfolio.id).where(Portfolio.user_id == user_id)
        )
        portfolio_ids = [row[0] for row in result.fetchall()]

        if not portfolio_ids:
            return 0

        # Anonymize transaction notes (keep financial data for compliance)
        result = await session.execute(
            update(Transaction)
            .where(Transaction.portfolio_id.in_(portfolio_ids))
            .values(notes=None)
        )
        return result.rowcount

    async def _anonymize_orders(
        self,
        session: AsyncSession,
        user_id: int
    ) -> int:
        """Anonymize order data"""
        result = await session.execute(
            update(Order)
            .where(Order.user_id == user_id)
            .values(
                rejection_reason=None,
                error_message=None
            )
        )
        return result.rowcount

    async def _anonymize_portfolios(
        self,
        session: AsyncSession,
        user_id: int
    ) -> int:
        """Anonymize portfolio data"""
        # Generate anonymous identifier
        anon_id = hashlib.sha256(str(user_id).encode()).hexdigest()[:12]

        result = await session.execute(
            update(Portfolio)
            .where(Portfolio.user_id == user_id)
            .values(
                name=f"Anonymized_{anon_id}",
                description=None,
                is_public=False
            )
        )
        return result.rowcount

    async def _retain_audit_logs(
        self,
        session: AsyncSession,
        user_id: int
    ) -> int:
        """Count retained audit logs (not deleted for compliance)"""
        result = await session.execute(
            select(func.count(AuditLog.id))
            .where(AuditLog.user_id == user_id)
        )
        return result.scalar() or 0

    async def _anonymize_user_profile(
        self,
        session: AsyncSession,
        user_id: int
    ) -> None:
        """Anonymize user profile"""
        anon_id = hashlib.sha256(str(user_id).encode()).hexdigest()[:12]

        await session.execute(
            update(User)
            .where(User.id == user_id)
            .values(
                email=f"deleted_{anon_id}@anonymized.local",
                username=f"deleted_{anon_id}",
                full_name=f"Deleted User {anon_id}",
                hashed_password=hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest(),
                phone_number=None,
                is_active=False,
                api_key=None,
                api_secret=None,
                two_factor_secret=None,
                preferences={},
                notification_settings={}
            )
        )

    def get_deletion_audit(
        self,
        request_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get the audit record for a deletion request.
        Maintains accountability without storing deleted PII.
        """
        request = (
            self._completed_requests.get(request_id) or
            self._pending_requests.get(request_id)
        )

        if not request:
            return None

        return {
            "request_id": request.request_id,
            "status": request.status.value,
            "request_date": request.request_date.isoformat(),
            "completion_date": (
                request.completion_date.isoformat()
                if request.completion_date else None
            ),
            "deleted_records": request.deleted_records,
            "retained_records": request.retained_records,
            "anonymized_records": request.anonymized_records,
            # User ID is hashed for anonymized audit trail
            "anonymized_user_reference": hashlib.sha256(
                str(request.user_id).encode()
            ).hexdigest()[:16]
        }
