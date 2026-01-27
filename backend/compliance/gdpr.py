"""
GDPR Compliance Services

Implements GDPR requirements including:
- Right to Data Portability (Article 20)
- Right to Erasure / Right to be Forgotten (Article 17)
- Consent Management (Article 7)
- Data Breach Notification (Articles 33-34)
- Data Subject Access Requests (DSAR)
- Data Retention Policy Enforcement
"""

import logging
import json
import csv
import io
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

from sqlalchemy import select, update, delete, func, and_, or_, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.config.database import get_db_session
from backend.models.unified_models import (
    User, Portfolio, Position, Transaction, Watchlist,
    Alert, Order, UserSession, AuditLog, Recommendation
)
from backend.security.audit_logging import (
    get_audit_logger, AuditEventType, AuditSeverity, ComplianceFramework
)

logger = logging.getLogger(__name__)


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


class GDPRDataPortability:
    """
    Implements GDPR Article 20 - Right to Data Portability

    Users have the right to receive their personal data in a structured,
    commonly used, and machine-readable format.
    """

    def __init__(self):
        self._data_categories = [
            "profile", "portfolios", "positions", "transactions",
            "orders", "watchlists", "alerts", "recommendations",
            "preferences", "consent_records", "sessions"
        ]

    async def export_user_data(
        self,
        user_id: int,
        session: Optional[AsyncSession] = None,
        include_categories: Optional[List[str]] = None
    ) -> DataExportResult:
        """
        Export all user data in a structured format.

        Args:
            user_id: The user's ID
            session: Optional database session
            include_categories: Categories to include (None = all)

        Returns:
            DataExportResult with all user data
        """
        export_id = str(uuid.uuid4())
        categories_to_export = include_categories or self._data_categories

        logger.info(f"Starting data export {export_id} for user {user_id}")

        async def _export(session: AsyncSession) -> DataExportResult:
            exported_data = {
                "export_metadata": {
                    "export_id": export_id,
                    "user_id": user_id,
                    "export_date": datetime.utcnow().isoformat(),
                    "format_version": "2.0",
                    "gdpr_article": "Article 20 - Right to Data Portability",
                    "categories_exported": categories_to_export
                }
            }
            record_counts = {}

            # Export each data category
            for category in categories_to_export:
                try:
                    data, count = await self._export_category(
                        session, user_id, category
                    )
                    exported_data[category] = data
                    record_counts[category] = count
                except Exception as e:
                    logger.error(
                        f"Error exporting {category} for user {user_id}: {e}"
                    )
                    exported_data[category] = {"error": str(e), "data": []}
                    record_counts[category] = 0

            # Log the export action
            audit_logger = get_audit_logger()
            await audit_logger.log_gdpr_request(
                request_type="data_request",
                user_id=str(user_id),
                details={
                    "export_id": export_id,
                    "categories": categories_to_export,
                    "record_counts": record_counts
                }
            )

            return DataExportResult(
                export_id=export_id,
                user_id=user_id,
                export_date=datetime.utcnow(),
                categories=categories_to_export,
                record_counts=record_counts,
                data=exported_data
            )

        if session:
            return await _export(session)
        else:
            async with get_db_session(readonly=True) as session:
                return await _export(session)

    async def _export_category(
        self,
        session: AsyncSession,
        user_id: int,
        category: str
    ) -> Tuple[Dict[str, Any], int]:
        """Export a specific data category for a user"""

        if category == "profile":
            return await self._export_profile(session, user_id)
        elif category == "portfolios":
            return await self._export_portfolios(session, user_id)
        elif category == "positions":
            return await self._export_positions(session, user_id)
        elif category == "transactions":
            return await self._export_transactions(session, user_id)
        elif category == "orders":
            return await self._export_orders(session, user_id)
        elif category == "watchlists":
            return await self._export_watchlists(session, user_id)
        elif category == "alerts":
            return await self._export_alerts(session, user_id)
        elif category == "recommendations":
            return await self._export_recommendations(session, user_id)
        elif category == "preferences":
            return await self._export_preferences(session, user_id)
        elif category == "consent_records":
            return await self._export_consent_records(session, user_id)
        elif category == "sessions":
            return await self._export_sessions(session, user_id)
        else:
            return {"category": category, "data": []}, 0

    async def _export_profile(
        self,
        session: AsyncSession,
        user_id: int
    ) -> Tuple[Dict[str, Any], int]:
        """Export user profile data"""
        result = await session.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()

        if not user:
            return {"data": None}, 0

        profile_data = {
            "user_id": user.id,
            "email": user.email,
            "username": user.username,
            "full_name": user.full_name,
            "role": user.role,
            "phone_number": user.phone_number,
            "country": user.country,
            "timezone": user.timezone,
            "is_active": user.is_active,
            "is_verified": user.is_verified,
            "two_factor_enabled": user.two_factor_enabled,
            "subscription_tier": user.subscription_tier,
            "subscription_end_date": (
                user.subscription_end_date.isoformat()
                if user.subscription_end_date else None
            ),
            "risk_tolerance": user.risk_tolerance,
            "investment_style": user.investment_style,
            "preferred_sectors": user.preferred_sectors,
            "excluded_sectors": user.excluded_sectors,
            "last_login": (
                user.last_login.isoformat() if user.last_login else None
            ),
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "updated_at": user.updated_at.isoformat() if user.updated_at else None
        }

        return {"data": profile_data}, 1

    async def _export_portfolios(
        self,
        session: AsyncSession,
        user_id: int
    ) -> Tuple[Dict[str, Any], int]:
        """Export portfolio data"""
        result = await session.execute(
            select(Portfolio)
            .where(Portfolio.user_id == user_id)
            .options(selectinload(Portfolio.positions))
        )
        portfolios = result.scalars().all()

        portfolio_data = []
        for portfolio in portfolios:
            portfolio_data.append({
                "id": portfolio.id,
                "portfolio_id": portfolio.portfolio_id,
                "name": portfolio.name,
                "description": portfolio.description,
                "is_public": portfolio.is_public,
                "is_default": portfolio.is_default,
                "benchmark": portfolio.benchmark,
                "total_value": float(portfolio.total_value) if portfolio.total_value else 0,
                "cash_balance": float(portfolio.cash_balance) if portfolio.cash_balance else 0,
                "total_return": float(portfolio.total_return) if portfolio.total_return else 0,
                "total_return_pct": portfolio.total_return_pct,
                "created_at": portfolio.created_at.isoformat() if portfolio.created_at else None,
                "positions_count": len(portfolio.positions) if portfolio.positions else 0
            })

        return {"data": portfolio_data}, len(portfolio_data)

    async def _export_positions(
        self,
        session: AsyncSession,
        user_id: int
    ) -> Tuple[Dict[str, Any], int]:
        """Export position data"""
        result = await session.execute(
            select(Position)
            .join(Portfolio)
            .where(Portfolio.user_id == user_id)
        )
        positions = result.scalars().all()

        position_data = []
        for pos in positions:
            position_data.append({
                "id": pos.id,
                "portfolio_id": pos.portfolio_id,
                "stock_id": pos.stock_id,
                "quantity": float(pos.quantity) if pos.quantity else 0,
                "avg_cost_basis": float(pos.avg_cost_basis) if pos.avg_cost_basis else 0,
                "current_price": float(pos.current_price) if pos.current_price else 0,
                "market_value": float(pos.market_value) if pos.market_value else 0,
                "unrealized_gain_loss": (
                    float(pos.unrealized_gain_loss) if pos.unrealized_gain_loss else 0
                ),
                "unrealized_gain_loss_pct": pos.unrealized_gain_loss_pct,
                "first_purchase_date": (
                    pos.first_purchase_date.isoformat()
                    if pos.first_purchase_date else None
                ),
                "last_transaction_date": (
                    pos.last_transaction_date.isoformat()
                    if pos.last_transaction_date else None
                )
            })

        return {"data": position_data}, len(position_data)

    async def _export_transactions(
        self,
        session: AsyncSession,
        user_id: int
    ) -> Tuple[Dict[str, Any], int]:
        """Export transaction history"""
        result = await session.execute(
            select(Transaction)
            .join(Portfolio)
            .where(Portfolio.user_id == user_id)
            .order_by(Transaction.trade_date.desc())
            .limit(10000)  # Limit for performance
        )
        transactions = result.scalars().all()

        transaction_data = []
        for txn in transactions:
            transaction_data.append({
                "id": txn.id,
                "transaction_id": txn.transaction_id,
                "portfolio_id": txn.portfolio_id,
                "stock_id": txn.stock_id,
                "transaction_type": txn.transaction_type,
                "quantity": float(txn.quantity) if txn.quantity else 0,
                "price": float(txn.price) if txn.price else 0,
                "total_amount": float(txn.total_amount) if txn.total_amount else 0,
                "commission": float(txn.commission) if txn.commission else 0,
                "fees": float(txn.fees) if txn.fees else 0,
                "trade_date": txn.trade_date.isoformat() if txn.trade_date else None,
                "settlement_date": (
                    txn.settlement_date.isoformat() if txn.settlement_date else None
                ),
                "notes": txn.notes
            })

        return {"data": transaction_data}, len(transaction_data)

    async def _export_orders(
        self,
        session: AsyncSession,
        user_id: int
    ) -> Tuple[Dict[str, Any], int]:
        """Export order history"""
        result = await session.execute(
            select(Order)
            .where(Order.user_id == user_id)
            .order_by(Order.created_at.desc())
            .limit(10000)
        )
        orders = result.scalars().all()

        order_data = []
        for order in orders:
            order_data.append({
                "id": order.id,
                "order_id": order.order_id,
                "stock_id": order.stock_id,
                "order_type": order.order_type,
                "order_side": order.order_side,
                "quantity": float(order.quantity) if order.quantity else 0,
                "limit_price": float(order.limit_price) if order.limit_price else None,
                "stop_price": float(order.stop_price) if order.stop_price else None,
                "status": order.status,
                "filled_quantity": (
                    float(order.filled_quantity) if order.filled_quantity else 0
                ),
                "average_fill_price": (
                    float(order.average_fill_price) if order.average_fill_price else None
                ),
                "created_at": order.created_at.isoformat() if order.created_at else None,
                "filled_at": order.filled_at.isoformat() if order.filled_at else None
            })

        return {"data": order_data}, len(order_data)

    async def _export_watchlists(
        self,
        session: AsyncSession,
        user_id: int
    ) -> Tuple[Dict[str, Any], int]:
        """Export watchlist data"""
        result = await session.execute(
            select(Watchlist).where(Watchlist.user_id == user_id)
        )
        watchlists = result.scalars().all()

        watchlist_data = []
        for wl in watchlists:
            watchlist_data.append({
                "id": wl.id,
                "stock_id": wl.stock_id,
                "name": wl.name,
                "notes": wl.notes,
                "tags": wl.tags,
                "priority": wl.priority,
                "target_price": float(wl.target_price) if wl.target_price else None,
                "stop_loss": float(wl.stop_loss) if wl.stop_loss else None,
                "added_date": wl.added_date.isoformat() if wl.added_date else None
            })

        return {"data": watchlist_data}, len(watchlist_data)

    async def _export_alerts(
        self,
        session: AsyncSession,
        user_id: int
    ) -> Tuple[Dict[str, Any], int]:
        """Export alert configurations"""
        result = await session.execute(
            select(Alert).where(Alert.user_id == user_id)
        )
        alerts = result.scalars().all()

        alert_data = []
        for alert in alerts:
            alert_data.append({
                "id": alert.id,
                "alert_id": alert.alert_id,
                "stock_id": alert.stock_id,
                "alert_type": alert.alert_type,
                "condition": alert.condition,
                "is_active": alert.is_active,
                "triggered_count": alert.triggered_count,
                "created_at": alert.created_at.isoformat() if alert.created_at else None
            })

        return {"data": alert_data}, len(alert_data)

    async def _export_recommendations(
        self,
        session: AsyncSession,
        user_id: int
    ) -> Tuple[Dict[str, Any], int]:
        """Export personalized recommendations (via user's portfolios)"""
        # Get recommendations for stocks the user has in portfolios
        result = await session.execute(
            select(Recommendation)
            .join(Position, Recommendation.stock_id == Position.stock_id)
            .join(Portfolio)
            .where(Portfolio.user_id == user_id)
            .order_by(Recommendation.created_at.desc())
            .limit(1000)
        )
        recommendations = result.scalars().all()

        rec_data = []
        for rec in recommendations:
            rec_data.append({
                "id": rec.id,
                "recommendation_id": rec.recommendation_id,
                "stock_id": rec.stock_id,
                "action": rec.action,
                "confidence": rec.confidence,
                "entry_price": float(rec.entry_price) if rec.entry_price else None,
                "target_price": float(rec.target_price) if rec.target_price else None,
                "stop_loss": float(rec.stop_loss) if rec.stop_loss else None,
                "reasoning": rec.reasoning,
                "created_at": rec.created_at.isoformat() if rec.created_at else None
            })

        return {"data": rec_data}, len(rec_data)

    async def _export_preferences(
        self,
        session: AsyncSession,
        user_id: int
    ) -> Tuple[Dict[str, Any], int]:
        """Export user preferences"""
        result = await session.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()

        if not user:
            return {"data": {}}, 0

        preferences_data = {
            "preferences": user.preferences or {},
            "notification_settings": user.notification_settings or {},
            "risk_tolerance": user.risk_tolerance,
            "investment_style": user.investment_style,
            "preferred_sectors": user.preferred_sectors,
            "excluded_sectors": user.excluded_sectors
        }

        return {"data": preferences_data}, 1

    async def _export_consent_records(
        self,
        session: AsyncSession,
        user_id: int
    ) -> Tuple[Dict[str, Any], int]:
        """Export consent records from audit logs"""
        result = await session.execute(
            select(AuditLog)
            .where(
                and_(
                    AuditLog.user_id == user_id,
                    AuditLog.action.like('%consent%')
                )
            )
            .order_by(AuditLog.created_at.desc())
        )
        consent_logs = result.scalars().all()

        consent_data = []
        for log in consent_logs:
            consent_data.append({
                "id": log.id,
                "action": log.action,
                "details": log.meta_data,
                "ip_address": log.ip_address,
                "created_at": log.created_at.isoformat() if log.created_at else None
            })

        return {"data": consent_data}, len(consent_data)

    async def _export_sessions(
        self,
        session: AsyncSession,
        user_id: int
    ) -> Tuple[Dict[str, Any], int]:
        """Export session history"""
        result = await session.execute(
            select(UserSession)
            .where(UserSession.user_id == user_id)
            .order_by(UserSession.created_at.desc())
            .limit(1000)
        )
        sessions = result.scalars().all()

        session_data = []
        for sess in sessions:
            session_data.append({
                "id": sess.id,
                "ip_address": sess.ip_address,
                "user_agent": sess.user_agent,
                "location": sess.location,
                "is_active": sess.is_active,
                "created_at": sess.created_at.isoformat() if sess.created_at else None,
                "last_activity": (
                    sess.last_activity.isoformat() if sess.last_activity else None
                )
            })

        return {"data": session_data}, len(session_data)

    def to_json(self, result: DataExportResult) -> str:
        """Convert exported data to JSON format"""
        return json.dumps(result.data, indent=2, default=str)

    def to_csv(self, result: DataExportResult) -> Dict[str, str]:
        """Convert exported data to CSV format (one file per category)"""
        csv_files = {}

        for category, category_data in result.data.items():
            if category == "export_metadata":
                continue

            if isinstance(category_data, dict) and "data" in category_data:
                records = category_data.get("data", [])

                if isinstance(records, list) and records:
                    output = io.StringIO()

                    # Handle list of dicts
                    if isinstance(records[0], dict):
                        writer = csv.DictWriter(
                            output,
                            fieldnames=records[0].keys()
                        )
                        writer.writeheader()
                        writer.writerows(records)

                    csv_files[category] = output.getvalue()
                elif isinstance(records, dict):
                    # Single record
                    output = io.StringIO()
                    writer = csv.DictWriter(output, fieldnames=records.keys())
                    writer.writeheader()
                    writer.writerow(records)
                    csv_files[category] = output.getvalue()

        return csv_files


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
            request_date=datetime.utcnow()
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
                datetime.utcnow() + timedelta(days=30)
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
                request.completion_date = datetime.utcnow()
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
                    "timestamp": datetime.utcnow().isoformat()
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
            session_cutoff = datetime.utcnow() - timedelta(
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
                "report_date": datetime.utcnow().isoformat(),
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
            "reported_at": datetime.utcnow().isoformat(),
            "breach_type": breach_details.get("breach_type", "unknown"),
            "affected_records": breach_details.get("affected_records", 0),
            "data_categories": breach_details.get("data_categories", []),
            "discovery_date": (
                breach_details.get("discovery_date", datetime.utcnow()).isoformat()
                if isinstance(breach_details.get("discovery_date"), datetime)
                else breach_details.get("discovery_date")
            ),
            "containment_measures": breach_details.get("containment_measures", ""),
            "notification_deadline": (
                datetime.utcnow() + timedelta(hours=72)
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
            "generated_at": datetime.utcnow().isoformat(),
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


# Module-level service instances
data_portability = GDPRDataPortability()
data_deletion = GDPRDataDeletion()
consent_manager = ConsentManager()
retention_manager = DataRetentionManager()
breach_notification = DataBreachNotification()
