"""
GDPR Data Portability - Article 20

Implements GDPR Article 20 (Right to Data Portability): users have the right
to receive their personal data in a structured, commonly used, and
machine-readable format.
"""

import csv
import io
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.compliance.gdpr_models import DataExportResult
from backend.config.database import get_db_session
from backend.models.unified_models import (
    Alert,
    AuditLog,
    Order,
    Portfolio,
    Position,
    Recommendation,
    Transaction,
    User,
    UserSession,
    Watchlist,
)
from backend.security.audit_logging import get_audit_logger

logger = logging.getLogger(__name__)


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
                    "export_date": datetime.now(timezone.utc).isoformat(),
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
                export_date=datetime.now(timezone.utc),
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
