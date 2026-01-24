"""
GDPR Compliance API Endpoints

Implements data subject rights under GDPR:
- Right to Access (Data Export)
- Right to Erasure (Right to be Forgotten)
- Consent Management
"""

from datetime import datetime
from typing import Optional, List, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import logging

from backend.api.routers.auth import get_current_user
from backend.models.tables import User, AuditLog, Portfolio, Transaction
from backend.utils.database import get_db_sync
from backend.utils.data_anonymization import (
    data_anonymizer,
    gdpr_compliance,
    GDPRCompliance,
    DataAnonymizer
)

router = APIRouter()
logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models
# =============================================================================

class ConsentRequest(BaseModel):
    """Request model for recording consent"""
    consent_type: Literal["data_processing", "marketing", "analytics", "third_party_sharing"]
    granted: bool = Field(..., description="Whether consent is granted")


class ConsentRecord(BaseModel):
    """Response model for consent records"""
    consent_id: str
    consent_type: str
    granted: bool
    timestamp: datetime
    ip_address: Optional[str] = None


class ConsentStatusResponse(BaseModel):
    """Response model for consent status"""
    user_id: int
    consents: List[ConsentRecord]
    last_updated: Optional[datetime] = None


class DataExportResponse(BaseModel):
    """Response model for data export"""
    user_id: int
    email: str
    full_name: str
    export_date: datetime
    profile_data: dict
    portfolio_data: List[dict]
    transaction_history: List[dict]
    consent_records: List[dict]
    preferences: dict
    data_categories: List[str]


class DeleteRequestResponse(BaseModel):
    """Response model for delete request"""
    request_id: str
    status: str
    message: str
    deletion_scheduled_at: datetime
    anonymization_complete: bool
    retained_for_compliance: List[str]


class GDPRAuditEntry(BaseModel):
    """Model for GDPR audit log entries"""
    action: str
    user_id: int
    timestamp: datetime
    ip_address: Optional[str]
    details: dict


# =============================================================================
# Helper Functions
# =============================================================================

def log_gdpr_action(
    db: Session,
    user_id: int,
    action: str,
    details: dict,
    request: Request
) -> None:
    """Log GDPR-related actions for audit compliance"""
    try:
        ip_address = None
        user_agent = None

        if request.client:
            ip_address = request.client.host
        user_agent = request.headers.get("user-agent", "")[:500]

        audit_entry = AuditLog(
            user_id=user_id,
            action=action,
            entity_type="gdpr",
            entity_id=user_id,
            details=details,
            ip_address=ip_address,
            user_agent=user_agent
        )
        db.add(audit_entry)
        db.commit()

        logger.info(f"GDPR action logged: {action} for user {user_id}")
    except Exception as e:
        logger.error(f"Failed to log GDPR action: {e}")
        db.rollback()


def get_client_ip(request: Request) -> Optional[str]:
    """Extract client IP address from request"""
    # Check for forwarded headers first (for proxy/load balancer scenarios)
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()

    if request.client:
        return request.client.host

    return None


# =============================================================================
# API Endpoints
# =============================================================================

@router.get(
    "/users/me/data-export",
    response_model=DataExportResponse,
    summary="Export user data (GDPR Right to Access)",
    description="Export all personal data associated with the authenticated user. "
                "This implements the GDPR Right to Access (Article 15).",
    responses={
        200: {"description": "User data exported successfully"},
        401: {"description": "Not authenticated"},
        500: {"description": "Internal server error during export"}
    }
)
async def export_user_data(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_sync)
) -> DataExportResponse:
    """
    Export all personal data for the authenticated user.

    GDPR Article 15 - Right of access by the data subject:
    The data subject shall have the right to obtain from the controller
    confirmation as to whether or not personal data concerning him or her
    are being processed.
    """
    try:
        logger.info(f"Data export requested for user {current_user.id}")

        # Get core user data from GDPR compliance utility
        gdpr_data = gdpr_compliance.export_user_data(str(current_user.id))

        # Gather profile data
        profile_data = {
            "email": current_user.email,
            "full_name": current_user.full_name,
            "role": str(current_user.role.value) if hasattr(current_user.role, 'value') else str(current_user.role),
            "is_active": current_user.is_active,
            "is_verified": current_user.is_verified,
            "phone_number": current_user.phone_number,
            "country": current_user.country,
            "timezone": current_user.timezone,
            "two_factor_enabled": current_user.two_factor_enabled,
            "subscription_tier": current_user.subscription_tier,
            "subscription_end_date": current_user.subscription_end_date.isoformat() if current_user.subscription_end_date else None,
            "last_login": current_user.last_login.isoformat() if current_user.last_login else None,
            "created_at": current_user.created_at.isoformat() if current_user.created_at else None,
            "updated_at": current_user.updated_at.isoformat() if current_user.updated_at else None,
        }

        # Gather portfolio data
        portfolios = db.query(Portfolio).filter(Portfolio.user_id == current_user.id).all()
        portfolio_data = []
        for portfolio in portfolios:
            portfolio_data.append({
                "id": portfolio.id,
                "name": portfolio.name,
                "description": portfolio.description,
                "cash_balance": float(portfolio.cash_balance) if portfolio.cash_balance else 0,
                "is_public": portfolio.is_public,
                "is_default": portfolio.is_default,
                "benchmark": portfolio.benchmark,
                "created_at": portfolio.created_at.isoformat() if portfolio.created_at else None,
                "positions": [
                    {
                        "stock_id": pos.stock_id,
                        "quantity": float(pos.quantity),
                        "average_cost": float(pos.average_cost)
                    }
                    for pos in portfolio.positions
                ] if portfolio.positions else []
            })

        # Gather transaction history
        transactions = db.query(Transaction).join(Portfolio).filter(
            Portfolio.user_id == current_user.id
        ).order_by(Transaction.executed_at.desc()).limit(1000).all()

        transaction_history = []
        for txn in transactions:
            transaction_history.append({
                "id": txn.id,
                "portfolio_id": txn.portfolio_id,
                "stock_id": txn.stock_id,
                "transaction_type": str(txn.transaction_type.value) if hasattr(txn.transaction_type, 'value') else str(txn.transaction_type),
                "quantity": float(txn.quantity),
                "price": float(txn.price),
                "commission": float(txn.commission) if txn.commission else 0,
                "fees": float(txn.fees) if txn.fees else 0,
                "executed_at": txn.executed_at.isoformat() if txn.executed_at else None,
            })

        # Get user preferences
        preferences = current_user.preferences or {}
        notification_settings = current_user.notification_settings or {}

        # Log the data export action
        log_gdpr_action(
            db=db,
            user_id=current_user.id,
            action="gdpr_data_export",
            details={
                "export_type": "full",
                "categories_exported": gdpr_data.get("data_categories", []),
                "portfolio_count": len(portfolio_data),
                "transaction_count": len(transaction_history)
            },
            request=request
        )

        return DataExportResponse(
            user_id=current_user.id,
            email=current_user.email,
            full_name=current_user.full_name,
            export_date=datetime.utcnow(),
            profile_data=profile_data,
            portfolio_data=portfolio_data,
            transaction_history=transaction_history,
            consent_records=gdpr_data.get("consent_records", []),
            preferences={
                "user_preferences": preferences,
                "notification_settings": notification_settings
            },
            data_categories=[
                "profile_data",
                "portfolio_data",
                "transaction_history",
                "consent_records",
                "preferences",
                "notification_settings"
            ]
        )

    except Exception as e:
        logger.error(f"Error exporting user data for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export user data. Please try again later."
        )


@router.post(
    "/users/me/delete-request",
    response_model=DeleteRequestResponse,
    summary="Request account deletion (GDPR Right to Erasure)",
    description="Initiate the right-to-erasure process. User data will be anonymized "
                "rather than deleted for audit compliance purposes.",
    responses={
        200: {"description": "Deletion request processed successfully"},
        401: {"description": "Not authenticated"},
        409: {"description": "Deletion already in progress"},
        500: {"description": "Internal server error"}
    }
)
async def request_deletion(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_sync)
) -> DeleteRequestResponse:
    """
    Initiate account deletion request.

    GDPR Article 17 - Right to erasure ('right to be forgotten'):
    The data subject shall have the right to obtain from the controller
    the erasure of personal data concerning him or her without undue delay.

    Note: For SEC and financial regulatory compliance, transaction data is
    anonymized rather than deleted to maintain audit trails.
    """
    try:
        logger.info(f"Deletion request initiated for user {current_user.id}")

        # Generate deletion request ID
        import hashlib
        request_id = hashlib.sha256(
            f"{current_user.id}:{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]

        # Anonymize user data using the DataAnonymizer
        anonymized_email = data_anonymizer.anonymize_email(current_user.email)
        anonymized_name = data_anonymizer.anonymize_name(current_user.full_name)

        # Update user record with anonymized data
        current_user.email = f"deleted_{request_id}@anonymized.local"
        current_user.full_name = f"Deleted User {request_id[:8]}"
        current_user.phone_number = None
        current_user.is_active = False
        current_user.api_key = None
        current_user.api_secret = None
        current_user.two_factor_secret = None
        current_user.preferences = {}
        current_user.notification_settings = {}

        # Call the forget_user method
        gdpr_compliance.anonymizer.forget_user(str(current_user.id))

        # Deactivate all sessions
        for session in current_user.sessions:
            session.is_active = False

        db.commit()

        # Log the deletion request
        log_gdpr_action(
            db=db,
            user_id=current_user.id,
            action="gdpr_deletion_request",
            details={
                "request_id": request_id,
                "anonymization_applied": True,
                "original_email_hash": hashlib.sha256(current_user.email.encode()).hexdigest()[:16],
                "retained_categories": [
                    "transaction_history (anonymized)",
                    "audit_logs",
                    "recommendation_performance"
                ]
            },
            request=request
        )

        logger.info(f"Deletion request completed for user {current_user.id}, request_id: {request_id}")

        return DeleteRequestResponse(
            request_id=request_id,
            status="completed",
            message="Your personal data has been anonymized. For SEC and financial regulatory "
                    "compliance, transaction records are retained in anonymized form.",
            deletion_scheduled_at=datetime.utcnow(),
            anonymization_complete=True,
            retained_for_compliance=[
                "Transaction history (anonymized for SEC compliance)",
                "Audit logs (retained for regulatory requirements)",
                "Recommendation performance data (anonymized for analytics)"
            ]
        )

    except Exception as e:
        logger.error(f"Error processing deletion request for user {current_user.id}: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process deletion request. Please contact support."
        )


@router.get(
    "/users/me/consent",
    response_model=ConsentStatusResponse,
    summary="Get consent status",
    description="Retrieve current consent status for all consent types.",
    responses={
        200: {"description": "Consent status retrieved successfully"},
        401: {"description": "Not authenticated"}
    }
)
async def get_consent_status(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_sync)
) -> ConsentStatusResponse:
    """
    Get the current consent status for the authenticated user.

    Returns all consent records including data processing, marketing,
    analytics, and third-party sharing consents.
    """
    try:
        logger.info(f"Consent status requested for user {current_user.id}")

        # Get consent records from GDPR compliance utility
        user_consents = []
        last_updated = None

        for consent_id, record in gdpr_compliance.consent_records.items():
            if record.get("user_id") == str(current_user.id):
                consent_timestamp = record.get("timestamp")
                user_consents.append(ConsentRecord(
                    consent_id=consent_id,
                    consent_type=record.get("purpose", "unknown"),
                    granted=record.get("granted", False),
                    timestamp=consent_timestamp,
                    ip_address=record.get("ip_address")
                ))

                if consent_timestamp and (last_updated is None or consent_timestamp > last_updated):
                    last_updated = consent_timestamp

        # Log the consent check
        log_gdpr_action(
            db=db,
            user_id=current_user.id,
            action="gdpr_consent_check",
            details={"consent_count": len(user_consents)},
            request=request
        )

        return ConsentStatusResponse(
            user_id=current_user.id,
            consents=user_consents,
            last_updated=last_updated
        )

    except Exception as e:
        logger.error(f"Error retrieving consent status for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve consent status."
        )


@router.post(
    "/users/me/consent",
    response_model=ConsentRecord,
    summary="Record consent",
    description="Record user consent for a specific purpose.",
    responses={
        200: {"description": "Consent recorded successfully"},
        401: {"description": "Not authenticated"},
        400: {"description": "Invalid consent type"}
    }
)
async def record_consent(
    consent_request: ConsentRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_sync)
) -> ConsentRecord:
    """
    Record user consent for a specific purpose.

    GDPR requires explicit consent for data processing activities.
    This endpoint records consent with:
    - Consent type (data_processing, marketing, analytics, third_party_sharing)
    - Whether consent is granted or denied
    - IP address for audit purposes
    - Timestamp of consent action
    """
    try:
        logger.info(f"Recording consent for user {current_user.id}: {consent_request.consent_type}")

        # Get client IP address
        ip_address = get_client_ip(request)

        # Record consent using GDPR compliance utility
        consent_id = gdpr_compliance.record_consent(
            user_id=str(current_user.id),
            purpose=consent_request.consent_type,
            granted=consent_request.granted,
            ip_address=ip_address
        )

        # Log the consent action
        log_gdpr_action(
            db=db,
            user_id=current_user.id,
            action="gdpr_consent_recorded",
            details={
                "consent_id": consent_id,
                "consent_type": consent_request.consent_type,
                "granted": consent_request.granted
            },
            request=request
        )

        # Return the consent record
        return ConsentRecord(
            consent_id=consent_id,
            consent_type=consent_request.consent_type,
            granted=consent_request.granted,
            timestamp=datetime.utcnow(),
            ip_address=data_anonymizer.anonymize_ip(ip_address) if ip_address else None
        )

    except Exception as e:
        logger.error(f"Error recording consent for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record consent."
        )


@router.delete(
    "/users/me/consent/{consent_type}",
    response_model=ConsentRecord,
    summary="Withdraw consent",
    description="Withdraw previously granted consent for a specific purpose.",
    responses={
        200: {"description": "Consent withdrawn successfully"},
        401: {"description": "Not authenticated"},
        404: {"description": "Consent type not found for user"}
    }
)
async def withdraw_consent(
    consent_type: Literal["data_processing", "marketing", "analytics", "third_party_sharing"],
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_sync)
) -> ConsentRecord:
    """
    Withdraw consent for a specific purpose.

    GDPR Article 7(3) - The data subject shall have the right to withdraw
    his or her consent at any time. The withdrawal of consent shall not
    affect the lawfulness of processing based on consent before its withdrawal.
    """
    try:
        logger.info(f"Withdrawing consent for user {current_user.id}: {consent_type}")

        # Get client IP address
        ip_address = get_client_ip(request)

        # Find existing consent for this type
        existing_consent = None
        for consent_id, record in gdpr_compliance.consent_records.items():
            if (record.get("user_id") == str(current_user.id) and
                record.get("purpose") == consent_type and
                record.get("granted") == True):
                existing_consent = (consent_id, record)
                break

        if not existing_consent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No active consent found for type: {consent_type}"
            )

        # Record the withdrawal (new record with granted=False)
        new_consent_id = gdpr_compliance.record_consent(
            user_id=str(current_user.id),
            purpose=consent_type,
            granted=False,
            ip_address=ip_address
        )

        # Log the consent withdrawal
        log_gdpr_action(
            db=db,
            user_id=current_user.id,
            action="gdpr_consent_withdrawn",
            details={
                "original_consent_id": existing_consent[0],
                "new_consent_id": new_consent_id,
                "consent_type": consent_type,
                "withdrawn_at": datetime.utcnow().isoformat()
            },
            request=request
        )

        return ConsentRecord(
            consent_id=new_consent_id,
            consent_type=consent_type,
            granted=False,
            timestamp=datetime.utcnow(),
            ip_address=data_anonymizer.anonymize_ip(ip_address) if ip_address else None
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error withdrawing consent for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to withdraw consent."
        )
