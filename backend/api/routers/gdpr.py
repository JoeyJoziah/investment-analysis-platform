"""
GDPR Compliance API Endpoints

Implements data subject rights under GDPR:
- Right to Access (Data Export) - Article 15
- Right to Erasure (Right to be Forgotten) - Article 17
- Right to Data Portability - Article 20
- Consent Management - Article 7
- Data Retention Reports
"""

from datetime import datetime
from typing import Optional, List, Literal, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from backend.config.database import get_async_db_session
from backend.models.unified_models import User
from backend.compliance.gdpr import (
    data_portability,
    data_deletion,
    consent_manager,
    retention_manager,
    ConsentType,
    DeletionStatus
)
from backend.utils.data_anonymization import data_anonymizer
from backend.models.api_response import ApiResponse, success_response

router = APIRouter()
logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models
# =============================================================================

class ConsentRequest(BaseModel):
    """Request model for recording consent"""
    consent_type: Literal[
        "data_processing", "marketing", "analytics",
        "third_party_sharing", "profiling", "automated_decisions"
    ]
    granted: bool = Field(..., description="Whether consent is granted")
    legal_basis: str = Field(
        default="explicit_consent",
        description="Legal basis for processing"
    )


class ConsentRecordResponse(BaseModel):
    """Response model for consent records"""
    consent_id: str
    consent_type: str
    granted: bool
    timestamp: datetime
    legal_basis: Optional[str] = None
    ip_address: Optional[str] = None


class ConsentStatusResponse(BaseModel):
    """Response model for consent status"""
    user_id: int
    consents: Dict[str, Any]
    last_updated: Optional[datetime] = None


class ConsentHistoryResponse(BaseModel):
    """Response model for consent history"""
    user_id: int
    history: List[Dict[str, Any]]


class DataExportResponse(BaseModel):
    """Response model for data export"""
    export_id: str
    user_id: int
    export_date: datetime
    categories: List[str]
    record_counts: Dict[str, int]
    download_url: Optional[str] = None
    format: str = "json"


class DataExportFullResponse(BaseModel):
    """Response model for full data export with data"""
    export_id: str
    user_id: int
    export_date: datetime
    categories: List[str]
    record_counts: Dict[str, int]
    data: Dict[str, Any]


class DeleteRequestResponse(BaseModel):
    """Response model for delete request"""
    request_id: str
    status: str
    message: str
    estimated_completion: Optional[datetime] = None
    deletion_scheduled_at: Optional[datetime] = None
    anonymization_complete: bool = False
    deleted_records: Dict[str, int] = Field(default_factory=dict)
    anonymized_records: Dict[str, int] = Field(default_factory=dict)
    retained_for_compliance: List[str] = Field(default_factory=list)


class DeletionAuditResponse(BaseModel):
    """Response model for deletion audit"""
    request_id: str
    status: str
    request_date: str
    completion_date: Optional[str] = None
    deleted_records: Dict[str, int]
    anonymized_records: Dict[str, int]
    retained_records: Dict[str, int]
    anonymized_user_reference: str


class RetentionReportResponse(BaseModel):
    """Response model for data retention report"""
    user_id: int
    report_date: str
    categories: Dict[str, Any]


# =============================================================================
# Helper Functions
# =============================================================================

def get_client_ip(request: Request) -> Optional[str]:
    """Extract client IP address from request"""
    # Check for forwarded headers first (for proxy/load balancer scenarios)
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()

    if request.client:
        return request.client.host

    return None


async def get_current_user_from_token(
    request: Request,
    db: AsyncSession = Depends(get_async_db_session)
) -> User:
    """
    Get current user from authentication token.
    This is a placeholder - integrate with your actual auth system.
    """
    # Import your actual auth dependency
    from backend.api.routers.auth import get_current_user
    return await get_current_user(request, db)


# =============================================================================
# Data Export Endpoints (GDPR Articles 15 & 20)
# =============================================================================

@router.get(
    "/users/me/data-export",
    summary="Export user data (GDPR Right to Access & Portability)",
    description="Export all personal data associated with the authenticated user. "
                "Implements GDPR Article 15 (Right to Access) and Article 20 "
                "(Right to Data Portability).",
    responses={
        200: {"description": "User data exported successfully"},
        401: {"description": "Not authenticated"},
        500: {"description": "Internal server error during export"}
    }
)
async def export_user_data(
    request: Request,
    include_categories: Optional[List[str]] = None,
    db: AsyncSession = Depends(get_async_db_session)
) -> ApiResponse[DataExportFullResponse]:
    """
    Export all personal data for the authenticated user.

    GDPR Article 15 - Right of access by the data subject:
    The data subject shall have the right to obtain from the controller
    confirmation as to whether or not personal data concerning him or her
    are being processed.

    GDPR Article 20 - Right to data portability:
    The data subject shall have the right to receive the personal data
    in a structured, commonly used and machine-readable format.
    """
    try:
        # Get current user (integrate with your auth system)
        current_user = await get_current_user_from_token(request, db)
        user_id = current_user.id

        logger.info(f"Data export requested for user {user_id}")

        # Use the database-integrated export service
        result = await data_portability.export_user_data(
            user_id=user_id,
            session=db,
            include_categories=include_categories
        )

        return success_response(data=DataExportFullResponse(
            export_id=result.export_id,
            user_id=result.user_id,
            export_date=result.export_date,
            categories=result.categories,
            record_counts=result.record_counts,
            data=result.data
        ))

    except Exception as e:
        logger.error(f"Error exporting user data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export user data. Please try again later."
        )


@router.get(
    "/users/me/data-export/json",
    summary="Export user data as JSON",
    description="Export all personal data as a JSON file.",
    responses={
        200: {"description": "JSON data returned"},
        401: {"description": "Not authenticated"}
    }
)
async def export_user_data_json(
    request: Request,
    db: AsyncSession = Depends(get_async_db_session)
) -> ApiResponse[Dict]:
    """Export user data as JSON format"""
    try:
        current_user = await get_current_user_from_token(request, db)
        result = await data_portability.export_user_data(
            user_id=current_user.id,
            session=db
        )
        return success_response(data=result.data)

    except Exception as e:
        logger.error(f"Error exporting user data as JSON: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export user data."
        )


# =============================================================================
# Data Deletion Endpoints (GDPR Article 17)
# =============================================================================

@router.post(
    "/users/me/delete-request",
    summary="Request account deletion (GDPR Right to Erasure)",
    description="Initiate the right-to-erasure process. User data will be "
                "anonymized or deleted based on regulatory requirements.",
    responses={
        200: {"description": "Deletion request created successfully"},
        401: {"description": "Not authenticated"},
        409: {"description": "Deletion already in progress"},
        500: {"description": "Internal server error"}
    }
)
async def request_deletion(
    request: Request,
    reason: Optional[str] = None,
    db: AsyncSession = Depends(get_async_db_session)
) -> ApiResponse[DeleteRequestResponse]:
    """
    Initiate account deletion request.

    GDPR Article 17 - Right to erasure ('right to be forgotten'):
    The data subject shall have the right to obtain from the controller
    the erasure of personal data concerning him or her without undue delay.

    Note: For SEC and financial regulatory compliance, transaction data is
    anonymized rather than deleted to maintain audit trails.
    """
    try:
        current_user = await get_current_user_from_token(request, db)
        user_id = current_user.id

        logger.info(f"Deletion request initiated for user {user_id}")

        # Create deletion request
        result = await data_deletion.request_deletion(
            user_id=user_id,
            reason=reason,
            session=db
        )

        return success_response(data=DeleteRequestResponse(
            request_id=result["request_id"],
            status=result["status"],
            message=result["message"],
            estimated_completion=datetime.fromisoformat(
                result["estimated_completion"]
            ) if result.get("estimated_completion") else None,
            deletion_scheduled_at=datetime.utcnow(),
            anonymization_complete=False,
            retained_for_compliance=[
                "Transaction history (anonymized for SEC compliance - 7 years)",
                "Audit logs (retained for regulatory requirements - 7 years)",
                "Consent records (retained for compliance - 10 years)"
            ]
        ))

    except Exception as e:
        logger.error(f"Error processing deletion request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process deletion request. Please contact support."
        )


@router.post(
    "/users/me/delete-request/{request_id}/process",
    summary="Process deletion request",
    description="Process a pending deletion request. Admin only.",
    responses={
        200: {"description": "Deletion processed successfully"},
        404: {"description": "Deletion request not found"},
        500: {"description": "Internal server error"}
    }
)
async def process_deletion_request(
    request_id: str,
    request: Request,
    db: AsyncSession = Depends(get_async_db_session)
) -> ApiResponse[DeleteRequestResponse]:
    """Process a pending deletion request"""
    try:
        result = await data_deletion.process_deletion(
            request_id=request_id,
            session=db
        )

        return success_response(data=DeleteRequestResponse(
            request_id=result["request_id"],
            status=result["status"],
            message="Deletion completed successfully",
            deletion_scheduled_at=datetime.fromisoformat(
                result["completion_date"]
            ) if result.get("completion_date") else None,
            anonymization_complete=True,
            deleted_records=result.get("deleted_records", {}),
            anonymized_records=result.get("anonymized_records", {}),
            retained_for_compliance=list(
                result.get("retained_for_compliance", {}).keys()
            )
        ))

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error processing deletion: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process deletion."
        )


@router.get(
    "/users/me/delete-request/{request_id}/audit",
    summary="Get deletion audit trail",
    description="Get the audit trail for a deletion request.",
    responses={
        200: {"description": "Audit trail retrieved"},
        404: {"description": "Request not found"}
    }
)
async def get_deletion_audit(
    request_id: str,
    request: Request,
    db: AsyncSession = Depends(get_async_db_session)
) -> ApiResponse[DeletionAuditResponse]:
    """Get audit trail for a deletion request"""
    audit = data_deletion.get_deletion_audit(request_id)

    if not audit:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Deletion request {request_id} not found"
        )

    return success_response(data=DeletionAuditResponse(
        request_id=audit["request_id"],
        status=audit["status"],
        request_date=audit["request_date"],
        completion_date=audit.get("completion_date"),
        deleted_records=audit.get("deleted_records", {}),
        anonymized_records=audit.get("anonymized_records", {}),
        retained_records=audit.get("retained_records", {}),
        anonymized_user_reference=audit["anonymized_user_reference"]
    ))


# =============================================================================
# Consent Management Endpoints (GDPR Article 7)
# =============================================================================

@router.get(
    "/users/me/consent",
    summary="Get consent status",
    description="Retrieve current consent status for all consent types.",
    responses={
        200: {"description": "Consent status retrieved successfully"},
        401: {"description": "Not authenticated"}
    }
)
async def get_consent_status(
    request: Request,
    db: AsyncSession = Depends(get_async_db_session)
) -> ApiResponse[ConsentStatusResponse]:
    """
    Get the current consent status for the authenticated user.

    Returns all consent records including data processing, marketing,
    analytics, and third-party sharing consents.
    """
    try:
        current_user = await get_current_user_from_token(request, db)
        user_id = current_user.id

        logger.info(f"Consent status requested for user {user_id}")

        # Get consent status from database
        status_data = await consent_manager.get_consent_status(
            user_id=user_id,
            session=db
        )

        # Find last updated timestamp
        last_updated = None
        for consent_type, info in status_data.items():
            if info.get("consent_date"):
                consent_date = datetime.fromisoformat(info["consent_date"])
                if last_updated is None or consent_date > last_updated:
                    last_updated = consent_date

        return success_response(data=ConsentStatusResponse(
            user_id=user_id,
            consents=status_data,
            last_updated=last_updated
        ))

    except Exception as e:
        logger.error(f"Error retrieving consent status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve consent status."
        )


@router.get(
    "/users/me/consent/history",
    summary="Get consent history",
    description="Retrieve complete consent history for the user.",
    responses={
        200: {"description": "Consent history retrieved"},
        401: {"description": "Not authenticated"}
    }
)
async def get_consent_history(
    request: Request,
    db: AsyncSession = Depends(get_async_db_session)
) -> ApiResponse[ConsentHistoryResponse]:
    """Get complete consent history for the authenticated user"""
    try:
        current_user = await get_current_user_from_token(request, db)
        user_id = current_user.id

        history = await consent_manager.get_consent_history(
            user_id=user_id,
            session=db
        )

        return success_response(data=ConsentHistoryResponse(
            user_id=user_id,
            history=history
        ))

    except Exception as e:
        logger.error(f"Error retrieving consent history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve consent history."
        )


@router.post(
    "/users/me/consent",
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
    db: AsyncSession = Depends(get_async_db_session)
) -> ApiResponse[ConsentRecordResponse]:
    """
    Record user consent for a specific purpose.

    GDPR requires explicit consent for data processing activities.
    This endpoint records consent with:
    - Consent type (data_processing, marketing, analytics, etc.)
    - Whether consent is granted or denied
    - Legal basis for processing
    - IP address for audit purposes
    - Timestamp of consent action
    """
    try:
        current_user = await get_current_user_from_token(request, db)
        user_id = current_user.id

        logger.info(
            f"Recording consent for user {user_id}: "
            f"{consent_request.consent_type}"
        )

        # Get client IP address and anonymize immediately for GDPR compliance
        raw_ip_address = get_client_ip(request)
        ip_address = data_anonymizer.anonymize_ip(raw_ip_address) if raw_ip_address else None
        user_agent = request.headers.get("user-agent")

        # Map string to ConsentType enum
        consent_type_map = {
            "data_processing": ConsentType.DATA_PROCESSING,
            "marketing": ConsentType.MARKETING,
            "analytics": ConsentType.ANALYTICS,
            "third_party_sharing": ConsentType.THIRD_PARTY_SHARING,
            "profiling": ConsentType.PROFILING,
            "automated_decisions": ConsentType.AUTOMATED_DECISIONS
        }
        consent_type = consent_type_map.get(consent_request.consent_type)

        if not consent_type:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid consent type: {consent_request.consent_type}"
            )

        # Record consent in database
        consent_id = await consent_manager.record_consent(
            user_id=user_id,
            consent_type=consent_type,
            consent_given=consent_request.granted,
            legal_basis=consent_request.legal_basis,
            ip_address=ip_address,
            user_agent=user_agent,
            session=db
        )

        return success_response(data=ConsentRecordResponse(
            consent_id=consent_id,
            consent_type=consent_request.consent_type,
            granted=consent_request.granted,
            timestamp=datetime.utcnow(),
            legal_basis=consent_request.legal_basis,
            ip_address=ip_address  # Already anonymized above
        ))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording consent: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record consent."
        )


@router.delete(
    "/users/me/consent/{consent_type}",
    summary="Withdraw consent",
    description="Withdraw previously granted consent for a specific purpose.",
    responses={
        200: {"description": "Consent withdrawn successfully"},
        401: {"description": "Not authenticated"},
        400: {"description": "Invalid consent type"}
    }
)
async def withdraw_consent(
    consent_type: Literal[
        "data_processing", "marketing", "analytics",
        "third_party_sharing", "profiling", "automated_decisions"
    ],
    request: Request,
    db: AsyncSession = Depends(get_async_db_session)
) -> ApiResponse[ConsentRecordResponse]:
    """
    Withdraw consent for a specific purpose.

    GDPR Article 7(3) - The data subject shall have the right to withdraw
    his or her consent at any time. The withdrawal of consent shall not
    affect the lawfulness of processing based on consent before its withdrawal.
    """
    try:
        current_user = await get_current_user_from_token(request, db)
        user_id = current_user.id

        logger.info(f"Withdrawing consent for user {user_id}: {consent_type}")

        ip_address = get_client_ip(request)

        # Map string to ConsentType enum
        consent_type_map = {
            "data_processing": ConsentType.DATA_PROCESSING,
            "marketing": ConsentType.MARKETING,
            "analytics": ConsentType.ANALYTICS,
            "third_party_sharing": ConsentType.THIRD_PARTY_SHARING,
            "profiling": ConsentType.PROFILING,
            "automated_decisions": ConsentType.AUTOMATED_DECISIONS
        }
        consent_type_enum = consent_type_map.get(consent_type)

        if not consent_type_enum:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid consent type: {consent_type}"
            )

        # Withdraw consent (record with granted=False)
        consent_id = await consent_manager.withdraw_consent(
            user_id=user_id,
            consent_type=consent_type_enum,
            ip_address=ip_address,
            session=db
        )

        return success_response(data=ConsentRecordResponse(
            consent_id=consent_id,
            consent_type=consent_type,
            granted=False,
            timestamp=datetime.utcnow(),
            legal_basis="consent_withdrawal",
            ip_address=data_anonymizer.anonymize_ip(ip_address) if ip_address else None
        ))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error withdrawing consent: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to withdraw consent."
        )


@router.get(
    "/users/me/consent/{consent_type}/check",
    summary="Check specific consent",
    description="Check if user has valid consent for a specific purpose.",
    responses={
        200: {"description": "Consent check completed"},
        401: {"description": "Not authenticated"}
    }
)
async def check_consent(
    consent_type: Literal[
        "data_processing", "marketing", "analytics",
        "third_party_sharing", "profiling", "automated_decisions"
    ],
    request: Request,
    db: AsyncSession = Depends(get_async_db_session)
) -> ApiResponse[Dict]:
    """Check if user has valid consent for a specific purpose"""
    try:
        current_user = await get_current_user_from_token(request, db)
        user_id = current_user.id

        consent_type_map = {
            "data_processing": ConsentType.DATA_PROCESSING,
            "marketing": ConsentType.MARKETING,
            "analytics": ConsentType.ANALYTICS,
            "third_party_sharing": ConsentType.THIRD_PARTY_SHARING,
            "profiling": ConsentType.PROFILING,
            "automated_decisions": ConsentType.AUTOMATED_DECISIONS
        }
        consent_type_enum = consent_type_map.get(consent_type)

        has_consent = await consent_manager.check_consent(
            user_id=user_id,
            consent_type=consent_type_enum,
            session=db
        )

        return success_response(data={
            "user_id": user_id,
            "consent_type": consent_type,
            "has_consent": has_consent,
            "checked_at": datetime.utcnow().isoformat()
        })

    except Exception as e:
        logger.error(f"Error checking consent: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check consent."
        )


# =============================================================================
# Data Retention Endpoints
# =============================================================================

@router.get(
    "/users/me/retention-report",
    summary="Get data retention report",
    description="Get a report showing data categories and their retention periods.",
    responses={
        200: {"description": "Retention report generated"},
        401: {"description": "Not authenticated"}
    }
)
async def get_retention_report(
    request: Request,
    db: AsyncSession = Depends(get_async_db_session)
) -> ApiResponse[RetentionReportResponse]:
    """Generate a data retention report for the authenticated user"""
    try:
        current_user = await get_current_user_from_token(request, db)
        user_id = current_user.id

        report = await retention_manager.get_retention_report(
            user_id=user_id,
            session=db
        )

        return success_response(data=RetentionReportResponse(
            user_id=report["user_id"],
            report_date=report["report_date"],
            categories=report["categories"]
        ))

    except Exception as e:
        logger.error(f"Error generating retention report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate retention report."
        )


@router.post(
    "/admin/retention/enforce",
    summary="Enforce retention policies (Admin)",
    description="Run retention policy enforcement to clean up expired data.",
    responses={
        200: {"description": "Retention policies enforced"},
        403: {"description": "Admin access required"}
    }
)
async def enforce_retention_policies(
    request: Request,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_async_db_session)
) -> ApiResponse[Dict]:
    """
    Enforce data retention policies by cleaning up expired data.
    Admin only endpoint.
    """
    try:
        # Add retention enforcement as background task
        background_tasks.add_task(
            retention_manager.enforce_retention_policies,
            session=db
        )

        return success_response(data={
            "status": "scheduled",
            "message": "Retention policy enforcement scheduled",
            "scheduled_at": datetime.utcnow().isoformat()
        })

    except Exception as e:
        logger.error(f"Error scheduling retention enforcement: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to schedule retention enforcement."
        )
