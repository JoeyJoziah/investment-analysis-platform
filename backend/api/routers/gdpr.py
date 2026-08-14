"""
GDPR Compliance API Endpoints

Implements data subject rights under GDPR:
- Right to Access (Data Export) - Article 15
- Right to Erasure (Right to be Forgotten) - Article 17
- Right to Data Portability - Article 20
- Consent Management - Article 7
- Data Retention Reports
"""

import hashlib
from datetime import datetime, timezone
from typing import Optional, List, Literal, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from backend.config.database import get_async_db_session
from backend.models.unified_models import User
from backend.auth.oauth2 import get_current_user, get_current_admin_user
from backend.compliance.gdpr import retention_manager
from backend.models.api_response import ApiResponse, success_response
from backend.security.rate_limiter import rate_limit, RateLimitCategory, RateLimitRule
import backend.services.gdpr_service as gdpr_service

# Finding #198: defense-in-depth. Every GDPR route handles PII or is destructive,
# so authentication is enforced at the router level. Individual endpoints add
# stronger, route-specific guards (admin-only or ownership) on top of this.
router = APIRouter(dependencies=[Depends(get_current_user)])
logger = logging.getLogger(__name__)


def _owns_deletion_audit(audit: Dict[str, Any], user: User) -> bool:
    """Return True if ``user`` is the subject of the given deletion audit record.

    The audit trail intentionally stores only a hashed, anonymized user
    reference (sha256(user_id)[:16]) so no raw PII is retained. To enforce
    ownership we recompute the same reference from the caller's id and compare.
    """
    expected_reference = hashlib.sha256(
        str(user.id).encode()
    ).hexdigest()[:16]
    return audit.get("anonymized_user_reference") == expected_reference

# Task 3: Rate limit rule for GDPR data exports
# 3 requests per hour, 10 requests per day
GDPR_EXPORT_RATE_LIMIT = RateLimitRule(
    requests=3,
    window_seconds=3600,  # 1 hour window
    block_duration_seconds=3600  # 1 hour block after violation
)


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



# get_client_ip lives in gdpr_service; local alias for backward compat
get_client_ip = gdpr_service.get_client_ip


@router.get(
    "/users/me/data-export",
    summary="Export user data (GDPR Right to Access & Portability)",
)
@rate_limit(category=RateLimitCategory.API_READ, custom_rule=GDPR_EXPORT_RATE_LIMIT)
async def export_user_data(
    request: Request,
    include_categories: Optional[List[str]] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db_session)
) -> ApiResponse[DataExportFullResponse]:
    """Export all personal data (GDPR Articles 15 & 20)."""
    try:
        result = await gdpr_service.export_user_data(
            user_id=current_user.id,
            session=db,
            include_categories=include_categories,
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


@router.get("/users/me/data-export/json", summary="Export user data as JSON")
async def export_user_data_json(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db_session)
) -> ApiResponse[Dict[str, Any]]:
    """Export user data as JSON format"""
    try:
        result = await gdpr_service.export_user_data(
            user_id=current_user.id,
            session=db,
        )
        return success_response(data=result.data)

    except Exception as e:
        logger.error(f"Error exporting user data as JSON: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export user data."
        )


@router.post(
    "/users/me/delete-request",
    summary="Request account deletion (GDPR Right to Erasure)",
)
async def request_deletion(
    request: Request,
    reason: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db_session)
) -> ApiResponse[DeleteRequestResponse]:
    """Initiate account deletion (GDPR Article 17 - Right to erasure)."""
    try:
        result = await gdpr_service.request_deletion(
            user_id=current_user.id,
            session=db,
            reason=reason,
        )

        resp = gdpr_service.build_deletion_request_response(result)
        return success_response(data=DeleteRequestResponse(**resp))

    except Exception as e:
        logger.error(f"Error processing deletion request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process deletion request. Please contact support."
        )


@router.post(
    "/users/me/delete-request/{request_id}/process",
    summary="Process deletion request",
)
async def process_deletion_request(
    request_id: str,
    request: Request,
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_async_db_session)
) -> ApiResponse[DeleteRequestResponse]:
    """Process a pending deletion request (admin only).

    Finding #198: irreversible erasure/anonymization. Restricted to admins so
    an anonymous caller cannot trigger destructive processing by guessing a
    ``request_id``.
    """
    try:
        result = await gdpr_service.process_deletion(
            request_id=request_id,
            session=db,
        )

        resp = gdpr_service.build_deletion_processed_response(result)
        return success_response(data=DeleteRequestResponse(**resp))

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
)
async def get_deletion_audit(
    request_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db_session)
) -> ApiResponse[DeletionAuditResponse]:
    """Get audit trail for a deletion request.

    Finding #198: prevents anonymous enumerable IDOR. The caller must be
    authenticated and may only read the audit for their own deletion request;
    admins may read any. Non-owners receive 404 so request ids are not
    enumerable.
    """
    audit = gdpr_service.get_deletion_audit(request_id)

    if not audit:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Deletion request {request_id} not found"
        )

    if not current_user.is_admin and not _owns_deletion_audit(audit, current_user):
        # Return 404 (not 403) so a non-owner cannot confirm a request id exists.
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


@router.get("/users/me/consent", summary="Get consent status")
async def get_consent_status(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db_session)
) -> ApiResponse[ConsentStatusResponse]:
    """Get current consent status for the authenticated user."""
    try:
        user_id = current_user.id

        status_data = await gdpr_service.get_consent_status(
            user_id=user_id,
            session=db,
        )
        last_updated = gdpr_service.derive_last_updated(status_data)

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


@router.get("/users/me/consent/history", summary="Get consent history")
async def get_consent_history(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db_session)
) -> ApiResponse[ConsentHistoryResponse]:
    """Get complete consent history for the authenticated user"""
    try:
        user_id = current_user.id

        history = await gdpr_service.get_consent_history(
            user_id=user_id,
            session=db,
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


@router.post("/users/me/consent", summary="Record consent")
async def record_consent(
    consent_request: ConsentRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db_session)
) -> ApiResponse[ConsentRecordResponse]:
    """Record user consent for a specific processing purpose."""
    try:
        user_id = current_user.id

        # Resolve and validate consent type
        consent_type = gdpr_service.resolve_consent_type(consent_request.consent_type)
        if not consent_type:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid consent type: {consent_request.consent_type}"
            )

        # Get client IP and anonymize immediately for GDPR compliance
        raw_ip = get_client_ip(request)
        ip_address = gdpr_service.anonymize_ip(raw_ip)
        user_agent = request.headers.get("user-agent")

        consent_id = await gdpr_service.record_consent(
            user_id=user_id,
            consent_type=consent_type,
            consent_given=consent_request.granted,
            legal_basis=consent_request.legal_basis,
            ip_address=ip_address,
            user_agent=user_agent,
            session=db,
        )

        return success_response(data=ConsentRecordResponse(
            consent_id=consent_id,
            consent_type=consent_request.consent_type,
            granted=consent_request.granted,
            timestamp=datetime.now(timezone.utc),
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


@router.delete("/users/me/consent/{consent_type}", summary="Withdraw consent")
async def withdraw_consent(
    consent_type: Literal[
        "data_processing", "marketing", "analytics",
        "third_party_sharing", "profiling", "automated_decisions"
    ],
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db_session)
) -> ApiResponse[ConsentRecordResponse]:
    """Withdraw consent for a specific purpose (GDPR Article 7(3))."""
    try:
        user_id = current_user.id

        consent_type_enum = gdpr_service.resolve_consent_type(consent_type)
        if not consent_type_enum:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid consent type: {consent_type}"
            )

        raw_ip = get_client_ip(request)
        ip_address = gdpr_service.anonymize_ip(raw_ip)

        consent_id = await gdpr_service.withdraw_consent(
            user_id=user_id,
            consent_type=consent_type_enum,
            ip_address=ip_address,
            session=db,
        )

        return success_response(data=ConsentRecordResponse(
            consent_id=consent_id,
            consent_type=consent_type,
            granted=False,
            timestamp=datetime.now(timezone.utc),
            legal_basis="consent_withdrawal",
            ip_address=ip_address
        ))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error withdrawing consent: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to withdraw consent."
        )


@router.get("/users/me/consent/{consent_type}/check", summary="Check specific consent")
async def check_consent(
    consent_type: Literal[
        "data_processing", "marketing", "analytics",
        "third_party_sharing", "profiling", "automated_decisions"
    ],
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db_session)
) -> ApiResponse[Dict[str, Any]]:
    """Check if user has valid consent for a specific purpose"""
    try:
        user_id = current_user.id

        consent_type_enum = gdpr_service.resolve_consent_type(consent_type)

        has_consent = await gdpr_service.check_consent(
            user_id=user_id,
            consent_type=consent_type_enum,
            session=db,
        )

        return success_response(data={
            "user_id": user_id,
            "consent_type": consent_type,
            "has_consent": has_consent,
            "checked_at": datetime.now(timezone.utc).isoformat()
        })

    except Exception as e:
        logger.error(f"Error checking consent: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check consent."
        )


@router.get("/users/me/retention-report", summary="Get data retention report")
async def get_retention_report(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db_session)
) -> ApiResponse[RetentionReportResponse]:
    """Generate a data retention report for the authenticated user"""
    try:
        user_id = current_user.id

        report = await gdpr_service.get_retention_report(
            user_id=user_id,
            session=db,
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


@router.post("/admin/retention/enforce", summary="Enforce retention policies (Admin)")
async def enforce_retention_policies(
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_async_db_session)
) -> ApiResponse[Dict[str, Any]]:
    """Enforce data retention policies (admin only).

    Finding #198: schedules destructive retention jobs. Restricted to admins so
    a non-admin caller cannot schedule data destruction.
    """
    try:
        background_tasks.add_task(
            retention_manager.enforce_retention_policies,
            session=db
        )

        return success_response(data={
            "status": "scheduled",
            "message": "Retention policy enforcement scheduled",
            "scheduled_at": datetime.now(timezone.utc).isoformat()
        })

    except Exception as e:
        logger.error(f"Error scheduling retention enforcement: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to schedule retention enforcement."
        )


class AnonymizeRequest(BaseModel):
    """Request model for anonymization"""
    confirm: bool = Field(..., description="Confirmation that user wants to anonymize data")
    reason: Optional[str] = Field(None, description="Optional reason for anonymization")


class AnonymizeResponse(BaseModel):
    """Response model for anonymization"""
    request_id: str
    status: str
    message: str
    anonymized_at: datetime
    anonymized_records: Dict[str, int]


@router.post("/users/me/anonymize", summary="Anonymize user data")
async def anonymize_user_data(
    anonymize_request: AnonymizeRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db_session)
) -> ApiResponse[AnonymizeResponse]:
    """Anonymize personal data while retaining records for SEC compliance."""
    try:
        if not anonymize_request.confirm:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Confirmation required. Set 'confirm: true' to proceed with anonymization."
            )

        user_id = current_user.id
        logger.info(f"Anonymization request for user {user_id}")

        result = await gdpr_service.anonymize_user_data(
            user_id=user_id,
            reason=anonymize_request.reason,
            session=db,
        )

        return success_response(data=AnonymizeResponse(
            request_id=result["request_id"],
            status="completed",
            message="User data anonymized successfully. Transaction data retained for compliance.",
            anonymized_at=datetime.now(timezone.utc),
            anonymized_records=result["anonymized_counts"]
        ))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error anonymizing user data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to anonymize user data. Please contact support."
        )


class AuditEntry(BaseModel):
    """Model for a single audit entry"""
    id: int
    action: str
    resource_type: str
    resource_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    meta_data: Optional[Dict[str, Any]] = None
    created_at: datetime


class AuditTrailResponse(BaseModel):
    """Response model for audit trail"""
    user_id: int
    total_entries: int
    entries: List[AuditEntry]
    page: int
    limit: int


@router.get("/users/me/audit", summary="Get audit trail")
async def get_audit_trail(
    request: Request,
    skip: int = 0,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db_session)
) -> ApiResponse[AuditTrailResponse]:
    """Get audit trail of data processing activities (GDPR Article 30)."""
    try:
        user_id = current_user.id

        trail = await gdpr_service.get_audit_trail(
            user_id=user_id,
            skip=skip,
            limit=limit,
            session=db,
        )

        entries = [AuditEntry(**e) for e in trail["entries"]]

        return success_response(data=AuditTrailResponse(
            user_id=user_id,
            total_entries=trail["total_entries"],
            entries=entries,
            page=trail["page"],
            limit=trail["limit"],
        ))

    except Exception as e:
        logger.error(f"Error retrieving audit trail: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve audit trail."
        )
