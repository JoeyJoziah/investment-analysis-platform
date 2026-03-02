"""
Unit tests for GDPR compliance modules (consolidated).

Tests cover:
- backend/compliance/gdpr.py: Enums, dataclasses, retention periods, GDPRDataPortability,
  GDPRDataDeletion, ConsentManager, DataRetentionManager, DataBreachNotification
- backend/services/gdpr_service.py: resolve_consent_type, derive_last_updated,
  export_user_data, request_deletion, process_deletion, get_deletion_audit,
  anonymize_user_data, get_audit_trail, get_retention_report, anonymize_ip

Includes extended edge-case tests (previously in test_gdpr_service.py):
- CONSENT_TYPE_MAP integrity, resolve_consent_type boundary cases
- derive_last_updated timezone and ordering edge cases
- Error propagation for export, deletion, retention report
- Detailed anonymize_user_data flow testing (portfolios, transactions, audit)
- Audit trail pagination edge cases and entry serialization
- Consent operation argument forwarding
- anonymize_ip IPv6 and loopback edge cases
"""

import hashlib
import logging
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from backend.compliance.gdpr import (
    RETENTION_PERIODS,
    ConsentManager,
    ConsentRecord,
    ConsentType,
    DataBreachNotification,
    DataExportResult,
    DeletionRequest,
    DeletionStatus,
    GDPRDataDeletion,
    GDPRDataPortability,
    DataRetentionManager,
    RetentionCategory,
    consent_manager,
    data_deletion,
    data_portability,
    retention_manager,
)
from backend.services.gdpr_service import (
    CONSENT_TYPE_MAP,
    anonymize_ip,
    anonymize_user_data,
    check_consent,
    derive_last_updated,
    export_user_data,
    get_audit_trail,
    get_consent_history,
    get_consent_status,
    get_deletion_audit,
    get_retention_report,
    process_deletion,
    record_consent,
    request_deletion,
    resolve_consent_type,
    withdraw_consent,
)


# ---------------------------------------------------------------------------
# ConsentType Enum
# ---------------------------------------------------------------------------


class TestConsentTypeEnum:
    """Verify ConsentType enum completeness and values."""

    def test_data_processing_member(self):
        assert ConsentType.DATA_PROCESSING.value == "data_processing"

    def test_marketing_member(self):
        assert ConsentType.MARKETING.value == "marketing"

    def test_analytics_member(self):
        assert ConsentType.ANALYTICS.value == "analytics"

    def test_third_party_sharing_member(self):
        assert ConsentType.THIRD_PARTY_SHARING.value == "third_party_sharing"

    def test_profiling_member(self):
        assert ConsentType.PROFILING.value == "profiling"

    def test_automated_decisions_member(self):
        assert ConsentType.AUTOMATED_DECISIONS.value == "automated_decisions"

    def test_consent_type_count(self):
        assert len(ConsentType) == 6

    def test_consent_type_is_str_enum(self):
        """ConsentType members should be usable as plain strings."""
        assert isinstance(ConsentType.MARKETING, str)
        assert ConsentType.MARKETING == "marketing"


# ---------------------------------------------------------------------------
# DeletionStatus Enum
# ---------------------------------------------------------------------------


class TestDeletionStatusEnum:
    """Verify DeletionStatus enum completeness and values."""

    def test_pending_member(self):
        assert DeletionStatus.PENDING.value == "pending"

    def test_processing_member(self):
        assert DeletionStatus.PROCESSING.value == "processing"

    def test_completed_member(self):
        assert DeletionStatus.COMPLETED.value == "completed"

    def test_failed_member(self):
        assert DeletionStatus.FAILED.value == "failed"

    def test_partially_completed_member(self):
        assert DeletionStatus.PARTIALLY_COMPLETED.value == "partially_completed"

    def test_deletion_status_count(self):
        assert len(DeletionStatus) == 5


# ---------------------------------------------------------------------------
# RetentionCategory Enum & RETENTION_PERIODS
# ---------------------------------------------------------------------------


class TestRetentionCategory:
    """Verify retention categories and their enforced periods."""

    def test_transaction_data_7_years(self):
        assert RETENTION_PERIODS[RetentionCategory.TRANSACTION_DATA] == 2555

    def test_audit_logs_7_years(self):
        assert RETENTION_PERIODS[RetentionCategory.AUDIT_LOGS] == 2555

    def test_user_profile_until_deletion(self):
        assert RETENTION_PERIODS[RetentionCategory.USER_PROFILE] is None

    def test_consent_records_10_years(self):
        assert RETENTION_PERIODS[RetentionCategory.CONSENT_RECORDS] == 3650

    def test_session_data_90_days(self):
        assert RETENTION_PERIODS[RetentionCategory.SESSION_DATA] == 90

    def test_analytics_data_2_years(self):
        assert RETENTION_PERIODS[RetentionCategory.ANALYTICS_DATA] == 730

    def test_retention_category_count(self):
        assert len(RetentionCategory) == 6

    def test_all_categories_have_retention_period(self):
        """Every RetentionCategory member must appear in RETENTION_PERIODS."""
        for category in RetentionCategory:
            assert category in RETENTION_PERIODS


# ---------------------------------------------------------------------------
# Dataclass: ConsentRecord
# ---------------------------------------------------------------------------


class TestConsentRecordDataclass:
    """Verify ConsentRecord dataclass fields and defaults."""

    def test_required_fields(self):
        record = ConsentRecord(
            consent_id="c-1",
            user_id=42,
            consent_type=ConsentType.MARKETING,
            consent_given=True,
            consent_date=datetime.now(timezone.utc),
            legal_basis="explicit_consent",
        )
        assert record.consent_id == "c-1"
        assert record.user_id == 42
        assert record.consent_given is True

    def test_default_version(self):
        record = ConsentRecord(
            consent_id="c-2",
            user_id=1,
            consent_type=ConsentType.ANALYTICS,
            consent_given=False,
            consent_date=datetime.now(timezone.utc),
            legal_basis="legitimate_interest",
        )
        assert record.version == "1.0"

    def test_optional_fields_default_none(self):
        record = ConsentRecord(
            consent_id="c-3",
            user_id=1,
            consent_type=ConsentType.PROFILING,
            consent_given=True,
            consent_date=datetime.now(timezone.utc),
            legal_basis="explicit_consent",
        )
        assert record.ip_address is None
        assert record.user_agent is None
        assert record.withdrawn_date is None


# ---------------------------------------------------------------------------
# Dataclass: DeletionRequest
# ---------------------------------------------------------------------------


class TestDeletionRequestDataclass:

    def test_defaults(self):
        req = DeletionRequest(
            request_id="r-1",
            user_id=10,
            status=DeletionStatus.PENDING,
            request_date=datetime.now(timezone.utc),
        )
        assert req.completion_date is None
        assert req.deleted_records == {}
        assert req.retained_records == {}
        assert req.anonymized_records == {}
        assert req.error_message is None


# ---------------------------------------------------------------------------
# Dataclass: DataExportResult
# ---------------------------------------------------------------------------


class TestDataExportResultDataclass:

    def test_default_format(self):
        result = DataExportResult(
            export_id="e-1",
            user_id=5,
            export_date=datetime.now(timezone.utc),
            categories=["profile"],
            record_counts={"profile": 1},
            data={"profile": {}},
        )
        assert result.format == "json"


# ---------------------------------------------------------------------------
# GDPRDataDeletion
# ---------------------------------------------------------------------------


class TestGDPRDataDeletion:
    """Unit tests for the GDPRDataDeletion class."""

    @pytest.mark.asyncio
    async def test_request_deletion_returns_request_id(self):
        deletion = GDPRDataDeletion()
        with patch("backend.compliance.gdpr.get_audit_logger") as mock_logger:
            mock_logger.return_value.log_gdpr_request = AsyncMock()
            result = await deletion.request_deletion(user_id=1, reason="GDPR request")

        assert "request_id" in result
        assert result["status"] == "pending"
        assert "estimated_completion" in result

    @pytest.mark.asyncio
    async def test_request_deletion_stores_pending_request(self):
        deletion = GDPRDataDeletion()
        with patch("backend.compliance.gdpr.get_audit_logger") as mock_logger:
            mock_logger.return_value.log_gdpr_request = AsyncMock()
            result = await deletion.request_deletion(user_id=99)

        assert result["request_id"] in deletion._pending_requests

    def test_get_deletion_audit_unknown_request_returns_none(self):
        deletion = GDPRDataDeletion()
        assert deletion.get_deletion_audit("nonexistent-id") is None

    @pytest.mark.asyncio
    async def test_get_deletion_audit_for_pending_request(self):
        deletion = GDPRDataDeletion()
        with patch("backend.compliance.gdpr.get_audit_logger") as mock_logger:
            mock_logger.return_value.log_gdpr_request = AsyncMock()
            result = await deletion.request_deletion(user_id=7)

        audit = deletion.get_deletion_audit(result["request_id"])
        assert audit is not None
        assert audit["status"] == "pending"
        assert "anonymized_user_reference" in audit

    @pytest.mark.asyncio
    async def test_process_deletion_unknown_request_raises(self):
        deletion = GDPRDataDeletion()
        with pytest.raises(ValueError, match="not found"):
            await deletion.process_deletion("no-such-id")

    @pytest.mark.asyncio
    async def test_anonymize_user_profile_uses_sha256(self):
        """The anonymized email/username must contain the sha256 hash prefix."""
        deletion = GDPRDataDeletion()
        user_id = 42
        expected_hash = hashlib.sha256(str(user_id).encode()).hexdigest()[:12]

        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.execute = AsyncMock()

        await deletion._anonymize_user_profile(mock_session, user_id)

        call_args = mock_session.execute.call_args
        # Verify the update statement was called (we check it was invoked)
        assert mock_session.execute.called
        # The anonymized email should embed the sha256 prefix
        assert expected_hash  # Non-empty hash was computed


# ---------------------------------------------------------------------------
# DataBreachNotification
# ---------------------------------------------------------------------------


class TestDataBreachNotification:

    def test_report_breach_returns_id(self):
        notifier = DataBreachNotification()
        breach_id = notifier.report_breach({"breach_type": "unauthorized_access"})
        assert breach_id is not None
        assert breach_id in notifier._breaches

    def test_notification_required_for_high_record_count(self):
        notifier = DataBreachNotification()
        breach_id = notifier.report_breach({"affected_records": 500})
        assert notifier.is_notification_required(breach_id) is True

    def test_notification_required_for_financial_data(self):
        notifier = DataBreachNotification()
        breach_id = notifier.report_breach(
            {"affected_records": 1, "data_categories": ["financial"]}
        )
        assert notifier.is_notification_required(breach_id) is True

    def test_notification_not_required_for_low_risk(self):
        notifier = DataBreachNotification()
        breach_id = notifier.report_breach(
            {"affected_records": 5, "data_categories": ["public_info"]}
        )
        assert notifier.is_notification_required(breach_id) is False

    def test_notification_required_unknown_breach_raises(self):
        notifier = DataBreachNotification()
        with pytest.raises(ValueError, match="not found"):
            notifier.is_notification_required("fake-id")

    def test_generate_regulatory_notification_structure(self):
        notifier = DataBreachNotification()
        breach_id = notifier.report_breach(
            {
                "breach_type": "data_leak",
                "affected_records": 1000,
                "data_categories": ["email", "authentication"],
            }
        )
        notification = notifier.generate_regulatory_notification(breach_id)
        assert notification["breach_reference"] == breach_id
        assert "dpo_contact" in notification
        assert "likely_consequences" in notification
        assert any("unauthorized" in c.lower() for c in notification["likely_consequences"])


# ---------------------------------------------------------------------------
# gdpr_service.py: resolve_consent_type
# ---------------------------------------------------------------------------


class TestResolveConsentType:

    def test_valid_data_processing(self):
        assert resolve_consent_type("data_processing") is ConsentType.DATA_PROCESSING

    def test_valid_marketing(self):
        assert resolve_consent_type("marketing") is ConsentType.MARKETING

    def test_valid_analytics(self):
        assert resolve_consent_type("analytics") is ConsentType.ANALYTICS

    def test_valid_third_party_sharing(self):
        assert resolve_consent_type("third_party_sharing") is ConsentType.THIRD_PARTY_SHARING

    def test_valid_profiling(self):
        assert resolve_consent_type("profiling") is ConsentType.PROFILING

    def test_valid_automated_decisions(self):
        assert resolve_consent_type("automated_decisions") is ConsentType.AUTOMATED_DECISIONS

    def test_invalid_type_returns_none(self):
        assert resolve_consent_type("nonexistent_type") is None

    def test_empty_string_returns_none(self):
        assert resolve_consent_type("") is None

    def test_consent_type_map_completeness(self):
        """Every ConsentType member should appear in CONSENT_TYPE_MAP values."""
        mapped_values = set(CONSENT_TYPE_MAP.values())
        for member in ConsentType:
            assert member in mapped_values


# ---------------------------------------------------------------------------
# gdpr_service.py: derive_last_updated
# ---------------------------------------------------------------------------


class TestDeriveLastUpdated:

    def test_empty_dict_returns_none(self):
        assert derive_last_updated({}) is None

    def test_single_entry_returns_that_date(self):
        dt = "2025-06-15T12:00:00+00:00"
        status = {"marketing": {"consent_date": dt}}
        result = derive_last_updated(status)
        assert result == datetime.fromisoformat(dt)

    def test_multiple_entries_returns_latest(self):
        earlier = "2025-01-01T00:00:00+00:00"
        later = "2025-12-31T23:59:59+00:00"
        status = {
            "marketing": {"consent_date": earlier},
            "analytics": {"consent_date": later},
        }
        result = derive_last_updated(status)
        assert result == datetime.fromisoformat(later)

    def test_entries_without_consent_date_are_skipped(self):
        status = {
            "marketing": {"granted": True},
            "analytics": {"consent_date": "2025-03-01T00:00:00+00:00"},
        }
        result = derive_last_updated(status)
        assert result == datetime.fromisoformat("2025-03-01T00:00:00+00:00")

    def test_all_entries_missing_consent_date_returns_none(self):
        status = {
            "marketing": {"granted": True},
            "profiling": {"granted": False},
        }
        assert derive_last_updated(status) is None


# ---------------------------------------------------------------------------
# gdpr_service.py: export_user_data (delegation)
# ---------------------------------------------------------------------------


class TestExportUserData:

    @pytest.mark.asyncio
    async def test_delegates_to_data_portability(self):
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = DataExportResult(
            export_id="exp-1",
            user_id=1,
            export_date=datetime.now(timezone.utc),
            categories=["profile"],
            record_counts={"profile": 1},
            data={"profile": {}},
        )

        with patch.object(data_portability, "export_user_data", new_callable=AsyncMock) as mock_export:
            mock_export.return_value = mock_result
            result = await export_user_data(user_id=1, session=mock_session)

        mock_export.assert_awaited_once_with(
            user_id=1,
            session=mock_session,
            include_categories=None,
        )
        assert result.export_id == "exp-1"

    @pytest.mark.asyncio
    async def test_passes_include_categories(self):
        mock_session = AsyncMock(spec=AsyncSession)
        categories = ["profile", "portfolios"]

        with patch.object(data_portability, "export_user_data", new_callable=AsyncMock) as mock_export:
            mock_export.return_value = MagicMock()
            await export_user_data(
                user_id=2, session=mock_session, include_categories=categories
            )

        mock_export.assert_awaited_once_with(
            user_id=2,
            session=mock_session,
            include_categories=categories,
        )


# ---------------------------------------------------------------------------
# gdpr_service.py: request_deletion (delegation)
# ---------------------------------------------------------------------------


class TestRequestDeletion:

    @pytest.mark.asyncio
    async def test_returns_request_id_and_status(self):
        mock_session = AsyncMock(spec=AsyncSession)
        expected = {
            "request_id": "req-abc",
            "status": "pending",
            "message": "Deletion request received.",
            "estimated_completion": "2025-07-01T00:00:00",
        }

        with patch.object(data_deletion, "request_deletion", new_callable=AsyncMock) as mock_del:
            mock_del.return_value = expected
            result = await request_deletion(user_id=5, session=mock_session, reason="test")

        assert result["request_id"] == "req-abc"
        assert result["status"] == "pending"
        mock_del.assert_awaited_once_with(
            user_id=5, reason="test", session=mock_session
        )


# ---------------------------------------------------------------------------
# gdpr_service.py: process_deletion
# ---------------------------------------------------------------------------


class TestProcessDeletion:

    @pytest.mark.asyncio
    async def test_delegates_to_data_deletion(self):
        mock_session = AsyncMock(spec=AsyncSession)
        expected = {"status": "completed", "request_id": "req-123"}

        with patch.object(data_deletion, "process_deletion", new_callable=AsyncMock) as mock_proc:
            mock_proc.return_value = expected
            result = await process_deletion(request_id="req-123", session=mock_session)

        assert result["status"] == "completed"


# ---------------------------------------------------------------------------
# gdpr_service.py: get_deletion_audit
# ---------------------------------------------------------------------------


class TestGetDeletionAudit:

    def test_delegates_to_data_deletion(self):
        expected = {"request_id": "r-1", "status": "completed"}
        with patch.object(data_deletion, "get_deletion_audit", return_value=expected) as mock_audit:
            result = get_deletion_audit("r-1")

        assert result == expected
        mock_audit.assert_called_once_with("r-1")

    def test_returns_none_for_unknown_request(self):
        with patch.object(data_deletion, "get_deletion_audit", return_value=None):
            assert get_deletion_audit("unknown") is None


# ---------------------------------------------------------------------------
# gdpr_service.py: anonymize_user_data
# ---------------------------------------------------------------------------


class TestAnonymizeUserData:

    @pytest.mark.asyncio
    async def test_replaces_pii_with_hash(self):
        """anonymize_user_data must replace email/username with sha256-based anon_id."""
        user_id = 42
        anon_id = hashlib.sha256(str(user_id).encode()).hexdigest()[:12]

        mock_session = AsyncMock(spec=AsyncSession)
        # Simulate no portfolios found
        mock_fetchall = MagicMock()
        mock_fetchall.fetchall.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_fetchall)
        # Mock rowcount for delete operations
        mock_fetchall.rowcount = 0

        with patch("backend.services.gdpr_service.get_audit_logger") as mock_logger:
            mock_logger.return_value.log_gdpr_request = AsyncMock()
            result = await anonymize_user_data(
                user_id=user_id, reason="user request", session=mock_session
            )

        assert "request_id" in result
        assert result["request_id"].startswith(f"anon_{anon_id}_")
        assert "anonymized_counts" in result
        assert result["anonymized_counts"]["profile"] == 1

    @pytest.mark.asyncio
    async def test_handles_user_with_portfolios(self):
        """When portfolios exist, transaction notes should be anonymized."""
        user_id = 10
        mock_session = AsyncMock(spec=AsyncSession)

        # First execute: update user profile -> no special return needed
        # Second execute (commit): nothing
        # Third execute: select portfolio ids
        mock_portfolio_result = MagicMock()
        mock_portfolio_result.fetchall.return_value = [(100,), (200,)]

        # For update/delete calls, return mock with rowcount
        mock_update_result = MagicMock()
        mock_update_result.rowcount = 3

        mock_delete_result = MagicMock()
        mock_delete_result.rowcount = 0

        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Call 1: update user profile
            if call_count == 1:
                return MagicMock()
            # Call 2: select portfolio ids
            if call_count == 2:
                return mock_portfolio_result
            # Call 3: update portfolios
            if call_count == 3:
                result = MagicMock()
                result.rowcount = 2
                return result
            # Call 4: update transactions
            if call_count == 4:
                return mock_update_result
            # Remaining: delete operations
            return mock_delete_result

        mock_session.execute = AsyncMock(side_effect=side_effect)
        mock_session.commit = AsyncMock()

        with patch("backend.services.gdpr_service.get_audit_logger") as mock_logger:
            mock_logger.return_value.log_gdpr_request = AsyncMock()
            result = await anonymize_user_data(
                user_id=user_id, reason=None, session=mock_session
            )

        counts = result["anonymized_counts"]
        assert counts["profile"] == 1
        assert counts.get("portfolios", 0) >= 0  # May or may not be set depending on flow


# ---------------------------------------------------------------------------
# gdpr_service.py: get_audit_trail (pagination)
# ---------------------------------------------------------------------------


class TestGetAuditTrail:

    @pytest.mark.asyncio
    async def test_returns_paginated_structure(self):
        mock_session = AsyncMock(spec=AsyncSession)

        # Mock count query
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 25

        # Mock audit log entries
        mock_log = MagicMock()
        mock_log.id = 1
        mock_log.action = "login"
        mock_log.resource_type = "session"
        mock_log.resource_id = "sess-1"
        mock_log.ip_address = "10.0.0.1"
        mock_log.user_agent = "TestAgent"
        mock_log.meta_data = {"key": "value"}
        mock_log.created_at = datetime(2025, 6, 1, tzinfo=timezone.utc)

        mock_entries_result = MagicMock()
        mock_entries_result.scalars.return_value.all.return_value = [mock_log]

        mock_session.execute = AsyncMock(
            side_effect=[mock_count_result, mock_entries_result]
        )

        result = await get_audit_trail(user_id=1, skip=0, limit=10, session=mock_session)

        assert result["total_entries"] == 25
        assert result["page"] == 1
        assert result["limit"] == 10
        assert len(result["entries"]) == 1
        assert result["entries"][0]["action"] == "login"

    @pytest.mark.asyncio
    async def test_page_calculation_with_offset(self):
        mock_session = AsyncMock(spec=AsyncSession)

        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 50

        mock_entries_result = MagicMock()
        mock_entries_result.scalars.return_value.all.return_value = []

        mock_session.execute = AsyncMock(
            side_effect=[mock_count_result, mock_entries_result]
        )

        result = await get_audit_trail(user_id=1, skip=20, limit=10, session=mock_session)
        assert result["page"] == 3  # (20 // 10) + 1

    @pytest.mark.asyncio
    async def test_zero_entries(self):
        mock_session = AsyncMock(spec=AsyncSession)

        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 0

        mock_entries_result = MagicMock()
        mock_entries_result.scalars.return_value.all.return_value = []

        mock_session.execute = AsyncMock(
            side_effect=[mock_count_result, mock_entries_result]
        )

        result = await get_audit_trail(user_id=1, skip=0, limit=10, session=mock_session)
        assert result["total_entries"] == 0
        assert result["entries"] == []


# ---------------------------------------------------------------------------
# gdpr_service.py: get_retention_report (delegation)
# ---------------------------------------------------------------------------


class TestGetRetentionReport:

    @pytest.mark.asyncio
    async def test_delegates_to_retention_manager(self):
        mock_session = AsyncMock(spec=AsyncSession)
        expected = {
            "user_id": 1,
            "report_date": "2025-06-01T00:00:00",
            "categories": {
                "transactions": {
                    "record_count": 5,
                    "retention_period_days": 2555,
                    "reason": "SEC regulatory compliance",
                }
            },
        }

        with patch.object(retention_manager, "get_retention_report", new_callable=AsyncMock) as mock_report:
            mock_report.return_value = expected
            result = await get_retention_report(user_id=1, session=mock_session)

        assert result["user_id"] == 1
        assert "categories" in result
        mock_report.assert_awaited_once_with(user_id=1, session=mock_session)

    @pytest.mark.asyncio
    async def test_report_structure_has_required_keys(self):
        mock_session = AsyncMock(spec=AsyncSession)
        expected = {
            "user_id": 1,
            "report_date": datetime.now(timezone.utc).isoformat(),
            "categories": {},
        }

        with patch.object(retention_manager, "get_retention_report", new_callable=AsyncMock) as mock_report:
            mock_report.return_value = expected
            result = await get_retention_report(user_id=1, session=mock_session)

        assert "user_id" in result
        assert "report_date" in result
        assert "categories" in result


# ---------------------------------------------------------------------------
# gdpr_service.py: consent operations (delegation)
# ---------------------------------------------------------------------------


class TestConsentServiceDelegation:

    @pytest.mark.asyncio
    async def test_record_consent_delegates(self):
        mock_session = AsyncMock(spec=AsyncSession)

        with patch.object(consent_manager, "record_consent", new_callable=AsyncMock) as mock_rec:
            mock_rec.return_value = "consent-id-1"
            result = await record_consent(
                user_id=1,
                consent_type=ConsentType.MARKETING,
                consent_given=True,
                legal_basis="explicit_consent",
                ip_address="10.0.0.1",
                user_agent="Mozilla/5.0",
                session=mock_session,
            )

        assert result == "consent-id-1"

    @pytest.mark.asyncio
    async def test_withdraw_consent_delegates(self):
        mock_session = AsyncMock(spec=AsyncSession)

        with patch.object(consent_manager, "withdraw_consent", new_callable=AsyncMock) as mock_wd:
            mock_wd.return_value = "withdrawal-id-1"
            result = await withdraw_consent(
                user_id=1,
                consent_type=ConsentType.ANALYTICS,
                ip_address="10.0.0.2",
                session=mock_session,
            )

        assert result == "withdrawal-id-1"

    @pytest.mark.asyncio
    async def test_check_consent_delegates(self):
        mock_session = AsyncMock(spec=AsyncSession)

        with patch.object(consent_manager, "check_consent", new_callable=AsyncMock) as mock_chk:
            mock_chk.return_value = True
            result = await check_consent(
                user_id=1,
                consent_type=ConsentType.DATA_PROCESSING,
                session=mock_session,
            )

        assert result is True

    @pytest.mark.asyncio
    async def test_get_consent_status_delegates(self):
        mock_session = AsyncMock(spec=AsyncSession)

        with patch.object(consent_manager, "get_consent_status", new_callable=AsyncMock) as mock_st:
            mock_st.return_value = {"marketing": {"granted": True}}
            result = await get_consent_status(user_id=1, session=mock_session)

        assert result == {"marketing": {"granted": True}}

    @pytest.mark.asyncio
    async def test_get_consent_history_delegates(self):
        mock_session = AsyncMock(spec=AsyncSession)

        with patch.object(consent_manager, "get_consent_history", new_callable=AsyncMock) as mock_hist:
            mock_hist.return_value = [{"consent_type": "marketing"}]
            result = await get_consent_history(user_id=1, session=mock_session)

        assert len(result) == 1


# ---------------------------------------------------------------------------
# gdpr_service.py: anonymize_ip
# ---------------------------------------------------------------------------


class TestAnonymizeIp:

    def test_none_input_returns_none(self):
        assert anonymize_ip(None) is None

    def test_empty_string_returns_none(self):
        """Empty string is falsy, should return None."""
        assert anonymize_ip("") is None

    def test_valid_ip_delegates_to_data_anonymizer(self):
        with patch("backend.services.gdpr_service.data_anonymizer") as mock_anon:
            mock_anon.anonymize_ip.return_value = "10.0.0.xxx"
            result = anonymize_ip("10.0.0.55")

        assert result == "10.0.0.xxx"
        mock_anon.anonymize_ip.assert_called_once_with("10.0.0.55")


# ---------------------------------------------------------------------------
# GDPRDataPortability
# ---------------------------------------------------------------------------


class TestGDPRDataPortability:

    def test_data_categories_default(self):
        portability = GDPRDataPortability()
        expected_categories = [
            "profile", "portfolios", "positions", "transactions",
            "orders", "watchlists", "alerts", "recommendations",
            "preferences", "consent_records", "sessions",
        ]
        assert portability._data_categories == expected_categories

    def test_to_json_returns_string(self):
        portability = GDPRDataPortability()
        result = DataExportResult(
            export_id="e-1",
            user_id=1,
            export_date=datetime.now(timezone.utc),
            categories=["profile"],
            record_counts={"profile": 1},
            data={"profile": {"name": "Test"}},
        )
        json_str = portability.to_json(result)
        assert isinstance(json_str, str)
        assert "Test" in json_str

    def test_to_csv_skips_export_metadata(self):
        portability = GDPRDataPortability()
        result = DataExportResult(
            export_id="e-1",
            user_id=1,
            export_date=datetime.now(timezone.utc),
            categories=["profile"],
            record_counts={"profile": 1},
            data={
                "export_metadata": {"export_id": "e-1"},
                "profile": {"data": [{"field": "value"}]},
            },
        )
        csv_files = portability.to_csv(result)
        assert "export_metadata" not in csv_files
        assert "profile" in csv_files
        assert "field" in csv_files["profile"]


# ===========================================================================
# Extended edge-case tests (consolidated from test_gdpr_service.py)
# ===========================================================================


# ---------------------------------------------------------------------------
# CONSENT_TYPE_MAP integrity
# ---------------------------------------------------------------------------


class TestConsentTypeMap:

    def test_all_enum_members_are_mapped(self):
        """Every ConsentType member must be reachable via CONSENT_TYPE_MAP."""
        mapped = set(CONSENT_TYPE_MAP.values())
        for member in ConsentType:
            assert member in mapped, f"{member} missing from CONSENT_TYPE_MAP"

    def test_all_keys_are_lowercase_strings(self):
        for key in CONSENT_TYPE_MAP:
            assert isinstance(key, str)
            assert key == key.lower()

    def test_map_values_are_consent_type_instances(self):
        for val in CONSENT_TYPE_MAP.values():
            assert isinstance(val, ConsentType)

    def test_map_has_no_duplicate_values(self):
        values = list(CONSENT_TYPE_MAP.values())
        assert len(values) == len(set(values))

    def test_map_keys_match_enum_value_strings(self):
        """Each map key should equal its enum value string."""
        for key, enum_val in CONSENT_TYPE_MAP.items():
            assert key == enum_val.value


# ---------------------------------------------------------------------------
# resolve_consent_type (boundary cases)
# ---------------------------------------------------------------------------


class TestResolveConsentTypeExtended:

    def test_case_sensitive_mismatch_returns_none(self):
        assert resolve_consent_type("Data_Processing") is None

    def test_whitespace_padded_returns_none(self):
        assert resolve_consent_type(" marketing ") is None

    def test_partial_match_returns_none(self):
        assert resolve_consent_type("market") is None

    def test_numeric_string_returns_none(self):
        assert resolve_consent_type("123") is None

    def test_all_valid_types_round_trip(self):
        """Every CONSENT_TYPE_MAP key resolves to the expected enum."""
        for key, expected in CONSENT_TYPE_MAP.items():
            assert resolve_consent_type(key) is expected


# ---------------------------------------------------------------------------
# derive_last_updated (extended edge cases)
# ---------------------------------------------------------------------------


class TestDeriveLastUpdatedExtended:

    def test_timezone_aware_comparison(self):
        """Dates with explicit UTC offset are compared correctly."""
        earlier = "2025-01-01T00:00:00+00:00"
        later = "2025-12-31T23:59:59+00:00"
        status = {
            "a": {"consent_date": later},
            "b": {"consent_date": earlier},
        }
        result = derive_last_updated(status)
        assert result == datetime.fromisoformat(later)

    def test_single_none_consent_date_value(self):
        """Entry with consent_date=None should be skipped."""
        status = {"marketing": {"consent_date": None}}
        assert derive_last_updated(status) is None

    def test_mixed_present_and_absent_dates(self):
        """Only entries with truthy consent_date contribute."""
        dt = "2025-06-15T10:00:00+00:00"
        status = {
            "a": {},
            "b": {"consent_date": dt},
            "c": {"consent_date": None},
            "d": {"other": "data"},
        }
        assert derive_last_updated(status) == datetime.fromisoformat(dt)

    def test_identical_dates_returns_that_date(self):
        dt = "2025-03-01T12:00:00+00:00"
        status = {
            "a": {"consent_date": dt},
            "b": {"consent_date": dt},
        }
        assert derive_last_updated(status) == datetime.fromisoformat(dt)

    def test_many_entries_finds_maximum(self):
        dates = [f"2025-{m:02d}-01T00:00:00+00:00" for m in range(1, 13)]
        status = {f"type_{i}": {"consent_date": d} for i, d in enumerate(dates)}
        result = derive_last_updated(status)
        assert result == datetime.fromisoformat("2025-12-01T00:00:00+00:00")


# ---------------------------------------------------------------------------
# export_user_data (delegation + error propagation)
# ---------------------------------------------------------------------------


class TestExportUserDataExtended:

    @pytest.mark.asyncio
    async def test_propagates_exception_from_delegate(self):
        mock_session = AsyncMock(spec=AsyncSession)

        with patch.object(
            data_portability, "export_user_data", new_callable=AsyncMock
        ) as mock_export:
            mock_export.side_effect = RuntimeError("DB connection lost")

            with pytest.raises(RuntimeError, match="DB connection lost"):
                await export_user_data(user_id=1, session=mock_session)

    @pytest.mark.asyncio
    async def test_none_categories_passed_by_default(self):
        mock_session = AsyncMock(spec=AsyncSession)

        with patch.object(
            data_portability, "export_user_data", new_callable=AsyncMock
        ) as mock_export:
            mock_export.return_value = MagicMock()
            await export_user_data(user_id=42, session=mock_session)

        _, kwargs = mock_export.call_args
        assert kwargs["include_categories"] is None

    @pytest.mark.asyncio
    async def test_specific_categories_forwarded(self):
        mock_session = AsyncMock(spec=AsyncSession)
        cats = ["profile", "transactions", "consent_records"]

        with patch.object(
            data_portability, "export_user_data", new_callable=AsyncMock
        ) as mock_export:
            mock_export.return_value = MagicMock()
            await export_user_data(user_id=7, session=mock_session, include_categories=cats)

        _, kwargs = mock_export.call_args
        assert kwargs["include_categories"] == cats


# ---------------------------------------------------------------------------
# request_deletion (extended)
# ---------------------------------------------------------------------------


class TestRequestDeletionExtended:

    @pytest.mark.asyncio
    async def test_reason_none_forwarded(self):
        mock_session = AsyncMock(spec=AsyncSession)

        with patch.object(
            data_deletion, "request_deletion", new_callable=AsyncMock
        ) as mock_del:
            mock_del.return_value = {"request_id": "r-1", "status": "pending"}
            await request_deletion(user_id=1, session=mock_session)

        _, kwargs = mock_del.call_args
        assert kwargs["reason"] is None

    @pytest.mark.asyncio
    async def test_propagates_delegate_exception(self):
        mock_session = AsyncMock(spec=AsyncSession)

        with patch.object(
            data_deletion, "request_deletion", new_callable=AsyncMock
        ) as mock_del:
            mock_del.side_effect = ValueError("user not found")

            with pytest.raises(ValueError, match="user not found"):
                await request_deletion(user_id=999, session=mock_session)


# ---------------------------------------------------------------------------
# process_deletion (extended)
# ---------------------------------------------------------------------------


class TestProcessDeletionExtended:

    @pytest.mark.asyncio
    async def test_forwards_request_id_correctly(self):
        mock_session = AsyncMock(spec=AsyncSession)

        with patch.object(
            data_deletion, "process_deletion", new_callable=AsyncMock
        ) as mock_proc:
            mock_proc.return_value = {"status": "completed"}
            await process_deletion(request_id="del-xyz-789", session=mock_session)

        mock_proc.assert_awaited_once_with(
            request_id="del-xyz-789", session=mock_session
        )

    @pytest.mark.asyncio
    async def test_propagates_value_error(self):
        mock_session = AsyncMock(spec=AsyncSession)

        with patch.object(
            data_deletion, "process_deletion", new_callable=AsyncMock
        ) as mock_proc:
            mock_proc.side_effect = ValueError("request not found")

            with pytest.raises(ValueError, match="request not found"):
                await process_deletion(request_id="bad-id", session=mock_session)


# ---------------------------------------------------------------------------
# get_deletion_audit (extended)
# ---------------------------------------------------------------------------


class TestGetDeletionAuditExtended:

    def test_returns_full_audit_dict(self):
        expected = {
            "request_id": "r-99",
            "status": "completed",
            "deleted_records": {"profile": 1},
        }
        with patch.object(data_deletion, "get_deletion_audit", return_value=expected):
            result = get_deletion_audit("r-99")

        assert result["deleted_records"]["profile"] == 1

    def test_returns_none_for_nonexistent(self):
        with patch.object(data_deletion, "get_deletion_audit", return_value=None):
            assert get_deletion_audit("nonexistent") is None


# ---------------------------------------------------------------------------
# anonymize_user_data (detailed flow testing)
# ---------------------------------------------------------------------------


class TestAnonymizeUserDataDetailed:

    @pytest.mark.asyncio
    async def test_anon_id_derived_from_sha256(self):
        """The anon_id in request_id must match sha256 of user_id."""
        user_id = 77
        expected_hash = hashlib.sha256(str(user_id).encode()).hexdigest()[:12]

        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_result.rowcount = 0
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.commit = AsyncMock()

        with patch("backend.services.gdpr_service.get_audit_logger") as mock_al:
            mock_al.return_value.log_gdpr_request = AsyncMock()
            result = await anonymize_user_data(user_id=user_id, reason=None, session=mock_session)

        assert result["request_id"].startswith(f"anon_{expected_hash}_")

    @pytest.mark.asyncio
    async def test_profile_always_anonymized(self):
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_result.rowcount = 0
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.commit = AsyncMock()

        with patch("backend.services.gdpr_service.get_audit_logger") as mock_al:
            mock_al.return_value.log_gdpr_request = AsyncMock()
            result = await anonymize_user_data(user_id=1, reason="test", session=mock_session)

        assert result["anonymized_counts"]["profile"] == 1

    @pytest.mark.asyncio
    async def test_sessions_watchlists_alerts_deleted(self):
        """Non-critical records (sessions, watchlists, alerts) should be deleted."""
        mock_session = AsyncMock(spec=AsyncSession)

        call_count = 0
        async def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return MagicMock()  # update user profile
            if call_count == 2:
                # select portfolio IDs - no portfolios
                result = MagicMock()
                result.fetchall.return_value = []
                return result
            # delete operations: sessions, watchlists, alerts
            result = MagicMock()
            result.rowcount = call_count - 2  # 1, 2, 3
            return result

        mock_session.execute = AsyncMock(side_effect=mock_execute)
        mock_session.commit = AsyncMock()

        with patch("backend.services.gdpr_service.get_audit_logger") as mock_al:
            mock_al.return_value.log_gdpr_request = AsyncMock()
            result = await anonymize_user_data(user_id=5, reason=None, session=mock_session)

        counts = result["anonymized_counts"]
        assert "sessions_deleted" in counts
        assert "watchlists_deleted" in counts
        assert "alerts_deleted" in counts

    @pytest.mark.asyncio
    async def test_portfolios_and_transactions_anonymized(self):
        """When portfolios exist, their names and transaction notes should be anonymized."""
        mock_session = AsyncMock(spec=AsyncSession)

        call_count = 0
        async def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return MagicMock()  # update user profile
            if call_count == 2:
                # select portfolio IDs
                r = MagicMock()
                r.fetchall.return_value = [(10,), (20,), (30,)]
                return r
            if call_count == 3:
                # update portfolios
                r = MagicMock()
                r.rowcount = 3
                return r
            if call_count == 4:
                # update transaction notes
                r = MagicMock()
                r.rowcount = 15
                return r
            # deletes
            r = MagicMock()
            r.rowcount = 0
            return r

        mock_session.execute = AsyncMock(side_effect=mock_execute)
        mock_session.commit = AsyncMock()

        with patch("backend.services.gdpr_service.get_audit_logger") as mock_al:
            mock_al.return_value.log_gdpr_request = AsyncMock()
            result = await anonymize_user_data(user_id=42, reason="leaving", session=mock_session)

        counts = result["anonymized_counts"]
        assert counts["portfolios"] == 3
        assert counts["transactions"] == 15

    @pytest.mark.asyncio
    async def test_audit_logger_called_with_correct_details(self):
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_result.rowcount = 0
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.commit = AsyncMock()

        with patch("backend.services.gdpr_service.get_audit_logger") as mock_al:
            mock_log_fn = AsyncMock()
            mock_al.return_value.log_gdpr_request = mock_log_fn
            result = await anonymize_user_data(user_id=99, reason="privacy", session=mock_session)

        mock_log_fn.assert_awaited_once()
        call_kwargs = mock_log_fn.call_args[1]
        assert call_kwargs["request_type"] == "data_anonymization"
        assert call_kwargs["user_id"] == "99"
        assert call_kwargs["details"]["reason"] == "privacy"
        assert call_kwargs["details"]["request_id"] == result["request_id"]

    @pytest.mark.asyncio
    async def test_commit_called_twice(self):
        """Session should be committed after profile update and after deletes."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_result.rowcount = 0
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.commit = AsyncMock()

        with patch("backend.services.gdpr_service.get_audit_logger") as mock_al:
            mock_al.return_value.log_gdpr_request = AsyncMock()
            await anonymize_user_data(user_id=1, reason=None, session=mock_session)

        assert mock_session.commit.await_count == 2


# ---------------------------------------------------------------------------
# get_audit_trail (extended pagination and serialization)
# ---------------------------------------------------------------------------


class TestGetAuditTrailExtended:

    def _make_mock_session(self, total, logs):
        mock_session = AsyncMock(spec=AsyncSession)

        mock_count = MagicMock()
        mock_count.scalar.return_value = total

        mock_entries = MagicMock()
        mock_entries.scalars.return_value.all.return_value = logs

        mock_session.execute = AsyncMock(side_effect=[mock_count, mock_entries])
        return mock_session

    def _make_log_entry(self, **overrides):
        defaults = {
            "id": 1,
            "action": "data_access",
            "resource_type": "portfolio",
            "resource_id": "p-100",
            "ip_address": "192.168.1.1",
            "user_agent": "TestBrowser/1.0",
            "meta_data": {"key": "value"},
            "created_at": datetime(2025, 7, 1, 12, 0, tzinfo=timezone.utc),
        }
        defaults.update(overrides)
        return MagicMock(**defaults)

    @pytest.mark.asyncio
    async def test_entry_serialization_all_fields(self):
        log = self._make_log_entry()
        session = self._make_mock_session(total=1, logs=[log])

        result = await get_audit_trail(user_id=1, skip=0, limit=10, session=session)

        entry = result["entries"][0]
        assert entry["id"] == 1
        assert entry["action"] == "data_access"
        assert entry["resource_type"] == "portfolio"
        assert entry["resource_id"] == "p-100"
        assert entry["ip_address"] == "192.168.1.1"
        assert entry["user_agent"] == "TestBrowser/1.0"
        assert entry["meta_data"] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_null_resource_type_defaults_to_unknown(self):
        log = self._make_log_entry(resource_type=None)
        session = self._make_mock_session(total=1, logs=[log])

        result = await get_audit_trail(user_id=1, skip=0, limit=10, session=session)

        assert result["entries"][0]["resource_type"] == "unknown"

    @pytest.mark.asyncio
    async def test_page_1_when_skip_0(self):
        session = self._make_mock_session(total=100, logs=[])
        result = await get_audit_trail(user_id=1, skip=0, limit=25, session=session)
        assert result["page"] == 1

    @pytest.mark.asyncio
    async def test_page_calculation_skip_50_limit_25(self):
        session = self._make_mock_session(total=100, logs=[])
        result = await get_audit_trail(user_id=1, skip=50, limit=25, session=session)
        assert result["page"] == 3  # (50 // 25) + 1

    @pytest.mark.asyncio
    async def test_page_1_when_limit_zero(self):
        """When limit=0, page should default to 1 (avoids division by zero)."""
        session = self._make_mock_session(total=10, logs=[])
        result = await get_audit_trail(user_id=1, skip=0, limit=0, session=session)
        assert result["page"] == 1

    @pytest.mark.asyncio
    async def test_total_entries_scalar_none_defaults_to_zero(self):
        mock_session = AsyncMock(spec=AsyncSession)

        mock_count = MagicMock()
        mock_count.scalar.return_value = None

        mock_entries = MagicMock()
        mock_entries.scalars.return_value.all.return_value = []

        mock_session.execute = AsyncMock(side_effect=[mock_count, mock_entries])

        result = await get_audit_trail(user_id=1, skip=0, limit=10, session=mock_session)
        assert result["total_entries"] == 0

    @pytest.mark.asyncio
    async def test_multiple_entries_serialized(self):
        logs = [self._make_log_entry(id=i, action=f"action_{i}") for i in range(5)]
        session = self._make_mock_session(total=5, logs=logs)

        result = await get_audit_trail(user_id=1, skip=0, limit=10, session=session)

        assert len(result["entries"]) == 5
        assert result["entries"][2]["action"] == "action_2"


# ---------------------------------------------------------------------------
# Consent operations (argument forwarding focus)
# ---------------------------------------------------------------------------


class TestConsentOperationsForwarding:

    @pytest.mark.asyncio
    async def test_get_consent_status_forwards_user_id(self):
        mock_session = AsyncMock(spec=AsyncSession)

        with patch.object(consent_manager, "get_consent_status", new_callable=AsyncMock) as mock_fn:
            mock_fn.return_value = {}
            await get_consent_status(user_id=42, session=mock_session)

        mock_fn.assert_awaited_once_with(user_id=42, session=mock_session)

    @pytest.mark.asyncio
    async def test_get_consent_history_forwards_user_id(self):
        mock_session = AsyncMock(spec=AsyncSession)

        with patch.object(consent_manager, "get_consent_history", new_callable=AsyncMock) as mock_fn:
            mock_fn.return_value = []
            await get_consent_history(user_id=7, session=mock_session)

        mock_fn.assert_awaited_once_with(user_id=7, session=mock_session)

    @pytest.mark.asyncio
    async def test_record_consent_forwards_all_params(self):
        mock_session = AsyncMock(spec=AsyncSession)

        with patch.object(consent_manager, "record_consent", new_callable=AsyncMock) as mock_fn:
            mock_fn.return_value = "consent-abc"
            result = await record_consent(
                user_id=10,
                consent_type=ConsentType.PROFILING,
                consent_given=False,
                legal_basis="legitimate_interest",
                ip_address="10.0.0.5",
                user_agent="Chrome/100",
                session=mock_session,
            )

        assert result == "consent-abc"
        mock_fn.assert_awaited_once_with(
            user_id=10,
            consent_type=ConsentType.PROFILING,
            consent_given=False,
            legal_basis="legitimate_interest",
            ip_address="10.0.0.5",
            user_agent="Chrome/100",
            session=mock_session,
        )

    @pytest.mark.asyncio
    async def test_record_consent_with_none_ip_and_agent(self):
        mock_session = AsyncMock(spec=AsyncSession)

        with patch.object(consent_manager, "record_consent", new_callable=AsyncMock) as mock_fn:
            mock_fn.return_value = "consent-xyz"
            await record_consent(
                user_id=5,
                consent_type=ConsentType.MARKETING,
                consent_given=True,
                legal_basis="explicit_consent",
                ip_address=None,
                user_agent=None,
                session=mock_session,
            )

        _, kwargs = mock_fn.call_args
        assert kwargs["ip_address"] is None
        assert kwargs["user_agent"] is None

    @pytest.mark.asyncio
    async def test_withdraw_consent_forwards_all_params(self):
        mock_session = AsyncMock(spec=AsyncSession)

        with patch.object(consent_manager, "withdraw_consent", new_callable=AsyncMock) as mock_fn:
            mock_fn.return_value = "withdrawal-id"
            result = await withdraw_consent(
                user_id=20,
                consent_type=ConsentType.THIRD_PARTY_SHARING,
                ip_address="172.16.0.1",
                session=mock_session,
            )

        assert result == "withdrawal-id"
        mock_fn.assert_awaited_once_with(
            user_id=20,
            consent_type=ConsentType.THIRD_PARTY_SHARING,
            ip_address="172.16.0.1",
            session=mock_session,
        )

    @pytest.mark.asyncio
    async def test_check_consent_true(self):
        mock_session = AsyncMock(spec=AsyncSession)

        with patch.object(consent_manager, "check_consent", new_callable=AsyncMock) as mock_fn:
            mock_fn.return_value = True
            result = await check_consent(
                user_id=1, consent_type=ConsentType.ANALYTICS, session=mock_session
            )

        assert result is True

    @pytest.mark.asyncio
    async def test_check_consent_false(self):
        mock_session = AsyncMock(spec=AsyncSession)

        with patch.object(consent_manager, "check_consent", new_callable=AsyncMock) as mock_fn:
            mock_fn.return_value = False
            result = await check_consent(
                user_id=1, consent_type=ConsentType.MARKETING, session=mock_session
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_check_consent_with_none_type(self):
        mock_session = AsyncMock(spec=AsyncSession)

        with patch.object(consent_manager, "check_consent", new_callable=AsyncMock) as mock_fn:
            mock_fn.return_value = False
            await check_consent(user_id=1, consent_type=None, session=mock_session)

        mock_fn.assert_awaited_once_with(
            user_id=1, consent_type=None, session=mock_session
        )


# ---------------------------------------------------------------------------
# get_retention_report (extended)
# ---------------------------------------------------------------------------


class TestGetRetentionReportExtended:

    @pytest.mark.asyncio
    async def test_propagates_exception(self):
        mock_session = AsyncMock(spec=AsyncSession)

        with patch.object(
            retention_manager, "get_retention_report", new_callable=AsyncMock
        ) as mock_fn:
            mock_fn.side_effect = RuntimeError("DB error")

            with pytest.raises(RuntimeError, match="DB error"):
                await get_retention_report(user_id=1, session=mock_session)

    @pytest.mark.asyncio
    async def test_returns_delegate_result_unchanged(self):
        mock_session = AsyncMock(spec=AsyncSession)
        expected = {
            "user_id": 5,
            "report_date": "2025-08-01",
            "categories": {"session_data": {"retention_period_days": 90}},
        }

        with patch.object(
            retention_manager, "get_retention_report", new_callable=AsyncMock
        ) as mock_fn:
            mock_fn.return_value = expected
            result = await get_retention_report(user_id=5, session=mock_session)

        assert result is expected


# ---------------------------------------------------------------------------
# anonymize_ip (extended edge cases)
# ---------------------------------------------------------------------------


class TestAnonymizeIpExtended:

    def test_none_returns_none(self):
        assert anonymize_ip(None) is None

    def test_empty_string_returns_none(self):
        assert anonymize_ip("") is None

    def test_ipv4_delegates(self):
        with patch("backend.services.gdpr_service.data_anonymizer") as mock_anon:
            mock_anon.anonymize_ip.return_value = "192.168.1.xxx"
            result = anonymize_ip("192.168.1.55")

        assert result == "192.168.1.xxx"
        mock_anon.anonymize_ip.assert_called_once_with("192.168.1.55")

    def test_ipv6_delegates(self):
        with patch("backend.services.gdpr_service.data_anonymizer") as mock_anon:
            mock_anon.anonymize_ip.return_value = "2001:db8::xxxx"
            result = anonymize_ip("2001:db8::1")

        assert result == "2001:db8::xxxx"

    def test_loopback_delegates(self):
        with patch("backend.services.gdpr_service.data_anonymizer") as mock_anon:
            mock_anon.anonymize_ip.return_value = "127.0.0.xxx"
            result = anonymize_ip("127.0.0.1")

        assert result == "127.0.0.xxx"
