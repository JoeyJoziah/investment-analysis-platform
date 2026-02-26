"""
Unit tests for backend/services/gdpr_service.py (SERVICE LAYER focus).

This file tests the gdpr_service module's OWN logic and delegation patterns.
It deliberately avoids duplicating coverage already in test_compliance_gdpr.py,
which covers enums, dataclasses, the compliance-layer classes (GDPRDataDeletion,
DataBreachNotification, GDPRDataPortability), and basic delegation smoke tests.

Focus here:
- CONSENT_TYPE_MAP completeness and edge cases
- resolve_consent_type boundary cases
- derive_last_updated edge cases (timezone handling, ordering)
- export_user_data error propagation
- request_deletion / process_deletion error propagation
- get_deletion_audit sync delegation edge cases
- anonymize_user_data detailed flow (profile + portfolios + deletes + audit logging)
- get_audit_trail pagination edge cases, entry serialization details
- get_consent_status / get_consent_history / record_consent / withdraw_consent /
  check_consent argument forwarding
- get_retention_report error propagation
- anonymize_ip edge cases
"""

import hashlib
import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from backend.compliance.gdpr import (
    ConsentType,
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
# resolve_consent_type (boundary cases beyond test_compliance_gdpr.py)
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
