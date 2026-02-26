"""
Unit tests for backend/services/admin_service.py

Tests all public functions of the admin_service module with mocked
dependencies. No database, network, or external service access required.
"""

import os
import random
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from backend.services.admin_service import (
    _mask_secret,
    apply_configuration_update,
    build_user_record,
    cancel_job,
    delete_user,
    disable_maintenance_mode,
    enable_maintenance_mode,
    execute_system_command,
    get_api_usage_stats,
    get_audit_logs,
    get_configuration,
    get_system_health_data,
    get_system_metrics_data,
    get_user_by_id,
    initiate_data_export,
    list_announcements,
    list_background_jobs,
    list_users,
    retry_job,
)


# ===========================================================================
# get_system_health_data
# ===========================================================================


class TestGetSystemHealthData:

    def test_returns_all_required_keys(self):
        """Health data should contain every expected top-level key."""
        data = get_system_health_data()
        expected_keys = {
            "status", "uptime", "cpu_usage", "memory_usage", "disk_usage",
            "active_connections", "request_rate", "error_rate",
            "response_time_avg", "services", "last_check",
        }
        assert expected_keys == set(data.keys())

    def test_status_is_operational(self):
        """System status should always be 'operational'."""
        data = get_system_health_data()
        assert data["status"] == "operational"

    def test_services_all_running(self):
        """Every service should report 'running'."""
        data = get_system_health_data()
        for name, state in data["services"].items():
            assert state == "running", f"Service {name} is not running"

    def test_services_list_complete(self):
        """All six monitored services should be present."""
        data = get_system_health_data()
        expected_services = {"api", "database", "cache", "worker", "scheduler", "websocket"}
        assert expected_services == set(data["services"].keys())

    def test_last_check_is_utc_datetime(self):
        """last_check should be a timezone-aware UTC datetime."""
        data = get_system_health_data()
        assert isinstance(data["last_check"], datetime)
        assert data["last_check"].tzinfo is not None

    def test_cpu_usage_within_range(self):
        """cpu_usage should be between 20 and 80."""
        random.seed(99)
        data = get_system_health_data()
        assert 20 <= data["cpu_usage"] <= 80

    def test_memory_usage_within_range(self):
        """memory_usage should be between 30 and 70."""
        random.seed(99)
        data = get_system_health_data()
        assert 30 <= data["memory_usage"] <= 70

    def test_uptime_is_positive_int(self):
        """uptime should be a positive integer (seconds)."""
        data = get_system_health_data()
        assert isinstance(data["uptime"], int)
        assert data["uptime"] > 0


# ===========================================================================
# build_user_record
# ===========================================================================


class TestBuildUserRecord:

    def test_record_has_all_fields(self):
        """A user record should have the expected set of keys."""
        random.seed(42)
        record = build_user_record(0)
        expected_keys = {
            "id", "email", "full_name", "role", "is_active", "is_verified",
            "created_at", "last_login", "subscription_tier",
            "api_calls_today", "storage_used_mb",
        }
        assert expected_keys == set(record.keys())

    def test_email_uses_index(self):
        """Email should be user{index}@example.com."""
        record = build_user_record(7)
        assert record["email"] == "user7@example.com"

    def test_full_name_uses_index(self):
        """Full name should be 'User {index}'."""
        record = build_user_record(42)
        assert record["full_name"] == "User 42"

    def test_id_is_valid_uuid(self):
        """The id field should be a valid UUID string."""
        record = build_user_record(0)
        parsed = uuid.UUID(record["id"])
        assert str(parsed) == record["id"]

    def test_role_is_valid_enum_value(self):
        """Role should be one of the defined _UserRole values."""
        random.seed(42)
        valid_roles = {"super_admin", "admin", "moderator", "analyst", "user"}
        for i in range(20):
            record = build_user_record(i)
            assert record["role"] in valid_roles

    def test_created_at_is_in_the_past(self):
        """created_at should be at most 365 days before now."""
        random.seed(42)
        record = build_user_record(0)
        assert record["created_at"] < datetime.now(timezone.utc)

    def test_is_active_is_boolean(self):
        """is_active should be a boolean."""
        record = build_user_record(0)
        assert isinstance(record["is_active"], bool)


# ===========================================================================
# list_users
# ===========================================================================


class TestListUsers:

    def test_default_returns_list(self):
        """Default call returns a list of dicts."""
        random.seed(42)
        users = list_users()
        assert isinstance(users, list)
        assert len(users) <= 50

    def test_limit_controls_page_size(self):
        """Specifying limit should cap the returned count."""
        random.seed(42)
        users = list_users(limit=5)
        assert len(users) <= 5

    def test_offset_skips_records(self):
        """offset=90 with limit=50 should return at most 10 users (from 100)."""
        random.seed(42)
        users = list_users(limit=50, offset=90)
        assert len(users) <= 10

    def test_offset_beyond_total_returns_empty(self):
        """Offset past all records returns an empty list."""
        random.seed(42)
        users = list_users(limit=50, offset=200)
        assert users == []

    def test_filter_by_role(self):
        """Filtering by role should return only matching users."""
        random.seed(42)
        users = list_users(role="admin")
        for u in users:
            assert u["role"] == "admin"

    def test_filter_by_is_active(self):
        """Filtering by is_active=True should return only active users."""
        random.seed(42)
        users = list_users(is_active=True)
        for u in users:
            assert u["is_active"] is True

    def test_filter_by_is_active_false(self):
        """Filtering by is_active=False should return only inactive users."""
        random.seed(42)
        users = list_users(is_active=False)
        for u in users:
            assert u["is_active"] is False

    def test_filter_by_role_and_active(self):
        """Both filters can be combined."""
        random.seed(42)
        users = list_users(role="user", is_active=True)
        for u in users:
            assert u["role"] == "user"
            assert u["is_active"] is True

    def test_nonexistent_role_returns_empty(self):
        """A role that doesn't exist should yield no results."""
        random.seed(42)
        users = list_users(role="nonexistent_role")
        assert users == []

    def test_zero_limit_returns_empty(self):
        """A limit of 0 should return an empty list."""
        random.seed(42)
        users = list_users(limit=0)
        assert users == []


# ===========================================================================
# get_user_by_id
# ===========================================================================


class TestGetUserById:

    def test_returns_user_with_supplied_id(self):
        """Returned user should have the same id as the input."""
        user = get_user_by_id("abc-123")
        assert user["id"] == "abc-123"

    def test_static_fields(self):
        """Email, full_name, role should be the deterministic defaults."""
        user = get_user_by_id("test-id")
        assert user["email"] == "user@example.com"
        assert user["full_name"] == "John Doe"
        assert user["role"] == "user"

    def test_is_active_and_verified(self):
        """Default user should be active and verified."""
        user = get_user_by_id("xyz")
        assert user["is_active"] is True
        assert user["is_verified"] is True

    def test_subscription_tier_premium(self):
        """Default subscription should be premium."""
        user = get_user_by_id("id-1")
        assert user["subscription_tier"] == "premium"

    def test_created_at_in_past(self):
        """created_at should be before now."""
        user = get_user_by_id("id-1")
        assert user["created_at"] < datetime.now(timezone.utc)


# ===========================================================================
# delete_user
# ===========================================================================


class TestDeleteUser:

    def test_success_response(self):
        """Should return success status with message containing user_id."""
        result = delete_user("user-42")
        assert result["status"] == "success"
        assert "user-42" in result["message"]

    def test_message_contains_sanitized_id(self):
        """The message should include the (sanitized) user ID."""
        result = delete_user("abc")
        assert "abc" in result["message"]


# ===========================================================================
# get_api_usage_stats
# ===========================================================================


class TestGetApiUsageStats:

    def test_returns_list_of_endpoints(self):
        """Should return one entry per monitored endpoint."""
        random.seed(42)
        stats = get_api_usage_stats()
        assert isinstance(stats, list)
        assert len(stats) == 6

    def test_entry_has_required_fields(self):
        """Each entry should contain all expected analytics fields."""
        random.seed(42)
        stats = get_api_usage_stats()
        required_keys = {
            "endpoint", "method", "total_calls", "successful_calls",
            "failed_calls", "avg_response_time", "p95_response_time",
            "p99_response_time", "total_data_transferred",
            "unique_users", "last_called",
        }
        for entry in stats:
            assert required_keys == set(entry.keys())

    def test_methods_are_get_or_post(self):
        """Only GET and POST methods should be present."""
        stats = get_api_usage_stats()
        for entry in stats:
            assert entry["method"] in ("GET", "POST")

    def test_last_called_is_datetime(self):
        """last_called should be a datetime."""
        stats = get_api_usage_stats()
        for entry in stats:
            assert isinstance(entry["last_called"], datetime)

    def test_days_back_parameter_accepted(self):
        """Function should accept a days_back argument without error."""
        stats = get_api_usage_stats(days_back=30)
        assert len(stats) == 6


# ===========================================================================
# get_system_metrics_data
# ===========================================================================


class TestGetSystemMetricsData:

    def test_top_level_keys(self):
        """Should have timestamp plus all subsystem metric groups."""
        random.seed(42)
        metrics = get_system_metrics_data()
        expected_keys = {
            "timestamp", "cpu", "memory", "disk",
            "network", "database", "cache", "queue",
        }
        assert expected_keys == set(metrics.keys())

    def test_timestamp_is_utc_datetime(self):
        """timestamp should be timezone-aware UTC datetime."""
        metrics = get_system_metrics_data()
        assert isinstance(metrics["timestamp"], datetime)
        assert metrics["timestamp"].tzinfo is not None

    def test_cpu_cores_is_eight(self):
        """CPU cores should be the fixed value 8."""
        metrics = get_system_metrics_data()
        assert metrics["cpu"]["cores"] == 8

    def test_memory_total_is_sixteen(self):
        """Total memory should be 16 GB."""
        metrics = get_system_metrics_data()
        assert metrics["memory"]["total_gb"] == 16

    def test_disk_total_is_five_hundred(self):
        """Total disk should be 500 GB."""
        metrics = get_system_metrics_data()
        assert metrics["disk"]["total_gb"] == 500

    def test_cache_hit_rate_between_zero_and_one(self):
        """Cache hit rate should be between 0.85 and 0.99."""
        random.seed(42)
        metrics = get_system_metrics_data()
        assert 0.85 <= metrics["cache"]["hit_rate"] <= 0.99

    def test_queue_pending_non_negative(self):
        """Queue pending count should be non-negative."""
        random.seed(42)
        metrics = get_system_metrics_data()
        assert metrics["queue"]["pending"] >= 0


# ===========================================================================
# list_background_jobs
# ===========================================================================


class TestListBackgroundJobs:

    def test_unfiltered_returns_twenty_jobs(self):
        """Without a status filter, 20 jobs should be returned."""
        random.seed(42)
        jobs = list_background_jobs()
        assert len(jobs) == 20

    def test_filtered_by_status_running(self):
        """Filtering by 'running' should return only running jobs."""
        random.seed(42)
        jobs = list_background_jobs(status="running")
        for j in jobs:
            assert j["status"] == "running"

    def test_filtered_by_status_completed(self):
        """Filtering by 'completed' should return only completed jobs."""
        random.seed(42)
        jobs = list_background_jobs(status="completed")
        for j in jobs:
            assert j["status"] == "completed"

    def test_completed_jobs_have_progress_100(self):
        """Completed jobs should have progress == 100."""
        random.seed(42)
        jobs = list_background_jobs(status="completed")
        for j in jobs:
            assert j["progress"] == 100

    def test_completed_jobs_have_completed_at(self):
        """Completed jobs should have a non-None completed_at timestamp."""
        random.seed(42)
        jobs = list_background_jobs(status="completed")
        for j in jobs:
            assert j["completed_at"] is not None

    def test_completed_jobs_have_result(self):
        """Completed jobs should include a result dict."""
        random.seed(42)
        jobs = list_background_jobs(status="completed")
        for j in jobs:
            assert j["result"] is not None
            assert "records_processed" in j["result"]

    def test_failed_jobs_have_error_message(self):
        """Failed jobs should have an error_message."""
        random.seed(42)
        jobs = list_background_jobs(status="failed")
        for j in jobs:
            assert j["error_message"] == "Connection timeout"

    def test_failed_jobs_progress_zero(self):
        """Failed jobs should have progress == 0."""
        random.seed(42)
        jobs = list_background_jobs(status="failed")
        for j in jobs:
            assert j["progress"] == 0

    def test_pending_jobs_progress_zero(self):
        """Pending jobs should have progress == 0."""
        random.seed(42)
        jobs = list_background_jobs(status="pending")
        for j in jobs:
            assert j["progress"] == 0

    def test_job_has_required_fields(self):
        """Each job should have all expected keys."""
        random.seed(42)
        jobs = list_background_jobs()
        required_keys = {
            "id", "name", "type", "status", "progress",
            "started_at", "completed_at", "error_message",
            "result", "retry_count",
        }
        for j in jobs:
            assert required_keys == set(j.keys())

    def test_job_type_is_known(self):
        """Job types should be from the known set."""
        random.seed(42)
        known_types = {"data_sync", "analysis", "report_generation", "cleanup", "backup"}
        jobs = list_background_jobs()
        for j in jobs:
            assert j["type"] in known_types


# ===========================================================================
# cancel_job
# ===========================================================================


class TestCancelJob:

    def test_success_response(self):
        """Should return success status."""
        result = cancel_job("job-99")
        assert result["status"] == "success"

    def test_message_contains_job_id(self):
        """Message should reference the cancelled job ID."""
        result = cancel_job("job-abc-123")
        assert "job-abc-123" in result["message"]


# ===========================================================================
# retry_job
# ===========================================================================


class TestRetryJob:

    def test_success_response(self):
        """Should return success status."""
        result = retry_job("job-fail-1")
        assert result["status"] == "success"

    def test_message_contains_job_id(self):
        """Message should reference the retried job ID."""
        result = retry_job("job-fail-1")
        assert "job-fail-1" in result["message"]

    def test_new_job_id_is_valid_uuid(self):
        """A new job ID (valid UUID) should be assigned for the retry."""
        result = retry_job("job-fail-1")
        parsed = uuid.UUID(result["new_job_id"])
        assert str(parsed) == result["new_job_id"]

    def test_new_job_id_differs_from_original(self):
        """The new job ID should not equal the original."""
        result = retry_job("original-id")
        assert result["new_job_id"] != "original-id"


# ===========================================================================
# _mask_secret
# ===========================================================================


class TestMaskSecret:

    def test_none_returns_not_set(self):
        """None value should produce ***NOT_SET***."""
        assert _mask_secret(None) == "***NOT_SET***"

    def test_empty_string_returns_not_set(self):
        """Empty string should produce ***NOT_SET***."""
        assert _mask_secret("") == "***NOT_SET***"

    def test_short_string_returns_not_set(self):
        """Strings shorter than 8 chars should produce ***NOT_SET***."""
        assert _mask_secret("abc") == "***NOT_SET***"
        assert _mask_secret("1234567") == "***NOT_SET***"

    def test_exactly_eight_chars_is_masked(self):
        """A string of exactly 8 chars should be masked."""
        result = _mask_secret("12345678")
        assert result == "1234...5678"

    def test_long_string_shows_bookends(self):
        """Long strings should show first 4 and last 4 chars."""
        result = _mask_secret("sk-proj-abcdefghijklmnop")
        assert result.startswith("sk-p")
        assert result.endswith("mnop")
        assert "..." in result

    def test_mask_hides_middle(self):
        """The masked value should not contain middle characters."""
        secret = "abcd____MIDDLE____wxyz"
        result = _mask_secret(secret)
        assert "MIDDLE" not in result


# ===========================================================================
# get_configuration
# ===========================================================================


class TestGetConfiguration:

    def test_all_sections_present(self):
        """Full config should contain all expected sections."""
        config = get_configuration()
        expected_sections = {
            "api_keys", "database", "cache", "security",
            "features", "limits", "monitoring",
        }
        assert expected_sections == set(config.keys())

    def test_section_filter_returns_only_that_section(self):
        """Requesting a section should return only that section."""
        config = get_configuration(section="database")
        assert set(config.keys()) == {"database"}

    def test_nonexistent_section_returns_empty_dict(self):
        """Requesting a missing section should return an empty dict for that key."""
        config = get_configuration(section="nonexistent")
        assert config == {"nonexistent": {}}

    def test_api_keys_are_masked(self):
        """API key values should be masked (never raw env values)."""
        config = get_configuration()
        for key_name, value in config["api_keys"].items():
            assert "..." in value or value == "***NOT_SET***"

    def test_api_keys_masked_with_env_set(self):
        """When env var is set, the masked value should show bookends."""
        with patch.dict(os.environ, {"ALPHA_VANTAGE_API_KEY": "abcd1234efgh5678"}):
            config = get_configuration()
            masked = config["api_keys"]["alpha_vantage"]
            assert masked.startswith("abcd")
            assert masked.endswith("5678")
            assert "..." in masked

    def test_database_defaults(self):
        """Database section should have expected default values."""
        config = get_configuration()
        db = config["database"]
        assert db["host"] == "postgres"
        assert db["port"] == 5432
        assert db["name"] == "investment_db"
        assert db["pool_size"] == 20

    def test_cache_defaults(self):
        """Cache section should have expected default values."""
        config = get_configuration()
        cache = config["cache"]
        assert cache["host"] == "redis"
        assert cache["port"] == 6379

    def test_features_section(self):
        """Features section should list feature flags as booleans."""
        config = get_configuration()
        features = config["features"]
        assert features["real_time_quotes"] is True
        assert features["options_trading"] is False

    def test_security_section(self):
        """Security section should have jwt expiration and password rules."""
        config = get_configuration()
        sec = config["security"]
        assert sec["jwt_expiration_minutes"] == 1440
        assert sec["password_min_length"] == 8

    def test_limits_section(self):
        """Limits section should define rate limits and capacity limits."""
        config = get_configuration()
        limits = config["limits"]
        assert limits["max_api_calls_per_minute"] == 60
        assert limits["max_portfolio_size"] == 100


# ===========================================================================
# apply_configuration_update
# ===========================================================================


class TestApplyConfigurationUpdate:

    def test_success_response(self):
        """Should return success status."""
        result = apply_configuration_update("features", "ml_predictions", False)
        assert result["status"] == "success"

    def test_message_contains_section_and_key(self):
        """Message should reference the updated section.key."""
        result = apply_configuration_update("limits", "max_api_calls_per_minute", 120)
        assert "limits.max_api_calls_per_minute" in result["message"]

    def test_database_section_requires_restart(self):
        """Updating the database section should require a restart."""
        result = apply_configuration_update("database", "pool_size", 30)
        assert result["requires_restart"] is True

    def test_cache_section_requires_restart(self):
        """Updating the cache section should require a restart."""
        result = apply_configuration_update("cache", "ttl_default", 600)
        assert result["requires_restart"] is True

    def test_features_section_no_restart(self):
        """Updating the features section should NOT require a restart."""
        result = apply_configuration_update("features", "crypto_trading", True)
        assert result["requires_restart"] is False

    def test_security_section_no_restart(self):
        """Updating the security section should NOT require a restart."""
        result = apply_configuration_update("security", "require_2fa", True)
        assert result["requires_restart"] is False

    def test_monitoring_section_no_restart(self):
        """Updating the monitoring section should NOT require a restart."""
        result = apply_configuration_update("monitoring", "log_level", "DEBUG")
        assert result["requires_restart"] is False


# ===========================================================================
# get_audit_logs
# ===========================================================================


class TestGetAuditLogs:

    def test_returns_list(self):
        """Should return a list of audit log entries."""
        random.seed(42)
        logs = get_audit_logs()
        assert isinstance(logs, list)

    def test_default_limit_caps_at_100(self):
        """Default limit of 100 should cap results."""
        random.seed(42)
        logs = get_audit_logs()
        assert len(logs) <= 100

    def test_custom_limit(self):
        """A smaller limit should cap results."""
        random.seed(42)
        logs = get_audit_logs(limit=10)
        assert len(logs) <= 10

    def test_sorted_newest_first(self):
        """Entries should be sorted by timestamp descending."""
        random.seed(42)
        logs = get_audit_logs()
        if len(logs) >= 2:
            for i in range(len(logs) - 1):
                assert logs[i]["timestamp"] >= logs[i + 1]["timestamp"]

    def test_filter_by_user_id(self):
        """All returned entries should match the specified user_id."""
        random.seed(42)
        uid = "specific-user-123"
        logs = get_audit_logs(user_id=uid)
        for entry in logs:
            assert entry["user_id"] == uid

    def test_filter_by_action(self):
        """All returned entries should match the specified action."""
        random.seed(42)
        logs = get_audit_logs(action="login")
        for entry in logs:
            assert entry["action"] == "login"

    def test_offset_skips_entries(self):
        """Using offset should skip early entries."""
        random.seed(42)
        all_logs = get_audit_logs(limit=200)
        random.seed(42)
        offset_logs = get_audit_logs(limit=200, offset=10)
        # With same seed, offset=10 should return 10 fewer entries
        assert len(offset_logs) == max(0, len(all_logs) - 10)

    def test_entry_has_required_fields(self):
        """Each entry should have all expected keys."""
        random.seed(42)
        logs = get_audit_logs(limit=5)
        required_keys = {
            "id", "timestamp", "user_id", "user_email", "action",
            "resource_type", "resource_id", "details", "ip_address",
            "user_agent", "success", "error_message",
        }
        for entry in logs:
            assert required_keys == set(entry.keys())

    def test_action_values_are_known(self):
        """Actions should be from the known set."""
        random.seed(42)
        known_actions = {"login", "logout", "create", "update", "delete", "export", "import"}
        logs = get_audit_logs()
        for entry in logs:
            assert entry["action"] in known_actions


# ===========================================================================
# list_announcements
# ===========================================================================


class TestListAnnouncements:

    def test_active_only_default(self):
        """Default call returns only active announcements."""
        announcements = list_announcements()
        for a in announcements:
            assert a["active"] is True

    def test_active_only_true_explicit(self):
        """Explicit active_only=True returns only active announcements."""
        announcements = list_announcements(active_only=True)
        for a in announcements:
            assert a["active"] is True

    def test_active_only_false_returns_all(self):
        """active_only=False should return all announcements (active and inactive)."""
        announcements = list_announcements(active_only=False)
        assert isinstance(announcements, list)
        assert len(announcements) >= 2

    def test_announcement_has_required_fields(self):
        """Each announcement should have all expected keys."""
        announcements = list_announcements()
        required_keys = {
            "id", "title", "message", "type", "active",
            "start_time", "end_time", "target_users",
        }
        for a in announcements:
            assert required_keys == set(a.keys())

    def test_announcement_types(self):
        """Announcement types should be 'warning' or 'info'."""
        announcements = list_announcements(active_only=False)
        for a in announcements:
            assert a["type"] in ("warning", "info")

    def test_ids_are_valid_uuids(self):
        """Announcement IDs should be valid UUIDs."""
        announcements = list_announcements()
        for a in announcements:
            parsed = uuid.UUID(a["id"])
            assert str(parsed) == a["id"]


# ===========================================================================
# initiate_data_export
# ===========================================================================


class TestInitiateDataExport:

    def test_returns_processing_status(self):
        """Export should start in 'processing' status."""
        result = initiate_data_export("users")
        assert result["status"] == "processing"

    def test_job_id_is_valid_uuid(self):
        """The job_id should be a valid UUID."""
        result = initiate_data_export("trades")
        parsed = uuid.UUID(result["job_id"])
        assert str(parsed) == result["job_id"]

    def test_download_url_contains_job_id(self):
        """The download URL should embed the job_id."""
        result = initiate_data_export("portfolios")
        assert result["job_id"] in result["download_url"]

    def test_estimated_time_is_positive(self):
        """Estimated time should be a positive integer."""
        random.seed(42)
        result = initiate_data_export("reports")
        assert isinstance(result["estimated_time_seconds"], int)
        assert result["estimated_time_seconds"] > 0

    def test_different_export_types_accepted(self):
        """Various export type strings should be accepted without error."""
        for etype in ("users", "trades", "portfolios", "audit_logs"):
            result = initiate_data_export(etype)
            assert result["status"] == "processing"


# ===========================================================================
# execute_system_command
# ===========================================================================


class TestExecuteSystemCommand:

    def test_executed_status(self):
        """Should return 'executed' status."""
        result = execute_system_command("cache_clear", 150)
        assert result["status"] == "executed"

    def test_command_echoed_back(self):
        """The command field should echo the input command."""
        result = execute_system_command("reindex", 500)
        assert result["command"] == "reindex"

    def test_result_success_true(self):
        """The result should report success=True."""
        result = execute_system_command("vacuum", 1000)
        assert result["result"]["success"] is True

    def test_result_message_contains_command(self):
        """The result message should reference the command."""
        result = execute_system_command("gc", 200)
        assert "gc" in result["result"]["message"]

    def test_execution_time_passthrough(self):
        """execution_time_ms should be passed through to the result."""
        result = execute_system_command("test", 42)
        assert result["result"]["execution_time_ms"] == 42


# ===========================================================================
# enable_maintenance_mode / disable_maintenance_mode
# ===========================================================================


class TestMaintenanceMode:

    def test_enable_default_message(self):
        """Enabling with default message should use the default."""
        result = enable_maintenance_mode()
        assert result["status"] == "maintenance_enabled"
        assert result["message"] == "System is under maintenance"

    def test_enable_custom_message(self):
        """Enabling with a custom message should use that message."""
        result = enable_maintenance_mode(message="Upgrading database")
        assert result["status"] == "maintenance_enabled"
        assert result["message"] == "Upgrading database"

    def test_disable(self):
        """Disabling maintenance mode should return operational status."""
        result = disable_maintenance_mode()
        assert result["status"] == "maintenance_disabled"
        assert result["message"] == "System is operational"

    def test_enable_then_disable_are_independent(self):
        """Enable and disable should return distinct statuses."""
        enabled = enable_maintenance_mode()
        disabled = disable_maintenance_mode()
        assert enabled["status"] != disabled["status"]
