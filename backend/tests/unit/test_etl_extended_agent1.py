"""
Unit tests for ETL modules: data_extractor.py and data_validation_pipeline.py.

Uses importlib bypass to avoid backend.etl.__init__.py import chain
which pulls in selenium, aiohttp, and other heavy dependencies.
"""

import asyncio
import importlib
import sys
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Stub ALL external dependencies before loading ETL modules.
# The import chain: data_extractor -> unlimited_extractor_with_fallbacks
#   -> unlimited_data_extractor (aiohttp, bs4, selenium, yfinance, requests)
#   -> intelligent_cache_system (aiofiles, redis, cachetools, psutil)
#   -> data_validation_pipeline (pure stdlib + pandas/numpy, no stubs needed)
# ---------------------------------------------------------------------------

_etl_dir = Path(__file__).resolve().parents[2] / "etl"

# Stub heavy deps used by unlimited_data_extractor
sys.modules.setdefault("aiohttp", MagicMock())
sys.modules.setdefault("bs4", MagicMock())
sys.modules.setdefault("selenium", MagicMock())
sys.modules.setdefault("selenium.webdriver", MagicMock())
sys.modules.setdefault("selenium.webdriver.chrome", MagicMock())
sys.modules.setdefault("selenium.webdriver.chrome.options", MagicMock())
sys.modules.setdefault("selenium.webdriver.common", MagicMock())
sys.modules.setdefault("selenium.webdriver.common.by", MagicMock())
sys.modules.setdefault("selenium.webdriver.support", MagicMock())
sys.modules.setdefault("selenium.webdriver.support.ui", MagicMock())
sys.modules.setdefault("selenium.webdriver.support.expected_conditions", MagicMock())
sys.modules.setdefault("yfinance", MagicMock())

# Stub deps used by intelligent_cache_system
sys.modules.setdefault("aiofiles", MagicMock())
sys.modules.setdefault("redis", MagicMock())
sys.modules.setdefault("cachetools", MagicMock())
sys.modules.setdefault("psutil", MagicMock())

# Stub dotenv (used by data_extractor itself)
sys.modules.setdefault("dotenv", MagicMock())

# ---------------------------------------------------------------------------
# Load data_validation_pipeline.py directly (no heavy deps)
# ---------------------------------------------------------------------------
_dvp_spec = importlib.util.spec_from_file_location(
    "data_validation_pipeline", _etl_dir / "data_validation_pipeline.py"
)
_dvp = importlib.util.module_from_spec(_dvp_spec)
_dvp_spec.loader.exec_module(_dvp)

ValidationLevel = _dvp.ValidationLevel
DataQualityIssue = _dvp.DataQualityIssue
ValidationRule = _dvp.ValidationRule
ValidationResult = _dvp.ValidationResult
DataQualityScore = _dvp.DataQualityScore
FinancialDataValidator = _dvp.FinancialDataValidator
validate_extraction_results = _dvp.validate_extraction_results
validate_batch_data = _dvp.validate_batch_data

# ---------------------------------------------------------------------------
# Load data_extractor.py -- requires stubs for its relative imports.
# We inject the already-loaded validation pipeline as the relative module,
# and mock the unlimited_extractor_with_fallbacks dependency.
# ---------------------------------------------------------------------------

# Create mock for the unlimited_extractor_with_fallbacks sibling module.
# data_extractor.py does: from .unlimited_extractor_with_fallbacks import ...
# and: from .data_validation_pipeline import ValidationLevel
_mock_unlimited = MagicMock()
_mock_unlimited.UnlimitedStockDataExtractor = MagicMock
_mock_unlimited.ExtractionResult = MagicMock()
_mock_unlimited.StockData = MagicMock()

# Build a fake "backend.etl" package so relative imports resolve.
import types as _types

_fake_backend = _types.ModuleType("backend")
_fake_backend.__path__ = []
_fake_etl = _types.ModuleType("backend.etl")
_fake_etl.__path__ = [str(_etl_dir)]
_fake_etl.__package__ = "backend.etl"

# Register stubs in sys.modules BEFORE exec_module
sys.modules["backend"] = sys.modules.get("backend", _fake_backend)
sys.modules["backend.etl"] = _fake_etl
sys.modules["backend.etl.unlimited_extractor_with_fallbacks"] = _mock_unlimited
sys.modules["backend.etl.data_validation_pipeline"] = _dvp

_de_spec = importlib.util.spec_from_file_location(
    "backend.etl.data_extractor",
    _etl_dir / "data_extractor.py",
)
_de = importlib.util.module_from_spec(_de_spec)
_de.__package__ = "backend.etl"
_de_spec.loader.exec_module(_de)

RateLimitConfig_DE = _de.RateLimitConfig
DataSourceConfig = _de.DataSourceConfig
MultiSourceDataExtractor = _de.MultiSourceDataExtractor
DataExtractor = _de.DataExtractor
DataValidator = _de.DataValidator
create_unlimited_extractor = _de.create_unlimited_extractor
print_migration_guide = _de.print_migration_guide


# ==========================================================================
# data_validation_pipeline.py
# ==========================================================================


class TestValidationLevel:
    """ValidationLevel enum coverage."""

    def test_basic_value(self):
        assert ValidationLevel.BASIC.value == "basic"

    def test_standard_value(self):
        assert ValidationLevel.STANDARD.value == "standard"

    def test_strict_value(self):
        assert ValidationLevel.STRICT.value == "strict"

    def test_comprehensive_value(self):
        assert ValidationLevel.COMPREHENSIVE.value == "comprehensive"

    def test_member_count(self):
        assert len(ValidationLevel) == 4


class TestDataQualityIssue:
    """DataQualityIssue enum coverage."""

    def test_all_members(self):
        expected = {
            "missing_required_field",
            "invalid_format",
            "out_of_range",
            "inconsistent_data",
            "suspicious_value",
            "stale_data",
            "incomplete_data",
            "duplicate_data",
        }
        actual = {member.value for member in DataQualityIssue}
        assert actual == expected

    def test_member_count(self):
        assert len(DataQualityIssue) == 8


class TestValidationRule:
    """ValidationRule dataclass coverage."""

    def test_defaults(self):
        rule = ValidationRule(field_name="price", rule_type="range")
        assert rule.field_name == "price"
        assert rule.rule_type == "range"
        assert rule.parameters == {}
        assert rule.severity == "error"
        assert rule.description == ""
        assert rule.validation_function is None

    def test_custom_params(self):
        fn = lambda data: None  # noqa: E731
        rule = ValidationRule(
            field_name="ticker",
            rule_type="custom",
            parameters={"min": 1},
            severity="warning",
            description="check ticker",
            validation_function=fn,
        )
        assert rule.parameters == {"min": 1}
        assert rule.severity == "warning"
        assert rule.validation_function is fn


class TestValidationResult:
    """ValidationResult dataclass coverage."""

    def test_defaults(self):
        vr = ValidationResult(
            field_name="ticker",
            issue_type=DataQualityIssue.MISSING_REQUIRED,
            severity="error",
            message="missing",
        )
        assert vr.suggested_fix is None
        assert vr.original_value is None
        assert vr.corrected_value is None

    def test_with_fix(self):
        vr = ValidationResult(
            field_name="price",
            issue_type=DataQualityIssue.OUT_OF_RANGE,
            severity="warning",
            message="price too high",
            suggested_fix="lower it",
            original_value=99999,
            corrected_value=100,
        )
        assert vr.suggested_fix == "lower it"
        assert vr.original_value == 99999
        assert vr.corrected_value == 100


class TestDataQualityScore:
    """DataQualityScore dataclass and pass_rate property."""

    def test_pass_rate_no_fields(self):
        score = DataQualityScore(
            overall_score=0,
            completeness_score=0,
            accuracy_score=0,
            consistency_score=0,
            timeliness_score=0,
            total_fields_checked=0,
            valid_fields=0,
        )
        assert score.pass_rate == 0.0

    def test_pass_rate_all_valid(self):
        score = DataQualityScore(
            overall_score=100,
            completeness_score=100,
            accuracy_score=100,
            consistency_score=100,
            timeliness_score=100,
            total_fields_checked=10,
            valid_fields=10,
        )
        assert score.pass_rate == 1.0

    def test_pass_rate_partial(self):
        score = DataQualityScore(
            overall_score=50,
            completeness_score=50,
            accuracy_score=50,
            consistency_score=50,
            timeliness_score=50,
            total_fields_checked=4,
            valid_fields=3,
        )
        assert score.pass_rate == pytest.approx(0.75)

    def test_issues_default(self):
        score = DataQualityScore(
            overall_score=80,
            completeness_score=80,
            accuracy_score=80,
            consistency_score=80,
            timeliness_score=80,
        )
        assert score.issues == []
        assert score.total_fields_checked == 0
        assert score.valid_fields == 0


class TestFinancialDataValidator:
    """FinancialDataValidator class tests."""

    def test_init_basic(self):
        v = FinancialDataValidator(ValidationLevel.BASIC)
        assert v.validation_level == ValidationLevel.BASIC
        assert v.cleaning_enabled is True
        assert isinstance(v.field_statistics, defaultdict)
        assert isinstance(v.common_issues, Counter)

    def test_init_standard_rules(self):
        v = FinancialDataValidator(ValidationLevel.STANDARD)
        assert "ticker" in v.validation_rules
        assert "current_price" in v.validation_rules
        assert "market_cap" in v.validation_rules

    def test_init_strict_has_beta_rules(self):
        v = FinancialDataValidator(ValidationLevel.STRICT)
        assert "beta" in v.validation_rules
        assert "dividend_yield" in v.validation_rules

    def test_init_comprehensive_has_consistency(self):
        v = FinancialDataValidator(ValidationLevel.COMPREHENSIVE)
        assert "consistency_checks" in v.validation_rules

    def test_market_ranges_structure(self):
        v = FinancialDataValidator()
        ranges = v.market_data_ranges
        assert "price_ranges" in ranges
        assert "volume_ranges" in ranges
        assert "market_cap_ranges" in ranges
        assert ranges["price_ranges"]["penny_stock_max"] == 5.0

    @pytest.mark.asyncio
    async def test_validate_good_data(self):
        v = FinancialDataValidator(ValidationLevel.STANDARD)
        data = {
            "ticker": "AAPL",
            "current_price": 150.0,
            "volume": 50_000_000,
            "market_cap": 2_500_000_000_000,
            "pe_ratio": 25.0,
            "timestamp": datetime.now().isoformat(),
        }
        score = await v.validate_stock_data(data)
        assert score.overall_score > 50
        assert score.completeness_score > 0
        assert isinstance(score.issues, list)

    @pytest.mark.asyncio
    async def test_validate_missing_ticker(self):
        v = FinancialDataValidator(ValidationLevel.BASIC)
        data = {"current_price": 100.0}
        score = await v.validate_stock_data(data)
        missing_issues = [
            i for i in score.issues
            if i.issue_type == DataQualityIssue.MISSING_REQUIRED
        ]
        assert len(missing_issues) >= 1

    @pytest.mark.asyncio
    async def test_validate_invalid_ticker_format(self):
        v = FinancialDataValidator(ValidationLevel.BASIC)
        data = {"ticker": "invalid_lower_123"}
        score = await v.validate_stock_data(data)
        format_issues = [
            i for i in score.issues
            if i.issue_type == DataQualityIssue.INVALID_FORMAT
        ]
        assert len(format_issues) >= 1

    @pytest.mark.asyncio
    async def test_validate_out_of_range_price(self):
        v = FinancialDataValidator(ValidationLevel.STANDARD)
        data = {"ticker": "TEST", "current_price": -5.0}
        score = await v.validate_stock_data(data)
        range_issues = [
            i for i in score.issues
            if i.issue_type == DataQualityIssue.OUT_OF_RANGE
        ]
        assert len(range_issues) >= 1

    @pytest.mark.asyncio
    async def test_validate_non_numeric_price(self):
        v = FinancialDataValidator(ValidationLevel.STANDARD)
        data = {"ticker": "TEST", "current_price": "not_a_number"}
        score = await v.validate_stock_data(data)
        format_issues = [
            i for i in score.issues
            if i.issue_type == DataQualityIssue.INVALID_FORMAT
        ]
        assert len(format_issues) >= 1

    @pytest.mark.asyncio
    async def test_validate_comprehensive_price_consistency(self):
        v = FinancialDataValidator(ValidationLevel.COMPREHENSIVE)
        data = {
            "ticker": "TEST",
            "current_price": 200.0,
            "day_high": 150.0,
            "day_low": 100.0,
        }
        score = await v.validate_stock_data(data)
        # current_price outside day range -> inconsistency
        inconsistency_issues = [
            i for i in score.issues
            if i.issue_type == DataQualityIssue.INCONSISTENT
        ]
        assert len(inconsistency_issues) >= 1

    @pytest.mark.asyncio
    async def test_validate_day_range_inverted(self):
        v = FinancialDataValidator(ValidationLevel.COMPREHENSIVE)
        data = {
            "ticker": "TEST",
            "current_price": 110.0,
            "day_high": 100.0,
            "day_low": 120.0,
        }
        score = await v.validate_stock_data(data)
        issues = [i for i in score.issues if "day range" in i.message.lower() or "outside" in i.message.lower()]
        assert len(issues) >= 1

    @pytest.mark.asyncio
    async def test_validate_extreme_price_movement(self):
        v = FinancialDataValidator(ValidationLevel.COMPREHENSIVE)
        data = {
            "ticker": "TEST",
            "current_price": 300.0,
            "previous_close": 100.0,
        }
        score = await v.validate_stock_data(data)
        suspicious = [
            i for i in score.issues
            if i.issue_type == DataQualityIssue.SUSPICIOUS
        ]
        assert len(suspicious) >= 1

    @pytest.mark.asyncio
    async def test_validate_pe_ratio_inconsistency(self):
        v = FinancialDataValidator(ValidationLevel.COMPREHENSIVE)
        data = {
            "ticker": "TEST",
            "current_price": 100.0,
            "pe_ratio": 50.0,
            "eps": 10.0,  # calculated PE = 100/10 = 10, far from 50
        }
        score = await v.validate_stock_data(data)
        inconsistent = [
            i for i in score.issues
            if i.issue_type == DataQualityIssue.INCONSISTENT
            and "P/E" in i.message
        ]
        assert len(inconsistent) >= 1

    def test_timeliness_score_recent(self):
        v = FinancialDataValidator()
        data = {"timestamp": datetime.now().isoformat()}
        score = v._calculate_timeliness_score(data)
        assert score == 100

    def test_timeliness_score_old(self):
        v = FinancialDataValidator()
        old_time = datetime.now() - timedelta(days=30)
        data = {"timestamp": old_time.isoformat()}
        score = v._calculate_timeliness_score(data)
        assert score == 20

    def test_timeliness_score_no_timestamp(self):
        v = FinancialDataValidator()
        score = v._calculate_timeliness_score({})
        assert score == 50

    def test_update_field_statistics(self):
        v = FinancialDataValidator()
        v._update_field_statistics({"price": 100.0, "volume": 5000})
        assert len(v.field_statistics["price"]) == 1
        assert v.field_statistics["price"][0] == 100.0

    def test_update_field_statistics_caps_at_1000(self):
        v = FinancialDataValidator()
        for i in range(1050):
            v._update_field_statistics({"price": float(i)})
        assert len(v.field_statistics["price"]) == 1000

    def test_get_validation_statistics(self):
        v = FinancialDataValidator(ValidationLevel.STANDARD)
        v._update_field_statistics({"price": 100.0, "volume": 5000.0})
        v._update_field_statistics({"price": 200.0, "volume": 6000.0})
        stats = v.get_validation_statistics()
        assert stats["validation_level"] == "standard"
        assert "field_statistics" in stats
        assert "rules_count" in stats
        assert stats["field_statistics"]["price"]["mean"] == 150.0

    @pytest.mark.asyncio
    async def test_clean_and_correct_disabled(self):
        v = FinancialDataValidator()
        v.cleaning_enabled = False
        data = {"ticker": "test"}
        qs = DataQualityScore(
            overall_score=50, completeness_score=50, accuracy_score=50,
            consistency_score=50, timeliness_score=50,
            issues=[
                ValidationResult(
                    field_name="ticker",
                    issue_type=DataQualityIssue.INVALID_FORMAT,
                    severity="error",
                    message="bad format",
                    original_value="test",
                )
            ],
        )
        result = await v.clean_and_correct_data(data, qs)
        # Should return data unchanged
        assert result["ticker"] == "test"

    @pytest.mark.asyncio
    async def test_fix_ticker_format(self):
        v = FinancialDataValidator()
        data = {"ticker": "aapl123"}
        issue = ValidationResult(
            field_name="ticker",
            issue_type=DataQualityIssue.INVALID_FORMAT,
            severity="error",
            message="bad ticker",
            original_value="aapl123",
        )
        correction = await v._fix_format_issue(data, issue)
        assert correction is not None
        assert data["ticker"] == "AAPL"

    @pytest.mark.asyncio
    async def test_fix_price_format_from_string(self):
        v = FinancialDataValidator()
        data = {"current_price": "$1,234.56"}
        issue = ValidationResult(
            field_name="current_price",
            issue_type=DataQualityIssue.INVALID_FORMAT,
            severity="error",
            message="not numeric",
            original_value="$1,234.56",
        )
        correction = await v._fix_format_issue(data, issue)
        assert correction is not None
        assert data["current_price"] == pytest.approx(1234.56)

    @pytest.mark.asyncio
    async def test_fix_price_from_cents(self):
        v = FinancialDataValidator()
        data = {"current_price": 15000}
        issue = ValidationResult(
            field_name="current_price",
            issue_type=DataQualityIssue.OUT_OF_RANGE,
            severity="error",
            message="out of range",
            original_value=15000,
        )
        correction = await v._fix_range_issue(data, issue)
        assert correction is not None
        assert data["current_price"] == 150.0

    @pytest.mark.asyncio
    async def test_fix_volume_from_millions(self):
        v = FinancialDataValidator()
        data = {"volume": 50}
        issue = ValidationResult(
            field_name="volume",
            issue_type=DataQualityIssue.OUT_OF_RANGE,
            severity="error",
            message="out of range",
            original_value=50,
        )
        correction = await v._fix_range_issue(data, issue)
        assert correction is not None
        assert data["volume"] == 50_000_000

    @pytest.mark.asyncio
    async def test_fix_missing_ticker_from_symbol(self):
        v = FinancialDataValidator()
        data = {"symbol": "MSFT"}
        issue = ValidationResult(
            field_name="ticker",
            issue_type=DataQualityIssue.MISSING_REQUIRED,
            severity="error",
            message="missing",
        )
        correction = await v._fix_missing_data(data, issue)
        assert data["ticker"] == "MSFT"

    @pytest.mark.asyncio
    async def test_fix_missing_price_from_close(self):
        v = FinancialDataValidator()
        data = {"close": 250.0}
        issue = ValidationResult(
            field_name="current_price",
            issue_type=DataQualityIssue.MISSING_REQUIRED,
            severity="error",
            message="missing",
        )
        correction = await v._fix_missing_data(data, issue)
        assert data["current_price"] == 250.0


class TestValidateExtractionResults:
    """Tests for the synchronous validate_extraction_results function."""

    def test_empty_list(self):
        result = validate_extraction_results([], min_quality_score=0.0)
        assert result["total_records"] == 0
        assert result["valid_records"] == 0
        assert result["pass_rate"] == 0

    def test_good_data_passes(self):
        results = [
            {
                "ticker": "AAPL",
                "current_price": 150.0,
                "volume": 50_000_000,
                "market_cap": 2_500_000_000_000,
                "timestamp": datetime.now().isoformat(),
            }
        ]
        out = validate_extraction_results(results, min_quality_score=30.0)
        assert out["total_records"] == 1
        assert out["valid_records"] >= 1
        assert "quality_scores" in out

    def test_bad_data_filtered(self):
        results = [
            {"not_a_ticker": True, "volume": "abc"},
        ]
        out = validate_extraction_results(results, min_quality_score=90.0)
        # The broken record should not pass a 90% threshold
        assert out["valid_records"] == 0
        assert out["filtered_records"] == 1

    def test_mixed_batch(self):
        results = [
            {
                "ticker": "AAPL",
                "current_price": 150.0,
                "volume": 50_000_000,
                "timestamp": datetime.now().isoformat(),
            },
            {"bad_field": True},
        ]
        out = validate_extraction_results(results, min_quality_score=40.0)
        assert out["total_records"] == 2
        assert 0 < out["avg_quality_score"]


# ==========================================================================
# data_extractor.py
# ==========================================================================


class TestRateLimitConfigDE:
    """Backward-compat RateLimitConfig in data_extractor.py."""

    def test_deprecated_init(self):
        cfg = RateLimitConfig_DE()
        # Constructor should succeed (no-op)
        assert cfg is not None

    def test_with_args(self):
        cfg = RateLimitConfig_DE("some", "args", key="val")
        assert cfg is not None


class TestDataSourceConfig:
    """Backward-compat DataSourceConfig."""

    def test_deprecated_init(self):
        dsc = DataSourceConfig()
        assert dsc is not None

    def test_with_args(self):
        dsc = DataSourceConfig("x", y=1)
        assert dsc is not None


class TestDataValidator:
    """DataValidator static validation logic."""

    def test_valid_data_with_ticker(self):
        data = {"ticker": "AAPL", "current_price": 150.0, "data_quality_score": 80}
        assert DataValidator.validate_stock_data(data) is True

    def test_missing_ticker(self):
        data = {"current_price": 100.0}
        assert DataValidator.validate_stock_data(data) is False

    def test_not_a_dict(self):
        assert DataValidator.validate_stock_data("string") is False
        assert DataValidator.validate_stock_data(None) is False
        assert DataValidator.validate_stock_data(42) is False

    def test_extraction_success_flag_true(self):
        data = {"ticker": "MSFT", "extraction_success": True}
        assert DataValidator.validate_stock_data(data) is True

    def test_extraction_success_flag_false(self):
        data = {"ticker": "MSFT", "extraction_success": False}
        assert DataValidator.validate_stock_data(data) is False

    def test_negative_price(self):
        data = {"ticker": "BAD", "current_price": -10.0}
        assert DataValidator.validate_stock_data(data) is False

    def test_zero_price(self):
        data = {"ticker": "ZERO", "current_price": 0}
        assert DataValidator.validate_stock_data(data) is False

    def test_non_numeric_price(self):
        data = {"ticker": "NAN", "current_price": "not_a_num"}
        assert DataValidator.validate_stock_data(data) is False

    def test_low_quality_score(self):
        data = {"ticker": "LOW", "data_quality_score": 30}
        assert DataValidator.validate_stock_data(data) is False

    def test_high_quality_score(self):
        data = {"ticker": "HI", "data_quality_score": 90}
        assert DataValidator.validate_stock_data(data) is True

    def test_boundary_quality_score_50(self):
        data = {"ticker": "MID", "data_quality_score": 50}
        assert DataValidator.validate_stock_data(data) is True

    def test_ticker_only(self):
        data = {"ticker": "ONLY"}
        assert DataValidator.validate_stock_data(data) is True


class TestMultiSourceDataExtractor:
    """MultiSourceDataExtractor backward-compat wrapper."""

    def test_init_sets_attributes(self):
        ext = MultiSourceDataExtractor(cache_dir="/tmp/test_cache")
        assert ext.cache_dir == "/tmp/test_cache"
        assert ext.data_sources == {}
        assert ext.api_keys == {}
        assert ext.call_history == {}
        assert ext.backoff_delays == {}

    def test_check_rate_limit_always_true(self):
        ext = MultiSourceDataExtractor()
        assert ext.check_rate_limit("yahoo") is True
        assert ext.check_rate_limit("anything") is True

    def test_cleanup_no_error(self):
        ext = MultiSourceDataExtractor()
        # cleanup should not raise
        ext.cleanup()

    @pytest.mark.asyncio
    async def test_extract_stock_data_exception(self):
        ext = MultiSourceDataExtractor()
        ext.unlimited_extractor = MagicMock()
        ext.unlimited_extractor.extract_stock_data = AsyncMock(
            side_effect=Exception("network down")
        )
        result = await ext.extract_stock_data("AAPL")
        assert result["extraction_success"] is False
        assert "network down" in result["error"]
        assert result["ticker"] == "AAPL"

    @pytest.mark.asyncio
    async def test_extract_stock_data_success(self):
        ext = MultiSourceDataExtractor()
        mock_data = MagicMock()
        mock_data.to_dict.return_value = {"ticker": "AAPL", "current_price": 150.0}
        mock_data.source = "yahoo"
        mock_data.data_quality_score = 85

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.data = mock_data
        mock_result.extraction_time_ms = 42
        mock_result.cache_hit = False

        ext.unlimited_extractor = MagicMock()
        ext.unlimited_extractor.extract_stock_data = AsyncMock(return_value=mock_result)

        result = await ext.extract_stock_data("AAPL")
        assert result["extraction_success"] is True
        assert result["extraction_source"] == "yahoo"
        assert result["extraction_time_ms"] == 42

    @pytest.mark.asyncio
    async def test_extract_stock_data_failure_result(self):
        ext = MultiSourceDataExtractor()
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.data = None
        mock_result.error = "source unavailable"
        mock_result.source = "yahoo"
        mock_result.extraction_time_ms = 100
        mock_result.ticker = "FAIL"

        ext.unlimited_extractor = MagicMock()
        ext.unlimited_extractor.extract_stock_data = AsyncMock(return_value=mock_result)

        result = await ext.extract_stock_data("FAIL")
        assert result["extraction_success"] is False
        assert result["error"] == "source unavailable"

    @pytest.mark.asyncio
    async def test_batch_extract_exception(self):
        ext = MultiSourceDataExtractor()
        ext.unlimited_extractor = MagicMock()
        ext.unlimited_extractor.extract_bulk_data = AsyncMock(
            side_effect=RuntimeError("bulk fail")
        )
        results = await ext.batch_extract(["AAPL", "MSFT"])
        assert len(results) == 2
        for r in results:
            assert r["extraction_success"] is False
            assert "bulk fail" in r["error"]

    def test_get_extraction_stats_error(self):
        ext = MultiSourceDataExtractor()
        ext.unlimited_extractor = MagicMock()
        ext.unlimited_extractor.get_comprehensive_stats = MagicMock(
            side_effect=Exception("stats fail")
        )
        stats = ext.get_extraction_stats()
        assert "error" in stats


class TestDataExtractor:
    """DataExtractor subclass (inherits MultiSourceDataExtractor)."""

    def test_inherits_multi_source(self):
        de = DataExtractor(cache_dir="/tmp/de_test")
        assert isinstance(de, MultiSourceDataExtractor)
        assert de.cache_dir == "/tmp/de_test"

    @pytest.mark.asyncio
    async def test_fetch_stock_data_delegates(self):
        de = DataExtractor()
        mock_data = MagicMock()
        mock_data.to_dict.return_value = {"ticker": "GOOG"}
        mock_data.source = "sec"
        mock_data.data_quality_score = 70

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.data = mock_data
        mock_result.extraction_time_ms = 10
        mock_result.cache_hit = True

        de.unlimited_extractor = MagicMock()
        de.unlimited_extractor.extract_stock_data = AsyncMock(return_value=mock_result)

        result = await de.fetch_stock_data("GOOG")
        assert result["ticker"] == "GOOG"
        assert result["cache_hit"] is True


class TestFactoryFunctions:
    """Tests for module-level factory/convenience functions."""

    def test_create_unlimited_extractor(self):
        ext = create_unlimited_extractor(cache_dir="/tmp/factory_test")
        assert isinstance(ext, DataExtractor)
        assert ext.cache_dir == "/tmp/factory_test"

    def test_print_migration_guide(self, capsys):
        print_migration_guide()
        captured = capsys.readouterr()
        assert "MIGRATION" in captured.out
        assert "UNLIMITED" in captured.out
