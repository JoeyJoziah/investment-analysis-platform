"""
Unit tests for backend/compliance/sec.py

Tests cover:
- SEC constant verification (risk warning, methodology template)
- RetentionPolicy and RecommendationDocumentation dataclasses
- DataRetentionManager: get/set policies, find expired records, cleanup
- InvestmentAdviceDocumentation: document and retrieve recommendations
- FiduciaryDutyChecker: conflicts, disclosure requirements, suitability
- SECDisclosureGenerator: all static disclosure generators
"""

import uuid
from dataclasses import asdict
from datetime import datetime, timedelta, timezone

import pytest

from backend.compliance.sec import (
    SEC_STANDARD_RISK_WARNING,
    SEC_METHODOLOGY_DISCLOSURE_TEMPLATE,
    RetentionPolicy,
    RecommendationDocumentation,
    DataRetentionManager,
    InvestmentAdviceDocumentation,
    FiduciaryDutyChecker,
    SECDisclosureGenerator,
)


# ---------------------------------------------------------------------------
# SEC Constants
# ---------------------------------------------------------------------------

class TestSECConstants:
    """Verify SEC-mandated warning text contains required regulatory phrases."""

    def test_risk_warning_contains_past_performance_disclaimer(self):
        assert "Past performance does not guarantee future results" in SEC_STANDARD_RISK_WARNING

    def test_risk_warning_contains_loss_of_principal(self):
        assert "loss of principal" in SEC_STANDARD_RISK_WARNING

    def test_risk_warning_contains_risk_statement(self):
        assert "All investments involve risk" in SEC_STANDARD_RISK_WARNING

    def test_risk_warning_contains_fluctuation_warning(self):
        assert "value of investments can fluctuate" in SEC_STANDARD_RISK_WARNING

    def test_methodology_template_has_placeholders(self):
        assert "{algorithm_type}" in SEC_METHODOLOGY_DISCLOSURE_TEMPLATE
        assert "{model_version}" in SEC_METHODOLOGY_DISCLOSURE_TEMPLATE
        assert "{training_date}" in SEC_METHODOLOGY_DISCLOSURE_TEMPLATE


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

class TestRetentionPolicy:
    """Test the RetentionPolicy dataclass defaults and serialization."""

    def test_required_fields(self):
        policy = RetentionPolicy(
            data_type="trade_records",
            retention_years=6,
            legal_requirement="SEC Rule 17a-4(b)(1)",
        )
        assert policy.data_type == "trade_records"
        assert policy.retention_years == 6
        assert policy.legal_requirement == "SEC Rule 17a-4(b)(1)"

    def test_default_auto_delete_is_false(self):
        policy = RetentionPolicy(
            data_type="audit_logs",
            retention_years=7,
            legal_requirement="SOX",
        )
        assert policy.auto_delete is False

    def test_default_requires_anonymization_is_false(self):
        policy = RetentionPolicy(
            data_type="audit_logs",
            retention_years=7,
            legal_requirement="SOX",
        )
        assert policy.requires_anonymization is False

    def test_asdict_roundtrip(self):
        policy = RetentionPolicy(
            data_type="customer_communications",
            retention_years=3,
            legal_requirement="SEC Rule 17a-4(b)(4)",
            auto_delete=True,
            requires_anonymization=True,
        )
        d = asdict(policy)
        assert d["data_type"] == "customer_communications"
        assert d["retention_years"] == 3
        assert d["auto_delete"] is True
        assert d["requires_anonymization"] is True


class TestRecommendationDocumentation:
    """Test the RecommendationDocumentation dataclass."""

    def test_all_required_fields_present(self):
        now = datetime.now(timezone.utc)
        doc = RecommendationDocumentation(
            recommendation_id="rec-001",
            stock="AAPL",
            recommendation="BUY",
            analyst_id="analyst-42",
            timestamp=now,
            rationale={"score": 0.85},
            model_version="2.1.0",
            data_sources=["alpha_vantage", "finnhub"],
            risk_disclosure=SEC_STANDARD_RISK_WARNING,
            conflicts_disclosed=[],
        )
        assert doc.recommendation_id == "rec-001"
        assert doc.stock == "AAPL"
        assert doc.recommendation == "BUY"
        assert doc.analyst_id == "analyst-42"
        assert doc.timestamp == now
        assert doc.model_version == "2.1.0"
        assert len(doc.data_sources) == 2
        assert doc.conflicts_disclosed == []

    def test_asdict_contains_all_keys(self):
        now = datetime.now(timezone.utc)
        doc = RecommendationDocumentation(
            recommendation_id="rec-002",
            stock="GOOG",
            recommendation="HOLD",
            analyst_id="analyst-7",
            timestamp=now,
            rationale={},
            model_version="1.0.0",
            data_sources=[],
            risk_disclosure="test",
            conflicts_disclosed=["firm_holdings"],
        )
        d = asdict(doc)
        expected_keys = {
            "recommendation_id", "stock", "recommendation", "analyst_id",
            "timestamp", "rationale", "model_version", "data_sources",
            "risk_disclosure", "conflicts_disclosed",
        }
        assert expected_keys == set(d.keys())


# ---------------------------------------------------------------------------
# DataRetentionManager
# ---------------------------------------------------------------------------

class TestDataRetentionManager:
    """Test retention policy management and cleanup logic."""

    def test_get_default_trade_records_policy(self):
        mgr = DataRetentionManager()
        policy = mgr.get_retention_policy("trade_records")
        assert policy is not None
        assert policy["years"] == 6
        assert "17a-4" in policy["legal_requirement"]

    def test_get_default_audit_logs_policy(self):
        mgr = DataRetentionManager()
        policy = mgr.get_retention_policy("audit_logs")
        assert policy is not None
        assert policy["years"] == 7

    def test_get_default_recommendation_rationale_policy(self):
        mgr = DataRetentionManager()
        policy = mgr.get_retention_policy("recommendation_rationale")
        assert policy is not None
        assert policy["years"] == 5
        assert "204-2" in policy["legal_requirement"]

    def test_get_unknown_data_type_returns_none(self):
        mgr = DataRetentionManager()
        assert mgr.get_retention_policy("nonexistent_type") is None

    def test_set_custom_policy_overrides_default(self):
        mgr = DataRetentionManager()
        mgr.set_retention_policy("trade_records", {
            "years": 10,
            "legal_requirement": "Custom Rule",
            "auto_delete": True,
            "requires_anonymization": True,
        })
        policy = mgr.get_retention_policy("trade_records")
        assert policy["years"] == 10
        assert policy["legal_requirement"] == "Custom Rule"
        assert policy["auto_delete"] is True
        assert policy["requires_anonymization"] is True

    def test_set_custom_policy_with_defaults(self):
        mgr = DataRetentionManager()
        mgr.set_retention_policy("new_type", {})
        policy = mgr.get_retention_policy("new_type")
        assert policy is not None
        assert policy["years"] == 7  # default
        assert policy["legal_requirement"] == "Custom policy"
        assert policy["auto_delete"] is False

    def test_find_expired_records_empty_when_no_cache(self):
        mgr = DataRetentionManager()
        result = mgr.find_expired_records("trade_records")
        assert result == []

    def test_find_expired_records_returns_cached(self):
        mgr = DataRetentionManager()
        cached = [{"id": 1, "created": "2018-01-01"}]
        mgr._expired_records_cache["trade_records"] = cached
        result = mgr.find_expired_records("trade_records")
        assert result == cached

    def test_find_expired_records_unknown_type(self):
        mgr = DataRetentionManager()
        result = mgr.find_expired_records("nonexistent_type")
        assert result == []

    def test_cleanup_no_expired_records(self):
        mgr = DataRetentionManager()
        result = mgr.cleanup_expired_data("trade_records")
        assert result["records_deleted"] == 0
        assert result["status"] == "no_expired_records"

    def test_cleanup_deletes_expired_records(self):
        mgr = DataRetentionManager()
        mgr._expired_records_cache["trade_records"] = [
            {"id": 1}, {"id": 2}, {"id": 3}
        ]
        result = mgr.cleanup_expired_data("trade_records")
        assert result["records_deleted"] == 3
        assert result["action"] == "deleted"
        assert "cleanup_date" in result

    def test_cleanup_anonymizes_when_policy_requires(self):
        mgr = DataRetentionManager()
        mgr.set_retention_policy("sensitive_data", {
            "years": 5,
            "requires_anonymization": True,
        })
        mgr._expired_records_cache["sensitive_data"] = [{"id": 1}, {"id": 2}]
        result = mgr.cleanup_expired_data("sensitive_data")
        assert result["records_deleted"] == 2
        assert result["action"] == "anonymized"

    def test_all_default_policies_present(self):
        expected = {
            "trade_records", "customer_communications",
            "portfolio_statements", "audit_logs",
            "recommendation_rationale",
        }
        assert set(DataRetentionManager.DEFAULT_POLICIES.keys()) == expected


# ---------------------------------------------------------------------------
# InvestmentAdviceDocumentation
# ---------------------------------------------------------------------------

class TestInvestmentAdviceDocumentation:
    """Test recommendation documentation and retrieval."""

    def test_document_recommendation_returns_confirmation(self):
        doc_mgr = InvestmentAdviceDocumentation()
        rationale = {
            "stock": "AAPL",
            "recommendation": "BUY",
            "model_version": "2.0.0",
            "data_sources": ["finnhub"],
            "conflicts": [],
        }
        result = doc_mgr.document_recommendation("rec-100", rationale, "analyst-1")
        assert result["status"] == "documented"
        assert result["recommendation_id"] == "rec-100"
        assert "documentation_id" in result
        assert "timestamp" in result

    def test_document_recommendation_stores_for_retrieval(self):
        doc_mgr = InvestmentAdviceDocumentation()
        rationale = {
            "stock": "TSLA",
            "recommendation": "SELL",
            "model_version": "1.5.0",
            "data_sources": ["polygon"],
            "conflicts": ["firm_holdings"],
        }
        doc_mgr.document_recommendation("rec-200", rationale, "analyst-2")
        retrieved = doc_mgr.get_recommendation_documentation("rec-200")

        assert retrieved is not None
        assert retrieved["stock"] == "TSLA"
        assert retrieved["recommendation"] == "SELL"
        assert retrieved["analyst_id"] == "analyst-2"
        assert retrieved["model_version"] == "1.5.0"
        assert retrieved["data_sources"] == ["polygon"]
        assert retrieved["conflicts_disclosed"] == ["firm_holdings"]

    def test_document_recommendation_attaches_risk_disclosure(self):
        doc_mgr = InvestmentAdviceDocumentation()
        doc_mgr.document_recommendation("rec-300", {"stock": "GOOG"}, "analyst-3")
        retrieved = doc_mgr.get_recommendation_documentation("rec-300")
        assert retrieved["risk_disclosure"] == SEC_STANDARD_RISK_WARNING

    def test_document_recommendation_defaults_for_missing_fields(self):
        doc_mgr = InvestmentAdviceDocumentation()
        doc_mgr.document_recommendation("rec-400", {}, "analyst-4")
        retrieved = doc_mgr.get_recommendation_documentation("rec-400")
        assert retrieved["stock"] == "UNKNOWN"
        assert retrieved["recommendation"] == "HOLD"
        assert retrieved["model_version"] == "1.0.0"
        assert retrieved["data_sources"] == []
        assert retrieved["conflicts_disclosed"] == []

    def test_get_nonexistent_recommendation_returns_none(self):
        doc_mgr = InvestmentAdviceDocumentation()
        assert doc_mgr.get_recommendation_documentation("does-not-exist") is None

    def test_documentation_timestamp_is_utc(self):
        doc_mgr = InvestmentAdviceDocumentation()
        before = datetime.now(timezone.utc)
        doc_mgr.document_recommendation("rec-500", {"stock": "MSFT"}, "analyst-5")
        after = datetime.now(timezone.utc)
        retrieved = doc_mgr.get_recommendation_documentation("rec-500")
        ts = datetime.fromisoformat(retrieved["timestamp"])
        assert before <= ts <= after

    def test_multiple_recommendations_stored_independently(self):
        doc_mgr = InvestmentAdviceDocumentation()
        doc_mgr.document_recommendation("rec-A", {"stock": "AAPL"}, "analyst-1")
        doc_mgr.document_recommendation("rec-B", {"stock": "GOOG"}, "analyst-2")

        a = doc_mgr.get_recommendation_documentation("rec-A")
        b = doc_mgr.get_recommendation_documentation("rec-B")
        assert a["stock"] == "AAPL"
        assert b["stock"] == "GOOG"


# ---------------------------------------------------------------------------
# FiduciaryDutyChecker
# ---------------------------------------------------------------------------

class TestFiduciaryDutyChecker:
    """Test conflict of interest checks and suitability analysis."""

    def test_no_conflicts_detected(self):
        checker = FiduciaryDutyChecker()
        result = checker.check_conflicts_of_interest({
            "stock": "AAPL",
            "action": "BUY",
        })
        assert result["conflicts_detected"] is False
        assert result["conflict_count"] == 0
        assert result["disclosure_required"] is False

    def test_firm_holdings_conflict_detected(self):
        checker = FiduciaryDutyChecker()
        result = checker.check_conflicts_of_interest({
            "stock": "AAPL",
            "firm_holdings": True,
        })
        assert result["conflicts_detected"] is True
        assert result["conflict_count"] == 1
        assert "firm_holdings" in result["conflict_types"]
        assert result["disclosure_required"] is True

    def test_compensation_conflict_detected(self):
        checker = FiduciaryDutyChecker()
        result = checker.check_conflicts_of_interest({
            "stock": "GOOG",
            "receives_compensation": True,
        })
        assert result["conflicts_detected"] is True
        assert "compensation" in result["conflict_types"]

    def test_multiple_conflicts_detected(self):
        checker = FiduciaryDutyChecker()
        result = checker.check_conflicts_of_interest({
            "stock": "TSLA",
            "firm_holdings": True,
            "receives_compensation": True,
        })
        assert result["conflict_count"] == 2
        assert len(result["conflict_details"]) == 2

    def test_requires_disclosure_true_when_conflicts(self):
        checker = FiduciaryDutyChecker()
        assert checker.requires_disclosure({"firm_holdings": True}) is True

    def test_requires_disclosure_false_when_no_conflicts(self):
        checker = FiduciaryDutyChecker()
        assert checker.requires_disclosure({"stock": "AAPL"}) is False

    def test_conflict_details_severity_is_high(self):
        checker = FiduciaryDutyChecker()
        result = checker.check_conflicts_of_interest({"firm_holdings": True})
        detail = result["conflict_details"][0]
        assert detail["severity"] == "high"
        assert detail["requires_disclosure"] is True

    def test_suitability_matching_risk_tolerance(self):
        checker = FiduciaryDutyChecker()
        result = checker.analyze_suitability(
            {"action": "BUY", "risk_level": "aggressive", "stock": "TSLA"},
            {"risk_tolerance": "aggressive", "investment_objective": "growth", "time_horizon": "long_term"},
        )
        assert result["suitable"] is True
        assert result["suitability_score"] >= 0.6
        assert result["factors"]["risk_alignment"] == 1.0

    def test_suitability_mismatched_risk_conservative_vs_aggressive(self):
        checker = FiduciaryDutyChecker()
        result = checker.analyze_suitability(
            {"action": "BUY", "risk_level": "aggressive", "stock": "TSLA"},
            {"risk_tolerance": "conservative", "investment_objective": "preservation", "time_horizon": "short_term"},
        )
        # Conservative client + aggressive recommendation = low risk alignment
        assert result["factors"]["risk_alignment"] < 0.5

    def test_suitability_returns_client_profile(self):
        checker = FiduciaryDutyChecker()
        result = checker.analyze_suitability(
            {"action": "HOLD", "risk_level": "moderate", "stock": "AAPL"},
            {"risk_tolerance": "moderate", "investment_objective": "income", "time_horizon": "medium_term"},
        )
        assert result["client_profile"]["risk_tolerance"] == "moderate"
        assert result["client_profile"]["investment_objective"] == "income"
        assert result["recommendation_characteristics"]["stock"] == "AAPL"

    def test_suitability_score_is_rounded(self):
        checker = FiduciaryDutyChecker()
        result = checker.analyze_suitability(
            {"action": "BUY", "risk_level": "moderate", "stock": "AAPL"},
            {"risk_tolerance": "moderate"},
        )
        # Scores are rounded to 3 decimal places
        score_str = str(result["suitability_score"])
        decimal_places = len(score_str.split(".")[-1]) if "." in score_str else 0
        assert decimal_places <= 3

    def test_suitability_growth_objective_favors_buy(self):
        checker = FiduciaryDutyChecker()
        buy_result = checker.analyze_suitability(
            {"action": "BUY", "risk_level": "moderate", "stock": "X"},
            {"risk_tolerance": "moderate", "investment_objective": "growth"},
        )
        sell_result = checker.analyze_suitability(
            {"action": "SELL", "risk_level": "moderate", "stock": "X"},
            {"risk_tolerance": "moderate", "investment_objective": "growth"},
        )
        assert buy_result["factors"]["objective_alignment"] >= sell_result["factors"]["objective_alignment"]

    def test_suitability_preservation_objective_favors_sell_or_hold(self):
        checker = FiduciaryDutyChecker()
        sell_result = checker.analyze_suitability(
            {"action": "SELL", "risk_level": "moderate", "stock": "X"},
            {"risk_tolerance": "moderate", "investment_objective": "preservation"},
        )
        buy_result = checker.analyze_suitability(
            {"action": "BUY", "risk_level": "moderate", "stock": "X"},
            {"risk_tolerance": "moderate", "investment_objective": "preservation"},
        )
        assert sell_result["factors"]["objective_alignment"] > buy_result["factors"]["objective_alignment"]

    def test_conflict_types_class_attribute(self):
        assert "firm_holdings" in FiduciaryDutyChecker.CONFLICT_TYPES
        assert "compensation" in FiduciaryDutyChecker.CONFLICT_TYPES
        assert "relationship" in FiduciaryDutyChecker.CONFLICT_TYPES
        assert "personal_holdings" in FiduciaryDutyChecker.CONFLICT_TYPES


# ---------------------------------------------------------------------------
# SECDisclosureGenerator
# ---------------------------------------------------------------------------

class TestSECDisclosureGenerator:
    """Test all static disclosure generators."""

    def test_generate_methodology_disclosure_default_args(self):
        result = SECDisclosureGenerator.generate_methodology_disclosure()
        assert "ML-powered quantitative" in result
        assert "1.0.0" in result
        # Default training_date is today
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        assert today_str in result

    def test_generate_methodology_disclosure_custom_args(self):
        result = SECDisclosureGenerator.generate_methodology_disclosure(
            algorithm_type="Deep Learning",
            model_version="3.2.1",
            training_date="2025-12-01",
        )
        assert "Deep Learning" in result
        assert "3.2.1" in result
        assert "2025-12-01" in result

    def test_generate_risk_warning_matches_constant(self):
        assert SECDisclosureGenerator.generate_risk_warning() == SEC_STANDARD_RISK_WARNING

    def test_generate_limitations_statement_content(self):
        result = SECDisclosureGenerator.generate_limitations_statement()
        assert "individual financial situation" in result
        assert "tax implications" in result
        assert "real-time market conditions" in result
        assert "non-public information" in result
        assert "geopolitical events" in result

    def test_generate_conflict_disclosure_no_conflicts(self):
        result = SECDisclosureGenerator.generate_conflict_disclosure(has_conflicts=False)
        assert "does not hold positions" in result

    def test_generate_conflict_disclosure_with_conflicts(self):
        result = SECDisclosureGenerator.generate_conflict_disclosure(has_conflicts=True)
        assert "CONFLICT DISCLOSURE" in result
        assert "may hold positions" in result

    def test_generate_conflict_disclosure_default_is_no_conflicts(self):
        result = SECDisclosureGenerator.generate_conflict_disclosure()
        assert "does not hold positions" in result
