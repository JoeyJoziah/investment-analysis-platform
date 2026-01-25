"""
SEC 2025 Compliance Services

Implements SEC requirements for algorithmic investment recommendations:
- Investment Adviser Act compliance
- Form ADV disclosure requirements
- Fiduciary duty obligations
- Data retention policies (7+ years)
- Audit trail requirements
"""

import logging
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


# SEC 2025 Required Disclosures
SEC_STANDARD_RISK_WARNING = (
    "IMPORTANT: Past performance does not guarantee future results. All investments "
    "involve risk, including possible loss of principal. The value of investments can "
    "fluctuate, and investors may not get back the amount originally invested."
)

SEC_METHODOLOGY_DISCLOSURE_TEMPLATE = (
    "This recommendation was generated using {algorithm_type} analysis. "
    "Model version: {model_version}. Last training date: {training_date}."
)


@dataclass
class RetentionPolicy:
    """Data retention policy configuration"""
    data_type: str
    retention_years: int
    legal_requirement: str
    auto_delete: bool = False
    requires_anonymization: bool = False


@dataclass
class RecommendationDocumentation:
    """SEC-required documentation for investment recommendations"""
    recommendation_id: str
    stock: str
    recommendation: str
    analyst_id: str
    timestamp: datetime
    rationale: Dict[str, Any]
    model_version: str
    data_sources: List[str]
    risk_disclosure: str
    conflicts_disclosed: List[str]


class DataRetentionManager:
    """
    Manages SEC-compliant data retention policies.

    SEC Rule 17a-4 and 17a-3 require:
    - Trade records: 6 years
    - Customer communications: 3 years
    - Portfolio statements: 3 years
    - Audit logs: 7 years (best practice)
    - Investment advice rationale: 5 years
    """

    # Default retention policies per SEC requirements
    DEFAULT_POLICIES = {
        "trade_records": RetentionPolicy(
            data_type="trade_records",
            retention_years=6,
            legal_requirement="SEC Rule 17a-4(b)(1)"
        ),
        "customer_communications": RetentionPolicy(
            data_type="customer_communications",
            retention_years=3,
            legal_requirement="SEC Rule 17a-4(b)(4)"
        ),
        "portfolio_statements": RetentionPolicy(
            data_type="portfolio_statements",
            retention_years=3,
            legal_requirement="SEC Rule 17a-4(b)(6)"
        ),
        "audit_logs": RetentionPolicy(
            data_type="audit_logs",
            retention_years=7,
            legal_requirement="SEC best practice / SOX compliance"
        ),
        "recommendation_rationale": RetentionPolicy(
            data_type="recommendation_rationale",
            retention_years=5,
            legal_requirement="Investment Advisers Act Rule 204-2"
        )
    }

    def __init__(self):
        self._policies: Dict[str, RetentionPolicy] = {}
        self._expired_records_cache: Dict[str, List[Dict]] = {}

    def set_retention_policy(self, data_type: str, policy: Dict[str, Any]) -> None:
        """Set retention policy for a data type"""
        years = policy.get("years", 7)
        legal_req = policy.get("legal_requirement", "Custom policy")

        self._policies[data_type] = RetentionPolicy(
            data_type=data_type,
            retention_years=years,
            legal_requirement=legal_req,
            auto_delete=policy.get("auto_delete", False),
            requires_anonymization=policy.get("requires_anonymization", False)
        )

        logger.info(f"Retention policy set for {data_type}: {years} years")

    def get_retention_policy(self, data_type: str) -> Optional[Dict[str, Any]]:
        """Get retention policy for a data type"""
        policy = self._policies.get(data_type) or self.DEFAULT_POLICIES.get(data_type)

        if policy:
            return {
                "data_type": policy.data_type,
                "years": policy.retention_years,
                "legal_requirement": policy.legal_requirement,
                "auto_delete": policy.auto_delete,
                "requires_anonymization": policy.requires_anonymization
            }

        return None

    def find_expired_records(self, data_type: str) -> List[Dict[str, Any]]:
        """
        Find records that have exceeded their retention period.

        In production, this would query the database for records
        older than the retention period.
        """
        policy = self.get_retention_policy(data_type)
        if not policy:
            return []

        retention_years = policy.get("years", 7)
        cutoff_date = datetime.utcnow() - timedelta(days=retention_years * 365)

        # In production, this would be a database query
        # For testing, return cached or empty list
        return self._expired_records_cache.get(data_type, [])

    def cleanup_expired_data(self, data_type: str) -> Dict[str, Any]:
        """
        Clean up expired data according to retention policy.

        For SEC compliance, we typically anonymize rather than delete
        to maintain audit trail integrity.
        """
        expired_records = self.find_expired_records(data_type)

        if not expired_records:
            return {
                "data_type": data_type,
                "records_deleted": 0,
                "status": "no_expired_records"
            }

        policy = self.get_retention_policy(data_type)

        # Determine action based on policy
        if policy and policy.get("requires_anonymization", False):
            # Anonymize records instead of deleting
            deleted_count = self._anonymize_records(data_type, expired_records)
            action = "anonymized"
        else:
            # Delete records
            deleted_count = self._delete_records(data_type, expired_records)
            action = "deleted"

        logger.info(f"Retention cleanup for {data_type}: {deleted_count} records {action}")

        return {
            "data_type": data_type,
            "records_deleted": deleted_count,
            "action": action,
            "cleanup_date": datetime.utcnow().isoformat()
        }

    def _anonymize_records(self, data_type: str, records: List[Dict]) -> int:
        """Anonymize records while preserving audit trail"""
        # In production, this would update database records
        return len(records)

    def _delete_records(self, data_type: str, records: List[Dict]) -> int:
        """Delete expired records"""
        # In production, this would delete from database
        return len(records)


class InvestmentAdviceDocumentation:
    """
    Documents investment advice for SEC compliance.

    SEC Investment Advisers Act Rule 204-2 requires:
    - Documentation of advice given
    - Rationale for recommendations
    - Conflicts of interest disclosure
    - Model/algorithm methodology
    """

    def __init__(self):
        self._documentation: Dict[str, RecommendationDocumentation] = {}

    def document_recommendation(
        self,
        recommendation_id: str,
        rationale: Dict[str, Any],
        analyst_id: str
    ) -> Dict[str, Any]:
        """
        Document a recommendation with full SEC-compliant rationale.

        Args:
            recommendation_id: Unique recommendation identifier
            rationale: Full analysis rationale and factors
            analyst_id: ID of the analyst/system generating recommendation

        Returns:
            Documentation confirmation
        """
        doc_id = str(uuid.uuid4())

        doc = RecommendationDocumentation(
            recommendation_id=recommendation_id,
            stock=rationale.get("stock", "UNKNOWN"),
            recommendation=rationale.get("recommendation", "HOLD"),
            analyst_id=analyst_id,
            timestamp=datetime.utcnow(),
            rationale=rationale,
            model_version=rationale.get("model_version", "1.0.0"),
            data_sources=rationale.get("data_sources", []),
            risk_disclosure=SEC_STANDARD_RISK_WARNING,
            conflicts_disclosed=rationale.get("conflicts", [])
        )

        self._documentation[recommendation_id] = doc

        logger.info(f"Documented recommendation {recommendation_id} by {analyst_id}")

        return {
            "status": "documented",
            "documentation_id": doc_id,
            "recommendation_id": recommendation_id,
            "timestamp": doc.timestamp.isoformat()
        }

    def get_recommendation_documentation(
        self,
        recommendation_id: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve documentation for a recommendation"""
        doc = self._documentation.get(recommendation_id)

        if not doc:
            return None

        return {
            "recommendation_id": doc.recommendation_id,
            "stock": doc.stock,
            "recommendation": doc.recommendation,
            "analyst_id": doc.analyst_id,
            "timestamp": doc.timestamp.isoformat(),
            "rationale": doc.rationale,
            "model_version": doc.model_version,
            "data_sources": doc.data_sources,
            "risk_disclosure": doc.risk_disclosure,
            "conflicts_disclosed": doc.conflicts_disclosed
        }


class FiduciaryDutyChecker:
    """
    Checks recommendations against fiduciary duty requirements.

    Investment advisers have a fiduciary duty to:
    - Act in client's best interest
    - Disclose conflicts of interest
    - Ensure suitability of recommendations
    - Provide fair and balanced information
    """

    # Potential conflict of interest types
    CONFLICT_TYPES = {
        "firm_holdings": "Firm or affiliates hold positions in recommended security",
        "compensation": "Adviser receives compensation from security issuer",
        "relationship": "Material business relationship with recommended company",
        "personal_holdings": "Adviser personally holds position in security"
    }

    def __init__(self):
        self._risk_tolerance_map = {
            "conservative": 0.3,
            "moderate": 0.6,
            "aggressive": 0.9
        }

    def check_conflicts_of_interest(
        self,
        recommendation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check for conflicts of interest in a recommendation.

        Args:
            recommendation: Recommendation details including potential conflicts

        Returns:
            Conflict analysis results
        """
        conflicts_detected = []
        conflict_details = []

        # Check for firm holdings conflict
        if recommendation.get("firm_holdings", False):
            conflicts_detected.append("firm_holdings")
            conflict_details.append({
                "type": "firm_holdings",
                "description": self.CONFLICT_TYPES["firm_holdings"],
                "severity": "high",
                "requires_disclosure": True
            })

        # Check for compensation conflicts
        if recommendation.get("receives_compensation", False):
            conflicts_detected.append("compensation")
            conflict_details.append({
                "type": "compensation",
                "description": self.CONFLICT_TYPES["compensation"],
                "severity": "high",
                "requires_disclosure": True
            })

        return {
            "conflicts_detected": len(conflicts_detected) > 0,
            "conflict_count": len(conflicts_detected),
            "conflict_types": conflicts_detected,
            "conflict_details": conflict_details,
            "disclosure_required": len(conflicts_detected) > 0,
            "checked_at": datetime.utcnow().isoformat()
        }

    def requires_disclosure(self, recommendation: Dict[str, Any]) -> bool:
        """Determine if a recommendation requires conflict disclosure"""
        conflict_check = self.check_conflicts_of_interest(recommendation)
        return conflict_check.get("disclosure_required", False)

    def analyze_suitability(
        self,
        recommendation: Dict[str, Any],
        client_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze if a recommendation is suitable for a client.

        SEC Regulation Best Interest requires advisers to:
        - Understand the client's investment profile
        - Have a reasonable basis for recommendations
        - Consider the client's financial situation and objectives

        Args:
            recommendation: The investment recommendation
            client_profile: Client's investment profile

        Returns:
            Suitability analysis results
        """
        client_risk_tolerance = client_profile.get("risk_tolerance", "moderate")
        recommendation_risk = recommendation.get("risk_level", "moderate")

        # Calculate suitability score
        client_risk_score = self._risk_tolerance_map.get(client_risk_tolerance, 0.5)
        rec_risk_score = self._risk_tolerance_map.get(recommendation_risk, 0.5)

        # Suitability is higher when recommendation risk matches client tolerance
        risk_alignment = 1.0 - abs(client_risk_score - rec_risk_score)

        # Check investment objective alignment
        objective_alignment = self._check_objective_alignment(
            recommendation,
            client_profile.get("investment_objective", "growth")
        )

        # Check time horizon alignment
        horizon_alignment = self._check_horizon_alignment(
            recommendation,
            client_profile.get("time_horizon", "long_term")
        )

        # Combine factors for overall suitability
        suitability_score = (
            risk_alignment * 0.4 +
            objective_alignment * 0.3 +
            horizon_alignment * 0.3
        )

        suitable = suitability_score >= 0.6

        return {
            "suitable": suitable,
            "suitability_score": round(suitability_score, 3),
            "factors": {
                "risk_alignment": round(risk_alignment, 3),
                "objective_alignment": round(objective_alignment, 3),
                "horizon_alignment": round(horizon_alignment, 3)
            },
            "client_profile": {
                "risk_tolerance": client_risk_tolerance,
                "investment_objective": client_profile.get("investment_objective"),
                "time_horizon": client_profile.get("time_horizon")
            },
            "recommendation_characteristics": {
                "action": recommendation.get("action"),
                "risk_level": recommendation_risk,
                "stock": recommendation.get("stock")
            },
            "analyzed_at": datetime.utcnow().isoformat()
        }

    def _check_objective_alignment(
        self,
        recommendation: Dict[str, Any],
        objective: str
    ) -> float:
        """Check alignment between recommendation and investment objective"""
        # Simple alignment check - in production would be more sophisticated
        action = recommendation.get("action", "HOLD").upper()

        if objective == "growth":
            return 0.9 if action in ["BUY", "STRONG_BUY"] else 0.5
        elif objective == "income":
            return 0.8 if action == "HOLD" else 0.6
        elif objective == "preservation":
            return 0.9 if action in ["SELL", "HOLD"] else 0.4

        return 0.5

    def _check_horizon_alignment(
        self,
        recommendation: Dict[str, Any],
        horizon: str
    ) -> float:
        """Check alignment between recommendation and time horizon"""
        # Get recommendation time horizon
        rec_horizon = recommendation.get("time_horizon", "medium")

        horizon_scores = {
            ("short_term", "short"): 0.9,
            ("short_term", "medium"): 0.7,
            ("short_term", "long"): 0.4,
            ("medium_term", "short"): 0.6,
            ("medium_term", "medium"): 0.9,
            ("medium_term", "long"): 0.7,
            ("long_term", "short"): 0.3,
            ("long_term", "medium"): 0.6,
            ("long_term", "long"): 0.9
        }

        # Normalize horizon names
        client_horizon = "long" if "long" in horizon.lower() else (
            "short" if "short" in horizon.lower() else "medium"
        )
        rec_h = "long" if "long" in str(rec_horizon).lower() else (
            "short" if "short" in str(rec_horizon).lower() else "medium"
        )

        return horizon_scores.get((horizon, rec_h), 0.5)


class SECDisclosureGenerator:
    """
    Generates SEC-compliant disclosure statements for recommendations.
    """

    @staticmethod
    def generate_methodology_disclosure(
        algorithm_type: str = "ML-powered quantitative",
        model_version: str = "1.0.0",
        training_date: str = None
    ) -> str:
        """Generate methodology disclosure statement"""
        if training_date is None:
            training_date = datetime.utcnow().strftime("%Y-%m-%d")

        return SEC_METHODOLOGY_DISCLOSURE_TEMPLATE.format(
            algorithm_type=algorithm_type,
            model_version=model_version,
            training_date=training_date
        )

    @staticmethod
    def generate_risk_warning() -> str:
        """Generate standard SEC risk warning"""
        return SEC_STANDARD_RISK_WARNING

    @staticmethod
    def generate_limitations_statement() -> str:
        """Generate limitations disclosure statement"""
        return (
            "This analysis does NOT consider: (1) your individual financial situation, "
            "(2) tax implications specific to your circumstances, (3) real-time market "
            "conditions that may have changed since data collection, (4) non-public "
            "information, (5) geopolitical events occurring after the analysis date."
        )

    @staticmethod
    def generate_conflict_disclosure(has_conflicts: bool = False) -> str:
        """Generate conflict of interest disclosure"""
        if has_conflicts:
            return (
                "CONFLICT DISCLOSURE: This platform or its affiliates may hold positions "
                "in the recommended securities. Please review the specific conflicts "
                "disclosed with each recommendation."
            )
        return (
            "This platform does not hold positions in any recommended securities. "
            "No material relationships exist between this platform and any recommended issuers."
        )
