"""
GDPR Compliance Services

Implements GDPR requirements including:
- Right to Data Portability (Article 20)
- Right to Erasure / Right to be Forgotten (Article 17)
- Consent Management (Article 7)
- Data Breach Notification (Articles 33-34)
"""

import logging
import json
import csv
import io
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class ConsentRecord:
    """Record of user consent"""
    user_id: int
    consent_type: str
    consent_given: bool
    consent_date: datetime
    legal_basis: str
    version: str = "1.0"
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


@dataclass
class DeletionRequest:
    """GDPR deletion request tracking"""
    request_id: str
    user_id: int
    status: str  # pending, processing, completed, failed
    request_date: datetime
    completion_date: Optional[datetime] = None
    deleted_records_count: int = 0
    error_message: Optional[str] = None


class GDPRDataPortability:
    """
    Implements GDPR Article 20 - Right to Data Portability

    Users have the right to receive their personal data in a structured,
    commonly used, and machine-readable format.
    """

    def __init__(self):
        self._data_categories = [
            "profile", "portfolio", "transactions",
            "recommendations", "preferences", "audit_logs"
        ]

    def export_user_data(self, user_id: int) -> Dict[str, Any]:
        """
        Export all user data in a structured format.

        Args:
            user_id: The user's ID

        Returns:
            Dictionary containing all user data categories
        """
        logger.info(f"Exporting data for user {user_id}")

        exported_data = {
            "export_metadata": {
                "user_id": user_id,
                "export_date": datetime.utcnow().isoformat(),
                "format_version": "1.0",
                "gdpr_article": "Article 20 - Right to Data Portability"
            }
        }

        # Export each data category
        for category in self._data_categories:
            try:
                exported_data[category] = self._export_category(user_id, category)
            except Exception as e:
                logger.error(f"Error exporting {category} for user {user_id}: {e}")
                exported_data[category] = {"error": str(e), "data": []}

        return exported_data

    def _export_category(self, user_id: int, category: str) -> Dict[str, Any]:
        """Export a specific data category for a user"""
        # In production, this would query the database
        # For now, return placeholder structure
        return {
            "category": category,
            "record_count": 0,
            "data": [],
            "exported_at": datetime.utcnow().isoformat()
        }

    def to_json(self, data: Dict[str, Any]) -> str:
        """Convert exported data to JSON format"""
        return json.dumps(data, indent=2, default=str)

    def to_csv(self, data: Dict[str, Any]) -> str:
        """Convert exported data to CSV format"""
        output = io.StringIO()

        # Flatten the data for CSV export
        for category, category_data in data.items():
            if category == "export_metadata":
                continue

            if isinstance(category_data, dict) and "data" in category_data:
                records = category_data.get("data", [])
                if records and isinstance(records, list):
                    writer = csv.DictWriter(output, fieldnames=records[0].keys())
                    writer.writeheader()
                    writer.writerows(records)
                    output.write("\n")

        return output.getvalue()


class GDPRDataDeletion:
    """
    Implements GDPR Article 17 - Right to Erasure (Right to be Forgotten)

    Users have the right to have their personal data erased when:
    - Data is no longer necessary for original purpose
    - User withdraws consent
    - User objects to processing
    - Data was unlawfully processed
    """

    def __init__(self):
        self._pending_requests: Dict[str, DeletionRequest] = {}
        self._completed_requests: Dict[str, DeletionRequest] = {}

    def request_deletion(self, user_id: int, reason: Optional[str] = None) -> Dict[str, Any]:
        """
        Submit a data deletion request.

        Args:
            user_id: The user's ID
            reason: Optional reason for deletion

        Returns:
            Dictionary with request_id and status
        """
        request_id = str(uuid.uuid4())

        request = DeletionRequest(
            request_id=request_id,
            user_id=user_id,
            status="pending",
            request_date=datetime.utcnow()
        )

        self._pending_requests[request_id] = request

        logger.info(f"Deletion request {request_id} created for user {user_id}")

        return {
            "request_id": request_id,
            "status": "pending",
            "message": "Deletion request received. Processing will begin within 30 days as per GDPR requirements.",
            "estimated_completion": (datetime.utcnow() + timedelta(days=30)).isoformat()
        }

    def process_deletion(self, request_id: str) -> Dict[str, Any]:
        """
        Process a pending deletion request.

        Args:
            request_id: The deletion request ID

        Returns:
            Dictionary with completion status
        """
        if request_id not in self._pending_requests:
            raise ValueError(f"Deletion request {request_id} not found")

        request = self._pending_requests[request_id]
        request.status = "processing"

        try:
            # In production, this would:
            # 1. Anonymize or delete user PII from all tables
            # 2. Remove from backups (where practical)
            # 3. Notify third parties if data was shared
            # 4. Create anonymized audit record

            deleted_count = self._execute_deletion(request.user_id)

            request.status = "completed"
            request.completion_date = datetime.utcnow()
            request.deleted_records_count = deleted_count

            # Move to completed
            self._completed_requests[request_id] = request
            del self._pending_requests[request_id]

            logger.info(f"Deletion request {request_id} completed. {deleted_count} records affected.")

            return {
                "status": "completed",
                "request_id": request_id,
                "deleted_records_count": deleted_count,
                "completion_date": request.completion_date.isoformat()
            }

        except Exception as e:
            request.status = "failed"
            request.error_message = str(e)
            logger.error(f"Deletion request {request_id} failed: {e}")
            raise

    def _execute_deletion(self, user_id: int) -> int:
        """Execute the actual data deletion/anonymization"""
        # In production, this would interact with the database
        # Placeholder returns a count
        return 42  # Number of affected records

    def get_deletion_audit(self, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the audit record for a deletion request.
        Maintains accountability without storing deleted PII.
        """
        request = self._completed_requests.get(request_id) or self._pending_requests.get(request_id)

        if not request:
            return None

        return {
            "request_id": request.request_id,
            "status": request.status,
            "request_date": request.request_date.isoformat(),
            "deletion_date": request.completion_date.isoformat() if request.completion_date else None,
            "records_deleted": request.deleted_records_count,
            # User ID is hashed for anonymized audit trail
            "anonymized_user_reference": hashlib.sha256(str(request.user_id).encode()).hexdigest()[:16]
        }


class ConsentManager:
    """
    Implements GDPR Article 7 - Conditions for Consent

    Manages user consent for data processing activities:
    - Recording consent given
    - Tracking consent withdrawal
    - Maintaining consent history
    - Verifying consent status
    """

    def __init__(self):
        self._consents: Dict[int, Dict[str, ConsentRecord]] = {}
        self._consent_history: Dict[int, List[ConsentRecord]] = {}

    def record_consent(
        self,
        user_id: int,
        consent_type: str,
        consent_given: bool,
        consent_date: datetime,
        legal_basis: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> None:
        """
        Record a user's consent decision.

        Args:
            user_id: The user's ID
            consent_type: Type of consent (e.g., 'data_processing', 'marketing')
            consent_given: Whether consent was given
            consent_date: When consent was given/withdrawn
            legal_basis: Legal basis for processing
            ip_address: Optional IP address for audit
            user_agent: Optional user agent for audit
        """
        record = ConsentRecord(
            user_id=user_id,
            consent_type=consent_type,
            consent_given=consent_given,
            consent_date=consent_date,
            legal_basis=legal_basis,
            ip_address=ip_address,
            user_agent=user_agent
        )

        # Initialize user's consent records if needed
        if user_id not in self._consents:
            self._consents[user_id] = {}
            self._consent_history[user_id] = []

        # Store current consent
        self._consents[user_id][consent_type] = record

        # Add to history
        self._consent_history[user_id].append(record)

        logger.info(f"Consent recorded for user {user_id}: {consent_type}={consent_given}")

    def get_consent_status(self, user_id: int) -> Dict[str, Any]:
        """
        Get current consent status for a user.

        Returns:
            Dictionary of consent types and their status
        """
        if user_id not in self._consents:
            return {}

        status = {}
        for consent_type, record in self._consents[user_id].items():
            status[consent_type] = record.consent_given
            status[f"{consent_type}_consent_date"] = record.consent_date.isoformat()

        return status

    def update_consent(
        self,
        user_id: int,
        consent_type: str,
        consent_given: bool
    ) -> None:
        """Update an existing consent record"""
        self.record_consent(
            user_id=user_id,
            consent_type=consent_type,
            consent_given=consent_given,
            consent_date=datetime.utcnow(),
            legal_basis="consent_update"
        )

    def get_consent_history(self, user_id: int) -> List[Dict[str, Any]]:
        """Get complete consent history for a user"""
        if user_id not in self._consent_history:
            return []

        return [asdict(record) for record in self._consent_history[user_id]]


class DataBreachNotification:
    """
    Implements GDPR Articles 33-34 - Data Breach Notification

    Article 33: Notification to supervisory authority within 72 hours
    Article 34: Communication to data subjects when high risk
    """

    def __init__(self):
        self._breaches: Dict[str, Dict[str, Any]] = {}

    def report_breach(self, breach_details: Dict[str, Any]) -> str:
        """
        Report a data breach incident.

        Args:
            breach_details: Dictionary containing breach information

        Returns:
            Unique breach ID
        """
        breach_id = str(uuid.uuid4())

        breach_record = {
            "breach_id": breach_id,
            "reported_at": datetime.utcnow().isoformat(),
            "breach_type": breach_details.get("breach_type", "unknown"),
            "affected_records": breach_details.get("affected_records", 0),
            "data_categories": breach_details.get("data_categories", []),
            "discovery_date": breach_details.get("discovery_date", datetime.utcnow()).isoformat()
                if isinstance(breach_details.get("discovery_date"), datetime)
                else breach_details.get("discovery_date"),
            "containment_measures": breach_details.get("containment_measures", ""),
            "notification_deadline": (datetime.utcnow() + timedelta(hours=72)).isoformat(),
            "status": "reported"
        }

        self._breaches[breach_id] = breach_record

        logger.critical(f"DATA BREACH REPORTED: {breach_id} - {breach_record['breach_type']}")

        return breach_id

    def is_notification_required(self, breach_id: str) -> bool:
        """
        Determine if regulatory notification is required.

        Per GDPR Article 33, notification is required unless the breach
        is unlikely to result in a risk to the rights and freedoms of
        natural persons.
        """
        if breach_id not in self._breaches:
            raise ValueError(f"Breach {breach_id} not found")

        breach = self._breaches[breach_id]

        # Notification required if:
        # - More than 500 records affected
        # - Sensitive data categories involved
        # - Financial data exposed

        high_risk_categories = {"financial", "portfolio_data", "personal_id", "authentication"}

        affected_records = breach.get("affected_records", 0)
        data_categories = set(breach.get("data_categories", []))

        if affected_records >= 500:
            return True

        if data_categories & high_risk_categories:
            return True

        return False

    def generate_regulatory_notification(self, breach_id: str) -> Dict[str, Any]:
        """
        Generate the regulatory notification document.

        Per GDPR Article 33(3), the notification must describe:
        - Nature of the breach
        - Categories and approximate number of data subjects
        - Likely consequences
        - Measures taken or proposed
        """
        if breach_id not in self._breaches:
            raise ValueError(f"Breach {breach_id} not found")

        breach = self._breaches[breach_id]

        return {
            "notification_type": "GDPR Article 33 - Supervisory Authority Notification",
            "breach_reference": breach_id,
            "generated_at": datetime.utcnow().isoformat(),

            "breach_description": f"Security incident of type '{breach['breach_type']}' "
                                 f"discovered on {breach['discovery_date']}",

            "affected_data_subjects": f"Approximately {breach['affected_records']} data subjects",

            "data_categories_affected": breach["data_categories"],

            "likely_consequences": self._assess_consequences(breach),

            "measures_taken": breach["containment_measures"],

            "dpo_contact": {
                "name": "Data Protection Officer",
                "email": "dpo@investmentplatform.com",
                "phone": "+1-XXX-XXX-XXXX"
            },

            "notification_deadline": breach["notification_deadline"]
        }

    def _assess_consequences(self, breach: Dict[str, Any]) -> List[str]:
        """Assess likely consequences of the breach"""
        consequences = []

        data_categories = breach.get("data_categories", [])

        if "email" in data_categories:
            consequences.append("Potential for phishing attacks targeting affected users")

        if "portfolio_data" in data_categories or "financial" in data_categories:
            consequences.append("Exposure of sensitive financial information")
            consequences.append("Potential for financial fraud or identity theft")

        if "authentication" in data_categories:
            consequences.append("Risk of unauthorized account access")
            consequences.append("Users should reset passwords immediately")

        if not consequences:
            consequences.append("Low risk - No sensitive personal data exposed")

        return consequences
