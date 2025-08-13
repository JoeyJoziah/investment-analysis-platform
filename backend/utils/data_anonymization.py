"""Data anonymization for GDPR compliance"""

import hashlib
import random
import string
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from cryptography.fernet import Fernet
from faker import Faker

from backend.config import settings
from backend.utils.monitoring import metrics

# Initialize Faker for generating fake data
fake = Faker()

# Encryption key for reversible anonymization
ENCRYPTION_KEY = settings.GDPR_ENCRYPTION_KEY.encode() if hasattr(settings, 'GDPR_ENCRYPTION_KEY') else Fernet.generate_key()
fernet = Fernet(ENCRYPTION_KEY)


class DataAnonymizer:
    """Handles data anonymization for GDPR compliance"""
    
    def __init__(self):
        self.anonymization_map: Dict[str, str] = {}
        self.encryption_key = ENCRYPTION_KEY
        
    def anonymize_email(self, email: str) -> str:
        """
        Anonymize email address
        
        Args:
            email: Original email address
            
        Returns:
            Anonymized email address
        """
        if not email:
            return email
            
        # Check if already anonymized
        if email in self.anonymization_map:
            return self.anonymization_map[email]
            
        # Split email
        parts = email.split('@')
        if len(parts) != 2:
            return self._hash_value(email)
            
        local, domain = parts
        
        # Anonymize local part
        if len(local) <= 3:
            anon_local = '*' * len(local)
        else:
            anon_local = local[0] + '*' * (len(local) - 2) + local[-1]
            
        # Keep domain for analytics
        anonymized = f"{anon_local}@{domain}"
        
        # Store mapping
        self.anonymization_map[email] = anonymized
        
        # Track operation
        metrics.data_anonymization_operations.labels(
            operation_type="email"
        ).inc()
        
        return anonymized
        
    def anonymize_name(self, name: str) -> str:
        """
        Anonymize person's name
        
        Args:
            name: Original name
            
        Returns:
            Anonymized name
        """
        if not name:
            return name
            
        # Generate consistent fake name based on hash
        hash_value = hashlib.md5(name.encode()).hexdigest()
        random.seed(hash_value)
        
        anonymized = fake.name()
        
        # Track operation
        metrics.data_anonymization_operations.labels(
            operation_type="name"
        ).inc()
        
        return anonymized
        
    def anonymize_ip(self, ip_address: str) -> str:
        """
        Anonymize IP address
        
        Args:
            ip_address: Original IP address
            
        Returns:
            Anonymized IP address
        """
        if not ip_address:
            return ip_address
            
        parts = ip_address.split('.')
        
        if len(parts) == 4:  # IPv4
            # Keep first two octets for geographic analysis
            anonymized = f"{parts[0]}.{parts[1]}.XXX.XXX"
        else:  # IPv6 or invalid
            # Hash the entire address
            anonymized = self._hash_value(ip_address)[:16]
            
        # Track operation
        metrics.data_anonymization_operations.labels(
            operation_type="ip_address"
        ).inc()
        
        return anonymized
        
    def anonymize_phone(self, phone: str) -> str:
        """
        Anonymize phone number
        
        Args:
            phone: Original phone number
            
        Returns:
            Anonymized phone number
        """
        if not phone:
            return phone
            
        # Remove non-digits
        digits = ''.join(filter(str.isdigit, phone))
        
        if len(digits) >= 10:
            # Keep country and area code
            anonymized = digits[:6] + 'X' * (len(digits) - 6)
        else:
            anonymized = 'X' * len(digits)
            
        # Track operation
        metrics.data_anonymization_operations.labels(
            operation_type="phone"
        ).inc()
        
        return anonymized
        
    def anonymize_financial_data(
        self,
        amount: float,
        precision: int = 2
    ) -> float:
        """
        Anonymize financial data by adding noise
        
        Args:
            amount: Original amount
            precision: Decimal precision
            
        Returns:
            Anonymized amount
        """
        if amount == 0:
            return amount
            
        # Add random noise (±5%)
        noise_factor = random.uniform(0.95, 1.05)
        anonymized = round(amount * noise_factor, precision)
        
        # Track operation
        metrics.data_anonymization_operations.labels(
            operation_type="financial"
        ).inc()
        
        return anonymized
        
    def anonymize_date(
        self,
        date: datetime,
        precision: str = "day"
    ) -> datetime:
        """
        Anonymize date by reducing precision
        
        Args:
            date: Original date
            precision: Level of precision ("year", "month", "day")
            
        Returns:
            Anonymized date
        """
        if precision == "year":
            return date.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        elif precision == "month":
            return date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:  # day
            return date.replace(hour=0, minute=0, second=0, microsecond=0)
            
    def anonymize_location(
        self,
        latitude: float,
        longitude: float,
        precision: int = 2
    ) -> tuple[float, float]:
        """
        Anonymize GPS coordinates
        
        Args:
            latitude: Original latitude
            longitude: Original longitude
            precision: Decimal places to keep
            
        Returns:
            Tuple of (anonymized_lat, anonymized_lon)
        """
        # Reduce precision
        anon_lat = round(latitude, precision)
        anon_lon = round(longitude, precision)
        
        # Track operation
        metrics.data_anonymization_operations.labels(
            operation_type="location"
        ).inc()
        
        return anon_lat, anon_lon
        
    def encrypt_sensitive_data(self, data: str) -> str:
        """
        Encrypt sensitive data (reversible)
        
        Args:
            data: Sensitive data to encrypt
            
        Returns:
            Encrypted data
        """
        if not data:
            return data
            
        encrypted = fernet.encrypt(data.encode()).decode()
        
        # Track operation
        metrics.data_anonymization_operations.labels(
            operation_type="encryption"
        ).inc()
        
        return encrypted
        
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """
        Decrypt sensitive data
        
        Args:
            encrypted_data: Encrypted data
            
        Returns:
            Decrypted data
        """
        if not encrypted_data:
            return encrypted_data
            
        try:
            decrypted = fernet.decrypt(encrypted_data.encode()).decode()
            return decrypted
        except Exception:
            # If decryption fails, return empty string
            return ""
            
    def pseudonymize(self, identifier: str, category: str = "user") -> str:
        """
        Create consistent pseudonym for identifier
        
        Args:
            identifier: Original identifier
            category: Category of identifier
            
        Returns:
            Pseudonymized identifier
        """
        # Create consistent hash
        hash_input = f"{category}:{identifier}"
        hash_value = hashlib.sha256(hash_input.encode()).hexdigest()
        
        # Create readable pseudonym
        pseudonym = f"{category}_{hash_value[:8]}"
        
        # Store mapping
        self.anonymization_map[identifier] = pseudonym
        
        # Track operation
        metrics.data_anonymization_operations.labels(
            operation_type="pseudonymization"
        ).inc()
        
        return pseudonym
        
    def _hash_value(self, value: str) -> str:
        """
        Create one-way hash of value
        
        Args:
            value: Value to hash
            
        Returns:
            Hashed value
        """
        return hashlib.sha256(value.encode()).hexdigest()
        
    def anonymize_user_data(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Anonymize complete user data object
        
        Args:
            user_data: User data dictionary
            
        Returns:
            Anonymized user data
        """
        anonymized = user_data.copy()
        
        # Anonymize PII fields
        if "email" in anonymized:
            anonymized["email"] = self.anonymize_email(anonymized["email"])
            
        if "name" in anonymized:
            anonymized["name"] = self.anonymize_name(anonymized["name"])
            
        if "phone" in anonymized:
            anonymized["phone"] = self.anonymize_phone(anonymized["phone"])
            
        if "ip_address" in anonymized:
            anonymized["ip_address"] = self.anonymize_ip(anonymized["ip_address"])
            
        if "date_of_birth" in anonymized:
            anonymized["date_of_birth"] = self.anonymize_date(
                anonymized["date_of_birth"],
                precision="year"
            )
            
        # Pseudonymize user ID
        if "user_id" in anonymized:
            anonymized["user_id"] = self.pseudonymize(
                str(anonymized["user_id"]),
                "user"
            )
            
        return anonymized
        
    def export_anonymization_map(self) -> Dict[str, str]:
        """
        Export anonymization mappings (for data subject requests)
        
        Returns:
            Dictionary of original -> anonymized mappings
        """
        return self.anonymization_map.copy()
        
    def forget_user(self, user_id: str) -> bool:
        """
        Implement right to be forgotten
        
        Args:
            user_id: User ID to forget
            
        Returns:
            Success status
        """
        # Remove from anonymization map
        keys_to_remove = [
            k for k, v in self.anonymization_map.items()
            if v.startswith(f"user_{user_id}")
        ]
        
        for key in keys_to_remove:
            del self.anonymization_map[key]
            
        # Track GDPR request
        metrics.gdpr_requests.labels(request_type="forget").inc()
        
        return True


class GDPRCompliance:
    """GDPR compliance utilities"""
    
    def __init__(self, anonymizer: DataAnonymizer):
        self.anonymizer = anonymizer
        self.consent_records: Dict[str, Dict[str, Any]] = {}
        
    def record_consent(
        self,
        user_id: str,
        purpose: str,
        granted: bool,
        ip_address: Optional[str] = None
    ) -> str:
        """
        Record user consent
        
        Args:
            user_id: User identifier
            purpose: Purpose of data processing
            granted: Whether consent was granted
            ip_address: User's IP address
            
        Returns:
            Consent record ID
        """
        consent_id = hashlib.sha256(
            f"{user_id}:{purpose}:{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]
        
        self.consent_records[consent_id] = {
            "user_id": user_id,
            "purpose": purpose,
            "granted": granted,
            "timestamp": datetime.utcnow(),
            "ip_address": self.anonymizer.anonymize_ip(ip_address) if ip_address else None,
            "consent_id": consent_id
        }
        
        # Track GDPR request
        metrics.gdpr_requests.labels(request_type="consent").inc()
        
        return consent_id
        
    def check_consent(self, user_id: str, purpose: str) -> bool:
        """
        Check if user has given consent for purpose
        
        Args:
            user_id: User identifier
            purpose: Purpose to check
            
        Returns:
            Whether consent is granted
        """
        for record in self.consent_records.values():
            if (record["user_id"] == user_id and
                record["purpose"] == purpose and
                record["granted"]):
                return True
        return False
        
    def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """
        Export all user data (GDPR right to access)
        
        Args:
            user_id: User identifier
            
        Returns:
            All user data
        """
        # Track GDPR request
        metrics.gdpr_requests.labels(request_type="export").inc()
        
        # In a real implementation, this would gather data from all systems
        return {
            "user_id": user_id,
            "export_date": datetime.utcnow().isoformat(),
            "consent_records": [
                record for record in self.consent_records.values()
                if record["user_id"] == user_id
            ],
            "data_categories": [
                "profile_data",
                "trading_history",
                "preferences",
                "analytics_data"
            ]
        }
        
    def data_retention_check(
        self,
        data_date: datetime,
        retention_days: int = 730  # 2 years default
    ) -> bool:
        """
        Check if data should be retained
        
        Args:
            data_date: Date of data
            retention_days: Retention period in days
            
        Returns:
            Whether data should be retained
        """
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        return data_date > cutoff_date


# Global instances
data_anonymizer = DataAnonymizer()
gdpr_compliance = GDPRCompliance(data_anonymizer)