#!/usr/bin/env python3
"""
Security Validation Script

This script validates that all security enhancements are properly configured
and functioning as expected.
"""

import os
import sys
import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def validate_secrets_manager():
    """Validate secrets management system"""
    logger.info("🔐 Validating secrets manager...")
    
    try:
        from backend.security.secrets_manager import get_secrets_manager, SecretType
        
        secrets_manager = get_secrets_manager()
        
        # Test storing and retrieving a secret
        test_secret = "test-secret-12345"
        success = secrets_manager.store_secret(
            "test_validation_secret",
            test_secret,
            SecretType.API_KEY,
            description="Test secret for validation"
        )
        
        if not success:
            logger.error("❌ Failed to store test secret")
            return False
        
        # Retrieve and verify
        retrieved = secrets_manager.get_secret("test_validation_secret")
        if retrieved != test_secret:
            logger.error("❌ Retrieved secret doesn't match stored secret")
            return False
        
        # Clean up test secret
        secrets_manager.delete_secret("test_validation_secret")
        
        # Check for required secrets
        required_secrets = [
            "api_key_alpha_vantage",
            "api_key_finnhub",
            "jwt_private_key",
            "jwt_public_key"
        ]
        
        missing_secrets = []
        for secret_name in required_secrets:
            if not secrets_manager.get_secret(secret_name):
                missing_secrets.append(secret_name)
        
        if missing_secrets:
            logger.warning(f"⚠️  Missing secrets: {', '.join(missing_secrets)}")
        
        logger.info("✅ Secrets manager validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Secrets manager validation failed: {e}")
        return False


async def validate_jwt_system():
    """Validate JWT authentication system"""
    logger.info("🔑 Validating JWT system...")
    
    try:
        from backend.security.jwt_manager import get_jwt_manager, TokenClaims, TokenType
        
        jwt_manager = get_jwt_manager()
        
        # Test token creation and verification
        test_claims = TokenClaims(
            user_id=1,
            username="test_user",
            email="test@example.com",
            roles=["user"],
            scopes=["read", "write"],
            is_admin=False
        )
        
        # Create access token
        access_token = jwt_manager.create_access_token(test_claims)
        if not access_token:
            logger.error("❌ Failed to create access token")
            return False
        
        # Verify token
        payload = jwt_manager.verify_token(access_token, TokenType.ACCESS)
        if not payload:
            logger.error("❌ Failed to verify access token")
            return False
        
        # Test refresh token
        refresh_token = jwt_manager.create_refresh_token(test_claims)
        refresh_payload = jwt_manager.verify_token(refresh_token, TokenType.REFRESH)
        if not refresh_payload:
            logger.error("❌ Failed to verify refresh token")
            return False
        
        # Test token revocation
        revoke_success = jwt_manager.revoke_token(access_token)
        if not revoke_success:
            logger.warning("⚠️  Token revocation may not be working (Redis might be unavailable)")
        
        # Test JWKS endpoint
        jwks = jwt_manager.get_public_key_jwks()
        if not jwks.get("keys"):
            logger.error("❌ JWKS public key not available")
            return False
        
        logger.info("✅ JWT system validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ JWT system validation failed: {e}")
        return False


async def validate_sql_injection_prevention():
    """Validate SQL injection prevention"""
    logger.info("🛡️  Validating SQL injection prevention...")
    
    try:
        from backend.security.sql_injection_prevention import (
            SQLInjectionPrevention, 
            SQLInjectionThreatLevel,
            validate_user_input
        )
        from fastapi import HTTPException
        
        prevention = SQLInjectionPrevention()
        
        # Test basic SQL injection patterns
        test_inputs = [
            ("normal input", False),
            ("'; DROP TABLE users; --", True),
            ("1' OR '1'='1", True),
            ("UNION SELECT * FROM passwords", True),
            ("<script>alert('xss')</script>", False),  # XSS, not SQL injection
            ("admin'--", True),
            ("1; DELETE FROM accounts", True)
        ]
        
        for input_text, should_be_threat in test_inputs:
            detection = prevention.detect_sql_injection(input_text)
            is_high_threat = detection.threat_level in [
                SQLInjectionThreatLevel.HIGH, 
                SQLInjectionThreatLevel.CRITICAL
            ]
            
            if should_be_threat and not is_high_threat:
                logger.error(f"❌ Failed to detect SQL injection in: '{input_text}'")
                return False
            elif not should_be_threat and is_high_threat:
                logger.warning(f"⚠️  False positive for: '{input_text}'")
        
        # Test input validation function
        try:
            validate_user_input("'; DROP TABLE users; --")
            logger.error("❌ Dangerous input was not blocked by validation")
            return False
        except HTTPException:
            # This is expected - dangerous input should be blocked
            pass
        
        # Test safe input passes through
        safe_input = validate_user_input("AAPL stock price")
        if not safe_input:
            logger.error("❌ Safe input was incorrectly blocked")
            return False
        
        logger.info("✅ SQL injection prevention validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ SQL injection prevention validation failed: {e}")
        return False


async def validate_rate_limiting():
    """Validate rate limiting system"""
    logger.info("🚦 Validating rate limiting...")
    
    try:
        from backend.security.rate_limiter import get_rate_limiter, RateLimitCategory
        from unittest.mock import Mock
        
        rate_limiter = get_rate_limiter()
        
        # Mock request object
        mock_request = Mock()
        mock_request.client.host = "127.0.0.1"
        mock_request.headers = {"user-agent": "test-client"}
        
        # Test normal rate limiting
        status = await rate_limiter.check_rate_limit(
            mock_request,
            RateLimitCategory.API_READ
        )
        
        if not status.allowed:
            logger.error("❌ Rate limiting blocked normal request")
            return False
        
        # Test rate limit rules exist
        if not rate_limiter.default_rules:
            logger.error("❌ No rate limit rules configured")
            return False
        
        # Check required categories
        required_categories = [
            RateLimitCategory.AUTHENTICATION,
            RateLimitCategory.API_READ,
            RateLimitCategory.API_WRITE,
            RateLimitCategory.ADMIN
        ]
        
        for category in required_categories:
            if category not in rate_limiter.default_rules:
                logger.error(f"❌ Missing rate limit rule for {category}")
                return False
        
        # Test trusted IP detection
        is_trusted = rate_limiter._is_trusted_ip("127.0.0.1")
        if not is_trusted:
            logger.warning("⚠️  Localhost not detected as trusted IP")
        
        logger.info("✅ Rate limiting validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Rate limiting validation failed: {e}")
        return False


async def validate_database_security():
    """Validate database security features"""
    logger.info("🗄️  Validating database security...")
    
    try:
        from backend.security.database_security import get_database_security_manager
        
        security_manager = get_database_security_manager()
        
        # Test SSL context creation
        try:
            ssl_context = security_manager._create_ssl_context()
            if not ssl_context:
                logger.warning("⚠️  SSL context creation returned None")
        except Exception as e:
            logger.warning(f"⚠️  SSL context creation failed: {e}")
        
        # Test audit log writing
        from backend.security.database_security import AuditLogEntry, AuditEventType
        
        test_entry = AuditLogEntry(
            timestamp=datetime.utcnow(),
            event_type=AuditEventType.QUERY,
            query="SELECT 1",
            success=True,
            risk_score=0
        )
        
        security_manager._write_audit_log(test_entry)
        
        # Verify audit log file exists
        if not security_manager.audit_log_path.exists():
            logger.error("❌ Audit log file not created")
            return False
        
        # Test security report generation
        report = security_manager.generate_security_report()
        if not report:
            logger.warning("⚠️  Security report generation returned empty result")
        
        logger.info("✅ Database security validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database security validation failed: {e}")
        return False


async def validate_configuration():
    """Validate security configuration"""
    logger.info("⚙️  Validating security configuration...")
    
    try:
        # Check required environment variables
        required_env_vars = [
            "MASTER_SECRET_KEY",
        ]
        
        missing_vars = []
        for var in required_env_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            logger.error(f"❌ Missing environment variables: {', '.join(missing_vars)}")
            return False
        
        # Check master key strength
        master_key = os.getenv("MASTER_SECRET_KEY")
        if len(master_key) < 20:
            logger.warning("⚠️  MASTER_SECRET_KEY is shorter than recommended (20+ characters)")
        
        # Check Redis availability
        try:
            import redis
            from backend.config.settings import settings
            redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
            redis_client.ping()
            logger.info("✅ Redis connection successful")
        except Exception as e:
            logger.warning(f"⚠️  Redis not available: {e}")
            logger.warning("Token blacklisting and distributed rate limiting will not work")
        
        # Check secrets directory
        secrets_dir = Path(os.getenv("SECRETS_DIR", "/app/secrets"))
        if not secrets_dir.exists():
            logger.warning(f"⚠️  Secrets directory does not exist: {secrets_dir}")
        elif not secrets_dir.is_dir():
            logger.error(f"❌ Secrets path is not a directory: {secrets_dir}")
            return False
        else:
            # Check permissions
            stat = secrets_dir.stat()
            if stat.st_mode & 0o077:
                logger.warning("⚠️  Secrets directory has overly permissive permissions")
        
        logger.info("✅ Configuration validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration validation failed: {e}")
        return False


async def validate_integration():
    """Validate security system integration"""
    logger.info("🔗 Validating security integration...")
    
    try:
        # Test that all security modules can be imported together
        from backend.security.secrets_manager import get_secrets_manager
        from backend.security.jwt_manager import get_jwt_manager
        from backend.security.rate_limiter import get_rate_limiter
        from backend.security.sql_injection_prevention import sql_injection_prevention
        from backend.security.database_security import get_database_security_manager
        
        # Test that components can interact
        secrets_manager = get_secrets_manager()
        jwt_manager = get_jwt_manager()
        
        # Verify JWT manager can access RSA keys from secrets manager
        jwt_keys_exist = (
            secrets_manager.get_secret("jwt_private_key") and
            secrets_manager.get_secret("jwt_public_key")
        )
        
        if not jwt_keys_exist:
            logger.warning("⚠️  JWT RSA keys not found in secrets manager")
        
        logger.info("✅ Security integration validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Security integration validation failed: {e}")
        return False


async def run_security_tests():
    """Run comprehensive security test suite"""
    logger.info("🧪 Running security test suite...")
    
    test_results = []
    
    # Define test cases
    security_tests = [
        ("Configuration", validate_configuration),
        ("Secrets Manager", validate_secrets_manager),
        ("JWT System", validate_jwt_system),
        ("SQL Injection Prevention", validate_sql_injection_prevention),
        ("Rate Limiting", validate_rate_limiting),
        ("Database Security", validate_database_security),
        ("Integration", validate_integration),
    ]
    
    # Run tests
    for test_name, test_func in security_tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = await test_func()
            test_results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            test_results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("🏁 SECURITY VALIDATION SUMMARY")
    logger.info("="*60)
    
    passed = 0
    failed = 0
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name:.<30} {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    logger.info("-" * 60)
    logger.info(f"Total tests: {len(test_results)}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    
    if failed == 0:
        logger.info("🎉 ALL SECURITY VALIDATIONS PASSED!")
        logger.info("Your security implementation is ready for production.")
        return True
    else:
        logger.error("💥 SOME SECURITY VALIDATIONS FAILED!")
        logger.error("Please fix the issues above before deploying to production.")
        return False


async def main():
    """Main validation function"""
    logger.info("🚀 Starting security validation...")
    
    try:
        success = await run_security_tests()
        
        if success:
            logger.info("\n✅ Security validation completed successfully!")
            sys.exit(0)
        else:
            logger.error("\n❌ Security validation failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Security validation crashed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())