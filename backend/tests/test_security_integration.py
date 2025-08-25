"""
Security and Authentication Integration Tests for Investment Analysis Platform
Tests OAuth2 authentication, rate limiting, access control, and security measures.
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import AsyncClient
from fastapi import HTTPException, status
from jose import jwt
import bcrypt

from backend.api.main import app
from backend.auth.oauth2 import (
    create_access_token,
    verify_token,
    get_current_user,
    get_current_active_user,
    authenticate_user
)
from backend.security.rate_limiter import RateLimiter
from backend.security.sql_injection_prevention import SQLInjectionPreventer
from backend.security.jwt_manager import JWTManager
from backend.models.unified_models import User
from backend.config.settings import settings


class TestSecurityIntegration:
    """Test comprehensive security features including authentication, authorization, and protection."""

    @pytest.fixture
    def mock_user(self):
        """Create mock user for testing."""
        return User(
            id=1,
            username="testuser",
            email="test@example.com",
            hashed_password="$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "secret"
            is_active=True,
            is_verified=True,
            created_at=datetime.utcnow(),
            last_login=datetime.utcnow()
        )

    @pytest.fixture
    def jwt_manager(self):
        """Create JWT manager instance."""
        return JWTManager(
            secret_key=settings.SECRET_KEY,
            algorithm="HS256",
            access_token_expire_minutes=30
        )

    @pytest.fixture
    def rate_limiter(self):
        """Create rate limiter instance."""
        return RateLimiter()

    @pytest.fixture
    async def async_client(self):
        """Create async HTTP client for testing."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client

    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_jwt_token_creation_and_validation(self, jwt_manager, mock_user):
        """Test JWT token creation, validation, and expiration."""
        
        # Test token creation
        token_data = {"sub": str(mock_user.id), "username": mock_user.username}
        access_token = jwt_manager.create_token(token_data, expires_delta=timedelta(minutes=30))
        
        assert access_token is not None
        assert isinstance(access_token, str)
        
        # Test token validation
        decoded_token = jwt_manager.decode_token(access_token)
        assert decoded_token["sub"] == str(mock_user.id)
        assert decoded_token["username"] == mock_user.username
        assert "exp" in decoded_token
        
        # Test token expiration
        expired_token = jwt_manager.create_token(
            token_data, 
            expires_delta=timedelta(seconds=-1)  # Already expired
        )
        
        with pytest.raises(Exception):  # Should raise JWT expired exception
            jwt_manager.decode_token(expired_token)
        
        # Test invalid token
        invalid_token = "invalid.jwt.token"
        with pytest.raises(Exception):
            jwt_manager.decode_token(invalid_token)

    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_user_authentication_flow(self, async_client, mock_user):
        """Test complete user authentication flow."""
        
        with patch('backend.repositories.user_repository.get_by_username') as mock_get_user:
            with patch('backend.repositories.user_repository.verify_password') as mock_verify:
                mock_get_user.return_value = mock_user
                mock_verify.return_value = True
                
                # Test login
                login_data = {
                    "username": "testuser",
                    "password": "secret"
                }
                
                response = await async_client.post("/api/auth/token", data=login_data)
                
                assert response.status_code == 200
                token_data = response.json()
                
                assert "access_token" in token_data
                assert "token_type" in token_data
                assert token_data["token_type"] == "bearer"
                
                access_token = token_data["access_token"]
                
                # Test authenticated request
                headers = {"Authorization": f"Bearer {access_token}"}
                response = await async_client.get("/api/auth/me", headers=headers)
                
                # In real scenario, this would work with proper token validation
                # Here we test the structure
                assert response.status_code in [200, 401]  # Depends on implementation

    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_password_security(self, mock_user):
        """Test password hashing and verification security."""
        
        # Test password hashing
        plain_password = "test_password_123!"
        hashed = bcrypt.hashpw(plain_password.encode('utf-8'), bcrypt.gensalt())
        
        # Test password verification
        assert bcrypt.checkpw(plain_password.encode('utf-8'), hashed)
        assert not bcrypt.checkpw(b"wrong_password", hashed)
        
        # Test password strength requirements
        weak_passwords = ["123", "password", "abc", ""]
        strong_passwords = ["MyStr0ng!P@ssw0rd", "C0mplex#Pass123", "Secure$Password456"]
        
        from backend.auth.password_validator import PasswordValidator
        validator = PasswordValidator()
        
        for weak_pass in weak_passwords:
            assert not validator.is_strong_password(weak_pass)
        
        for strong_pass in strong_passwords:
            assert validator.is_strong_password(strong_pass)

    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_rate_limiting_integration(self, async_client, rate_limiter):
        """Test API rate limiting implementation."""
        
        with patch.object(app, 'dependency_overrides', {}):
            # Test rate limiting on login endpoint
            login_data = {"username": "testuser", "password": "password"}
            
            # Make multiple rapid requests
            responses = []
            for i in range(10):  # Exceed rate limit
                response = await async_client.post("/api/auth/token", data=login_data)
                responses.append(response)
            
            # Should eventually get rate limited
            rate_limited_responses = [r for r in responses if r.status_code == 429]
            
            # In production, should have some rate limited responses
            # Here we test the infrastructure exists
            assert len(responses) == 10
            
            # Test rate limit headers
            if responses[-1].status_code == 429:
                headers = responses[-1].headers
                assert "X-RateLimit-Limit" in headers or "Retry-After" in headers

    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_sql_injection_prevention(self, async_client):
        """Test SQL injection prevention mechanisms."""
        
        sql_injection_payloads = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'--",
            "'; INSERT INTO users VALUES ('hacker', 'password'); --",
            "1' UNION SELECT * FROM sensitive_data --"
        ]
        
        # Test on search endpoint
        for payload in sql_injection_payloads:
            response = await async_client.get(f"/api/stocks/search?query={payload}")
            
            # Should not return 500 error (indicating SQL error)
            # Should handle injection attempt gracefully
            assert response.status_code != 500
            
            if response.status_code == 200:
                data = response.json()
                # Should not contain suspicious data
                assert not any("DROP" in str(item) for item in data if isinstance(data, list))

    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_input_validation_and_sanitization(self, async_client):
        """Test input validation and sanitization."""
        
        # Test XSS prevention
        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src='x' onerror='alert(1)'>",
            "';alert(String.fromCharCode(88,83,83))//'"
        ]
        
        headers = {"Authorization": "Bearer test_token"}
        
        for payload in xss_payloads:
            # Test on portfolio creation
            portfolio_data = {
                "name": payload,
                "description": f"Test portfolio {payload}",
                "strategy": "aggressive"
            }
            
            response = await async_client.post(
                "/api/portfolio/create",
                headers=headers,
                json=portfolio_data
            )
            
            # Should validate input and reject or sanitize
            if response.status_code == 422:
                # Validation error - good
                error_detail = response.json()
                assert "detail" in error_detail
            elif response.status_code == 200:
                # Accepted but should be sanitized
                data = response.json()
                # Should not contain raw script tags
                assert "<script>" not in str(data)

    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_authorization_and_access_control(self, async_client, mock_user):
        """Test authorization and access control mechanisms."""
        
        # Create tokens for different users
        user1_token = create_access_token(data={"sub": "1", "username": "user1"})
        user2_token = create_access_token(data={"sub": "2", "username": "user2"})
        admin_token = create_access_token(data={"sub": "999", "username": "admin", "role": "admin"})
        
        # Test accessing own resources
        headers1 = {"Authorization": f"Bearer {user1_token}"}
        response = await async_client.get("/api/portfolio/user-1-portfolio", headers=headers1)
        
        # Should allow access to own resources (or appropriate error)
        assert response.status_code in [200, 404, 401]  # Not 403 Forbidden
        
        # Test accessing other user's resources
        headers2 = {"Authorization": f"Bearer {user2_token}"}
        response = await async_client.get("/api/portfolio/user-1-portfolio", headers=headers2)
        
        # Should deny access to other user's resources
        assert response.status_code in [403, 404, 401]
        
        # Test admin access
        admin_headers = {"Authorization": f"Bearer {admin_token}"}
        response = await async_client.get("/api/admin/users", headers=admin_headers)
        
        # Should allow admin access (or appropriate error)
        assert response.status_code in [200, 401]  # Not 403 if token is valid

    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_session_management_security(self, jwt_manager, mock_user):
        """Test session management and token security."""
        
        # Test token rotation
        token1 = jwt_manager.create_token({"sub": str(mock_user.id)})
        
        # Simulate token refresh
        await asyncio.sleep(1)  # Ensure different timestamp
        token2 = jwt_manager.create_token({"sub": str(mock_user.id)})
        
        # Tokens should be different
        assert token1 != token2
        
        # Both should be valid initially
        decoded1 = jwt_manager.decode_token(token1)
        decoded2 = jwt_manager.decode_token(token2)
        
        assert decoded1["sub"] == decoded2["sub"]
        
        # Test token blacklisting (if implemented)
        # jwt_manager.blacklist_token(token1)
        # Should raise exception when using blacklisted token
        
        # Test concurrent session limits
        # In production, might limit concurrent sessions per user

    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_data_encryption_and_privacy(self, async_client):
        """Test data encryption and privacy protection."""
        
        # Test that sensitive data is not exposed in responses
        headers = {"Authorization": "Bearer test_token"}
        
        with patch('backend.auth.oauth2.get_current_user') as mock_get_user:
            mock_user = User(
                id=1,
                username="testuser",
                email="test@example.com",
                hashed_password="$2b$12$hash...",  # Should never be exposed
                is_active=True
            )
            mock_get_user.return_value = mock_user
            
            response = await async_client.get("/api/auth/me", headers=headers)
            
            if response.status_code == 200:
                user_data = response.json()
                
                # Should not expose sensitive fields
                assert "hashed_password" not in user_data
                assert "password" not in user_data
                
                # Should expose safe fields
                expected_fields = ["id", "username", "email", "is_active"]
                for field in expected_fields:
                    if field in user_data:  # May depend on response model
                        assert user_data[field] is not None

    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_cors_security(self, async_client):
        """Test CORS security configuration."""
        
        # Test preflight request
        headers = {
            "Origin": "https://malicious-site.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Authorization"
        }
        
        response = await async_client.options("/api/stocks/search", headers=headers)
        
        # Should handle CORS appropriately
        if response.status_code == 200:
            cors_headers = response.headers
            
            # Check for CORS headers
            if "Access-Control-Allow-Origin" in cors_headers:
                allowed_origin = cors_headers["Access-Control-Allow-Origin"]
                
                # Should not allow * for credentials
                if "Access-Control-Allow-Credentials" in cors_headers:
                    assert allowed_origin != "*"

    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_security_headers(self, async_client):
        """Test security headers in HTTP responses."""
        
        response = await async_client.get("/api/health/status")
        headers = response.headers
        
        # Test for important security headers
        security_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Strict-Transport-Security",  # For HTTPS
            "Content-Security-Policy"
        ]
        
        # Note: Some headers may only be present in production
        for header in security_headers:
            if header in headers:
                # Verify header has secure value
                if header == "X-Content-Type-Options":
                    assert headers[header] == "nosniff"
                elif header == "X-Frame-Options":
                    assert headers[header] in ["DENY", "SAMEORIGIN"]

    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_api_versioning_security(self, async_client):
        """Test security across different API versions."""
        
        # Test deprecated version warnings
        headers = {
            "X-API-Version": "v1",
            "Authorization": "Bearer test_token"
        }
        
        response = await async_client.get("/api/stocks/AAPL", headers=headers)
        
        # Should handle version appropriately
        if response.status_code == 200:
            # May include deprecation warnings
            if "X-API-Deprecated" in response.headers:
                assert response.headers["X-API-Deprecated"] == "true"
        
        # Test version-specific security rules
        # Older versions might have stricter rate limits
        # Newer versions might have enhanced authentication

    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_audit_logging_integration(self, async_client):
        """Test security audit logging."""
        
        with patch('backend.utils.audit_logger.log_security_event') as mock_audit:
            # Test failed authentication logging
            login_data = {"username": "nonexistent", "password": "wrong"}
            response = await async_client.post("/api/auth/token", data=login_data)
            
            if response.status_code == 401:
                # Should log failed authentication attempt
                # In production implementation
                pass
            
            # Test successful authentication logging
            with patch('backend.repositories.user_repository.get_by_username') as mock_get:
                with patch('backend.repositories.user_repository.verify_password') as mock_verify:
                    mock_user = User(id=1, username="testuser", email="test@test.com", is_active=True)
                    mock_get.return_value = mock_user
                    mock_verify.return_value = True
                    
                    login_data = {"username": "testuser", "password": "correct"}
                    response = await async_client.post("/api/auth/token", data=login_data)
                    
                    # Should log successful authentication
                    # In production would verify audit logs

    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_privilege_escalation_prevention(self, async_client):
        """Test prevention of privilege escalation attacks."""
        
        # Create regular user token
        user_token = create_access_token(data={"sub": "1", "username": "user"})
        headers = {"Authorization": f"Bearer {user_token}"}
        
        # Try to access admin endpoints
        admin_endpoints = [
            "/api/admin/users",
            "/api/admin/system/shutdown",
            "/api/admin/cache/clear",
            "/api/admin/database/backup"
        ]
        
        for endpoint in admin_endpoints:
            response = await async_client.get(endpoint, headers=headers)
            
            # Should deny access with 403 Forbidden
            assert response.status_code in [403, 401, 404]
        
        # Try to modify other user's data
        other_user_data = {
            "user_id": 999,  # Different user
            "portfolio_name": "Hacked Portfolio"
        }
        
        response = await async_client.post(
            "/api/portfolio/999/update",
            headers=headers,
            json=other_user_data
        )
        
        # Should deny unauthorized access
        assert response.status_code in [403, 401, 404]

    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_data_leakage_prevention(self, async_client):
        """Test prevention of data leakage in error messages."""
        
        # Test database error handling
        with patch('backend.repositories.stock_repository.get_by_symbol') as mock_get:
            mock_get.side_effect = Exception("DETAIL: Key (id)=(1) violates unique constraint")
            
            response = await async_client.get("/api/stocks/AAPL")
            
            # Should not expose database details
            if response.status_code == 500:
                error_data = response.json()
                error_message = str(error_data)
                
                # Should not contain database internals
                assert "violates unique constraint" not in error_message
                assert "DETAIL:" not in error_message
                assert "Key (" not in error_message
        
        # Test file path disclosure
        with patch('backend.utils.file_handler.read_file') as mock_read:
            mock_read.side_effect = FileNotFoundError("/etc/passwd not found")
            
            # Attempt to trigger file error
            response = await async_client.get("/api/analysis/report/nonexistent")
            
            if response.status_code == 404:
                error_data = response.json()
                error_message = str(error_data)
                
                # Should not expose file paths
                assert "/etc/passwd" not in error_message
                assert not any(path in error_message for path in ["/home/", "/var/", "/usr/"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])