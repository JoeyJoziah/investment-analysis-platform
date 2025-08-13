"""Security configuration and hardening"""

import os
import secrets
from datetime import timedelta
from typing import Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.sessions import SessionMiddleware

from backend.config.settings import settings


class SecurityConfig:
    """Security configuration and hardening settings"""
    
    # HTTPS Settings
    FORCE_HTTPS = os.getenv("FORCE_HTTPS", "false").lower() == "true"
    
    # CORS Settings
    ALLOWED_ORIGINS = [
        "http://localhost:3000",
        "https://investment-analysis.com",
        "https://api.investment-analysis.com"
    ]
    
    if settings.ENVIRONMENT == "development":
        ALLOWED_ORIGINS.extend([
            "http://localhost:8000",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8000"
        ])
    
    ALLOWED_METHODS = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    ALLOWED_HEADERS = [
        "Authorization",
        "Content-Type",
        "X-Requested-With",
        "X-API-Key"
    ]
    
    # Session Settings
    SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY", secrets.token_urlsafe(32))
    SESSION_MAX_AGE = 3600  # 1 hour
    
    # Rate Limiting
    RATE_LIMIT_ENABLED = True
    DEFAULT_RATE_LIMIT = "100/hour"
    STRICT_RATE_LIMIT = "10/minute"
    
    # Security Headers
    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' data:; "
            "connect-src 'self' ws: wss:"
        ),
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": (
            "geolocation=(), "
            "microphone=(), "
            "camera=(), "
            "payment=(), "
            "usb=(), "
            "magnetometer=(), "
            "gyroscope=(), "
            "accelerometer=()"
        )
    }
    
    # Password Policy
    PASSWORD_MIN_LENGTH = 12
    PASSWORD_REQUIRE_UPPERCASE = True
    PASSWORD_REQUIRE_LOWERCASE = True
    PASSWORD_REQUIRE_DIGITS = True
    PASSWORD_REQUIRE_SPECIAL = True
    PASSWORD_MAX_AGE_DAYS = 90
    
    # JWT Settings
    JWT_ALGORITHM = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS = 7
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
    
    # API Key Settings
    API_KEY_LENGTH = 32
    API_KEY_PREFIX = "sk_"
    
    # File Upload Security
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_FILE_TYPES = [".csv", ".json", ".pdf"]
    UPLOAD_SCAN_ENABLED = True
    
    # Database Security
    DB_CONNECTION_TIMEOUT = 30
    DB_MAX_CONNECTIONS = 20
    DB_SSL_REQUIRE = os.getenv("DB_SSL_REQUIRE", "false").lower() == "true"
    
    # Audit Settings
    AUDIT_LOG_RETENTION_DAYS = 2555  # 7 years for SEC compliance
    AUDIT_LOG_ENCRYPTION = True
    
    # IP Filtering
    BLOCKED_IPS: List[str] = []
    ALLOWED_IPS: Optional[List[str]] = None  # None = allow all
    
    # Trusted Hosts
    TRUSTED_HOSTS = [
        "localhost",
        "127.0.0.1",
        "investment-analysis.com",
        "api.investment-analysis.com"
    ]


def add_security_middleware(app: FastAPI) -> None:
    """Add security middleware to FastAPI app"""
    
    # Add security headers middleware
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        
        # Add security headers
        for header, value in SecurityConfig.SECURITY_HEADERS.items():
            response.headers[header] = value
            
        # Add custom headers
        response.headers["X-API-Version"] = "1.0"
        response.headers["X-RateLimit-Policy"] = SecurityConfig.DEFAULT_RATE_LIMIT
        
        return response
    
    # HTTPS redirect (production only)
    if SecurityConfig.FORCE_HTTPS and settings.ENVIRONMENT == "production":
        app.add_middleware(HTTPSRedirectMiddleware)
    
    # Trusted hosts
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=SecurityConfig.TRUSTED_HOSTS
    )
    
    # GZIP compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=SecurityConfig.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=SecurityConfig.ALLOWED_METHODS,
        allow_headers=SecurityConfig.ALLOWED_HEADERS,
        expose_headers=["X-RateLimit-Remaining", "X-RateLimit-Reset"]
    )
    
    # Session middleware
    app.add_middleware(
        SessionMiddleware,
        secret_key=SecurityConfig.SESSION_SECRET_KEY,
        max_age=SecurityConfig.SESSION_MAX_AGE,
        same_site="strict",
        https_only=SecurityConfig.FORCE_HTTPS
    )
    
    # IP filtering middleware
    @app.middleware("http")
    async def ip_filter_middleware(request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        
        # Check blocked IPs
        if client_ip in SecurityConfig.BLOCKED_IPS:
            from fastapi import HTTPException, status
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="IP address blocked"
            )
        
        # Check allowed IPs (if configured)
        if (SecurityConfig.ALLOWED_IPS is not None and 
            client_ip not in SecurityConfig.ALLOWED_IPS):
            from fastapi import HTTPException, status
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="IP address not allowed"
            )
        
        return await call_next(request)


class PasswordValidator:
    """Password validation and strength checking"""
    
    @staticmethod
    def validate_password(password: str) -> Dict[str, bool]:
        """Validate password against security policy"""
        results = {
            "length": len(password) >= SecurityConfig.PASSWORD_MIN_LENGTH,
            "uppercase": any(c.isupper() for c in password) if SecurityConfig.PASSWORD_REQUIRE_UPPERCASE else True,
            "lowercase": any(c.islower() for c in password) if SecurityConfig.PASSWORD_REQUIRE_LOWERCASE else True,
            "digits": any(c.isdigit() for c in password) if SecurityConfig.PASSWORD_REQUIRE_DIGITS else True,
            "special": any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password) if SecurityConfig.PASSWORD_REQUIRE_SPECIAL else True
        }
        
        results["valid"] = all(results.values())
        return results
    
    @staticmethod
    def calculate_strength(password: str) -> int:
        """Calculate password strength score (0-100)"""
        score = 0
        
        # Length scoring
        if len(password) >= 12:
            score += 25
        elif len(password) >= 8:
            score += 15
        
        # Character variety
        if any(c.isupper() for c in password):
            score += 15
        if any(c.islower() for c in password):
            score += 15
        if any(c.isdigit() for c in password):
            score += 15
        if any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            score += 20
        
        # Bonus for length
        score += min(10, (len(password) - 12) * 2)
        
        return min(100, score)
    
    @staticmethod
    def generate_secure_password(length: int = 16) -> str:
        """Generate a secure password"""
        import string
        import random
        
        # Ensure we have all required character types
        chars = ""
        password = ""
        
        if SecurityConfig.PASSWORD_REQUIRE_UPPERCASE:
            chars += string.ascii_uppercase
            password += random.choice(string.ascii_uppercase)
            
        if SecurityConfig.PASSWORD_REQUIRE_LOWERCASE:
            chars += string.ascii_lowercase
            password += random.choice(string.ascii_lowercase)
            
        if SecurityConfig.PASSWORD_REQUIRE_DIGITS:
            chars += string.digits
            password += random.choice(string.digits)
            
        if SecurityConfig.PASSWORD_REQUIRE_SPECIAL:
            special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
            chars += special_chars
            password += random.choice(special_chars)
        
        # Fill remaining length
        for _ in range(length - len(password)):
            password += random.choice(chars)
        
        # Shuffle the password
        password_list = list(password)
        random.shuffle(password_list)
        
        return ''.join(password_list)


class APIKeyManager:
    """API key management and validation"""
    
    @staticmethod
    def generate_api_key() -> str:
        """Generate a new API key"""
        key = secrets.token_urlsafe(SecurityConfig.API_KEY_LENGTH)
        return f"{SecurityConfig.API_KEY_PREFIX}{key}"
    
    @staticmethod
    def validate_api_key_format(api_key: str) -> bool:
        """Validate API key format"""
        if not api_key.startswith(SecurityConfig.API_KEY_PREFIX):
            return False
        
        # Remove prefix and check length
        key_part = api_key[len(SecurityConfig.API_KEY_PREFIX):]
        return len(key_part) == SecurityConfig.API_KEY_LENGTH
    
    @staticmethod
    def hash_api_key(api_key: str) -> str:
        """Hash API key for storage"""
        import hashlib
        return hashlib.sha256(api_key.encode()).hexdigest()


class SecurityScanner:
    """Security scanning and vulnerability detection"""
    
    @staticmethod
    def scan_file_upload(file_content: bytes, filename: str) -> Dict[str, Any]:
        """Scan uploaded file for security issues"""
        results = {
            "safe": True,
            "issues": [],
            "file_type": "unknown",
            "size": len(file_content)
        }
        
        # Check file size
        if len(file_content) > SecurityConfig.MAX_FILE_SIZE:
            results["safe"] = False
            results["issues"].append("File size exceeds limit")
        
        # Check file extension
        import os
        _, ext = os.path.splitext(filename.lower())
        if ext not in SecurityConfig.ALLOWED_FILE_TYPES:
            results["safe"] = False
            results["issues"].append(f"File type {ext} not allowed")
        
        # Simple malware detection (basic patterns)
        suspicious_patterns = [
            b"<script",
            b"javascript:",
            b"vbscript:",
            b"onload=",
            b"onerror=",
            b"eval(",
            b"exec("
        ]
        
        for pattern in suspicious_patterns:
            if pattern in file_content.lower():
                results["safe"] = False
                results["issues"].append(f"Suspicious pattern detected: {pattern.decode()}")
        
        return results
    
    @staticmethod
    def check_sql_injection(query: str) -> bool:
        """Check for potential SQL injection patterns"""
        suspicious_patterns = [
            "union select",
            "drop table",
            "delete from",
            "insert into",
            "update set",
            "exec(",
            "execute(",
            "--",
            ";--",
            "/*",
            "*/"
        ]
        
        query_lower = query.lower()
        return any(pattern in query_lower for pattern in suspicious_patterns)
    
    @staticmethod
    def validate_input(data: str, max_length: int = 1000) -> Dict[str, Any]:
        """Validate user input for security issues"""
        results = {
            "safe": True,
            "issues": []
        }
        
        # Check length
        if len(data) > max_length:
            results["safe"] = False
            results["issues"].append("Input too long")
        
        # Check for XSS patterns
        xss_patterns = [
            "<script",
            "javascript:",
            "vbscript:",
            "onload=",
            "onerror=",
            "onclick=",
            "onmouseover="
        ]
        
        data_lower = data.lower()
        for pattern in xss_patterns:
            if pattern in data_lower:
                results["safe"] = False
                results["issues"].append(f"XSS pattern detected: {pattern}")
        
        # Check for SQL injection
        if SecurityScanner.check_sql_injection(data):
            results["safe"] = False
            results["issues"].append("Potential SQL injection detected")
        
        return results


# Global security instances
password_validator = PasswordValidator()
api_key_manager = APIKeyManager()
security_scanner = SecurityScanner()