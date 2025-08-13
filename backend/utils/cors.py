"""
CORS Configuration for Production
Secure and flexible CORS handling with environment-based configuration
"""

import logging
from typing import List, Union
from urllib.parse import urlparse
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
import re
import os

logger = logging.getLogger(__name__)

class SecureCORSMiddleware:
    """
    Enhanced CORS middleware with security features for production
    """
    
    def __init__(self):
        self.allowed_origins = self._parse_allowed_origins()
        self.allowed_origin_regex = self._parse_origin_regex()
        self.allow_credentials = self._should_allow_credentials()
        self.allowed_methods = self._get_allowed_methods()
        self.allowed_headers = self._get_allowed_headers()
        self.exposed_headers = self._get_exposed_headers()
        self.max_age = self._get_max_age()
        
    def _parse_allowed_origins(self) -> List[str]:
        """Parse allowed origins from environment"""
        origins = []
        
        # Get from environment
        env_origins = os.getenv("CORS_ORIGINS", "").strip()
        if env_origins:
            # Handle JSON array format
            if env_origins.startswith("["):
                import json
                try:
                    origins = json.loads(env_origins)
                except json.JSONDecodeError:
                    logger.error(f"Invalid CORS_ORIGINS JSON: {env_origins}")
            else:
                # Handle comma-separated format
                origins = [o.strip() for o in env_origins.split(",") if o.strip()]
        
        # Add default origins based on environment
        app_env = os.getenv("APP_ENV", "development")
        
        if app_env == "development":
            # Development defaults
            dev_origins = [
                "http://localhost:3000",
                "http://localhost:3001",
                "http://127.0.0.1:3000",
                "http://127.0.0.1:3001",
                "http://localhost:8000",
                "http://127.0.0.1:8000"
            ]
            origins.extend(dev_origins)
        elif app_env == "production":
            # Production origins (should be configured via env)
            prod_domains = os.getenv("PRODUCTION_DOMAINS", "").strip()
            if prod_domains:
                for domain in prod_domains.split(","):
                    domain = domain.strip()
                    origins.extend([
                        f"https://{domain}",
                        f"https://www.{domain}"
                    ])
        
        # Remove duplicates and empty strings
        origins = list(set(filter(None, origins)))
        
        logger.info(f"Configured CORS origins: {origins}")
        return origins
    
    def _parse_origin_regex(self) -> Union[str, None]:
        """Parse regex pattern for dynamic origin matching"""
        # Allow regex pattern for dynamic subdomains
        regex_pattern = os.getenv("CORS_ORIGIN_REGEX", "")
        
        if not regex_pattern:
            # Default patterns based on environment
            app_env = os.getenv("APP_ENV", "development")
            
            if app_env == "production":
                # Example: Allow all subdomains of your domain
                prod_domain = os.getenv("PRODUCTION_DOMAIN", "")
                if prod_domain:
                    # Escape special characters and create pattern
                    escaped_domain = re.escape(prod_domain)
                    regex_pattern = f"https://.*\\.{escaped_domain}"
        
        if regex_pattern:
            try:
                # Validate regex
                re.compile(regex_pattern)
                logger.info(f"CORS origin regex: {regex_pattern}")
                return regex_pattern
            except re.error as e:
                logger.error(f"Invalid CORS regex pattern: {e}")
        
        return None
    
    def _should_allow_credentials(self) -> bool:
        """Determine if credentials should be allowed"""
        # Default to True for authenticated APIs
        return os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true"
    
    def _get_allowed_methods(self) -> List[str]:
        """Get allowed HTTP methods"""
        methods = os.getenv("CORS_ALLOWED_METHODS", "GET,POST,PUT,DELETE,OPTIONS,PATCH")
        return [m.strip() for m in methods.split(",")]
    
    def _get_allowed_headers(self) -> List[str]:
        """Get allowed request headers"""
        default_headers = [
            "Accept",
            "Accept-Language",
            "Content-Type",
            "Authorization",
            "X-Requested-With",
            "X-CSRF-Token",
            "X-API-Key"
        ]
        
        custom_headers = os.getenv("CORS_ALLOWED_HEADERS", "")
        if custom_headers:
            additional = [h.strip() for h in custom_headers.split(",")]
            default_headers.extend(additional)
        
        return list(set(default_headers))
    
    def _get_exposed_headers(self) -> List[str]:
        """Get headers exposed to the client"""
        default_exposed = [
            "Content-Length",
            "Content-Range",
            "X-Total-Count",
            "X-Page-Count",
            "X-Rate-Limit-Remaining",
            "X-Rate-Limit-Reset"
        ]
        
        custom_exposed = os.getenv("CORS_EXPOSED_HEADERS", "")
        if custom_exposed:
            additional = [h.strip() for h in custom_exposed.split(",")]
            default_exposed.extend(additional)
        
        return list(set(default_exposed))
    
    def _get_max_age(self) -> int:
        """Get preflight cache duration"""
        # Default to 1 hour
        return int(os.getenv("CORS_MAX_AGE", "3600"))
    
    def is_origin_allowed(self, origin: str) -> bool:
        """Check if an origin is allowed"""
        if not origin:
            return False
        
        # Check exact match
        if origin in self.allowed_origins:
            return True
        
        # Check wildcard
        if "*" in self.allowed_origins:
            return True
        
        # Check regex pattern
        if self.allowed_origin_regex:
            if re.match(self.allowed_origin_regex, origin):
                return True
        
        # Check for subdomain matching
        parsed = urlparse(origin)
        if parsed.scheme and parsed.netloc:
            # Check if it's a subdomain of allowed origins
            for allowed in self.allowed_origins:
                allowed_parsed = urlparse(allowed)
                if allowed_parsed.netloc:
                    # Check if origin is subdomain of allowed
                    if parsed.netloc.endswith(f".{allowed_parsed.netloc}"):
                        return True
        
        return False
    
    def get_cors_config(self) -> dict:
        """Get CORS configuration for FastAPI middleware"""
        config = {
            "allow_origins": self.allowed_origins if "*" not in self.allowed_origins else ["*"],
            "allow_credentials": self.allow_credentials,
            "allow_methods": self.allowed_methods,
            "allow_headers": self.allowed_headers,
            "expose_headers": self.exposed_headers,
            "max_age": self.max_age
        }
        
        # Add regex if configured
        if self.allowed_origin_regex and "*" not in self.allowed_origins:
            config["allow_origin_regex"] = self.allowed_origin_regex
        
        return config
    
    def validate_config(self) -> tuple[bool, List[str]]:
        """Validate CORS configuration for security issues"""
        issues = []
        
        # Check for overly permissive configurations
        if "*" in self.allowed_origins:
            if self.allow_credentials:
                issues.append("CRITICAL: Cannot use wildcard origin (*) with credentials")
            elif os.getenv("APP_ENV") == "production":
                issues.append("WARNING: Using wildcard origin (*) in production")
        
        # Check for HTTP origins in production
        if os.getenv("APP_ENV") == "production":
            for origin in self.allowed_origins:
                if origin.startswith("http://") and "localhost" not in origin:
                    issues.append(f"WARNING: Insecure HTTP origin in production: {origin}")
        
        # Check for missing origins
        if not self.allowed_origins and not self.allowed_origin_regex:
            issues.append("WARNING: No CORS origins configured")
        
        # Check methods
        if "*" in self.allowed_methods and os.getenv("APP_ENV") == "production":
            issues.append("WARNING: All HTTP methods allowed in production")
        
        # Check headers
        if "*" in self.allowed_headers and os.getenv("APP_ENV") == "production":
            issues.append("WARNING: All headers allowed in production")
        
        is_valid = not any("CRITICAL" in issue for issue in issues)
        return is_valid, issues

def setup_cors(app) -> None:
    """
    Setup CORS for the FastAPI application with production-ready configuration
    """
    cors_config = SecureCORSMiddleware()
    
    # Validate configuration
    is_valid, issues = cors_config.validate_config()
    
    if not is_valid:
        logger.error("CORS configuration has critical issues:")
        for issue in issues:
            logger.error(f"  - {issue}")
        
        # In production, fail fast on critical issues
        if os.getenv("APP_ENV") == "production":
            raise ValueError("Critical CORS configuration issues detected")
    elif issues:
        logger.warning("CORS configuration warnings:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    
    # Apply CORS middleware
    config = cors_config.get_cors_config()
    
    app.add_middleware(
        CORSMiddleware,
        **config
    )
    
    logger.info(f"CORS configured: origins={len(cors_config.allowed_origins)}, " 
                f"credentials={cors_config.allow_credentials}")
    
    # Log configuration in debug mode
    if os.getenv("DEBUG", "false").lower() == "true":
        logger.debug(f"CORS configuration: {config}")

def create_cors_error_response(origin: str = None) -> dict:
    """Create a CORS error response with helpful information"""
    app_env = os.getenv("APP_ENV", "development")
    
    response = {
        "error": "CORS policy violation",
        "message": "The request origin is not allowed by the CORS policy"
    }
    
    if app_env == "development":
        # Provide more details in development
        response["details"] = {
            "origin": origin,
            "help": "Add the origin to CORS_ORIGINS environment variable or update .env file"
        }
    
    return response