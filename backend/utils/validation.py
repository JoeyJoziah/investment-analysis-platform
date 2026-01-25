"""
Comprehensive Input Validation and Sanitization
Prevents SQL injection, XSS, and other security vulnerabilities
"""

import re
import logging
import html
import urllib.parse
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, date
from decimal import Decimal
import bleach
from pydantic import BaseModel, validator, ValidationError
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
import sqlparse

logger = logging.getLogger(__name__)

class InputValidator:
    """Comprehensive input validation and sanitization"""
    
    # Regex patterns for validation
    PATTERNS = {
        "email": re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"),
        "phone": re.compile(r"^\+?1?\d{9,15}$"),
        "ticker": re.compile(r"^[A-Z]{1,5}$"),
        "uuid": re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"),
        "alphanumeric": re.compile(r"^[a-zA-Z0-9]+$"),
        "alpha": re.compile(r"^[a-zA-Z]+$"),
        "numeric": re.compile(r"^[0-9]+$"),
        "decimal": re.compile(r"^-?\d+(\.\d+)?$"),
        "date": re.compile(r"^\d{4}-\d{2}-\d{2}$"),
        "datetime": re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"),
        "url": re.compile(r"^https?://[^\s/$.?#].[^\s]*$"),
        "safe_string": re.compile(r"^[a-zA-Z0-9\s\-_.,!?@#$%^&*()+=\[\]{}|;:'\"/\\]+$")
    }
    
    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        re.compile(r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER|CREATE|EXEC|EXECUTE)\b)", re.IGNORECASE),
        re.compile(r"(--|#|\/\*|\*\/)", re.IGNORECASE),
        re.compile(r"(\bOR\b\s*\d+\s*=\s*\d+)", re.IGNORECASE),
        re.compile(r"(\bAND\b\s*\d+\s*=\s*\d+)", re.IGNORECASE),
        re.compile(r"(;|'|\"|`|\\x00|\\n|\\r|\\x1a)", re.IGNORECASE),
        re.compile(r"(\bxp_cmdshell\b|\bsp_executesql\b)", re.IGNORECASE)
    ]
    
    # XSS patterns
    XSS_PATTERNS = [
        re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL),
        re.compile(r"javascript:", re.IGNORECASE),
        re.compile(r"on\w+\s*=", re.IGNORECASE),
        re.compile(r"<iframe[^>]*>.*?</iframe>", re.IGNORECASE | re.DOTALL),
        re.compile(r"<object[^>]*>.*?</object>", re.IGNORECASE | re.DOTALL),
        re.compile(r"<embed[^>]*>", re.IGNORECASE),
        re.compile(r"<img[^>]*onerror[^>]*>", re.IGNORECASE)
    ]
    
    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        re.compile(r"\.\./"),
        re.compile(r"\.\\.\\"),
        re.compile(r"%2e%2e%2f", re.IGNORECASE),
        re.compile(r"%252e%252e%252f", re.IGNORECASE),
        re.compile(r"\.\.%2f", re.IGNORECASE),
        re.compile(r"\.\.\\\\"),
        re.compile(r"/etc/passwd"),
        re.compile(r"C:\\\\", re.IGNORECASE)
    ]
    
    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000, allow_html: bool = False) -> str:
        """Sanitize string input"""
        if not isinstance(value, str):
            raise ValueError(f"Expected string, got {type(value)}")
        
        # Trim and limit length
        value = value.strip()[:max_length]
        
        # Remove null bytes
        value = value.replace("\x00", "")
        
        # HTML sanitization
        if not allow_html:
            # Remove all HTML tags
            value = bleach.clean(value, tags=[], strip=True)
        else:
            # Allow safe HTML tags only
            allowed_tags = ['p', 'br', 'strong', 'em', 'u', 'a', 'ul', 'ol', 'li']
            allowed_attributes = {'a': ['href', 'title']}
            value = bleach.clean(value, tags=allowed_tags, attributes=allowed_attributes)
        
        # Escape special characters
        value = html.escape(value)
        
        return value
    
    @staticmethod
    def validate_email(email: str) -> str:
        """Validate and normalize email address"""
        email = email.strip().lower()
        
        if not InputValidator.PATTERNS["email"].match(email):
            raise ValueError(f"Invalid email format: {email}")
        
        # Additional checks
        if len(email) > 255:
            raise ValueError("Email address too long")
        
        # Check for suspicious patterns
        if any(pattern in email for pattern in ["<script", "javascript:", "onclick"]):
            raise ValueError("Invalid characters in email")
        
        return email
    
    @staticmethod
    def validate_ticker(ticker: str) -> str:
        """Validate stock ticker symbol"""
        ticker = ticker.strip().upper()
        
        if not InputValidator.PATTERNS["ticker"].match(ticker):
            raise ValueError(f"Invalid ticker format: {ticker}")
        
        return ticker
    
    @staticmethod
    def validate_phone(phone: str) -> str:
        """Validate phone number"""
        # Remove common formatting characters
        phone = re.sub(r"[\s\-\(\)]+", "", phone)
        
        if not InputValidator.PATTERNS["phone"].match(phone):
            raise ValueError(f"Invalid phone number: {phone}")
        
        return phone
    
    @staticmethod
    def validate_uuid(uuid_str: str) -> str:
        """Validate UUID format"""
        uuid_str = uuid_str.strip().lower()
        
        if not InputValidator.PATTERNS["uuid"].match(uuid_str):
            raise ValueError(f"Invalid UUID format: {uuid_str}")
        
        return uuid_str
    
    @staticmethod
    def validate_numeric(value: Union[str, int, float], 
                        min_value: Optional[float] = None,
                        max_value: Optional[float] = None) -> float:
        """Validate numeric input"""
        try:
            num_value = float(value)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid numeric value: {value}")
        
        if min_value is not None and num_value < min_value:
            raise ValueError(f"Value {num_value} is below minimum {min_value}")
        
        if max_value is not None and num_value > max_value:
            raise ValueError(f"Value {num_value} exceeds maximum {max_value}")
        
        # Check for special values
        if not (-1e308 < num_value < 1e308):
            raise ValueError(f"Numeric value out of range: {num_value}")
        
        return num_value
    
    @staticmethod
    def validate_date(date_str: str, 
                     min_date: Optional[date] = None,
                     max_date: Optional[date] = None) -> date:
        """Validate date input"""
        try:
            if isinstance(date_str, date):
                date_value = date_str
            else:
                date_value = datetime.strptime(str(date_str), "%Y-%m-%d").date()
        except ValueError:
            raise ValueError(f"Invalid date format: {date_str}")
        
        if min_date and date_value < min_date:
            raise ValueError(f"Date {date_value} is before minimum {min_date}")
        
        if max_date and date_value > max_date:
            raise ValueError(f"Date {date_value} is after maximum {max_date}")
        
        return date_value
    
    @staticmethod
    def validate_url(url: str, allowed_domains: Optional[List[str]] = None) -> str:
        """Validate URL"""
        url = url.strip()
        
        if not InputValidator.PATTERNS["url"].match(url):
            raise ValueError(f"Invalid URL format: {url}")
        
        # Parse URL
        parsed = urllib.parse.urlparse(url)
        
        # Check scheme
        if parsed.scheme not in ["http", "https"]:
            raise ValueError(f"Invalid URL scheme: {parsed.scheme}")
        
        # Check domain whitelist
        if allowed_domains:
            if not any(parsed.netloc.endswith(domain) for domain in allowed_domains):
                raise ValueError(f"Domain not allowed: {parsed.netloc}")
        
        # Check for suspicious patterns
        if any(pattern in url.lower() for pattern in ["javascript:", "data:", "vbscript:"]):
            raise ValueError("Potentially malicious URL")
        
        return url
    
    @staticmethod
    def check_sql_injection(value: str) -> bool:
        """Check for SQL injection attempts"""
        if not isinstance(value, str):
            return False
        
        # Check against SQL patterns
        for pattern in InputValidator.SQL_INJECTION_PATTERNS:
            if pattern.search(value):
                logger.warning(f"Potential SQL injection detected: {value[:100]}")
                return True
        
        # Use sqlparse for additional validation
        try:
            parsed = sqlparse.parse(value)
            if parsed and len(parsed) > 0:
                # If sqlparse can parse it as SQL, it might be an injection attempt
                for statement in parsed:
                    if statement.get_type() != 'UNKNOWN':
                        logger.warning(f"SQL statement detected in input: {statement.get_type()}")
                        return True
        except:
            pass
        
        return False
    
    @staticmethod
    def check_xss(value: str) -> bool:
        """Check for XSS attempts"""
        if not isinstance(value, str):
            return False
        
        for pattern in InputValidator.XSS_PATTERNS:
            if pattern.search(value):
                logger.warning(f"Potential XSS detected: {value[:100]}")
                return True
        
        return False
    
    @staticmethod
    def check_path_traversal(value: str) -> bool:
        """Check for path traversal attempts"""
        if not isinstance(value, str):
            return False
        
        for pattern in InputValidator.PATH_TRAVERSAL_PATTERNS:
            if pattern.search(value):
                logger.warning(f"Potential path traversal detected: {value[:100]}")
                return True
        
        return False
    
    @staticmethod
    def validate_dict(data: Dict[str, Any], 
                     required_fields: Optional[List[str]] = None,
                     allowed_fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """Validate dictionary input"""
        if not isinstance(data, dict):
            raise ValueError("Input must be a dictionary")
        
        # Check required fields
        if required_fields:
            missing = set(required_fields) - set(data.keys())
            if missing:
                raise ValueError(f"Missing required fields: {missing}")
        
        # Check allowed fields
        if allowed_fields:
            extra = set(data.keys()) - set(allowed_fields)
            if extra:
                raise ValueError(f"Unexpected fields: {extra}")
        
        # Validate each field
        validated = {}
        for key, value in data.items():
            # Validate key
            if not isinstance(key, str) or len(key) > 100:
                raise ValueError(f"Invalid field name: {key}")
            
            # Check for injection in key
            if InputValidator.check_sql_injection(key) or InputValidator.check_xss(key):
                raise ValueError(f"Invalid characters in field name: {key}")
            
            # Validate value recursively
            if isinstance(value, str):
                if InputValidator.check_sql_injection(value) or InputValidator.check_xss(value):
                    raise ValueError(f"Invalid content in field: {key}")
                validated[key] = InputValidator.sanitize_string(value)
            elif isinstance(value, dict):
                validated[key] = InputValidator.validate_dict(value)
            elif isinstance(value, list):
                validated[key] = InputValidator.validate_list(value)
            else:
                validated[key] = value
        
        return validated
    
    @staticmethod
    def validate_list(data: List[Any], max_items: int = 1000) -> List[Any]:
        """Validate list input"""
        if not isinstance(data, list):
            raise ValueError("Input must be a list")
        
        if len(data) > max_items:
            raise ValueError(f"List exceeds maximum size of {max_items}")
        
        validated = []
        for item in data:
            if isinstance(item, str):
                if InputValidator.check_sql_injection(item) or InputValidator.check_xss(item):
                    raise ValueError(f"Invalid content in list item")
                validated.append(InputValidator.sanitize_string(item))
            elif isinstance(item, dict):
                validated.append(InputValidator.validate_dict(item))
            elif isinstance(item, list):
                validated.append(InputValidator.validate_list(item))
            else:
                validated.append(item)
        
        return validated

class ValidationMiddleware:
    """Middleware for request validation"""
    
    def __init__(self):
        self.validator = InputValidator()
        
    async def __call__(self, request: Request, call_next):
        """Validate incoming requests"""
        # Skip validation for certain paths
        skip_paths = ["/api/health", "/api/docs", "/api/redoc", "/openapi.json"]
        if request.url.path in skip_paths:
            return await call_next(request)
        
        try:
            # Validate headers
            self._validate_headers(request.headers)
            
            # Validate query parameters
            if request.query_params:
                self._validate_query_params(dict(request.query_params))
            
            # Validate path parameters
            if request.path_params:
                self._validate_path_params(dict(request.path_params))
            
            # For POST/PUT/PATCH requests, validate body
            if request.method in ["POST", "PUT", "PATCH"]:
                # Note: Body validation should be done in route handlers
                # using Pydantic models for better integration
                pass
            
            response = await call_next(request)
            return response
            
        except ValueError as e:
            logger.warning(f"Validation error: {e}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"error": "Validation error", "detail": str(e)}
            )
        except Exception as e:
            logger.error(f"Unexpected validation error: {e}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": "Internal server error"}
            )
    
    def _validate_headers(self, headers: Dict[str, str]):
        """Validate request headers"""
        # Check for suspicious headers
        suspicious_headers = ["X-Forwarded-Host", "X-Original-URL", "X-Rewrite-URL"]
        
        for header in suspicious_headers:
            if header in headers:
                value = headers[header]
                if self.validator.check_path_traversal(value):
                    raise ValueError(f"Invalid header value: {header}")
    
    def _validate_query_params(self, params: Dict[str, str]):
        """Validate query parameters"""
        for key, value in params.items():
            # Check key
            if self.validator.check_sql_injection(key) or self.validator.check_xss(key):
                raise ValueError(f"Invalid query parameter name: {key}")
            
            # Check value
            if self.validator.check_sql_injection(value) or self.validator.check_xss(value):
                raise ValueError(f"Invalid query parameter value: {key}")
    
    def _validate_path_params(self, params: Dict[str, str]):
        """Validate path parameters"""
        for key, value in params.items():
            # Check for path traversal
            if self.validator.check_path_traversal(value):
                raise ValueError(f"Invalid path parameter: {key}")
            
            # Check for injection
            if self.validator.check_sql_injection(value) or self.validator.check_xss(value):
                raise ValueError(f"Invalid path parameter value: {key}")

# Pydantic validators for common fields
class SecureStringField(str):
    """Secure string field with validation"""
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    
    @classmethod
    def validate(cls, v):
        if not isinstance(v, str):
            raise TypeError("String required")
        
        # Check for injection
        if InputValidator.check_sql_injection(v) or InputValidator.check_xss(v):
            raise ValueError("Invalid characters detected")
        
        # Sanitize
        return InputValidator.sanitize_string(v)

class SecureEmailField(str):
    """Secure email field with validation"""
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    
    @classmethod
    def validate(cls, v):
        return InputValidator.validate_email(v)

class SecureTickerField(str):
    """Secure ticker field with validation"""
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    
    @classmethod
    def validate(cls, v):
        return InputValidator.validate_ticker(v)

def setup_validation(app):
    """Setup validation middleware for the application"""
    app.add_middleware(ValidationMiddleware)
    logger.info("Input validation middleware configured")


def validate_financial_data(
    data: Dict[str, Any],
    required_fields: Optional[List[str]] = None,
    numeric_fields: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Validate financial data dictionary.

    Args:
        data: Financial data to validate
        required_fields: List of required fields
        numeric_fields: List of fields that must be numeric

    Returns:
        Dictionary with validation status and issues
    """
    errors = []
    warnings = []

    if required_fields is None:
        required_fields = ['ticker', 'price', 'volume']

    if numeric_fields is None:
        numeric_fields = ['price', 'volume', 'market_cap', 'pe_ratio']

    # Check required fields
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")

    # Check numeric fields
    for field in numeric_fields:
        if field in data:
            value = data[field]
            if value is not None:
                try:
                    float(value)
                except (ValueError, TypeError):
                    errors.append(f"Field {field} must be numeric")

    # Check price validity
    if 'price' in data and data['price'] is not None:
        try:
            price = float(data['price'])
            if price <= 0:
                errors.append("Price must be positive")
        except (ValueError, TypeError):
            pass

    # Check volume validity
    if 'volume' in data and data['volume'] is not None:
        try:
            volume = float(data['volume'])
            if volume < 0:
                errors.append("Volume cannot be negative")
        except (ValueError, TypeError):
            pass

    # Calculate data quality score
    total_fields = len(data)
    valid_fields = total_fields - len(errors)
    quality_score = valid_fields / total_fields if total_fields > 0 else 0

    return {
        'is_valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'data_quality_score': quality_score
    }