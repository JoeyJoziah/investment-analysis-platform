"""
Comprehensive Input Validation and Sanitization System
Protects against injection attacks, XSS, and data validation issues
"""

import re
import html
import json
import uuid
from typing import Any, Dict, List, Optional, Union, Callable, Set
from datetime import datetime, date
from decimal import Decimal, InvalidOperation
from enum import Enum
import validators
from pydantic import BaseModel, validator, ValidationError
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
import logging
import bleach
from urllib.parse import quote, unquote
import base64
from dataclasses import dataclass
import asyncio
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger = logging.getLogger(__name__)


class ValidationError(HTTPException):
    """Custom validation error"""
    def __init__(self, detail: str = "Validation failed", errors: List[str] = None):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": detail, "errors": errors or []}
        )


class SanitizationLevel(str, Enum):
    """Sanitization levels for different contexts"""
    STRICT = "strict"      # Remove all HTML and special characters
    MODERATE = "moderate"  # Allow safe HTML tags
    MINIMAL = "minimal"    # Basic XSS protection only
    NONE = "none"         # No sanitization (for trusted input)


class InputType(str, Enum):
    """Input data types for specific validation"""
    TEXT = "text"
    EMAIL = "email"
    URL = "url"
    USERNAME = "username"
    PASSWORD = "password"
    PHONE = "phone"
    TICKER_SYMBOL = "ticker_symbol"
    CURRENCY_CODE = "currency_code"
    AMOUNT = "amount"
    PERCENTAGE = "percentage"
    DATE = "date"
    DATETIME = "datetime"
    JSON = "json"
    SQL_IDENTIFIER = "sql_identifier"
    FILE_PATH = "file_path"
    IP_ADDRESS = "ip_address"
    UUID = "uuid"


@dataclass
class ValidationRule:
    """Validation rule definition"""
    field_name: str
    input_type: InputType
    required: bool = True
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[Union[int, float, Decimal]] = None
    max_value: Optional[Union[int, float, Decimal]] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    sanitization_level: SanitizationLevel = SanitizationLevel.MODERATE
    custom_validator: Optional[Callable] = None


class SecurityPatterns:
    """Common security patterns for validation"""
    
    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r"(\b(ALTER|CREATE|DELETE|DROP|EXEC|EXECUTE|INSERT|MERGE|SELECT|UPDATE|UNION|TRUNCATE)\b)",
        r"(\b(AND|OR)\s+\d+\s*=\s*\d+)",
        r"(\b(AND|OR)\s+[\"']?\d+[\"']?\s*=\s*[\"']?\d+[\"']?)",
        r"(SCRIPT\s*:)",
        r"(JAVASCRIPT\s*:)",
        r"(VBSCRIPT\s*:)",
        r"(ONLOAD\s*=)",
        r"(ONERROR\s*=)",
        r"(<SCRIPT[^>]*>.*?</SCRIPT>)",
    ]
    
    # XSS patterns
    XSS_PATTERNS = [
        r"(<script[^>]*>.*?</script>)",
        r"(<iframe[^>]*>.*?</iframe>)",
        r"(<object[^>]*>.*?</object>)",
        r"(<embed[^>]*>.*?</embed>)",
        r"(<applet[^>]*>.*?</applet>)",
        r"(<meta[^>]*>)",
        r"(<link[^>]*>)",
        r"(javascript:)",
        r"(vbscript:)",
        r"(onload\s*=)",
        r"(onerror\s*=)",
        r"(onclick\s*=)",
        r"(onmouseover\s*=)",
        r"(onfocus\s*=)",
    ]
    
    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        r"(\.\.[\\/])",
        r"([\\/]\.\.)",
        r"(%2e%2e[\\/])",
        r"([\\/]%2e%2e)",
        r"(\.\.[%2f%5c])",
        r"([%2f%5c]\.\.)",
    ]
    
    # Command injection patterns
    COMMAND_INJECTION_PATTERNS = [
        r"(\s*[;&|`$(){}[\]<>]|\s*\n)",
        r"(\$\([^)]*\))",
        r"(`[^`]*`)",
        r"(\${[^}]*})",
    ]


class InputSanitizer:
    """Input sanitization and cleaning utilities"""
    
    def __init__(self):
        # Configure bleach for HTML sanitization
        self.allowed_tags = {
            'minimal': [],
            'moderate': ['p', 'br', 'strong', 'em', 'u', 'ol', 'ul', 'li'],
            'strict': []
        }
        
        self.allowed_attributes = {
            'minimal': {},
            'moderate': {},
            'strict': {}
        }
    
    def sanitize_html(self, text: str, level: SanitizationLevel = SanitizationLevel.MODERATE) -> str:
        """Sanitize HTML content based on level"""
        if level == SanitizationLevel.NONE:
            return text
        
        if level == SanitizationLevel.STRICT:
            # Strip all HTML tags
            return bleach.clean(text, tags=[], attributes={}, strip=True)
        
        # Use configured tags and attributes for other levels
        return bleach.clean(
            text,
            tags=self.allowed_tags.get(level.value, []),
            attributes=self.allowed_attributes.get(level.value, {}),
            strip=True
        )
    
    def sanitize_sql_identifier(self, identifier: str) -> str:
        """Sanitize SQL identifier (table, column names)"""
        # Only allow alphanumeric, underscore, and hyphen
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', identifier)
        
        # Ensure it doesn't start with a number
        if sanitized and sanitized[0].isdigit():
            sanitized = f"_{sanitized}"
        
        return sanitized[:63]  # PostgreSQL limit
    
    def sanitize_file_path(self, path: str) -> str:
        """Sanitize file path to prevent traversal attacks"""
        # Remove path traversal attempts
        sanitized = re.sub(r'\.\.[\\/]', '', path)
        sanitized = re.sub(r'[\\/]\.\.', '', sanitized)
        
        # Remove dangerous characters
        sanitized = re.sub(r'[<>:"|?*]', '', sanitized)
        
        return sanitized
    
    def escape_json(self, obj: Any) -> str:
        """Safely serialize object to JSON"""
        try:
            return json.dumps(obj, default=str, ensure_ascii=True)
        except (TypeError, ValueError) as e:
            logger.error(f"JSON serialization error: {e}")
            return "{}"
    
    def normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters"""
        import unicodedata
        return unicodedata.normalize('NFKC', text)


class InputValidator:
    """Comprehensive input validation system"""
    
    def __init__(self):
        self.sanitizer = InputSanitizer()
        self.security_patterns = SecurityPatterns()
    
    def detect_injection_attempt(self, value: str) -> List[str]:
        """Detect potential injection attacks"""
        threats = []
        value_lower = value.lower()
        
        # Check SQL injection patterns
        for pattern in self.security_patterns.SQL_INJECTION_PATTERNS:
            if re.search(pattern, value_lower, re.IGNORECASE):
                threats.append(f"Potential SQL injection: {pattern}")
        
        # Check XSS patterns
        for pattern in self.security_patterns.XSS_PATTERNS:
            if re.search(pattern, value_lower, re.IGNORECASE):
                threats.append(f"Potential XSS attack: {pattern}")
        
        # Check path traversal
        for pattern in self.security_patterns.PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, value_lower, re.IGNORECASE):
                threats.append(f"Potential path traversal: {pattern}")
        
        # Check command injection
        for pattern in self.security_patterns.COMMAND_INJECTION_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                threats.append(f"Potential command injection: {pattern}")
        
        return threats
    
    def validate_by_type(self, value: Any, input_type: InputType) -> tuple[bool, str, Any]:
        """Validate input by specific type"""
        try:
            if value is None:
                return True, "", None
            
            # Convert to string for most validations
            str_value = str(value).strip()
            
            if input_type == InputType.EMAIL:
                if not validators.email(str_value):
                    return False, "Invalid email format", None
                return True, "", str_value.lower()
            
            elif input_type == InputType.URL:
                if not validators.url(str_value):
                    return False, "Invalid URL format", None
                return True, "", str_value
            
            elif input_type == InputType.USERNAME:
                # Username: 3-30 chars, alphanumeric + underscore/hyphen
                if not re.match(r'^[a-zA-Z0-9_-]{3,30}$', str_value):
                    return False, "Username must be 3-30 characters, alphanumeric with underscore/hyphen only", None
                return True, "", str_value.lower()
            
            elif input_type == InputType.PASSWORD:
                if len(str_value) < 12:
                    return False, "Password must be at least 12 characters long", None
                if not re.search(r'[A-Z]', str_value):
                    return False, "Password must contain at least one uppercase letter", None
                if not re.search(r'[a-z]', str_value):
                    return False, "Password must contain at least one lowercase letter", None
                if not re.search(r'[0-9]', str_value):
                    return False, "Password must contain at least one digit", None
                if not re.search(r'[!@#$%^&*()_+\-=\[\]{}|;:,.<>?]', str_value):
                    return False, "Password must contain at least one special character", None
                return True, "", str_value
            
            elif input_type == InputType.PHONE:
                # Remove all non-digits
                phone_digits = re.sub(r'[^0-9]', '', str_value)
                if len(phone_digits) < 10 or len(phone_digits) > 15:
                    return False, "Phone number must be 10-15 digits", None
                return True, "", phone_digits
            
            elif input_type == InputType.TICKER_SYMBOL:
                # Stock ticker: 1-10 uppercase letters
                if not re.match(r'^[A-Z]{1,10}$', str_value.upper()):
                    return False, "Invalid ticker symbol format", None
                return True, "", str_value.upper()
            
            elif input_type == InputType.CURRENCY_CODE:
                # ISO 4217 currency codes (3 uppercase letters)
                if not re.match(r'^[A-Z]{3}$', str_value.upper()):
                    return False, "Invalid currency code format", None
                return True, "", str_value.upper()
            
            elif input_type == InputType.AMOUNT:
                try:
                    amount = Decimal(str_value)
                    if amount < 0:
                        return False, "Amount cannot be negative", None
                    return True, "", amount
                except InvalidOperation:
                    return False, "Invalid amount format", None
            
            elif input_type == InputType.PERCENTAGE:
                try:
                    percentage = float(str_value)
                    if percentage < 0 or percentage > 100:
                        return False, "Percentage must be between 0 and 100", None
                    return True, "", percentage
                except ValueError:
                    return False, "Invalid percentage format", None
            
            elif input_type == InputType.DATE:
                try:
                    parsed_date = datetime.strptime(str_value, "%Y-%m-%d").date()
                    return True, "", parsed_date
                except ValueError:
                    return False, "Invalid date format (YYYY-MM-DD required)", None
            
            elif input_type == InputType.DATETIME:
                try:
                    parsed_datetime = datetime.fromisoformat(str_value.replace('Z', '+00:00'))
                    return True, "", parsed_datetime
                except ValueError:
                    return False, "Invalid datetime format (ISO 8601 required)", None
            
            elif input_type == InputType.JSON:
                try:
                    parsed_json = json.loads(str_value)
                    return True, "", parsed_json
                except json.JSONDecodeError as e:
                    return False, f"Invalid JSON format: {e}", None
            
            elif input_type == InputType.SQL_IDENTIFIER:
                sanitized = self.sanitizer.sanitize_sql_identifier(str_value)
                if not sanitized:
                    return False, "Invalid SQL identifier", None
                return True, "", sanitized
            
            elif input_type == InputType.FILE_PATH:
                sanitized = self.sanitizer.sanitize_file_path(str_value)
                if not sanitized:
                    return False, "Invalid file path", None
                return True, "", sanitized
            
            elif input_type == InputType.IP_ADDRESS:
                if not (validators.ipv4(str_value) or validators.ipv6(str_value)):
                    return False, "Invalid IP address format", None
                return True, "", str_value
            
            elif input_type == InputType.UUID:
                try:
                    uuid_obj = uuid.UUID(str_value)
                    return True, "", str(uuid_obj)
                except ValueError:
                    return False, "Invalid UUID format", None
            
            else:  # InputType.TEXT or default
                return True, "", str_value
                
        except Exception as e:
            logger.error(f"Validation error for {input_type}: {e}")
            return False, f"Validation error: {e}", None
    
    def validate_field(self, value: Any, rule: ValidationRule) -> tuple[bool, List[str], Any]:
        """Validate a single field against its rule"""
        errors = []
        
        # Check if required field is missing
        if rule.required and (value is None or str(value).strip() == ""):
            errors.append(f"{rule.field_name} is required")
            return False, errors, None
        
        # Skip validation for empty non-required fields
        if not rule.required and (value is None or str(value).strip() == ""):
            return True, [], None
        
        str_value = str(value).strip()
        
        # Detect injection attempts
        threats = self.detect_injection_attempt(str_value)
        if threats:
            errors.extend([f"{rule.field_name}: {threat}" for threat in threats])
            logger.warning(f"Security threat detected in {rule.field_name}: {threats}")
            return False, errors, None
        
        # Sanitize the input
        sanitized_value = self.sanitizer.sanitize_html(str_value, rule.sanitization_level)
        
        # Length validation
        if rule.min_length is not None and len(sanitized_value) < rule.min_length:
            errors.append(f"{rule.field_name} must be at least {rule.min_length} characters")
        
        if rule.max_length is not None and len(sanitized_value) > rule.max_length:
            errors.append(f"{rule.field_name} must not exceed {rule.max_length} characters")
        
        # Pattern validation
        if rule.pattern and not re.match(rule.pattern, sanitized_value):
            errors.append(f"{rule.field_name} does not match required pattern")
        
        # Allowed values validation
        if rule.allowed_values and sanitized_value not in rule.allowed_values:
            errors.append(f"{rule.field_name} must be one of: {', '.join(map(str, rule.allowed_values))}")
        
        # Type-specific validation
        valid, error_msg, validated_value = self.validate_by_type(sanitized_value, rule.input_type)
        if not valid:
            errors.append(f"{rule.field_name}: {error_msg}")
            return False, errors, None
        
        # Value range validation (for numeric types)
        if rule.min_value is not None or rule.max_value is not None:
            try:
                numeric_value = float(validated_value)
                if rule.min_value is not None and numeric_value < rule.min_value:
                    errors.append(f"{rule.field_name} must be at least {rule.min_value}")
                if rule.max_value is not None and numeric_value > rule.max_value:
                    errors.append(f"{rule.field_name} must not exceed {rule.max_value}")
            except (TypeError, ValueError):
                pass  # Skip range validation for non-numeric types
        
        # Custom validation
        if rule.custom_validator:
            try:
                custom_valid, custom_error = rule.custom_validator(validated_value)
                if not custom_valid:
                    errors.append(f"{rule.field_name}: {custom_error}")
            except Exception as e:
                errors.append(f"{rule.field_name}: Custom validation error: {e}")
        
        return len(errors) == 0, errors, validated_value
    
    def validate_data(self, data: Dict[str, Any], rules: List[ValidationRule]) -> tuple[bool, List[str], Dict[str, Any]]:
        """Validate entire data dictionary against rules"""
        all_errors = []
        validated_data = {}
        
        # Create rule lookup
        rule_lookup = {rule.field_name: rule for rule in rules}
        
        # Validate each field in the data
        for field_name, value in data.items():
            if field_name in rule_lookup:
                rule = rule_lookup[field_name]
                valid, errors, validated_value = self.validate_field(value, rule)
                
                if valid:
                    validated_data[field_name] = validated_value
                else:
                    all_errors.extend(errors)
            else:
                # Handle unknown fields - sanitize as text
                str_value = str(value).strip()
                threats = self.detect_injection_attempt(str_value)
                if threats:
                    all_errors.append(f"Unknown field {field_name} contains security threats")
                    logger.warning(f"Security threat in unknown field {field_name}: {threats}")
                else:
                    validated_data[field_name] = self.sanitizer.sanitize_html(str_value)
        
        # Check for missing required fields
        for rule in rules:
            if rule.required and rule.field_name not in data:
                all_errors.append(f"{rule.field_name} is required")
        
        return len(all_errors) == 0, all_errors, validated_data


class ValidationMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for automatic input validation and sanitization"""
    
    def __init__(self, app, validation_config: Optional[Dict[str, List[ValidationRule]]] = None):
        super().__init__(app)
        self.validator = InputValidator()
        self.validation_config = validation_config or {}
        
        # Define default validation rules for common endpoints
        self.default_rules = {
            "/api/auth/login": [
                ValidationRule("username", InputType.USERNAME, required=True, max_length=50),
                ValidationRule("password", InputType.PASSWORD, required=True, max_length=128),
            ],
            "/api/auth/register": [
                ValidationRule("username", InputType.USERNAME, required=True, max_length=50),
                ValidationRule("email", InputType.EMAIL, required=True, max_length=254),
                ValidationRule("password", InputType.PASSWORD, required=True, max_length=128),
                ValidationRule("full_name", InputType.TEXT, required=True, max_length=100, 
                              sanitization_level=SanitizationLevel.STRICT),
            ],
            "/api/stocks/*": [
                ValidationRule("symbol", InputType.TICKER_SYMBOL, required=True, max_length=10),
                ValidationRule("limit", InputType.AMOUNT, required=False, min_value=1, max_value=1000),
                ValidationRule("offset", InputType.AMOUNT, required=False, min_value=0),
            ],
            "/api/portfolio/*": [
                ValidationRule("amount", InputType.AMOUNT, required=False, min_value=0.01),
                ValidationRule("percentage", InputType.PERCENTAGE, required=False, min_value=0, max_value=100),
            ],
        }
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request through validation pipeline"""
        try:
            # Skip validation for certain paths
            skip_paths = ["/api/health", "/api/metrics", "/api/docs", "/api/redoc", "/api/openapi.json"]
            if any(request.url.path.startswith(path) for path in skip_paths):
                return await call_next(request)
            
            # Skip validation for GET requests (query params handled separately)
            if request.method == "GET":
                return await call_next(request)
            
            # Get validation rules for this endpoint
            rules = self._get_validation_rules(request.url.path)
            if not rules:
                return await call_next(request)
            
            # Parse request body
            try:
                body = await request.body()
                if not body:
                    return await call_next(request)
                
                # Parse JSON body
                try:
                    data = json.loads(body.decode('utf-8'))
                except json.JSONDecodeError:
                    return JSONResponse(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        content={"error": "Invalid JSON format"}
                    )
                
                # Validate data
                valid, errors, validated_data = self.validator.validate_data(data, rules)
                
                if not valid:
                    logger.warning(f"Validation failed for {request.url.path}: {errors}")
                    return JSONResponse(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        content={
                            "error": "Validation failed",
                            "details": errors,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    )
                
                # Replace request body with validated data
                validated_body = json.dumps(validated_data).encode('utf-8')
                request._body = validated_body
                
            except Exception as e:
                logger.error(f"Validation middleware error: {e}")
                return JSONResponse(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content={"error": "Validation processing failed"}
                )
            
            return await call_next(request)
            
        except Exception as e:
            logger.error(f"Validation middleware critical error: {e}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": "Internal server error"}
            )
    
    def _get_validation_rules(self, path: str) -> Optional[List[ValidationRule]]:
        """Get validation rules for a specific path"""
        # Exact match first
        if path in self.validation_config:
            return self.validation_config[path]
        
        # Check default rules
        if path in self.default_rules:
            return self.default_rules[path]
        
        # Pattern matching for wildcard rules
        for pattern, rules in {**self.validation_config, **self.default_rules}.items():
            if pattern.endswith("/*"):
                prefix = pattern[:-2]
                if path.startswith(prefix):
                    return rules
        
        return None


# Utility functions for manual validation
def validate_ticker_symbol(symbol: str) -> str:
    """Validate and normalize ticker symbol"""
    validator = InputValidator()
    valid, error, validated = validator.validate_by_type(symbol, InputType.TICKER_SYMBOL)
    if not valid:
        raise ValidationError(f"Invalid ticker symbol: {error}")
    return validated


def validate_email_address(email: str) -> str:
    """Validate email address"""
    validator = InputValidator()
    valid, error, validated = validator.validate_by_type(email, InputType.EMAIL)
    if not valid:
        raise ValidationError(f"Invalid email address: {error}")
    return validated


def validate_currency_amount(amount: Union[str, int, float, Decimal]) -> Decimal:
    """Validate currency amount"""
    validator = InputValidator()
    valid, error, validated = validator.validate_by_type(amount, InputType.AMOUNT)
    if not valid:
        raise ValidationError(f"Invalid amount: {error}")
    return validated


def sanitize_user_input(text: str, level: SanitizationLevel = SanitizationLevel.MODERATE) -> str:
    """Sanitize user input text"""
    sanitizer = InputSanitizer()
    return sanitizer.sanitize_html(text, level)


def detect_security_threats(text: str) -> List[str]:
    """Detect potential security threats in text"""
    validator = InputValidator()
    return validator.detect_injection_attempt(text)